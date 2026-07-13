"""Single-file deterministic media, subtitle, OCR, and storage pipeline.

The module intentionally owns only the infrastructure layer. It does not
infer speakers, personalities, relationships, writing rules, or WebGAL
formatting. Downstream analyzers consume the files written per video.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import difflib
import hashlib
import html
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Protocol, Sequence

import numpy as np
import yt_dlp
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from yt_dlp.networking.impersonate import ImpersonateTarget


SCHEMA_VERSION = 3
PIPELINE_VERSION = "3.0.0"
MIN_SAMPLE_INTERVAL_SECONDS = 3.0
MIN_SAMPLE_INTERVAL = MIN_SAMPLE_INTERVAL_SECONDS

BILIBILI_REFERER = "https://www.bilibili.com/"
BILIBILI_ORIGIN = "https://www.bilibili.com"
CHROME_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
VIDEO_SUFFIXES = {".mp4", ".mkv", ".flv", ".webm", ".mov"}
SUPPORTED_SUBTITLE_SUFFIXES = {".srt", ".vtt", ".ass", ".ssa", ".json", ".xml"}
YTDLP_COMMON_ARGS = (
    "--ignore-config",
    "--no-playlist",
    "--impersonate",
    "chrome",
    "--add-header",
    f"Referer: {BILIBILI_REFERER}",
    "--add-header",
    f"Origin: {BILIBILI_ORIGIN}",
)
BBDOWN_SUBTITLE_ARGS = ("--sub-only", "--skip-ai=false", "-F", "<bvid>")
BBDOWN_VIDEO_ARGS = (
    "--video-only",
    "--skip-mux",
    "-q",
    "360P 流畅,480P 清晰",
)

BV_RE = re.compile(r"\b(BV[0-9A-Za-z]{10})\b")
AV_RE = re.compile(r"\b(?:av|AV)(\d+)\b")
VIDEO_ID_RE = re.compile(r"(BV[0-9A-Za-z]{10}|av\d+)", re.IGNORECASE)
NUMBER_PREFIX_RE = re.compile(
    r"^\s*(?:(?:\d+)\s*[.\uFF0E\u3001)\uFF09:]|[\uFF08(]\s*\d+\s*[\uFF09)])\s*"
)
TIME_RANGE_RE = re.compile(
    r"(?P<start>\d{1,2}:\d{2}(?::\d{2})?[,.]\d{1,3})\s*-->\s*"
    r"(?P<end>\d{1,2}:\d{2}(?::\d{2})?[,.]\d{1,3})"
)
NON_DIALOGUE_RE = re.compile(
    r"^[\s\u266a\u266b\u266c\u3010\u3011\[\]()<>\-_.:\uFF1A,\uFF0C]*"
    r"(?:music|bgm|soundtrack|instrumental|\u97f3\u4e50|\u7eaf\u97f3\u4e50|\u97f3\u6548)"
    r"[\s\u266a\u266b\u266c\u3010\u3011\[\]()<>\-_.:\uFF1A,\uFF0C]*$",
    re.IGNORECASE,
)
MEANINGFUL_CHAR_RE = re.compile(r"[A-Za-z0-9\u3040-\u30ff\u3400-\u9fff]")
TERMINAL_PUNCTUATION = (".", "!", "?", "\u3002", "\uff01", "\uff1f", "\u2026")


# ---------------------------------------------------------------------------
# Configuration and data records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CropRegion:
    top: float = 0.50
    bottom: float = 0.94
    left: float = 0.04
    right: float = 0.96

    def __post_init__(self) -> None:
        if not (
            0.0 <= self.left < self.right <= 1.0
            and 0.0 <= self.top < self.bottom <= 1.0
        ):
            raise ValueError("crop boundaries must form a non-empty region inside 0..1")

    def as_dict(self) -> dict[str, float]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class PipelineConfig:
    sample_interval_seconds: float = MIN_SAMPLE_INTERVAL_SECONDS
    start_time: float | None = None
    end_time: float | None = None
    crop: CropRegion = field(default_factory=CropRegion)
    prefer_subtitle_track: bool = True
    enable_hardsub_ocr: bool = True
    keep_frames: Literal["none", "changed", "all"] = "changed"
    max_frame_width: int = 1280
    frame_similarity_threshold: float = 0.018
    adaptive_ocr_confidence: float = 0.78
    force: bool = False

    def __post_init__(self) -> None:
        if self.sample_interval_seconds < MIN_SAMPLE_INTERVAL_SECONDS:
            raise ValueError(
                f"sample interval must be at least "
                f"{MIN_SAMPLE_INTERVAL_SECONDS:g} seconds"
            )
        if self.start_time is not None and self.start_time < 0:
            raise ValueError("start time cannot be negative")
        if self.end_time is not None and self.end_time < 0:
            raise ValueError("end time cannot be negative")
        if (
            self.start_time is not None
            and self.end_time is not None
            and self.end_time <= self.start_time
        ):
            raise ValueError("end time must be greater than start time")
        if self.max_frame_width < 640:
            raise ValueError("max frame width must be at least 640 pixels")
        if not 0.0 <= self.frame_similarity_threshold <= 1.0:
            raise ValueError("frame similarity threshold must be inside 0..1")
        if not 0.0 <= self.adaptive_ocr_confidence <= 1.0:
            raise ValueError("adaptive OCR confidence must be inside 0..1")

    def public_dict(self, *, include_runtime_flags: bool = True) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        if not include_runtime_flags:
            data.pop("force", None)
            data.pop("keep_frames", None)
        return data

    def fingerprint(self, source_key: str) -> str:
        payload = {
            "pipeline_version": PIPELINE_VERSION,
            "schema_version": SCHEMA_VERSION,
            "source_key": source_key,
            "config": self.public_dict(include_runtime_flags=False),
        }
        encoded = json.dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class VideoRef:
    raw_input: str
    url: str
    video_id: str | None
    output_key: str


@dataclass
class OcrLine:
    text: str
    confidence: float
    box: list[list[float]]
    role: Literal["dialogue", "speaker_hint", "unknown"] = "dialogue"

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class OcrCandidate:
    text: str
    confidence: float
    quality: float
    lines: list[OcrLine]
    variant: str = "base"

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 6),
            "quality": round(self.quality, 6),
            "variant": self.variant,
            "lines": [line.as_dict() for line in self.lines],
        }


@dataclass
class OcrSample:
    time: float
    text: str
    confidence: float
    quality: float
    signature: str
    reused: bool
    ocr_calls: int
    variant: str
    lines: list[OcrLine] = field(default_factory=list)
    saved_frame: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "time": round(self.time, 3),
            "text": self.text,
            "confidence": round(self.confidence, 6),
            "quality": round(self.quality, 6),
            "signature": self.signature,
            "reused": self.reused,
            "ocr_calls": self.ocr_calls,
            "variant": self.variant,
            "lines": [line.as_dict() for line in self.lines],
            "saved_frame": self.saved_frame,
        }


@dataclass
class Segment:
    start: float
    end: float
    text: str
    confidence: float
    sample_times: list[float]
    reconstruction: str
    source: Literal["subtitle_track", "hard_subtitle_ocr"]
    alternatives: list[str] = field(default_factory=list)

    def as_dict(self, segment_id: int | None = None) -> dict[str, Any]:
        data = {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
            "confidence": round(self.confidence, 6),
            "sample_times": [round(value, 3) for value in self.sample_times],
            "reconstruction": self.reconstruction,
            "source": self.source,
            "alternatives": self.alternatives,
        }
        if segment_id is not None:
            data = {"segment_id": segment_id, **data}
        return data


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    width: int | None
    height: int | None
    duration: float | None


@dataclass
class SubtitleCue:
    start: float
    end: float
    text: str


@dataclass
class SubtitleTrackResult:
    selected_path: Path | None = None
    cues: list[SubtitleCue] = field(default_factory=list)
    files_found: list[str] = field(default_factory=list)
    rejected: list[dict[str, str]] = field(default_factory=list)
    bbdown_exit_code: int | None = None
    title: str | None = None
    video_id: str | None = None
    log_tail: str | None = None
    download_error: str | None = None
    error_traceback: str | None = None

    @property
    def usable(self) -> bool:
        return bool(self.cues)


@dataclass(frozen=True)
class FetchResult:
    subtitle_path: Path | None = None
    video_path: Path | None = None
    source: str = ""


@dataclass(frozen=True)
class AdaptiveOcrResult:
    candidate: OcrCandidate
    calls: int
    tried_variants: list[str]


@dataclass
class AssemblyStats:
    input_samples: int = 0
    readable_samples: int = 0
    temporal_consensus_groups: int = 0
    overlap_joins: int = 0
    dropped_low_confidence_fragments: int = 0

    def as_dict(self) -> dict[str, int]:
        return dataclasses.asdict(self)


@dataclass
class AssemblyResult:
    segments: list[Segment] = field(default_factory=list)
    stats: AssemblyStats = field(default_factory=AssemblyStats)


@dataclass(frozen=True)
class ProcessResult:
    input: str
    output_key: str
    output_directory: str
    success: bool
    skipped: bool
    source_type: str | None
    segment_count: int
    failure_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Input normalization
# ---------------------------------------------------------------------------


def parse_video_inputs(raw_input: str) -> list[str]:
    """Parse newline, whitespace, comma, and numbered input without duplicates."""
    items: list[str] = []
    seen: set[str] = set()
    for raw_line in raw_input.splitlines():
        line = NUMBER_PREFIX_RE.sub("", raw_line.strip(), count=1).strip()
        for token in re.split(r"[\s,\uFF0C]+", line):
            token = NUMBER_PREFIX_RE.sub("", token.strip(), count=1).strip()
            if not token or re.fullmatch(r"\d+[.\uFF0E\u3001)\uFF09:]?", token):
                continue
            if token not in seen:
                seen.add(token)
                items.append(token)
    return items


def is_supported_input(value: str) -> bool:
    value = value.strip()
    return bool(
        re.match(r"^https?://\S+$", value, re.IGNORECASE)
        or re.fullmatch(r"BV[0-9A-Za-z]{10}", value)
        or re.fullmatch(r"(?:av|AV)\d+", value)
    )


def normalize_input(raw: str) -> str:
    """Return a canonical Bilibili URL for a raw id or retain a full URL."""
    raw = raw.strip()
    if re.fullmatch(r"BV[0-9A-Za-z]{10}", raw):
        return f"https://www.bilibili.com/video/{raw}/"
    if re.fullmatch(r"(?:av|AV)\d+", raw):
        return f"https://www.bilibili.com/video/{raw}/"
    if re.match(r"^https?://\S+$", raw, re.IGNORECASE):
        return raw
    raise ValueError(f"unrecognized bilibili input: {raw!r}")


def extract_video_id(url_or_id: str) -> str:
    match = VIDEO_ID_RE.search(url_or_id)
    if not match:
        raise ValueError(f"could not find a BV/AV id in: {url_or_id!r}")
    return match.group(0)


def resolve_video_input(value: str) -> VideoRef:
    value = value.strip()
    if not is_supported_input(value):
        raise ValueError("input must be a Bilibili URL, BV ID, or AV ID")

    bv_match = BV_RE.search(value)
    if bv_match:
        video_id = bv_match.group(1)
        return VideoRef(
            value,
            f"https://www.bilibili.com/video/{video_id}/",
            video_id,
            video_id,
        )

    av_match = AV_RE.search(value)
    if av_match:
        video_id = f"av{av_match.group(1)}"
        return VideoRef(
            value,
            f"https://www.bilibili.com/video/{video_id}/",
            video_id,
            video_id,
        )

    if re.match(
        r"^https?://(?:www\.)?(?:b23\.tv|bilibili\.com)/",
        value,
        re.IGNORECASE,
    ):
        # Let yt-dlp follow short-link redirects so every network request uses
        # the same impersonation and Bilibili headers.
        return _from_resolved_url(value, value)

    raise ValueError("URL must use bilibili.com or b23.tv")


def _from_resolved_url(raw_input: str, url: str) -> VideoRef:
    bv_match = BV_RE.search(url)
    av_match = AV_RE.search(url)
    video_id = (
        bv_match.group(1)
        if bv_match
        else f"av{av_match.group(1)}" if av_match else None
    )
    output_key = video_id or f"url-{hashlib.sha256(url.encode()).hexdigest()[:12]}"
    return VideoRef(raw_input, url, video_id, output_key)


# ---------------------------------------------------------------------------
# yt-dlp network layer
# ---------------------------------------------------------------------------


def yt_dlp_common_args() -> list[str]:
    """CLI-equivalent options kept for tests and compatibility callers."""
    return list(YTDLP_COMMON_ARGS)


def _ydl_base_opts(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "ignoreconfig": True,
        "noplaylist": True,
        "retries": 5,
        "fragment_retries": 5,
        "socket_timeout": 30,
        "http_headers": {
            "Referer": BILIBILI_REFERER,
            "Origin": BILIBILI_ORIGIN,
            "User-Agent": CHROME_UA,
        },
        "impersonate": ImpersonateTarget.from_str("chrome"),
    }
    if extra:
        opts.update(extra)
    return opts


def _run_ydl(url: str, opts: dict[str, Any], *, download: bool) -> dict[str, Any]:
    """Run one library call and retry transient Bilibili 412 responses."""
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=download)
            return info if isinstance(info, dict) else {}
        except yt_dlp.utils.DownloadError as exc:
            last_error = exc
            message = str(exc)
            is_412 = "412" in message or "Precondition Failed" in message
            if is_412 and attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            raise
    if last_error is not None:
        raise last_error
    raise RuntimeError("yt-dlp call ended without a result")


def _download_with_command(
    command: list[str],
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]],
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    """Test seam; production uses the library call above."""
    return runner(
        command,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def download_low_quality_video(
    video: VideoRef,
    destination: Path,
    *,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    existing = _find_videos(destination)
    if existing:
        return existing[0]

    bbdown_error: str | None = None
    bbdown = shutil.which("BBDown")
    if bbdown:
        command = [
            bbdown,
            video.url,
            *BBDOWN_VIDEO_ARGS,
            "--work-dir",
            str(destination),
        ]
        process = _download_with_command(
            command, runner=command_runner, timeout=900
        )
        bbdown_log = (process.stdout or "") + "\n" + (process.stderr or "")
        videos = _find_videos(destination)
        if process.returncode == 0 and videos:
            return videos[0]
        bbdown_error = (
            f"BBDown exited with code {process.returncode}: "
            f"{_last_log_line(bbdown_log)}"
        )
        print(bbdown_error, file=sys.stderr, flush=True)

    if command_runner is not subprocess.run:
        command = [
            "yt-dlp",
            *yt_dlp_common_args(),
            "--no-part",
            "--concurrent-fragments",
            "4",
            "--retries",
            "5",
            "--fragment-retries",
            "5",
            "--format",
            "bestvideo[height>=360][height<=480]/bestvideo[height<=480]/worstvideo/worst",
            "--output",
            str(destination / "source.%(ext)s"),
            video.url,
        ]
        process = _download_with_command(
            command, runner=command_runner, timeout=900
        )
        if process.returncode != 0:
            message = _last_log_line(
                (process.stdout or "") + "\n" + (process.stderr or "")
            )
            detail = (
                f"; yt-dlp fallback: {message}"
                if bbdown_error
                else f": {message}"
            )
            raise RuntimeError(
                f"low-quality video download failed with exit code "
                f"{process.returncode}{detail}"
            )
    else:
        try:
            _run_ydl(
                video.url,
                _ydl_base_opts(
                    {
                        "format": (
                            "bestvideo[height>=360][height<=480]/"
                            "bestvideo[height<=480]/worstvideo/worst"
                        ),
                        "outtmpl": str(destination / "source.%(ext)s"),
                        "nopart": True,
                        "concurrent_fragment_downloads": 4,
                    }
                ),
                download=True,
            )
        except Exception as exc:
            if bbdown_error:
                raise RuntimeError(
                    f"{bbdown_error}; yt-dlp fallback failed: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            raise

    videos = _find_videos(destination)
    if not videos:
        raise RuntimeError(
            "download completed without producing a non-empty video file"
        )
    return videos[0]


def fetch_subtitle_or_video(url: str, workdir: Path) -> FetchResult:
    """Use subtitle tracks first, then the low-quality video fallback."""
    workdir.mkdir(parents=True, exist_ok=True)
    video = resolve_video_input(url)
    track = download_best_subtitle_track(video, workdir / "subtitle-track")
    if track.usable and track.selected_path is not None:
        return FetchResult(
            subtitle_path=track.selected_path,
            source="subtitle-track",
        )

    video_path = download_low_quality_video(video, workdir / "video")
    return FetchResult(video_path=video_path, source="video-360p")


# ---------------------------------------------------------------------------
# Subtitle-track parsing and selection
# ---------------------------------------------------------------------------


def parse_timecode(value: str) -> float:
    raw = value.strip().replace(",", ".")
    parts = raw.split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    raise ValueError(f"unsupported subtitle timecode: {value}")


def clean_subtitle_text(value: str) -> str:
    value = value.replace(r"\N", "\n").replace(r"\n", "\n")
    value = re.sub(r"\{[^}]*\}", "", value)
    value = re.sub(r"<[^>]+>", "", value)
    value = html.unescape(value)
    lines = [re.sub(r"\s+", " ", line).strip() for line in value.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def parse_srt_or_vtt(text: str) -> list[SubtitleCue]:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    cues: list[SubtitleCue] = []
    index = 0
    while index < len(lines):
        match = TIME_RANGE_RE.search(lines[index])
        if not match:
            index += 1
            continue
        start = parse_timecode(match.group("start"))
        end = parse_timecode(match.group("end"))
        index += 1
        payload: list[str] = []
        while index < len(lines) and lines[index].strip():
            payload.append(lines[index])
            index += 1
        cleaned = clean_subtitle_text("\n".join(payload))
        if cleaned and end > start:
            cues.append(SubtitleCue(start, end, cleaned))
        index += 1
    return cues


def parse_ass(text: str) -> list[SubtitleCue]:
    cues: list[SubtitleCue] = []
    for raw_line in text.replace("\r\n", "\n").splitlines():
        if not raw_line.startswith("Dialogue:"):
            continue
        fields = raw_line.split(",", 9)
        if len(fields) < 10:
            continue
        try:
            start = parse_timecode(fields[1])
            end = parse_timecode(fields[2])
        except ValueError:
            continue
        cleaned = clean_subtitle_text(fields[9])
        if cleaned and end > start:
            cues.append(SubtitleCue(start, end, cleaned))
    return cues


def parse_json_track(text: str) -> list[SubtitleCue]:
    data = json.loads(text)
    rows = data.get("body", data) if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return []
    cues: list[SubtitleCue] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        start = row.get("from", row.get("start", row.get("start_time")))
        end = row.get("to", row.get("end", row.get("end_time")))
        content = row.get("content", row.get("text", row.get("body")))
        if start is None or end is None or content is None:
            continue
        try:
            start_value, end_value = float(start), float(end)
        except (TypeError, ValueError):
            continue
        cleaned = clean_subtitle_text(str(content))
        if cleaned and end_value > start_value:
            cues.append(SubtitleCue(start_value, end_value, cleaned))
    return cues


def parse_xml_track(text: str) -> list[SubtitleCue]:
    root = ET.fromstring(text)
    cues: list[SubtitleCue] = []
    for node in root.iter():
        if node.tag.lower().split("}")[-1] not in {"text", "p", "d"}:
            continue
        start_value = node.attrib.get("start") or node.attrib.get("from")
        if start_value is None and "p" in node.attrib:
            start_value = node.attrib["p"].split(",", 1)[0]
        if start_value is None:
            continue
        duration = node.attrib.get("dur") or node.attrib.get("duration")
        end_value = node.attrib.get("end") or node.attrib.get("to")
        try:
            start = float(start_value)
            end = (
                float(end_value)
                if end_value is not None
                else start + float(duration or 0)
            )
        except (TypeError, ValueError):
            continue
        cleaned = clean_subtitle_text("".join(node.itertext()))
        if cleaned and end > start:
            cues.append(SubtitleCue(start, end, cleaned))
    return cues


def parse_subtitle_file(path: Path) -> list[SubtitleCue]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    suffix = path.suffix.lower()
    if suffix in {".ass", ".ssa"}:
        return parse_ass(text)
    if suffix == ".json":
        return parse_json_track(text)
    if suffix == ".xml":
        return parse_xml_track(text)
    return parse_srt_or_vtt(text)


def segments_from_subtitle_file(path: Path) -> list[dict[str, Any]]:
    return [
        {
            "start": cue.start,
            "end": cue.end,
            "text": cue.text,
            "confidence": 1.0,
            "sample_times": [cue.start],
            "reconstruction": "direct_subtitle_track",
            "source": "subtitle_track",
            "alternatives": [],
        }
        for cue in parse_subtitle_file(path)
        if is_dialogue_cue(cue)
    ]


def is_dialogue_cue(cue: SubtitleCue) -> bool:
    compact = re.sub(r"\s+", "", cue.text)
    return bool(compact) and not NON_DIALOGUE_RE.fullmatch(compact)


def track_score(path: Path, dialogue_count: int) -> tuple[int, int, str]:
    name = path.name.casefold()
    ai = any(token in name for token in ("ai", "auto", "asr", "\u81ea\u52a8"))
    chinese = any(
        token in name for token in ("zh", "chi", "chs", "cht", "\u4e2d\u6587")
    )
    japanese = any(token in name for token in ("ja", "jp", "jpn", "\u65e5\u6587"))
    priority = 0 if chinese and not ai else 1 if chinese else 2 if japanese else 3
    return priority, -dialogue_count, name


def extract_title(log: str) -> str | None:
    patterns = (r"(?:Title)\s*[:\uFF1A]\s*(.+)", r"\[P\d+\](.+)")
    for line in log.splitlines():
        for pattern in patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                return match.group(1).strip()
    return None


def download_best_subtitle_track(
    video: VideoRef,
    work_dir: Path,
    *,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> SubtitleTrackResult:
    """Download and select the best usable subtitle without downloading video."""
    work_dir.mkdir(parents=True, exist_ok=True)
    result = SubtitleTrackResult(video_id=video.video_id)
    info: dict[str, Any] = {}

    bbdown_error: str | None = None
    bbdown = shutil.which("BBDown")
    if bbdown:
        command = [
            bbdown,
            video.url,
            *BBDOWN_SUBTITLE_ARGS,
            "--work-dir",
            str(work_dir),
        ]
        process = _download_with_command(
            command, runner=command_runner, timeout=240
        )
        log = (process.stdout or "") + "\n" + (process.stderr or "")
        result.bbdown_exit_code = process.returncode
        result.title = extract_title(log)
        result.log_tail = _last_log_line(log)
        if process.returncode != 0:
            bbdown_error = (
                f"BBDown exited with code {process.returncode}: "
                f"{result.log_tail}"
            )
            print(bbdown_error, file=sys.stderr, flush=True)

    if not bbdown or not _has_usable_subtitle_file(work_dir):
        command = [
            "yt-dlp",
            *yt_dlp_common_args(),
            "--skip-download",
            "--write-subs",
            "--write-auto-subs",
            "--sub-langs",
            "all,-danmaku",
            "--sub-format",
            "srt/vtt/best",
            "--write-info-json",
            "--output",
            str(work_dir / "%(id)s.%(language)s.%(ext)s"),
            video.url,
        ]
        if command_runner is not subprocess.run:
            process = _download_with_command(
                command, runner=command_runner, timeout=240
            )
            log = (process.stdout or "") + "\n" + (process.stderr or "")
            result.log_tail = _last_log_line(log)
            if process.returncode != 0:
                result.download_error = result.log_tail
        else:
            try:
                info = _run_ydl(
                    video.url,
                    _ydl_base_opts(
                        {
                            "skip_download": True,
                            "writesubtitles": True,
                            "writeautomaticsub": True,
                            "subtitleslangs": ["all", "-danmaku"],
                            "subtitlesformat": "srt/vtt/best",
                            "writeinfojson": True,
                            "outtmpl": str(work_dir / "%(id)s.%(language)s.%(ext)s"),
                        }
                    ),
                    download=True,
                )
            except Exception as exc:
                result.error_traceback = traceback.format_exc()
                print(result.error_traceback, file=sys.stderr, flush=True)
                result.download_error = f"{type(exc).__name__}: {exc}"
                result.log_tail = str(exc)

    metadata = _read_info_metadata(work_dir)
    merged_info = {**metadata, **info}
    result.title = (
        str(merged_info.get("title"))
        if merged_info.get("title")
        else result.title
    )
    result.video_id = (
        str(merged_info.get("id"))
        if merged_info.get("id")
        else result.video_id
    )

    files = sorted(
        path
        for path in work_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_SUBTITLE_SUFFIXES
        and not path.name.endswith(".info.json")
    )
    result.files_found = [str(path.relative_to(work_dir)) for path in files]

    candidates: list[tuple[tuple[int, int, str], Path, list[SubtitleCue]]] = []
    for path in files:
        try:
            cues = [cue for cue in parse_subtitle_file(path) if is_dialogue_cue(cue)]
        except (
            OSError,
            ValueError,
            TypeError,
            json.JSONDecodeError,
            ET.ParseError,
        ) as exc:
            result.rejected.append(
                {"file": str(path.relative_to(work_dir)), "reason": str(exc)}
            )
            continue
        if not cues:
            result.rejected.append(
                {
                    "file": str(path.relative_to(work_dir)),
                    "reason": "no usable dialogue cues",
                }
            )
            continue
        candidates.append((track_score(path, len(cues)), path, cues))

    if candidates:
        _score, result.selected_path, result.cues = min(
            candidates, key=lambda item: item[0]
        )
    elif bbdown_error and not result.download_error:
        result.download_error = bbdown_error
    return result


def cues_to_segments(cues: Sequence[SubtitleCue]) -> list[Segment]:
    return [
        Segment(
            start=cue.start,
            end=cue.end,
            text=cue.text,
            confidence=1.0,
            sample_times=[cue.start],
            reconstruction="direct_subtitle_track",
            source="subtitle_track",
        )
        for cue in cues
    ]


# ---------------------------------------------------------------------------
# Video probing, screenshot sampling, and evidence-frame retention
# ---------------------------------------------------------------------------


def probe_video(
    video_path: Path,
    *,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> VideoInfo:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height:format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    process = command_runner(command, text=True, capture_output=True, check=False)
    if process.returncode != 0:
        return VideoInfo(video_path, None, None, None)
    try:
        payload = json.loads(process.stdout or "{}")
        stream = (payload.get("streams") or [{}])[0]
        duration_value = (payload.get("format") or {}).get("duration")
        return VideoInfo(
            video_path,
            _optional_int(stream.get("width")),
            _optional_int(stream.get("height")),
            float(duration_value) if duration_value is not None else None,
        )
    except (ValueError, TypeError, json.JSONDecodeError):
        return VideoInfo(video_path, None, None, None)


def build_frame_extraction_command(
    video_path: Path,
    frame_dir: Path,
    config: PipelineConfig,
) -> list[str]:
    """Build one sparse FFmpeg decode; the config enforces the 3s floor."""
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    if config.start_time is not None:
        command.extend(["-ss", _number(config.start_time)])
    command.extend(["-i", str(video_path)])
    if config.end_time is not None:
        duration = config.end_time - (config.start_time or 0.0)
        command.extend(["-t", _number(duration)])
    interval = _number(config.sample_interval_seconds)
    filters = (
        f"select='isnan(prev_selected_t)+gte(t-prev_selected_t\\,{interval})',"
        f"scale='min({config.max_frame_width},iw)':-2"
    )
    command.extend(
        [
            "-an",
            "-vf",
            filters,
            "-q:v",
            "3",
            "-fps_mode",
            "vfr",
            str(frame_dir / "frame_%08d.jpg"),
        ]
    )
    return command


def extract_frames(
    video_path: Path,
    frame_dir: Path,
    config: PipelineConfig,
    *,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> list[Path]:
    frame_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(frame_dir.glob("frame_*.jpg"))
    if existing:
        return existing
    command = build_frame_extraction_command(video_path, frame_dir, config)
    process = command_runner(
        command,
        text=True,
        capture_output=True,
        timeout=1800,
        check=False,
    )
    if process.returncode != 0:
        message = _last_log_line(
            (process.stdout or "") + "\n" + (process.stderr or "")
        )
        raise RuntimeError(f"FFmpeg frame extraction failed: {message}")
    frames = sorted(frame_dir.glob("frame_*.jpg"))
    if not frames:
        raise RuntimeError("FFmpeg completed without producing screenshots")
    return frames


def sample_frames(
    video_path: Path,
    out_dir: Path,
    interval: float,
    start: str | float | None = None,
    end: str | float | None = None,
) -> list[Path]:
    config = PipelineConfig(
        sample_interval_seconds=interval,
        start_time=_coerce_time(start),
        end_time=_coerce_time(end),
    )
    return extract_frames(video_path, out_dir, config)


def crop_relative(image: Image.Image, region: CropRegion) -> Image.Image:
    width, height = image.size
    box = (
        round(width * region.left),
        round(height * region.top),
        round(width * region.right),
        round(height * region.bottom),
    )
    return image.crop(box)


def subtitle_signature(image: Image.Image) -> bytes:
    """Compact edge signature for conservative near-static-frame reuse."""
    grayscale = ImageOps.grayscale(image).resize((128, 32))
    edges = grayscale.filter(ImageFilter.FIND_EDGES)
    pixels = (
        edges.get_flattened_data()
        if hasattr(edges, "get_flattened_data")
        else edges.getdata()
    )
    return bytes(1 if value >= 42 else 0 for value in pixels)


def _subtitle_band_signature(image: Image.Image) -> str:
    width, height = image.size
    band = image.crop((0, int(height * 0.82), width, height))
    band = band.convert("L").resize((32, 8))
    return hashlib.sha1(band.tobytes()).hexdigest()


def signature_hex(signature: bytes) -> str:
    packed = bytearray()
    for offset in range(0, len(signature), 8):
        value = 0
        for bit in signature[offset : offset + 8]:
            value = (value << 1) | int(bool(bit))
        packed.append(value)
    return hashlib.blake2s(packed, digest_size=8).hexdigest()


def signature_distance(left: bytes, right: bytes) -> float:
    if not left or len(left) != len(right):
        return 1.0
    return sum(a != b for a, b in zip(left, right)) / len(left)


def signatures_are_similar(
    left: bytes | None, right: bytes, threshold: float
) -> bool:
    return left is not None and signature_distance(left, right) <= threshold


def save_evidence_frame(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image.convert("RGB").save(
            destination, format="JPEG", quality=82, optimize=True
        )


def frame_time(index: int, config: PipelineConfig) -> float:
    return (config.start_time or 0.0) + index * config.sample_interval_seconds


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------


SPEAKER_HINT_REGIONS = {
    "left_upper": CropRegion(top=0.54, bottom=0.68, left=0.14, right=0.42),
    "left_lower": CropRegion(top=0.62, bottom=0.76, left=0.06, right=0.30),
    "center": CropRegion(top=0.62, bottom=0.76, left=0.38, right=0.64),
}


class OcrEngine(Protocol):
    name: str

    def recognize(self, image: Image.Image) -> Sequence[object]: ...


class RapidOcrEngine:
    name = "rapidocr_3_onnxruntime"

    def __init__(self) -> None:
        from rapidocr import RapidOCR

        self._engine = RapidOCR()

    def recognize(self, image: Image.Image) -> Sequence[object]:
        rgb = image.convert("RGB")
        bgr = np.asarray(rgb)[:, :, ::-1].copy()
        raw = self._engine(bgr)
        return _parse_rapidocr_result(raw)


def _parse_rapidocr_result(raw: object) -> list[object]:
    if raw is None:
        return []
    if hasattr(raw, "boxes") and hasattr(raw, "txts") and hasattr(raw, "scores"):
        boxes = getattr(raw, "boxes")
        texts = getattr(raw, "txts")
        scores = getattr(raw, "scores")
        boxes = boxes if boxes is not None else []
        texts = texts if texts is not None else []
        scores = scores if scores is not None else []
        return [
            [
                box.tolist() if hasattr(box, "tolist") else box,
                text,
                score,
            ]
            for box, text, score in zip(boxes, texts, scores)
        ]
    rows: object = raw[0] if isinstance(raw, tuple) else raw
    if rows is None:
        return []
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
        if len(rows) == 3 and all(
            isinstance(value, Sequence) and not isinstance(value, (str, bytes))
            for value in rows
        ):
            first, second, third = rows
            if len(first) == len(second) == len(third):
                return [
                    [
                        box.tolist() if hasattr(box, "tolist") else box,
                        text,
                        score,
                    ]
                    for box, text, score in zip(first, second, third)
                ]
        return list(rows)
    return []


def recognize_adaptive(
    image: Image.Image,
    crop_region: CropRegion,
    engine: OcrEngine,
    *,
    confidence_threshold: float = 0.78,
) -> AdaptiveOcrResult:
    """Use one normal pass, then at most two targeted enhancement passes."""
    candidates: list[OcrCandidate] = []
    tried: list[str] = []

    base = candidate_from_result(
        engine.recognize(image), image.size, crop_region, variant="base"
    )
    candidates.append(base)
    tried.append("base")

    if _needs_enhancement(base, confidence_threshold):
        enhanced_image = enhanced_variant(image)
        enhanced = candidate_from_result(
            engine.recognize(enhanced_image),
            enhanced_image.size,
            crop_region,
            variant="enhanced",
        )
        candidates.append(enhanced)
        tried.append("enhanced")

    best_so_far = max(candidates, key=_candidate_rank)
    if _needs_binary_pass(best_so_far):
        binary_image = binary_variant(image)
        binary = candidate_from_result(
            engine.recognize(binary_image),
            binary_image.size,
            crop_region,
            variant="binary",
        )
        candidates.append(binary)
        tried.append("binary")

    best = max(candidates, key=_candidate_rank)
    return AdaptiveOcrResult(best, len(candidates), tried)


def candidate_from_result(
    raw_result: Sequence[object],
    image_size: tuple[int, int],
    crop_region: CropRegion,
    *,
    variant: str,
) -> OcrCandidate:
    parsed: list[OcrLine] = []
    for raw_line in raw_result or []:
        try:
            box_value = raw_line[0]  # type: ignore[index]
            text = str(raw_line[1]).strip()  # type: ignore[index]
            confidence = float(raw_line[2])  # type: ignore[index]
            box = [
                [float(point[0]), float(point[1])]
                for point in box_value
            ]
        except (IndexError, TypeError, ValueError):
            continue
        if not text or not box:
            continue
        parsed.append(
            OcrLine(text=text, confidence=confidence, box=box, role="unknown")
        )

    ordered: list[OcrLine] = []
    for row in cluster_reading_rows(parsed):
        role = classify_row_role(row, image_size, crop_region)
        for line in sorted(row, key=_line_left):
            line.role = role
            ordered.append(line)
    dialogue_lines = [line for line in ordered if line.role != "speaker_hint"]
    text = join_ocr_lines([line.text for line in dialogue_lines])
    confidence = weighted_confidence(dialogue_lines)
    quality = candidate_quality(text, confidence, dialogue_lines)
    return OcrCandidate(text, confidence, quality, ordered, variant)


def classify_line_role(
    text: str,
    box: list[list[float]],
    image_size: tuple[int, int],
    crop_region: CropRegion,
) -> str:
    return classify_row_role(
        [OcrLine(text=text, confidence=0.0, box=box, role="unknown")],
        image_size,
        crop_region,
    )


def classify_row_role(
    row: Sequence[OcrLine],
    image_size: tuple[int, int],
    crop_region: CropRegion,
) -> str:
    """Mask only a short row fully contained by a known speaker-label region."""
    width, height = image_size
    if width <= 0 or height <= 0 or not row:
        return "unknown"
    points = [point for line in row for point in line.box]
    local_left = min(point[0] for point in points)
    local_right = max(point[0] for point in points)
    local_top = min(point[1] for point in points)
    local_bottom = max(point[1] for point in points)
    global_left = crop_region.left + (local_left / width) * (
        crop_region.right - crop_region.left
    )
    global_right = crop_region.left + (local_right / width) * (
        crop_region.right - crop_region.left
    )
    global_top = crop_region.top + (local_top / height) * (
        crop_region.bottom - crop_region.top
    )
    global_bottom = crop_region.top + (local_bottom / height) * (
        crop_region.bottom - crop_region.top
    )
    compact = re.sub(r"\s+", "", join_ocr_lines([line.text for line in row]))
    margin = 0.025
    if len(compact) <= 10 and any(
        global_left >= region.left - margin
        and global_right <= region.right + margin
        and global_top >= region.top - margin
        and global_bottom <= region.bottom + margin
        for region in SPEAKER_HINT_REGIONS.values()
    ):
        return "speaker_hint"
    return "dialogue"


def join_ocr_lines(lines: Sequence[str]) -> str:
    output = ""
    for raw in lines:
        value = re.sub(r"\s+", " ", raw).strip()
        if not value:
            continue
        if output and _needs_word_space(output[-1], value[0]):
            output += " "
        output += value
    return output.strip()


def weighted_confidence(lines: Sequence[OcrLine]) -> float:
    weights = [max(1, meaningful_length(line.text)) for line in lines]
    total = sum(weights)
    if not total:
        return 0.0
    return sum(
        line.confidence * weight for line, weight in zip(lines, weights)
    ) / total


def candidate_quality(
    text: str, confidence: float, lines: Sequence[OcrLine]
) -> float:
    length = meaningful_length(text)
    if length == 0:
        return 0.0
    length_score = min(1.0, math.log2(length + 1) / 4.0)
    line_score = min(1.0, len(lines) / 2.0)
    score = confidence * 0.72 + length_score * 0.23 + line_score * 0.05
    if length == 1:
        score -= 0.22
    elif length == 2:
        score -= 0.06
    return max(0.0, min(1.0, score))


def meaningful_length(text: str) -> int:
    return len(MEANINGFUL_CHAR_RE.findall(text or ""))


def sort_reading_order(lines: Sequence[OcrLine]) -> list[OcrLine]:
    return [
        line
        for row in cluster_reading_rows(lines)
        for line in sorted(row, key=_line_left)
    ]


def cluster_reading_rows(lines: Sequence[OcrLine]) -> list[list[OcrLine]]:
    pending = sorted(
        lines, key=lambda line: (_line_center_y(line), _line_left(line))
    )
    rows: list[list[OcrLine]] = []
    for line in pending:
        if not rows:
            rows.append([line])
            continue
        current = rows[-1]
        row_center = sum(_line_center_y(item) for item in current) / len(current)
        row_height = max(_line_height(item) for item in current)
        tolerance = max(row_height, _line_height(line)) * 0.65
        if abs(_line_center_y(line) - row_center) <= tolerance:
            current.append(line)
        else:
            rows.append([line])
    return rows


def enhanced_variant(image: Image.Image) -> Image.Image:
    grayscale = ImageOps.grayscale(image)
    width, height = grayscale.size
    enlarged = grayscale.resize(
        (max(1, width * 2), max(1, height * 2)),
        Image.Resampling.LANCZOS,
    )
    contrasted = ImageOps.autocontrast(enlarged, cutoff=1)
    contrasted = ImageEnhance.Contrast(contrasted).enhance(1.35)
    return contrasted.filter(
        ImageFilter.UnsharpMask(radius=1.2, percent=170, threshold=2)
    )


def binary_variant(image: Image.Image) -> Image.Image:
    enhanced = enhanced_variant(image)
    threshold = otsu_threshold(enhanced.histogram())
    return enhanced.point(
        lambda value: 255 if value >= threshold else 0,
        mode="1",
    ).convert("L")


def otsu_threshold(histogram: Sequence[int]) -> int:
    total = sum(histogram)
    if total <= 0:
        return 128
    weighted_sum = sum(index * count for index, count in enumerate(histogram))
    background_weight = 0
    background_sum = 0.0
    best_variance = -1.0
    best_threshold = 128
    for threshold, count in enumerate(histogram):
        background_weight += count
        if background_weight == 0:
            continue
        foreground_weight = total - background_weight
        if foreground_weight == 0:
            break
        background_sum += threshold * count
        background_mean = background_sum / background_weight
        foreground_mean = (weighted_sum - background_sum) / foreground_weight
        variance = (
            background_weight
            * foreground_weight
            * (background_mean - foreground_mean) ** 2
        )
        if variance > best_variance:
            best_variance = variance
            best_threshold = threshold
    return best_threshold


class OcrRunner:
    """Cache-aware OCR adapter with no more than three passes per new frame."""

    def __init__(
        self,
        engine: OcrEngine | None = None,
        *,
        similarity_threshold: float = 0.018,
    ) -> None:
        self._engine = engine or RapidOcrEngine()
        self._similarity_threshold = similarity_threshold
        self._cache: dict[bytes, OcrCandidate] = {}
        self._last_signature: bytes | None = None
        self._last_candidate: OcrCandidate | None = None
        self.calls = 0
        self.reused = 0
        self.enhance_calls = 0

    def run(
        self,
        frame_path: Path,
        crop_region: CropRegion = CropRegion(),
        *,
        confidence_threshold: float = 0.78,
    ) -> list[OcrLine]:
        with Image.open(frame_path) as image:
            crop = crop_relative(image.convert("RGB"), crop_region)
            candidate, _reused, _calls, _variants = self.run_image(
                crop,
                crop_region,
                confidence_threshold=confidence_threshold,
            )
        return candidate.lines

    def run_image(
        self,
        image: Image.Image,
        crop_region: CropRegion,
        *,
        confidence_threshold: float = 0.78,
    ) -> tuple[OcrCandidate, bool, int, list[str]]:
        signature = subtitle_signature(image)
        cached = self._cache.get(signature)
        if cached is not None:
            self.reused += 1
            self._last_signature = signature
            self._last_candidate = cached
            return cached, True, 0, ["reused"]
        if (
            self._last_signature is not None
            and self._last_candidate is not None
            and signatures_are_similar(
                self._last_signature, signature, self._similarity_threshold
            )
        ):
            self.reused += 1
            self._last_signature = signature
            return self._last_candidate, True, 0, ["reused"]

        adaptive = recognize_adaptive(
            image,
            crop_region,
            self._engine,
            confidence_threshold=confidence_threshold,
        )
        self.calls += adaptive.calls
        self.enhance_calls += max(0, adaptive.calls - 1)
        self._cache[signature] = adaptive.candidate
        self._last_signature = signature
        self._last_candidate = adaptive.candidate
        return (
            adaptive.candidate,
            False,
            adaptive.calls,
            adaptive.tried_variants,
        )

    @staticmethod
    def _enhance(image: Image.Image, attempt: int) -> Image.Image:
        if attempt == 0:
            return ImageEnhance.Contrast(image).enhance(1.8)
        if attempt == 1:
            grayscale = image.convert("L")
            return ImageOps.autocontrast(grayscale).point(
                lambda pixel: 255 if pixel > 150 else 0
            )
        return ImageOps.equalize(image.convert("L"))


def _candidate_rank(candidate: OcrCandidate) -> tuple[float, int, float]:
    length = meaningful_length(candidate.text)
    complete_sentence_bonus = min(length, 12) * 0.008
    return (
        candidate.quality + complete_sentence_bonus,
        length,
        candidate.confidence,
    )


def _needs_enhancement(candidate: OcrCandidate, threshold: float) -> bool:
    return (
        not candidate.text
        or meaningful_length(candidate.text) < 4
        or candidate.confidence < threshold
        or candidate.quality < 0.68
    )


def _needs_binary_pass(candidate: OcrCandidate) -> bool:
    return (
        not candidate.text
        or meaningful_length(candidate.text) <= 1
        or candidate.confidence < 0.58
        or candidate.quality < 0.50
    )


def _line_center_y(line: OcrLine) -> float:
    return sum(point[1] for point in line.box) / len(line.box)


def _line_height(line: OcrLine) -> float:
    values = [point[1] for point in line.box]
    return max(values) - min(values)


def _line_left(line: OcrLine) -> float:
    return min(point[0] for point in line.box)


def _needs_word_space(left: str, right: str) -> bool:
    return (
        left.isascii()
        and right.isascii()
        and left.isalnum()
        and right.isalnum()
    )


# ---------------------------------------------------------------------------
# Conservative sentence reconstruction
# ---------------------------------------------------------------------------


def assemble_sentences(
    samples: Sequence[OcrSample], sample_interval: float
) -> AssemblyResult:
    """Choose supported whole captions and merge only reliable overlaps."""
    stats = AssemblyStats(input_samples=len(samples))
    groups: list[list[OcrSample]] = []
    active: list[OcrSample] = []

    for sample in samples:
        if not is_readable_text(sample.text):
            if active:
                groups.append(active)
                active = []
            continue
        stats.readable_samples += 1
        if active and samples_describe_same_caption(active[-1], sample):
            active.append(sample)
        else:
            if active:
                groups.append(active)
            active = [sample]
    if active:
        groups.append(active)

    segments: list[Segment] = []
    for group in groups:
        segment = segment_from_group(group, sample_interval)
        if should_drop_fragment(segment):
            stats.dropped_low_confidence_fragments += 1
            continue
        if len(group) > 1:
            stats.temporal_consensus_groups += 1
        segments.append(segment)

    repaired: list[Segment] = []
    for segment in segments:
        if repaired:
            joined = join_overlapping_segments(
                repaired[-1], segment, sample_interval
            )
            if joined is not None:
                repaired[-1] = joined
                stats.overlap_joins += 1
                continue
        repaired.append(segment)

    for previous, current in zip(repaired, repaired[1:]):
        previous.end = min(previous.end, current.start)
    return AssemblyResult(repaired, stats)


def reconstruct_sentences(
    frame_lines: Sequence[tuple[float, Sequence[OcrLine]]],
    sample_interval: float = MIN_SAMPLE_INTERVAL_SECONDS,
) -> list[dict[str, Any]]:
    """Compatibility facade returning the compact dict form from the proposal."""
    samples: list[OcrSample] = []
    for timestamp, lines in frame_lines:
        dialogue = [line for line in lines if line.role != "speaker_hint"]
        text = join_ocr_lines([line.text for line in dialogue])
        confidence = weighted_confidence(dialogue)
        samples.append(
            OcrSample(
                time=timestamp,
                text=text,
                confidence=confidence,
                quality=candidate_quality(text, confidence, dialogue),
                signature="",
                reused=False,
                ocr_calls=1,
                variant="base",
                lines=list(lines),
            )
        )
    return [
        {
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
            "confidence": segment.confidence,
            "sample_times": segment.sample_times,
            "reconstruction": segment.reconstruction,
            "source": segment.source,
            "alternatives": segment.alternatives,
        }
        for segment in assemble_sentences(samples, sample_interval).segments
    ]


def _overlap_merge(left: str, right: str) -> str:
    for size in range(min(len(left), len(right)), 1, -1):
        if left[-size:] == right[:size]:
            return left + right[size:]
    return ""


def samples_describe_same_caption(left: OcrSample, right: OcrSample) -> bool:
    left_key = comparison_key(left.text)
    right_key = comparison_key(right.text)
    if not left_key or not right_key:
        return False
    if left_key == right_key or left_key in right_key or right_key in left_key:
        return True
    ratio = difflib.SequenceMatcher(None, left_key, right_key).ratio()
    shorter = min(len(left_key), len(right_key))
    return ratio >= (0.88 if shorter >= 8 else 0.92)


def segment_from_group(
    group: Sequence[OcrSample], sample_interval: float
) -> Segment:
    best = choose_whole_sentence_candidate(group)
    unique_texts = _unique_preserving_order(
        sample.text for sample in group if sample.text
    )
    confidence = sum(sample.confidence for sample in group) / len(group)
    reconstruction = "temporal_consensus" if len(group) > 1 else "direct_ocr"
    return Segment(
        start=group[0].time,
        end=group[-1].time + sample_interval,
        text=best.text.strip(),
        confidence=confidence,
        sample_times=[sample.time for sample in group],
        reconstruction=reconstruction,
        source="hard_subtitle_ocr",
        alternatives=[text for text in unique_texts if text != best.text],
    )


def choose_whole_sentence_candidate(group: Sequence[OcrSample]) -> OcrSample:
    best_quality = max(sample.quality for sample in group)
    credible = [
        sample
        for sample in group
        if sample.confidence >= 0.55
        and sample.quality >= best_quality - 0.16
    ]
    if credible:
        return max(
            credible,
            key=lambda sample: (
                meaningful_length(sample.text),
                _group_support(sample, group),
                sample.quality,
            ),
        )
    return max(group, key=_sample_rank)


def join_overlapping_segments(
    left: Segment, right: Segment, sample_interval: float
) -> Segment | None:
    if right.start - left.end > sample_interval * 0.5:
        return None
    if left.text.rstrip().endswith(TERMINAL_PUNCTUATION):
        return None

    left_text = normalized_spacing(left.text)
    right_text = normalized_spacing(right.text)
    overlap = suffix_prefix_overlap(left_text, right_text)
    if overlap < 2:
        return None
    merged_text = left_text + right_text[overlap:]
    if merged_text == left_text or merged_text == right_text:
        return None
    total_samples = len(left.sample_times) + len(right.sample_times)
    confidence = (
        left.confidence * len(left.sample_times)
        + right.confidence * len(right.sample_times)
    ) / total_samples
    return Segment(
        start=left.start,
        end=right.end,
        text=merged_text,
        confidence=confidence,
        sample_times=left.sample_times + right.sample_times,
        reconstruction="overlap_join",
        source="hard_subtitle_ocr",
        alternatives=_unique_preserving_order(
            [*left.alternatives, left.text, *right.alternatives, right.text]
        ),
    )


def suffix_prefix_overlap(left: str, right: str, max_window: int = 12) -> int:
    left_normalized = normalized_spacing(left)
    right_normalized = normalized_spacing(right)
    limit = min(len(left_normalized), len(right_normalized), max_window)
    for size in range(limit, 0, -1):
        if left_normalized[-size:] == right_normalized[:size]:
            return size
    return 0


def should_drop_fragment(segment: Segment) -> bool:
    length = meaningful_length(segment.text)
    return length == 0 or (length == 1 and segment.confidence < 0.72)


def clamp_segment_end_times(
    segments: Sequence[Segment], timeline_end: float
) -> list[Segment]:
    output: list[Segment] = []
    for segment in segments:
        segment.end = min(segment.end, timeline_end)
        if segment.end > segment.start:
            output.append(segment)
    return output


def is_readable_text(text: str) -> bool:
    compact = compact_text(text)
    return bool(compact) and compact not in {
        "[unrecognized]",
        "[unabletorecognize]",
    }


def compact_text(text: str) -> str:
    return re.sub(r"\s+", "", text or "").strip()


def normalized_spacing(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def comparison_key(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", text or "", flags=re.UNICODE).casefold()


def _sample_rank(sample: OcrSample) -> tuple[float, int, float]:
    support = sample.quality * 0.65 + sample.confidence * 0.35
    return support, meaningful_length(sample.text), sample.confidence


def _group_support(
    candidate: OcrSample, group: Sequence[OcrSample]
) -> float:
    candidate_key = comparison_key(candidate.text)
    if not candidate_key:
        return 0.0
    return sum(
        difflib.SequenceMatcher(
            None, candidate_key, comparison_key(other.text)
        ).ratio()
        for other in group
    ) / len(group)


def _unique_preserving_order(values: Iterable[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return output


# ---------------------------------------------------------------------------
# Atomic output and manifest cache
# ---------------------------------------------------------------------------


CORE_OUTPUTS = {
    "segments": "segments.jsonl",
    "subtitle": "subtitle.srt",
    "transcript": "transcript.md",
    "source_card": "source-card.md",
    "frame_index": "frames-index.jsonl",
    "ocr_status": "ocr-status.json",
}


class OutputStore:
    def __init__(self, root: Path, output_key: str) -> None:
        self.root = root
        self.output_key = output_key
        self.directory = root / output_key
        self.directory.mkdir(parents=True, exist_ok=True)

    @property
    def manifest_path(self) -> Path:
        return self.directory / "manifest.json"

    @property
    def frames_directory(self) -> Path:
        return self.directory / "frames"

    def cache_is_valid(self, fingerprint: str) -> bool:
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return False
        if (
            manifest.get("status") != "success"
            or manifest.get("fingerprint") != fingerprint
        ):
            return False
        return all(
            (self.directory / filename).is_file()
            and (self.directory / filename).stat().st_size > 0
            for key, filename in CORE_OUTPUTS.items()
            if key != "frame_index"
        )

    def write_segments(self, segments: Sequence[Segment]) -> None:
        content = "".join(
            json.dumps(
                segment.as_dict(index),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
            for index, segment in enumerate(segments, start=1)
        )
        atomic_write_text(self.directory / CORE_OUTPUTS["segments"], content)
        atomic_write_text(
            self.directory / CORE_OUTPUTS["subtitle"],
            segments_to_srt(segments),
        )
        atomic_write_text(
            self.directory / CORE_OUTPUTS["transcript"],
            segments_to_markdown(segments),
        )

    def write_frame_index(self, samples: Sequence[OcrSample]) -> None:
        content = "".join(
            json.dumps(
                sample.as_dict(),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
            for sample in samples
        )
        atomic_write_text(self.directory / CORE_OUTPUTS["frame_index"], content)

    def write_ocr_status(self, payload: dict[str, Any]) -> None:
        atomic_write_json(self.directory / CORE_OUTPUTS["ocr_status"], payload)

    def write_source_card(
        self,
        video: VideoRef,
        *,
        title: str | None,
        source_type: str,
        config: PipelineConfig,
    ) -> None:
        rows = [
            "# Source Card",
            "",
            f"- Input: {video.raw_input}",
            f"- Resolved URL: {video.url}",
            f"- Video ID: {video.video_id or 'unresolved'}",
            f"- Title: {title or 'unknown'}",
            f"- Source type: {source_type}",
            f"- Sample interval: {config.sample_interval_seconds:g} seconds",
            f"- Processed range: {config.start_time or 0:g} to "
            + (
                f"{config.end_time:g} seconds"
                if config.end_time is not None
                else "video end"
            ),
            "",
            "This file contains source metadata only. Character interpretation belongs to a downstream analyzer.",
            "",
        ]
        atomic_write_text(
            self.directory / CORE_OUTPUTS["source_card"],
            "\n".join(rows),
        )

    def write_manifest(self, payload: dict[str, Any]) -> None:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "pipeline_version": PIPELINE_VERSION,
            **payload,
        }
        atomic_write_json(self.manifest_path, manifest)


def base_manifest(
    video: VideoRef,
    config: PipelineConfig,
    fingerprint: str,
    *,
    status: str,
) -> dict[str, Any]:
    return {
        "status": status,
        "fingerprint": fingerprint,
        "input": video.raw_input,
        "resolved_url": video.url,
        "video_id": video.video_id,
        "output_key": video.output_key,
        "config": config.public_dict(),
        "outputs": CORE_OUTPUTS,
        "updated_at": utc_now(),
    }


def segments_to_srt(segments: Sequence[Segment]) -> str:
    rows: list[str] = []
    for index, segment in enumerate(segments, start=1):
        rows.extend(
            [
                str(index),
                f"{srt_timestamp(segment.start)} --> {srt_timestamp(segment.end)}",
                segment.text,
                "",
            ]
        )
    return "\n".join(rows).rstrip() + "\n"


def segments_to_markdown(segments: Sequence[Segment]) -> str:
    rows = ["# Transcript", ""]
    for segment in segments:
        rows.extend(
            [
                f"[{display_timestamp(segment.start)} --> {display_timestamp(segment.end)}]",
                segment.text,
                (
                    f"<!-- source={segment.source}; confidence={segment.confidence:.3f}; "
                    f"reconstruction={segment.reconstruction} -->"
                ),
                "",
            ]
        )
    return "\n".join(rows).rstrip() + "\n"


def write_outputs(
    out_dir: Path,
    video_id: str,
    source: str,
    segments: Sequence[Segment | dict[str, Any]],
    frames_index: Sequence[OcrSample | dict[str, Any]],
    ocr_stats: dict[str, Any],
) -> None:
    """Small direct writer matching the proposal's single-file API."""
    out_dir.mkdir(parents=True, exist_ok=True)
    normalized_segments = [_coerce_segment(item) for item in segments]
    atomic_write_text(
        out_dir / "segments.jsonl",
        "".join(
            json.dumps(
                item.as_dict(index),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
            for index, item in enumerate(normalized_segments, start=1)
        ),
    )
    atomic_write_text(
        out_dir / "subtitle.srt",
        segments_to_srt(normalized_segments),
    )
    atomic_write_text(
        out_dir / "transcript.md",
        "\n".join(
            [
                f"# {video_id}",
                f"Source: {source}",
                "",
                *[
                    f"- {item.start:.1f}s-{item.end:.1f}s {item.text}"
                    for item in normalized_segments
                ],
                "",
            ]
        ),
    )
    atomic_write_text(
        out_dir / "frames-index.jsonl",
        "".join(
            json.dumps(
                item.as_dict() if isinstance(item, OcrSample) else item,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
            for item in frames_index
        ),
    )
    atomic_write_json(out_dir / "ocr-status.json", ocr_stats)
    atomic_write_text(
        out_dir / "source-card.md",
        f"# {video_id}\n\nSource: {source}\nSegments: {len(normalized_segments)}\n",
    )


def srt_timestamp(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def display_timestamp(seconds: float) -> str:
    return srt_timestamp(seconds).replace(",", ".")


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_write(path: Path, content: str) -> None:
    atomic_write_text(path, content)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def process_video(
    video: VideoRef | str,
    config_or_output: PipelineConfig | Path | None = None,
    *,
    output_root: Path | None = None,
    work_root: Path | None = None,
    interval: float = MIN_SAMPLE_INTERVAL_SECONDS,
    start: str | float | None = None,
    end: str | float | None = None,
    keep_frames: Literal["none", "changed", "all"] = "changed",
    force: bool = False,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> ProcessResult:
    """Process one video; successful manifests make later runs no-ops."""
    if isinstance(config_or_output, PipelineConfig):
        config = config_or_output
    else:
        if isinstance(config_or_output, Path) and output_root is None:
            output_root = config_or_output
        config = PipelineConfig(
            sample_interval_seconds=interval,
            start_time=_coerce_time(start),
            end_time=_coerce_time(end),
            keep_frames=keep_frames,
            force=force,
        )

    output_root = output_root or Path("outputs")
    work_root = work_root or Path("work")
    video_ref = (
        video if isinstance(video, VideoRef) else resolve_video_input(video)
    )
    store = OutputStore(output_root, video_ref.output_key)
    fingerprint = config.fingerprint(video_ref.output_key)
    if not config.force and store.cache_is_valid(fingerprint):
        manifest = _read_manifest(store)
        return ProcessResult(
            video_ref.raw_input,
            video_ref.output_key,
            str(store.directory),
            True,
            True,
            manifest.get("source_type"),
            int((manifest.get("stats") or {}).get("segment_count", 0)),
        )

    started = time.monotonic()
    started_at = utc_now()
    store.write_manifest(
        base_manifest(video_ref, config, fingerprint, status="running")
        | {"started_at": started_at}
    )
    work_root.mkdir(parents=True, exist_ok=True)
    source_type: str | None = None
    title: str | None = None
    diagnostics: dict[str, Any] = {}

    try:
        with tempfile.TemporaryDirectory(
            prefix=f"{video_ref.output_key}-",
            dir=work_root,
        ) as temporary_name:
            temporary = Path(temporary_name)
            track: SubtitleTrackResult | None = None

            if config.prefer_subtitle_track:
                track = download_best_subtitle_track(
                    video_ref,
                    temporary / "subtitle-track",
                    command_runner=command_runner,
                )
                title = track.title
                diagnostics["subtitle_track"] = {
                    "download_error": track.download_error,
                    "bbdown_exit_code": track.bbdown_exit_code,
                    "files_found": track.files_found,
                    "selected_file": (
                        track.selected_path.name
                        if track.selected_path is not None
                        else None
                    ),
                    "rejected": track.rejected,
                    "log_tail": track.log_tail,
                    "traceback": track.error_traceback,
                }
                if track.usable:
                    source_type = "subtitle_track"
                    segments = cues_to_segments(track.cues)
                    store.write_segments(segments)
                    store.write_frame_index([])
                    store.write_ocr_status(
                        {
                            "status": "skipped",
                            "reason": "usable platform subtitle track found",
                            "ocr_engine": None,
                            "sample_interval_seconds": (
                                config.sample_interval_seconds
                            ),
                        }
                    )
                    store.write_source_card(
                        video_ref,
                        title=title,
                        source_type=source_type,
                        config=config,
                    )
                    return _finish_success(
                        store,
                        video_ref,
                        config,
                        fingerprint,
                        source_type,
                        segments_count=len(segments),
                        diagnostics=diagnostics,
                        started_at=started_at,
                        started=started,
                    )

            if not config.enable_hardsub_ocr:
                track_error = (
                    f"; subtitle track: {track.download_error}"
                    if track is not None and track.download_error
                    else ""
                )
                raise RuntimeError(
                    "no usable subtitle track was found and hard-subtitle OCR "
                    f"is disabled{track_error}"
                )

            source_type = "hard_subtitle_ocr"
            video_path = download_low_quality_video(
                video_ref,
                temporary / "video",
                command_runner=command_runner,
            )
            video_info = probe_video(video_path, command_runner=command_runner)
            frames = extract_frames(
                video_path,
                temporary / "frames",
                config,
                command_runner=command_runner,
            )
            diagnostics["video"] = {
                "downloaded_file": video_path.name,
                "quality": "360P preferred, 480P fallback",
                "width": video_info.width,
                "height": video_info.height,
                "duration": video_info.duration,
                "ffmpeg_frame_extraction_passes": 1,
                "frame_count": len(frames),
            }

            samples, ocr_status = _ocr_frames(frames, store, config)
            assembly = assemble_sentences(
                samples, config.sample_interval_seconds
            )
            store.write_frame_index(samples)
            ocr_status["assembly"] = assembly.stats.as_dict()
            timeline_end = (
                config.end_time
                if config.end_time is not None
                else video_info.duration
            )
            if timeline_end is not None:
                assembly.segments = clamp_segment_end_times(
                    assembly.segments, timeline_end
                )
            ocr_status["segment_count"] = len(assembly.segments)
            if not assembly.segments:
                ocr_status["status"] = "failed"
                ocr_status["failure_reason"] = (
                    "OCR produced no reliable dialogue sentence"
                )
                store.write_ocr_status(ocr_status)
                raise RuntimeError("OCR produced no reliable dialogue sentence")

            store.write_segments(assembly.segments)
            ocr_status["status"] = "success"
            store.write_ocr_status(ocr_status)
            store.write_source_card(
                video_ref,
                title=title,
                source_type=source_type,
                config=config,
            )
            diagnostics["ocr"] = {
                key: ocr_status[key]
                for key in (
                    "ocr_engine",
                    "frame_count",
                    "ocr_call_count",
                    "reused_frame_count",
                    "enhanced_frame_count",
                    "binary_frame_count",
                    "saved_frame_count",
                )
            }
            return _finish_success(
                store,
                video_ref,
                config,
                fingerprint,
                source_type,
                segments_count=len(assembly.segments),
                diagnostics=diagnostics,
                started_at=started_at,
                started=started,
            )
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        error_traceback = traceback.format_exc()
        print(error_traceback, file=sys.stderr, flush=True)
        store.write_manifest(
            base_manifest(video_ref, config, fingerprint, status="failed")
            | {
                "source_type": source_type,
                "started_at": started_at,
                "completed_at": utc_now(),
                "duration_seconds": round(time.monotonic() - started, 3),
                "failure_reason": reason,
                "traceback": error_traceback,
                "diagnostics": diagnostics,
            }
        )
        return ProcessResult(
            video_ref.raw_input,
            video_ref.output_key,
            str(store.directory),
            False,
            False,
            source_type,
            0,
            reason,
        )


def _ocr_frames(
    frames: list[Path],
    store: OutputStore,
    config: PipelineConfig,
    ) -> tuple[list[OcrSample], dict[str, Any]]:
    engine = RapidOcrEngine()
    runner = OcrRunner(
        engine,
        similarity_threshold=config.frame_similarity_threshold,
    )
    samples: list[OcrSample] = []
    saved = 0
    enhanced_frames = 0
    binary_frames = 0

    for index, frame_path in enumerate(frames):
        with Image.open(frame_path) as frame:
            crop = crop_relative(frame.convert("RGB"), config.crop)
            signature = subtitle_signature(crop)
            candidate, reused, calls, variants = runner.run_image(
                crop,
                config.crop,
                confidence_threshold=config.adaptive_ocr_confidence,
            )
            enhanced_frames += int("enhanced" in variants)
            binary_frames += int("binary" in variants)

        sample = OcrSample(
            time=frame_time(index, config),
            text=candidate.text,
            confidence=candidate.confidence,
            quality=candidate.quality,
            signature=signature_hex(signature),
            reused=reused,
            ocr_calls=calls,
            variant=candidate.variant if not reused else "reused",
            lines=candidate.lines,
        )
        if _should_save_frame(
            sample,
            samples[-1] if samples else None,
            config.keep_frames,
        ):
            destination = (
                store.frames_directory
                / f"frame_{sample.time:010.3f}.jpg"
            )
            save_evidence_frame(frame_path, destination)
            sample.saved_frame = str(destination.relative_to(store.directory))
            saved += 1
        samples.append(sample)

        if (index + 1) % 100 == 0 or index + 1 == len(frames):
            print(
                f"OCR progress {index + 1}/{len(frames)}; "
                f"calls={runner.calls}; reused={runner.reused}",
                flush=True,
            )

    readable = [sample for sample in samples if sample.text.strip()]
    average_confidence = (
        sum(sample.confidence for sample in readable) / len(readable)
        if readable
        else 0.0
    )
    return samples, {
        "status": "running",
        "ocr_engine": engine.name,
        "sample_interval_seconds": config.sample_interval_seconds,
        "frame_count": len(frames),
        "ocr_call_count": runner.calls,
        "reused_frame_count": runner.reused,
        "enhanced_frame_count": enhanced_frames,
        "binary_frame_count": binary_frames,
        "enhance_call_count": runner.enhance_calls,
        "saved_frame_count": saved,
        "average_confidence": average_confidence,
        "crop_region": config.crop.as_dict(),
        "frame_similarity_threshold": config.frame_similarity_threshold,
    }


def _should_save_frame(
    sample: OcrSample,
    previous: OcrSample | None,
    keep_frames: str,
) -> bool:
    if keep_frames == "none":
        return False
    if keep_frames == "all" or previous is None:
        return True
    if not sample.text.strip() and not previous.text.strip():
        return False
    return (
        not samples_describe_same_caption(previous, sample)
        or sample.confidence < 0.65
    )


def _finish_success(
    store: OutputStore,
    video: VideoRef,
    config: PipelineConfig,
    fingerprint: str,
    source_type: str,
    *,
    segments_count: int,
    diagnostics: dict[str, Any],
    started_at: str,
    started: float,
) -> ProcessResult:
    store.write_manifest(
        base_manifest(video, config, fingerprint, status="success")
        | {
            "source_type": source_type,
            "started_at": started_at,
            "completed_at": utc_now(),
            "duration_seconds": round(time.monotonic() - started, 3),
            "stats": {"segment_count": segments_count},
            "diagnostics": diagnostics,
            "failure_reason": None,
        }
    )
    return ProcessResult(
        video.raw_input,
        video.output_key,
        str(store.directory),
        True,
        False,
        source_type,
        segments_count,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_time_value(value: str) -> float:
    raw = value.strip()
    try:
        if ":" not in raw:
            seconds = float(raw)
        else:
            parts = [float(part) for part in raw.split(":")]
            if len(parts) == 2:
                minutes, seconds_part = parts
                if seconds_part >= 60:
                    raise ValueError
                seconds = minutes * 60 + seconds_part
            elif len(parts) == 3:
                hours, minutes, seconds_part = parts
                if minutes >= 60 or seconds_part >= 60:
                    raise ValueError
                seconds = hours * 3600 + minutes * 60 + seconds_part
            else:
                raise ValueError
        if seconds < 0:
            raise ValueError
        return seconds
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"invalid time {value!r}; use seconds, MM:SS, or HH:MM:SS"
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract Bilibili subtitle tracks or run sparse hard-subtitle OCR."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Bilibili URLs, BV IDs, or AV IDs",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs"),
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path("work"),
    )
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=MIN_SAMPLE_INTERVAL_SECONDS,
        help="seconds between screenshots; minimum and default are 3",
    )
    parser.add_argument("--start-time", type=parse_time_value)
    parser.add_argument("--end-time", type=parse_time_value)
    parser.add_argument("--crop-top", type=float, default=0.50)
    parser.add_argument("--crop-bottom", type=float, default=0.94)
    parser.add_argument("--crop-left", type=float, default=0.04)
    parser.add_argument("--crop-right", type=float, default=0.96)
    parser.add_argument(
        "--keep-frames",
        choices=["none", "changed", "all"],
        default="changed",
    )
    parser.add_argument("--max-frame-width", type=int, default=1280)
    parser.add_argument("--skip-subtitle-track", action="store_true")
    parser.add_argument("--disable-hardsub-ocr", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    raw_env = os.environ.get("VIDEO_URLS", "")
    values = _deduplicate([*args.inputs, *parse_video_inputs(raw_env)])
    if not values:
        parser.error("provide at least one input or set VIDEO_URLS")

    try:
        config = PipelineConfig(
            sample_interval_seconds=args.sample_interval,
            start_time=args.start_time,
            end_time=args.end_time,
            crop=CropRegion(
                top=args.crop_top,
                bottom=args.crop_bottom,
                left=args.crop_left,
                right=args.crop_right,
            ),
            prefer_subtitle_track=not args.skip_subtitle_track,
            enable_hardsub_ocr=not args.disable_hardsub_ocr,
            keep_frames=args.keep_frames,
            max_frame_width=args.max_frame_width,
            force=args.force,
        )
    except ValueError as exc:
        parser.error(str(exc))

    results: list[ProcessResult] = []
    invalid: list[dict[str, Any]] = []
    for index, value in enumerate(values, start=1):
        print(f"Processing {index}/{len(values)}: {value}", flush=True)
        try:
            video = resolve_video_input(value)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            invalid.append(
                {
                    "input": value,
                    "success": False,
                    "skipped": False,
                    "failure_reason": reason,
                }
            )
            print(f"Input failed: {reason}", file=sys.stderr, flush=True)
            continue
        result = process_video(
            video,
            config,
            output_root=args.output_root,
            work_root=args.work_root,
        )
        results.append(result)
        state = (
            "cached"
            if result.skipped
            else "success"
            if result.success
            else "failed"
        )
        print(
            f"Result: {state}; output={result.output_directory}",
            flush=True,
        )
        if result.failure_reason:
            print(result.failure_reason, file=sys.stderr, flush=True)

    summary_path = args.summary_json or args.output_root / "run-summary.json"
    summary = {
        "processed_at": utc_now(),
        "processed": len(values),
        "successful": sum(result.success for result in results),
        "cached": sum(result.skipped for result in results),
        "failed": sum(not result.success for result in results) + len(invalid),
        "items": [result.as_dict() for result in results] + invalid,
    }
    atomic_write_json(summary_path, summary)
    return 0 if any(result.success for result in results) else 2


# ---------------------------------------------------------------------------
# Internal helpers and compatibility functions
# ---------------------------------------------------------------------------


def _find_videos(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in VIDEO_SUFFIXES
        and _is_nonempty_file(path)
    )


def _has_usable_subtitle_file(work_dir: Path) -> bool:
    for path in work_dir.rglob("*"):
        if not (
            path.is_file()
            and path.suffix.lower() in SUPPORTED_SUBTITLE_SUFFIXES
            and _is_nonempty_file(path)
        ):
            continue
        try:
            if any(is_dialogue_cue(cue) for cue in parse_subtitle_file(path)):
                return True
        except (
            OSError,
            ValueError,
            TypeError,
            json.JSONDecodeError,
            ET.ParseError,
        ):
            continue
    return False


def _is_nonempty_file(path: Path) -> bool:
    try:
        return path.stat().st_size > 0
    except OSError:
        return False


def _read_info_metadata(work_dir: Path) -> dict[str, object]:
    for path in sorted(work_dir.glob("*.info.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _last_log_line(log: str) -> str:
    return next(
        (line.strip() for line in reversed(log.splitlines()) if line.strip()),
        "unknown error",
    )


def _optional_int(value: object) -> int | None:
    return int(value) if value is not None else None


def _number(value: float) -> str:
    return f"{value:g}"


def _coerce_time(value: str | float | None) -> float | None:
    if value is None or value == "":
        return None
    return parse_time_value(str(value)) if isinstance(value, str) else float(value)


def _read_manifest(store: OutputStore) -> dict[str, Any]:
    try:
        payload = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _coerce_segment(value: Segment | dict[str, Any]) -> Segment:
    if isinstance(value, Segment):
        return value
    source = value.get("source", "hard_subtitle_ocr")
    if source not in {"subtitle_track", "hard_subtitle_ocr"}:
        source = "hard_subtitle_ocr"
    return Segment(
        start=float(value.get("start", 0)),
        end=float(value.get("end", value.get("start", 0))),
        text=str(value.get("text", "")),
        confidence=float(value.get("confidence", 1.0)),
        sample_times=[
            float(item)
            for item in value.get("sample_times", [value.get("start", 0)])
        ],
        reconstruction=str(value.get("reconstruction", "direct_ocr")),
        source=source,
        alternatives=[str(item) for item in value.get("alternatives", [])],
    )


def _deduplicate(values: Iterable[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = value.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            output.append(cleaned)
    return output


def config_fingerprint(
    video_id: str,
    interval: float,
    start: str | float | None,
    end: str | float | None,
) -> str:
    raw = f"{video_id}|{interval}|{start}|{end}|{SCHEMA_VERSION}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def already_done(out_dir: Path, fingerprint: str, force: bool) -> bool:
    if force:
        return False
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return (
        manifest.get("status") == "success"
        and manifest.get("fingerprint") == fingerprint
    )


__all__ = [
    "AdaptiveOcrResult",
    "AssemblyResult",
    "AssemblyStats",
    "CORE_OUTPUTS",
    "CropRegion",
    "FetchResult",
    "MIN_SAMPLE_INTERVAL",
    "MIN_SAMPLE_INTERVAL_SECONDS",
    "NON_DIALOGUE_RE",
    "OcrCandidate",
    "OcrEngine",
    "OcrLine",
    "OcrRunner",
    "OcrSample",
    "OutputStore",
    "PIPELINE_VERSION",
    "PipelineConfig",
    "ProcessResult",
    "RapidOcrEngine",
    "SCHEMA_VERSION",
    "Segment",
    "SubtitleCue",
    "SubtitleTrackResult",
    "VideoInfo",
    "VideoRef",
    "already_done",
    "assemble_sentences",
    "atomic_write_json",
    "atomic_write_text",
    "base_manifest",
    "binary_variant",
    "build_frame_extraction_command",
    "build_parser",
    "candidate_from_result",
    "classify_line_role",
    "classify_row_role",
    "clamp_segment_end_times",
    "clean_subtitle_text",
    "compact_text",
    "comparison_key",
    "config_fingerprint",
    "cues_to_segments",
    "crop_relative",
    "download_best_subtitle_track",
    "download_low_quality_video",
    "display_timestamp",
    "extract_frames",
    "extract_video_id",
    "enhanced_variant",
    "fetch_subtitle_or_video",
    "frame_time",
    "is_dialogue_cue",
    "is_readable_text",
    "is_supported_input",
    "join_overlapping_segments",
    "join_ocr_lines",
    "main",
    "meaningful_length",
    "normalize_input",
    "normalized_spacing",
    "otsu_threshold",
    "parse_ass",
    "parse_json_track",
    "parse_subtitle_file",
    "parse_srt_or_vtt",
    "parse_time_value",
    "parse_timecode",
    "parse_video_inputs",
    "process_video",
    "probe_video",
    "reconstruct_sentences",
    "resolve_video_input",
    "sample_frames",
    "samples_describe_same_caption",
    "save_evidence_frame",
    "segments_from_subtitle_file",
    "segments_to_markdown",
    "segments_to_srt",
    "should_drop_fragment",
    "signature_distance",
    "signature_hex",
    "signatures_are_similar",
    "subtitle_signature",
    "suffix_prefix_overlap",
    "track_score",
    "utc_now",
    "weighted_confidence",
    "write_outputs",
    "yt_dlp_common_args",
]
