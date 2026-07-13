from __future__ import annotations

import html
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from .media import yt_dlp_common_args
from .models import Segment, VideoRef


SUPPORTED_SUBTITLE_SUFFIXES = {".srt", ".vtt", ".ass", ".ssa", ".json", ".xml"}
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


@dataclass(frozen=True)
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

    @property
    def usable(self) -> bool:
        return bool(self.cues)


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
        cleaned = clean_subtitle_text(str(content))
        if cleaned and float(end) > float(start):
            cues.append(SubtitleCue(float(start), float(end), cleaned))
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
        start = float(start_value)
        end = (
            float(end_value) if end_value is not None else start + float(duration or 0)
        )
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
    work_dir.mkdir(parents=True, exist_ok=True)
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
    process = command_runner(
        command,
        text=True,
        capture_output=True,
        timeout=240,
        check=False,
    )
    log = (process.stdout or "") + "\n" + (process.stderr or "")
    result = SubtitleTrackResult(
        bbdown_exit_code=process.returncode,
        log_tail=_last_log_line(log),
    )
    metadata = _read_info_metadata(work_dir)
    result.title = (
        str(metadata.get("title")) if metadata.get("title") else extract_title(log)
    )
    result.video_id = (
        str(metadata.get("id"))
        if metadata.get("id")
        else (_extract_video_id(log) or video.video_id)
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


def _extract_video_id(log: str) -> str | None:
    match = re.search(r"\b(BV[0-9A-Za-z]{10})\b", log)
    return match.group(1) if match else None


def _last_log_line(log: str) -> str | None:
    return next(
        (line.strip() for line in reversed(log.splitlines()) if line.strip()), None
    )


def _read_info_metadata(work_dir: Path) -> dict[str, object]:
    for path in sorted(work_dir.glob("*.info.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            return payload
    return {}

