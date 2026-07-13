from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image, ImageFilter, ImageOps

from .models import CropRegion, PipelineConfig, VideoRef


VIDEO_SUFFIXES = {".mp4", ".mkv", ".flv", ".webm"}
YTDLP_COMMON_ARGS = (
    "--ignore-config",
    "--no-playlist",
    "--impersonate",
    "chrome",
    "--add-header",
    "Referer: https://www.bilibili.com/",
    "--add-header",
    "Origin: https://www.bilibili.com",
)


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    width: int | None
    height: int | None
    duration: float | None


def yt_dlp_common_args() -> list[str]:
    """Use browser-like requests for Bilibili's anti-bot API checks."""
    return list(YTDLP_COMMON_ARGS)


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
        "worstvideo[height>=360][height<=480]/bestvideo[height<=480]/worstvideo/worst",
        "--output",
        str(destination / "source.%(ext)s"),
        video.url,
    ]
    process = command_runner(
        command,
        text=True,
        capture_output=True,
        timeout=900,
        check=False,
    )
    if process.returncode != 0:
        message = _last_log_line((process.stdout or "") + "\n" + (process.stderr or ""))
        raise RuntimeError(
            f"low-quality video download failed with exit code {process.returncode}: {message}"
        )
    videos = _find_videos(destination)
    if not videos:
        raise RuntimeError("yt-dlp completed without producing a video file")
    return videos[0]


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
        message = _last_log_line((process.stdout or "") + "\n" + (process.stderr or ""))
        raise RuntimeError(f"FFmpeg frame extraction failed: {message}")
    frames = sorted(frame_dir.glob("frame_*.jpg"))
    if not frames:
        raise RuntimeError("FFmpeg completed without producing screenshots")
    return frames


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
    """Return a compact edge signature used only for conservative OCR reuse."""
    grayscale = ImageOps.grayscale(image).resize((128, 32))
    edges = grayscale.filter(ImageFilter.FIND_EDGES)
    pixels = (
        edges.get_flattened_data()
        if hasattr(edges, "get_flattened_data")
        else edges.getdata()
    )
    return bytes(1 if value >= 42 else 0 for value in pixels)


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


def signatures_are_similar(left: bytes | None, right: bytes, threshold: float) -> bool:
    return left is not None and signature_distance(left, right) <= threshold


def save_evidence_frame(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image.convert("RGB").save(destination, format="JPEG", quality=82, optimize=True)


def frame_time(index: int, config: PipelineConfig) -> float:
    return (config.start_time or 0.0) + index * config.sample_interval_seconds


def _find_videos(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )


def _last_log_line(log: str) -> str:
    return next(
        (line.strip() for line in reversed(log.splitlines()) if line.strip()),
        "unknown error",
    )


def _optional_int(value: object) -> int | None:
    return int(value) if value is not None else None


def _number(value: float) -> str:
    return f"{value:g}"

