"""Compatibility facade for the consolidated core pipeline."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Callable

from . import core as _core
from .core import ProcessResult

# Kept as module attributes so existing tests and callers can patch the
# original seam without needing to know that the implementation moved.
download_best_subtitle_track = _core.download_best_subtitle_track
download_low_quality_video = _core.download_low_quality_video


def process_video(
    video: _core.VideoRef | str,
    config_or_output: _core.PipelineConfig | Path | None = None,
    *,
    output_root: Path | None = None,
    work_root: Path | None = None,
    interval: float = _core.MIN_SAMPLE_INTERVAL_SECONDS,
    start: str | float | None = None,
    end: str | float | None = None,
    keep_frames: str = "changed",
    force: bool = False,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> ProcessResult:
    patched_track = download_best_subtitle_track
    original_track = _core.download_best_subtitle_track
    _core.download_best_subtitle_track = patched_track
    try:
        return _core.process_video(
            video,
            config_or_output,
            output_root=output_root,
            work_root=work_root,
            interval=interval,
            start=start,
            end=end,
            keep_frames=keep_frames,
            force=force,
            command_runner=command_runner,
        )
    finally:
        _core.download_best_subtitle_track = original_track


__all__ = [
    "ProcessResult",
    "download_best_subtitle_track",
    "download_low_quality_video",
    "process_video",
]
