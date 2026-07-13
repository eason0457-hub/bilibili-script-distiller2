"""Compatibility exports for storage handling in core.py."""

from .core import (
    CORE_OUTPUTS,
    OutputStore,
    atomic_write_json,
    atomic_write_text,
    base_manifest,
    display_timestamp,
    segments_to_markdown,
    segments_to_srt,
    srt_timestamp,
    utc_now,
)

__all__ = [
    "CORE_OUTPUTS",
    "OutputStore",
    "atomic_write_json",
    "atomic_write_text",
    "base_manifest",
    "display_timestamp",
    "segments_to_markdown",
    "segments_to_srt",
    "srt_timestamp",
    "utc_now",
]
