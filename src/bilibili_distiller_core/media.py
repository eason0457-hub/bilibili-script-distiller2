"""Compatibility exports for media handling in core.py."""

from .core import (
    VideoInfo,
    build_frame_extraction_command,
    crop_relative,
    download_low_quality_video,
    extract_frames,
    frame_time,
    probe_video,
    sample_frames,
    save_evidence_frame,
    signature_distance,
    signature_hex,
    signatures_are_similar,
    subtitle_signature,
    yt_dlp_common_args,
)

__all__ = [
    "VideoInfo",
    "build_frame_extraction_command",
    "crop_relative",
    "download_low_quality_video",
    "extract_frames",
    "frame_time",
    "probe_video",
    "sample_frames",
    "save_evidence_frame",
    "signature_distance",
    "signature_hex",
    "signatures_are_similar",
    "subtitle_signature",
    "yt_dlp_common_args",
]
