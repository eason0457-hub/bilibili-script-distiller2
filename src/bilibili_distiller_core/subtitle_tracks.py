"""Compatibility exports for subtitle-track handling in core.py."""

from .core import (
    SubtitleCue,
    SubtitleTrackResult,
    clean_subtitle_text,
    cues_to_segments,
    download_best_subtitle_track,
    extract_title,
    is_dialogue_cue,
    parse_ass,
    parse_json_track,
    parse_subtitle_file,
    parse_srt_or_vtt,
    parse_timecode,
    parse_xml_track,
    track_score,
)

__all__ = [
    "SubtitleCue",
    "SubtitleTrackResult",
    "clean_subtitle_text",
    "cues_to_segments",
    "download_best_subtitle_track",
    "extract_title",
    "is_dialogue_cue",
    "parse_ass",
    "parse_json_track",
    "parse_subtitle_file",
    "parse_srt_or_vtt",
    "parse_timecode",
    "parse_xml_track",
    "track_score",
]
