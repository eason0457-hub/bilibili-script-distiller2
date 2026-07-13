"""Compatibility exports for sentence reconstruction in core.py."""

from .core import (
    AssemblyResult,
    AssemblyStats,
    assemble_sentences,
    clamp_segment_end_times,
    compact_text,
    comparison_key,
    is_readable_text,
    join_overlapping_segments,
    normalized_spacing,
    reconstruct_sentences,
    samples_describe_same_caption,
    segment_from_group,
    should_drop_fragment,
    suffix_prefix_overlap,
)

__all__ = [
    "AssemblyResult",
    "AssemblyStats",
    "assemble_sentences",
    "clamp_segment_end_times",
    "compact_text",
    "comparison_key",
    "is_readable_text",
    "join_overlapping_segments",
    "normalized_spacing",
    "reconstruct_sentences",
    "samples_describe_same_caption",
    "segment_from_group",
    "should_drop_fragment",
    "suffix_prefix_overlap",
]
