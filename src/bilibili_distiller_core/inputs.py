"""Compatibility exports for input handling in core.py."""

from .core import (
    is_supported_input,
    parse_video_inputs,
    resolve_video_input,
)

__all__ = ["is_supported_input", "parse_video_inputs", "resolve_video_input"]
