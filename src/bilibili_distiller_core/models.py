from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal


SCHEMA_VERSION = 1
PIPELINE_VERSION = "2.0.0"
MIN_SAMPLE_INTERVAL_SECONDS = 3.0


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
                f"sample interval must be at least {MIN_SAMPLE_INTERVAL_SECONDS:g} seconds"
            )
        if self.start_time is not None and self.start_time < 0:
            raise ValueError("start time cannot be negative")
        if self.end_time is not None and self.end_time < 0:
            raise ValueError("end time cannot be negative")
        if self.start_time is not None and self.end_time is not None:
            if self.end_time <= self.start_time:
                raise ValueError("end time must be greater than start time")
        if self.max_frame_width < 640:
            raise ValueError("max frame width must be at least 640 pixels")
        if not 0.0 <= self.frame_similarity_threshold <= 1.0:
            raise ValueError("frame similarity threshold must be inside 0..1")
        if not 0.0 <= self.adaptive_ocr_confidence <= 1.0:
            raise ValueError("adaptive OCR confidence must be inside 0..1")

    def fingerprint(self, source_key: str) -> str:
        payload = {
            "pipeline_version": PIPELINE_VERSION,
            "source_key": source_key,
            "config": self.public_dict(include_runtime_flags=False),
        }
        encoded = json.dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def public_dict(self, *, include_runtime_flags: bool = True) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        if not include_runtime_flags:
            data.pop("force", None)
            data.pop("keep_frames", None)
        return data


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
