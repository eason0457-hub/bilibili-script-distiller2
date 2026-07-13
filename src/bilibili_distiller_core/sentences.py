from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .models import OcrSample, Segment
from .ocr import meaningful_length


TERMINAL_PUNCTUATION = (".", "!", "?", "\u3002", "\uff01", "\uff1f", "\u2026")


@dataclass
class AssemblyStats:
    input_samples: int = 0
    readable_samples: int = 0
    temporal_consensus_groups: int = 0
    overlap_joins: int = 0
    dropped_low_confidence_fragments: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "input_samples": self.input_samples,
            "readable_samples": self.readable_samples,
            "temporal_consensus_groups": self.temporal_consensus_groups,
            "overlap_joins": self.overlap_joins,
            "dropped_low_confidence_fragments": self.dropped_low_confidence_fragments,
        }


@dataclass
class AssemblyResult:
    segments: list[Segment] = field(default_factory=list)
    stats: AssemblyStats = field(default_factory=AssemblyStats)


def assemble_sentences(
    samples: Sequence[OcrSample], sample_interval: float
) -> AssemblyResult:
    """Build stable sentences from sparse OCR without inventing missing words."""
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
            joined = join_overlapping_segments(repaired[-1], segment, sample_interval)
            if joined is not None:
                repaired[-1] = joined
                stats.overlap_joins += 1
                continue
        repaired.append(segment)

    for previous, current in zip(repaired, repaired[1:]):
        previous.end = min(previous.end, current.start)
    return AssemblyResult(repaired, stats)


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


def segment_from_group(group: Sequence[OcrSample], sample_interval: float) -> Segment:
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
    """Prefer a supported longer reading when OCR captured progressive fragments."""
    best_quality = max(sample.quality for sample in group)
    credible = [
        sample
        for sample in group
        if sample.confidence >= 0.55 and sample.quality >= best_quality - 0.16
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
    if length == 0:
        return True
    return length == 1 and segment.confidence < 0.72


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
    return bool(compact) and compact not in {"[unrecognized]", "[unabletorecognize]"}


def compact_text(text: str) -> str:
    return re.sub(r"\s+", "", text or "").strip()


def normalized_spacing(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def comparison_key(text: str) -> str:
    return re.sub(
        r"[\s\W_]+",
        "",
        text or "",
        flags=re.UNICODE,
    ).casefold()


def _sample_rank(sample: OcrSample) -> tuple[float, int, float]:
    support = sample.quality * 0.65 + sample.confidence * 0.35
    return support, meaningful_length(sample.text), sample.confidence


def _group_support(candidate: OcrSample, group: Sequence[OcrSample]) -> float:
    candidate_key = comparison_key(candidate.text)
    if not candidate_key:
        return 0.0
    return sum(
        difflib.SequenceMatcher(None, candidate_key, comparison_key(other.text)).ratio()
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
