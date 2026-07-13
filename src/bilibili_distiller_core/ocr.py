from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Protocol, Sequence

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from .models import CropRegion, OcrCandidate, OcrLine


SPEAKER_HINT_REGIONS = {
    "left_upper": CropRegion(top=0.54, bottom=0.68, left=0.14, right=0.42),
    "left_lower": CropRegion(top=0.62, bottom=0.76, left=0.06, right=0.30),
    "center": CropRegion(top=0.62, bottom=0.76, left=0.38, right=0.64),
}
MEANINGFUL_CHAR_RE = re.compile(r"[A-Za-z0-9\u3040-\u30ff\u3400-\u9fff]")


class OcrEngine(Protocol):
    name: str

    def recognize(self, image: Image.Image) -> Sequence[object]: ...


class RapidOcrEngine:
    name = "rapidocr_3_onnxruntime"

    def __init__(self) -> None:
        from rapidocr import RapidOCR

        self._engine = RapidOCR()

    def recognize(self, image: Image.Image) -> Sequence[object]:
        import numpy as np

        rgb = image.convert("RGB")
        bgr = np.asarray(rgb)[:, :, ::-1].copy()
        result = self._engine(bgr)
        boxes = getattr(result, "boxes", None)
        texts = getattr(result, "txts", None)
        scores = getattr(result, "scores", None)
        if boxes is None or texts is None or scores is None:
            return []
        return [
            [box.tolist() if hasattr(box, "tolist") else box, text, score]
            for box, text, score in zip(boxes, texts, scores)
        ]


@dataclass(frozen=True)
class AdaptiveOcrResult:
    candidate: OcrCandidate
    calls: int
    tried_variants: list[str]


def recognize_adaptive(
    image: Image.Image,
    crop_region: CropRegion,
    engine: OcrEngine,
    *,
    confidence_threshold: float = 0.78,
) -> AdaptiveOcrResult:
    """Run one OCR pass normally and add preprocessing only when quality is poor."""
    candidates: list[OcrCandidate] = []
    tried: list[str] = []

    base = candidate_from_result(
        engine.recognize(image), image.size, crop_region, variant="base"
    )
    candidates.append(base)
    tried.append("base")

    if _needs_enhancement(base, confidence_threshold):
        enhanced_image = enhanced_variant(image)
        enhanced = candidate_from_result(
            engine.recognize(enhanced_image),
            enhanced_image.size,
            crop_region,
            variant="enhanced",
        )
        candidates.append(enhanced)
        tried.append("enhanced")

    best_so_far = max(candidates, key=_candidate_rank)
    if _needs_binary_pass(best_so_far):
        binary_image = binary_variant(image)
        binary = candidate_from_result(
            engine.recognize(binary_image),
            binary_image.size,
            crop_region,
            variant="binary",
        )
        candidates.append(binary)
        tried.append("binary")

    best = max(candidates, key=_candidate_rank)
    return AdaptiveOcrResult(best, len(candidates), tried)


def candidate_from_result(
    raw_result: Sequence[object],
    image_size: tuple[int, int],
    crop_region: CropRegion,
    *,
    variant: str,
) -> OcrCandidate:
    parsed: list[OcrLine] = []
    for raw_line in raw_result or []:
        try:
            box_value = raw_line[0]  # type: ignore[index]
            text = str(raw_line[1]).strip()  # type: ignore[index]
            confidence = float(raw_line[2])  # type: ignore[index]
            box = [[float(point[0]), float(point[1])] for point in box_value]
        except (IndexError, TypeError, ValueError):
            continue
        if not text or not box:
            continue
        parsed.append(
            OcrLine(text=text, confidence=confidence, box=box, role="unknown")
        )

    rows = cluster_reading_rows(parsed)
    parsed = []
    for row in rows:
        role = classify_row_role(row, image_size, crop_region)
        for line in sorted(row, key=_line_left):
            line.role = role
            parsed.append(line)
    dialogue_lines = [line for line in parsed if line.role != "speaker_hint"]
    text = join_ocr_lines([line.text for line in dialogue_lines])
    confidence = weighted_confidence(dialogue_lines)
    quality = candidate_quality(text, confidence, dialogue_lines)
    return OcrCandidate(text, confidence, quality, parsed, variant)


def classify_line_role(
    text: str,
    box: list[list[float]],
    image_size: tuple[int, int],
    crop_region: CropRegion,
) -> str:
    return classify_row_role(
        [OcrLine(text=text, confidence=0.0, box=box, role="unknown")],
        image_size,
        crop_region,
    )


def classify_row_role(
    row: Sequence[OcrLine],
    image_size: tuple[int, int],
    crop_region: CropRegion,
) -> str:
    """Mark only a complete short row contained by a known name-label region."""
    width, height = image_size
    if width <= 0 or height <= 0 or not row:
        return "unknown"
    points = [point for line in row for point in line.box]
    local_left = min(point[0] for point in points)
    local_right = max(point[0] for point in points)
    local_top = min(point[1] for point in points)
    local_bottom = max(point[1] for point in points)
    global_left = crop_region.left + (local_left / width) * (
        crop_region.right - crop_region.left
    )
    global_right = crop_region.left + (local_right / width) * (
        crop_region.right - crop_region.left
    )
    global_top = crop_region.top + (local_top / height) * (
        crop_region.bottom - crop_region.top
    )
    global_bottom = crop_region.top + (local_bottom / height) * (
        crop_region.bottom - crop_region.top
    )
    compact = re.sub(r"\s+", "", join_ocr_lines([line.text for line in row]))
    margin = 0.025
    if len(compact) <= 10 and any(
        global_left >= region.left - margin
        and global_right <= region.right + margin
        and global_top >= region.top - margin
        and global_bottom <= region.bottom + margin
        for region in SPEAKER_HINT_REGIONS.values()
    ):
        return "speaker_hint"
    return "dialogue"


def join_ocr_lines(lines: Sequence[str]) -> str:
    output = ""
    for raw in lines:
        value = re.sub(r"\s+", " ", raw).strip()
        if not value:
            continue
        if output and _needs_word_space(output[-1], value[0]):
            output += " "
        output += value
    return output.strip()


def weighted_confidence(lines: Sequence[OcrLine]) -> float:
    weights = [max(1, meaningful_length(line.text)) for line in lines]
    total = sum(weights)
    if not total:
        return 0.0
    return sum(line.confidence * weight for line, weight in zip(lines, weights)) / total


def candidate_quality(text: str, confidence: float, lines: Sequence[OcrLine]) -> float:
    length = meaningful_length(text)
    if length == 0:
        return 0.0
    length_score = min(1.0, math.log2(length + 1) / 4.0)
    line_score = min(1.0, len(lines) / 2.0)
    score = confidence * 0.72 + length_score * 0.23 + line_score * 0.05
    if length == 1:
        score -= 0.22
    elif length == 2:
        score -= 0.06
    return max(0.0, min(1.0, score))


def meaningful_length(text: str) -> int:
    return len(MEANINGFUL_CHAR_RE.findall(text or ""))


def sort_reading_order(lines: Sequence[OcrLine]) -> list[OcrLine]:
    """Cluster boxes into visual rows, then read each row from left to right."""
    return [
        line
        for row in cluster_reading_rows(lines)
        for line in sorted(row, key=_line_left)
    ]


def cluster_reading_rows(lines: Sequence[OcrLine]) -> list[list[OcrLine]]:
    pending = sorted(lines, key=lambda line: (_line_center_y(line), _line_left(line)))
    rows: list[list[OcrLine]] = []
    for line in pending:
        if not rows:
            rows.append([line])
            continue
        current = rows[-1]
        row_center = sum(_line_center_y(item) for item in current) / len(current)
        row_height = max(_line_height(item) for item in current)
        tolerance = max(row_height, _line_height(line)) * 0.65
        if abs(_line_center_y(line) - row_center) <= tolerance:
            current.append(line)
        else:
            rows.append([line])
    return rows


def enhanced_variant(image: Image.Image) -> Image.Image:
    grayscale = ImageOps.grayscale(image)
    width, height = grayscale.size
    enlarged = grayscale.resize((width * 2, height * 2), Image.Resampling.LANCZOS)
    contrasted = ImageOps.autocontrast(enlarged, cutoff=1)
    contrasted = ImageEnhance.Contrast(contrasted).enhance(1.35)
    return contrasted.filter(
        ImageFilter.UnsharpMask(radius=1.2, percent=170, threshold=2)
    )


def binary_variant(image: Image.Image) -> Image.Image:
    enhanced = enhanced_variant(image)
    threshold = otsu_threshold(enhanced.histogram())
    return enhanced.point(
        lambda value: 255 if value >= threshold else 0, mode="1"
    ).convert("L")


def otsu_threshold(histogram: Sequence[int]) -> int:
    total = sum(histogram)
    if total <= 0:
        return 128
    weighted_sum = sum(index * count for index, count in enumerate(histogram))
    background_weight = 0
    background_sum = 0.0
    best_variance = -1.0
    best_threshold = 128
    for threshold, count in enumerate(histogram):
        background_weight += count
        if background_weight == 0:
            continue
        foreground_weight = total - background_weight
        if foreground_weight == 0:
            break
        background_sum += threshold * count
        background_mean = background_sum / background_weight
        foreground_mean = (weighted_sum - background_sum) / foreground_weight
        variance = (
            background_weight
            * foreground_weight
            * (background_mean - foreground_mean) ** 2
        )
        if variance > best_variance:
            best_variance = variance
            best_threshold = threshold
    return best_threshold


def _candidate_rank(candidate: OcrCandidate) -> tuple[float, int, float]:
    length = meaningful_length(candidate.text)
    complete_sentence_bonus = min(length, 12) * 0.008
    return candidate.quality + complete_sentence_bonus, length, candidate.confidence


def _needs_enhancement(candidate: OcrCandidate, threshold: float) -> bool:
    return (
        not candidate.text
        or meaningful_length(candidate.text) < 4
        or candidate.confidence < threshold
        or candidate.quality < 0.68
    )


def _needs_binary_pass(candidate: OcrCandidate) -> bool:
    return (
        not candidate.text
        or meaningful_length(candidate.text) <= 1
        or candidate.confidence < 0.58
        or candidate.quality < 0.50
    )


def _line_center_y(line: OcrLine) -> float:
    return sum(point[1] for point in line.box) / len(line.box)


def _line_height(line: OcrLine) -> float:
    values = [point[1] for point in line.box]
    return max(values) - min(values)


def _line_left(line: OcrLine) -> float:
    return min(point[0] for point in line.box)


def _needs_word_space(left: str, right: str) -> bool:
    return left.isascii() and right.isascii() and left.isalnum() and right.isalnum()
