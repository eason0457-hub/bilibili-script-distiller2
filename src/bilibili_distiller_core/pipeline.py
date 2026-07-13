from __future__ import annotations

import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from .media import (
    crop_relative,
    download_low_quality_video,
    extract_frames,
    frame_time,
    probe_video,
    save_evidence_frame,
    signature_hex,
    signatures_are_similar,
    subtitle_signature,
)
from .models import OcrCandidate, OcrSample, PipelineConfig, VideoRef
from .ocr import RapidOcrEngine, recognize_adaptive
from .sentences import (
    assemble_sentences,
    clamp_segment_end_times,
    samples_describe_same_caption,
)
from .storage import OutputStore, base_manifest, utc_now
from .subtitle_tracks import cues_to_segments, download_best_subtitle_track


@dataclass(frozen=True)
class ProcessResult:
    input: str
    output_key: str
    output_directory: str
    success: bool
    skipped: bool
    source_type: str | None
    segment_count: int
    failure_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "input": self.input,
            "output_key": self.output_key,
            "output_directory": self.output_directory,
            "success": self.success,
            "skipped": self.skipped,
            "source_type": self.source_type,
            "segment_count": self.segment_count,
            "failure_reason": self.failure_reason,
        }


def process_video(
    video: VideoRef,
    config: PipelineConfig,
    *,
    output_root: Path,
    work_root: Path,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> ProcessResult:
    store = OutputStore(output_root, video.output_key)
    fingerprint = config.fingerprint(video.output_key)
    if not config.force and store.cache_is_valid(fingerprint):
        manifest = _read_manifest(store)
        return ProcessResult(
            video.raw_input,
            video.output_key,
            str(store.directory),
            True,
            True,
            manifest.get("source_type"),
            int((manifest.get("stats") or {}).get("segment_count", 0)),
        )

    started = time.monotonic()
    started_at = utc_now()
    store.write_manifest(
        base_manifest(video, config, fingerprint, status="running")
        | {"started_at": started_at}
    )
    work_root.mkdir(parents=True, exist_ok=True)

    source_type: str | None = None
    title: str | None = None
    diagnostics: dict[str, Any] = {}
    try:
        with tempfile.TemporaryDirectory(
            prefix=f"{video.output_key}-", dir=work_root
        ) as temporary_name:
            temporary = Path(temporary_name)

            if config.prefer_subtitle_track:
                track = download_best_subtitle_track(
                    video, temporary / "subtitle-track", command_runner=command_runner
                )
                title = track.title
                diagnostics["subtitle_track"] = {
                    "bbdown_exit_code": track.bbdown_exit_code,
                    "files_found": track.files_found,
                    "selected_file": (
                        track.selected_path.name
                        if track.selected_path is not None
                        else None
                    ),
                    "rejected": track.rejected,
                    "log_tail": track.log_tail,
                }
                if track.usable:
                    source_type = "subtitle_track"
                    segments = cues_to_segments(track.cues)
                    store.write_segments(segments)
                    store.write_frame_index([])
                    store.write_ocr_status(
                        {
                            "status": "skipped",
                            "reason": "usable platform subtitle track found",
                            "ocr_engine": None,
                            "sample_interval_seconds": config.sample_interval_seconds,
                        }
                    )
                    store.write_source_card(
                        video, title=title, source_type=source_type, config=config
                    )
                    return _finish_success(
                        store,
                        video,
                        config,
                        fingerprint,
                        source_type,
                        segments_count=len(segments),
                        diagnostics=diagnostics,
                        started_at=started_at,
                        started=started,
                    )

            if not config.enable_hardsub_ocr:
                raise RuntimeError(
                    "no usable subtitle track was found and hard-subtitle OCR is disabled"
                )

            source_type = "hard_subtitle_ocr"
            video_path = download_low_quality_video(
                video, temporary / "video", command_runner=command_runner
            )
            video_info = probe_video(video_path, command_runner=command_runner)
            frames = extract_frames(
                video_path,
                temporary / "frames",
                config,
                command_runner=command_runner,
            )
            diagnostics["video"] = {
                "downloaded_file": video_path.name,
                "quality": "360P preferred, 480P fallback",
                "width": video_info.width,
                "height": video_info.height,
                "duration": video_info.duration,
                "ffmpeg_frame_extraction_passes": 1,
                "frame_count": len(frames),
            }

            samples, ocr_status = _ocr_frames(frames, store, config)
            assembly = assemble_sentences(samples, config.sample_interval_seconds)
            store.write_frame_index(samples)
            ocr_status["assembly"] = assembly.stats.as_dict()
            timeline_end = (
                config.end_time if config.end_time is not None else video_info.duration
            )
            if timeline_end is not None:
                assembly.segments = clamp_segment_end_times(
                    assembly.segments, timeline_end
                )
            ocr_status["segment_count"] = len(assembly.segments)
            if not assembly.segments:
                ocr_status["status"] = "failed"
                ocr_status["failure_reason"] = (
                    "OCR produced no reliable dialogue sentence"
                )
                store.write_ocr_status(ocr_status)
                raise RuntimeError("OCR produced no reliable dialogue sentence")

            store.write_segments(assembly.segments)
            ocr_status["status"] = "success"
            store.write_ocr_status(ocr_status)
            store.write_source_card(
                video, title=title, source_type=source_type, config=config
            )
            diagnostics["ocr"] = {
                key: ocr_status[key]
                for key in (
                    "ocr_engine",
                    "frame_count",
                    "ocr_call_count",
                    "reused_frame_count",
                    "enhanced_frame_count",
                    "binary_frame_count",
                    "saved_frame_count",
                )
            }
            return _finish_success(
                store,
                video,
                config,
                fingerprint,
                source_type,
                segments_count=len(assembly.segments),
                diagnostics=diagnostics,
                started_at=started_at,
                started=started,
            )
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        failure_manifest = base_manifest(
            video, config, fingerprint, status="failed"
        ) | {
            "source_type": source_type,
            "started_at": started_at,
            "completed_at": utc_now(),
            "duration_seconds": round(time.monotonic() - started, 3),
            "failure_reason": reason,
            "diagnostics": diagnostics,
        }
        store.write_manifest(failure_manifest)
        return ProcessResult(
            video.raw_input,
            video.output_key,
            str(store.directory),
            False,
            False,
            source_type,
            0,
            reason,
        )


def _ocr_frames(
    frames: list[Path], store: OutputStore, config: PipelineConfig
) -> tuple[list[OcrSample], dict[str, Any]]:
    engine = RapidOcrEngine()
    samples: list[OcrSample] = []
    previous_signature: bytes | None = None
    previous_candidate: OcrCandidate | None = None
    ocr_calls = 0
    reused = 0
    enhanced = 0
    binary = 0
    saved = 0

    for index, frame_path in enumerate(frames):
        with Image.open(frame_path) as frame:
            crop = crop_relative(frame.convert("RGB"), config.crop)
            signature = subtitle_signature(crop)
            if (
                signatures_are_similar(
                    previous_signature, signature, config.frame_similarity_threshold
                )
                and previous_candidate is not None
            ):
                candidate = previous_candidate
                calls = 0
                reused_frame = True
                variants = ["reused"]
                reused += 1
            else:
                adaptive = recognize_adaptive(
                    crop,
                    config.crop,
                    engine,
                    confidence_threshold=config.adaptive_ocr_confidence,
                )
                candidate = adaptive.candidate
                calls = adaptive.calls
                variants = adaptive.tried_variants
                reused_frame = False
                ocr_calls += calls
                enhanced += int("enhanced" in variants)
                binary += int("binary" in variants)

        sample = OcrSample(
            time=frame_time(index, config),
            text=candidate.text,
            confidence=candidate.confidence,
            quality=candidate.quality,
            signature=signature_hex(signature),
            reused=reused_frame,
            ocr_calls=calls,
            variant=candidate.variant if not reused_frame else "reused",
            lines=candidate.lines,
        )
        if _should_save_frame(
            sample, samples[-1] if samples else None, config.keep_frames
        ):
            destination = store.frames_directory / f"frame_{sample.time:010.3f}.jpg"
            save_evidence_frame(frame_path, destination)
            sample.saved_frame = str(destination.relative_to(store.directory))
            saved += 1
        samples.append(sample)
        previous_signature = signature
        previous_candidate = candidate

        if (index + 1) % 100 == 0 or index + 1 == len(frames):
            print(
                f"OCR progress {index + 1}/{len(frames)}; calls={ocr_calls}; reused={reused}",
                flush=True,
            )

    readable = [sample for sample in samples if sample.text.strip()]
    average_confidence = (
        sum(sample.confidence for sample in readable) / len(readable)
        if readable
        else 0.0
    )
    return samples, {
        "status": "running",
        "ocr_engine": engine.name,
        "sample_interval_seconds": config.sample_interval_seconds,
        "frame_count": len(frames),
        "ocr_call_count": ocr_calls,
        "reused_frame_count": reused,
        "enhanced_frame_count": enhanced,
        "binary_frame_count": binary,
        "saved_frame_count": saved,
        "average_confidence": average_confidence,
        "crop_region": config.crop.as_dict(),
        "frame_similarity_threshold": config.frame_similarity_threshold,
    }


def _should_save_frame(
    sample: OcrSample, previous: OcrSample | None, keep_frames: str
) -> bool:
    if keep_frames == "none":
        return False
    if keep_frames == "all" or previous is None:
        return True
    if not sample.text.strip() and not previous.text.strip():
        return False
    return (
        not samples_describe_same_caption(previous, sample) or sample.confidence < 0.65
    )


def _finish_success(
    store: OutputStore,
    video: VideoRef,
    config: PipelineConfig,
    fingerprint: str,
    source_type: str,
    *,
    segments_count: int,
    diagnostics: dict[str, Any],
    started_at: str,
    started: float,
) -> ProcessResult:
    store.write_manifest(
        base_manifest(video, config, fingerprint, status="success")
        | {
            "source_type": source_type,
            "started_at": started_at,
            "completed_at": utc_now(),
            "duration_seconds": round(time.monotonic() - started, 3),
            "stats": {"segment_count": segments_count},
            "diagnostics": diagnostics,
            "failure_reason": None,
        }
    )
    return ProcessResult(
        video.raw_input,
        video.output_key,
        str(store.directory),
        True,
        False,
        source_type,
        segments_count,
    )


def _read_manifest(store: OutputStore) -> dict[str, Any]:
    import json

    try:
        return json.loads(store.manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
