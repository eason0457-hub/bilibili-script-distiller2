from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

from .models import (
    PIPELINE_VERSION,
    SCHEMA_VERSION,
    OcrSample,
    PipelineConfig,
    Segment,
    VideoRef,
)


CORE_OUTPUTS = {
    "segments": "segments.jsonl",
    "subtitle": "subtitle.srt",
    "transcript": "transcript.md",
    "source_card": "source-card.md",
    "frame_index": "frames-index.jsonl",
    "ocr_status": "ocr-status.json",
}


class OutputStore:
    def __init__(self, root: Path, output_key: str) -> None:
        self.root = root
        self.output_key = output_key
        self.directory = root / output_key
        self.directory.mkdir(parents=True, exist_ok=True)

    @property
    def manifest_path(self) -> Path:
        return self.directory / "manifest.json"

    @property
    def frames_directory(self) -> Path:
        return self.directory / "frames"

    def cache_is_valid(self, fingerprint: str) -> bool:
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return False
        if (
            manifest.get("status") != "success"
            or manifest.get("fingerprint") != fingerprint
        ):
            return False
        return all(
            (self.directory / filename).is_file()
            and (self.directory / filename).stat().st_size > 0
            for key, filename in CORE_OUTPUTS.items()
            if key not in {"frame_index"}
        )

    def write_segments(self, segments: Sequence[Segment]) -> None:
        jsonl = "".join(
            json.dumps(
                segment.as_dict(index), ensure_ascii=False, separators=(",", ":")
            )
            + "\n"
            for index, segment in enumerate(segments, start=1)
        )
        atomic_write_text(self.directory / CORE_OUTPUTS["segments"], jsonl)
        atomic_write_text(
            self.directory / CORE_OUTPUTS["subtitle"], segments_to_srt(segments)
        )
        atomic_write_text(
            self.directory / CORE_OUTPUTS["transcript"], segments_to_markdown(segments)
        )

    def write_frame_index(self, samples: Sequence[OcrSample]) -> None:
        content = "".join(
            json.dumps(sample.as_dict(), ensure_ascii=False, separators=(",", ":"))
            + "\n"
            for sample in samples
        )
        atomic_write_text(self.directory / CORE_OUTPUTS["frame_index"], content)

    def write_ocr_status(self, payload: dict[str, Any]) -> None:
        atomic_write_json(self.directory / CORE_OUTPUTS["ocr_status"], payload)

    def write_source_card(
        self,
        video: VideoRef,
        *,
        title: str | None,
        source_type: str,
        config: PipelineConfig,
    ) -> None:
        rows = [
            "# Source Card",
            "",
            f"- Input: `{video.raw_input}`",
            f"- Resolved URL: {video.url}",
            f"- Video ID: `{video.video_id or 'unresolved'}`",
            f"- Title: {title or 'unknown'}",
            f"- Source type: `{source_type}`",
            f"- Sample interval: {config.sample_interval_seconds:g} seconds",
            f"- Processed range: {config.start_time or 0:g} to "
            + (
                f"{config.end_time:g} seconds"
                if config.end_time is not None
                else "video end"
            ),
            "",
            "This file contains source metadata only. Character interpretation belongs to a downstream analyzer.",
            "",
        ]
        atomic_write_text(self.directory / CORE_OUTPUTS["source_card"], "\n".join(rows))

    def write_manifest(self, payload: dict[str, Any]) -> None:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "pipeline_version": PIPELINE_VERSION,
            **payload,
        }
        atomic_write_json(self.manifest_path, manifest)


def base_manifest(
    video: VideoRef,
    config: PipelineConfig,
    fingerprint: str,
    *,
    status: str,
) -> dict[str, Any]:
    return {
        "status": status,
        "fingerprint": fingerprint,
        "input": video.raw_input,
        "resolved_url": video.url,
        "video_id": video.video_id,
        "output_key": video.output_key,
        "config": config.public_dict(),
        "outputs": CORE_OUTPUTS,
        "updated_at": utc_now(),
    }


def segments_to_srt(segments: Sequence[Segment]) -> str:
    rows: list[str] = []
    for index, segment in enumerate(segments, start=1):
        rows.extend(
            [
                str(index),
                f"{srt_timestamp(segment.start)} --> {srt_timestamp(segment.end)}",
                segment.text,
                "",
            ]
        )
    return "\n".join(rows).rstrip() + "\n"


def segments_to_markdown(segments: Sequence[Segment]) -> str:
    rows = ["# Transcript", ""]
    for segment in segments:
        rows.extend(
            [
                f"[{display_timestamp(segment.start)} --> {display_timestamp(segment.end)}]",
                segment.text,
                (
                    f"<!-- source={segment.source}; confidence={segment.confidence:.3f}; "
                    f"reconstruction={segment.reconstruction} -->"
                ),
                "",
            ]
        )
    return "\n".join(rows).rstrip() + "\n"


def srt_timestamp(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def display_timestamp(seconds: float) -> str:
    return srt_timestamp(seconds).replace(",", ".")


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
