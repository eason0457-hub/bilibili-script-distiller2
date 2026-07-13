from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .inputs import parse_video_inputs, resolve_video_input
from .models import MIN_SAMPLE_INTERVAL_SECONDS, CropRegion, PipelineConfig
from .pipeline import ProcessResult, process_video
from .storage import atomic_write_json, utc_now


def parse_time_value(value: str) -> float:
    raw = value.strip()
    try:
        if ":" not in raw:
            seconds = float(raw)
        else:
            parts = [float(part) for part in raw.split(":")]
            if len(parts) == 2:
                minutes, seconds_part = parts
                if seconds_part >= 60:
                    raise ValueError
                seconds = minutes * 60 + seconds_part
            elif len(parts) == 3:
                hours, minutes, seconds_part = parts
                if minutes >= 60 or seconds_part >= 60:
                    raise ValueError
                seconds = hours * 3600 + minutes * 60 + seconds_part
            else:
                raise ValueError
        if seconds < 0:
            raise ValueError
        return seconds
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid time {value!r}; use seconds, MM:SS, or HH:MM:SS"
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract Bilibili subtitle tracks or run sparse hard-subtitle OCR."
    )
    parser.add_argument("inputs", nargs="*", help="Bilibili URLs, BV IDs, or AV IDs")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--work-root", type=Path, default=Path("work"))
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=MIN_SAMPLE_INTERVAL_SECONDS,
        help="seconds between screenshots; minimum and default are 3",
    )
    parser.add_argument("--start-time", type=parse_time_value)
    parser.add_argument("--end-time", type=parse_time_value)
    parser.add_argument("--crop-top", type=float, default=0.50)
    parser.add_argument("--crop-bottom", type=float, default=0.94)
    parser.add_argument("--crop-left", type=float, default=0.04)
    parser.add_argument("--crop-right", type=float, default=0.96)
    parser.add_argument(
        "--keep-frames", choices=["none", "changed", "all"], default="changed"
    )
    parser.add_argument("--max-frame-width", type=int, default=1280)
    parser.add_argument("--skip-subtitle-track", action="store_true")
    parser.add_argument("--disable-hardsub-ocr", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    raw_env = os.environ.get("VIDEO_URLS", "")
    values = _deduplicate([*args.inputs, *parse_video_inputs(raw_env)])
    if not values:
        parser.error("provide at least one input or set VIDEO_URLS")

    try:
        config = PipelineConfig(
            sample_interval_seconds=args.sample_interval,
            start_time=args.start_time,
            end_time=args.end_time,
            crop=CropRegion(
                top=args.crop_top,
                bottom=args.crop_bottom,
                left=args.crop_left,
                right=args.crop_right,
            ),
            prefer_subtitle_track=not args.skip_subtitle_track,
            enable_hardsub_ocr=not args.disable_hardsub_ocr,
            keep_frames=args.keep_frames,
            max_frame_width=args.max_frame_width,
            force=args.force,
        )
    except ValueError as exc:
        parser.error(str(exc))

    results: list[ProcessResult] = []
    invalid: list[dict[str, object]] = []
    for index, value in enumerate(values, start=1):
        print(f"Processing {index}/{len(values)}: {value}", flush=True)
        try:
            video = resolve_video_input(value)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            invalid.append({"input": value, "success": False, "failure_reason": reason})
            print(f"Input failed: {reason}", file=sys.stderr, flush=True)
            continue
        result = process_video(
            video,
            config,
            output_root=args.output_root,
            work_root=args.work_root,
        )
        results.append(result)
        state = (
            "cached" if result.skipped else "success" if result.success else "failed"
        )
        print(f"Result: {state}; output={result.output_directory}", flush=True)
        if result.failure_reason:
            print(result.failure_reason, file=sys.stderr, flush=True)

    summary_path = args.summary_json or args.output_root / "run-summary.json"
    summary = {
        "processed_at": utc_now(),
        "processed": len(values),
        "successful": sum(result.success for result in results),
        "cached": sum(result.skipped for result in results),
        "failed": sum(not result.success for result in results) + len(invalid),
        "items": [result.as_dict() for result in results] + invalid,
    }
    atomic_write_json(summary_path, summary)
    return 0 if any(result.success for result in results) else 2


def _deduplicate(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = value.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            output.append(cleaned)
    return output


if __name__ == "__main__":
    raise SystemExit(main())
