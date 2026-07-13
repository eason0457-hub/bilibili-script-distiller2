import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from bilibili_distiller_core.models import PipelineConfig, Segment, VideoRef
from bilibili_distiller_core.pipeline import process_video
from bilibili_distiller_core.storage import OutputStore, base_manifest
from bilibili_distiller_core.subtitle_tracks import SubtitleCue, SubtitleTrackResult


class StorageAndPipelineTests(unittest.TestCase):
    def setUp(self):
        self.video = VideoRef(
            "BV1uknVz9EeN",
            "https://www.bilibili.com/video/BV1uknVz9EeN/",
            "BV1uknVz9EeN",
            "BV1uknVz9EeN",
        )

    def test_output_store_writes_atomic_handoff_files_and_valid_cache(self):
        config = PipelineConfig()
        fingerprint = config.fingerprint(self.video.output_key)
        segment = Segment(0, 3, "text", 0.9, [0], "direct_ocr", "hard_subtitle_ocr")
        with tempfile.TemporaryDirectory() as temporary:
            store = OutputStore(Path(temporary), self.video.output_key)
            store.write_segments([segment])
            store.write_frame_index([])
            store.write_ocr_status({"status": "success"})
            store.write_source_card(
                self.video,
                title="Example",
                source_type="hard_subtitle_ocr",
                config=config,
            )
            store.write_manifest(
                base_manifest(self.video, config, fingerprint, status="success")
                | {"source_type": "hard_subtitle_ocr", "stats": {"segment_count": 1}}
            )
            self.assertTrue(store.cache_is_valid(fingerprint))
            self.assertIn("text", (store.directory / "subtitle.srt").read_text())
            row = json.loads(
                (store.directory / "segments.jsonl").read_text().splitlines()[0]
            )
            self.assertEqual(row["text"], "text")

    def test_pipeline_uses_subtitle_track_and_second_run_skips_everything(self):
        track = SubtitleTrackResult(
            cues=[SubtitleCue(1, 4, "platform subtitle")],
            title="Example",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch(
                "bilibili_distiller_core.pipeline.download_best_subtitle_track",
                return_value=track,
            ) as download:
                first = process_video(
                    self.video,
                    PipelineConfig(),
                    output_root=root / "outputs",
                    work_root=root / "work",
                )
                second = process_video(
                    self.video,
                    PipelineConfig(),
                    output_root=root / "outputs",
                    work_root=root / "work",
                )
        self.assertTrue(first.success)
        self.assertFalse(first.skipped)
        self.assertEqual(first.source_type, "subtitle_track")
        self.assertTrue(second.skipped)
        self.assertEqual(download.call_count, 1)


if __name__ == "__main__":
    unittest.main()
