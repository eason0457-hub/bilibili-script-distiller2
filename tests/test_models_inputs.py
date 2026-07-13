import unittest

from bilibili_distiller_core.inputs import parse_video_inputs, resolve_video_input
from bilibili_distiller_core.models import CropRegion, PipelineConfig


class ModelsAndInputsTests(unittest.TestCase):
    def test_sample_interval_has_hard_three_second_minimum(self):
        self.assertEqual(PipelineConfig().sample_interval_seconds, 3.0)
        with self.assertRaisesRegex(ValueError, "at least 3"):
            PipelineConfig(sample_interval_seconds=2.99)
        self.assertEqual(
            PipelineConfig(sample_interval_seconds=6).sample_interval_seconds, 6
        )

    def test_invalid_time_range_and_crop_are_rejected(self):
        with self.assertRaises(ValueError):
            PipelineConfig(start_time=10, end_time=5)
        with self.assertRaises(ValueError):
            CropRegion(top=0.9, bottom=0.5)

    def test_fingerprint_changes_for_ocr_configuration_not_runtime_flags(self):
        base = PipelineConfig()
        self.assertEqual(
            base.fingerprint("BV1uknVz9EeN"),
            PipelineConfig(force=True, keep_frames="all").fingerprint("BV1uknVz9EeN"),
        )
        self.assertNotEqual(
            base.fingerprint("BV1uknVz9EeN"),
            PipelineConfig(sample_interval_seconds=5).fingerprint("BV1uknVz9EeN"),
        )

    def test_batch_input_parser_handles_numbered_mixed_separators(self):
        values = parse_video_inputs(
            "1. BV1uknVz9EeN\n2) https://www.bilibili.com/video/BV1C7YjzDEvr/, BV1uknVz9EeN"
        )
        self.assertEqual(
            values,
            ["BV1uknVz9EeN", "https://www.bilibili.com/video/BV1C7YjzDEvr/"],
        )

    def test_bv_id_resolves_without_network(self):
        video = resolve_video_input("BV1uknVz9EeN")
        self.assertEqual(video.video_id, "BV1uknVz9EeN")
        self.assertEqual(video.output_key, "BV1uknVz9EeN")
        self.assertIn("/video/BV1uknVz9EeN/", video.url)


if __name__ == "__main__":
    unittest.main()
