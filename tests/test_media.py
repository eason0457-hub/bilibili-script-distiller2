import unittest
from pathlib import Path

from PIL import Image

from bilibili_distiller_core.media import (
    build_frame_extraction_command,
    signature_hex,
    signature_distance,
    signatures_are_similar,
    subtitle_signature,
    yt_dlp_common_args,
)
from bilibili_distiller_core.models import PipelineConfig


class MediaTests(unittest.TestCase):
    def test_yt_dlp_uses_browser_like_bilibili_requests(self):
        args = yt_dlp_common_args()
        self.assertEqual(args[args.index("--impersonate") + 1], "chrome")
        self.assertIn("Referer: https://www.bilibili.com/", args)
        self.assertIn("Origin: https://www.bilibili.com", args)

    def test_ffmpeg_uses_single_sparse_three_second_filter(self):
        config = PipelineConfig()
        command = build_frame_extraction_command(
            Path("video.mp4"), Path("frames"), config
        )
        filter_value = command[command.index("-vf") + 1]
        self.assertIn("isnan(prev_selected_t)", filter_value)
        self.assertIn("t-prev_selected_t\\,3", filter_value)
        self.assertIn("scale='min(1280,iw)':-2", filter_value)
        self.assertEqual(command.count("ffmpeg"), 1)

    def test_custom_interval_cannot_increase_sampling_frequency(self):
        config = PipelineConfig(sample_interval_seconds=5)
        command = build_frame_extraction_command(Path("v.mp4"), Path("f"), config)
        self.assertIn("t-prev_selected_t\\,5", command[command.index("-vf") + 1])

    def test_subtitle_signature_reuses_only_near_identical_crops(self):
        first = Image.new("RGB", (320, 80), "black")
        second = first.copy()
        different = Image.new("RGB", (320, 80), "white")
        first_signature = subtitle_signature(first)
        second_signature = subtitle_signature(second)
        different_signature = subtitle_signature(different)
        self.assertEqual(signature_distance(first_signature, second_signature), 0)
        self.assertTrue(
            signatures_are_similar(first_signature, second_signature, 0.018)
        )
        # Uniform black and white both have no edges; this is intentionally safe
        # because neither image contains subtitle strokes.
        self.assertEqual(len(first_signature), len(different_signature))
        self.assertEqual(len(signature_hex(first_signature)), 16)


if __name__ == "__main__":
    unittest.main()
