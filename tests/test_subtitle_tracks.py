import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from bilibili_distiller_core import core
from bilibili_distiller_core.models import VideoRef
from bilibili_distiller_core.subtitle_tracks import (
    SubtitleCue,
    download_best_subtitle_track,
    is_dialogue_cue,
    parse_ass,
    parse_json_track,
    parse_srt_or_vtt,
)


class SubtitleTrackTests(unittest.TestCase):
    def test_srt_vtt_ass_and_json_are_parsed_to_cues(self):
        srt = "1\n00:00:01,000 --> 00:00:03,000\nHello <i>world</i>\n"
        self.assertEqual(parse_srt_or_vtt(srt)[0], SubtitleCue(1.0, 3.0, "Hello world"))
        ass = "Dialogue: 0,0:00:02.00,0:00:04.00,Default,,0,0,0,,{\\i1}Line\\Ntwo"
        self.assertEqual(parse_ass(ass)[0].text, "Line\ntwo")
        json_text = '{"body":[{"from":3,"to":5,"content":"text"}]}'
        self.assertEqual(parse_json_track(json_text)[0], SubtitleCue(3.0, 5.0, "text"))

    def test_music_only_cue_is_not_dialogue(self):
        self.assertFalse(is_dialogue_cue(SubtitleCue(0, 2, "[BGM]")))
        self.assertTrue(is_dialogue_cue(SubtitleCue(0, 2, "actual dialogue")))

    def test_best_track_prefers_human_chinese_and_rejects_music(self):
        seen_command = []

        def fake_runner(command, **_kwargs):
            seen_command.append(command)
            work_dir = Path(command[command.index("--output") + 1]).parent
            (work_dir / "video.ai.zh.srt").write_text(
                "1\n00:00:00,000 --> 00:00:02,000\nauto line\n", encoding="utf-8"
            )
            (work_dir / "video.zh.srt").write_text(
                "1\n00:00:00,000 --> 00:00:02,000\nhuman line\n", encoding="utf-8"
            )
            (work_dir / "video.ja.srt").write_text(
                "1\n00:00:00,000 --> 00:00:02,000\n[BGM]\n", encoding="utf-8"
            )
            (work_dir / "video.NA.info.json").write_text(
                '{"id":"BV1uknVz9EeN","title":"Example"}', encoding="utf-8"
            )
            return subprocess.CompletedProcess(command, 0, "", "")

        video = VideoRef(
            "BV1uknVz9EeN", "https://example.invalid", "BV1uknVz9EeN", "BV1uknVz9EeN"
        )
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.object(core.shutil, "which", return_value=None):
                result = download_best_subtitle_track(
                    video, Path(temporary), command_runner=fake_runner
                )
        self.assertTrue(result.usable)
        self.assertEqual(result.selected_path.name, "video.zh.srt")
        self.assertEqual(result.cues[0].text, "human line")
        self.assertEqual(result.title, "Example")
        self.assertIn("--impersonate", seen_command[0])
        self.assertIn("chrome", seen_command[0])

    def test_bbdown_subtitle_command_is_preferred_and_validates_output(self):
        seen_command = []

        def fake_runner(command, **_kwargs):
            seen_command.append(command)
            work_dir = Path(command[command.index("--work-dir") + 1])
            (work_dir / "video.zh.srt").write_text(
                "1\n00:00:00,000 --> 00:00:02,000\nBBDown line\n",
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(command, 0, "Title: Example", "")

        video = VideoRef(
            "BV1uknVz9EeN", "https://example.invalid", "BV1uknVz9EeN", "BV1uknVz9EeN"
        )
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.object(
                core.shutil, "which", return_value="/usr/local/bin/BBDown"
            ):
                result = download_best_subtitle_track(
                    video, Path(temporary), command_runner=fake_runner
                )

        self.assertTrue(result.usable)
        self.assertEqual(result.cues[0].text, "BBDown line")
        self.assertEqual(seen_command[0][0], "/usr/local/bin/BBDown")
        self.assertIn("--sub-only", seen_command[0])
        self.assertIn("--skip-ai=false", seen_command[0])
        self.assertIn("-F", seen_command[0])

    def test_bbdown_video_command_is_low_quality_and_requires_file(self):
        seen_command = []

        def fake_runner(command, **_kwargs):
            seen_command.append(command)
            work_dir = Path(command[command.index("--work-dir") + 1])
            (work_dir / "video.mp4").write_bytes(b"video")
            return subprocess.CompletedProcess(command, 0, "", "")

        video = VideoRef(
            "BV1uknVz9EeN", "https://example.invalid", "BV1uknVz9EeN", "BV1uknVz9EeN"
        )
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.object(
                core.shutil, "which", return_value="/usr/local/bin/BBDown"
            ):
                path = core.download_low_quality_video(
                    video, Path(temporary), command_runner=fake_runner
                )

        self.assertEqual(path.name, "video.mp4")
        self.assertIn("--video-only", seen_command[0])
        self.assertIn("--skip-mux", seen_command[0])
        self.assertIn("360P 流畅,480P 清晰", seen_command[0])


if __name__ == "__main__":
    unittest.main()
