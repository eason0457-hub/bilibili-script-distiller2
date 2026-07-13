import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yt_dlp

from bilibili_distiller_core import core


class CoreNetworkTests(unittest.TestCase):
    def test_library_options_are_browser_like(self):
        opts = core._ydl_base_opts()
        self.assertTrue(opts["ignoreconfig"])
        self.assertTrue(opts["noplaylist"])
        self.assertEqual(opts["impersonate"], "chrome")
        self.assertEqual(
            opts["http_headers"]["Referer"],
            "https://www.bilibili.com/",
        )
        self.assertEqual(
            opts["http_headers"]["Origin"],
            "https://www.bilibili.com",
        )

    def test_412_is_retried_on_the_same_library_call(self):
        attempts = []

        class FakeYoutubeDL:
            def __init__(self, options):
                attempts.append(options)

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def extract_info(self, _url, *, download):
                if len(attempts) < 3:
                    raise yt_dlp.utils.DownloadError(
                        "HTTP Error 412: Precondition Failed"
                    )
                return {"id": "BV1uknVz9EeN", "download": download}

        with mock.patch.object(core.yt_dlp, "YoutubeDL", FakeYoutubeDL):
            with mock.patch.object(core.time, "sleep") as sleep:
                result = core._run_ydl(
                    "https://www.bilibili.com/video/BV1uknVz9EeN/",
                    core._ydl_base_opts(),
                    download=False,
                )

        self.assertEqual(result["id"], "BV1uknVz9EeN")
        self.assertEqual(len(attempts), 3)
        self.assertEqual(sleep.call_count, 2)
        self.assertTrue(all(item["impersonate"] == "chrome" for item in attempts))


class CoreOutputTests(unittest.TestCase):
    def test_failed_manifest_is_not_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            video = core.VideoRef(
                "BV1uknVz9EeN",
                "https://www.bilibili.com/video/BV1uknVz9EeN/",
                "BV1uknVz9EeN",
                "BV1uknVz9EeN",
            )
            config = core.PipelineConfig()
            fingerprint = config.fingerprint(video.output_key)
            store = core.OutputStore(output, video.output_key)
            store.write_manifest(
                core.base_manifest(
                    video,
                    config,
                    fingerprint,
                    status="failed",
                )
            )
            self.assertFalse(store.cache_is_valid(fingerprint))
            self.assertFalse(
                core.already_done(store.directory, fingerprint, force=False)
            )


if __name__ == "__main__":
    unittest.main()
