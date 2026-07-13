import unittest

from PIL import Image

from bilibili_distiller_core.models import CropRegion
from bilibili_distiller_core.ocr import candidate_from_result, recognize_adaptive


def line(text, confidence, left=10, top=70, right=90, bottom=90):
    return [
        [[left, top], [right, top], [right, bottom], [left, bottom]],
        text,
        confidence,
    ]


class FakeEngine:
    name = "fake"

    def __init__(self, results):
        self.results = list(results)
        self.calls = 0

    def recognize(self, _image):
        result = self.results[min(self.calls, len(self.results) - 1)]
        self.calls += 1
        return result


class OcrTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (100, 100), "white")
        self.crop = CropRegion()

    def test_good_base_ocr_uses_one_model_call(self):
        engine = FakeEngine([[line("complete sentence", 0.95)]])
        result = recognize_adaptive(self.image, self.crop, engine)
        self.assertEqual(result.candidate.text, "complete sentence")
        self.assertEqual(result.calls, 1)

    def test_short_base_ocr_uses_enhancement_and_prefers_whole_sentence(self):
        engine = FakeEngine(
            [
                [line("I", 0.99)],
                [line("I understand the whole sentence", 0.84)],
            ]
        )
        result = recognize_adaptive(self.image, self.crop, engine)
        self.assertEqual(result.candidate.text, "I understand the whole sentence")
        self.assertEqual(result.calls, 2)
        self.assertEqual(result.candidate.variant, "enhanced")

    def test_very_poor_results_trigger_at_most_three_passes(self):
        engine = FakeEngine([[line("x", 0.2)]])
        result = recognize_adaptive(self.image, self.crop, engine)
        self.assertEqual(result.calls, 3)
        self.assertEqual(engine.calls, 3)

    def test_name_hint_is_retained_but_excluded_from_dialogue_text(self):
        # Local coordinates map to the configured global left-upper name region.
        raw = [
            line("Name", 0.96, left=20, top=15, right=32, bottom=35),
            line("dialogue body", 0.94, left=12, top=78, right=88, bottom=94),
        ]
        candidate = candidate_from_result(raw, (100, 100), self.crop, variant="base")
        self.assertEqual(candidate.text, "dialogue body")
        self.assertEqual(candidate.lines[0].role, "speaker_hint")

    def test_name_only_result_is_kept_as_raw_evidence_not_dialogue(self):
        raw = [line("Name", 0.96, left=20, top=15, right=32, bottom=35)]
        candidate = candidate_from_result(raw, (100, 100), self.crop, variant="base")
        self.assertEqual(candidate.text, "")
        self.assertEqual(candidate.lines[0].role, "speaker_hint")

    def test_words_on_same_visual_row_are_sorted_left_to_right(self):
        raw = [
            line("third", 0.9, left=70, top=68, right=95, bottom=88),
            line("first", 0.9, left=5, top=72, right=25, bottom=92),
            line("second", 0.9, left=35, top=70, right=65, bottom=90),
        ]
        candidate = candidate_from_result(raw, (100, 100), self.crop, variant="base")
        self.assertEqual(candidate.text, "first second third")

    def test_dialogue_row_crossing_name_regions_is_kept_whole(self):
        raw = [
            line("whole", 0.95, left=5, top=20, right=28, bottom=40),
            line("sentence", 0.95, left=33, top=18, right=68, bottom=40),
            line("test", 0.95, left=73, top=21, right=95, bottom=41),
        ]
        candidate = candidate_from_result(raw, (100, 100), self.crop, variant="base")
        self.assertEqual(candidate.text, "whole sentence test")
        self.assertTrue(all(item.role == "dialogue" for item in candidate.lines))


if __name__ == "__main__":
    unittest.main()
