import unittest

from bilibili_distiller_core.models import OcrSample
from bilibili_distiller_core.sentences import (
    assemble_sentences,
    clamp_segment_end_times,
)


def sample(time, text, confidence=0.9, quality=0.85):
    return OcrSample(
        time=time,
        text=text,
        confidence=confidence,
        quality=quality,
        signature="00",
        reused=False,
        ocr_calls=1,
        variant="base",
    )


class SentenceAssemblyTests(unittest.TestCase):
    def test_progressive_caption_prefers_supported_whole_sentence(self):
        result = assemble_sentences(
            [
                sample(0, "I want", confidence=0.98, quality=0.90),
                sample(3, "I want to tell you", confidence=0.84, quality=0.82),
            ],
            3,
        )
        self.assertEqual(len(result.segments), 1)
        self.assertEqual(result.segments[0].text, "I want to tell you")
        self.assertEqual(result.segments[0].reconstruction, "temporal_consensus")

    def test_overlapping_fragments_form_one_sentence(self):
        result = assemble_sentences([sample(0, "abcdef"), sample(3, "defghij")], 3)
        self.assertEqual([item.text for item in result.segments], ["abcdefghij"])
        self.assertEqual(result.stats.overlap_joins, 1)

    def test_unrelated_adjacent_lines_are_not_forced_together(self):
        result = assemble_sentences(
            [sample(0, "first line"), sample(3, "second line")], 3
        )
        self.assertEqual(len(result.segments), 2)

    def test_distinct_sentences_with_long_shared_suffix_stay_separate(self):
        result = assemble_sentences(
            [
                sample(0, "FIRST COMPLETE SENTENCE"),
                sample(3, "SECOND COMPLETE SENTENCE"),
                sample(6, "FINAL COMPLETE SENTENCE"),
            ],
            3,
        )
        self.assertEqual(
            [item.text for item in result.segments],
            [
                "FIRST COMPLETE SENTENCE",
                "SECOND COMPLETE SENTENCE",
                "FINAL COMPLETE SENTENCE",
            ],
        )

    def test_low_confidence_single_character_noise_is_dropped(self):
        result = assemble_sentences([sample(0, "x", confidence=0.4, quality=0.2)], 3)
        self.assertEqual(result.segments, [])
        self.assertEqual(result.stats.dropped_low_confidence_fragments, 1)

    def test_high_confidence_single_character_utterance_is_preserved(self):
        result = assemble_sentences([sample(0, "x", confidence=0.95, quality=0.8)], 3)
        self.assertEqual([item.text for item in result.segments], ["x"])

    def test_last_segment_is_clamped_to_video_end(self):
        result = assemble_sentences([sample(6, "last line")], 3)
        clamped = clamp_segment_end_times(result.segments, 8)
        self.assertEqual(clamped[0].end, 8)


if __name__ == "__main__":
    unittest.main()
