import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from writing_style import measure_style


class WritingStyleTests(unittest.TestCase):
    def test_reference_example(self):
        result = measure_style("I'm glad. Are you? I’d read it again!")
        self.assertEqual((result["words"], result["sentences"]), (8, 3))
        self.assertEqual(result["mean_sentence_words"], 8 / 3)
        self.assertEqual(result["question_ending_pct"], 100 / 3)
        self.assertEqual(result["sentence_words_squared_sum"], 24)
        self.assertAlmostEqual(result["sd_sentence_words"], (8 ** 0.5) / 3)
        self.assertEqual(result["first_singular_per_1000_words"], 250)
        self.assertEqual(result["second_person"], 1)
        self.assertEqual(result["contractions"], 2)
        self.assertEqual(result["exclamations"], 1)
        self.assertFalse(result["has_quote_marker"])

    def test_quoted_speech_html_and_punctuation(self):
        result = measure_style('<p>Dr. Scott paid 3.50.</p><blockquote>&#34;Did you?!&#34;</blockquote>')
        self.assertEqual((result["words"], result["sentences"]), (7, 2))
        self.assertTrue(result["has_quote_marker"])
        self.assertEqual(result["question_endings"], 1)
        self.assertEqual(result["exclamations"], 1)
        self.assertEqual(result["semicolons"], 0)

    def test_ellipses_possessives_and_emphatic_punctuation(self):
        result = measure_style("The author's book... well, it's fun!!!<br/>We’re fans")
        self.assertEqual((result["words"], result["sentences"]), (8, 2))
        self.assertEqual(result["ellipses"], 1)
        self.assertEqual(result["exclamations"], 3)
        self.assertEqual(result["contractions"], 2)
        self.assertEqual(result["first_plural"], 1)

    def test_undefined_measurements_and_zero(self):
        empty = measure_style(" <br/> ... ")
        self.assertIsNone(empty["mean_sentence_words"])
        self.assertIsNone(empty["contractions_per_1000_words"])
        single = measure_style("Worth reading")
        self.assertIsNone(single["sd_sentence_words"])
        self.assertEqual(single["contractions_per_1000_words"], 0)
        self.assertEqual(single["question_ending_pct"], 0)

    def test_soft_wrap_and_paragraph_boundary(self):
        text = "All of<br />these books were good<br /><br />Worth reading"
        result = measure_style(text)
        self.assertEqual((result["words"], result["sentences"]), (8, 2))
