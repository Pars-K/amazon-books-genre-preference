from collections import Counter
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from style_indicators import counts_from_text, metrics
from writing_style import measure_style


class LargerStyleInventoryTests(unittest.TestCase):
    def test_preserves_pilot_definitions(self):
        text = 'I’m not sure... really? <p>Dr. Wang thinks it’s fun!!!</p> “Do you?”'
        c = counts_from_text(text)
        old = measure_style(text)
        for key, value in metrics(c).items():
            if key in old:
                self.assertAlmostEqual(value, old[key], msg=key)
        self.assertEqual(c["quote_reviews"], 1)
        self.assertEqual(c["emphatic_runs"], 1)

    def test_fixed_windows_and_user_denominators(self):
        a = counts_from_text(" ".join(["word"] * 20))
        b = counts_from_text(" ".join("token" + str(i) for i in range(41)))
        c = counts_from_text("Tiny text.")
        self.assertIsNone(metrics(c)["vocabulary_diversity_20_pct"])
        combined = Counter(a) + Counter(b) + Counter(c)
        self.assertEqual(combined["windows20"], 3)
        self.assertEqual(combined["distinct_in_windows"], 41)
        self.assertAlmostEqual(metrics(combined)["vocabulary_diversity_20_pct"], 100 * 41 / 60)
        self.assertEqual(metrics(combined)["words_per_review"], 63 / 3)

    def test_sentence_functions_overlap_and_do_not_cross_boundaries(self):
        c = counts_from_text("I highly recommend this because it is well-written, but I highly recommend a sequel too. "
                             "I highly. Recommend this?")
        for key in ("recommendation", "evaluation", "causal", "contrast", "subordination"):
            self.assertEqual(c[key + "_sentences"], 1)
        self.assertEqual(c["sentences"], 3)
        self.assertAlmostEqual(metrics(c)["recommendation_sentence_pct"], 100 / 3)

    def test_phrases_and_emphasis_use_whole_tokens(self):
        c = counts_from_text("USA, I, 3D, noisy... nobody is gonna go?! Why howeverish?")
        self.assertEqual((c["upper_tokens"], c["colloquial_tokens"], c["emphatic_runs"]), (1, 1, 1))
        self.assertEqual(c["contrast_sentences"], 0)
