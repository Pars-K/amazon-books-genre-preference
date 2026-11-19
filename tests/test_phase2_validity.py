"""Check sparse, tied comparisons against exact cases with known answers."""

from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.stats import fisher_exact

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from audit_phase2_validity import rank_permutation_p


class ValidityCalibrationTests(unittest.TestCase):
    def test_identical_constant_rates_have_no_distribution_difference(self):
        p = rank_permutation_p(np.zeros(135),np.zeros(22),np.random.default_rng(62),repeats=199)
        self.assertEqual(p,1)

    def test_rare_tied_nonzero_values_match_the_exact_binary_case(self):
        a = np.zeros(135)
        b = np.r_[np.ones(3),np.zeros(19)]
        exact = fisher_exact([[0,135],[3,19]]).pvalue
        estimated = rank_permutation_p(a,b,np.random.default_rng(62),repeats=19999)
        self.assertAlmostEqual(estimated,exact,delta=.0015)


if __name__ == "__main__":
    unittest.main()
