"""Independent numerical checks for the custom classifier and evaluation."""

import sys
from pathlib import Path
import unittest

import numpy as np
from scipy import sparse
from scipy.optimize import check_grad

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from predictive_feasibility import logistic, objective, score_matrix


class PredictiveFeasibilityTests(unittest.TestCase):
    def test_softmax_gradient_matches_finite_differences(self):
        rng = np.random.default_rng(311)
        x = sparse.csr_matrix(np.column_stack([rng.normal(size=(9, 4)), np.ones(9)]))
        y = np.array([0, 1, 2, 0, 2, 1, 0, 1, 2])
        theta = rng.normal(size=15) / 3
        error = check_grad(lambda z: objective(z, x, y, 3, .03)[0],
                           lambda z: objective(z, x, y, 3, .03)[1], theta)
        self.assertLess(error, 1e-6)

    def test_known_confusion_scores(self):
        # F1 = (2/3, 2/3, 1); recalls = (1/2, 1, 1).
        c = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 2]])
        result = score_matrix(c)
        self.assertAlmostEqual(result["macro_f1"], 7 / 9)
        self.assertAlmostEqual(result["balanced_accuracy"], 5 / 6)
        self.assertAlmostEqual(result["accuracy"], 4 / 5)

    def test_logistic_separates_known_unseen_points(self):
        x = sparse.csr_matrix(np.array([-3, -2, -1, 1, 2, 3])[:, None])
        xt = sparse.csr_matrix(np.array([-4, -.5, .5, 4])[:, None])
        pred, diagnostics = logistic(x, np.array([0, 0, 0, 1, 1, 1]), xt, 2, .01)
        np.testing.assert_array_equal(pred, [0, 0, 1, 1])
        self.assertLess(diagnostics["gradient_max"], 1e-5)


if __name__ == "__main__":
    unittest.main()
