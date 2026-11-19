"""Numerical checks of training weights and weighted softmax gradients."""

from pathlib import Path
import sys
import unittest

import numpy as np
from scipy import sparse
from scipy.optimize import check_grad

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from balanced_genre_feasibility import group_weights, weighted_objective
from predictive_feasibility import objective


class BalancedTrainingTests(unittest.TestCase):
    def test_each_group_has_equal_mass_without_changing_mean_weight(self):
        y = np.array([0] * 12 + [1] * 3 + [2])
        _, weight = group_weights(y, 3)
        self.assertAlmostEqual(weight.mean(), 1)
        np.testing.assert_allclose([weight[y == j].sum() for j in range(3)], np.repeat(len(y) / 3, 3))

    def test_weighted_gradient_matches_finite_differences(self):
        rng = np.random.default_rng(113)
        x = sparse.csr_matrix(np.column_stack([rng.normal(size=(12, 4)), np.ones(12)]))
        y = np.array([0] * 8 + [1] * 3 + [2])
        _, weight = group_weights(y, 3)
        theta = rng.normal(size=15) / 4
        error = check_grad(lambda z: weighted_objective(z, x, y, 3, .03, weight)[0],
                           lambda z: weighted_objective(z, x, y, 3, .03, weight)[1], theta)
        self.assertLess(error, 1e-6)

    def test_equal_weights_reproduce_original_loss_and_gradient(self):
        rng = np.random.default_rng(114)
        x = sparse.csr_matrix(np.column_stack([rng.normal(size=(9, 2)), np.ones(9)]))
        y = np.array([0, 1, 2] * 3)
        theta = rng.normal(size=9)
        a = objective(theta, x, y, 3, .01)
        b = weighted_objective(theta, x, y, 3, .01, np.ones(9))
        self.assertAlmostEqual(a[0], b[0])
        np.testing.assert_allclose(a[1], b[1], atol=1e-14)

    def test_balancing_removes_frequency_only_intercept_gradient(self):
        y = np.array([0] * 7 + [1])
        x = sparse.csr_matrix(np.ones((8, 1)))
        _, weight = group_weights(y, 2)
        _, balanced_gradient = weighted_objective(np.zeros(2), x, y, 2, 0, weight)
        _, raw_gradient = objective(np.zeros(2), x, y, 2, 0)
        np.testing.assert_allclose(balanced_gradient, [0, 0], atol=1e-14)
        self.assertGreater(np.linalg.norm(raw_gradient), .1)


if __name__ == '__main__':
    unittest.main()
