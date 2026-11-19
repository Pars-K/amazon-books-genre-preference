"""Regression checks for label evidence, sparse transforms and unsupervised metrics."""

import sys
from pathlib import Path
import unittest

import numpy as np
from scipy import sparse
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phase2_patterns import ari, silhouette, svd_fit, terms, vectorize
from phase2_sample import qualifying
from phase2_interpret import conditional_info


class PhaseTwoChecks(unittest.TestCase):
    def test_held_out_book_can_change_label_and_preserve_other_evidence(self):
        day = 86400000
        start = 1600000000000
        mapping = {"a":"Fantasy", "b":"Fantasy", "c":"Fantasy", "d":"Science"}
        history = {"a":start, "b":start+190*day, "c":start+370*day, "d":start+500*day}
        rules = dict(minimum_mapping_coverage=.5, minimum_genre_books=3,
                     minimum_share_all_books=.5, minimum_dated_genre_books=3,
                     minimum_active_quarters=2, minimum_span_days=180)
        self.assertEqual(qualifying(history, mapping, rules), ["Fantasy"])
        without_science = {k:v for k,v in history.items() if k!="d"}
        self.assertEqual(qualifying(without_science, mapping, rules), ["Fantasy"])
        without_fantasy = {k:v for k,v in history.items() if k!="b"}
        self.assertEqual(qualifying(without_fantasy, mapping, rules), [])

    def test_frozen_idf_oov_empty_document_and_l2_norm(self):
        x = vectorize([{"known":2, "other":99}, {"unseen":7}, {}], ["known"], np.array([2.0]))
        np.testing.assert_allclose(x.toarray(), [[1], [0], [0]])
        self.assertTrue(np.isfinite(x.data).all())

    def test_svd_retains_a_full_rank_small_example(self):
        x = sparse.csr_matrix(np.array([[1.,0,0],[0,2.,0],[0,0,3.],[1.,1.,1.]]))
        singular, axes = svd_fit(x, rank=3)
        reconstructed = (x @ axes.T) @ axes
        np.testing.assert_allclose(reconstructed, x.toarray(), atol=1e-10)
        np.testing.assert_allclose(singular, np.linalg.svd(x.toarray(), compute_uv=False), atol=1e-10)

    def test_cluster_label_permutation_and_singleton_silhouette(self):
        self.assertEqual(ari([0,0,1,1],[2,2,5,5]), 1)
        self.assertAlmostEqual(ari([0,0,1,1],[0,1,0,1]), -.5)
        x = np.array([[0.],[1.],[5.]])
        # First two samples contribute 0.8 and 0.75; the singleton contributes zero.
        self.assertAlmostEqual(silhouette(cdist(x,x), np.array([0,0,1])), (0.8+0.75)/3)

    def test_decorative_unicode_does_not_create_separate_lexical_features(self):
        self.assertEqual(terms("Ｗｏｎｄｅｒｆｕｌ 𝐬𝐭𝐨𝐫𝐲"),terms("Wonderful story"))

    def test_conditioning_can_remove_one_bit_of_shared_information(self):
        labels = np.array([0,0,1,1])
        shared = conditional_info(labels, labels, np.zeros(4), repeats=9)
        conditioned = conditional_info(labels, labels, labels, repeats=9)
        self.assertAlmostEqual(shared["information_bits"],1)
        self.assertAlmostEqual(conditioned["information_bits"],0)


if __name__ == "__main__":
    unittest.main()
