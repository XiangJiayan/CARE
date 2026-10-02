import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retrieval_models import fused_score
from care.semantic_ids import make_unique_codes
from care.metrics import catalog_coverage, ranking_metrics


class CoreTests(unittest.TestCase):
    def test_semantic_ids_are_unique_after_collision_suffix(self):
        result = make_unique_codes([3, 1, 2], [[1, 2], [1, 2], [5, 6]])
        self.assertEqual(len({tuple(value) for value in result.values()}), 3)

    def test_fusion_keeps_shape_and_finite_values(self):
        a = np.asarray([[1, 2, 4], [2, 1, 0]], dtype=np.float32)
        b = np.asarray([[3, 0, 1], [4, 3, 2]], dtype=np.float32)
        result = fused_score(a, b, np.asarray([.2, .8]))
        self.assertEqual(result.shape, a.shape)
        self.assertTrue(np.isfinite(result).all())

    def test_metrics_use_requested_cutoffs_and_only_requested_families(self):
        result = ranking_metrics({1: [10, 30, 20]}, {1: [10, 20]})
        expected = {f"{name}@{k}" for name in ("precision", "recall", "ndcg")
                    for k in (5, 10, 15, 20)} | {"queries"}
        self.assertEqual(set(result), expected)

    def test_catalog_coverage_uses_each_top_k_prefix(self):
        prediction = {1: [1, 2, 3], 2: [2, 4, 5]}
        result = catalog_coverage(prediction, catalog_size=10, cutoffs=(1, 2, 3))
        self.assertEqual(result, {"coverage@1": .2, "coverage@2": .3, "coverage@3": .5})


if __name__ == "__main__":
    unittest.main()
