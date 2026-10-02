from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence


def catalog_coverage(
    predictions: Mapping[int, Sequence[int]],
    catalog_size: int,
    cutoffs: Sequence[int] = (5, 10, 15, 20),
) -> dict[str, float]:
    """Fraction of catalog APIs appearing at least once in all Top-K lists."""
    if catalog_size <= 0:
        raise ValueError("catalog_size must be positive")
    result = {}
    for k in cutoffs:
        recommended = {
            int(api)
            for ranking in predictions.values()
            for api in ranking[:k]
            if api is not None
        }
        result[f"coverage@{k}"] = len(recommended) / catalog_size
    return result


def ranking_metrics(
    predictions: Mapping[int, Sequence[int]],
    ground_truth: Mapping[int, Iterable[int]],
    cutoffs: Sequence[int] = (5, 10, 15, 20),
) -> dict[str, float]:
    totals = {f"precision@{k}": 0.0 for k in cutoffs}
    totals.update({f"recall@{k}": 0.0 for k in cutoffs})
    totals.update({f"ndcg@{k}": 0.0 for k in cutoffs})
    evaluated = 0
    for mashup_id, targets_iter in ground_truth.items():
        targets = set(int(value) for value in targets_iter)
        if not targets:
            continue
        ranking = [int(value) if value is not None else None for value in predictions.get(mashup_id, predictions.get(str(mashup_id), []))]
        evaluated += 1
        for k in cutoffs:
            top = ranking[:k]
            hits = [1 if item in targets else 0 for item in top]
            totals[f"precision@{k}"] += sum(hits) / k
            totals[f"recall@{k}"] += sum(hits) / len(targets)
            dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hits))
            ideal = sum(1.0 / math.log2(rank + 2) for rank in range(min(len(targets), k)))
            totals[f"ndcg@{k}"] += dcg / ideal if ideal else 0.0
    if not evaluated:
        return {**{key: 0.0 for key in totals}, "queries": 0}
    return {**{key: value / evaluated for key, value in totals.items()}, "queries": evaluated}
