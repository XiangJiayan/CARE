"""Run the deterministic Popularity baseline on ProgrammableWeb.

APIs are ranked by invocation count in training Mashups only. Validation and
test interactions never contribute to popularity, preventing data leakage.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from care.data import load_pw_data, relation_map, split_mashups
from care.metrics import catalog_coverage, ranking_metrics


def evaluate_popularity(data_dir: Path, split_seed: int, cutoffs: Sequence[int]) -> tuple[dict, dict]:
    data = load_pw_data(data_dir)
    splits = split_mashups(data, seed=split_seed)
    train_set = set(splits["train"])
    train_edges = data.mashup_api[data.mashup_api["MashupID"].isin(train_set)]
    counts = Counter(int(api) for api in train_edges["ApiID"].tolist())

    api_ids = sorted(int(api) for api in data.apis["ID"].tolist())
    ranking = sorted(api_ids, key=lambda api: (-counts.get(api, 0), api))
    max_k = max(cutoffs)
    predictions = {int(mid): ranking[:max_k] for mid in splits["test"]}

    all_truth = relation_map(data.mashup_api, "MashupID", "ApiID")
    truth = {int(mid): all_truth[int(mid)] for mid in splits["test"]}
    metrics = ranking_metrics(predictions, truth, cutoffs)
    metrics.update(catalog_coverage(predictions, len(api_ids), cutoffs))

    seen = sorted((api for api in api_ids if counts.get(api, 0) > 0),
                  key=lambda api: (-counts[api], api))
    head_size = math.ceil(0.2 * len(seen))
    tail = set(seen[head_size:])
    tail_truth = {
        mid: [api for api in targets if api in tail]
        for mid, targets in truth.items()
    }
    tail_truth = {mid: targets for mid, targets in tail_truth.items() if targets}
    tail_predictions = {mid: predictions[mid] for mid in tail_truth}
    tail_metrics = ranking_metrics(tail_predictions, tail_truth, cutoffs)
    for k in cutoffs:
        metrics[f"tail_recall@{k}"] = tail_metrics[f"recall@{k}"]
    metrics["tail_queries"] = len(tail_truth)

    metadata = {
        "method": "Popularity",
        "split_seed": int(split_seed),
        "train_mashups": len(splits["train"]),
        "valid_mashups": len(splits["valid"]),
        "test_mashups": len(splits["test"]),
        "candidate_apis": len(api_ids),
        "training_seen_apis": len(seen),
        "head_apis": head_size,
        "tail_apis": len(tail),
        "ranking_rule": "descending training invocation count, then ascending API ID",
        "tail_definition": "training-seen APIs outside the top 20% by training frequency",
    }
    return metrics, {"metadata": metadata, "predictions": predictions, "counts": counts}


def summarize(rows: list[dict], metric_keys: Sequence[str]) -> dict:
    result = {}
    for key in metric_keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        result[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }
    return result


def write_report(path: Path, metadata: dict, summary: dict, cutoffs: Sequence[int], seeds: Sequence[int]) -> None:
    lines = [
        "# Popularity Baseline Report", "",
        f"- Split seed: `{metadata['split_seed']}`",
        f"- Reporting seeds: `{', '.join(map(str, seeds))}`",
        "- Popularity is deterministic, so repeated runs have identical values and zero standard deviation.",
        "- API frequency is calculated from training Mashups only.",
        f"- Test Mashups: `{metadata['test_mashups']}`; candidate APIs: `{metadata['candidate_apis']}`.",
        f"- Tail definition: {metadata['tail_definition']}.", "",
        "| K | Precision | Recall | NDCG | Coverage | Tail Recall |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for k in cutoffs:
        values = []
        for metric in ("precision", "recall", "ndcg", "coverage", "tail_recall"):
            item = summary[f"{metric}@{k}"]
            values.append(f"{item['mean']:.6f} ± {item['std']:.6f}")
        lines.append(f"| {k} | " + " | ".join(values) + " |")
    lines += ["", "## Interpretation", "",
              "Popularity recommends the same globally frequent APIs to every Mashup. "
              "It is a non-personalized lower baseline and normally has low catalog coverage and tail recall.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(data_dir: Path, output_dir: Path, split_seed: int,
        seeds: Sequence[int], cutoffs: Sequence[int]) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics, details = evaluate_popularity(data_dir, split_seed, cutoffs)
    rows = [{"method": "Popularity", "seed": int(seed), **metrics} for seed in seeds]
    metric_keys = [f"{metric}@{k}" for k in cutoffs
                   for metric in ("precision", "recall", "ndcg", "coverage", "tail_recall")]
    summary = summarize(rows, metric_keys)

    fields = ["method", "seed", *metric_keys, "queries", "tail_queries"]
    with (output_dir / "per_seed_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "metadata": details["metadata"],
        "seeds": [int(seed) for seed in seeds],
        "metrics": summary,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "top20_ranking.json").write_text(
        json.dumps({"ranking": next(iter(details["predictions"].values()), []),
                    "training_counts": {str(api): int(details["counts"].get(api, 0))
                                        for api in next(iter(details["predictions"].values()), [])}},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(output_dir / "REPORT.md", details["metadata"], summary, cutoffs, seeds)
    print(f"Popularity experiment complete: {output_dir}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ProgrammableWeb Popularity baseline")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43, 71, 101])
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[5, 10, 15, 20])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.data_dir, args.output_dir, args.split_seed, args.seeds, args.cutoffs)

