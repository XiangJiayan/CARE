"""Evaluate direct Sentence-T5 cosine retrieval on ProgrammableWeb."""
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
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from care.metrics import catalog_coverage, ranking_metrics


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_pack(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def rank_scores(scores: np.ndarray, mashup_ids: Sequence[int],
                api_ids: Sequence[int], k: int) -> dict[int, list[int]]:
    api_array = np.asarray(api_ids, dtype=np.int64)
    result = {}
    for row, mashup_id in enumerate(mashup_ids):
        order = np.lexsort((api_array, -scores[row]))[:k]
        result[int(mashup_id)] = api_array[order].astype(int).tolist()
    return result


def evaluate(workspace: Path, split_seed: int,
             cutoffs: Sequence[int]) -> tuple[dict, dict]:
    required = [
        workspace / "prepared.json",
        workspace / "api_embeddings.pt",
        workspace / "strict" / "mashup_embeddings.pt",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing frozen final-experiment files: " + ", ".join(missing))

    prepared = load_json(workspace / "prepared.json")
    stored_seed = int(prepared.get("split_seed", split_seed))
    if stored_seed != split_seed:
        raise ValueError(f"Workspace split seed is {stored_seed}, expected {split_seed}")

    api_pack = load_pack(workspace / "api_embeddings.pt")
    mashup_pack = load_pack(workspace / "strict" / "mashup_embeddings.pt")
    api_ids = [int(value) for value in api_pack["api_ids"].tolist()]
    mashup_ids = [int(value) for value in mashup_pack["mashup_ids"].tolist()]
    mashup_index = {mid: index for index, mid in enumerate(mashup_ids)}
    splits = {name: [int(value) for value in values]
              for name, values in prepared["splits"].items()}
    links = {int(mid): [int(api) for api in values]
             for mid, values in prepared["links"].items()}

    test_ids = splits["test"]
    api_vectors = F.normalize(api_pack["embeddings"].float(), p=2, dim=1)
    query_vectors = torch.stack([
        mashup_pack["embeddings"][mashup_index[mid]].float() for mid in test_ids
    ])
    query_vectors = F.normalize(query_vectors, p=2, dim=1)
    scores = (query_vectors @ api_vectors.T).numpy()
    predictions = rank_scores(scores, test_ids, api_ids, max(cutoffs))
    truth = {mid: links[mid] for mid in test_ids}

    metrics = ranking_metrics(predictions, truth, cutoffs)
    metrics.update(catalog_coverage(predictions, len(api_ids), cutoffs))

    counts = Counter(api for mid in splits["train"] for api in links[mid])
    seen = sorted((api for api in api_ids if counts.get(api, 0) > 0),
                  key=lambda api: (-counts[api], api))
    head_size = math.ceil(0.2 * len(seen))
    tail = set(seen[head_size:])
    tail_truth = {mid: [api for api in targets if api in tail]
                  for mid, targets in truth.items()}
    tail_truth = {mid: targets for mid, targets in tail_truth.items() if targets}
    tail_metrics = ranking_metrics(
        {mid: predictions[mid] for mid in tail_truth}, tail_truth, cutoffs)
    for k in cutoffs:
        metrics[f"tail_recall@{k}"] = tail_metrics[f"recall@{k}"]
    metrics["tail_queries"] = len(tail_truth)

    metadata = {
        "method": "Sentence-T5 Direct",
        "split_seed": split_seed,
        "train_mashups": len(splits["train"]),
        "valid_mashups": len(splits["valid"]),
        "test_mashups": len(test_ids),
        "candidate_apis": len(api_ids),
        "embedding_dim": int(api_vectors.shape[1]),
        "encoder": str(api_pack.get("encoder", "sentence-t5-base")),
        "query_variant": str(mashup_pack.get("variant", "strict")),
        "similarity": "cosine similarity between frozen Mashup and API embeddings",
        "supervised_training": False,
        "popularity_fusion": False,
        "tail_definition": "training-seen APIs outside the top 20% by training frequency",
    }
    return metrics, {"metadata": metadata, "predictions": predictions}


def summarize(rows: list[dict], metric_keys: Sequence[str]) -> dict:
    result = {}
    for key in metric_keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        result[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }
    return result


def write_report(path: Path, metadata: dict, summary: dict,
                 cutoffs: Sequence[int], seeds: Sequence[int]) -> None:
    lines = [
        "# Sentence-T5 Direct Baseline Report", "",
        f"- Split seed: `{metadata['split_seed']}`",
        f"- Reporting seeds: `{', '.join(map(str, seeds))}`",
        "- Sentence-T5 Direct is deterministic; repeated reports have zero standard deviation.",
        f"- Encoder: `{metadata['encoder']}`",
        f"- Embedding dimension: `{metadata['embedding_dim']}`; query variant: `{metadata['query_variant']}`.",
        "- Ranking uses cosine similarity only; no interaction supervision or popularity fusion.",
        f"- Test Mashups: `{metadata['test_mashups']}`; candidate APIs: `{metadata['candidate_apis']}`.", "",
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
              "Sentence-T5 Direct measures the value of pretrained semantic embeddings alone. "
              "Improvements over this baseline can be attributed to recommendation supervision, "
              "semantic-ID generation, catalog grounding, retrieval alignment, and score fusion.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(workspace: Path, output_dir: Path, split_seed: int,
        seeds: Sequence[int], cutoffs: Sequence[int]) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics, details = evaluate(workspace, split_seed, cutoffs)
    rows = [{"method": "Sentence-T5 Direct", "seed": int(seed), **metrics}
            for seed in seeds]
    metric_keys = [f"{metric}@{k}" for k in cutoffs
                   for metric in ("precision", "recall", "ndcg", "coverage", "tail_recall")]
    summary = summarize(rows, metric_keys)

    fields = ["method", "seed", *metric_keys, "queries", "tail_queries"]
    with (output_dir / "per_seed_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    payload = {"metadata": details["metadata"],
               "seeds": [int(seed) for seed in seeds], "metrics": summary}
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "test_top20_predictions.json").write_text(
        json.dumps({str(mid): ranking for mid, ranking in details["predictions"].items()},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(output_dir / "REPORT.md", details["metadata"], summary, cutoffs, seeds)
    print(f"Sentence-T5 Direct experiment complete: {output_dir}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Sentence-T5 Direct baseline")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43, 71, 101])
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[5, 10, 15, 20])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.workspace, args.output_dir, args.split_seed, args.seeds, args.cutoffs)

