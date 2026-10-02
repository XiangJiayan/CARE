"""Run the direct content-based TF-IDF baseline on ProgrammableWeb."""
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
from sklearn.feature_extraction.text import TfidfVectorizer

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from care.data import (
    build_api_texts,
    build_mashup_texts,
    load_pw_data,
    relation_map,
    split_mashups,
)
from care.metrics import catalog_coverage, ranking_metrics


def rank_scores(scores: np.ndarray, mashup_ids: Sequence[int], api_ids: Sequence[int], k: int) -> dict[int, list[int]]:
    api_array = np.asarray(api_ids, dtype=np.int64)
    predictions = {}
    for row, mashup_id in enumerate(mashup_ids):
        # Primary key: descending similarity; deterministic tie-break: ascending API ID.
        order = np.lexsort((api_array, -scores[row]))[:k]
        predictions[int(mashup_id)] = api_array[order].astype(int).tolist()
    return predictions


def evaluate_tfidf(
    data_dir: Path,
    split_seed: int,
    cutoffs: Sequence[int],
    max_features: int,
    ngram_range: tuple[int, int],
    stop_words: str | None,
    sublinear_tf: bool,
) -> tuple[dict, dict]:
    data = load_pw_data(data_dir)
    splits = split_mashups(data, seed=split_seed)
    api_texts = build_api_texts(data)
    mashup_texts = build_mashup_texts(data)
    api_ids = sorted(int(api) for api in data.apis["ID"].tolist())

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words=stop_words,
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf,
        norm="l2",
    )
    # Candidate API content is available at recommendation time. Test Mashup text is
    # transformed only and is not used to estimate vocabulary or IDF statistics.
    fit_texts = [api_texts[api] for api in api_ids]
    fit_texts += [mashup_texts[mid] for mid in splits["train"]]
    vectorizer.fit(fit_texts)
    api_matrix = vectorizer.transform([api_texts[api] for api in api_ids])
    test_matrix = vectorizer.transform([mashup_texts[mid] for mid in splits["test"]])
    scores = (test_matrix @ api_matrix.T).toarray().astype(np.float32)
    predictions = rank_scores(scores, splits["test"], api_ids, max(cutoffs))

    all_truth = relation_map(data.mashup_api, "MashupID", "ApiID")
    truth = {int(mid): all_truth[int(mid)] for mid in splits["test"]}
    metrics = ranking_metrics(predictions, truth, cutoffs)
    metrics.update(catalog_coverage(predictions, len(api_ids), cutoffs))

    train_set = set(splits["train"])
    train_edges = data.mashup_api[data.mashup_api["MashupID"].isin(train_set)]
    counts = Counter(int(api) for api in train_edges["ApiID"].tolist())
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
        "method": "TF-IDF Direct",
        "split_seed": int(split_seed),
        "train_mashups": len(splits["train"]),
        "valid_mashups": len(splits["valid"]),
        "test_mashups": len(splits["test"]),
        "candidate_apis": len(api_ids),
        "vocabulary_size": len(vectorizer.vocabulary_),
        "max_features": max_features,
        "ngram_range": list(ngram_range),
        "stop_words": stop_words,
        "sublinear_tf": bool(sublinear_tf),
        "similarity": "cosine similarity via L2-normalized TF-IDF dot product",
        "fit_corpus": "all candidate API texts plus training Mashup texts",
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
        "# TF-IDF Direct Baseline Report", "",
        f"- Split seed: `{metadata['split_seed']}`",
        f"- Reporting seeds: `{', '.join(map(str, seeds))}`",
        "- TF-IDF Direct is deterministic, so repeated reports have zero standard deviation.",
        f"- Fit corpus: {metadata['fit_corpus']}.",
        f"- Vocabulary size: `{metadata['vocabulary_size']}`; n-grams: `{tuple(metadata['ngram_range'])}`.",
        "- Test Mashup labels and invocation edges are never used for fitting or ranking.",
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
              "The method recommends APIs solely from lexical similarity between Mashup requirements "
              "and API catalog content. It uses neither Mashup-API interaction supervision nor popularity fusion.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(data_dir: Path, output_dir: Path, split_seed: int,
        seeds: Sequence[int], cutoffs: Sequence[int], max_features: int,
        ngram_range: tuple[int, int], stop_words: str | None,
        sublinear_tf: bool) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics, details = evaluate_tfidf(
        data_dir, split_seed, cutoffs, max_features,
        ngram_range, stop_words, sublinear_tf)
    rows = [{"method": "TF-IDF Direct", "seed": int(seed), **metrics} for seed in seeds]
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
    print(f"TF-IDF experiment complete: {output_dir}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ProgrammableWeb TF-IDF baseline")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43, 71, 101])
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[5, 10, 15, 20])
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument("--ngram-min", type=int, default=1)
    parser.add_argument("--ngram-max", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.data_dir, args.output_dir, args.split_seed, args.seeds, args.cutoffs,
        args.max_features, (args.ngram_min, args.ngram_max), "english", True)

