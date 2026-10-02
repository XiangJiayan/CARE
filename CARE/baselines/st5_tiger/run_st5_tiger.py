"""Run the direct Sentence-T5 adaptation of TIGER without retrieval additions."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import numpy as np
import torch
from transformers import Adafactor

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from models import SemanticGenerator, generate
from pipeline import Data, generator_scores, rankings
from care.io_utils import load_checkpoint, save_checkpoint, seed_all
from care.metrics import catalog_coverage, ranking_metrics


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def trusted_load(path: Path) -> dict:
    """Load only project-produced score caches that contain NumPy arrays."""
    return torch.load(path, map_location="cpu", weights_only=False)


def training_args(args):
    return SimpleNamespace(
        device=args.device, generator_steps=args.generator_steps,
        generator_min_steps=args.generator_min_steps,
        generator_eval_every=args.generator_eval_every,
        generator_batch=args.generator_batch, eval_batch=args.eval_batch,
        patience=args.patience,
    )


def train_generator(seed: int, data: Data, params: dict,
                    output_dir: Path, args) -> dict:
    local = output_dir / "models" / "checkpoints" / f"generator_st5_tiger_seed{seed}.pt"
    local.parent.mkdir(parents=True, exist_ok=True)
    if local.exists():
        payload = load_checkpoint(local)
        return {"checkpoint": str(local), "validation": payload["validation"],
                "seconds": payload.get("seconds", 0.0)}
    seed_all(seed)
    started = time.time()
    model = SemanticGenerator(data.vocab.state["size"],
                              memory_tokens=params["memory_tokens"],
                              dropout_rate=params["dropout"]).to(args.device)
    optimizer = Adafactor(model.parameters(), lr=params["lr"], relative_step=False,
                          scale_parameter=False, warmup_init=False)
    edges = [(mid, api) for mid in data.splits["train"] for api in data.links[mid]]
    valid = data.splits["valid"]
    queries = {mid: data.query[mid] for mid in valid}
    truth = data.truth("valid")
    rng = np.random.default_rng(seed + 202)
    best, best_metric, best_state, stale, history = -1.0, None, None, 0, []
    for step in range(1, args.generator_steps + 1):
        chosen = rng.integers(len(edges), size=args.generator_batch)
        rows = [edges[index] for index in chosen]
        query = torch.stack([data.query[mid] for mid, _ in rows]).to(args.device)
        labels = data.labels([api for _, api in rows]).to(args.device)
        positive = torch.ones(len(rows), dtype=torch.bool, device=args.device)
        weights = torch.ones(len(rows), device=args.device)
        optimizer.zero_grad(set_to_none=True)
        lr = params["lr"] * min(1.0, math.sqrt(1000 / max(1, step)))
        for group in optimizer.param_groups:
            group["lr"] = lr
        loss, generation, _, _ = model.loss(query, labels, positive, [], weights, 0.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % args.generator_eval_every == 0 or step == args.generator_steps:
            prediction = generate(model, data.vocab, queries, args.device,
                                  batch_size=args.eval_batch)
            metric = ranking_metrics(prediction, truth, args.cutoffs)
            history.append({"step": step, "loss": float(generation.detach()),
                            "validation": metric})
            if metric["ndcg@10"] > best:
                best, best_metric, stale = metric["ndcg@10"], metric, 0
                best_state = {key: value.detach().cpu()
                              for key, value in model.state_dict().items()}
            else:
                stale += 1
            if step >= args.generator_min_steps and stale >= args.patience:
                break
    seconds = time.time() - started
    save_checkpoint(local, {"model": best_state, "validation": best_metric,
                            "params": params, "seed": seed,
                            "seconds": seconds, "history": history})
    return {"checkpoint": str(local), "validation": best_metric, "seconds": seconds}


@torch.no_grad()
def score_seed(seed: int, row: dict, data: Data, params: dict,
               output_dir: Path, args) -> dict:
    cache = output_dir / "score_cache" / f"test_seed{seed}.pt"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        return trusted_load(cache)
    model = SemanticGenerator(data.vocab.state["size"],
                              memory_tokens=params["memory_tokens"],
                              dropout_rate=params["dropout"]).to(args.device)
    model.load_state_dict(load_checkpoint(row["checkpoint"])["model"])
    mids = data.splits["test"]
    query = torch.stack([data.query[mid] for mid in mids])
    payload = {"mids": mids, "generated": generator_scores(
        model, data.vocab, query, data.api_ids, args.device,
        args.score_query_batch, args.score_api_batch)}
    torch.save(payload, cache)
    print(f"SCORED ST5-TIGER seed={seed}", flush=True)
    return payload


def evaluate(score: np.ndarray, mids: Sequence[int], data: Data,
             cutoffs: Sequence[int]) -> tuple[dict, dict[int, list[int]]]:
    prediction = rankings(score, mids, data.api_ids, k=max(cutoffs))
    truth = data.truth("test")
    result = ranking_metrics(prediction, truth, cutoffs)
    result.update(catalog_coverage(prediction, len(data.api_ids), cutoffs))
    seen = sorted((api for api in data.api_ids if data.counts[api]),
                  key=lambda api: (-data.counts[api], api))
    tail = set(seen[math.ceil(0.2 * len(seen)):])
    tail_truth = {mid: [api for api in targets if api in tail]
                  for mid, targets in truth.items()}
    tail_truth = {mid: values for mid, values in tail_truth.items() if values}
    tail_metrics = ranking_metrics(
        {mid: prediction[mid] for mid in tail_truth}, tail_truth, cutoffs)
    for k in cutoffs:
        result[f"tail_recall@{k}"] = tail_metrics[f"recall@{k}"]
    result["tail_queries"] = len(tail_truth)
    return result, prediction


def summarize(rows: list[dict], keys: Sequence[str]) -> dict:
    output = {}
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows])
        output[key] = {"mean": float(values.mean()),
                       "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0}
    return output


def write_report(path: Path, summary: dict, metadata: dict,
                 cutoffs: Sequence[int], seeds: Sequence[int]) -> None:
    lines = ["# ST5-TIGER Baseline Report", "",
             f"- Split seed: `{metadata['split_seed']}`",
             f"- Training seeds: `{', '.join(map(str, seeds))}`",
             "- Definition: TIGER semantic-ID generator directly adapted to Mashup-to-API recommendation.",
             "- Sentence encoder: Sentence-T5; API identifiers: RQ-VAE semantic IDs from the shared workspace.",
             "- No retriever, catalog grounding, confidence gate, score fusion, or supervised alignment branch.",
             f"- Generator parameters: lr=`{metadata['params']['lr']}`, memory tokens=`{metadata['params']['memory_tokens']}`, dropout=`{metadata['params']['dropout']}`.",
             f"- Validation-selected steps: `{metadata['best_steps']}`.", "",
             "| K | Precision | Recall | NDCG | Coverage | Tail Recall |",
             "|---:|---:|---:|---:|---:|---:|"]
    for k in cutoffs:
        cells = []
        for metric in ("precision", "recall", "ndcg", "coverage", "tail_recall"):
            value = summary[f"{metric}@{k}"]
            cells.append((f"{100*value['mean']:.2f}% ± {100*value['std']:.2f}%"
                          if metric == "coverage" else f"{value['mean']:.6f} ± {value['std']:.6f}"))
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    lines += ["", "## Scope", "",
              "This is a method-level direct migration, not an unchanged execution of the original sequential-recommendation code. ProgrammableWeb has unordered API sets, so the Mashup Sentence-T5 representation conditions the SID generator and all called APIs are positive targets.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args) -> dict:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = Data(SimpleNamespace(output_dir=str(args.workspace)), "strict")
    params = {"lr": args.generator_lr,
              "memory_tokens": args.generator_memory_tokens,
              "dropout": args.generator_dropout}
    rows, runs, predictions = [], [], {}
    for seed in args.seeds:
        trial = train_generator(seed, data, params, output_dir, args)
        payload = score_seed(seed, trial, data, params, output_dir, args)
        metrics, ranking = evaluate(payload["generated"], payload["mids"], data, args.cutoffs)
        rows.append({"method": "ST5-TIGER", "seed": seed, **metrics})
        checkpoint = load_checkpoint(trial["checkpoint"])
        history = checkpoint.get("history", [])
        best_step = max(history, key=lambda x: x["validation"]["ndcg@10"])["step"] if history else None
        runs.append({"seed": seed, "checkpoint": trial["checkpoint"],
                     "best_step": best_step, "validation": trial["validation"],
                     "seconds": trial["seconds"]})
        predictions[str(seed)] = {str(mid): values for mid, values in ranking.items()}
    keys = [f"{metric}@{k}" for k in args.cutoffs
            for metric in ("precision", "recall", "ndcg", "coverage", "tail_recall")]
    summary = summarize(rows, keys)
    with (output_dir / "per_seed_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "seed", *keys,
                                                     "queries", "tail_queries"], extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    metadata = {"split_seed": args.split_seed, "seeds": args.seeds, "params": params,
                "best_steps": [r["best_step"] for r in runs], "runs": runs,
                "rqvae_levels": 4, "sentence_encoder": "sentence-t5-base",
                "method_scope": "generator only; no retrieval augmentation"}
    result = {"metadata": metadata, "metrics": summary}
    (output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "test_top20_predictions.json").write_text(json.dumps(predictions), encoding="utf-8")
    write_report(output_dir / "REPORT.md", summary, metadata, args.cutoffs, args.seeds)
    print(f"ST5-TIGER experiment complete: {output_dir}")
    return result


def parser():
    parser = argparse.ArgumentParser(description="Run ST5-TIGER baseline")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43, 71, 101])
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[5, 10, 15, 20])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--generator-lr", type=float, default=0.001)
    parser.add_argument("--generator-memory-tokens", type=int, default=8)
    parser.add_argument("--generator-dropout", type=float, default=0.1)
    parser.add_argument("--generator-steps", type=int, default=4000)
    parser.add_argument("--generator-min-steps", type=int, default=2000)
    parser.add_argument("--generator-eval-every", type=int, default=500)
    parser.add_argument("--generator-batch", type=int, default=64)
    parser.add_argument("--eval-batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--score-query-batch", type=int, default=8)
    parser.add_argument("--score-api-batch", type=int, default=192)
    return parser


if __name__ == "__main__":
    values = parser().parse_args()
    torch.set_num_threads(4)
    if values.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    run(values)
