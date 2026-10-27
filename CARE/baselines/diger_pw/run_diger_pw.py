"""Run DIGER adapted to unordered Mashup-to-API recommendation (DIGER-PW)."""
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

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from care.io_utils import load_checkpoint, save_checkpoint, seed_all
from care.metrics import catalog_coverage, ranking_metrics
from care.rqvae import RQVAE, RQVAEConfig
from care.semantic_ids import make_unique_codes
from models import SIDVocabulary, SemanticGenerator
from pipeline import Data, generator_scores, rankings
from diger_model import DifferentiableResidualTokenizer, differentiable_generation_loss


def trusted_load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def current_codes(tokenizer: DifferentiableResidualTokenizer, data: Data,
                  device: str) -> dict[int, list[int]]:
    tokenizer.eval()
    with torch.no_grad():
        base = tokenizer.sinkhorn_codes(data.api_vectors.to(device)).cpu().tolist()
    return make_unique_codes(data.api_ids, base)


def vocabulary_for(codes: dict[int, list[int]], state: dict) -> SIDVocabulary:
    vocab = SIDVocabulary(codes, state=state)
    for path in vocab.paths.values():
        for position, token in enumerate(path):
            start = state["starts"][position]
            if not start <= token < start + state["sizes"][position]:
                raise RuntimeError("Dynamic SID exceeded the shared positional vocabulary")
    return vocab


def dynamic_vocab_state(data: Data, levels: int, codebook_size: int) -> dict:
    """Reserve a full-catalog suffix range because differentiable SIDs can drift/collide."""
    sizes = [int(codebook_size)] * int(levels) + [len(data.api_ids)]
    starts, cursor = [], 2
    for size in sizes:
        starts.append(cursor)
        cursor += size
    return {"starts": starts, "sizes": sizes, "size": cursor}


def validation_score(generator, tokenizer, data: Data, vocab_state: dict,
                     args) -> tuple[dict, dict]:
    codes = current_codes(tokenizer, data, args.device)
    vocab = vocabulary_for(codes, vocab_state)
    mids = data.splits["valid"]
    query = torch.stack([data.query[mid] for mid in mids])
    score = generator_scores(generator, vocab, query, data.api_ids, args.device,
                             args.score_query_batch, args.score_api_batch)
    prediction = rankings(score, mids, data.api_ids, k=max(args.cutoffs))
    return ranking_metrics(prediction, data.truth("valid"), args.cutoffs), codes


def train_seed(seed: int, data: Data, output_dir: Path, args) -> dict:
    checkpoint_path = output_dir / "models" / "checkpoints" / f"diger_pw_seed{seed}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_path.exists():
        payload = load_checkpoint(checkpoint_path)
        return {"checkpoint": str(checkpoint_path), "validation": payload["validation"],
                "seconds": payload.get("seconds", 0.0), "best_step": payload.get("best_step")}

    seed_all(seed)
    started = time.time()
    rq_pack = load_checkpoint(Path(args.workspace) / "rqvae_best.pt")
    rqvae = RQVAE(RQVAEConfig.from_dict(rq_pack["config"]))
    rqvae.load_state_dict(rq_pack["model"])
    tokenizer = DifferentiableResidualTokenizer(
        rqvae, temperature=args.gumbel_temperature,
        frequency_ratio=args.frequency_ratio, usage_ema=args.usage_ema,
    ).to(args.device)
    tokenizer.initialize_usage(data.api_vectors.to(args.device))
    vocab_state = dynamic_vocab_state(data, tokenizer.levels, tokenizer.codebook_size)
    generator = SemanticGenerator(
        vocab_state["size"], memory_tokens=args.memory_tokens,
        dropout_rate=args.dropout,
    ).to(args.device)
    optimizer = torch.optim.AdamW([
        {"params": generator.parameters(), "lr": args.generator_lr},
        {"params": tokenizer.parameters(), "lr": args.tokenizer_lr},
    ], weight_decay=0.01)

    edges = [(mid, api) for mid in data.splits["train"] for api in data.links[mid]]
    refresh_every = (args.sid_refresh_every if args.sid_refresh_every > 0
                     else max(1, math.ceil(len(edges) / args.batch_size)))
    rng = np.random.default_rng(seed + 202)
    # Corrected DIGER uses cached pretrained hard IDs in the forward path and
    # current soft assignments in the backward path. The final deterministic
    # IDs are recomputed for validation/test with Gumbel noise disabled.
    initial_codes = current_codes(tokenizer, data, args.device)
    cached_prefix = {api: initial_codes[api][:-1] for api in data.api_ids}
    cached_suffix = {api: initial_codes[api][-1] for api in data.api_ids}
    best_metric, best_payload, stale, history = -1.0, None, 0, []
    generator.train(); tokenizer.train()
    for step in range(1, args.training_steps + 1):
        selected = rng.integers(len(edges), size=args.batch_size)
        batch = [edges[index] for index in selected]
        query = torch.stack([data.query[mid] for mid, _ in batch]).to(args.device)
        apis = [api for _, api in batch]
        api_indices = torch.tensor([data.ai[api] for api in apis], device=args.device)
        api_vectors = data.api_vectors[api_indices.cpu()].to(args.device)
        hard_prefix = torch.tensor([cached_prefix[api] for api in apis], device=args.device)
        suffix = torch.tensor([cached_suffix[api] for api in apis], device=args.device)

        generator.train(); tokenizer.train()
        optimizer.zero_grad(set_to_none=True)
        assignment = tokenizer(api_vectors, add_noise=True, cached_hard_codes=hard_prefix)
        generation = differentiable_generation_loss(
            generator, query, assignment.probabilities, suffix,
            vocab_state["starts"],
        )
        total = (generation
                 + args.reconstruction_weight * assignment.reconstruction_loss
                 + args.quantization_weight * assignment.quantization_loss)
        total.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), 1.0)
        optimizer.step()
        if step % refresh_every == 0:
            refreshed = current_codes(tokenizer, data, args.device)
            cached_prefix = {api: refreshed[api][:-1] for api in data.api_ids}
            cached_suffix = {api: refreshed[api][-1] for api in data.api_ids}

        if step % args.eval_every == 0 or step == args.training_steps:
            validation, codes = validation_score(generator, tokenizer, data, vocab_state, args)
            # DIGER refreshes the cached full-catalog Sinkhorn map after each
            # training epoch; our fixed-step interval is the PW epoch analogue.
            cached_prefix = {api: codes[api][:-1] for api in data.api_ids}
            cached_suffix = {api: codes[api][-1] for api in data.api_ids}
            hard = np.asarray([codes[api][:-1] for api in data.api_ids])
            row = {
                "step": step,
                "loss": float(total.detach()),
                "generation": float(generation.detach()),
                "reconstruction": float(assignment.reconstruction_loss.detach()),
                "quantization": float(assignment.quantization_loss.detach()),
                "validation": validation,
                "unique_codes_per_level": [int(len(np.unique(hard[:, i]))) for i in range(hard.shape[1])],
                "unique_sid_prefixes": int(len({tuple(value) for value in hard.tolist()})),
            }
            history.append(row)
            print("DIGER-PW", json.dumps({"seed": seed, **row}), flush=True)
            metric = validation["ndcg@10"]
            if metric > best_metric:
                best_metric, stale = metric, 0
                best_payload = {
                    "generator": {k: v.detach().cpu() for k, v in generator.state_dict().items()},
                    "tokenizer": {k: v.detach().cpu() for k, v in tokenizer.state_dict().items()},
                    "codes": codes,
                    "vocab_state": vocab_state,
                    "validation": validation,
                    "best_step": step,
                }
            else:
                stale += 1
            if step >= args.min_steps and stale >= args.patience:
                break

    seconds = time.time() - started
    if best_payload is None:
        raise RuntimeError("Training completed without a validation checkpoint")
    best_payload.update({
        "seed": seed, "seconds": seconds, "history": history,
        "method": "DIGER-PW (FrqUD, corrected hard-forward/soft-backward)",
        "official_reference_branch": "gradient-fix",
        "adaptation": "Mashup Sentence-T5 condition; unordered positive API targets",
    })
    save_checkpoint(checkpoint_path, best_payload)
    return {"checkpoint": str(checkpoint_path), "validation": best_payload["validation"],
            "seconds": seconds, "best_step": best_payload["best_step"]}


@torch.no_grad()
def score_seed(seed: int, trial: dict, data: Data, output_dir: Path, args) -> dict:
    cache = output_dir / "score_cache" / f"test_seed{seed}.pt"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        return trusted_load(cache)
    checkpoint = load_checkpoint(trial["checkpoint"])
    vocab_state = checkpoint["vocab_state"]
    generator = SemanticGenerator(
        vocab_state["size"], memory_tokens=args.memory_tokens,
        dropout_rate=args.dropout,
    ).to(args.device)
    generator.load_state_dict(checkpoint["generator"])
    vocab = vocabulary_for({int(k): v for k, v in checkpoint["codes"].items()}, vocab_state)
    mids = data.splits["test"]
    query = torch.stack([data.query[mid] for mid in mids])
    payload = {"mids": mids, "generated": generator_scores(
        generator, vocab, query, data.api_ids, args.device,
        args.score_query_batch, args.score_api_batch,
    )}
    torch.save(payload, cache)
    print(f"SCORED DIGER-PW seed={seed}", flush=True)
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
    result = {}
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows])
        result[key] = {"mean": float(values.mean()),
                       "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0}
    return result


def write_report(path: Path, summary: dict, metadata: dict,
                 cutoffs: Sequence[int], seeds: Sequence[int]) -> None:
    lines = [
        "# DIGER-PW Baseline Report", "",
        f"- Split seed: `{metadata['split_seed']}`",
        f"- Training seeds: `{', '.join(map(str, seeds))}`",
        "- Method: corrected hard-forward/soft-backward differentiable SID with FrqUD.",
        "- PW adaptation: Mashup description conditions generation; each called API is an unordered positive target.",
        "- Pure generative baseline: no CARE catalog loss, retriever, alignment loss, or score fusion.",
        f"- Tokenizer: RQ levels=`{metadata['tokenizer']['levels']}`, codebook size=`{metadata['tokenizer']['codebook_size']}`, collision-suffix capacity=`{metadata['tokenizer']['suffix_capacity']}`.",
        f"- Full-catalog Sinkhorn SID refresh interval: `{metadata['tokenizer']['sid_refresh_steps']}` optimizer steps.",
        f"- Best steps selected by validation NDCG@10: `{metadata['best_steps']}`.", "",
        "| K | Precision | Recall | NDCG | Coverage | Tail Recall |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for k in cutoffs:
        cells = []
        for metric in ("precision", "recall", "ndcg", "coverage", "tail_recall"):
            value = summary[f"{metric}@{k}"]
            cells.append(
                f"{100 * value['mean']:.2f}% ± {100 * value['std']:.2f}%"
                if metric == "coverage"
                else f"{value['mean']:.6f} ± {value['std']:.6f}"
            )
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    lines += ["", "## Reproducibility note", "",
              "This is a method-level adaptation of DIGER to an unordered multi-positive service-recommendation task, not an unchanged execution of its sequential user-history code.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args) -> dict:
    workspace = Path(args.workspace)
    required = ["prepared.json", "api_embeddings.pt", "content_semantic_ids.json", "rqvae_best.pt"]
    missing = [name for name in required if not (workspace / name).exists()]
    if missing:
        raise FileNotFoundError(f"Workspace is missing: {missing}. Set CARE_WORKSPACE_DIR.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = Data(SimpleNamespace(output_dir=str(workspace)), "strict")
    rows, runs, predictions = [], [], {}
    for seed in args.seeds:
        trial = train_seed(seed, data, output_dir, args)
        payload = score_seed(seed, trial, data, output_dir, args)
        metrics, ranking = evaluate(payload["generated"], payload["mids"], data, args.cutoffs)
        rows.append({"method": "DIGER-PW", "seed": seed, **metrics})
        runs.append(trial)
        predictions[str(seed)] = {str(mid): values for mid, values in ranking.items()}

    keys = [f"{metric}@{k}" for k in args.cutoffs
            for metric in ("precision", "recall", "ndcg", "coverage", "tail_recall")]
    summary = summarize(rows, keys)
    with (output_dir / "per_seed_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "seed", *keys,
                                                     "queries", "tail_queries"], extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    rq_config = load_checkpoint(workspace / "rqvae_best.pt")["config"]
    training_edges = sum(len(data.links[mid]) for mid in data.splits["train"])
    refresh_steps = (args.sid_refresh_every if args.sid_refresh_every > 0
                     else max(1, math.ceil(training_edges / args.batch_size)))
    metadata = {
        "split_seed": args.split_seed, "seeds": args.seeds,
        "best_steps": [row["best_step"] for row in runs], "runs": runs,
        "official_reference_branch": "gradient-fix",
        "tokenizer": {
            "levels": int(rq_config["num_levels"]),
            "codebook_size": int(rq_config["codebook_size"]),
            "suffix_capacity": len(data.api_ids),
            "sid_refresh_steps": refresh_steps,
            "sinkhorn_epsilon": 0.003,
            "sinkhorn_iterations": 50,
        },
        "settings": {key: value for key, value in vars(args).items()
                     if key not in {"workspace", "output_dir"}},
    }
    result = {"metadata": metadata, "metrics": summary}
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "test_top20_predictions.json").write_text(
        json.dumps(predictions), encoding="utf-8")
    write_report(output_dir / "REPORT.md", summary, metadata, args.cutoffs, args.seeds)
    print(f"DIGER-PW experiment complete: {output_dir}", flush=True)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Run DIGER-PW baseline")
    value.add_argument("--workspace", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--split-seed", type=int, default=2026)
    value.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43, 71, 101])
    value.add_argument("--cutoffs", type=int, nargs="+", default=[5, 10, 15, 20])
    value.add_argument("--device", default="cuda")
    value.add_argument("--generator-lr", type=float, default=1e-3)
    value.add_argument("--tokenizer-lr", type=float, default=1e-5)
    value.add_argument("--memory-tokens", type=int, default=8)
    value.add_argument("--dropout", type=float, default=0.1)
    value.add_argument("--gumbel-temperature", type=float, default=2.0)
    value.add_argument("--frequency-ratio", type=float, default=1.5)
    value.add_argument("--usage-ema", type=float, default=0.99)
    value.add_argument("--reconstruction-weight", type=float, default=768.0)
    value.add_argument("--quantization-weight", type=float, default=32.0)
    value.add_argument("--training-steps", type=int, default=4000)
    value.add_argument("--min-steps", type=int, default=2000)
    value.add_argument("--eval-every", type=int, default=500)
    value.add_argument("--sid-refresh-every", type=int, default=0)
    value.add_argument("--batch-size", type=int, default=64)
    value.add_argument("--patience", type=int, default=2)
    value.add_argument("--score-query-batch", type=int, default=8)
    value.add_argument("--score-api-batch", type=int, default=192)
    return value


if __name__ == "__main__":
    arguments = parser().parse_args()
    torch.set_num_threads(4)
    if arguments.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    run(arguments)
