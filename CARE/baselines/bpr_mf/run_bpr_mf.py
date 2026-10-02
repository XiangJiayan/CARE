"""Train and evaluate BPR matrix factorization on the Mashup-API graph."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from care.metrics import catalog_coverage, ranking_metrics


class BPRMF(nn.Module):
    def __init__(self, users: int, items: int, dim: int):
        super().__init__()
        self.user_embedding = nn.Embedding(users, dim)
        self.item_embedding = nn.Embedding(items, dim)
        self.item_bias = nn.Embedding(items, 1)
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.zeros_(self.item_bias.weight)

    def score(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return (self.user_embedding(users) * self.item_embedding(items)).sum(-1) + self.item_bias(items).squeeze(-1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_workspace(path: Path, split_seed: int) -> dict:
    prepared_path = path / "prepared.json"
    if not prepared_path.exists():
        raise FileNotFoundError(f"Missing final experiment workspace: {prepared_path}")
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    stored_seed = int(prepared.get("split_seed", split_seed))
    if stored_seed != split_seed:
        raise ValueError(f"Workspace split seed is {stored_seed}, expected {split_seed}")
    return prepared


def sample_negatives(users: np.ndarray, positive_sets: list[set[int]],
                     item_count: int, rng: np.random.Generator) -> np.ndarray:
    negatives = rng.integers(0, item_count, size=len(users), dtype=np.int64)
    invalid = np.fromiter((int(item) in positive_sets[int(user)]
                           for user, item in zip(users, negatives)),
                          dtype=bool, count=len(users))
    while invalid.any():
        negatives[invalid] = rng.integers(0, item_count, size=int(invalid.sum()), dtype=np.int64)
        invalid = np.fromiter((int(item) in positive_sets[int(user)]
                               for user, item in zip(users, negatives)),
                              dtype=bool, count=len(users))
    return negatives


def train_one(prepared: dict, seed: int, dim: int, lr: float, reg: float,
              epochs: int, batch_size: int, device_name: str) -> tuple[BPRMF, dict, dict]:
    seed_everything(seed)
    splits = {name: [int(value) for value in values]
              for name, values in prepared["splits"].items()}
    links = {int(mid): [int(api) for api in values]
             for mid, values in prepared["links"].items()}
    api_ids = sorted({api for values in links.values() for api in values})
    api_index = {api: index for index, api in enumerate(api_ids)}
    train_mashups = splits["train"]
    user_index = {mid: index for index, mid in enumerate(train_mashups)}

    pairs = [(user_index[mid], api_index[api])
             for mid in train_mashups for api in links[mid]]
    users = np.asarray([pair[0] for pair in pairs], dtype=np.int64)
    positives = np.asarray([pair[1] for pair in pairs], dtype=np.int64)
    positive_sets = [set() for _ in train_mashups]
    for user, item in pairs:
        positive_sets[user].add(item)

    device = torch.device(device_name if device_name.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model = BPRMF(len(train_mashups), len(api_ids), dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    history = []

    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(users))
        epoch_loss = 0.0
        examples = 0
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            batch_users_np = users[indices]
            batch_pos_np = positives[indices]
            batch_neg_np = sample_negatives(batch_users_np, positive_sets, len(api_ids), rng)
            batch_users = torch.from_numpy(batch_users_np).to(device)
            batch_pos = torch.from_numpy(batch_pos_np).to(device)
            batch_neg = torch.from_numpy(batch_neg_np).to(device)

            pos_score = model.score(batch_users, batch_pos)
            neg_score = model.score(batch_users, batch_neg)
            ranking_loss = -F.logsigmoid(pos_score - neg_score).mean()
            regularization = (
                model.user_embedding(batch_users).pow(2).sum()
                + model.item_embedding(batch_pos).pow(2).sum()
                + model.item_embedding(batch_neg).pow(2).sum()
            ) / len(batch_users)
            loss = ranking_loss + reg * regularization
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(indices)
            examples += len(indices)
        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            row = {"epoch": epoch, "loss": epoch_loss / max(examples, 1)}
            history.append(row)
            print(f"BPR-MF seed={seed} epoch={epoch} loss={row['loss']:.6f}", flush=True)

    metadata = {
        "device": str(device), "users": len(train_mashups), "items": len(api_ids),
        "training_edges": len(pairs), "embedding_dim": dim, "learning_rate": lr,
        "weight_decay": reg, "epochs": epochs, "batch_size": batch_size,
    }
    return model.cpu(), {"api_ids": api_ids, "history": history}, metadata


def evaluate_model(model: BPRMF, model_data: dict, prepared: dict,
                   cutoffs: Sequence[int]) -> tuple[dict, dict]:
    splits = {name: [int(value) for value in values]
              for name, values in prepared["splits"].items()}
    links = {int(mid): [int(api) for api in values]
             for mid, values in prepared["links"].items()}
    api_ids = model_data["api_ids"]

    # Test Mashups are unseen. The mean learned training-Mashup factor is the
    # fixed, content-free cold-start representation; no test edge is consumed.
    with torch.no_grad():
        cold_user = model.user_embedding.weight.mean(0)
        scores = model.item_embedding.weight @ cold_user + model.item_bias.weight.squeeze(-1)
    api_array = np.asarray(api_ids, dtype=np.int64)
    score_array = scores.numpy()
    order = np.lexsort((api_array, -score_array))[:max(cutoffs)]
    ranking = api_array[order].astype(int).tolist()
    predictions = {mid: list(ranking) for mid in splits["test"]}
    truth = {mid: links[mid] for mid in splits["test"]}

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
    tail_metric = ranking_metrics(
        {mid: predictions[mid] for mid in tail_truth}, tail_truth, cutoffs)
    for k in cutoffs:
        metrics[f"tail_recall@{k}"] = tail_metric[f"recall@{k}"]
    metrics["tail_queries"] = len(tail_truth)
    return metrics, {"predictions": predictions, "cold_start_ranking": ranking}


def summarize(rows: list[dict], metric_keys: Sequence[str]) -> dict:
    result = {}
    for key in metric_keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        result[key] = {"mean": float(values.mean()),
                       "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0}
    return result


def write_report(path: Path, summary: dict, metadata: dict,
                 cutoffs: Sequence[int], seeds: Sequence[int]) -> None:
    lines = [
        "# BPR-MF Baseline Report", "",
        f"- Split seed: `{metadata['split_seed']}`",
        f"- Training seeds: `{', '.join(map(str, seeds))}`",
        f"- Parameters: dimension=`{metadata['embedding_dim']}`, lr=`{metadata['learning_rate']}`, "
        f"L2=`{metadata['weight_decay']}`, epochs=`{metadata['epochs']}`.",
        f"- Training graph: `{metadata['users']}` Mashups, `{metadata['items']}` APIs, "
        f"`{metadata['training_edges']}` positive edges.",
        "- Cold-start protocol: every unseen test Mashup uses the mean training-user factor.",
        "- No validation/test API invocation is provided to the model.", "",
        "| K | Precision | Recall | NDCG | Coverage | Tail Recall |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for k in cutoffs:
        cells = []
        for metric in ("precision", "recall", "ndcg", "coverage", "tail_recall"):
            value = summary[f"{metric}@{k}"]
            cells.append(f"{value['mean']:.6f} ± {value['std']:.6f}")
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    lines += ["", "## Interpretation", "",
              "BPR-MF learns collaborative latent factors from the training Mashup-API graph. "
              "Because the official split contains completely unseen test Mashups, it must use "
              "one shared cold-start user factor and therefore cannot personalize test rankings.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(workspace: Path, output_dir: Path, split_seed: int, seeds: Sequence[int],
        cutoffs: Sequence[int], dim: int, lr: float, reg: float, epochs: int,
        batch_size: int, device: str) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    prepared = load_workspace(workspace, split_seed)
    rows, rankings, train_meta = [], {}, None
    for seed in seeds:
        model, model_data, metadata = train_one(
            prepared, int(seed), dim, lr, reg, epochs, batch_size, device)
        metrics, details = evaluate_model(model, model_data, prepared, cutoffs)
        rows.append({"method": "BPR-MF", "seed": int(seed), **metrics})
        rankings[str(seed)] = details["cold_start_ranking"]
        train_meta = metadata
        torch.save({"model": model.state_dict(), "api_ids": model_data["api_ids"],
                    "history": model_data["history"], "seed": int(seed),
                    "metadata": metadata}, checkpoint_dir / f"bpr_mf_seed{seed}.pt")

    metric_keys = [f"{metric}@{k}" for k in cutoffs
                   for metric in ("precision", "recall", "ndcg", "coverage", "tail_recall")]
    summary = summarize(rows, metric_keys)
    fields = ["method", "seed", *metric_keys, "queries", "tail_queries"]
    with (output_dir / "per_seed_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)

    metadata = {"method": "BPR-MF", "split_seed": split_seed,
                "cold_start_strategy": "mean learned training-user factor", **train_meta}
    payload = {"metadata": metadata, "seeds": list(map(int, seeds)), "metrics": summary}
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "cold_start_rankings.json").write_text(
        json.dumps(rankings, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(output_dir / "REPORT.md", summary, metadata, cutoffs, seeds)
    print(f"BPR-MF experiment complete: {output_dir}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BPR-MF baseline")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43, 71, 101])
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[5, 10, 15, 20])
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--reg", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.workspace, args.output_dir, args.split_seed, args.seeds, args.cutoffs,
        args.dim, args.lr, args.reg, args.epochs, args.batch_size, args.device)

