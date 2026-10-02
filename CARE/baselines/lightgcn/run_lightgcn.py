"""Train and evaluate LightGCN on the ProgrammableWeb Mashup-API graph."""
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


class LightGCN(nn.Module):
    def __init__(self, users: int, items: int, dim: int, layers: int):
        super().__init__()
        self.users = users
        self.items = items
        self.layers = layers
        self.user_embedding = nn.Embedding(users, dim)
        self.item_embedding = nn.Embedding(items, dim)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    def propagate(self, adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        layers = [values]
        for _ in range(self.layers):
            values = torch.sparse.mm(adjacency, values)
            layers.append(values)
        output = torch.stack(layers, dim=0).mean(dim=0)
        return output[:self.users], output[self.users:]


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_workspace(path: Path, split_seed: int) -> dict:
    source = path / "prepared.json"
    if not source.exists():
        raise FileNotFoundError(f"Missing final experiment workspace: {source}")
    prepared = json.loads(source.read_text(encoding="utf-8"))
    stored_seed = int(prepared.get("split_seed", split_seed))
    if stored_seed != split_seed:
        raise ValueError(f"Workspace split seed is {stored_seed}, expected {split_seed}")
    return prepared


def build_graph(prepared: dict, device: torch.device) -> tuple[torch.Tensor, dict]:
    splits = {name: [int(value) for value in values]
              for name, values in prepared["splits"].items()}
    links = {int(mid): [int(api) for api in values]
             for mid, values in prepared["links"].items()}
    train_mids = splits["train"]
    api_ids = sorted({api for values in links.values() for api in values})
    user_index = {mid: index for index, mid in enumerate(train_mids)}
    api_index = {api: index for index, api in enumerate(api_ids)}
    pairs = [(user_index[mid], api_index[api]) for mid in train_mids for api in links[mid]]
    users = np.asarray([user for user, _ in pairs], dtype=np.int64)
    items = np.asarray([item for _, item in pairs], dtype=np.int64)

    node_count = len(train_mids) + len(api_ids)
    item_nodes = items + len(train_mids)
    row = np.concatenate([users, item_nodes])
    col = np.concatenate([item_nodes, users])
    degree = np.bincount(row, minlength=node_count).astype(np.float32)
    weight = 1.0 / np.sqrt(np.maximum(degree[row], 1.0) * np.maximum(degree[col], 1.0))
    indices = torch.tensor(np.stack([row, col]), dtype=torch.long, device=device)
    values = torch.tensor(weight, dtype=torch.float32, device=device)
    adjacency = torch.sparse_coo_tensor(indices, values, (node_count, node_count), device=device).coalesce()
    positive_sets = [set() for _ in train_mids]
    for user, item in pairs:
        positive_sets[user].add(item)
    return adjacency, {
        "train_mids": train_mids, "api_ids": api_ids, "users": users,
        "items": items, "positive_sets": positive_sets, "links": links,
        "splits": splits, "pairs": pairs,
    }


def sample_negatives(users: np.ndarray, positive_sets: list[set[int]],
                     item_count: int, rng: np.random.Generator) -> np.ndarray:
    values = rng.integers(0, item_count, size=len(users), dtype=np.int64)
    invalid = np.fromiter((int(item) in positive_sets[int(user)]
                           for user, item in zip(users, values)), bool, len(users))
    while invalid.any():
        values[invalid] = rng.integers(0, item_count, size=int(invalid.sum()), dtype=np.int64)
        invalid = np.fromiter((int(item) in positive_sets[int(user)]
                               for user, item in zip(users, values)), bool, len(users))
    return values


def rank_cold_start(user_vectors: torch.Tensor, item_vectors: torch.Tensor,
                    api_ids: Sequence[int], k: int) -> list[int]:
    scores = item_vectors @ user_vectors.mean(dim=0)
    scores = scores.detach().cpu().numpy()
    api_array = np.asarray(api_ids, dtype=np.int64)
    order = np.lexsort((api_array, -scores))[:k]
    return api_array[order].astype(int).tolist()


def truth_for(graph: dict, split: str) -> dict[int, list[int]]:
    return {mid: graph["links"][mid] for mid in graph["splits"][split]}


def train_one(prepared: dict, seed: int, config: dict,
              device_name: str) -> tuple[LightGCN, dict]:
    seed_everything(seed)
    device = torch.device(device_name if device_name.startswith("cuda") and torch.cuda.is_available() else "cpu")
    adjacency, graph = build_graph(prepared, device)
    model = LightGCN(len(graph["train_mids"]), len(graph["api_ids"]),
                     config["dim"], config["layers"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    rng = np.random.default_rng(seed)
    validation_truth = truth_for(graph, "valid")
    users_np, positives_np = graph["users"], graph["items"]
    users = torch.from_numpy(users_np).to(device)
    positives = torch.from_numpy(positives_np).to(device)
    best_ndcg, best_epoch, best_state = -1.0, 0, None
    history, stale = [], 0

    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        negative_np = sample_negatives(users_np, graph["positive_sets"], len(graph["api_ids"]), rng)
        negatives = torch.from_numpy(negative_np).to(device)
        user_final, item_final = model.propagate(adjacency)
        pos_score = (user_final[users] * item_final[positives]).sum(dim=1)
        neg_score = (user_final[users] * item_final[negatives]).sum(dim=1)
        ranking_loss = -F.logsigmoid(pos_score - neg_score).mean()
        regularization = (
            model.user_embedding(users).pow(2).sum()
            + model.item_embedding(positives).pow(2).sum()
            + model.item_embedding(negatives).pow(2).sum()
        ) / len(users)
        loss = ranking_loss + config["weight_decay"] * regularization
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()

        if epoch % config["eval_every"] == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                user_final, item_final = model.propagate(adjacency)
                ranking = rank_cold_start(user_final, item_final, graph["api_ids"], 20)
            predictions = {mid: ranking for mid in validation_truth}
            validation = ranking_metrics(predictions, validation_truth, (10,))
            ndcg = float(validation["ndcg@10"])
            row = {"epoch": epoch, "loss": float(loss.detach()),
                   "validation_ndcg@10": ndcg}
            history.append(row)
            print(f"LightGCN seed={seed} epoch={epoch} loss={row['loss']:.6f} val_ndcg@10={ndcg:.6f}", flush=True)
            if ndcg > best_ndcg + 1e-8:
                best_ndcg, best_epoch = ndcg, epoch
                best_state = {key: value.detach().cpu().clone()
                              for key, value in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if epoch >= 30 and stale >= config["patience"]:
                break

    model.load_state_dict(best_state)
    metadata = {
        **config, "device": str(device), "users": len(graph["train_mids"]),
        "items": len(graph["api_ids"]), "training_edges": len(graph["pairs"]),
        "best_epoch": best_epoch, "best_validation_ndcg@10": best_ndcg,
        "history": history,
    }
    return model.cpu(), {"graph": graph, "metadata": metadata}


def evaluate_model(model: LightGCN, model_data: dict, prepared: dict,
                   cutoffs: Sequence[int]) -> tuple[dict, list[int]]:
    adjacency, graph = build_graph(prepared, torch.device("cpu"))
    model.eval()
    with torch.no_grad():
        user_final, item_final = model.propagate(adjacency)
        ranking = rank_cold_start(user_final, item_final, graph["api_ids"], max(cutoffs))
    truth = truth_for(graph, "test")
    predictions = {mid: list(ranking) for mid in truth}
    metrics = ranking_metrics(predictions, truth, cutoffs)
    metrics.update(catalog_coverage(predictions, len(graph["api_ids"]), cutoffs))

    counts = Counter(api for mid in graph["splits"]["train"] for api in graph["links"][mid])
    seen = sorted((api for api in graph["api_ids"] if counts.get(api, 0) > 0),
                  key=lambda api: (-counts[api], api))
    tail = set(seen[math.ceil(0.2 * len(seen)):])
    tail_truth = {mid: [api for api in targets if api in tail]
                  for mid, targets in truth.items()}
    tail_truth = {mid: targets for mid, targets in tail_truth.items() if targets}
    tail_metrics = ranking_metrics(
        {mid: predictions[mid] for mid in tail_truth}, tail_truth, cutoffs)
    for k in cutoffs:
        metrics[f"tail_recall@{k}"] = tail_metrics[f"recall@{k}"]
    metrics["tail_queries"] = len(tail_truth)
    return metrics, ranking


def summarize(rows: list[dict], keys: Sequence[str]) -> dict:
    output = {}
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows])
        output[key] = {"mean": float(values.mean()),
                       "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0}
    return output


def write_report(path: Path, summary: dict, metadata: dict,
                 cutoffs: Sequence[int], seeds: Sequence[int]) -> None:
    best_epochs = [row["best_epoch"] for row in metadata["runs"]]
    lines = [
        "# LightGCN Baseline Report", "",
        f"- Split seed: `{metadata['split_seed']}`",
        f"- Training seeds: `{', '.join(map(str, seeds))}`",
        f"- Embedding dimension=`{metadata['dim']}`, propagation layers=`{metadata['layers']}`.",
        f"- lr=`{metadata['lr']}`, L2=`{metadata['weight_decay']}`.",
        f"- Best epochs selected by validation NDCG@10: `{best_epochs}`.",
        "- Cold-start protocol: unseen validation/test Mashups use the mean propagated training-Mashup embedding.",
        "- No validation/test invocation edge is added to the graph.", "",
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
              "LightGCN captures high-order collaborative structure among training Mashups and APIs. "
              "Under the official Mashup-level cold-start split, however, all test Mashups share one "
              "fallback query representation and cannot receive personalized graph recommendations.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(workspace: Path, output_dir: Path, split_seed: int, seeds: Sequence[int],
        cutoffs: Sequence[int], config: dict, device: str) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"; checkpoint_dir.mkdir(exist_ok=True)
    prepared = load_workspace(workspace, split_seed)
    rows, rankings, run_metadata = [], {}, []
    for seed in seeds:
        model, model_data = train_one(prepared, int(seed), config, device)
        metrics, ranking = evaluate_model(model, model_data, prepared, cutoffs)
        rows.append({"method": "LightGCN", "seed": int(seed), **metrics})
        rankings[str(seed)] = ranking
        run_metadata.append(model_data["metadata"])
        torch.save({"model": model.state_dict(), "api_ids": model_data["graph"]["api_ids"],
                    "seed": int(seed), "metadata": model_data["metadata"]},
                   checkpoint_dir / f"lightgcn_seed{seed}.pt")

    keys = [f"{metric}@{k}" for k in cutoffs
            for metric in ("precision", "recall", "ndcg", "coverage", "tail_recall")]
    summary = summarize(rows, keys)
    with (output_dir / "per_seed_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "seed", *keys, "queries", "tail_queries"],
                                extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    metadata = {"method": "LightGCN", "split_seed": split_seed,
                "cold_start_strategy": "mean propagated training-Mashup embedding",
                **config, "runs": run_metadata}
    payload = {"metadata": metadata, "seeds": list(map(int, seeds)), "metrics": summary}
    (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "cold_start_rankings.json").write_text(json.dumps(rankings, indent=2), encoding="utf-8")
    write_report(output_dir / "REPORT.md", summary, metadata, cutoffs, seeds)
    print(f"LightGCN experiment complete: {output_dir}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LightGCN baseline")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43, 71, 101])
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    params = {"dim": 64, "layers": 3, "lr": 0.001, "weight_decay": 1e-4,
              "max_epochs": 300, "eval_every": 10, "patience": 8}
    run(args.workspace, args.output_dir, args.split_seed, args.seeds,
        (5, 10, 15, 20), params, args.device)

