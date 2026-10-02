"""Train and evaluate Neural Matrix Factorization on Mashup-API feedback."""
from __future__ import annotations

import argparse
import copy
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

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from care.metrics import catalog_coverage, ranking_metrics


class NeuMF(nn.Module):
    def __init__(self, users: int, items: int, gmf_dim: int, mlp_dim: int,
                 layers: Sequence[int], dropout: float):
        super().__init__()
        self.gmf_user = nn.Embedding(users, gmf_dim)
        self.gmf_item = nn.Embedding(items, gmf_dim)
        self.mlp_user = nn.Embedding(users, mlp_dim)
        self.mlp_item = nn.Embedding(items, mlp_dim)
        modules: list[nn.Module] = []
        input_dim = 2 * mlp_dim
        for output_dim in layers:
            modules.extend([nn.Linear(input_dim, output_dim), nn.ReLU()])
            if dropout > 0:
                modules.append(nn.Dropout(dropout))
            input_dim = output_dim
        self.mlp = nn.Sequential(*modules)
        self.output = nn.Linear(gmf_dim + input_dim, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for embedding in (self.gmf_user, self.gmf_item, self.mlp_user, self.mlp_item):
            nn.init.normal_(embedding.weight, std=0.01)
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        gmf = self.gmf_user(users) * self.gmf_item(items)
        mlp = self.mlp(torch.cat([self.mlp_user(users), self.mlp_item(items)], dim=-1))
        return self.output(torch.cat([gmf, mlp], dim=-1)).squeeze(-1)

    @torch.no_grad()
    def cold_start_scores(self, device: torch.device, batch_size: int = 4096) -> torch.Tensor:
        self.eval()
        mean_gmf = self.gmf_user.weight.mean(0, keepdim=True).to(device)
        mean_mlp = self.mlp_user.weight.mean(0, keepdim=True).to(device)
        results = []
        for start in range(0, self.gmf_item.num_embeddings, batch_size):
            end = min(start + batch_size, self.gmf_item.num_embeddings)
            gmf_item = self.gmf_item.weight[start:end].to(device)
            mlp_item = self.mlp_item.weight[start:end].to(device)
            gmf = mean_gmf.expand(end - start, -1) * gmf_item
            mlp = self.mlp(torch.cat([mean_mlp.expand(end - start, -1), mlp_item], dim=-1))
            results.append(self.output(torch.cat([gmf, mlp], dim=-1)).squeeze(-1).cpu())
        return torch.cat(results)


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


def shared_ranking(model: NeuMF, api_ids: Sequence[int], device: torch.device,
                   k: int) -> list[int]:
    scores = model.cold_start_scores(device).numpy()
    api_array = np.asarray(api_ids, dtype=np.int64)
    order = np.lexsort((api_array, -scores))[:k]
    return api_array[order].astype(int).tolist()


def split_truth(prepared: dict, split: str) -> dict[int, list[int]]:
    mids = [int(mid) for mid in prepared["splits"][split]]
    links = {int(mid): [int(api) for api in values]
             for mid, values in prepared["links"].items()}
    return {mid: links[mid] for mid in mids}


def train_one(prepared: dict, seed: int, config: dict, device_name: str) -> tuple[NeuMF, dict]:
    seed_everything(seed)
    splits = {name: [int(value) for value in values]
              for name, values in prepared["splits"].items()}
    links = {int(mid): [int(api) for api in values]
             for mid, values in prepared["links"].items()}
    api_ids = sorted({api for values in links.values() for api in values})
    api_index = {api: index for index, api in enumerate(api_ids)}
    train_mids = splits["train"]
    user_index = {mid: index for index, mid in enumerate(train_mids)}
    pairs = [(user_index[mid], api_index[api]) for mid in train_mids for api in links[mid]]
    pos_users = np.asarray([u for u, _ in pairs], dtype=np.int64)
    pos_items = np.asarray([i for _, i in pairs], dtype=np.int64)
    positive_sets = [set() for _ in train_mids]
    for user, item in pairs:
        positive_sets[user].add(item)

    device = torch.device(device_name if device_name.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model = NeuMF(len(train_mids), len(api_ids), config["gmf_dim"], config["mlp_dim"],
                  config["layers"], config["dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"],
                                 weight_decay=config["weight_decay"])
    criterion = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)
    validation_truth = split_truth(prepared, "valid")
    best_ndcg, best_epoch, best_state = -1.0, 0, None
    history, stale = [], 0

    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        negative_users = np.repeat(pos_users, config["negative_ratio"])
        negative_items = sample_negatives(negative_users, positive_sets, len(api_ids), rng)
        users = np.concatenate([pos_users, negative_users])
        items = np.concatenate([pos_items, negative_items])
        labels = np.concatenate([np.ones(len(pos_users), dtype=np.float32),
                                 np.zeros(len(negative_users), dtype=np.float32)])
        order = rng.permutation(len(users))
        epoch_loss, examples = 0.0, 0
        for start in range(0, len(order), config["batch_size"]):
            idx = order[start:start + config["batch_size"]]
            batch_users = torch.from_numpy(users[idx]).to(device)
            batch_items = torch.from_numpy(items[idx]).to(device)
            batch_labels = torch.from_numpy(labels[idx]).to(device)
            loss = criterion(model(batch_users, batch_items), batch_labels)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            epoch_loss += float(loss.detach()) * len(idx); examples += len(idx)

        if epoch % config["eval_every"] == 0 or epoch == 1:
            ranking = shared_ranking(model, api_ids, device, 20)
            predictions = {mid: ranking for mid in validation_truth}
            validation = ranking_metrics(predictions, validation_truth, (10,))
            ndcg = float(validation["ndcg@10"])
            row = {"epoch": epoch, "loss": epoch_loss / max(examples, 1),
                   "validation_ndcg@10": ndcg}
            history.append(row)
            print(f"NeuMF seed={seed} epoch={epoch} loss={row['loss']:.6f} val_ndcg@10={ndcg:.6f}", flush=True)
            if ndcg > best_ndcg + 1e-8:
                best_ndcg, best_epoch = ndcg, epoch
                best_state = {key: value.detach().cpu().clone()
                              for key, value in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if epoch >= 20 and stale >= config["patience"]:
                break

    model.load_state_dict(best_state)
    metadata = {
        **config, "device": str(device), "users": len(train_mids), "items": len(api_ids),
        "training_edges": len(pairs), "best_epoch": best_epoch,
        "best_validation_ndcg@10": best_ndcg, "history": history,
    }
    return model.cpu(), {"api_ids": api_ids, "metadata": metadata}


def evaluate_model(model: NeuMF, api_ids: Sequence[int], prepared: dict,
                   cutoffs: Sequence[int]) -> tuple[dict, list[int]]:
    device = torch.device("cpu")
    ranking = shared_ranking(model, api_ids, device, max(cutoffs))
    truth = split_truth(prepared, "test")
    predictions = {mid: list(ranking) for mid in truth}
    metrics = ranking_metrics(predictions, truth, cutoffs)
    metrics.update(catalog_coverage(predictions, len(api_ids), cutoffs))

    splits = {name: [int(value) for value in values]
              for name, values in prepared["splits"].items()}
    links = {int(mid): [int(api) for api in values]
             for mid, values in prepared["links"].items()}
    counts = Counter(api for mid in splits["train"] for api in links[mid])
    seen = sorted((api for api in api_ids if counts.get(api, 0) > 0),
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
        "# NeuMF Baseline Report", "",
        f"- Split seed: `{metadata['split_seed']}`",
        f"- Training seeds: `{', '.join(map(str, seeds))}`",
        f"- GMF/MLP dimensions: `{metadata['gmf_dim']}/{metadata['mlp_dim']}`; "
        f"MLP layers: `{metadata['layers']}`.",
        f"- lr=`{metadata['lr']}`, L2=`{metadata['weight_decay']}`, negatives=`{metadata['negative_ratio']}`.",
        f"- Best epochs selected by validation NDCG@10: `{best_epochs}`.",
        "- Cold-start protocol: unseen validation/test Mashups use mean learned user embeddings.",
        "- No test invocation edge is used for parameter selection or ranking.", "",
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
              "NeuMF models nonlinear collaborative interactions, but under the Mashup-level "
              "cold-start split it still cannot construct query-specific test user factors.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(workspace: Path, output_dir: Path, split_seed: int, seeds: Sequence[int],
        cutoffs: Sequence[int], config: dict, device: str) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"; checkpoint_dir.mkdir(exist_ok=True)
    prepared = load_workspace(workspace, split_seed)
    rows, rankings, run_metadata = [], {}, []
    for seed in seeds:
        model, model_data = train_one(prepared, int(seed), config, device)
        metrics, ranking = evaluate_model(model, model_data["api_ids"], prepared, cutoffs)
        rows.append({"method": "NeuMF", "seed": int(seed), **metrics})
        rankings[str(seed)] = ranking
        run_metadata.append(model_data["metadata"])
        torch.save({"model": model.state_dict(), "api_ids": model_data["api_ids"],
                    "seed": int(seed), "metadata": model_data["metadata"]},
                   checkpoint_dir / f"neumf_seed{seed}.pt")

    keys = [f"{metric}@{k}" for k in cutoffs
            for metric in ("precision", "recall", "ndcg", "coverage", "tail_recall")]
    summary = summarize(rows, keys)
    with (output_dir / "per_seed_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "seed", *keys, "queries", "tail_queries"],
                                extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    metadata = {"method": "NeuMF", "split_seed": split_seed,
                "cold_start_strategy": "mean learned GMF and MLP user embeddings",
                **config, "runs": run_metadata}
    payload = {"metadata": metadata, "seeds": list(map(int, seeds)), "metrics": summary}
    (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "cold_start_rankings.json").write_text(json.dumps(rankings, indent=2), encoding="utf-8")
    write_report(output_dir / "REPORT.md", summary, metadata, cutoffs, seeds)
    print(f"NeuMF experiment complete: {output_dir}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NeuMF baseline")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43, 71, 101])
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    default = {"gmf_dim": 32, "mlp_dim": 32, "layers": [64, 32, 16],
               "dropout": 0.1, "lr": 0.001, "weight_decay": 1e-5,
               "negative_ratio": 4, "max_epochs": 100, "eval_every": 5,
               "patience": 8, "batch_size": 2048}
    run(args.workspace, args.output_dir, args.split_seed, args.seeds,
        (5, 10, 15, 20), default, args.device)

