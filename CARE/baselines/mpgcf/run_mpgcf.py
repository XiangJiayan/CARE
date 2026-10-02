"""Train and evaluate MPGCF on the fixed ProgrammableWeb split."""
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
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    # Keep this baseline directory first so its local config/model modules are
    # not shadowed by the project-level config.py.
    sys.path.append(str(PROJECT_ROOT))

from care.metrics import catalog_coverage, ranking_metrics
from model import MPGCF, indexed_info_nce, symmetric_indexed_info_nce


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_workspace(path: Path, split_seed: int) -> dict:
    prepared = json.loads((path / "prepared.json").read_text(encoding="utf-8"))
    if int(prepared.get("split_seed", split_seed)) != split_seed:
        raise ValueError("Workspace split seed does not match the requested split seed")
    api_pack = torch.load(path / "api_embeddings.pt", map_location="cpu", weights_only=False)
    mashup_pack = torch.load(path / "strict" / "mashup_embeddings.pt",
                             map_location="cpu", weights_only=False)
    prepared["api_pack"] = api_pack
    prepared["mashup_pack"] = mashup_pack
    return prepared


def sparse_adjacency(users: np.ndarray, items: np.ndarray, user_count: int,
                     item_count: int, exponent: float, device: torch.device) -> torch.Tensor:
    item_nodes = items + user_count
    row = np.concatenate([users, item_nodes])
    col = np.concatenate([item_nodes, users])
    degree = np.bincount(row, minlength=user_count + item_count).astype(np.float32)
    left = np.power(np.maximum(degree[row], 1e-7), -exponent)
    right = np.power(np.maximum(degree[col], 1e-7), -(1.0 - exponent))
    values = torch.tensor(left * right, dtype=torch.float32, device=device)
    indices = torch.tensor(np.stack([row, col]), dtype=torch.long, device=device)
    return torch.sparse_coo_tensor(
        indices, values, (user_count + item_count, user_count + item_count),
        device=device).coalesce()


def build_data(prepared: dict, device: torch.device, r: float) -> dict:
    splits = {key: [int(v) for v in values] for key, values in prepared["splits"].items()}
    links = {int(mid): [int(a) for a in values] for mid, values in prepared["links"].items()}
    train_mids = splits["train"]
    api_ids = [int(v) for v in prepared["api_pack"]["api_ids"].tolist()]
    user_index = {mid: i for i, mid in enumerate(train_mids)}
    api_index = {aid: i for i, aid in enumerate(api_ids)}
    pairs = [(user_index[mid], api_index[aid]) for mid in train_mids for aid in links[mid]]
    edge_users = np.asarray([u for u, _ in pairs], dtype=np.int64)
    edge_items = np.asarray([i for _, i in pairs], dtype=np.int64)
    positives = [set() for _ in train_mids]
    for u, i in pairs:
        positives[u].add(i)

    counts = np.bincount(edge_items, minlength=len(api_ids))
    seen = np.flatnonzero(counts > 0)
    seen = seen[np.lexsort((seen, -counts[seen]))]
    head_n = math.ceil(0.2 * len(seen))
    head, tail = seen[:head_n], seen[head_n:]

    api_pack, mashup_pack = prepared["api_pack"], prepared["mashup_pack"]
    packed_api_ids = [int(v) for v in api_pack["api_ids"].tolist()]
    packed_mid_ids = [int(v) for v in mashup_pack["mashup_ids"].tolist()]
    api_row = {aid: i for i, aid in enumerate(packed_api_ids)}
    mid_row = {mid: i for i, mid in enumerate(packed_mid_ids)}
    api_text = torch.stack([api_pack["embeddings"][api_row[aid]].float() for aid in api_ids]).to(device)
    train_text = torch.stack([mashup_pack["embeddings"][mid_row[mid]].float() for mid in train_mids]).to(device)

    return {
        "splits": splits, "links": links, "train_mids": train_mids,
        "api_ids": api_ids, "api_index": api_index, "mid_row": mid_row,
        "mashup_embeddings": mashup_pack["embeddings"].float(),
        "edge_users": edge_users, "edge_items": edge_items, "positives": positives,
        "head": head, "tail": tail, "counts": counts, "pairs": pairs,
        "api_text": api_text, "train_text": train_text,
        "acc_adj": sparse_adjacency(edge_users, edge_items, len(train_mids), len(api_ids), 0.5, device),
        "nacc_adj": sparse_adjacency(edge_users, edge_items, len(train_mids), len(api_ids), r, device),
    }


def sample_from_pool(users: np.ndarray, positives: list[set[int]], pool: np.ndarray,
                     rng: np.random.Generator) -> np.ndarray:
    sampled = pool[rng.integers(0, len(pool), size=len(users))]
    invalid = np.fromiter((int(i) in positives[int(u)] for u, i in zip(users, sampled)),
                          dtype=bool, count=len(users))
    while invalid.any():
        sampled[invalid] = pool[rng.integers(0, len(pool), size=int(invalid.sum()))]
        invalid = np.fromiter((int(i) in positives[int(u)] for u, i in zip(users, sampled)),
                              dtype=bool, count=len(users))
    return sampled


def rank_split(model: MPGCF, data: dict, split: str, device: torch.device,
               max_k: int) -> dict[int, list[int]]:
    model.eval()
    mids = data["splits"][split]
    with torch.no_grad():
        _, _, _, _, graph_users, graph_items = model.propagate(data["acc_adj"], data["nacc_adj"])
        cold_graph = graph_users.mean(dim=0, keepdim=True)
        rows = [data["mid_row"][mid] for mid in mids]
        query_raw = data["mashup_embeddings"][rows].to(device)
        query_text = model.project_text(query_raw)
        query_graph = cold_graph.expand(len(mids), -1)
        query = model.fuse(query_graph, query_text)
        item_text = model.project_text(data["api_text"])
        items = model.fuse(graph_items, item_text)
        scores = query @ items.T
        top = torch.topk(scores, k=max_k, dim=1).indices.cpu().numpy()
    api_array = np.asarray(data["api_ids"], dtype=np.int64)
    return {mid: api_array[top[row]].astype(int).tolist() for row, mid in enumerate(mids)}


def truth_for(data: dict, split: str) -> dict[int, list[int]]:
    return {mid: data["links"][mid] for mid in data["splits"][split]}


def train_one(prepared: dict, seed: int, cfg: dict, device_name: str):
    seed_everything(seed)
    device = torch.device(device_name if device_name.startswith("cuda") and torch.cuda.is_available() else "cpu")
    data = build_data(prepared, device, cfg["r"])
    model = MPGCF(len(data["train_mids"]), len(data["api_ids"]),
                  data["api_text"].shape[1], cfg["dim"], cfg["layers"],
                  cfg["alpha"], cfg["text_weight"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    rng = np.random.default_rng(seed)
    order_base = np.arange(len(data["edge_users"]))
    best_score, best_epoch, best_state, stale = -1.0, 0, None, 0
    history = []

    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        order = rng.permutation(order_base)
        head_neg = sample_from_pool(data["edge_users"], data["positives"], data["head"], rng)
        tail_neg = sample_from_pool(data["edge_users"], data["positives"], data["tail"], rng)
        totals = []
        for start in range(0, len(order), cfg["batch_size"]):
            batch = order[start:start + cfg["batch_size"]]
            u = torch.tensor(data["edge_users"][batch], dtype=torch.long, device=device)
            p = torch.tensor(data["edge_items"][batch], dtype=torch.long, device=device)
            nh = torch.tensor(head_neg[batch], dtype=torch.long, device=device)
            nt = torch.tensor(tail_neg[batch], dtype=torch.long, device=device)
            acc_u, acc_i, nacc_u, nacc_i, graph_u, graph_i = model.propagate(data["acc_adj"], data["nacc_adj"])
            text_u_all = model.project_text(data["train_text"])
            text_i_all = model.project_text(data["api_text"])
            uu, pp = torch.unique(u), torch.unique(p)

            structure = cfg["lambda_structure"] * (
                cfg["structure_user_weight"] * indexed_info_nce(
                    acc_u[uu], nacc_u[uu], nacc_u, uu, cfg["structure_temp"])
                + cfg["structure_item_weight"] * indexed_info_nce(
                    acc_i[pp], nacc_i[pp], nacc_i, pp, cfg["structure_temp"])
            )
            semantic = cfg["lambda_semantic"] * (
                cfg["semantic_user_weight"] * symmetric_indexed_info_nce(
                    graph_u[uu], text_u_all[uu], graph_u, text_u_all, uu, cfg["semantic_temp"])
                + cfg["semantic_item_weight"] * symmetric_indexed_info_nce(
                    graph_i[pp], text_i_all[pp], graph_i, text_i_all, pp, cfg["semantic_temp"])
            )

            fu = model.fuse(graph_u[u], text_u_all[u])
            fp = model.fuse(graph_i[p], text_i_all[p])
            fnh = model.fuse(graph_i[nh], text_i_all[nh])
            fnt = model.fuse(graph_i[nt], text_i_all[nt])
            pos_score = (fu * fp).sum(dim=1)
            head_score = (fp * fnh).sum(dim=1) * (fu * fnh).sum(dim=1)
            tail_score = (fp * fnt).sum(dim=1) * (fu * fnt).sum(dim=1)
            bpr = 0.5 * (-F.logsigmoid(pos_score - head_score).mean()
                         -F.logsigmoid(pos_score - tail_score).mean())
            reg = (model.user_embedding(u).pow(2).sum()
                   + model.item_embedding(p).pow(2).sum()
                   + model.item_embedding(nh).pow(2).sum()
                   + model.item_embedding(nt).pow(2).sum()) / len(u)
            loss = bpr + structure + semantic + cfg["lambda_reg"] * reg
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            totals.append([float(loss.detach()), float(bpr.detach()),
                           float(structure.detach()), float(semantic.detach())])

        if epoch % cfg["eval_every"] == 0 or epoch == 1:
            predictions = rank_split(model, data, "valid", device, 20)
            val = ranking_metrics(predictions, truth_for(data, "valid"), (10,))
            ndcg = float(val["ndcg@10"])
            avg = np.asarray(totals).mean(axis=0)
            history.append({"epoch": epoch, "loss": float(avg[0]), "bpr": float(avg[1]),
                            "structure": float(avg[2]), "semantic": float(avg[3]),
                            "validation_ndcg@10": ndcg})
            print(f"MPGCF seed={seed} epoch={epoch} loss={avg[0]:.6f} val_ndcg@10={ndcg:.6f}", flush=True)
            if ndcg > best_score + 1e-8:
                best_score, best_epoch, stale = ndcg, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            if epoch >= 30 and stale >= cfg["patience"]:
                break

    model.load_state_dict(best_state)
    model = model.cpu()
    metadata = {**cfg, "device": str(device), "users": len(data["train_mids"]),
                "items": len(data["api_ids"]), "training_edges": len(data["pairs"]),
                "best_epoch": best_epoch, "best_validation_ndcg@10": best_score,
                "history": history}
    return model, metadata


def evaluate(model: MPGCF, prepared: dict, cfg: dict, cutoffs: Sequence[int]):
    device = torch.device("cpu")
    data = build_data(prepared, device, cfg["r"])
    predictions = rank_split(model, data, "test", device, max(cutoffs))
    truth = truth_for(data, "test")
    metrics = ranking_metrics(predictions, truth, cutoffs)
    metrics.update(catalog_coverage(predictions, len(data["api_ids"]), cutoffs))
    seen = np.flatnonzero(data["counts"] > 0)
    seen = seen[np.lexsort((seen, -data["counts"][seen]))]
    tail_ids = {data["api_ids"][i] for i in seen[math.ceil(0.2 * len(seen)): ]}
    tail_truth = {mid: [aid for aid in values if aid in tail_ids] for mid, values in truth.items()}
    tail_truth = {mid: values for mid, values in tail_truth.items() if values}
    tail_metrics = ranking_metrics({mid: predictions[mid] for mid in tail_truth}, tail_truth, cutoffs)
    for k in cutoffs:
        metrics[f"tail_recall@{k}"] = tail_metrics[f"recall@{k}"]
    metrics["tail_queries"] = len(tail_truth)
    return metrics, predictions


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
        "# MPGCF Baseline Report", "",
        f"- Split seed: `{metadata['split_seed']}`",
        f"- Training seeds: `{', '.join(map(str, seeds))}`",
        "- Source method: IntelligentServiceLab/MPGCF (official GitHub repository).",
        f"- Embedding dimension=`{metadata['dim']}`, graph layers=`{metadata['layers']}`, lr=`{metadata['lr']}`.",
        f"- Best epochs selected by validation NDCG@10: `{metadata['best_epochs']}`.",
        "- Strict cold start: validation/test Mashup invocation edges are never added to the graph.",
        "- Unseen Mashups use their Sentence-T5 description plus the mean training-Mashup graph vector.", "",
        "| K | Precision | Recall | NDCG | Coverage | Tail Recall |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for k in cutoffs:
        values = []
        for metric in ("precision", "recall", "ndcg", "coverage", "tail_recall"):
            value = summary[f"{metric}@{k}"]
            if metric == "coverage":
                values.append(f"{100 * value['mean']:.2f}% ± {100 * value['std']:.2f}%")
            else:
                values.append(f"{value['mean']:.6f} ± {value['std']:.6f}")
        lines.append(f"| {k} | " + " | ".join(values) + " |")
    lines += ["", "## Protocol note", "",
              "The official repository evaluates held-out APIs for graph-observed Mashups. "
              "This run instead uses the project's common Mashup-level cold-start split so all baselines are directly comparable.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(workspace: Path, output_dir: Path, split_seed: int, seeds: Sequence[int],
        cutoffs: Sequence[int], cfg: dict, device: str) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    prepared = load_workspace(workspace, split_seed)
    rows, run_metadata, all_predictions = [], [], {}
    for seed in seeds:
        model, metadata = train_one(prepared, int(seed), cfg, device)
        metrics, predictions = evaluate(model, prepared, cfg, cutoffs)
        rows.append({"method": "MPGCF", "seed": int(seed), **metrics})
        run_metadata.append(metadata)
        all_predictions[str(seed)] = {str(mid): values for mid, values in predictions.items()}
        torch.save({"state_dict": model.state_dict(), "metadata": metadata},
                   output_dir / "checkpoints" / f"mpgcf_seed{seed}.pt")

    metric_keys = [f"{metric}@{k}" for k in cutoffs
                   for metric in ("precision", "recall", "ndcg", "coverage", "tail_recall")]
    summary = summarize(rows, metric_keys)
    with (output_dir / "per_seed_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "seed", *metric_keys,
                                                     "queries", "tail_queries"], extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    metadata = {**cfg, "split_seed": split_seed, "seeds": list(map(int, seeds)),
                "best_epochs": [m["best_epoch"] for m in run_metadata],
                "runs": run_metadata, "source_url": "https://github.com/IntelligentServiceLab/MPGCF"}
    payload = {"metadata": metadata, "metrics": summary}
    (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "test_top20_predictions.json").write_text(
        json.dumps(all_predictions, ensure_ascii=False), encoding="utf-8")
    write_report(output_dir / "REPORT.md", summary, metadata, cutoffs, seeds)
    print(f"MPGCF experiment complete: {output_dir}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MPGCF baseline")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43, 71, 101])
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[5, 10, 15, 20])
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    import config
    args = parse_args()
    configuration = {
        "dim": config.EMBEDDING_DIM, "layers": config.PROPAGATION_LAYERS,
        "lr": config.LEARNING_RATE, "lambda_structure": config.STRUCTURE_LOSS_WEIGHT,
        "lambda_semantic": config.SEMANTIC_LOSS_WEIGHT, "lambda_reg": config.REGULARIZATION_WEIGHT,
        "structure_temp": config.STRUCTURE_TEMPERATURE, "semantic_temp": config.SEMANTIC_TEMPERATURE,
        "structure_user_weight": config.STRUCTURE_USER_WEIGHT,
        "structure_item_weight": config.STRUCTURE_ITEM_WEIGHT,
        "semantic_user_weight": config.SEMANTIC_USER_WEIGHT,
        "semantic_item_weight": config.SEMANTIC_ITEM_WEIGHT,
        "alpha": config.ACCURACY_FUSION_ALPHA, "r": config.NON_ACCURACY_EXPONENT,
        "text_weight": config.TEXT_FUSION_WEIGHT, "batch_size": config.BATCH_SIZE,
        "max_epochs": config.MAX_EPOCHS, "eval_every": config.EVAL_EVERY,
        "patience": config.PATIENCE,
    }
    run(args.workspace, args.output_dir, args.split_seed, args.seeds,
        args.cutoffs, configuration, args.device)
