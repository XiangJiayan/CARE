"""Train and evaluate TSSGCF on the common ProgrammableWeb split."""
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
import pandas as pd
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from model import TSSGCF, textual_similarity_loss
from care.metrics import catalog_coverage, ranking_metrics


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode_texts(data_dir: Path, model_path: Path, cache_path: Path) -> dict:
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu", weights_only=False)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(str(model_path), device="cuda" if torch.cuda.is_available() else "cpu")
    mashups = pd.read_csv(data_dir / "mashup.csv", encoding="utf-8")
    apis = pd.read_csv(data_dir / "api.csv", encoding="utf-8")
    mashup_texts = (mashups["Name"].fillna("") + ". " + mashups["Description"].fillna("")).tolist()
    api_texts = (apis["Name"].fillna("") + ". " + apis["Description"].fillna("")).tolist()
    pack = {
        "mashup_ids": torch.tensor(mashups["ID"].astype(int).tolist()),
        "mashup_embeddings": torch.tensor(model.encode(
            mashup_texts, batch_size=128, show_progress_bar=True, normalize_embeddings=True)),
        "api_ids": torch.tensor(apis["ID"].astype(int).tolist()),
        "api_embeddings": torch.tensor(model.encode(
            api_texts, batch_size=128, show_progress_bar=True, normalize_embeddings=True)),
        "encoder": "sentence-transformers/all-MiniLM-L6-v2",
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pack, cache_path)
    return pack


def load_workspace(workspace: Path, split_seed: int, data_dir: Path,
                   model_path: Path, cache_path: Path) -> dict:
    prepared = json.loads((workspace / "prepared.json").read_text(encoding="utf-8"))
    if int(prepared.get("split_seed", split_seed)) != split_seed:
        raise ValueError("Workspace split seed does not match")
    prepared["text_pack"] = encode_texts(data_dir, model_path, cache_path)
    return prepared


def similarity_graph(text: torch.Tensor, threshold: float,
                     device: torch.device) -> torch.Tensor:
    text = F.normalize(text.float(), p=2, dim=1)
    similarity = text @ text.T
    mask = similarity >= threshold
    mask.fill_diagonal_(True)
    row, col = torch.where(mask)
    values = similarity[row, col].clamp_min(0)
    degree = torch.zeros(text.shape[0], dtype=torch.float32)
    degree.scatter_add_(0, row, values)
    # Symmetric degree normalization stabilizes repeated LightGCN propagation.
    values = values / torch.sqrt(degree[row].clamp_min(1e-8) * degree[col].clamp_min(1e-8))
    adjacency = torch.sparse_coo_tensor(torch.stack([row, col]), values,
                                        (len(text), len(text))).coalesce()
    return adjacency.to(device)


def build_data(prepared: dict, threshold: float, device: torch.device) -> dict:
    splits = {k: [int(v) for v in values] for k, values in prepared["splits"].items()}
    links = {int(mid): [int(a) for a in values] for mid, values in prepared["links"].items()}
    pack = prepared["text_pack"]
    all_mids = [int(v) for v in pack["mashup_ids"].tolist()]
    api_ids = [int(v) for v in pack["api_ids"].tolist()]
    mid_row = {mid: i for i, mid in enumerate(all_mids)}
    api_row = {aid: i for i, aid in enumerate(api_ids)}
    train_mids = splits["train"]
    mashup_text_all = pack["mashup_embeddings"].float()
    api_text = pack["api_embeddings"].float()
    train_text = torch.stack([mashup_text_all[mid_row[mid]] for mid in train_mids])
    user_index = {mid: i for i, mid in enumerate(train_mids)}
    pairs = [(user_index[mid], api_row[aid]) for mid in train_mids for aid in links[mid]]
    edge_users = np.asarray([u for u, _ in pairs], dtype=np.int64)
    edge_items = np.asarray([i for _, i in pairs], dtype=np.int64)
    positives = [set() for _ in train_mids]
    for u, i in pairs:
        positives[u].add(i)
    return {
        "splits": splits, "links": links, "train_mids": train_mids,
        "api_ids": api_ids, "mid_row": mid_row, "pairs": pairs,
        "edge_users": edge_users, "edge_items": edge_items, "positives": positives,
        "mashup_text_all": mashup_text_all, "train_text": train_text.to(device),
        "api_text": api_text.to(device),
        "mashup_adj": similarity_graph(train_text, threshold, device),
        "api_adj": similarity_graph(api_text, threshold, device),
    }


def sample_negatives(users: np.ndarray, positives: list[set[int]], item_count: int,
                     rng: np.random.Generator) -> np.ndarray:
    values = rng.integers(0, item_count, size=len(users), dtype=np.int64)
    bad = np.fromiter((int(i) in positives[int(u)] for u, i in zip(users, values)), bool, len(users))
    while bad.any():
        values[bad] = rng.integers(0, item_count, size=int(bad.sum()), dtype=np.int64)
        bad = np.fromiter((int(i) in positives[int(u)] for u, i in zip(users, values)), bool, len(users))
    return values


def limit_indices(indices: torch.Tensor, limit: int) -> torch.Tensor:
    indices = torch.unique(indices)
    if len(indices) > limit:
        indices = indices[:limit]
    return indices


def rank_split(model: TSSGCF, data: dict, split: str, device: torch.device,
               max_k: int, threshold: float) -> dict[int, list[int]]:
    model.eval()
    mids = data["splits"][split]
    with torch.no_grad():
        graph_m, graph_a, _, final_a = model(
            data["mashup_adj"], data["api_adj"], data["train_text"], data["api_text"])
        rows = [data["mid_row"][mid] for mid in mids]
        query_text_raw = data["mashup_text_all"][rows].to(device)
        query_text = F.normalize(model.text_mlp(query_text_raw), p=2, dim=1)
        train_text = F.normalize(data["train_text"], p=2, dim=1)
        similarity = F.normalize(query_text_raw, p=2, dim=1) @ train_text.T
        mask = similarity >= threshold
        weights = similarity.clamp_min(0) * mask
        empty = weights.sum(dim=1) <= 0
        graph_query = weights @ graph_m / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        if empty.any():
            graph_query[empty] = graph_m.mean(dim=0)
        graph_query = F.normalize(graph_query, p=2, dim=1)
        final_query = ((1.0 - model.text_weight) * graph_query
                       + model.text_weight * query_text)
        scores = final_query @ final_a.T
        top = torch.topk(scores, k=max_k, dim=1).indices.cpu().numpy()
    api_array = np.asarray(data["api_ids"], dtype=np.int64)
    return {mid: api_array[top[i]].astype(int).tolist() for i, mid in enumerate(mids)}


def truth_for(data: dict, split: str) -> dict[int, list[int]]:
    return {mid: data["links"][mid] for mid in data["splits"][split]}


def train_one(prepared: dict, seed: int, cfg: dict, device_name: str):
    seed_everything(seed)
    device = torch.device(device_name if device_name.startswith("cuda") and torch.cuda.is_available() else "cpu")
    data = build_data(prepared, cfg["threshold"], device)
    model = TSSGCF(len(data["train_mids"]), len(data["api_ids"]),
                   data["api_text"].shape[1], cfg["dim"], cfg["layers"],
                   cfg["text_weight"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    rng = np.random.default_rng(seed)
    base_order = np.arange(len(data["edge_users"]))
    best_score, best_epoch, best_state, stale, history = -1.0, 0, None, 0, []

    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        order = rng.permutation(base_order)
        negatives = sample_negatives(data["edge_users"], data["positives"], len(data["api_ids"]), rng)
        losses = []
        for start in range(0, len(order), cfg["batch_size"]):
            batch = order[start:start + cfg["batch_size"]]
            u = torch.tensor(data["edge_users"][batch], dtype=torch.long, device=device)
            p = torch.tensor(data["edge_items"][batch], dtype=torch.long, device=device)
            n = torch.tensor(negatives[batch], dtype=torch.long, device=device)
            graph_m, graph_a, final_m, final_a = model(
                data["mashup_adj"], data["api_adj"], data["train_text"], data["api_text"])
            pos = (final_m[u] * final_a[p]).sum(dim=1)
            neg = (final_m[u] * final_a[n]).sum(dim=1)
            bpr = -F.logsigmoid(pos - neg).mean()
            ui = limit_indices(u, cfg["tss_sample_size"])
            ai = limit_indices(torch.cat([p, n]), cfg["tss_sample_size"])
            tss = cfg["lambda_tss"] * (
                textual_similarity_loss(graph_m, data["train_text"], ui,
                                        cfg["threshold"], cfg["beta"])
                + textual_similarity_loss(graph_a, data["api_text"], ai,
                                          cfg["threshold"], cfg["beta"])
            )
            explicit_l2 = sum(parameter.pow(2).sum() for parameter in model.parameters())
            loss = bpr + tss + cfg["explicit_l2"] * explicit_l2
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            losses.append([float(loss.detach()), float(bpr.detach()), float(tss.detach())])

        if epoch % cfg["eval_every"] == 0 or epoch == 1:
            predictions = rank_split(model, data, "valid", device, 20, cfg["threshold"])
            score = float(ranking_metrics(predictions, truth_for(data, "valid"), (10,))["ndcg@10"])
            avg = np.asarray(losses).mean(axis=0)
            history.append({"epoch": epoch, "loss": float(avg[0]), "bpr": float(avg[1]),
                            "tss": float(avg[2]), "validation_ndcg@10": score})
            print(f"TSSGCF seed={seed} epoch={epoch} loss={avg[0]:.6f} val_ndcg@10={score:.6f}", flush=True)
            if score > best_score + 1e-8:
                best_score, best_epoch, stale = score, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            if epoch >= 30 and stale >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    metadata = {**cfg, "device": str(device), "users": len(data["train_mids"]),
                "items": len(data["api_ids"]), "training_edges": len(data["pairs"]),
                "mashup_similarity_edges": int(data["mashup_adj"]._nnz()),
                "api_similarity_edges": int(data["api_adj"]._nnz()),
                "best_epoch": best_epoch, "best_validation_ndcg@10": best_score,
                "history": history}
    return model.cpu(), metadata


def evaluate(model: TSSGCF, prepared: dict, cfg: dict, cutoffs: Sequence[int]):
    data = build_data(prepared, cfg["threshold"], torch.device("cpu"))
    predictions = rank_split(model, data, "test", torch.device("cpu"), max(cutoffs), cfg["threshold"])
    truth = truth_for(data, "test")
    metrics = ranking_metrics(predictions, truth, cutoffs)
    metrics.update(catalog_coverage(predictions, len(data["api_ids"]), cutoffs))
    counts = Counter(a for mid in data["splits"]["train"] for a in data["links"][mid])
    seen = sorted((a for a in data["api_ids"] if counts.get(a, 0) > 0),
                  key=lambda a: (-counts[a], a))
    tail = set(seen[math.ceil(0.2 * len(seen)):])
    tail_truth = {mid: [a for a in values if a in tail] for mid, values in truth.items()}
    tail_truth = {mid: values for mid, values in tail_truth.items() if values}
    tm = ranking_metrics({mid: predictions[mid] for mid in tail_truth}, tail_truth, cutoffs)
    for k in cutoffs:
        metrics[f"tail_recall@{k}"] = tm[f"recall@{k}"]
    metrics["tail_queries"] = len(tail_truth)
    return metrics, predictions


def summarize(rows: list[dict], keys: Sequence[str]) -> dict:
    out = {}
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows])
        out[key] = {"mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0}
    return out


def write_report(path: Path, summary: dict, metadata: dict,
                 cutoffs: Sequence[int], seeds: Sequence[int]) -> None:
    lines = ["# TSSGCF Baseline Report", "",
             f"- Split seed: `{metadata['split_seed']}`",
             f"- Training seeds: `{', '.join(map(str, seeds))}`",
             "- Source method: IntelligentServiceLab/TSSGCF (official GitHub repository).",
             f"- MiniLM similarity threshold=`{metadata['threshold']}`, embedding dimension=`{metadata['dim']}`, graph layers=`{metadata['layers']}`.",
             f"- Best epochs selected by validation NDCG@10: `{metadata['best_epochs']}`.",
             "- Strict cold start: no validation/test invocation edge is used.",
             "- An unseen Mashup graph vector is induced from text-similar training Mashups.", "",
             "| K | Precision | Recall | NDCG | Coverage | Tail Recall |",
             "|---:|---:|---:|---:|---:|---:|"]
    for k in cutoffs:
        cells = []
        for metric in ("precision", "recall", "ndcg", "coverage", "tail_recall"):
            value = summary[f"{metric}@{k}"]
            cells.append((f"{100*value['mean']:.2f}% ± {100*value['std']:.2f}%"
                          if metric == "coverage" else f"{value['mean']:.6f} ± {value['std']:.6f}"))
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    lines += ["", "## Protocol note", "",
              "The official code holds out API edges for already observed Mashups. This adaptation uses the project's common Mashup-level cold-start split for a fair comparison.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(workspace: Path, data_dir: Path, model_path: Path, output_dir: Path,
        split_seed: int, seeds: Sequence[int], cutoffs: Sequence[int],
        cfg: dict, device: str) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True); (output_dir / "checkpoints").mkdir(exist_ok=True)
    prepared = load_workspace(workspace, split_seed, data_dir, model_path,
                              HERE / "cache" / "minilm_embeddings.pt")
    rows, runs, predictions_all = [], [], {}
    for seed in seeds:
        model, metadata = train_one(prepared, int(seed), cfg, device)
        metrics, predictions = evaluate(model, prepared, cfg, cutoffs)
        rows.append({"method": "TSSGCF", "seed": int(seed), **metrics})
        runs.append(metadata)
        predictions_all[str(seed)] = {str(mid): values for mid, values in predictions.items()}
        torch.save({"state_dict": model.state_dict(), "metadata": metadata},
                   output_dir / "checkpoints" / f"tssgcf_seed{seed}.pt")
    keys = [f"{m}@{k}" for k in cutoffs for m in ("precision", "recall", "ndcg", "coverage", "tail_recall")]
    summary = summarize(rows, keys)
    with (output_dir / "per_seed_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "seed", *keys, "queries", "tail_queries"], extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    metadata = {**cfg, "split_seed": split_seed, "seeds": list(map(int, seeds)),
                "best_epochs": [m["best_epoch"] for m in runs], "runs": runs,
                "encoder": "sentence-transformers/all-MiniLM-L6-v2",
                "source_url": "https://github.com/IntelligentServiceLab/TSSGCF"}
    payload = {"metadata": metadata, "metrics": summary}
    (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "test_top20_predictions.json").write_text(json.dumps(predictions_all), encoding="utf-8")
    write_report(output_dir / "REPORT.md", summary, metadata, cutoffs, seeds)
    print(f"TSSGCF experiment complete: {output_dir}")
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description="Run TSSGCF baseline")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43, 71, 101])
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[5, 10, 15, 20])
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    import config
    args = parse_args()
    cfg = {"dim": config.EMBEDDING_DIM, "layers": config.PROPAGATION_LAYERS,
           "threshold": config.SIMILARITY_THRESHOLD, "text_weight": config.TEXT_FUSION_WEIGHT,
           "lambda_tss": config.TSS_WEIGHT, "beta": config.TSS_BETA,
           "lr": config.LEARNING_RATE, "weight_decay": config.WEIGHT_DECAY,
           "explicit_l2": config.EXPLICIT_L2_WEIGHT, "batch_size": config.BATCH_SIZE,
           "max_epochs": config.MAX_EPOCHS, "eval_every": config.EVAL_EVERY,
           "patience": config.PATIENCE, "tss_sample_size": config.TSS_SAMPLE_SIZE}
    run(args.workspace, args.data_dir, args.model_path, args.output_dir,
        args.split_seed, args.seeds, args.cutoffs, cfg, args.device)

