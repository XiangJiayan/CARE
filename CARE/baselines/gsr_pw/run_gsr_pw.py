"""Train and evaluate GSR_REC on the fixed ProgrammableWeb split."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from care.data import (
    build_mashup_texts,
    load_pw_data,
    relation_map,
    split_mashups,
)
from care.metrics import catalog_coverage, ranking_metrics
from model import GSR


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[-_.][A-Za-z0-9]+)*")
PAD, UNK = "<pad>", "<unk>"


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(str(text))]


def requirement_and_category_texts(data) -> tuple[dict[int, str], dict[int, str]]:
    full = build_mashup_texts(data)
    names = data.categories.set_index("ID")["Name"].fillna("").astype(str).to_dict()
    categories = relation_map(data.mashup_category, "MashupID", "CateID")
    category_text = {
        int(mid): " ".join(names[cid] for cid in categories.get(int(mid), []) if names.get(cid))
        or "uncategorized"
        for mid in data.mashup_api["MashupID"].unique()
    }
    requirements = {}
    for mid, text in full.items():
        requirements[int(mid)] = text.split(" Categories:", 1)[0]
    return requirements, category_text


def build_vocabulary(
    train_ids: Sequence[int], requirements: dict[int, str], categories: dict[int, str]
) -> dict[str, int]:
    counts = Counter()
    for mid in train_ids:
        counts.update(tokenize(requirements[mid]))
        counts.update(tokenize(categories[mid]))
    vocabulary = {PAD: 0, UNK: 1}
    for token, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        vocabulary[token] = len(vocabulary)
    return vocabulary


def encode_text(text: str, vocabulary: dict[str, int], max_length: int) -> list[int]:
    values = [vocabulary.get(token, 1) for token in tokenize(text)][:max_length]
    return values or [1]


def compute_retrieval(
    requirements: dict[int, str],
    splits: dict[str, list[int]],
    encoder_name: str,
    device: str,
    whitening_dim: int,
    cache_path: Path,
) -> tuple[dict[int, int], dict[int, float], dict]:
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        return (
            {int(k): int(v) for k, v in cached["retrieved"].items()},
            {int(k): float(v) for k, v in cached["similarity"].items()},
            cached["metadata"],
        )
    from sentence_transformers import SentenceTransformer

    ids = sorted(requirements)
    model = SentenceTransformer(encoder_name, device=device)
    model.max_seq_length = 256
    vectors = model.encode(
        [requirements[mid] for mid in ids],
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(np.float64)
    id_to_row = {mid: row for row, mid in enumerate(ids)}
    train_ids = [int(mid) for mid in splits["train"]]
    train_rows = np.asarray([id_to_row[mid] for mid in train_ids])
    train_vectors = vectors[train_rows]
    mean = train_vectors.mean(axis=0, keepdims=True)
    centered = train_vectors - mean
    covariance = centered.T @ centered / max(1, len(centered))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1][: min(whitening_dim, vectors.shape[1])]
    transform = eigenvectors[:, order] / np.sqrt(eigenvalues[order].clip(1e-8))[None, :]
    whitened = (vectors - mean) @ transform
    train_whitened = whitened[train_rows]

    retrieved: dict[int, int] = {}
    distances: dict[int, float] = {}
    train_position = {mid: index for index, mid in enumerate(train_ids)}
    for start in range(0, len(ids), 256):
        block_ids = ids[start : start + 256]
        block = whitened[start : start + len(block_ids)]
        distance = (
            (block * block).sum(axis=1, keepdims=True)
            + (train_whitened * train_whitened).sum(axis=1)[None, :]
            - 2.0 * block @ train_whitened.T
        )
        for row, mid in enumerate(block_ids):
            if mid in train_position:
                distance[row, train_position[mid]] = np.inf
            nearest = int(np.argmin(distance[row]))
            retrieved[mid] = train_ids[nearest]
            distances[mid] = float(max(0.0, distance[row, nearest]))
    scale = float(np.median(list(distances.values()))) or 1.0
    similarity = {mid: 1.0 / (1.0 + value / scale) for mid, value in distances.items()}
    metadata = {
        "encoder": "Sentence-T5",
        "whitening_dim": int(len(order)),
        "retrieval_repository": "training Mashups only",
        "distance": "whitened squared Euclidean",
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {"retrieved": retrieved, "similarity": similarity, "metadata": metadata},
            indent=2,
        ),
        encoding="utf-8",
    )
    return retrieved, similarity, metadata


class MashupDataset(Dataset):
    def __init__(self, ids: Sequence[int], features: dict[int, dict]):
        self.ids = list(map(int, ids))
        self.features = features

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict:
        mid = self.ids[index]
        return {"mid": mid, **self.features[mid]}


def pad(values: list[list[int]], padding: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([len(value) for value in values], dtype=torch.long)
    width = int(lengths.max())
    result = torch.full((len(values), width), padding, dtype=torch.long)
    for row, value in enumerate(values):
        result[row, : len(value)] = torch.tensor(value, dtype=torch.long)
    return result, lengths


def collate(rows: list[dict]) -> dict:
    requirement, requirement_length = pad([row["requirement"] for row in rows])
    category, category_length = pad([row["category"] for row in rows])
    retrieved_requirement, retrieved_requirement_length = pad(
        [row["retrieved_requirement"] for row in rows]
    )
    retrieved_category, retrieved_category_length = pad(
        [row["retrieved_category"] for row in rows]
    )
    target, _ = pad([row["target"] for row in rows], padding=-100)
    target_set, _ = pad([row["target_set"] for row in rows], padding=-1)
    return {
        "mids": [row["mid"] for row in rows],
        "requirement": requirement,
        "requirement_length": requirement_length,
        "category": category,
        "category_length": category_length,
        "retrieved_requirement": retrieved_requirement,
        "retrieved_requirement_length": retrieved_requirement_length,
        "retrieved_category": retrieved_category,
        "retrieved_category_length": retrieved_category_length,
        "similarity": torch.tensor([row["similarity"] for row in rows]),
        "target": target,
        "target_set": target_set,
    }


def to_device(batch: dict, device: str) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def encode_contexts(model: GSR, batch: dict):
    current = model.encode(
        batch["requirement"],
        batch["requirement_length"],
        batch["category"],
        batch["category_length"],
    )
    retrieved = model.encode(
        batch["retrieved_requirement"],
        batch["retrieved_requirement_length"],
        batch["retrieved_category"],
        batch["retrieved_category_length"],
    )
    return current, retrieved


@torch.no_grad()
def rankings_for_loader(
    model: GSR,
    loader: DataLoader,
    api_ids: Sequence[int],
    device: str,
    gamma: float,
    top_k: int,
) -> dict[int, list[int]]:
    model.eval()
    predictions = {}
    api_ids = np.asarray(api_ids, dtype=np.int64)
    for batch in loader:
        batch = to_device(batch, device)
        current, retrieved = encode_contexts(model, batch)
        score = model.first_step_scores(
            current, retrieved, batch["similarity"], gamma
        ).cpu().numpy()
        order = np.argsort(-score, axis=1)[:, :top_k]
        for mid, row in zip(batch["mids"], order):
            predictions[int(mid)] = api_ids[row].astype(int).tolist()
    return predictions


def evaluate(
    predictions: dict[int, list[int]],
    truth: dict[int, list[int]],
    api_ids: Sequence[int],
    train_counts: Counter,
    cutoffs: Sequence[int],
) -> dict:
    result = ranking_metrics(predictions, truth, cutoffs)
    result.update(catalog_coverage(predictions, len(api_ids), cutoffs))
    seen = sorted(
        (api for api in api_ids if train_counts[int(api)]),
        key=lambda api: (-train_counts[int(api)], int(api)),
    )
    tail = set(map(int, seen[math.ceil(0.2 * len(seen)) :]))
    tail_truth = {
        mid: [api for api in targets if api in tail]
        for mid, targets in truth.items()
    }
    tail_truth = {mid: values for mid, values in tail_truth.items() if values}
    tail_metrics = ranking_metrics(
        {mid: predictions[mid] for mid in tail_truth}, tail_truth, cutoffs
    )
    for k in cutoffs:
        result[f"tail_recall@{k}"] = tail_metrics[f"recall@{k}"]
    result["tail_queries"] = len(tail_truth)
    return result


def train_seed(
    seed: int,
    features: dict[int, dict],
    splits: dict[str, list[int]],
    truth: dict[str, dict[int, list[int]]],
    api_ids: list[int],
    train_counts: Counter,
    text_vocab_size: int,
    output_dir: Path,
    args,
) -> tuple[dict, dict[int, list[int]]]:
    seed_all(seed)
    checkpoint = output_dir / "checkpoints" / f"gsr_seed{seed}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    train_loader = DataLoader(
        MashupDataset(splits["train"], features),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(seed),
    )
    valid_loader = DataLoader(
        MashupDataset(splits["valid"], features),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        MashupDataset(splits["test"], features),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    model = GSR(
        text_vocab_size,
        len(api_ids),
        word_dim=args.word_dim,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
    ).to(args.device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_score, best_state, best_epoch, stale = -1.0, None, 0, 0
    history = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = np.zeros(4, dtype=np.float64)
        batches = 0
        for batch in train_loader:
            batch = to_device(batch, args.device)
            current, retrieved = encode_contexts(model, batch)
            nll = model.teacher_forced_nll(
                current,
                retrieved,
                batch["similarity"],
                batch["target"],
                args.retrieval_gamma,
            )
            rl, sampled_reward, greedy_reward = model.scst_loss(
                current,
                retrieved,
                batch["similarity"],
                batch["target_set"],
                args.retrieval_gamma,
                args.max_generated_services,
            )
            loss = args.nll_weight * nll + (1.0 - args.nll_weight) * rl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            totals += [float(loss.detach()), float(nll.detach()), float(rl.detach()), float(sampled_reward)]
            batches += 1
        valid_predictions = rankings_for_loader(
            model,
            valid_loader,
            api_ids,
            args.device,
            args.retrieval_gamma,
            max(args.cutoffs),
        )
        valid_metrics = ranking_metrics(valid_predictions, truth["valid"], args.cutoffs)
        row = {
            "epoch": epoch,
            "loss": float(totals[0] / batches),
            "nll": float(totals[1] / batches),
            "rl": float(totals[2] / batches),
            "sample_reward": float(totals[3] / batches),
            "valid_ndcg@10": float(valid_metrics["ndcg@10"]),
        }
        history.append(row)
        print(f"GSR seed={seed} " + json.dumps(row), flush=True)
        if valid_metrics["ndcg@10"] > best_score:
            best_score = float(valid_metrics["ndcg@10"])
            best_epoch = epoch
            stale = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale += 1
        if stale >= args.patience:
            break
    model.load_state_dict(best_state)
    test_predictions = rankings_for_loader(
        model,
        test_loader,
        api_ids,
        args.device,
        args.retrieval_gamma,
        max(args.cutoffs),
    )
    metrics = evaluate(
        test_predictions,
        truth["test"],
        api_ids,
        train_counts,
        args.cutoffs,
    )
    torch.save(
        {
            "model": best_state,
            "seed": seed,
            "best_epoch": best_epoch,
            "validation_ndcg@10": best_score,
            "history": history,
        },
        checkpoint,
    )
    return (
        {
            "method": "GSR_REC",
            "seed": seed,
            "best_epoch": best_epoch,
            "seconds": time.time() - started,
            **metrics,
        },
        test_predictions,
    )


def summarize(rows: list[dict], keys: Iterable[str]) -> dict:
    summary = {}
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows])
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }
    return summary


def write_report(path: Path, summary: dict, metadata: dict, cutoffs: Sequence[int]) -> None:
    lines = [
        "# GSR_REC ProgrammableWeb Baseline Report",
        "",
        f"- Split seed: `{metadata['split_seed']}`",
        f"- Training seeds: `{', '.join(map(str, metadata['seeds']))}`",
        "- Recommendation output: first-step (<SOS>) service distribution, as defined by GSR_REC.",
        "- Similar historical requirements are retrieved exclusively from the training split.",
        "- Training objective: hybrid negative log-likelihood and SCST set reward.",
        "",
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
    lines += [
        "",
        "## Reproduction scope",
        "",
        "This is a clean-room ProgrammableWeb adaptation of the published GSR method. "
        "It retains bi-GRU requirement/category encoding, attention decoding, training-only "
        "semantic retrieval, probability fusion, autoregressive teacher forcing, and SCST. "
        "The retrieval encoder is Sentence-T5 because the original BERT/GloVe artifacts are "
        "not distributed with the official repository; the recommender architecture and "
        "reported 300-dimensional trainable word-embedding capacity are retained.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args) -> dict:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_pw_data(args.data_dir)
    splits = split_mashups(data, seed=args.split_seed)
    requirements, categories = requirement_and_category_texts(data)
    links = relation_map(data.mashup_api, "MashupID", "ApiID")
    linked_ids = set(links)
    requirements = {
        mid: text for mid, text in requirements.items() if mid in linked_ids
    }
    categories = {
        mid: text for mid, text in categories.items() if mid in linked_ids
    }
    api_ids = sorted(map(int, data.mashup_api["ApiID"].unique()))
    api_to_index = {api: index for index, api in enumerate(api_ids)}
    train_counts = Counter(
        api for mid in splits["train"] for api in links[int(mid)]
    )
    vocabulary = build_vocabulary(splits["train"], requirements, categories)
    retrieved, similarity, retrieval_metadata = compute_retrieval(
        requirements,
        splits,
        args.retrieval_encoder,
        args.device,
        args.whitening_dim,
        output_dir / "retrieval_cache.json",
    )
    encoded_requirement = {
        mid: encode_text(text, vocabulary, args.max_requirement_length)
        for mid, text in requirements.items()
    }
    encoded_category = {
        mid: encode_text(text, vocabulary, args.max_category_length)
        for mid, text in categories.items()
    }
    eos_id = len(api_ids)
    features = {}
    for mid in sorted(requirements):
        historical = retrieved[mid]
        ordered = sorted(
            (api_to_index[api] for api in links[mid]),
            key=lambda index: (-train_counts[api_ids[index]], api_ids[index]),
        )
        target_set = list(dict.fromkeys(ordered))
        target = target_set[: args.max_generated_services - 1] + [eos_id]
        features[mid] = {
            "requirement": encoded_requirement[mid],
            "category": encoded_category[mid],
            "retrieved_requirement": encoded_requirement[historical],
            "retrieved_category": encoded_category[historical],
            "similarity": similarity[mid],
            "target": target,
            "target_set": target_set,
        }
    truth = {
        split: {int(mid): list(map(int, links[int(mid)])) for mid in ids}
        for split, ids in splits.items()
    }
    rows, predictions = [], {}
    for seed in args.seeds:
        row, ranking = train_seed(
            int(seed),
            features,
            splits,
            truth,
            api_ids,
            train_counts,
            len(vocabulary),
            output_dir,
            args,
        )
        rows.append(row)
        predictions[str(seed)] = {str(k): v for k, v in ranking.items()}
    keys = [
        f"{metric}@{k}"
        for k in args.cutoffs
        for metric in ("precision", "recall", "ndcg", "coverage", "tail_recall")
    ]
    summary = summarize(rows, keys)
    with (output_dir / "per_seed_metrics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "seed",
                *keys,
                "queries",
                "tail_queries",
                "best_epoch",
                "seconds",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "method": "GSR_REC",
        "split_seed": args.split_seed,
        "seeds": list(map(int, args.seeds)),
        "cutoffs": list(map(int, args.cutoffs)),
        "parameters": {
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "hidden_size": args.hidden_size,
            "word_dim": args.word_dim,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "nll_weight": args.nll_weight,
            "retrieval_gamma": args.retrieval_gamma,
            "max_generated_services": args.max_generated_services,
        },
        "retrieval": retrieval_metadata,
        "best_epochs": [row["best_epoch"] for row in rows],
    }
    result = {"metadata": metadata, "metrics": summary}
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "test_top20_predictions.json").write_text(
        json.dumps(predictions), encoding="utf-8"
    )
    write_report(output_dir / "REPORT.md", summary, metadata, args.cutoffs)
    print(f"GSR experiment complete: {output_dir}", flush=True)
    return result


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the GSR_REC PW baseline")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--retrieval-encoder", required=True)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43, 71, 101])
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[5, 10, 15, 20])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-size", type=int, default=200)
    parser.add_argument("--word-dim", type=int, default=300)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max-requirement-length", type=int, default=80)
    parser.add_argument("--max-category-length", type=int, default=20)
    parser.add_argument("--max-generated-services", type=int, default=10)
    parser.add_argument("--nll-weight", type=float, default=0.2)
    parser.add_argument("--retrieval-gamma", type=float, default=1.0)
    parser.add_argument("--whitening-dim", type=int, default=256)
    return parser


if __name__ == "__main__":
    run(parser().parse_args())

