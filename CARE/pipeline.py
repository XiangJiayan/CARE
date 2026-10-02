"""Standalone CARE pipeline for ProgrammableWeb."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor"))

import numpy as np
import pandas as pd
import torch
from transformers import Adafactor
from models import SIDVocabulary, SemanticGenerator, generate
from retrieval_models import SemanticRetriever, fused_score
from care.data import build_api_texts, load_pw_data, relation_map, split_mashups
from care.io_utils import dump, load_checkpoint, read, save_checkpoint, seed_all
from care.metrics import catalog_coverage, ranking_metrics
from care.rqvae import RQVAE, RQVAEConfig
from care.semantic_ids import make_unique_codes


VARIANTS = {"strict": False}


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def clean(value):
    if pd.isna(value):
        return ""
    return " ".join(unicodedata.normalize("NFKC", str(value)).replace("\\n", " ").split())


def compile_api_patterns(api_frame):
    patterns = []
    names = sorted({clean(value) for value in api_frame["Name"] if clean(value)}, key=lambda x: (-len(x), x.casefold()))
    for name in names:
        if len(name) < 3:
            continue
        left = r"(?<![A-Za-z0-9])" if name[0].isalnum() else ""
        right = r"(?![A-Za-z0-9])" if name[-1].isalnum() else ""
        patterns.append((name, re.compile(left + re.escape(name) + right, re.IGNORECASE)))
    return patterns


def mask_catalog_names(text, patterns):
    matched = []
    for name, pattern in patterns:
        text, count = pattern.subn(" service ", text)
        if count:
            matched.extend([name] * count)
    return " ".join(text.split()), matched


def category_map(data):
    names = data.categories.set_index("ID")["Name"].fillna("").astype(str).to_dict()
    links = relation_map(data.mashup_category, "MashupID", "CateID")
    return {mid: sorted({clean(names[cid]) for cid in values if clean(names[cid])}) for mid, values in links.items()}


def build_clean_texts(data, links, splits):
    patterns = compile_api_patterns(data.apis)
    api_names = data.apis.set_index("ID")["Name"].fillna("").astype(str).to_dict()
    categories = category_map(data)
    texts = {name: {} for name in VARIANTS}
    affected, occurrences, target_edges = 0, 0, 0
    affected_by_split = Counter()
    split_of = {mid: split for split, mids in splits.items() for mid in mids}
    examples = []
    for row in data.mashups.itertuples(index=False):
        mid = int(row.ID)
        original_name, original_description = clean(row.Name), clean(row.Description)
        masked_name, found_name = mask_catalog_names(original_name, patterns)
        masked_description, found_description = mask_catalog_names(original_description, patterns)
        found = found_name + found_description
        if found:
            affected += 1
            occurrences += len(found)
            if mid in split_of:
                affected_by_split[split_of[mid]] += 1
            if len(examples) < 20:
                examples.append({"mashup_id": mid, "before": f"{original_name} {original_description}",
                                 "after": f"{masked_name} {masked_description}", "matched_catalog_names": found})
        raw_combined = f"{original_name} {original_description}".casefold()
        for api in links.get(mid, []):
            name = clean(api_names[api]).casefold()
            if len(name) >= 3 and name in raw_combined:
                target_edges += 1
        for variant, keep_categories in VARIANTS.items():
            parts = [f"Mashup name: {masked_name or 'service mashup'}"]
            if masked_description:
                parts.append(f"Description: {masked_description}")
            if keep_categories and categories.get(mid):
                parts.append("Categories: " + ", ".join(categories[mid]))
            texts[variant][str(mid)] = " ".join(parts)
    audit = {"mask_rule": "replace every catalog API name (length >=3, case-insensitive bounded literal) in Mashup name and description with 'service'",
             "uses_target_labels_for_masking": False, "catalog_patterns": len(patterns),
             "mashups_affected": affected, "mask_occurrences": occurrences,
             "affected_linked_mashups_by_split": dict(affected_by_split),
             "target_edges_with_literal_name_before_masking": target_edges,
             "examples": examples}
    return texts, audit


def prepare(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "prepared.json").exists():
        print("PREPARE already complete", flush=True)
        return
    data = load_pw_data(args.data_dir)
    links = relation_map(data.mashup_api, "MashupID", "ApiID")
    splits = split_mashups(data, seed=args.split_seed)
    texts, audit = build_clean_texts(data, links, splits)
    api_texts = build_api_texts(data)
    api_meta = {str(int(row.ID)): {"name": clean(row.Name), "description": clean(row.Description)}
                for row in data.apis.itertuples(index=False)}
    folder = output / "strict"
    folder.mkdir(exist_ok=True)
    dump(folder / "mashup_texts.json", texts["strict"])
    csv_hash = {name: digest(Path(args.data_dir) / name) for name in
                ("category.csv", "mashup.csv", "api.csv", "mashupapi.csv", "mashupcate.csv", "apicate.csv")}
    dump(output / "prepared.json", {"api_texts": api_texts, "api_metadata": api_meta,
         "links": links, "splits": splits, "fingerprints": csv_hash,
         "split_seed": args.split_seed, "train_seed": args.seed,
         "cleaning": "strict_catalog_name_masking_no_mashup_categories"})
    dump(output / "cleaning_audit.json", audit)
    dump(output / "run_config.json", vars(args))
    print("CLEANING_AUDIT", json.dumps({key: value for key, value in audit.items() if key != "examples"}), flush=True)


def encode(args):
    output = Path(args.output_dir)
    api_target = output / "api_embeddings.pt"
    query_targets = [output / variant / "mashup_embeddings.pt" for variant in VARIANTS]
    if api_target.exists() and all(path.exists() for path in query_targets):
        print("ENCODE already complete", flush=True)
        return
    from sentence_transformers import SentenceTransformer
    prepared = read(output / "prepared.json")
    mids = sorted(map(int, read(output / "strict/mashup_texts.json")))
    model = SentenceTransformer(args.encoder, device=args.device)
    model.max_seq_length = 256
    if not api_target.exists():
        api_ids = sorted(map(int, prepared["api_texts"]))
        embedding = model.encode([prepared["api_texts"][str(api)] for api in api_ids],
                                 batch_size=args.encoder_batch, show_progress_bar=True,
                                 convert_to_tensor=True, normalize_embeddings=False).cpu().float()
        save_checkpoint(api_target, {"api_ids": torch.tensor(api_ids),
                        "embeddings": embedding, "encoder": args.encoder})
        print("ENCODED api", tuple(embedding.shape), flush=True)
    for variant in VARIANTS:
        target = output / variant / "mashup_embeddings.pt"
        if target.exists():
            continue
        texts = read(output / variant / "mashup_texts.json")
        embedding = model.encode([texts[str(mid)] for mid in mids], batch_size=32, show_progress_bar=True,
                                 convert_to_tensor=True, normalize_embeddings=False).cpu().float()
        save_checkpoint(target, {"mashup_ids": torch.tensor(mids), "embeddings": embedding,
                                 "encoder": args.encoder, "variant": variant})
        print("ENCODED", variant, tuple(embedding.shape), flush=True)


def train_rqvae(args):
    output = Path(args.output_dir)
    if (output / "content_semantic_ids.json").exists():
        print("RQ-VAE already complete", flush=True)
        return
    seed_all(args.seed)
    pack = load_checkpoint(output / "api_embeddings.pt")
    values = pack["embeddings"].float()
    config = RQVAEConfig(
        input_dim=values.shape[1],
        num_levels=args.rq_levels,
        codebook_size=args.rq_codebook_size,
    )
    model = RQVAE(config).to(args.device)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.rq_lr,
                                    initial_accumulator_value=args.rq_initial_accumulator)
    rng = np.random.default_rng(args.seed + 101)
    best, best_state, history = float("inf"), None, []
    for epoch in range(1, args.rq_epochs + 1):
        order = rng.permutation(len(values))
        losses = []
        for start in range(0, len(order), args.rq_batch):
            batch = values[order[start:start + args.rq_batch]].to(args.device)
            optimizer.zero_grad(set_to_none=True)
            total, reconstruction, quantization, _ = model(batch, initialize=(epoch == 1 and start == 0))
            loss = reconstruction * values.shape[1] + quantization * 32.0
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch % args.rq_eval_every == 0 or epoch == args.rq_epochs:
            model.eval()
            with torch.no_grad():
                total, reconstruction, quantization, codes = model(values.to(args.device))
            row = {"epoch": epoch, "loss": float(np.mean(losses)),
                   "reconstruction": float(reconstruction), "quantization": float(quantization),
                   "unique_base_codes": len({tuple(x) for x in codes.cpu().tolist()})}
            history.append(row)
            print("RQ-VAE", json.dumps(row), flush=True)
            if row["reconstruction"] < best:
                best = row["reconstruction"]
                best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    with torch.no_grad():
        base_codes = model.encode_codes(values.to(args.device)).cpu().tolist()
    unique = make_unique_codes(pack["api_ids"].tolist(), base_codes)
    dump(output / "content_semantic_ids.json", unique)
    save_checkpoint(output / "rqvae_best.pt", {"model": best_state, "config": config.to_dict(),
                    "history": history, "best_reconstruction": best})
    assert len({tuple(code) for code in unique.values()}) == len(unique)


class Data:
    def __init__(self, args, variant):
        self.root, self.variant = Path(args.output_dir), variant
        self.prepared = read(self.root / "prepared.json")
        apack = load_checkpoint(self.root / "api_embeddings.pt")
        qpack = load_checkpoint(self.root / variant / "mashup_embeddings.pt")
        self.api_ids = list(map(int, apack["api_ids"].tolist()))
        self.ai = {api: index for index, api in enumerate(self.api_ids)}
        self.api_vectors = apack["embeddings"].float()
        self.query = {int(mid): vector.float() for mid, vector in zip(qpack["mashup_ids"], qpack["embeddings"])}
        self.links = {int(mid): list(map(int, values)) for mid, values in self.prepared["links"].items()}
        self.splits = {name: list(map(int, values)) for name, values in self.prepared["splits"].items()}
        self.counts = Counter(api for mid in self.splits["train"] for api in self.links[mid])
        self.count_array = np.asarray([self.counts[api] for api in self.api_ids], dtype=np.float32)
        codes = {int(key): value for key, value in read(self.root / "content_semantic_ids.json").items()}
        self.vocab = SIDVocabulary(codes)

    def truth(self, split):
        return {mid: self.links[mid] for mid in self.splits[split]}

    def incidence(self, mids):
        target = torch.zeros((len(mids), len(self.api_ids)), dtype=torch.bool)
        for row, mid in enumerate(mids):
            target[row, [self.ai[api] for api in self.links[mid]]] = True
        return target

    def labels(self, apis):
        return torch.tensor([self.vocab.paths[api] + [1] for api in apis], dtype=torch.long)


def rankings(score, mids, api_ids, k=20):
    order = np.argsort(-np.asarray(score), axis=1, kind="stable")[:, :k]
    return {int(mid): [int(api_ids[index]) for index in row] for mid, row in zip(mids, order)}


def train_generator(args, variant):
    folder = Path(args.output_dir) / variant
    if (folder / "generator_best.pt").exists():
        return
    seed_all(args.seed)
    data = Data(args, variant)
    model = SemanticGenerator(data.vocab.state["size"], memory_tokens=args.generator_memory_tokens,
                              dropout_rate=args.generator_dropout).to(args.device)
    optimizer = Adafactor(model.parameters(), lr=args.generator_lr, relative_step=False,
                          scale_parameter=False, warmup_init=False)
    edges = [(mid, api) for mid in data.splits["train"] for api in data.links[mid]]
    api_labels = data.labels(data.api_ids)
    valid = data.splits["valid"]
    queries, truth = {mid: data.query[mid] for mid in valid}, data.truth("valid")
    rng = np.random.default_rng(args.seed + 202)
    history, best, best_state, start_time = [], -1.0, None, time.time()
    running = []
    for step in range(1, args.generator_steps + 1):
        chosen = rng.integers(len(edges), size=args.generator_batch)
        rows = [edges[index] for index in chosen]
        query = torch.stack([data.query[mid] for mid, _ in rows]).to(args.device)
        labels = data.labels([api for _, api in rows]).to(args.device)
        positive = torch.ones(len(rows), dtype=torch.bool, device=args.device)
        weight = torch.ones(len(rows), device=args.device)
        catalog_indices = rng.integers(len(data.api_ids), size=args.catalog_batch)
        catalog_query = data.api_vectors[catalog_indices].to(args.device)
        catalog_labels = api_labels[catalog_indices].to(args.device)
        catalog_positive = torch.ones(len(catalog_indices), dtype=torch.bool, device=args.device)
        catalog_weight = torch.ones(len(catalog_indices), device=args.device)

        optimizer.zero_grad(set_to_none=True)
        lr = args.generator_lr * min(1.0, math.sqrt(1000 / max(1, step)))
        for group in optimizer.param_groups:
            group["lr"] = lr
        interaction_loss, generation, _, _ = model.loss(
            query, labels, positive, [], weight, 0.0
        )
        catalog_loss, catalog_generation, _, _ = model.loss(
            catalog_query, catalog_labels, catalog_positive, [], catalog_weight, 0.0
        )
        loss = interaction_loss + args.catalog_weight * catalog_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running.append(float(generation.detach()))
        if step % args.generator_eval_every == 0 or step == args.generator_steps:
            prediction = generate(model, data.vocab, queries, args.device, batch_size=args.eval_batch)
            metric = ranking_metrics(prediction, truth)
            row = {"step": step, "loss": float(np.mean(running)),
                   "catalog_loss": float(catalog_generation.detach()), "validation": metric,
                   "seconds": time.time() - start_time}
            running = []
            history.append(row)
            print("GENERATOR", variant, json.dumps(row), flush=True)
            if metric["ndcg@10"] > best:
                best = metric["ndcg@10"]
                best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                save_checkpoint(folder / "generator_best.pt", {"model": best_state, "vocabulary": data.vocab.state,
                                "step": step, "validation": metric,
                                "catalog_weight": args.catalog_weight})
    dump(folder / "generator_history.json", history)


@torch.no_grad()
def retrieval_scores(model, query, api, device, batch=128):
    model.eval()
    encoded_api, _ = model.encode_api(api.to(device))
    values = []
    for start in range(0, len(query), batch):
        encoded_query, _ = model.encode_query(query[start:start + batch].to(device))
        values.append((encoded_query @ encoded_api.T).cpu())
    return torch.cat(values).numpy()


def train_retriever(args, variant):
    folder = Path(args.output_dir) / variant
    if (folder / "retriever_best.pt").exists():
        return
    seed_all(args.seed + 1)
    data = Data(args, variant)
    model = SemanticRetriever(bottleneck=args.retriever_bottleneck).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.retriever_lr, weight_decay=1e-4)
    train, valid = data.splits["train"], data.splits["valid"]
    validation_query = torch.stack([data.query[mid] for mid in valid])
    weight = torch.tensor(1.0 / np.power(1.0 + data.count_array, args.retriever_gamma),
                          device=args.device)
    api = data.api_vectors.to(args.device)
    rng = np.random.default_rng(args.seed + 811)
    best, state, history = -1.0, None, []
    for step in range(1, args.retriever_steps + 1):
        mids = rng.choice(train, size=args.retriever_batch, replace=True).tolist()
        query = torch.stack([data.query[mid] for mid in mids]).to(args.device)
        positive = data.incidence(mids).to(args.device)
        optimizer.zero_grad(set_to_none=True)
        loss, alignment, anchor, temperature = model.loss(
            query, api, positive, weight, anchor_weight=args.retriever_anchor_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % args.retriever_eval_every == 0 or step == args.retriever_steps:
            score = retrieval_scores(model, validation_query, data.api_vectors, args.device)
            metric = ranking_metrics(rankings(score, valid, data.api_ids), data.truth("valid"))
            row = {"step": step, "loss": float(loss.detach()), "temperature": float(temperature.detach()),
                   "validation": metric}
            history.append(row)
            print("RETRIEVER", variant, json.dumps(row), flush=True)
            if metric["ndcg@10"] > best:
                best = metric["ndcg@10"]
                state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    save_checkpoint(folder / "retriever_best.pt", {"model": state, "history": history, "best": best})


def train(args):
    for variant in VARIANTS:
        train_generator(args, variant)
        train_retriever(args, variant)


@torch.no_grad()
def generator_scores(model, vocab, query, api_ids, device, qbatch=8, abatch=192):
    labels = torch.tensor([vocab.paths[api] + [1] for api in api_ids], dtype=torch.long)
    result = []
    model.eval()
    for qstart in range(0, len(query), qbatch):
        q = query[qstart:qstart + qbatch].to(device)
        parts = []
        for astart in range(0, len(labels), abatch):
            target = labels[astart:astart + abatch].to(device)
            nq, na = len(q), len(target)
            repeated_q = q[:, None, :].expand(nq, na, -1).reshape(nq * na, -1)
            repeated_target = target[None, :, :].expand(nq, na, -1).reshape(nq * na, -1)
            value, _ = model.sequence_scores(repeated_q, repeated_target)
            parts.append(value.reshape(nq, na).cpu())
        result.append(torch.cat(parts, 1))
    return torch.cat(result).numpy()


def score(args):
    for variant in VARIANTS:
        folder = Path(args.output_dir) / variant
        if (folder / "scores.pt").exists():
            continue
        data = Data(args, variant)
        generator_checkpoint = load_checkpoint(folder / "generator_best.pt")
        generator = SemanticGenerator(data.vocab.state["size"], memory_tokens=args.generator_memory_tokens,
                                      dropout_rate=args.generator_dropout).to(args.device)
        generator.load_state_dict(generator_checkpoint["model"])
        retriever = SemanticRetriever(bottleneck=args.retriever_bottleneck).to(args.device)
        retriever.load_state_dict(load_checkpoint(folder / "retriever_best.pt")["model"])
        payload = {}
        for split in ("valid", "test"):
            mids = data.splits[split]
            query = torch.stack([data.query[mid] for mid in mids])
            aligned = retrieval_scores(retriever, query, data.api_vectors, args.device)
            generated = generator_scores(generator, data.vocab, query, data.api_ids, args.device,
                                         args.score_query_batch, args.score_api_batch)
            payload[split] = {"mids": mids, "aligned": torch.from_numpy(aligned),
                              "generator": torch.from_numpy(generated)}
            print("SCORED", variant, split, flush=True)
        save_checkpoint(folder / "scores.pt", payload)


def metrics(score, mids, data, split):
    return ranking_metrics(rankings(score, mids, data.api_ids), data.truth(split))


def select(args):
    for variant in VARIANTS:
        folder = Path(args.output_dir) / variant
        if (folder / "fusion_config.json").exists():
            continue
        data = Data(args, variant)
        matrix = load_checkpoint(folder / "scores.pt")["valid"]
        mids, generated, aligned = matrix["mids"], matrix["generator"].numpy(), matrix["aligned"].numpy()
        care_validation = metrics(
            fused_score(generated, aligned, args.fusion_alpha), mids, data, "valid"
        )
        config = {
            "method": "CARE",
            "alpha": args.fusion_alpha,
            "validation": care_validation,
            "selection": "fixed configuration; test data not used",
        }
        dump(folder / "fusion_config.json", config)
        print("SELECTED", variant, json.dumps(config), flush=True)


def group_metrics(prediction, truth, data):
    overall = ranking_metrics(prediction, truth)
    overall.update(catalog_coverage(prediction, len(data.api_ids)))
    seen = sorted((api for api in data.api_ids if data.counts[api]), key=lambda api: (-data.counts[api], api))
    head = set(seen[:math.ceil(0.2 * len(seen))])
    tail = set(seen) - head
    tail_truth = {mid: [api for api in values if api in tail] for mid, values in truth.items()}
    tail_truth = {mid: values for mid, values in tail_truth.items() if values}
    tail_all = ranking_metrics({mid: prediction[mid] for mid in tail_truth}, tail_truth)
    tail_recall = {f"recall@{k}": tail_all[f"recall@{k}"] for k in (5, 10, 15, 20)}
    tail_recall["queries_with_target"] = len(tail_truth)
    return {"overall": overall, "tail_recall": tail_recall,
            "tail_definition": "training-seen APIs outside the top 20% by training frequency"}


def evaluate(args):
    output = Path(args.output_dir)
    if (output / "test_results.json").exists():
        result = read(output / "test_results.json")
        data = Data(args, "strict")
        truth = {int(mid): list(map(int, values)) for mid, values in result["truth"].items()}
        for name, payload in result["methods"].items():
            prediction = {int(mid): list(map(int, values)) for mid, values in payload["predictions"].items()}
            payload["metrics"] = group_metrics(prediction, truth, data)
        result["metric_protocol"] = {
            "cutoffs": [5, 10, 15, 20],
            "overall": ["precision", "recall", "ndcg"],
            "catalog": ["coverage"],
            "long_tail": "tail_recall",
            "tail_definition": "training-seen APIs outside the top 20% by training frequency",
        }
        dump(output / "test_results.json", result)
        print("EVALUATE metrics refreshed for K=5,10,15,20", flush=True)
        return
    methods, selected = {}, {}
    for variant in VARIANTS:
        data = Data(args, variant)
        matrix = load_checkpoint(output / variant / "scores.pt")["test"]
        config = read(output / variant / "fusion_config.json")
        mids = matrix["mids"]
        generated, aligned = matrix["generator"].numpy(), matrix["aligned"].numpy()
        care_score = fused_score(generated, aligned, config["alpha"])
        truth = data.truth("test")
        name = f"{variant}_care"
        prediction = rankings(care_score, mids, data.api_ids)
        methods[name] = {"metrics": group_metrics(prediction, truth, data), "predictions": prediction}
        print("TEST", name, json.dumps(methods[name]["metrics"]["overall"]), flush=True)
        selected[variant] = name
    truth = Data(args, "strict").truth("test")
    dump(output / "test_results.json", {"methods": methods, "truth": truth, "selected_by_validation": selected,
         "metric_protocol": {"cutoffs": [5, 10, 15, 20],
             "overall": ["precision", "recall", "ndcg"], "catalog": ["coverage"],
             "long_tail": "tail_recall",
             "tail_definition": "training-seen APIs outside the top 20% by training frequency"},
         "scope": "CARE trained from raw ProgrammableWeb CSV files and Sentence-T5 embeddings",
         "environment": {"python": platform.python_version(), "torch": torch.__version__,
                         "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"}})


def report(args):
    output = Path(args.output_dir)
    prepared = read(output / "prepared.json")
    current = {name: digest(Path(args.data_dir) / name) for name in prepared["fingerprints"]}
    if current != prepared["fingerprints"]:
        raise AssertionError("CSV changed")
    result = read(output / "test_results.json")
    truth = {int(mid): values for mid, values in result["truth"].items()}
    catalog_size = len(Data(args, "strict").api_ids)
    for name, payload in result["methods"].items():
        prediction = {int(mid): values for mid, values in payload["predictions"].items()}
        recomputed = ranking_metrics(prediction, truth)
        recomputed.update(catalog_coverage(prediction, catalog_size))
        for metric, value in recomputed.items():
            if abs(value - payload["metrics"]["overall"][metric]) > 1e-10:
                raise AssertionError((name, metric))
    strict_name = result["selected_by_validation"]["strict"]
    dump(output / "verification.json", {"passed": True, "csv_unchanged": True,
         "metrics_recomputed": True, "selected_method": strict_name})
    audit = read(output / "cleaning_audit.json")
    metric = result["methods"][strict_name]["metrics"]
    overall = metric["overall"]
    lines = ["# CARE ProgrammableWeb Experiment Report", "",
             f"清洗规则：{audit['mask_rule']}。不使用目标标签；原始 CSV 未修改。", "",
             f"共修改 {audit['mashups_affected']} 个 Mashup、替换 {audit['mask_occurrences']} 处目录 API 名称；原始关联中有 {audit['target_edges_with_literal_name_before_masking']} 条直接包含目标 API 名称。", "",
             f"训练随机种子：{args.seed}；数据划分种子：{args.split_seed}。", "",
             "| K | Precision | Recall | NDCG | Coverage | Tail Recall |",
             "|---:|---:|---:|---:|---:|---:|"]
    tail = metric["tail_recall"]
    for k in (5, 10, 15, 20):
        lines.append(f"| {k} | {overall[f'precision@{k}']:.6f} | {overall[f'recall@{k}']:.6f} | "
                     f"{overall[f'ndcg@{k}']:.6f} | {overall[f'coverage@{k}']:.2%} | "
                     f"{tail[f'recall@{k}']:.6f} |")
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("VERIFIED", strict_name, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run", "prepare", "encode", "rqvae", "train", "score", "select", "evaluate", "report"])
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--encoder", default="work/models/sentence-t5-base")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--encoder-batch", type=int, default=32)
    parser.add_argument("--rq-epochs", type=int, default=20000)
    parser.add_argument("--rq-lr", type=float, default=.4)
    parser.add_argument("--rq-initial-accumulator", type=float, default=10.)
    parser.add_argument("--rq-batch", type=int, default=1024)
    parser.add_argument("--rq-eval-every", type=int, default=500)
    parser.add_argument("--rq-levels", type=int, default=4)
    parser.add_argument("--rq-codebook-size", type=int, default=512)
    parser.add_argument("--generator-steps", type=int, default=4000)
    parser.add_argument("--generator-batch", type=int, default=64)
    parser.add_argument("--generator-lr", type=float, default=0.001)
    parser.add_argument("--generator-memory-tokens", type=int, default=8)
    parser.add_argument("--generator-dropout", type=float, default=0.1)
    parser.add_argument("--generator-eval-every", type=int, default=500)
    parser.add_argument("--catalog-weight", type=float, default=0.5)
    parser.add_argument("--catalog-batch", type=int, default=64)
    parser.add_argument("--eval-batch", type=int, default=16)
    parser.add_argument("--retriever-steps", type=int, default=2000)
    parser.add_argument("--retriever-batch", type=int, default=128)
    parser.add_argument("--retriever-lr", type=float, default=3e-4)
    parser.add_argument("--retriever-bottleneck", type=int, default=32)
    parser.add_argument("--retriever-gamma", type=float, default=0.0)
    parser.add_argument("--retriever-anchor-weight", type=float, default=0.001)
    parser.add_argument("--retriever-eval-every", type=int, default=200)
    parser.add_argument("--fusion-alpha", type=float, default=0.6)
    parser.add_argument("--score-query-batch", type=int, default=8)
    parser.add_argument("--score-api-batch", type=int, default=192)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    actions = {"prepare": lambda: prepare(args), "encode": lambda: encode(args), "rqvae": lambda: train_rqvae(args), "train": lambda: train(args),
               "score": lambda: score(args), "select": lambda: select(args), "evaluate": lambda: evaluate(args),
               "report": lambda: report(args)}
    if args.command == "run":
        for stage in ("prepare", "encode", "rqvae", "train", "score", "select", "evaluate", "report"):
            print("STAGE", stage, flush=True)
            actions[stage]()
    else:
        actions[args.command]()


if __name__ == "__main__":
    main()
