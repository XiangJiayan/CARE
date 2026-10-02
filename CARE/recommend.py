"""使用训练好的 CARE，为一条新 Mashup 需求推荐 API。"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

import config
from models import SemanticGenerator
from pipeline import Data, generator_scores, rankings
from retrieval_models import SemanticRetriever, fused_score
from care.io_utils import load_checkpoint, read


def mask_api_names(text: str, metadata: dict) -> str:
    names = sorted({value["name"] for value in metadata.values() if len(value["name"]) >= 3},
                   key=lambda value: (-len(value), value.casefold()))
    for name in names:
        left = r"(?<![A-Za-z0-9])" if name[0].isalnum() else ""
        right = r"(?![A-Za-z0-9])" if name[-1].isalnum() else ""
        text = re.sub(left + re.escape(name) + right, " service ", text, flags=re.IGNORECASE)
    return " ".join(text.split())


@torch.no_grad()
def recommend(text: str, run_dir: Path, top_k: int = 20):
    run = Path(run_dir)
    prepared = read(run / "prepared.json")
    cleaned = mask_api_names(text, prepared["api_metadata"])
    encoder = SentenceTransformer(str(config.SENTENCE_T5_MODEL), device=config.DEVICE)
    encoder.max_seq_length = 256
    query = encoder.encode([f"Mashup name: service mashup Description: {cleaned}"],
                           convert_to_tensor=True, normalize_embeddings=False).cpu().float()

    args = SimpleNamespace(output_dir=str(run))
    data = Data(args, "strict")
    generator = SemanticGenerator(
        data.vocab.state["size"],
        memory_tokens=config.GENERATOR_MEMORY_TOKENS,
        dropout_rate=config.GENERATOR_DROPOUT,
    ).to(config.DEVICE)
    generator.load_state_dict(load_checkpoint(run / "strict/generator_best.pt")["model"])
    retriever = SemanticRetriever(bottleneck=config.RETRIEVER_BOTTLENECK).to(config.DEVICE)
    retriever.load_state_dict(load_checkpoint(run / "strict/retriever_best.pt")["model"])
    retriever.eval()
    encoded_query, _ = retriever.encode_query(query.to(config.DEVICE))
    encoded_api, _ = retriever.encode_api(data.api_vectors.to(config.DEVICE))
    aligned = (encoded_query @ encoded_api.T).cpu().numpy()
    generated = generator_scores(generator, data.vocab, query, data.api_ids, config.DEVICE)

    fusion = read(run / "strict/fusion_config.json")
    selected = fusion["method"]
    alpha = float(fusion["alpha"])
    final = fused_score(generated, aligned, alpha)

    order = np.argsort(-final[0], kind="stable")[:top_k]
    result = []
    for rank, index in enumerate(order, 1):
        api = data.api_ids[int(index)]
        meta = prepared["api_metadata"][str(api)]
        result.append({"rank": rank, "api_id": api, "name": meta["name"],
                       "description": meta["description"], "score": float(final[0, index])})
    return cleaned, selected, alpha, result


def main():
    parser = argparse.ArgumentParser(description="CARE 新 Mashup API 推荐")
    parser.add_argument("--text", help="新 Mashup 的英文需求描述")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--run-dir", type=Path,
                        default=config.OUTPUT_ROOT / f"seed{config.TRAIN_SEEDS[0]}")
    values = parser.parse_args()
    text = values.text or input("请输入新 Mashup 的英文需求描述：\n> ").strip()
    if not text:
        sys.exit("需求描述不能为空。")
    cleaned, selected, alpha, rows = recommend(text, values.run_dir, values.top_k)
    print(f"\n严格清洗后的输入：{cleaned}")
    print(f"验证集选择的方法：{selected}；生成分支权重 alpha={alpha:.4f}\n")
    for row in rows:
        description = row["description"][:120].replace("\n", " ")
        print(f"{row['rank']:>2}. [{row['api_id']}] {row['name']}  score={row['score']:.4f}")
        if description:
            print(f"    {description}")


if __name__ == "__main__":
    main()
