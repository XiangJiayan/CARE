from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PWData:
    categories: pd.DataFrame
    mashups: pd.DataFrame
    apis: pd.DataFrame
    mashup_api: pd.DataFrame
    mashup_category: pd.DataFrame
    api_category: pd.DataFrame


FILES = {
    "categories": ("category.csv", ("ID", "Name")),
    "mashups": ("mashup.csv", ("ID", "Name", "Description")),
    "apis": ("api.csv", ("ID", "Name", "Description")),
    "mashup_api": ("mashupapi.csv", ("MashupID", "ApiID")),
    "mashup_category": ("mashupcate.csv", ("MashupID", "CateID")),
    "api_category": ("apicate.csv", ("ApiID", "CateID")),
}


def _read_csv(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    frame = frame.loc[:, list(columns)].copy()
    for column in columns:
        if column.endswith("ID") or column == "ID":
            frame[column] = pd.to_numeric(frame[column], errors="raise").astype("int64")
    return frame


def load_pw_data(data_dir: str | Path, validate: bool = True) -> PWData:
    root = Path(data_dir)
    values = {
        name: _read_csv(root / filename, columns)
        for name, (filename, columns) in FILES.items()
    }
    data = PWData(**values)
    if validate:
        validate_pw_data(data)
    return data


def validate_pw_data(data: PWData) -> None:
    for name, frame in (
        ("category", data.categories),
        ("mashup", data.mashups),
        ("api", data.apis),
    ):
        if frame["ID"].duplicated().any():
            raise ValueError(f"Duplicate IDs found in {name}.csv")

    checks = (
        (data.mashup_api, "MashupID", data.mashups, "mashupapi.csv"),
        (data.mashup_api, "ApiID", data.apis, "mashupapi.csv"),
        (data.mashup_category, "MashupID", data.mashups, "mashupcate.csv"),
        (data.mashup_category, "CateID", data.categories, "mashupcate.csv"),
        (data.api_category, "ApiID", data.apis, "apicate.csv"),
        (data.api_category, "CateID", data.categories, "apicate.csv"),
    )
    for relations, column, entities, filename in checks:
        invalid = ~relations[column].isin(entities["ID"])
        if invalid.any():
            examples = relations.loc[invalid, column].head().tolist()
            raise ValueError(f"Invalid {column} values in {filename}: {examples}")

    for filename, relations in (
        ("mashupapi.csv", data.mashup_api),
        ("mashupcate.csv", data.mashup_category),
        ("apicate.csv", data.api_category),
    ):
        if relations.duplicated().any():
            raise ValueError(f"Duplicate relations found in {filename}")


def relation_map(frame: pd.DataFrame, source: str, target: str) -> Dict[int, List[int]]:
    grouped = frame.groupby(source, sort=True)[target].apply(list)
    return {int(key): [int(value) for value in values] for key, values in grouped.items()}


def _category_names(data: PWData, relations: pd.DataFrame, entity_column: str) -> Dict[int, List[str]]:
    names = data.categories.set_index("ID")["Name"].fillna("").astype(str).to_dict()
    mapping = relation_map(relations, entity_column, "CateID")
    return {
        entity_id: sorted({names[cate_id].strip() for cate_id in category_ids if names[cate_id].strip()})
        for entity_id, category_ids in mapping.items()
    }


def _clean(value: object) -> str:
    if pd.isna(value):
        return ""
    normalized = unicodedata.normalize("NFKC", str(value))
    return " ".join(normalized.replace("\\n", " ").split())


def build_api_texts(data: PWData) -> Dict[int, str]:
    category_names = _category_names(data, data.api_category, "ApiID")
    texts: Dict[int, str] = {}
    for row in data.apis.itertuples(index=False):
        parts = [f"API name: {_clean(row.Name)}"]
        description = _clean(row.Description)
        if description:
            parts.append(f"Description: {description}")
        categories = category_names.get(int(row.ID), [])
        if categories:
            parts.append("Categories: " + ", ".join(categories))
        texts[int(row.ID)] = " ".join(parts)
    return texts


def build_mashup_texts(data: PWData) -> Dict[int, str]:
    category_names = _category_names(data, data.mashup_category, "MashupID")
    texts: Dict[int, str] = {}
    for row in data.mashups.itertuples(index=False):
        parts = [f"Mashup name: {_clean(row.Name)}"]
        description = _clean(row.Description)
        if description:
            parts.append(f"Description: {description}")
        categories = category_names.get(int(row.ID), [])
        if categories:
            parts.append("Categories: " + ", ".join(categories))
        texts[int(row.ID)] = " ".join(parts)
    return texts


def split_mashups(
    data: PWData,
    seed: int = 42,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
) -> Dict[str, List[int]]:
    if not 0 < train_ratio < 1 or not 0 <= valid_ratio < 1:
        raise ValueError("train_ratio and valid_ratio must be valid fractions")
    if train_ratio + valid_ratio >= 1:
        raise ValueError("train_ratio + valid_ratio must be below 1")
    ids = np.array(sorted(data.mashup_api["MashupID"].unique()), dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    train_end = int(len(ids) * train_ratio)
    valid_end = train_end + int(len(ids) * valid_ratio)
    return {
        "train": ids[:train_end].astype(int).tolist(),
        "valid": ids[train_end:valid_end].astype(int).tolist(),
        "test": ids[valid_end:].astype(int).tolist(),
    }


def save_splits(splits: Mapping[str, Iterable[int]], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    serializable = {name: [int(value) for value in values] for name, values in splits.items()}
    output.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def load_splits(path: str | Path) -> Dict[str, List[int]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {name: [int(value) for value in values] for name, values in raw.items()}


def audit_summary(data: PWData) -> dict:
    counts = data.mashup_api.groupby("MashupID").size()
    api_counts = data.mashup_api.groupby("ApiID").size()
    return {
        "mashups": len(data.mashups),
        "apis": len(data.apis),
        "categories": len(data.categories),
        "mashup_api_edges": len(data.mashup_api),
        "mashups_with_api": int(counts.size),
        "mashups_without_api": int(len(data.mashups) - counts.size),
        "mashups_with_multiple_apis": int((counts >= 2).sum()),
        "mean_apis_per_linked_mashup": float(counts.mean()),
        "max_apis_per_mashup": int(counts.max()),
        "mean_mashups_per_api": float(api_counts.mean()),
        "max_mashups_per_api": int(api_counts.max()),
        "missing_mashup_descriptions": int(data.mashups["Description"].isna().sum()),
        "missing_api_descriptions": int(data.apis["Description"].isna().sum()),
    }
