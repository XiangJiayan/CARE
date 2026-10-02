from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


def make_unique_codes(api_ids: Sequence[int], base_codes: Sequence[Sequence[int]]) -> Dict[int, List[int]]:
    if len(api_ids) != len(base_codes):
        raise ValueError("api_ids and base_codes must have identical lengths")
    groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for api_id, code in zip(api_ids, base_codes):
        groups[tuple(int(value) for value in code)].append(int(api_id))
    result: Dict[int, List[int]] = {}
    for base_code, members in groups.items():
        for suffix, api_id in enumerate(sorted(members)):
            result[api_id] = [*base_code, suffix]
    return result


def semantic_token(position: int, code: int) -> str:
    return f"<sid_{position}_{code}>"


def code_to_tokens(code: Sequence[int]) -> List[str]:
    return [semantic_token(position, int(value)) for position, value in enumerate(code)]


def all_semantic_tokens(codes: Mapping[int, Sequence[int]]) -> List[str]:
    return sorted({token for code in codes.values() for token in code_to_tokens(code)})


def save_semantic_ids(
    codes: Mapping[int, Sequence[int]],
    output_path: str | Path,
    metadata: Mapping[str, object] | None = None,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": dict(metadata or {}),
        "codes": {str(api_id): [int(value) for value in code] for api_id, code in codes.items()},
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_semantic_ids(path: str | Path) -> tuple[Dict[int, List[int]], dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    codes = {int(api_id): [int(value) for value in code] for api_id, code in payload["codes"].items()}
    return codes, dict(payload.get("metadata", {}))


class SemanticTrie:
    """Prefix tree used by Hugging Face constrained beam search."""

    END = "__end__"

    def __init__(self, token_paths: Mapping[int, Sequence[int]], eos_token_id: int) -> None:
        self.root: dict = {}
        self.eos_token_id = int(eos_token_id)
        for path in token_paths.values():
            node = self.root
            for token_id in path:
                node = node.setdefault(int(token_id), {})
            node[self.END] = True

    def allowed(self, prefix: Sequence[int]) -> List[int]:
        node = self.root
        for token_id in prefix:
            token_id = int(token_id)
            if token_id not in node:
                return [self.eos_token_id]
            node = node[token_id]
        allowed = [key for key in node if key != self.END]
        if self.END in node:
            allowed.append(self.eos_token_id)
        return [int(value) for value in allowed]
