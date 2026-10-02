"""Sentence-T5 Direct baseline configuration for one-click PyCharm execution."""
from pathlib import Path

BASELINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASELINE_DIR.parent.parent

# Reuse the exact frozen Sentence-T5 embeddings and strict split used by the final model.
WORKSPACE_DIR = PROJECT_ROOT / "outputs" / "care" / "split2026" / "seed17"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "baselines" / "sentence_t5_direct" / "split2026"

SPLIT_SEED = 2026
REPORTING_SEEDS = (17, 29, 43, 71, 101)
CUTOFFS = (5, 10, 15, 20)
