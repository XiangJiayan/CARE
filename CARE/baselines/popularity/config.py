"""Popularity baseline configuration for one-click PyCharm execution."""
from pathlib import Path

BASELINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASELINE_DIR.parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "programmableweb"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "baselines" / "popularity" / "split2026"

SPLIT_SEED = 2026
TRAIN_SEEDS = (17, 29, 43, 71, 101)
CUTOFFS = (5, 10, 15, 20)
