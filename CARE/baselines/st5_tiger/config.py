"""ST5-TIGER generator-only baseline configuration."""
from pathlib import Path

BASELINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASELINE_DIR.parent.parent
WORKSPACE_DIR = PROJECT_ROOT / "outputs" / "care" / "split2026" / "seed17"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "baselines" / "st5_tiger" / "split2026"

SPLIT_SEED = 2026
TRAIN_SEEDS = (17, 29, 43, 71, 101)
CUTOFFS = (5, 10, 15, 20)
DEVICE = "cuda"

GENERATOR_LR = 0.001
GENERATOR_MEMORY_TOKENS = 8
GENERATOR_DROPOUT = 0.1

GENERATOR_STEPS = 4000
GENERATOR_MIN_STEPS = 2000
GENERATOR_EVAL_EVERY = 500
GENERATOR_BATCH = 64
EVAL_BATCH = 16
PATIENCE = 2
SCORE_QUERY_BATCH = 8
SCORE_API_BATCH = 192
