"""LightGCN baseline configuration for one-click PyCharm execution."""
from pathlib import Path

BASELINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASELINE_DIR.parent.parent
WORKSPACE_DIR = PROJECT_ROOT / "outputs" / "care" / "split2026" / "seed17"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "baselines" / "lightgcn" / "split2026"

SPLIT_SEED = 2026
TRAIN_SEEDS = (17, 29, 43, 71, 101)
CUTOFFS = (5, 10, 15, 20)

EMBEDDING_DIM = 64
PROPAGATION_LAYERS = 3
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 300
EVAL_EVERY = 10
PATIENCE = 8
DEVICE = "cuda"
