"""NeuMF baseline configuration for one-click PyCharm execution."""
from pathlib import Path

BASELINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASELINE_DIR.parent.parent
WORKSPACE_DIR = PROJECT_ROOT / "outputs" / "care" / "split2026" / "seed17"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "baselines" / "neumf" / "split2026"

SPLIT_SEED = 2026
TRAIN_SEEDS = (17, 29, 43, 71, 101)
CUTOFFS = (5, 10, 15, 20)

GMF_DIM = 32
MLP_DIM = 32
MLP_LAYERS = (64, 32, 16)
DROPOUT = 0.1
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
NEGATIVE_RATIO = 4
MAX_EPOCHS = 100
EVAL_EVERY = 5
PATIENCE = 8
BATCH_SIZE = 2048
DEVICE = "cuda"
