"""TSSGCF baseline configuration for one-click execution."""
import os
from pathlib import Path

BASELINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASELINE_DIR.parent.parent
WORKSPACE_DIR = PROJECT_ROOT / "outputs" / "care" / "split2026" / "seed17"
DATA_DIR = PROJECT_ROOT / "data" / "programmableweb"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "baselines" / "tssgcf" / "split2026"
MINILM_MODEL = os.environ.get(
    "TSSGCF_SENTENCE_ENCODER", "sentence-transformers/all-MiniLM-L6-v2"
)

SPLIT_SEED = 2026
TRAIN_SEEDS = (17, 29, 43, 71, 101)
CUTOFFS = (5, 10, 15, 20)

EMBEDDING_DIM = 256
PROPAGATION_LAYERS = 5
SIMILARITY_THRESHOLD = 0.34
TEXT_FUSION_WEIGHT = 0.5
TSS_WEIGHT = 0.25
TSS_BETA = 1.0
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EXPLICIT_L2_WEIGHT = 1e-5
BATCH_SIZE = 16384
MAX_EPOCHS = 300
EVAL_EVERY = 10
PATIENCE = 8
TSS_SAMPLE_SIZE = 1024
DEVICE = "cuda"
