"""Fixed CARE configuration used by the paper experiments."""
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("CARE_DATA_DIR", PROJECT_DIR / "data" / "programmableweb"))
SENTENCE_T5_MODEL = os.environ.get(
    "CARE_SENTENCE_ENCODER", "sentence-transformers/sentence-t5-base"
)
OUTPUT_ROOT = PROJECT_DIR / "outputs" / "care" / "split2026"
DEVICE = os.environ.get("CARE_DEVICE", "cuda")

SPLIT_SEED = 2026
TRAIN_SEEDS = (17, 29, 43, 71, 101)
CUTOFFS = (5, 10, 15, 20)

RQ_EPOCHS = 20_000
RQ_LEVELS = 4
RQ_CODEBOOK_SIZE = 512

GENERATOR_STEPS = 4_000
GENERATOR_LR = 0.001
GENERATOR_MEMORY_TOKENS = 8
GENERATOR_DROPOUT = 0.1
CATALOG_LOSS_WEIGHT = 0.5
CATALOG_BATCH_SIZE = 64

RETRIEVER_STEPS = 2_000
RETRIEVER_LR = 0.0003
RETRIEVER_BOTTLENECK = 32
RETRIEVER_GAMMA = 0.0
RETRIEVER_ANCHOR_WEIGHT = 0.001

FUSION_ALPHA = 0.6
