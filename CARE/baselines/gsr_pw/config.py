"""Portable configuration for the GSR ProgrammableWeb baseline."""
from __future__ import annotations

import os
from pathlib import Path


BASELINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASELINE_DIR.parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent.parent

DATA_DIR = Path(
    os.environ.get("CARE_DATA_DIR", PROJECT_ROOT / "data" / "programmableweb")
)
OUTPUT_DIR = Path(
    os.environ.get(
        "CARE_RESULT_DIR",
        PROJECT_ROOT.parent / "CARE_experiment_results" / "baselines" / "gsr_pw" / "split2026",
    )
)

_local_encoder = WORKSPACE_ROOT / "work" / "models" / "sentence-t5-base"
RETRIEVAL_ENCODER = os.environ.get(
    "CARE_SENTENCE_ENCODER",
    str(_local_encoder) if _local_encoder.exists() else "sentence-transformers/sentence-t5-base",
)

SPLIT_SEED = 2026
TRAIN_SEEDS = (17, 29, 43, 71, 101)
CUTOFFS = (5, 10, 15, 20)
DEVICE = os.environ.get("CARE_DEVICE", "cuda")

# Settings reported in Table II of the GSR paper.
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
HIDDEN_SIZE = 200
WORD_DIM = 300
DROPOUT = 0.2
BATCH_SIZE = 128
EPOCHS = 30
MAX_REQUIREMENT_LENGTH = 80
MAX_CATEGORY_LENGTH = 20
MAX_GENERATED_SERVICES = 10

# Best hybrid-loss setting reported by the paper.
NLL_WEIGHT = 0.2
RETRIEVAL_GAMMA = 1.0
WHITENING_DIM = 256
PATIENCE = 5

