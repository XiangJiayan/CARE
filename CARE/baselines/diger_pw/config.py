"""DIGER-PW baseline configuration."""
from __future__ import annotations

import os
from pathlib import Path


BASELINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASELINE_DIR.parent.parent

_portable_workspace = PROJECT_ROOT / "outputs" / "care" / "split2026" / "seed17"
WORKSPACE_DIR = Path(
    os.environ.get(
        "CARE_WORKSPACE_DIR",
        _portable_workspace,
    )
)

# Keep experimental products outside the blind-review source-code package.
OUTPUT_DIR = Path(
    os.environ.get(
        "CARE_RESULT_DIR",
        PROJECT_ROOT.parent / "CARE_experiment_results" / "baselines" / "diger_pw" / "split2026",
    )
)

SPLIT_SEED = 2026
TRAIN_SEEDS = (17, 29, 43, 71, 101)
CUTOFFS = (5, 10, 15, 20)
DEVICE = os.environ.get("CARE_DEVICE", "cuda")

# The generator is identical in capacity to the ST5-TIGER baseline.
GENERATOR_LR = 1e-3
TOKENIZER_LR = 1e-5
GENERATOR_MEMORY_TOKENS = 8
GENERATOR_DROPOUT = 0.1

# FrqUD settings. The official paper uses tau=2.0 and dataset-dependent ratios.
GUMBEL_TEMPERATURE = 2.0
FREQUENCY_RATIO = 1.5
USAGE_EMA = 0.99

# Preserve the loss normalization used to pretrain the shared PW tokenizer.
# These convert mean losses back to the feature/latent scales used by CARE.
RECONSTRUCTION_WEIGHT = 768.0
QUANTIZATION_WEIGHT = 32.0

TRAINING_STEPS = 4000
MIN_STEPS = 2000
EVAL_EVERY = 500
SID_REFRESH_EVERY = 0
BATCH_SIZE = 64
PATIENCE = 2
SCORE_QUERY_BATCH = 8
SCORE_API_BATCH = 192
