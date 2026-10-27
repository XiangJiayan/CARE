"""One-click PyCharm entry point for the DIGER-PW baseline."""
from __future__ import annotations

import subprocess
import sys

import config


def main() -> None:
    command = [
        sys.executable,
        str(config.BASELINE_DIR / "run_diger_pw.py"),
        "--workspace", str(config.WORKSPACE_DIR),
        "--output-dir", str(config.OUTPUT_DIR),
        "--split-seed", str(config.SPLIT_SEED),
        "--seeds", *map(str, config.TRAIN_SEEDS),
        "--cutoffs", *map(str, config.CUTOFFS),
        "--device", config.DEVICE,
        "--generator-lr", str(config.GENERATOR_LR),
        "--tokenizer-lr", str(config.TOKENIZER_LR),
        "--memory-tokens", str(config.GENERATOR_MEMORY_TOKENS),
        "--dropout", str(config.GENERATOR_DROPOUT),
        "--gumbel-temperature", str(config.GUMBEL_TEMPERATURE),
        "--frequency-ratio", str(config.FREQUENCY_RATIO),
        "--usage-ema", str(config.USAGE_EMA),
        "--reconstruction-weight", str(config.RECONSTRUCTION_WEIGHT),
        "--quantization-weight", str(config.QUANTIZATION_WEIGHT),
        "--training-steps", str(config.TRAINING_STEPS),
        "--min-steps", str(config.MIN_STEPS),
        "--eval-every", str(config.EVAL_EVERY),
        "--sid-refresh-every", str(config.SID_REFRESH_EVERY),
        "--batch-size", str(config.BATCH_SIZE),
        "--patience", str(config.PATIENCE),
        "--score-query-batch", str(config.SCORE_QUERY_BATCH),
        "--score-api-batch", str(config.SCORE_API_BATCH),
    ]
    print("Starting DIGER-PW experiment:", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=config.BASELINE_DIR)


if __name__ == "__main__":
    main()
