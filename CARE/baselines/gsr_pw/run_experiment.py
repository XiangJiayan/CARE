"""One-click PyCharm entry point for the GSR_REC baseline."""
import subprocess
import sys

import config


def main() -> None:
    command = [
        sys.executable,
        str(config.BASELINE_DIR / "run_gsr_pw.py"),
        "--data-dir",
        str(config.DATA_DIR),
        "--output-dir",
        str(config.OUTPUT_DIR),
        "--retrieval-encoder",
        config.RETRIEVAL_ENCODER,
        "--split-seed",
        str(config.SPLIT_SEED),
        "--seeds",
        *map(str, config.TRAIN_SEEDS),
        "--cutoffs",
        *map(str, config.CUTOFFS),
        "--device",
        config.DEVICE,
        "--learning-rate",
        str(config.LEARNING_RATE),
        "--weight-decay",
        str(config.WEIGHT_DECAY),
        "--hidden-size",
        str(config.HIDDEN_SIZE),
        "--word-dim",
        str(config.WORD_DIM),
        "--dropout",
        str(config.DROPOUT),
        "--batch-size",
        str(config.BATCH_SIZE),
        "--epochs",
        str(config.EPOCHS),
        "--patience",
        str(config.PATIENCE),
        "--max-requirement-length",
        str(config.MAX_REQUIREMENT_LENGTH),
        "--max-category-length",
        str(config.MAX_CATEGORY_LENGTH),
        "--max-generated-services",
        str(config.MAX_GENERATED_SERVICES),
        "--nll-weight",
        str(config.NLL_WEIGHT),
        "--retrieval-gamma",
        str(config.RETRIEVAL_GAMMA),
        "--whitening-dim",
        str(config.WHITENING_DIM),
    ]
    print("Starting GSR_REC experiment", flush=True)
    subprocess.run(command, check=True, cwd=config.BASELINE_DIR)


if __name__ == "__main__":
    main()

