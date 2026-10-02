"""Run the five-seed CARE experiment with the fixed paper configuration."""
import subprocess
import sys

import config


def main() -> None:
    for seed in config.TRAIN_SEEDS:
        run_dir = config.OUTPUT_ROOT / f"seed{seed}"
        command = [
            sys.executable, str(config.PROJECT_DIR / "pipeline.py"), "run",
            "--data-dir", str(config.DATA_DIR),
            "--output-dir", str(run_dir),
            "--encoder", config.SENTENCE_T5_MODEL,
            "--device", config.DEVICE,
            "--split-seed", str(config.SPLIT_SEED),
            "--seed", str(seed),
            "--rq-epochs", str(config.RQ_EPOCHS),
            "--rq-levels", str(config.RQ_LEVELS),
            "--rq-codebook-size", str(config.RQ_CODEBOOK_SIZE),
            "--generator-steps", str(config.GENERATOR_STEPS),
            "--generator-lr", str(config.GENERATOR_LR),
            "--generator-memory-tokens", str(config.GENERATOR_MEMORY_TOKENS),
            "--generator-dropout", str(config.GENERATOR_DROPOUT),
            "--catalog-weight", str(config.CATALOG_LOSS_WEIGHT),
            "--catalog-batch", str(config.CATALOG_BATCH_SIZE),
            "--retriever-steps", str(config.RETRIEVER_STEPS),
            "--retriever-lr", str(config.RETRIEVER_LR),
            "--retriever-bottleneck", str(config.RETRIEVER_BOTTLENECK),
            "--retriever-gamma", str(config.RETRIEVER_GAMMA),
            "--retriever-anchor-weight", str(config.RETRIEVER_ANCHOR_WEIGHT),
            "--fusion-alpha", str(config.FUSION_ALPHA),
        ]
        print(f"Starting CARE seed={seed}", flush=True)
        subprocess.run(command, check=True, cwd=config.PROJECT_DIR)


if __name__ == "__main__":
    main()
