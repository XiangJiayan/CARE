"""One-click PyCharm entry point for ST5-TIGER."""
import subprocess
import sys
import config


def main() -> None:
    command = [sys.executable, str(config.BASELINE_DIR / "run_st5_tiger.py"),
               "--workspace", str(config.WORKSPACE_DIR),
               "--output-dir", str(config.OUTPUT_DIR),
               "--split-seed", str(config.SPLIT_SEED),
               "--seeds", *map(str, config.TRAIN_SEEDS),
               "--cutoffs", *map(str, config.CUTOFFS),
               "--device", config.DEVICE,
               "--generator-lr", str(config.GENERATOR_LR),
               "--generator-memory-tokens", str(config.GENERATOR_MEMORY_TOKENS),
               "--generator-dropout", str(config.GENERATOR_DROPOUT),
               "--generator-steps", str(config.GENERATOR_STEPS),
               "--generator-min-steps", str(config.GENERATOR_MIN_STEPS),
               "--generator-eval-every", str(config.GENERATOR_EVAL_EVERY),
               "--generator-batch", str(config.GENERATOR_BATCH),
               "--eval-batch", str(config.EVAL_BATCH),
               "--patience", str(config.PATIENCE),
               "--score-query-batch", str(config.SCORE_QUERY_BATCH),
               "--score-api-batch", str(config.SCORE_API_BATCH)]
    print("Starting ST5-TIGER experiment:", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=config.BASELINE_DIR)


if __name__ == "__main__":
    main()
