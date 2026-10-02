"""One-click PyCharm entry point for MPGCF."""
import subprocess
import sys

import config


def main() -> None:
    command = [
        sys.executable, str(config.BASELINE_DIR / "run_mpgcf.py"),
        "--workspace", str(config.WORKSPACE_DIR),
        "--output-dir", str(config.OUTPUT_DIR),
        "--split-seed", str(config.SPLIT_SEED),
        "--seeds", *map(str, config.TRAIN_SEEDS),
        "--cutoffs", *map(str, config.CUTOFFS),
        "--device", config.DEVICE,
    ]
    print("Starting MPGCF experiment:", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=config.BASELINE_DIR)


if __name__ == "__main__":
    main()

