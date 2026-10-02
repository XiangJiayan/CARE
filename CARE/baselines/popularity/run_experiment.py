"""One-click PyCharm entry point for the Popularity baseline."""
from config import CUTOFFS, DATA_DIR, OUTPUT_DIR, SPLIT_SEED, TRAIN_SEEDS
from run_popularity import run


if __name__ == "__main__":
    run(DATA_DIR, OUTPUT_DIR, SPLIT_SEED, TRAIN_SEEDS, CUTOFFS)

