"""One-click PyCharm entry point for Sentence-T5 Direct."""
from config import CUTOFFS, OUTPUT_DIR, REPORTING_SEEDS, SPLIT_SEED, WORKSPACE_DIR
from run_sentence_t5_direct import run


if __name__ == "__main__":
    run(WORKSPACE_DIR, OUTPUT_DIR, SPLIT_SEED, REPORTING_SEEDS, CUTOFFS)

