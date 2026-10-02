"""One-click PyCharm entry point for the TF-IDF baseline."""
from config import (
    CUTOFFS,
    DATA_DIR,
    MAX_FEATURES,
    NGRAM_RANGE,
    OUTPUT_DIR,
    REPORTING_SEEDS,
    SPLIT_SEED,
    STOP_WORDS,
    SUBLINEAR_TF,
)
from run_tfidf import run


if __name__ == "__main__":
    run(DATA_DIR, OUTPUT_DIR, SPLIT_SEED, REPORTING_SEEDS, CUTOFFS,
        MAX_FEATURES, NGRAM_RANGE, STOP_WORDS, SUBLINEAR_TF)

