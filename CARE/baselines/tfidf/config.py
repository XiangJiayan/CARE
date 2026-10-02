"""TF-IDF baseline configuration for one-click PyCharm execution."""
from pathlib import Path

BASELINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASELINE_DIR.parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "programmableweb"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "baselines" / "tfidf" / "split2026"

SPLIT_SEED = 2026
REPORTING_SEEDS = (17, 29, 43, 71, 101)
CUTOFFS = (5, 10, 15, 20)

MAX_FEATURES = 20_000
NGRAM_RANGE = (1, 2)
STOP_WORDS = "english"
SUBLINEAR_TF = True
