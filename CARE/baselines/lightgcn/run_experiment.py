"""One-click PyCharm entry point for LightGCN."""
from config import *
from run_lightgcn import run


if __name__ == "__main__":
    params = {
        "dim": EMBEDDING_DIM, "layers": PROPAGATION_LAYERS,
        "lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
        "max_epochs": MAX_EPOCHS, "eval_every": EVAL_EVERY, "patience": PATIENCE,
    }
    run(WORKSPACE_DIR, OUTPUT_DIR, SPLIT_SEED, TRAIN_SEEDS, CUTOFFS, params, DEVICE)

