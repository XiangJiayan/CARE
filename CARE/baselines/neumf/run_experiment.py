"""One-click PyCharm entry point for NeuMF."""
from config import *
from run_neumf import run


if __name__ == "__main__":
    params = {
        "gmf_dim": GMF_DIM, "mlp_dim": MLP_DIM, "layers": list(MLP_LAYERS),
        "dropout": DROPOUT, "lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
        "negative_ratio": NEGATIVE_RATIO, "max_epochs": MAX_EPOCHS,
        "eval_every": EVAL_EVERY, "patience": PATIENCE, "batch_size": BATCH_SIZE,
    }
    run(WORKSPACE_DIR, OUTPUT_DIR, SPLIT_SEED, TRAIN_SEEDS, CUTOFFS, params, DEVICE)

