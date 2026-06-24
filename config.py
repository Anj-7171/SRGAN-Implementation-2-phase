import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# dataset
DATA_DIR = "data/DIV2K"
SCALE = 4

# training
BATCH_SIZE = 16
NUM_WORKERS = 4

PRETRAIN_EPOCHS = 50
GAN_EPOCHS = 100
LR = 1e-4

# paths
CHECKPOINT_DIR = "checkpoints"
SAMPLE_DIR = "samples"
LOG_FILE = "logs/metrics.csv"