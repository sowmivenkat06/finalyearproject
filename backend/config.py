import torch
import os

# ==========================
# Dataset
# ==========================

DATASET_PATH = "dataset"

# ==========================
# Models
# ==========================

MODEL_PATH = "models"

# ==========================
# Outputs
# ==========================

OUTPUT_PATH = "outputs"

CHECKPOINT_PATH = "checkpoints"

FUSION_PATH = "fusion"

# ==========================
# Classes
# ==========================

CLASS_NAMES = [

    "Dyskeratotic",
    "Koilocytotic",
    "Metaplastic",
    "Parabasal",
    "Superficial-Intermediate"

]

NUM_CLASSES = len(CLASS_NAMES)

# ==========================
# Image Sizes
# ==========================

IMAGE_SIZE = 224

INCEPTION_SIZE = 299

# ==========================
# Training
# ==========================

BATCH_SIZE = 32

LEARNING_RATE = 0.0001

EPOCHS = 20

# ==========================
# Device
# ==========================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using Device :", DEVICE)

# ==========================
# Create Output Folders
# ==========================

os.makedirs(OUTPUT_PATH, exist_ok=True)

os.makedirs(CHECKPOINT_PATH, exist_ok=True)

os.makedirs(FUSION_PATH, exist_ok=True)