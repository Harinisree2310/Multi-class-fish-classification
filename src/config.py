# src/config.py
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Dataset (your structure)
DATA_ROOT = os.path.join(BASE_DIR, "dataset", "data")
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR = os.path.join(DATA_ROOT, "val")
TEST_DIR = os.path.join(DATA_ROOT, "test")

# Models and metadata
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Where we store the single chosen best model for deployment
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.h5")
# File with class names (one per line, order must match training)
CLASS_NAMES_FILE = os.path.join(MODELS_DIR, "class_names.txt")

# Training hyperparams
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 8   # change to 15-30 for real runs
LEARNING_RATE = 1e-4

# List of transfer backbones to try
BACKBONES = [ "MobileNetV2", "InceptionV3"]