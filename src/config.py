from pathlib import Path
import torch
import os
from dotenv import load_dotenv

load_dotenv()

# --------------------
# General
# --------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0     # easier to handle for reproducibility
SEED = 51

# --------------------
# General experiment
# --------------------


# --------------------
# Paths
# --------------------
PROJECT_ROOT = Path(
    os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[1])
)

# Dataset root (can be outside the repo)
DATA_DIR = Path(
    os.getenv("DATA_DIR", PROJECT_ROOT / "data")
)

# subpaths
PLACES_DATASET_DIR = DATA_DIR / "Inpainting/places2/val_256"
#JAGUAR_DATASET_DIR = DATA_DIR / "jaguars"
#EXPORTS_DIR = PROJECT_ROOT / "exports"
#CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

# Ensure directories exist (safe)
#EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
#CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)