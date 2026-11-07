
import os
from pathlib import Path


# --------------------------------------------------
# PROJECT ROOT
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent


# --------------------------------------------------
# APPLICATION SETTINGS
# --------------------------------------------------

class Config:

    SECRET_KEY = os.environ.get(
        "EDGEDEPTH_SECRET_KEY",
        "edge-depth-development-key-change-before-deployment"
    )

    DEBUG = False

    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100 MB

    JSON_SORT_KEYS = False


# --------------------------------------------------
# DIRECTORY CONFIGURATION
# --------------------------------------------------

class Paths:

    BACKEND_DIR = BASE_DIR / "backend"
    CLASSICAL_DIR = BASE_DIR / "classical"
    MODELS_DIR = BASE_DIR / "models"
    CHECKPOINT_DIR = MODELS_DIR / "checkpoints"
    TRAINING_DIR = BASE_DIR / "training"

    DATASETS_DIR = BASE_DIR / "datasets"

    NYU_DIR = DATASETS_DIR / "NYU_Depth_V2"
    KITTI_DIR = DATASETS_DIR / "KITTI"
    MIDDLEBURY_DIR = DATASETS_DIR / "Middlebury"

    UPLOAD_DIR = BASE_DIR / "uploads"
    OUTPUT_DIR = BASE_DIR / "outputs"

    TEMPLATE_DIR = BASE_DIR / "templates"
    STATIC_DIR = BASE_DIR / "static"

    CSS_DIR = STATIC_DIR / "css"
    JS_DIR = STATIC_DIR / "js"
    IMAGE_DIR = STATIC_DIR / "images"

    # Trained model checkpoints
    DEPTH_MODEL = CHECKPOINT_DIR / "edgedepth_best.pth"


# --------------------------------------------------
# SUPPORTED FILE FORMATS
# --------------------------------------------------

SUPPORTED_RGB_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff"
}

SUPPORTED_DEPTH_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".pfm",
    ".npy"
}


# --------------------------------------------------
# RECONSTRUCTION SETTINGS
# --------------------------------------------------

UPSAMPLING_FACTORS = [2, 4, 8]

RECONSTRUCTION_METHODS = [
    "nearest",
    "bilinear",
    "bicubic",
    "joint_bilateral",
    "guided",
    "deep_learning",
    "hybrid"
]


# --------------------------------------------------
# DEPTH PROCESSING SETTINGS
# --------------------------------------------------

DEFAULT_DEPTH_SCALE = 1.0

MIN_DEPTH = 0.0

DEFAULT_GUIDED_RADIUS = 8

DEFAULT_GUIDED_EPSILON = 0.001

DEFAULT_BILATERAL_DIAMETER = 9

DEFAULT_NOISE_REDUCTION = True

DEFAULT_HOLE_FILLING = True

DEFAULT_EDGE_PRESERVATION = True

DEFAULT_BOUNDARY_REFINEMENT = True


# --------------------------------------------------
# MODEL SETTINGS
# --------------------------------------------------

MODEL_INPUT_CHANNELS = 5

# RGB: 3 channels
# Low-resolution depth: 1 channel
# Validity mask: 1 channel

MODEL_OUTPUT_CHANNELS = 1

MODEL_AVAILABLE = Paths.DEPTH_MODEL.is_file()


# --------------------------------------------------
# DEVICE CONFIGURATION
# --------------------------------------------------

def get_device():

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"

    except ImportError:
        pass

    return "cpu"


DEVICE = get_device()


# --------------------------------------------------
# CREATE REQUIRED DIRECTORIES
# --------------------------------------------------

def create_directories():

    directories = [
        Paths.BACKEND_DIR,
        Paths.CLASSICAL_DIR,
        Paths.MODELS_DIR,
        Paths.CHECKPOINT_DIR,
        Paths.TRAINING_DIR,
        Paths.DATASETS_DIR,
        Paths.NYU_DIR,
        Paths.KITTI_DIR,
        Paths.MIDDLEBURY_DIR,
        Paths.UPLOAD_DIR,
        Paths.OUTPUT_DIR,
        Paths.TEMPLATE_DIR,
        Paths.STATIC_DIR,
        Paths.CSS_DIR,
        Paths.JS_DIR,
        Paths.IMAGE_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":

    create_directories()

    print("EdgeDepth AI Configuration")
    print("--------------------------")

    print("Project:", BASE_DIR)
    print("Device:", DEVICE)
    print("Model checkpoint:", Paths.DEPTH_MODEL)
    print("Model available:", MODEL_AVAILABLE)

    print("Supported factors:", UPSAMPLING_FACTORS)

    print("Directories initialized successfully.")
