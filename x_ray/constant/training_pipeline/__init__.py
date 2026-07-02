# Instead of hardcoding values throughout the project, we can store them in one place and import them wherever needed.

from datetime import datetime
from typing import List
import torch

# Generates a timestamp to use for various purpose like project tracking, monitoring, store logs with timestamp etc
TIMESTAMP: datetime = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

# Data Ingestion Constants: Stores outputs of pipeline stages.
ARTIFACT_DIR: str = "artifacts"

# AWS S3 Configuration : Used by our S3 utility.
BUCKET_NAME: str = "lungxray"
S3_DATA_FOLDER: str = "data"

# Class Labels
CLASS_LABEL_1: str = "NORMAL"
CLASS_LABEL_2: str = "PNEUMONIA"

# Data Augmentation Parameters
BRIGHTNESS: int = 0.10
CONTRAST: int = 0.1
SATURATION: int = 0.10
HUE: int = 0.1
# Input Image Size
RESIZE: int = 224
# Extracts the center region.
CENTERCROP: int = 224 
RANDOMROTATION: int = 10
# Normalization of images using mean and std
NORMALIZE_LIST_1: List[int] = [0.485, 0.456, 0.406] # mean
NORMALIZE_LIST_2: List[int] = [0.229, 0.224, 0.225] # std

# Transform File Names
TRAIN_TRANSFORMS_KEY: str = "xray_train_transforms"
TRAIN_TRANSFORMS_FILE: str = "train_transforms.pkl"
TEST_TRANSFORMS_FILE: str = "test_transforms.pkl"

# DataLoader Parameters
BATCH_SIZE: int = 16
SHUFFLE: bool = True # for training
PIN_MEMORY: bool = True # Speeds up CPU → GPU transfer.

# Model Training Constants: Model Storage
TRAINED_MODEL_DIR: str = "trained_model"
TRAINED_MODEL_NAME: str = "model.pt"

# Device slection for training
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Learning Rate Scheduler
# # StepLR
# STEP_SIZE: int = 6 # Reduce LR every 6 epochs.
# GAMMA: int = 0.5 # Multiply LR by 0.5.

# ReduceLROnPlateau
LR_FACTOR: float = 0.5
LR_PATIENCE: int = 2
MIN_LR: float = 1e-6
EPOCH: int = 10

# BentoML Configuration: used for mlops
BENTOML_MODEL_NAME: str = "xray_model" # Name used when saving model into BentoML.
BENTOML_SERVICE_NAME: str = "xray_service" # API service name.
BENTOML_ECR_IMAGE: str = "xray_bento_image" # Docker image name for deployment.

# Prediction Labels: used after training
PREDICTION_LABEL: dict = {0: CLASS_LABEL_1, 1: CLASS_LABEL_2}

# AWS ECR Configuration: used for mlops
AWS_ACCOUNT_ID: str = "851236939228"
AWS_REGION: str = "ap-southeast-2"
