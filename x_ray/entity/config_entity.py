import os
from dataclasses import dataclass
from torch import device
from x_ray.constant.training_pipeline import *


@dataclass
class DataIngestionConfig: # Stores paths related to downloading data.
    def __init__(self):
        self.s3_data_folder: str = S3_DATA_FOLDER

        self.bucket_name: str = BUCKET_NAME

        self.artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)

        self.data_path: str = os.path.join(
            self.artifact_dir, "data_ingestion", self.s3_data_folder
        )

        self.train_data_path: str = os.path.join(self.data_path, "train")

        self.test_data_path: str = os.path.join(self.data_path, "test")


@dataclass
class DataTransformationConfig: # stores the datatransfomation hard coded value
    def __init__(self):
        # use ** to unpack the dictionary and pass the values as keyword arguments to the ColorJitter transform.
        self.color_jitter_transforms: dict = {
            "brightness": BRIGHTNESS,
            "contrast": CONTRAST,
            "saturation": SATURATION,
            "hue": HUE,
        }

        self.RESIZE: int = RESIZE

        self.CENTERCROP: int = CENTERCROP

        self.RANDOMROTATION: int = RANDOMROTATION
        # Use ** to unpack the dictionary and pass the values as keyword arguments to the Normalize transform.
        self.normalize_transforms: dict = {
            "mean": NORMALIZE_LIST_1,
            "std": NORMALIZE_LIST_2,
        }

        self.train_data_loader_params: dict = {
            "batch_size": BATCH_SIZE,
            "shuffle": SHUFFLE,
            "pin_memory": PIN_MEMORY,
        }

        self.test_data_loader_params: dict = {
            "batch_size": BATCH_SIZE,
            "shuffle": False,
            "pin_memory": PIN_MEMORY,
        }

        self.artifact_dir: str = os.path.join(
            ARTIFACT_DIR, TIMESTAMP, "data_transformation"
        )

        self.train_transforms_file: str = os.path.join(
            self.artifact_dir, TRAIN_TRANSFORMS_FILE
        )

        self.test_transforms_file: str = os.path.join(
            self.artifact_dir, TEST_TRANSFORMS_FILE
        )


@dataclass
class ModelTrainerConfig:
    def __init__(self):
        self.artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP, "model_training")

        self.trained_bentoml_model_name: str = "xray_model"

        self.trained_model_path: str = os.path.join(
            self.artifact_dir, TRAINED_MODEL_NAME
        )

        self.train_transforms_key: str = TRAIN_TRANSFORMS_KEY

        self.epochs: int = EPOCH

        self.optimizer_params: dict = {"lr": 0.001}

        self.scheduler_params: dict = {
            "mode": "min",
            "factor": LR_FACTOR,
            "patience": LR_PATIENCE,
            "min_lr": MIN_LR
        }

        self.device: device = DEVICE


@dataclass
class ModelEvaluationConfig:
    def __init__(self):
        self.device: device = DEVICE
        
        self.optimizer_params: dict = {"lr": 0.001}


# Model Pusher Configurations
@dataclass
class ModelPusherConfig: # Used when pushing model to BentoML or Docker.
    def __init__(self):
        self.bentoml_model_name: str = BENTOML_MODEL_NAME

        self.bentoml_service_name: str = BENTOML_SERVICE_NAME

        self.train_transforms_key: str = TRAIN_TRANSFORMS_KEY

        self.bentoml_ecr_image: str = BENTOML_ECR_IMAGE
