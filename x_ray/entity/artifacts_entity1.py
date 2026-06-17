# A dataclass automatically creates: __init__(), __repr__(), __eq__()
from dataclasses import dataclass
from torch.utils.data.dataloader import DataLoader


@dataclass
class DataIngestionArtifact: # Represents the output of the Data Ingestion stage.
    train_file_path: str

    test_file_path: str


@dataclass
class DataTransformationArtifact: # Output of Data Transformation stage.
    transformed_train_object: DataLoader

    transformed_test_object: DataLoader

    train_transform_file_path: str

    test_transform_file_path: str


@dataclass
class ModelTrainerArtifact: # Output of Model Training stage.
    trained_model_path: str # Stores location of trained saved model


@dataclass
class ModelEvaluationArtifact: # Output of Model evaluation stage.
    model_accuracy: float


@dataclass
class ModelPusherArtifact: # Output of deployment stage. it includes BentoML, Docker, Aws, Kubernets
    bentoml_model_name: str

    bentoml_service_name: str
