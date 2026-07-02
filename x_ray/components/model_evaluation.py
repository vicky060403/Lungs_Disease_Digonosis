import sys
from typing import Tuple

import torch
from torch.nn import CrossEntropyLoss, Module
from torch.utils.data import DataLoader

from x_ray.entity.artifacts_entity import (
    DataTransformationArtifact,
    ModelEvaluationArtifact,
    ModelTrainerArtifact,
)
from x_ray.entity.config_entity import ModelEvaluationConfig
from x_ray.exception import XRayException
from x_ray.logger import logging


class ModelEvaluation:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_evaluation_config: ModelEvaluationConfig,
        model_trainer_artifact: ModelTrainerArtifact,
    ):

        self.data_transformation_artifact = data_transformation_artifact
        self.model_evaluation_config = model_evaluation_config
        self.model_trainer_artifact = model_trainer_artifact

    def configuration(self) -> Tuple[DataLoader, Module, Module]:
        """
        Configure the model evaluation process.
        Loads the trained model, loss function and test dataloader.
        """

        logging.info("Entered configuration method of ModelEvaluation.")

        try:
            test_dataloader = (
                self.data_transformation_artifact.transformed_test_object
            )

            model: Module = torch.load(
                self.model_trainer_artifact.trained_model_path,
                map_location=self.model_evaluation_config.device,
            )

            model.to(self.model_evaluation_config.device)
            model.eval()

            criterion = CrossEntropyLoss()

            logging.info("Exited configuration method.")

            return test_dataloader, model, criterion

        except Exception as e:
            raise XRayException(e, sys)

    def test_net(self) -> float:
        """
        Evaluate the trained model on the test dataset.
        Returns the overall test accuracy.
        """

        logging.info("Entered test_net method.")

        try:

            test_dataloader, model, criterion = self.configuration()

            test_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():

                for batch_idx, (images, labels) in enumerate(test_dataloader):

                    images = images.to(self.model_evaluation_config.device)
                    labels = labels.to(self.model_evaluation_config.device)

                    outputs = model(images)

                    loss = criterion(outputs, labels)

                    test_loss += loss.item()

                    _, predictions = torch.max(outputs, 1)

                    correct += (predictions == labels).sum().item()

                    total += labels.size(0)

                    logging.info(
                        f"Batch {batch_idx + 1}/{len(test_dataloader)} | "
                        f"Loss: {loss.item():.4f}"
                    )

            avg_test_loss = test_loss / len(test_dataloader)

            accuracy = 100 * correct / total

            logging.info(
                f"Average Test Loss: {avg_test_loss:.4f}"
            )

            logging.info(
                f"Test Accuracy: {accuracy:.2f}%"
            )

            logging.info("Exited test_net method.")

            return accuracy

        except Exception as e:
            raise XRayException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Initiates model evaluation and returns the evaluation artifact.
        """

        logging.info(
            "Entered initiate_model_evaluation method."
        )

        try:

            accuracy = self.test_net()

            model_evaluation_artifact = ModelEvaluationArtifact(
                model_accuracy=accuracy
            )

            logging.info(
                "Exited initiate_model_evaluation method."
            )

            return model_evaluation_artifact

        except Exception as e:
            raise XRayException(e, sys)