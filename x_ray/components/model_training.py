import os
import sys

import bentoml
import joblib
import torch
from torch.nn import Module, CrossEntropyLoss
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from x_ray.constant.training_pipeline import *
from x_ray.entity.artifacts_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from x_ray.entity.config_entity import ModelTrainerConfig
from x_ray.exception import XRayException
from x_ray.logger import logging
from x_ray.ml.model.arch import Net


class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact, model_trainer_config: ModelTrainerConfig):

        self.model_trainer_config: ModelTrainerConfig = model_trainer_config

        self.data_transformation_artifact: DataTransformationArtifact = (
            data_transformation_artifact
        )
        self.criterion: CrossEntropyLoss = CrossEntropyLoss()
        self.model: Module = Net()

    def train(self, optimizer: Optimizer) -> None:
        """
        Description: To train the model

        input: model,device,train_loader,optimizer,epoch

        output: loss, batch id and accuracy
        """
        logging.info("Entered the train method of Model trainer class")

        try:
            self.model.train()

            pbar = tqdm(self.data_transformation_artifact.transformed_train_object)

            correct: int = 0

            processed = 0

            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(DEVICE), target.to(DEVICE)

                # Initialization of gradient
                optimizer.zero_grad()

                # In PyTorch, gradient is accumulated over backprop and even though thats used in RNN generally not used in CNN
                # or specific requirements
                ## prediction on data

                y_pred = self.model(data)

                # Calculating loss given the prediction
                loss = self.criterion(y_pred, target)

                # Backprop
                loss.backward()

                optimizer.step()

                # get the index of the log-probability corresponding to the max value
                pred = y_pred.argmax(dim=1, keepdim=True)

                correct += pred.eq(target.view_as(pred)).sum().item()

                processed += len(data)

                pbar.set_description(
                    desc=f"Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}"
                )

            logging.info("Exited the train method of Model trainer class")


        except Exception as e:
            raise XRayException(e, sys)

    def test(self) -> tuple[float, float]:
        """
        Evaluate the model on the test dataset.

        Returns:
            tuple:
                test_loss (float)
                test_accuracy (float)
        """

        logging.info("Entered the test method of Model trainer class")

        try:
            self.model.eval()

            test_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():

                for data, target in self.data_transformation_artifact.transformed_test_object:

                    data = data.to(self.model_trainer_config.device)
                    target = target.to(self.model_trainer_config.device)

                    output = self.model(data)

                    loss = self.criterion(output, target)

                    test_loss += loss.item()

                    _, pred = torch.max(output, 1)

                    correct += (pred == target).sum().item()

                    total += target.size(0)

            test_loss /= len(self.data_transformation_artifact.transformed_test_object)

            accuracy = 100 * correct / total

            print(
                f"Test Loss: {test_loss:.4f} | Test Accuracy: {accuracy:.2f}%"
            )

            logging.info(
                f"Test Loss: {test_loss:.4f} | Test Accuracy: {accuracy:.2f}%"
            )

            logging.info("Exited the test method of Model trainer class")

            return test_loss, accuracy

        except Exception as e:
            raise XRayException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info(
                "Entered the initiate_model_trainer method of Model trainer class"
            )

            model: Module = self.model.to(self.model_trainer_config.device)

            optimizer: Optimizer = torch.optim.Adam(
                model.parameters(), 
                **self.model_trainer_config.optimizer_params
            )

            scheduler: ReduceLROnPlateau = ReduceLROnPlateau(
                optimizer=optimizer, 
                **self.model_trainer_config.scheduler_params
            )

            best_loss = float("inf")

            for epoch in range(1, self.model_trainer_config.epochs + 1):

                print(f"\nEpoch {epoch}/{self.model_trainer_config.epochs}")

                self.train(optimizer=optimizer)

                test_loss, test_accuracy = self.test()

                scheduler.step(test_loss)

                current_lr = optimizer.param_groups[0]["lr"]

                print(f"Current Learning Rate: {current_lr:.6f}")

                if test_loss < best_loss:

                    best_loss = test_loss

                    torch.save(
                        model,
                        self.model_trainer_config.trained_model_path
                    )

                    logging.info("Best model saved.")

            os.makedirs(self.model_trainer_config.artifact_dir, exist_ok=True)

            torch.save(model, self.model_trainer_config.trained_model_path)

            train_transforms_obj = joblib.load(
                self.data_transformation_artifact.train_transform_file_path
            )

            bentoml.pytorch.save_model(
                name=BENTOML_MODEL_NAME,
                model=model,
                custom_objects={
                    TRAIN_TRANSFORMS_KEY: train_transforms_obj
                }
            )
        

            model_trainer_artifact: ModelTrainerArtifact = ModelTrainerArtifact(
                trained_model_path=self.model_trainer_config.trained_model_path
            )

            logging.info(
                "Exited the initiate_model_trainer method of Model trainer class"
            )

            return model_trainer_artifact

        except Exception as e:
            raise XRayException(e, sys)
