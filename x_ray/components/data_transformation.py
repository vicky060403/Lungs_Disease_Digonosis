import os
import sys
from typing import Tuple

import joblib
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from x_ray.entity.artifacts_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
)
from x_ray.entity.config_entity import DataTransformationConfig
from x_ray.exception import XRayException
from x_ray.logger import logging


class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig, data_ingestion_artifact: DataIngestionArtifact):
        self.data_transformation_config = data_transformation_config

        self.data_ingestion_artifact = data_ingestion_artifact
    # transforming the training data
    def transforming_training_data(self) -> transforms.Compose:
        try:
            logging.info(
                "Entered the transforming_training_data method of Data transformation class"
            )

            train_transform: transforms.Compose = transforms.Compose([
                    # resize the image to a fixed size
                    transforms.Resize(self.data_transformation_config.RESIZE),
                    # center crop the image to a fixed size
                    transforms.CenterCrop(self.data_transformation_config.CENTERCROP),
                    # apply random color jittering to the image so that brithness, contrast, saturation and hue of the image is changed randomly
                    transforms.ColorJitter(
                        **self.data_transformation_config.color_jitter_transforms
                    ),
                    # apply random horizontal flip to the image
                    transforms.RandomHorizontalFlip(),
                    # apply random rotation to the image
                    transforms.RandomRotation(
                        self.data_transformation_config.RANDOMROTATION
                    ),
                    # convert the image to a tensor
                    transforms.ToTensor(),
                    # normalize the image using mean and std
                    transforms.Normalize(
                        **self.data_transformation_config.normalize_transforms
                    )
                ])

            logging.info(
                "Exited the transforming_training_data method of Data transformation class"
            )
            return train_transform

        except Exception as e:
            raise XRayException(e, sys)

# Here we use transforms.Compose after self -> as type hinting to indicate that the method returns a transforms.Compose object.
    # transforming the testing data
    def transforming_testing_data(self) -> transforms.Compose:
        logging.info(
            "Entered the transforming_testing_data method of Data transformation class"
        )

        try:
            test_transform: transforms.Compose = transforms.Compose([
                    # resize the image to a fixed size
                    transforms.Resize(self.data_transformation_config.RESIZE),
                    # center crop the image to a fixed size
                    transforms.CenterCrop(self.data_transformation_config.CENTERCROP),
                    # convert the image to a tensor
                    transforms.ToTensor(),
                    # normalize the image using mean and std
                    transforms.Normalize(
                        **self.data_transformation_config.normalize_transforms
                    )
                ])

            logging.info(
                "Exited the transforming_testing_data method of Data transformation class"
            )

            return test_transform

        except Exception as e:
            raise XRayException(e, sys)
    # load to the dataloader and transform the train and test data and return the train_loader and test_loader in tuple format(train_loader, test_loader)
    def data_loader(self, train_transform: transforms.Compose, test_transform: transforms.Compose) -> Tuple[DataLoader, DataLoader]:
        try:
            logging.info("Entered the data_loader method of Data transformation class")
            # load the train and test data using ImageFolder and apply the transforms to the data
            train_data: Dataset = ImageFolder(
                os.path.join(self.data_ingestion_artifact.train_file_path),
                transform=train_transform,
            )
            # load the test data using ImageFolder and apply the transforms to the data
            test_data: Dataset = ImageFolder(
                os.path.join(self.data_ingestion_artifact.test_file_path),
                transform=test_transform,
            )

            logging.info("Created train data and test data paths")

            # create the train and test dataloaders using the DataLoader class and the data_loader_params from the config file
            train_loader: DataLoader = DataLoader(
                train_data, **self.data_transformation_config.train_data_loader_params
            )

            test_loader: DataLoader = DataLoader(
                test_data, **self.data_transformation_config.test_data_loader_params
            )

            logging.info("Exited the data_loader method of Data transformation class")

            return train_loader, test_loader

        except Exception as e:
            raise XRayException(e, sys)

    # initiate_data_transformation method is responsible for initiating the data transformation process. It applies the transformations to the training and testing data, saves the transformation objects, and returns a DataTransformationArtifact containing the transformed data loaders and file paths.
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info(
                "Entered the initiate_data_transformation method of Data transformation class"
            )
            # Apply transformations to the training data 
            train_transform: transforms.Compose = self.transforming_training_data()
            # Apply transformations to the testing data
            test_transform: transforms.Compose = self.transforming_testing_data()
            # Create the artifact directory if it doesn't exist
            os.makedirs(self.data_transformation_config.artifact_dir, exist_ok=True)

            # Save the transformation objects to files using joblib
            joblib.dump(
                train_transform, self.data_transformation_config.train_transforms_file
            )

            joblib.dump(
                test_transform, self.data_transformation_config.test_transforms_file
            )

            # Load the transformed training and testing data into DataLoader objects
            train_loader, test_loader = self.data_loader(
                train_transform=train_transform, test_transform=test_transform
            )
            # Create a DataTransformationArtifact object to store the transformed data loaders and file paths
            data_transformation_artifact: DataTransformationArtifact = DataTransformationArtifact(
                transformed_train_object=train_loader,
                transformed_test_object=test_loader,
                train_transform_file_path=self.data_transformation_config.train_transforms_file,
                test_transform_file_path=self.data_transformation_config.test_transforms_file,
            )

            logging.info(
                "Exited the initiate_data_transformation method of Data transformation class"
            )

            return data_transformation_artifact

        except Exception as e:
            raise XRayException(e, sys)
