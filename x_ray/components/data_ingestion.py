import sys

from x_ray.cloud_storage.s3_operations import S3Operation
from x_ray.constant.training_pipeline import *
from x_ray.entity.artifacts_entity import DataIngestionArtifact
# Imports the configuration class containing settings like bucket name, S3 folder, local paths
from x_ray.entity.config_entity import DataIngestionConfig
from x_ray.exception import XRayException
from x_ray.logger import logging


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config

        self.s3 = S3Operation()

    def get_data_from_s3(self) -> None:
        try:
            logging.info("Entered the get_data_from_s3 method of Data ingestion class")
            # Downloading the data from S3 bucket to local artifact directory
            self.s3.sync_folder_from_s3(
                folder=self.data_ingestion_config.data_path,
                bucket_name=self.data_ingestion_config.bucket_name,
                bucket_folder_name=self.data_ingestion_config.s3_data_folder,
            )

            logging.info("Exited the get_data_from_s3 method of Data ingestion class")

        except Exception as e:
            raise XRayException(e, sys)
        
    # This method initiates the data ingestion process by downloading data from S3 and creating a DataIngestionArtifact object which is used to store the data locally into artifacts folder that contains paths to the train and test datasets.
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        logging.info(
            "Entered the initiate_data_ingestion method of Data ingestion class"
        )

        try:
            self.get_data_from_s3()

            data_ingestion_artifact: DataIngestionArtifact = DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.train_data_path,
                test_file_path=self.data_ingestion_config.test_data_path,
            )

            logging.info(
                "Exited the initiate_data_ingestion method of Data ingestion class"
            )

            return data_ingestion_artifact

        except Exception as e:
            raise XRayException(e, sys)
