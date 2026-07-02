import os
import sys

from x_ray.constant.training_pipeline import (
    AWS_ACCOUNT_ID,
    AWS_REGION,
)
from x_ray.entity.artifacts_entity import ModelPusherArtifact
from x_ray.entity.config_entity import ModelPusherConfig
from x_ray.exception import XRayException
from x_ray.logger import logging


class ModelPusher:
    """
    Responsible for building the BentoML service,
    containerizing it, and pushing the Docker image to AWS ECR.
    """

    def __init__(self, model_pusher_config: ModelPusherConfig):
        self.model_pusher_config = model_pusher_config

        self.ecr_repository = (
            f"{AWS_ACCOUNT_ID}.dkr.ecr.{AWS_REGION}.amazonaws.com"
        )

    def build_and_push_bento_image(self) -> None:

        logging.info(
            "Entered build_and_push_bento_image method."
        )

        try:

            # Build Bento
            logging.info("Building Bento...")

            status = os.system("bentoml build")

            if status != 0:
                raise Exception("Failed to build Bento.")

            logging.info("Bento built successfully.")

            # Containerize
            logging.info("Creating Docker image...")

            status = os.system(
                f"bentoml containerize "
                f"{self.model_pusher_config.bentoml_service_name}:latest "
                f"-t {self.ecr_repository}/"
                f"{self.model_pusher_config.bentoml_ecr_image}:latest"
            )

            if status != 0:
                raise Exception("Failed to create Docker image.")

            logging.info("Docker image created successfully.")

            # Login to ECR
            logging.info("Logging into AWS ECR...")

            status = os.system(
                f"aws ecr get-login-password --region {AWS_REGION} | "
                f"docker login --username AWS "
                f"--password-stdin {self.ecr_repository}"
            )

            if status != 0:
                raise Exception("Failed to login to ECR.")

            logging.info("Logged into AWS ECR.")

            # Push Docker image
            logging.info("Pushing Docker image...")

            status = os.system(
                f"docker push "
                f"{self.ecr_repository}/"
                f"{self.model_pusher_config.bentoml_ecr_image}:latest"
            )

            if status != 0:
                raise Exception("Failed to push Docker image.")

            logging.info("Docker image pushed successfully.")

            logging.info(
                "Exited build_and_push_bento_image method."
            )

        except Exception as e:
            raise XRayException(e, sys)

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Build the BentoML service and push the Docker image to AWS ECR.

        Returns:
            ModelPusherArtifact
        """

        logging.info(
            "Entered initiate_model_pusher method."
        )

        try:

            self.build_and_push_bento_image()

            model_pusher_artifact = ModelPusherArtifact(
                bentoml_model_name=self.model_pusher_config.bentoml_model_name,
                bentoml_service_name=self.model_pusher_config.bentoml_service_name,
            )

            logging.info(
                "Exited initiate_model_pusher method."
            )

            return model_pusher_artifact

        except Exception as e:
            raise XRayException(e, sys)