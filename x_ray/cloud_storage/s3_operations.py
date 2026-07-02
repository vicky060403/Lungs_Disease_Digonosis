import os
import sys
# import exception custom module from x_ray.exception
from x_ray.exception import XRayException

# create class of S3-Bucket operation to run S3Operation together
class S3Operation:
    # Upload Folder to S3
    # bucket_name = "lungxray"
    def sync_folder_to_s3(self, folder: str, bucket_name: str, bucket_folder_name: str) -> None:
        try:
            # Build AWS command
            """
                folder = "artifacts"
                bucket_name = "lungxray"
                bucket_folder_name = "model_registry"
            """
            command: str = (
                f"aws s3 sync {folder} s3://{bucket_name}/{bucket_folder_name}/ "
            )
            # execute command
            os.system(command)

        except Exception as e:
            raise XRayException(e, sys)

    # Download folder from S3 bucket
    def sync_folder_from_s3(self, folder: str, bucket_name: str, bucket_folder_name: str) -> None:
        try:
            # Build command
            command: str = (
                f"aws s3 sync s3://{bucket_name}/{bucket_folder_name}/ {folder} "
            )
            # execute command
            os.system(command)

        except Exception as e:
            raise XRayException(e, sys)
