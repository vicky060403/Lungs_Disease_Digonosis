import logging # Python's built-in module for recording events.
import os

from x_ray.constant.training_pipeline import TIMESTAMP

LOG_FILE: str = f"{TIMESTAMP}.log"
# Create Logs Folder Path
logs_path = os.path.join(os.getcwd(), "logs", TIMESTAMP)
# os.getcwd() Returns the current project directory.

# Create Folder
os.makedirs(logs_path, exist_ok=True)
# Create Full Log File Path to store the logs
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)
# Configure Logging to store the each log files into LOG_FILE_PATH
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
