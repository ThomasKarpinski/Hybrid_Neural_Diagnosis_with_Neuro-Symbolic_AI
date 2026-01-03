import logging
import os
from datetime import datetime

# Ensure logs directory exists
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Log filename with timestamp
LOG_FILE = os.path.join(LOG_DIR, "project.log")


def setup_logging():
    """
    Configures global logging for the entire project.
    Writes logs to console and logs/project.log.
    """

    # Format used for both console and file
    log_format = "[%(asctime)s] [%(levelname)s] (%(name)s): %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Root logger
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(LOG_FILE, mode="a"),
            logging.StreamHandler()
        ]
    )


def get_logger(name: str):
    """
    Returns a logger instance for a specific module.
    Ensures global logging is initialized before use.
    """
    # If logging has no handlers yet → initialize
    if not logging.getLogger().handlers:
        setup_logging()

    return logging.getLogger(name)


