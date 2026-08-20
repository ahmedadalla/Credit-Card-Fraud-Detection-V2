import logging
import logging.config
from pathlib import Path
import sys
from  src.fraud_detection.configs import Paths

# Create logs directory if it doesn't exist
log_dir = Paths.LOGS

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": sys.stdout
        },
        "file": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": log_dir +"/"+ "ml_project.log",
            "mode": "a"
        }
    },
    "root": {
        "handlers": ["console", "file"],
        "level": "DEBUG",
        "propagate": False
    }
}

def configure_logging():
    """Configure logging for the application"""
    logging.config.dictConfig(LOGGING_CONFIG)
    logger = logging.getLogger(__name__)
    logger.info("Logging configured")

# Configure logging when this module is imported
configure_logging()