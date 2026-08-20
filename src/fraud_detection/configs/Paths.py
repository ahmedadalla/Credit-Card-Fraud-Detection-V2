import os
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).parent.parent.parent.parent

# Data
DATA=os.path.join(ROOT_DIR,"assets/data")
RAW_DATA=os.path.join(DATA,"raw/")
PROCESSED=os.path.join(DATA,"processed/")

#
LOGS=os.path.join(ROOT_DIR,"Logs")

#models
MODELS=os.path.join(ROOT_DIR,"assets/models")
