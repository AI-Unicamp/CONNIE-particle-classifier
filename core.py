import os
from enum import Enum
from pathlib import Path


MAIN_DIR = Path(__file__).parents[0]
CATALOG_FOLDERPATH = os.path.join(MAIN_DIR, "data", "catalog")
DATABASE_FOLDER = os.path.join(MAIN_DIR, "database")
DATA_FOLDER = os.path.join(MAIN_DIR, "data")
MODELS_FOLDER = os.path.join(MAIN_DIR, "models")
SCHEMA_FILE = os.path.join(MAIN_DIR, "database", "db_schema.sqlite")

class ClassLabel(Enum):
    Muon = 1
    Electron = 2
    Diffusion_Hit = 3
    Blob = 4
    Alpha = 5
    Others = 6
