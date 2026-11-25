from pathlib import Path
from os import environ

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset" / "anhtraicay" / "data"
RESULTS = ROOT / "results"
LEGACY_RESULTS = RESULTS / "legacy_root"
MODELS = ROOT / "models"
IMAGE_SIZE = (224, 224)
WEB_MODEL_DIR = Path(environ.get("FRUIT_MODEL_DIR", MODELS))
if not WEB_MODEL_DIR.is_absolute():
    WEB_MODEL_DIR = ROOT / WEB_MODEL_DIR
WEB_RESULTS_DIR = RESULTS if WEB_MODEL_DIR.resolve() == MODELS.resolve() else WEB_MODEL_DIR
CONFIDENCE_THRESHOLD = 0.70
FRUIT_PRESENCE_THRESHOLD = 0.35
MAX_IMAGE_BYTES = 5 * 1024 * 1024
