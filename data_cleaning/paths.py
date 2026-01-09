import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA, exist_ok=True)

RAW_DATA = os.path.join(DATA, "raw")
os.makedirs(RAW_DATA, exist_ok=True)

RAW_IMGS = os.path.join(RAW_DATA, "ISIC_2019_Training_Input")
os.makedirs(RAW_IMGS, exist_ok=True)

PROCESSED_DATA = os.path.join(DATA, "processed")
os.makedirs(PROCESSED_DATA, exist_ok=True)

PROCESSED_IMGS = os.path.join(PROCESSED_DATA, "images")
os.makedirs(PROCESSED_IMGS, exist_ok=True)

WORKED_IMGS = os.path.join(PROCESSED_DATA, "worked_images")
os.makedirs(WORKED_IMGS, exist_ok=True)

OUTPUTS = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUTPUTS, exist_ok=True)

CONFUSION_METRICS = os.path.join(OUTPUTS, "confusion_metrics")
os.makedirs(CONFUSION_METRICS, exist_ok=True)

METRICS_FOLDER = os.path.join(OUTPUTS, "metrics")
os.makedirs(METRICS_FOLDER, exist_ok=True)

SAVED_MODELS = os.path.join(OUTPUTS, "saved_models")
os.makedirs(SAVED_MODELS, exist_ok=True)

TENSORFLOW_MODEL_FOLDER = os.path.join(SAVED_MODELS, "tensorflow_model_metrics")
os.makedirs(TENSORFLOW_MODEL_FOLDER, exist_ok=True)

PYTORCH_MODEL_FOLDER = os.path.join(SAVED_MODELS, "pytorch_model_metrics")
os.makedirs(PYTORCH_MODEL_FOLDER, exist_ok=True)