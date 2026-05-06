import torch

MODEL_PATH: str = "Models/R50.pth"
MODEL_VERSION: str = "1.0.0"

IMAGE_SIZE: int = 224
MEAN: list[float] = [0.485, 0.456, 0.406]
STD: list[float] = [0.229, 0.224, 0.225]

CLASS_NAMES: list[str] = ["NORMAL", "PNEUMONIA"]
CONFIDENCE_THRESHOLD: float = 0.5

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

API_HOST: str = "0.0.0.0"
API_PORT: int = 8000

DRIFT_WINDOW: int = 100
CONFIDENCE_DRIFT_THRESHOLD: float = 0.75
EXPECTED_PNEUMONIA_RATE: float = 0.75
LABEL_DRIFT_TOLERANCE: float = 0.15

PREDICTIONS_LOG_PATH: str = "logs/predictions.log"

MODEL_INFO: dict = {
    "architecture": "ResNet50",
    "training_dataset": "Chest X-Ray Pneumonia (Kaggle, ~5800 paediatric X-rays)",
    "input_size": [IMAGE_SIZE, IMAGE_SIZE],
    "metrics": {
        "accuracy": 0.918,
        "precision": 0.906,
        "recall": 0.969,
        "f1": 0.937,
        "auc": 0.974,
    },
}
