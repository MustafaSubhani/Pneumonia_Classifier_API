import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src.config as config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def preprocess(image_path: str) -> torch.Tensor:
    """Load an image from disk and return a normalised (1, 3, 224, 224) tensor.

    Parameters
    ----------
    image_path:
        Absolute or project-relative path to the image file.

    Raises
    ------
    FileNotFoundError:
        If the path does not exist on disk.
    ValueError:
        If Pillow cannot open the file as an image.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: '{image_path}'")

    try:
        img = Image.open(path).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Cannot open '{image_path}' as an image: {exc}") from exc

    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD),
    ])
    return transform(img).unsqueeze(0)


def predict(image_path: str, model: torch.nn.Module, cfg) -> dict:
    """Run inference on a single image and return structured results.

    Parameters
    ----------
    image_path:
        Path to the image file.
    model:
        A loaded, eval-mode PyTorch model.
    cfg:
        Project config module (exposes CLASS_NAMES, DEVICE, CONFIDENCE_THRESHOLD).

    Returns
    -------
    dict with keys:
        class (str), confidence (float), probabilities (dict), latency_ms (float)
    """
    tensor = preprocess(image_path).to(cfg.DEVICE)

    start = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)
    latency_ms = (time.perf_counter() - start) * 1000

    probs = torch.softmax(logits, dim=1).squeeze().cpu().tolist()
    predicted_idx = int(torch.argmax(torch.tensor(probs)))
    predicted_class = cfg.CLASS_NAMES[predicted_idx]
    confidence = float(probs[predicted_idx])

    logger.info(
        "image='%s' class='%s' confidence=%.4f latency_ms=%.2f",
        image_path,
        predicted_class,
        confidence,
        latency_ms,
    )

    if confidence < cfg.CONFIDENCE_THRESHOLD:
        logger.warning(
            "Low confidence %.4f (threshold %.2f) for '%s'.",
            confidence,
            cfg.CONFIDENCE_THRESHOLD,
            image_path,
        )

    return {
        "class": predicted_class,
        "confidence": round(confidence, 6),
        "probabilities": {
            name: round(float(p), 6)
            for name, p in zip(cfg.CLASS_NAMES, probs)
        },
        "latency_ms": round(latency_ms, 3),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chest X-ray inference")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Override MODEL_PATH from config (optional)",
    )
    args = parser.parse_args()

    if args.model_path:
        config.MODEL_PATH = args.model_path

    from src.model import load_model
    model = load_model(config)

    result = predict(args.image, model, config)
    print(json.dumps(result, indent=2))
