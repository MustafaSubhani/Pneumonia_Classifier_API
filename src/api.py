import logging
import os
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src.config as config
from src.model import load_model, count_parameters, frozen_layer_names
from src.inference import predict
from src.monitoring import PredictionLogger, DriftMonitor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MAX_BATCH_SIZE = 16
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/tiff", "image/webp"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model(config)
    app.state.start_time = time.time()
    app.state.prediction_logger = PredictionLogger(
        config.PREDICTIONS_LOG_PATH, config.MODEL_VERSION
    )
    app.state.drift_monitor = DriftMonitor(config.PREDICTIONS_LOG_PATH, config)
    logger.info("API startup complete.")
    yield
    logger.info("API shutting down.")


app = FastAPI(
    title="Chest X-Ray Classifier",
    description="ResNet50-based pneumonia detection API.",
    version=config.MODEL_VERSION,
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    """Return service liveness and high-level runtime stats."""
    total = app.state.drift_monitor.get_summary_stats().get("total_predictions", 0)
    return {
        "status": "ok",
        "model_version": config.MODEL_VERSION,
        "device": config.DEVICE,
        "uptime_seconds": round(time.time() - app.state.start_time, 2),
        "total_predictions": total,
    }


@app.get("/model/info")
def model_info() -> dict:
    """Return architecture metadata and training metrics."""
    model = app.state.model
    info = dict(config.MODEL_INFO)
    info["parameter_count"] = count_parameters(model)
    info["frozen_layers"] = frozen_layer_names(model)
    info["normalization_constants"] = {"mean": config.MEAN, "std": config.STD}
    return info


@app.get("/metrics")
def metrics() -> dict:
    """Return summary statistics plus confidence and label drift results."""
    monitor: DriftMonitor = app.state.drift_monitor
    return {
        **monitor.get_summary_stats(),
        "confidence_drift": monitor.check_confidence_drift(),
        "label_drift": monitor.check_label_drift(),
    }


@app.post("/predict")
async def predict_single(file: UploadFile = File(...)) -> dict:
    """Accept a single image and return a prediction dict."""
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

    suffix = Path(file.filename or "upload").suffix or ".jpg"
    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        result = predict(tmp_path, app.state.model, config)
        app.state.prediction_logger.log(file.filename or "unknown", result)
        return result
    except Exception as exc:
        logger.error("Inference failed for '%s': %s", file.filename, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/predict/batch")
async def predict_batch(files: list[UploadFile] = File(...)) -> dict:
    """Accept up to 16 images and return per-image predictions plus aggregates."""
    if len(files) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {len(files)} exceeds the maximum of {MAX_BATCH_SIZE}.",
        )

    results = []
    class_counts: dict[str, int] = {}

    for file in files:
        if file.content_type not in ALLOWED_CONTENT_TYPES:
            results.append({"filename": file.filename, "error": f"Unsupported type: {file.content_type}"})
            continue

        suffix = Path(file.filename or "upload").suffix or ".jpg"
        tmp_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(await file.read())
                tmp_path = tmp.name

            pred = predict(tmp_path, app.state.model, config)
            pred["filename"] = file.filename
            app.state.prediction_logger.log(file.filename or "unknown", pred)
            results.append(pred)
            cls = pred["class"]
            class_counts[cls] = class_counts.get(cls, 0) + 1
        except Exception as exc:
            logger.error("Batch inference failed for '%s': %s", file.filename, exc)
            results.append({"filename": file.filename, "error": str(exc)})
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    successful = [r for r in results if "error" not in r]
    avg_conf = (
        round(sum(r["confidence"] for r in successful) / len(successful), 6)
        if successful else None
    )

    return {
        "predictions": results,
        "aggregate": {
            "count": len(files),
            "class_distribution": class_counts,
            "average_confidence": avg_conf,
        },
    }


@app.get("/predictions/history")
def predictions_history(
    limit: int = 50,
    class_filter: Optional[str] = None,
    min_confidence: Optional[float] = None,
) -> dict:
    """Return filtered prediction history from the log.

    Parameters
    ----------
    limit:
        Maximum records to return (1–200).
    class_filter:
        'NORMAL' or 'PNEUMONIA' — restricts results to that class.
    min_confidence:
        Lower bound on confidence (0.0–1.0).
    """
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 200.")

    if class_filter is not None and class_filter not in config.CLASS_NAMES:
        raise HTTPException(
            status_code=400,
            detail=f"class_filter must be one of {config.CLASS_NAMES}.",
        )

    if min_confidence is not None and not (0.0 <= min_confidence <= 1.0):
        raise HTTPException(status_code=400, detail="min_confidence must be between 0 and 1.")

    records = app.state.drift_monitor.get_recent_predictions(
        limit=limit,
        class_filter=class_filter,
        min_confidence=min_confidence,
    )
    return {"count": len(records), "predictions": records}
