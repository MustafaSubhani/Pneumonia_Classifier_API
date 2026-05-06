import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class PredictionLogger:
    """Appends structured prediction records to a JSONL log file."""

    def __init__(self, log_path: str, model_version: str) -> None:
        self._log_path = Path(log_path)
        self._model_version = model_version
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, filename: str, prediction: dict) -> None:
        """Append one prediction record to the log file.

        Parameters
        ----------
        filename:
            Original filename of the submitted image.
        prediction:
            Dict returned by inference.predict() — must contain 'class',
            'confidence', 'probabilities', and 'latency_ms'.
        """
        record = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "filename": filename,
            "predicted_class": prediction.get("class"),
            "confidence": prediction.get("confidence"),
            "probabilities": prediction.get("probabilities"),
            "latency_ms": prediction.get("latency_ms"),
            "model_version": self._model_version,
        }
        try:
            with open(self._log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except OSError as exc:
            logger.warning("Failed to write prediction log: %s", exc)


class DriftMonitor:
    """Reads the prediction log and surfaces basic data-drift signals."""

    def __init__(self, log_path: str, config) -> None:
        self._log_path = Path(log_path)
        self._config = config

    def _load_records(self, window: Optional[int] = None) -> list[dict]:
        """Return the most recent *window* records from the log (all if None)."""
        if not self._log_path.exists():
            return []
        try:
            with open(self._log_path, "r", encoding="utf-8") as fh:
                lines = [ln.strip() for ln in fh if ln.strip()]
            records = [json.loads(ln) for ln in lines]
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not read prediction log: %s", exc)
            return []

        if window is not None and window > 0:
            records = records[-window:]
        return records

    def get_summary_stats(self, window: Optional[int] = None) -> dict:
        """Return aggregate statistics over the most recent *window* predictions.

        Returns
        -------
        dict with keys: total_predictions, average_confidence,
        class_distribution, average_latency_ms, window_size.
        """
        records = self._load_records(window)
        if not records:
            return {
                "total_predictions": 0,
                "average_confidence": None,
                "class_distribution": {},
                "average_latency_ms": None,
                "window_size": window,
            }

        confidences = [r["confidence"] for r in records if r.get("confidence") is not None]
        latencies = [r["latency_ms"] for r in records if r.get("latency_ms") is not None]

        class_counts: dict[str, int] = {}
        for r in records:
            cls = r.get("predicted_class", "UNKNOWN")
            class_counts[cls] = class_counts.get(cls, 0) + 1

        total = len(records)
        class_distribution = {
            cls: {"count": cnt, "percentage": round(cnt / total * 100, 2)}
            for cls, cnt in class_counts.items()
        }

        return {
            "total_predictions": total,
            "average_confidence": round(sum(confidences) / len(confidences), 6) if confidences else None,
            "class_distribution": class_distribution,
            "average_latency_ms": round(sum(latencies) / len(latencies), 3) if latencies else None,
            "window_size": window if window is not None else total,
        }

    def check_confidence_drift(self, window: Optional[int] = None) -> dict:
        """Detect whether average confidence has fallen below the threshold.

        Returns
        -------
        dict with keys: avg_confidence, threshold, drift_detected, message.
        """
        cfg = self._config
        effective_window = window if window is not None else cfg.DRIFT_WINDOW
        records = self._load_records(effective_window)

        if not records:
            return {
                "avg_confidence": None,
                "threshold": cfg.CONFIDENCE_DRIFT_THRESHOLD,
                "drift_detected": False,
                "message": "No predictions available.",
            }

        confidences = [r["confidence"] for r in records if r.get("confidence") is not None]
        avg = round(sum(confidences) / len(confidences), 6) if confidences else 0.0
        drift = avg < cfg.CONFIDENCE_DRIFT_THRESHOLD
        message = (
            f"Confidence drift detected: avg={avg:.4f} < threshold={cfg.CONFIDENCE_DRIFT_THRESHOLD}."
            if drift
            else f"Confidence is healthy: avg={avg:.4f} >= threshold={cfg.CONFIDENCE_DRIFT_THRESHOLD}."
        )

        if drift:
            logger.warning(message)

        return {
            "avg_confidence": avg,
            "threshold": cfg.CONFIDENCE_DRIFT_THRESHOLD,
            "drift_detected": drift,
            "message": message,
        }

    def check_label_drift(self, window: Optional[int] = None) -> dict:
        """Detect whether the PNEUMONIA rate has shifted beyond tolerance.

        Returns
        -------
        dict with keys: observed_rate, expected_rate, tolerance,
        drift_detected, message.
        """
        cfg = self._config
        effective_window = window if window is not None else cfg.DRIFT_WINDOW
        records = self._load_records(effective_window)

        if not records:
            return {
                "observed_rate": None,
                "expected_rate": cfg.EXPECTED_PNEUMONIA_RATE,
                "tolerance": cfg.LABEL_DRIFT_TOLERANCE,
                "drift_detected": False,
                "message": "No predictions available.",
            }

        pneumonia_count = sum(
            1 for r in records if r.get("predicted_class") == "PNEUMONIA"
        )
        observed_rate = round(pneumonia_count / len(records), 6)
        drift = abs(observed_rate - cfg.EXPECTED_PNEUMONIA_RATE) > cfg.LABEL_DRIFT_TOLERANCE
        message = (
            f"Label drift detected: observed={observed_rate:.4f}, "
            f"expected={cfg.EXPECTED_PNEUMONIA_RATE:.4f} ± {cfg.LABEL_DRIFT_TOLERANCE}."
            if drift
            else f"Label distribution is within tolerance: observed={observed_rate:.4f}."
        )

        if drift:
            logger.warning(message)

        return {
            "observed_rate": observed_rate,
            "expected_rate": cfg.EXPECTED_PNEUMONIA_RATE,
            "tolerance": cfg.LABEL_DRIFT_TOLERANCE,
            "drift_detected": drift,
            "message": message,
        }

    def get_recent_predictions(
        self,
        limit: int = 50,
        class_filter: Optional[str] = None,
        min_confidence: Optional[float] = None,
    ) -> list[dict]:
        """Return filtered recent predictions from the log.

        Parameters
        ----------
        limit:
            Maximum number of records to return.
        class_filter:
            If set, only return records matching this predicted class.
        min_confidence:
            If set, only return records with confidence >= this value.
        """
        records = self._load_records()
        if class_filter:
            records = [r for r in records if r.get("predicted_class") == class_filter]
        if min_confidence is not None:
            records = [r for r in records if (r.get("confidence") or 0) >= min_confidence]
        return records[-limit:]
