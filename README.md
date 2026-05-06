# Chest X-Ray Pneumonia Classifier

## Project Overview

This project trains and serves a binary image classifier that detects pneumonia in paediatric chest X-rays. Early and accurate pneumonia detection is critical in resource-limited settings where radiologist access is constrained; an automated screening tool can triage high-risk cases for priority review.

The dataset is the [Chest X-Ray Pneumonia dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) from Kaggle, containing approximately 5,800 frontal-view X-rays of children aged one to five. Positive (PNEUMONIA) cases outnumber negative (NORMAL) cases by roughly 3:1, which was addressed through weighted sampling and class-weighted loss during training.

---

## Repository Structure

```
chest-xray-classifier/
├── Models/
│   ├── CNN.pth          # Custom CNN weights
│   ├── R18.pth          # ResNet18 weights
│   └── R50.pth          # ResNet50 weights (production model)
├── Notebooks/
│   └── ...              # Training and error-analysis notebooks
├── src/
│   ├── config.py        # All configurable parameters
│   ├── model.py         # Architecture definition and weight loading
│   ├── inference.py     # Preprocessing, prediction, CLI entry point
│   ├── monitoring.py    # Prediction logging and drift detection
│   └── api.py           # FastAPI application
├── logs/
│   └── predictions.log  # Created at runtime, not committed
├── venv/
├── .gitignore
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Model Results

| Model      | Params | Accuracy | Precision | Recall | F1    | AUC   |
|------------|--------|----------|-----------|--------|-------|-------|
| Custom CNN | 26M    | 0.827    | 0.794     | 0.977  | 0.876 | 0.931 |
| ResNet18   | 11M    | 0.728    | 0.699     | 0.992  | 0.820 | 0.946 |
| ResNet50   | 24M    | 0.918    | 0.906     | 0.969  | 0.937 | 0.974 |

**Train–test accuracy gap:**

| Model      | Train Acc | Test Acc | Gap   |
|------------|-----------|----------|-------|
| Custom CNN | 0.956     | 0.827    | 0.129 |
| ResNet18   | 0.958     | 0.728    | 0.230 |
| ResNet50   | 0.969     | 0.918    | 0.051 |

ResNet18 shows the largest generalisation gap (0.230), suggesting it memorised training-specific features. The Custom CNN also overfits moderately (0.129). ResNet50's gap of 0.051 indicates that the regularisation strategy was most effective for this architecture and that it generalises reliably to unseen X-rays.

---

## Why ResNet50

ResNet50 was selected as the production model for the following reasons:

- **Highest test accuracy (0.918)** and **AUC (0.974)** across all three models.
- **Best F1 score (0.937)**, balancing precision and recall on the imbalanced dataset.
- **Smallest train–test gap (0.051)**, demonstrating the most reliable generalisation.
- **Clinically acceptable recall (0.969)**: fewer than 4% of pneumonia cases are missed.
- **Effective regularisation**: the combination of BatchNorm, dual Dropout, and partial layer freezing controlled overfitting without sacrificing predictive power.

---

## Technical Decisions

### Class Imbalance

The 3:1 positive-to-negative ratio was addressed with two complementary mechanisms:

- **WeightedRandomSampler** rebalances the mini-batch composition at the data-loading stage, ensuring the model sees a balanced distribution per epoch without duplicating samples in memory.
- **Class-weighted cross-entropy loss** applies a higher penalty to errors on the minority class during the gradient update. Using both ensures the imbalance is corrected both in what the model sees and in how it is penalised for mistakes.

### Data Augmentation

Training augmentations were chosen with clinical plausibility in mind:

- **Random horizontal flip**: chest X-rays can be acquired from either side.
- **Small random rotation (±10°)**: accounts for patient positioning variation.
- **Colour jitter (brightness, contrast)**: simulates variation across X-ray machines and exposure settings.
- No vertical flips or extreme distortions, which would produce anatomically implausible images.

### Regularisation

| Technique               | Setting            | Purpose                                       |
|-------------------------|--------------------|-----------------------------------------------|
| Batch Normalisation     | After Linear(2048→512) | Stabilises activations, mild regulariser   |
| Dropout                 | 0.5 and 0.3        | Prevents co-adaptation in the FC head         |
| L2 weight decay         | 5e-4               | Shrinks weights toward zero globally          |
| L1 regularisation       | 1e-5               | Promotes sparsity in weight magnitudes        |
| Early stopping          | Patience 4 epochs  | Halts training when validation loss plateaus  |

### Training Strategies Across Architectures

Three different architectures and training methodologies were explored to find the optimal balance between model capacity and generalisation:

- **Custom CNN (Trained from Scratch):** A bespoke architecture trained entirely from random initialization. This served as a baseline to understand the fundamental challenges of the dataset and to verify that our data augmentation and class-balancing strategies were effective before introducing complex pre-trained models.
- **ResNet18 (Fully Fine-Tuned):** Initialised with ImageNet weights, all layers of the ResNet18 model were fully unfrozen and fine-tuned. Because it is a relatively smaller network (11M parameters), it was feasible to train the entire network to see how well it could adapt specifically to medical imagery. However, this approach resulted in significant overfitting (the largest train-test gap), as the model memorised training-specific noise without sufficient regularisation.
- **ResNet50 (Partially Frozen):** For the larger ResNet50 model (24M parameters), fully fine-tuning on a small dataset (~5,800 images) would lead to severe overfitting. To counter this, a partial freezing strategy was adopted. The initial blocks (like `layer1`), which act as generic edge and texture detectors learned from ImageNet, were frozen. Only the deeper, more semantic layers and the custom classification head were updated. This drastically reduced the trainable parameter count, effectively regularising the network and yielding the best generalisation performance.

---

## Quickstart

### Installation

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

### Run Inference from the CLI

```bash
python src/inference.py --image path/to/chest_xray.jpg

# Override the model path
python src/inference.py --image path/to/chest_xray.jpg --model-path Models/CNN.pth
```

### Run the FastAPI Server Locally

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

### API Endpoints

```bash
# Service health
curl http://localhost:8000/health

# Model architecture and metrics
curl http://localhost:8000/model/info

# Drift and summary statistics
curl http://localhost:8000/metrics

# Single-image prediction
curl -X POST http://localhost:8000/predict \
     -F "file=@path/to/chest_xray.jpg"

# Batch prediction (up to 16 images)
curl -X POST http://localhost:8000/predict/batch \
     -F "files=@image1.jpg" \
     -F "files=@image2.jpg"

# Prediction history (most recent 50)
curl "http://localhost:8000/predictions/history?limit=50"

# History filtered by class and minimum confidence
curl "http://localhost:8000/predictions/history?class_filter=PNEUMONIA&min_confidence=0.9&limit=20"
```

---

## Testing

`test_api.py` runs 36 automated checks against every endpoint using the six sample images in the project root.

### Against a local server (no Docker required)

```bash
# Terminal 1 — start the server
uvicorn src.api:app --host 0.0.0.0 --port 8000

# Terminal 2 — run the tests
python test_api.py --no-docker
```

### Against Docker

> **Prerequisite**: Docker Desktop must be open and fully started before running this command. If it is not running you will see a daemon connection error.

```bash
python test_api.py --docker-tag chest-xray-classifier
```

The script will build the image, start a container, run all tests, and stop the container automatically.

---

## Docker

> **Prerequisite**: Docker Desktop must be open and fully started.

### Build

```bash
docker build -t chest-xray-classifier .
```

### Run

```bash
docker run -p 8000:8000 chest-xray-classifier
```

### Example Request

```bash
curl -X POST http://localhost:8000/predict \
     -F "file=@path/to/chest_xray.jpg"
```

---

## Production & Monitoring

Every prediction appended to `logs/predictions.log` contains:

| Field             | Description                                              |
|-------------------|----------------------------------------------------------|
| `timestamp`       | UTC ISO-8601 time of the prediction                      |
| `filename`        | Original filename submitted by the client                |
| `predicted_class` | `NORMAL` or `PNEUMONIA`                                  |
| `confidence`      | Softmax probability of the predicted class               |
| `probabilities`   | Full softmax distribution over both classes              |
| `latency_ms`      | Model forward-pass time in milliseconds                  |
| `model_version`   | Version string from config for reproducibility           |

### Confidence Drift

If the rolling average confidence over the last `DRIFT_WINDOW` predictions falls below `CONFIDENCE_DRIFT_THRESHOLD` (default 0.75), the `/metrics` endpoint reports `drift_detected: true`. This typically signals that the incoming image distribution has shifted away from the training domain (e.g., a different X-ray machine, new acquisition protocol). **Recommended action**: retrain or fine-tune on recent labelled samples.

```bash
curl http://localhost:8000/metrics
```

### Label Drift

If the observed PNEUMONIA rate deviates from `EXPECTED_PNEUMONIA_RATE` (0.75) by more than `LABEL_DRIFT_TOLERANCE` (0.15), label drift is flagged. In a clinical setting, a sustained shift in the positive rate can indicate a seasonal disease outbreak or a change in the patient population being screened. **Recommended action**: investigate the referral pipeline and alert clinical staff.

```bash
curl "http://localhost:8000/predictions/history?class_filter=PNEUMONIA&min_confidence=0.8&limit=100"
```

**Production upgrade path**: replace the flat log file with InfluxDB or Prometheus for time-series querying; run drift checks on a schedule via Airflow or APScheduler; route alerts to PagerDuty or Slack.

---

## Future Improvements

- **Larger dataset**: train on NIH ChestX-ray14 (112,000 images, 14 pathologies) for broader coverage.
- **Modern architectures**: benchmark EfficientNetV2 and Vision Transformers (ViT-B/16) against ResNet50.
- **Robust evaluation**: replace the single train/test split with 5-fold cross-validation.
- **Ensembling**: combine CNN, ResNet18, and ResNet50 predictions via soft voting.
- **Explainability**: integrate Grad-CAM to generate heatmaps highlighting the lung regions driving each prediction.
- **Continuous retraining**: build an MLOps pipeline that automatically retrains the model when drift is sustained beyond a configurable threshold.
