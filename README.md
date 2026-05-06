# Chest X-Ray Pneumonia Classifier

## Project Overview

This project trains and serves a binary image classifier that detects pneumonia in paediatric chest X-rays. Early and accurate pneumonia detection is critical in resource-limited settings where radiologist access is constrained; an automated screening tool can triage high-risk cases for priority review.

The dataset is the [Chest X-Ray Pneumonia dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) from Kaggle, containing approximately 5,800 frontal-view paediatric chest X‑rays (age range not explicitly specified in the dataset documentation). Positive (PNEUMONIA) cases outnumber negative (NORMAL) cases by roughly 3:1, which was addressed through weighted sampling and class-weighted loss during training.

**Dataset Attribution:** Chest X-Ray Images (Pneumonia) by Paul Mooney, CC BY 4.0.

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
├── test_api.py          # Automated API test suite
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

- **WeightedRandomSampler** rebalances the mini-batch composition at the data-loading stage.
- **Class-weighted cross-entropy loss** applies a higher penalty to errors on the minority class.

This ensures the imbalance is addressed at both the data-level (sampling) and the loss-level (weighted penalty).

### Data Augmentation

Training augmentations were chosen with clinical plausibility in mind:

- **Random horizontal flip**: chest X-rays can be acquired from either side.
- **Small random rotation (±20°)**: accounts for patient positioning variation.
- **Colour jitter (brightness, contrast)**: simulates variation across X-ray machines and exposure settings.
- No vertical flips or extreme distortions, which would produce anatomically implausible images.

### Model Selection and Training Methodology

Three distinct architectures were trained and evaluated to understand the trade‑offs between model capacity, transfer learning benefit, and overfitting risk on a moderately sized medical imaging dataset.

#### Custom CNN (Baseline)
**Why used:** A convolutional neural network built from scratch, with four convolutional blocks (filters: 32 → 64 → 128 → 256) followed by a classifier head containing dense layers, Batch Normalisation, and Dropout (0.5 and 0.3). This model served as a lower‑bound baseline to assess the inherent difficulty of the classification task and to verify that data augmentation and class‑balancing strategies were effective before introducing pre‑trained models.

**How trained:**
- Initialised randomly (no pre‑training).
- Trained for up to 15 epochs (fully completed).
- Learning rate = 1e‑3.
- L2 weight decay = 5e‑4, L1 regularisation = 1e‑5.
- Weighted sampler and class‑weighted loss applied.
- Early stopping with patience 4 (not triggered).

#### ResNet18 (Full Fine‑Tuning)
**Why used:** ResNet18 is a relatively lightweight (≈11M parameters) pre‑trained model that offers a good balance between representational power and training speed. All layers were fully fine‑tuned to assess how well a moderately sized pre‑trained network can adapt to chest X‑ray features without architectural modifications.

**How trained:**
- Initialised with ImageNet weights.
- Standard ResNet18 architecture; final fully connected layer replaced with a new classifier (Linear 512→256→2) containing BatchNorm and Dropout.
- Trained for up to 10 epochs (fully completed).
- Learning rate = 1e‑3.
- Same regularisation (L2, L1, Dropout, BatchNorm) and class‑balancing as the custom CNN.

#### ResNet50 (Partial Freezing)
**Why used:** ResNet50 has higher capacity (≈24M parameters), but full fine‑tuning on a dataset of only ≈5,800 images would risk severe overfitting. To retain the benefits of pre‑training while controlling overfitting, a partial freezing strategy was adopted: the early layers (`layer1`), which learn generic low‑level features (edges, textures), were frozen, while all subsequent layers and the new classifier head were fine‑tuned.

**How trained:**
- Initialised with ImageNet weights.
- `layer1` parameters frozen; `requires_grad = False`.
- New classifier head: `Linear(2048→512) → ReLU → BatchNorm → Dropout(0.5) → Linear(512→128) → ReLU → Dropout(0.3) → Linear(128→2)`.
- Trained for up to 15 epochs (early stopping triggered after 7 epochs).
- Learning rate = 3e‑4 (lower due to larger model capacity).
- Same regularisation and class‑balancing as other models.

#### Training Commonality
All three models were trained with:
- Batch size = 32.
- Optimiser = Adam with weight decay (5e‑4).
- Loss = class‑weighted CrossEntropyLoss + L1 penalty (1e‑5).
- Learning rate scheduler = ReduceLROnPlateau (mode = 'max', patience = 2, factor = 0.5).
- Early stopping = patience 4, monitoring validation AUC.
- Data augmentation (training only): random rotation (±20°), horizontal flip, colour jitter.
- WeightedRandomSampler and class weights to address imbalance.

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

> **Note on Authentication:** These endpoints are currently unauthenticated for easy local testing. For production deployments, you should add an API key mechanism, OAuth2, or place the API behind a reverse-proxy with rate-limiting.

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

> **Note:** The `logs/predictions.log` file will grow indefinitely. For a long-term production environment, we recommend implementing log rotation (e.g. via `logrotate`) or switching to a time-series database like InfluxDB or Prometheus.

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
- **Ensembling**: combine CNN, ResNet18, and ResNet50 predictions via soft voting.
- **Continuous retraining**: build an MLOps pipeline that automatically retrains the model when drift is sustained beyond a configurable threshold.
