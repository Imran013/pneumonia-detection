<div align="center">

# 🏥 Pneumonia Detection from Chest X-Rays

**An end-to-end deep learning system for automated pneumonia detection — from model training to real-time web deployment.**

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

[Features](#-features) · [Results](#-results) · [Architecture](#-architecture) · [Quick Start](#-quick-start) · [API](#-api-reference) · [Dataset](#-dataset)

</div>

---

## 📌 Overview

This project implements a production-ready pneumonia detection system using transfer learning on chest X-ray images. Unlike typical notebook-based approaches, this system addresses real-world challenges including **class imbalance**, **threshold optimization**, and **deployment** — resulting in a model that is not just accurate, but clinically reliable.

> **⚠️ Medical Disclaimer:** This project is for educational and research purposes only. It is not intended for clinical diagnosis. Always consult a qualified healthcare professional for medical decisions.

---

## ✨ Features

- **Transfer Learning** — ResNet50 backbone pre-trained on ImageNet with partial layer freezing and differential learning rates
- **Class Imbalance Handling** — WeightedRandomSampler + inverse-frequency class weights + class-aware augmentation for the minority class
- **Threshold Optimization** — Systematic analysis across probability thresholds to find the optimal sensitivity/specificity trade-off
- **Comprehensive Evaluation** — Confusion matrix, ROC curve, PR curve, confidence distributions, per-sample predictions, and full threshold analysis exported to XLSX
- **Production API** — FastAPI backend with rate limiting, input validation, CORS, and real-time inference (<300ms)
- **Web Interface** — Clean, responsive frontend for uploading X-rays and viewing predictions

---

## 📊 Results

### Performance Metrics

| Metric | Value |
|---|---|
| Validation Accuracy | **98.47%** |
| Test ROC AUC | **0.9575** |
| Average Precision | **0.9691** |
| Inference Time | **<300ms** |

### Confusion Matrix — The Impact of Threshold Optimization

| | Default (t=0.5) | Optimized (t=0.7) | Improvement |
|---|---|---|---|
| **Specificity** (True Negative Rate) | 69.2% | **85.0%** | +15.8% |
| **Sensitivity** (True Positive Rate) | 99.0% | **94.6%** | -4.4% |
| **False Positives** | 72 | **35** | **-51.4%** |
| **False Negatives** | 4 | 21 | +17 |

At the default threshold of 0.5, the model flagged 30.8% of healthy patients as pneumonia. After threshold optimization, false positives dropped by 51% while sensitivity remained above 94%.

### Training History

<div align="center">
<img src="plots/training_history.png" width="800"/>
</div>

### Confusion Matrices (Default vs Optimized Threshold)

<div align="center">
<img src="plots/confusion_matrices.png" width="800"/>
</div>

### ROC Curve & Precision-Recall Curve

<div align="center">
<img src="plots/roc.png" width="400"/> <img src="plots/pr.png" width="400"/>
</div>

### Threshold Analysis

<div align="center">
<img src="plots/threshold.png" width="800"/>
</div>

### Confidence & Probability Distributions

<div align="center">
<img src="plots/confidence.png" width="800"/>
<img src="plots/prob_dist.png" width="800"/>
</div>

---

## 🏗 Architecture

### System Overview

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│  Chest X-ray │────▶│  Preprocessing│────▶│  ResNet50    │────▶│  Prediction  │
│  Upload      │     │  & Validation │     │  + Custom FC │     │  + Threshold │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
                                                                       │
                                                              ┌────────▼────────┐
                                                              │  NORMAL /       │
                                                              │  PNEUMONIA      │
                                                              │  + Confidence % │
                                                              └─────────────────┘
```

### Model Architecture

- **Backbone:** ResNet50 (pre-trained on ImageNet)
- **Frozen Layers:** conv1, bn1, layer1, layer2 (partial freezing strategy)
- **Custom Classifier:**
  - Dropout (0.3) → Linear (2048 → 512) → ReLU → BatchNorm1d → Dropout (0.3) → Linear (512 → 2)
- **Optimizer:** Adam with differential learning rates (backbone: lr×0.01, classifier: lr)

### Project Structure

```
├── config.py                  # All hyperparameters and paths
├── data_loader.py             # ClassAwareDataset + WeightedRandomSampler
├── fine_tuning_model.py       # ResNet50 transfer learning model
├── trainer.py                 # Training loop with class weights + MixUp/CutMix
├── evaluator.py               # Basic evaluation
├── evaluate_pipeline.py       # Comprehensive evaluation + XLSX export
├── visualizer.py              # Training plots
├── utils.py                   # Seed, timing, parameter counting
├── main.py                    # Training pipeline (transfer learning)
├── pneumonia_api.py           # FastAPI production API
├── checkpoints/               # Saved model weights
├── plots/                     # Generated evaluation plots
└── outputs/                   # XLSX results, JSON predictions
```

---

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
CUDA-capable GPU (recommended, CPU works but slower)
```

### Installation

```bash
git clone https://github.com/yourusername/pneumonia-detection.git
cd pneumonia-detection
pip install -r requirements.txt
```

### Dependencies

```
torch>=2.0
torchvision>=0.15
scikit-learn
matplotlib
openpyxl
fastapi
uvicorn
python-multipart
pillow
numpy
```

### Dataset Setup

Download the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle and extract it:

```
chest_xray/
├── train/
│   ├── NORMAL/       (1,341 images)
│   └── PNEUMONIA/    (3,875 images)
├── val/
│   ├── NORMAL/       (8 images)
│   └── PNEUMONIA/    (8 images)
└── test/
    ├── NORMAL/       (234 images)
    └── PNEUMONIA/    (390 images)
```

### Training

```bash
python main.py
```

### Evaluation (No Training — Uses Saved Checkpoint)

```bash
python evaluate_pipeline.py
```

This generates all plots, XLSX with per-sample predictions, threshold analysis, and confusion matrices at both default (0.5) and optimized thresholds.

### Start the API

```bash
uvicorn pneumonia_api:app --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000/docs` for the interactive API documentation.

---

## 🔌 API Reference

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Root — API info |
| `GET` | `/health` | Health check — model status |
| `POST` | `/predict` | Upload X-ray → get prediction |
| `GET` | `/model/info` | Model metadata and parameters |

### Prediction Request

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@chest_xray.jpg" \
  -F "patient_id=PAT-12345"
```

### Prediction Response

```json
{
  "status": "success",
  "prediction": "PNEUMONIA",
  "confidence": 88.5,
  "probabilities": {
    "NORMAL": 0.115,
    "PNEUMONIA": 0.885
  },
  "processing_time_ms": 245.3,
  "patient_id": "PAT-12345",
  "warnings": null
}
```

### Rate Limiting

The API enforces **10 requests per 60 seconds** per IP address.

---

## 📁 Dataset

This project uses the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset by Paul Mooney on Kaggle.

- **Total:** 5,856 images (JPEG)
- **Classes:** NORMAL, PNEUMONIA
- **Imbalance:** ~75% PNEUMONIA, ~25% NORMAL
- **Source:** Guangzhou Women and Children's Medical Center

---

## 🔧 Key Engineering Decisions

### Why Threshold Optimization Matters

The default decision boundary of 0.5 treats false positives and false negatives equally. In medical contexts, the cost of each error type is different. By analyzing the full threshold curve, we found that **0.7** provides the best F1 score while significantly reducing false positives — a critical improvement for clinical reliability.

### Handling the 3:1 Class Imbalance

Three complementary strategies were used:

1. **WeightedRandomSampler** — Ensures balanced class exposure every epoch without discarding any data
2. **Inverse-frequency class weights** — Applied in `CrossEntropyLoss` so the loss function penalizes minority-class errors proportionally more
3. **Class-aware augmentation** — The NORMAL (minority) class receives aggressive augmentation (stronger rotation, affine transforms, perspective distortion, color jitter, Gaussian blur, random erasing) while PNEUMONIA receives standard augmentation

### Transfer Learning Strategy

With ~4,400 training images, training from scratch is not viable. We use ResNet50 pre-trained on ImageNet with partial freezing: early layers (conv1, bn1, layer1, layer2) are frozen to preserve learned low-level features, while deeper layers (layer3, layer4) and the custom classifier are fine-tuned with differential learning rates.

---

## 📈 Evaluation Pipeline

The evaluation script (`evaluate_pipeline.py`) loads a saved checkpoint and generates:

- **Confusion matrices** at default (0.5) and optimized thresholds
- **ROC curve** with AUC score
- **Precision-Recall curve** with Average Precision
- **Threshold analysis** — sensitivity, specificity, precision, F1, accuracy for thresholds from 0.1 to 0.95
- **Confidence distributions** — correct vs wrong predictions, by class
- **Probability distributions** — P(PNEUMONIA) and P(NORMAL) by true class
- **XLSX export** with all results across 6 sheets (Summary, Confusion Matrices, Threshold Analysis, Per-Sample Predictions, Training History)
- **JSON exports** — per-sample predictions, threshold data, evaluation summary

---

## 🤝 Contributing

Contributions are welcome. Please open an issue first to discuss what you'd like to change.

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for details.

---

<div align="center">

**Built with PyTorch, FastAPI, and a lot of confusion matrix staring.**

</div>
