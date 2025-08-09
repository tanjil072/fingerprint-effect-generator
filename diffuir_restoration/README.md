# DiffUIR Fingerprint Restoration Experiment

This folder contains the implementation for using DiffUIR model to remove fingerprint effects from images, designed for experimental evaluation and benchmarking.

## 🎯 Project Overview

This experiment evaluates the effectiveness of the DiffUIR (Diffusion-based Underwater Image Restoration) model for removing artificially generated fingerprint effects from images. The setup allows for comprehensive benchmarking against other restoration methods.

## 📁 Structure

```
diffuir_restoration/
├── README.md                   # This file
├── requirements.txt           # Python dependencies
├── setup.py                   # Setup and installation script
├── config.yaml               # Configuration file
├── src/                      # Core implementation
│   ├── __init__.py
│   ├── diffuir_model.py      # DiffUIR model wrapper
│   ├── data_loader.py        # Dataset loading and preprocessing
│   ├── metrics.py            # PSNR, SSIM evaluation metrics
│   ├── restoration.py        # Main restoration pipeline
│   └── utils.py              # Utility functions
├── models/                   # Pre-trained model weights
│   └── diffuir_weights/      # DiffUIR model checkpoints
├── data/                     # Dataset organization
│   ├── clean/                # Original clean images
│   ├── fingerprinted/        # Images with fingerprint effects
│   └── restored/             # Restored images output
├── results/                  # Evaluation results
│   ├── metrics/              # PSNR/SSIM scores
│   ├── comparisons/          # Side-by-side comparisons
│   └── reports/              # Evaluation reports
└── evaluation/              # Evaluation scripts
    ├── run_evaluation.py    # Main evaluation script
    ├── benchmark.py         # Benchmarking against baselines
    └── visualize_results.py # Results visualization
```

## 🚀 Quick Start

### 1. Installation

```bash
cd diffuir_restoration
pip install -r requirements.txt
python setup.py
```

### 2. Prepare Data

```bash
python src/data_loader.py --prepare-dataset
```

### 3. Run Restoration

```bash
python evaluation/run_evaluation.py --config config.yaml
```

### 4. Generate Benchmarks

```bash
python evaluation/benchmark.py --compare-methods
```

## 📊 Evaluation Metrics

- **PSNR (Peak Signal-to-Noise Ratio)**: Higher is better
- **SSIM (Structural Similarity Index)**: Higher is better (range: 0-1)
- **Processing Time**: Per image restoration time
- **Visual Quality**: Perceptual assessment

## 🔧 Configuration

Edit `config.yaml` to customize:

- Model parameters
- Dataset paths
- Evaluation settings
- Output directories

## 📈 Expected Results

Based on similar restoration tasks, expected performance:

- PSNR: 23-29 dB (varies by scene complexity)
- SSIM: 0.82-0.90 (varies by degradation level)

## 🤝 Integration

This experiment folder is designed to be:

- **Portable**: Can be moved to any location
- **Self-contained**: All dependencies and data included
- **Reproducible**: Fixed seeds and configurations
- **Extensible**: Easy to add new baseline methods
