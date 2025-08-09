# DiffUIR Fingerprint Restoration - Project Overview

This folder contains a complete, standalone implementation for evaluating the DiffUIR model's effectiveness in removing fingerprint effects from images. The project is designed to be portable and self-contained, making it easy to move to any location for experimental evaluation.

## 🎯 Quick Start (5 minutes)

```bash
# 1. Navigate to this directory
cd diffuir_restoration

# 2. Run the quick start demo
python quick_start.py

# 3. Check results
ls -la results/
```

## 📁 Project Structure

```
diffuir_restoration/
├── quick_start.py           # One-click demo script
├── setup.py                # Environment setup
├── config.yaml             # Configuration file
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── PROJECT_OVERVIEW.md     # Detailed project documentation
├── src/                    # Core implementation
│   ├── __init__.py
│   ├── diffuir_model.py    # DiffUIR model wrapper
│   ├── data_loader.py      # Dataset handling
│   ├── metrics.py          # PSNR, SSIM, LPIPS evaluation
│   ├── restoration.py      # Main restoration pipeline
│   └── utils.py            # Utility functions
├── evaluation/             # Evaluation scripts
│   ├── run_evaluation.py   # Main evaluation runner
│   ├── benchmark.py        # Baseline comparison
│   └── visualize_results.py # Results visualization
├── models/                 # Model weights directory
│   └── diffuir_weights/    # Pre-trained model files
├── data/                   # Dataset organization
│   ├── clean/              # Original clean images
│   ├── fingerprinted/      # Images with fingerprint effects
│   └── restored/           # DiffUIR restored outputs
└── results/                # Evaluation results
    ├── metrics/            # JSON metric files
    ├── comparisons/        # Side-by-side image comparisons
    ├── reports/            # Markdown evaluation reports
    └── visualizations/     # Plots and dashboards
```

## 🚀 Usage Examples

### 1. Complete Evaluation Pipeline

```bash
python evaluation/run_evaluation.py --mode complete
```

### 2. Single Image Restoration

```bash
python evaluation/run_evaluation.py --mode single \
    --input-image /path/to/degraded_image.png \
    --output-image /path/to/restored_image.png
```

### 3. Benchmark Against Baselines

```bash
python evaluation/benchmark.py --compare-methods
```

### 4. Visualize Results

```bash
python evaluation/visualize_results.py
```

## 📊 Expected Performance

Based on similar restoration tasks, expected metrics:

| Method           | PSNR (dB) | SSIM      | Processing Time (s) |
| ---------------- | --------- | --------- | ------------------- |
| DiffUIR          | 23-29     | 0.82-0.90 | 0.1-0.5             |
| Gaussian Blur    | 21-25     | 0.75-0.85 | 0.001-0.01          |
| Bilateral Filter | 22-26     | 0.78-0.87 | 0.01-0.05           |

## 🔧 Configuration

Key configuration options in `config.yaml`:

```yaml
# Model settings
model:
  device: "cuda" # or "cpu"
  mixed_precision: true
  checkpoint_path: "./models/diffuir_weights/model_best.pth"

# Evaluation settings
evaluation:
  metrics: ["psnr", "ssim", "lpips", "processing_time"]
  save_comparisons: true

# Baseline comparison
baselines:
  enabled: true
  methods:
    - name: "gaussian_blur"
    - name: "bilateral_filter"
    - name: "non_local_means"
```

## 📈 Output Files

### Metrics (`results/metrics/`)

- `test_diffuir_metrics.json` - DiffUIR performance metrics
- `test_benchmark_comparison.json` - All methods comparison

### Visual Comparisons (`results/comparisons/`)

- Side-by-side image comparisons showing degraded, restored, and clean images
- Format: `test_[filename].png`

### Reports (`results/reports/`)

- `evaluation_summary.md` - Overall evaluation summary
- `test_benchmark_report.md` - Detailed comparison report

### Visualizations (`results/visualizations/`)

- `evaluation_dashboard.png` - Comprehensive performance dashboard
- `metrics_comparison.png` - Metric comparison plots

## 🔬 Research Integration

This evaluation framework is designed for research use and can be easily integrated into academic papers:

### Performance Table Generation

```python
# Load results for paper
import json
with open('results/metrics/test_benchmark_comparison.json') as f:
    results = json.load(f)

# Generate LaTeX table
for method, metrics in results.items():
    print(f"{method} & {metrics['psnr']['mean']:.2f} & {metrics['ssim']['mean']:.4f} \\\\")
```

### Citation-Ready Results

The evaluation generates standardized metrics that can be directly used in papers:

- PSNR and SSIM values with standard deviations
- Processing time measurements
- Statistical significance testing data

## 🧪 Experimental Reproducibility

### Random Seeds

All random operations are seeded for reproducibility:

```yaml
experiment:
  random_seed: 42
```

### Environment Information

The system automatically logs:

- Python version
- PyTorch version
- CUDA version (if available)
- Hardware specifications
- Model parameters

### Data Versioning

```bash
# Save dataset information
python -c "from src.data_loader import DatasetPreparator; prep = DatasetPreparator(config); prep.save_dataset_info()"
```

## 🔄 Moving to Another Location

This project is fully portable:

```bash
# Copy entire folder
cp -r diffuir_restoration /new/location/

# Setup in new location
cd /new/location/diffuir_restoration
python setup.py
python quick_start.py
```

## 🐛 Troubleshooting

### Common Issues

1. **ImportError for torch/CUDA**

   ```bash
   pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
   ```

2. **Memory issues with large images**

   - Reduce `batch_size` in config.yaml
   - Enable `mixed_precision: true`

3. **Slow processing on CPU**
   - Set `device: "cpu"` in config.yaml
   - Reduce image resolution in preprocessing

### Getting Help

```bash
# Check setup
python setup.py --verify-only

# Run with debug mode
python evaluation/run_evaluation.py --mode complete --config config.yaml
```

## 📚 Additional Resources

- **DiffUIR Paper**: [Link to original paper]
- **Fingerprint Generation**: See parent directory for fingerprint effect generation
- **Metrics Documentation**: See `src/metrics.py` for implementation details

## 🤝 Contributing

To extend this evaluation framework:

1. **Add new baseline methods**: Edit `src/metrics.py` `BaselineMetrics` class
2. **Add new metrics**: Edit `src/metrics.py` `MetricsCalculator` class
3. **Add new models**: Create new model wrapper in `src/`
4. **Custom visualizations**: Edit `evaluation/visualize_results.py`

## 📄 License & Citation

If you use this evaluation framework in your research, please cite:

```bibtex
@misc{diffuir_fingerprint_evaluation,
  title={DiffUIR Fingerprint Restoration Evaluation Framework},
  author={[Your Name]},
  year={2024},
  howpublished={\\url{[Repository URL]}}
}
```

---

_Generated by DiffUIR Fingerprint Restoration Evaluation Framework_
