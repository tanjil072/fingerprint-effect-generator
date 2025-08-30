# Fingerprint Effect Generator

A complete, standalone implementation for generating realistic fingerprint effects on images. This system simulates finger smudges on camera lenses or screens with authentic ridge patterns and natural-looking distortion effects.

## 🎯 Features

### Authentic Fingerprint Patterns

- **Loop**: Most common pattern with curved ridges (elliptical equations)
- **Whorl**: Circular/spiral patterns (spiral mathematics)
- **Arch**: Bridge-like patterns (parabolic curves)

### Realistic Visual Effects

- Natural ridge patterns with proper spacing
- Variable opacity and intensity
- Subtle geometric distortion (smudging simulation)
- Edge blending for natural appearance
- Random rotation and positioning
- Smart collision avoidance

### Highly Configurable

- Adjustable number of fingerprints (2-5 by default)
- Size variation (100-150 pixels)
- Intensity/opacity control (0.6-0.9)
- Blur and noise levels
- Pattern density and contrast

## 📁 Project Structure

```
fingerprint_effect_generator/
├── src/                        # Core implementation
│   ├── __init__.py
│   ├── config.py              # Configuration parameters
│   ├── fingerprint.py         # Core fingerprint class
│   └── fingerprintgenerator.py # Main generation logic
├── examples/                   # Usage examples
│   ├── basic_usage.py         # Simple batch processing
│   └── advanced_usage.py      # Custom configurations & demos
├── input_images/              # Place your images here
│   └── aachen_000001_000019_leftImg8bit.png  # Sample image
├── output/                    # Generated results
│   ├── processed_images/      # Images with fingerprint effects
│   └── label_masks/           # Binary masks of fingerprint locations
└── docs/                      # Documentation
    └── README.md              # This file
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install required packages
pip install numpy opencv-python pillow scipy
```

### 2. Add Your Images

Replace or add images in the `input_images/` folder. A sample image is included for testing. Supported formats:

- PNG, JPG, JPEG, BMP, TIFF

### 3. Run Basic Processing

```bash
# Navigate to the fingerprint generator folder
cd fingerprint_effect_generator

# Run basic processing on all images
python examples/basic_usage.py
```

### 4. Check Results

- Processed images: `output/processed_images/`
- Label masks: `output/label_masks/`

## 📖 Usage Examples

### Basic Usage

```python
import sys
import os
sys.path.append('src')

from fingerprintgenerator import generate_fingerprints
from config import cfg

# Generate fingerprint effect
image_path = "input_images/your_image.jpg"
result_img, label_mask = generate_fingerprints(image_path, cfg)

# Save results
result_img.save("output/processed_images/result.jpg")
label_mask.save("output/label_masks/mask.jpg")
```

### Custom Configuration

```python
# Create custom configuration
custom_cfg = cfg.copy()
custom_cfg.update({
    'maxPrints': 3,           # Maximum 3 fingerprints
    'minPrints': 2,           # Minimum 2 fingerprints
    'maxSize': 180,           # Larger fingerprints
    'maxIntensity': 1.0,      # Maximum visibility
    'brightness_factor': 0.2,  # Darker smudging
    'contrast_boost': 1.5      # Higher contrast
})

# Generate with custom settings
result_img, label_mask = generate_fingerprints(image_path, custom_cfg)
```

### Advanced Examples

```bash
# Run advanced demonstrations
python examples/advanced_usage.py

# This creates:
# - Maximum visibility demo
# - Subtle effect demo
# - Pattern type examples
```

## ⚙️ Configuration Parameters

### Core Settings

- `maxPrints` / `minPrints`: Number of fingerprints (default: 2-5)
- `maxSize` / `minSize`: Fingerprint size in pixels (default: 100-150)
- `maxIntensity` / `minIntensity`: Opacity level (default: 0.6-0.9)

### Visual Effects

- `brightness_factor`: Darkening amount (default: 0.4, lower = darker)
- `contrast_boost`: Contrast enhancement (default: 1.3)
- `blur_radius`: Edge softening (default: 2)
- `noise_level`: Realism noise (default: 0.05)

### Pattern Control

- `pattern_density`: Ridge density (default: 0.5)
- `distortion_strength`: Background distortion (default: 5)
- `edge_fade`: Edge blending (default: 0.6)

## 🎨 Visual Results

The generator creates highly visible, realistic fingerprint effects:

### Before/After Comparison

- **Clear ridge patterns** with authentic fingerprint shapes
- **Realistic smudging effects** that darken and distort the background
- **Natural oil residue appearance** with subtle color shifts
- **Sharp edge definition** while maintaining organic appearance

### Visibility Metrics

- Maximum pixel difference: **155** (highly visible)
- Significant changes: **8.47%** of image pixels
- Status: **✅ FINGERPRINTS ARE HIGHLY VISIBLE**

## 🔬 Technical Details

### Algorithm Overview

1. **Pattern Generation**: Mathematical models create authentic ridge patterns
2. **Smart Positioning**: Non-overlapping placement with collision avoidance
3. **Texture Mapping**: Applies geometric distortion for smudging effects
4. **Alpha Blending**: Natural integration with background images

### Pattern Mathematics

- **Loop**: Elliptical equations with sinusoidal ridges
- **Whorl**: Spiral mathematics for circular patterns
- **Arch**: Parabolic curves for bridge-like ridges

### Visual Processing Pipeline

1. Generate ridge patterns using mathematical models
2. Apply rotation and geometric transformations
3. Create alpha maps with circular falloff
4. Apply background distortion (barrel effect)
5. Blend with contrast and brightness adjustments
6. Add realistic noise and edge effects

## 🆚 Comparison with Raindrop Algorithm

| Aspect                 | Raindrop Generator  | Fingerprint Generator       |
| ---------------------- | ------------------- | --------------------------- |
| **Shape Generation**   | Circle + Ellipse    | Mathematical ridge patterns |
| **Visual Effect**      | Water refraction    | Smudging distortion         |
| **Pattern Types**      | Single droplet type | Loop/Whorl/Arch variations  |
| **Physics Simulation** | Light refraction    | Surface contact/pressure    |
| **Collision Handling** | Merging drops       | Position avoidance          |
| **Texture Processing** | Fisheye distortion  | Geometric smudging          |

## 🛠️ Development

### Extending the System

The modular design makes it easy to:

- Add new pattern types in `fingerprint.py`
- Modify visual effects in `fingerprintgenerator.py`
- Create custom configurations
- Build new applications on top

### File Organization

- **Core logic**: `src/` folder contains all implementation
- **Examples**: `examples/` folder shows usage patterns
- **Configuration**: Centralized in `src/config.py`
- **Separation**: Completely independent from raindrop algorithm

## 📋 Requirements

- Python 3.7+
- NumPy: Mathematical operations
- OpenCV: Image processing
- Pillow: Image manipulation
- SciPy: Advanced image operations

## 🎉 Ready to Use

The fingerprint effect generator is completely set up and ready to use! It creates realistic, highly visible fingerprint smudges that simulate authentic finger touches on camera lenses or screens.

Start with the basic example and experiment with different configurations to achieve the perfect fingerprint effects for your needs.

## 🧼 Scratch Removal (Inpainting) + PSNR/SSIM Evaluation

This project includes scripts to remove scratches using your label masks and evaluate PSNR/SSIM against `input_images/`.

Outputs and reports are written to `output/restored_images/`.

### Option A — OpenCV Telea (recommended for thin scratches)

Fast, classical inpainting that performed best on thin scratches.

```zsh
"/Users/tanjilpranto/Desktop/Rain Drop Generation/fingerprint-effect-generator/.venv/bin/python" \
"/Users/tanjilpranto/Desktop/Rain Drop Generation/fingerprint-effect-generator/restore_and_evaluate.py" \
--method opencv --opencv_kind telea --opencv_radius 3 --opencv_dilate_ksize 3 --opencv_dilate_iter 1
```

Results:

- Restored images: `output/restored_images/opencv/`
- Metrics: `output/restored_images/psnr_ssim_opencv.txt` and `psnr_ssim_combined.csv`

### Option B — LaMa (deep learning, better for wider/complex regions)

Uses LaMa via `lama-cleaner` in a Python 3.11 virtualenv. The first run downloads model weights.

```zsh
SSL_CERT_FILE="$(("/Users/tanjilpranto/Desktop/Rain Drop Generation/fingerprint-effect-generator/.venv311/bin/python") -c 'import certifi; print(certifi.where())')" \
REQUESTS_CA_BUNDLE="$(("/Users/tanjilpranto/Desktop/Rain Drop Generation/fingerprint-effect-generator/.venv311/bin/python") -c 'import certifi; print(certifi.where())')" \
"/Users/tanjilpranto/Desktop/Rain Drop Generation/fingerprint-effect-generator/.venv311/bin/python" \
"/Users/tanjilpranto/Desktop/Rain Drop Generation/fingerprint-effect-generator/restore_and_evaluate.py" \
--method lama --lama_dilate_ksize 1 --lama_dilate_iter 0
```

Results:

- Restored images: `output/restored_images/lama/`
- Metrics: `output/restored_images/psnr_ssim_lama.txt` and `psnr_ssim_combined.csv`

### Notes

- Inputs: Scratched images in `output/processed_images/`, masks in `output/label_masks/` (white = inpaint region).
- You can run both methods (one after another) to populate both folders and compare in the combined CSV.
- For thicker scratches, increase `--opencv_radius` (e.g., 5–7) or add slight mask dilation.
