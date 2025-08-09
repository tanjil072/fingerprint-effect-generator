#!/usr/bin/env python3
"""
Setup script for DiffUIR fingerprint restoration experiment
"""

import os
import sys
import subprocess
import argparse
import shutil
from pathlib import Path


def run_command(command, description, check=True):
    """Run a shell command with description"""
    print(f"🔧 {description}...")
    try:
        if isinstance(command, list):
            result = subprocess.run(command, check=check, capture_output=True, text=True)
        else:
            result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"⚠️  {description} completed with warnings")
            if result.stderr:
                print(f"   Error: {result.stderr.strip()}")
            return False
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stderr:
            print(f"   Error: {e.stderr.strip()}")
        return False


def check_python_version():
    """Check if Python version is suitable"""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected. Python 3.8+ required.")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True


def check_gpu_availability():
    """Check GPU availability"""
    print("🖥️  Checking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ {gpu_count} GPU(s) available: {gpu_name}")
            return True
        else:
            print("⚠️  No GPU available, will use CPU (slower)")
            return False
    except ImportError:
        print("ℹ️  PyTorch not installed yet, GPU check will be performed after installation")
        return None


def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    # Check if requirements.txt exists
    if not os.path.exists('requirements.txt'):
        print("❌ requirements.txt not found")
        return False
    
    # Install dependencies
    success = run_command(
        [sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'],
        "Installing Python packages"
    )
    
    return success


def create_directories():
    """Create necessary directories"""
    print("📁 Creating directory structure...")
    
    directories = [
        'data/clean',
        'data/fingerprinted', 
        'data/restored',
        'results/metrics',
        'results/comparisons',
        'results/reports',
        'results/visualizations',
        'models/diffuir_weights'
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"   ✅ Created: {directory}")
        except Exception as e:
            print(f"   ❌ Failed to create {directory}: {e}")
            return False
    
    print("✅ Directory structure created")
    return True


def setup_model_weights():
    """Setup model weights (placeholder for actual model download)"""
    print("🤖 Setting up model weights...")
    
    weights_dir = Path('models/diffuir_weights')
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a placeholder file indicating where to place model weights
    readme_path = weights_dir / 'README.md'
    readme_content = """# DiffUIR Model Weights

## Instructions

To use a pre-trained DiffUIR model:

1. Download the model weights from the official repository
2. Place the weights file here as `model_best.pth`
3. Update the `checkpoint_path` in `config.yaml` to point to this file

## Model Format

The model file should be a PyTorch state dictionary saved with `torch.save()`.

Expected structure:
```
model_best.pth  # Main model file
```

## Alternative Setup

If you don't have pre-trained weights, the system will use randomly initialized weights.
This is suitable for:
- Testing the evaluation pipeline
- Fine-tuning on your specific dataset
- Experimenting with the architecture

## Note

For best results with fingerprint restoration, consider:
- Pre-trained weights from underwater image restoration tasks
- Models trained on similar degradation patterns
- Fine-tuning on fingerprint-specific data
"""
    
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print("ℹ️  Model weights setup complete (see models/diffuir_weights/README.md)")
    return True


def prepare_sample_data():
    """Prepare sample data from parent project"""
    print("🎯 Preparing sample data...")
    
    # Check if parent project has input images
    parent_input_dir = Path('..') / 'input_images'
    
    if parent_input_dir.exists():
        # Copy sample images to clean directory
        clean_dir = Path('data/clean')
        
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        copied_count = 0
        
        for img_file in parent_input_dir.iterdir():
            if img_file.suffix.lower() in image_extensions:
                shutil.copy2(img_file, clean_dir / img_file.name)
                copied_count += 1
        
        if copied_count > 0:
            print(f"✅ Copied {copied_count} sample images from parent project")
            return True
        else:
            print("⚠️  No images found in parent project")
    else:
        print("ℹ️  Parent project input_images not found")
    
    # Create a note about manual data preparation
    data_readme = Path('data/README.md')
    data_content = """# Dataset Preparation

## Manual Setup

1. **Clean Images**: Place original/clean images in `clean/` directory
2. **Fingerprinted Images**: Will be auto-generated from clean images
3. **Restored Images**: Output directory for restoration results

## Auto-Generation

Run the following command to prepare the dataset:

```bash
python src/data_loader.py --prepare-dataset
```

Or use the main evaluation script:

```bash
python evaluation/run_evaluation.py --prepare-dataset --source-images /path/to/your/images
```

## Supported Formats

- PNG, JPEG, JPG, BMP, TIFF
- RGB images recommended
- Any resolution (will be resized according to config)

## Dataset Structure

```
data/
├── clean/              # Original images (ground truth)
├── fingerprinted/      # Auto-generated fingerprinted versions
└── restored/           # DiffUIR restored outputs
```
"""
    
    with open(data_readme, 'w') as f:
        f.write(data_content)
    
    print("ℹ️  Sample data preparation complete (see data/README.md)")
    return True


def verify_installation():
    """Verify that installation is working"""
    print("🔍 Verifying installation...")
    
    try:
        # Test imports
        print("   Testing imports...")
        import torch
        import numpy as np
        import cv2
        import PIL
        from skimage import metrics
        print("   ✅ Core packages imported successfully")
        
        # Test GPU availability (if installed)
        if torch.cuda.is_available():
            print(f"   ✅ GPU available: {torch.cuda.get_device_name(0)}")
        else:
            print("   ℹ️  CPU mode (no GPU detected)")
        
        # Test configuration loading
        if os.path.exists('config.yaml'):
            print("   ✅ Configuration file found")
        else:
            print("   ⚠️  Configuration file not found")
        
        print("✅ Installation verification complete")
        return True
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Verification error: {e}")
        return False


def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(
        description='Setup DiffUIR fingerprint restoration experiment',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--skip-install',
        action='store_true',
        help='Skip dependency installation'
    )
    
    parser.add_argument(
        '--skip-data',
        action='store_true',
        help='Skip sample data preparation'
    )
    
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only run verification (skip setup)'
    )
    
    args = parser.parse_args()
    
    print("🚀 DiffUIR Fingerprint Restoration Setup")
    print("=" * 45)
    
    if args.verify_only:
        success = verify_installation()
        sys.exit(0 if success else 1)
    
    success_steps = []
    
    # Step 1: Check Python version
    if check_python_version():
        success_steps.append("python_version")
    else:
        print("❌ Setup failed: Python version incompatible")
        sys.exit(1)
    
    # Step 2: Install dependencies
    if not args.skip_install:
        if install_dependencies():
            success_steps.append("dependencies")
        else:
            print("⚠️  Dependency installation failed, continuing with setup...")
    else:
        print("⏭️  Skipping dependency installation")
    
    # Step 3: Check GPU after installation
    gpu_available = check_gpu_availability()
    if gpu_available:
        success_steps.append("gpu")
    
    # Step 4: Create directories
    if create_directories():
        success_steps.append("directories")
    else:
        print("❌ Setup failed: Could not create directories")
        sys.exit(1)
    
    # Step 5: Setup model weights
    if setup_model_weights():
        success_steps.append("model_weights")
    
    # Step 6: Prepare sample data
    if not args.skip_data:
        if prepare_sample_data():
            success_steps.append("sample_data")
    else:
        print("⏭️  Skipping sample data preparation")
    
    # Step 7: Verify installation
    if verify_installation():
        success_steps.append("verification")
    
    # Summary
    print("\n" + "=" * 45)
    print("📋 Setup Summary:")
    
    steps = {
        "python_version": "Python Version Check",
        "dependencies": "Dependency Installation", 
        "gpu": "GPU Detection",
        "directories": "Directory Creation",
        "model_weights": "Model Weights Setup",
        "sample_data": "Sample Data Preparation",
        "verification": "Installation Verification"
    }
    
    for step_id, step_name in steps.items():
        status = "✅" if step_id in success_steps else "❌" if step_id != "gpu" else "⚠️"
        print(f"   {status} {step_name}")
    
    if len(success_steps) >= 4:  # Minimum required steps
        print("\n🎉 Setup completed successfully!")
        print("\n📚 Next Steps:")
        print("1. Place your images in data/clean/ directory")
        print("2. Run: python evaluation/run_evaluation.py --prepare-dataset")
        print("3. Run: python evaluation/run_evaluation.py --mode complete")
        print("4. Check results/ directory for outputs")
        
        if "gpu" not in success_steps and gpu_available is not None:
            print("\n💡 Tip: Install CUDA-compatible PyTorch for GPU acceleration")
    else:
        print("\n⚠️  Setup completed with issues. Check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
