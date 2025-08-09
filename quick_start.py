#!/usr/bin/env python3
"""
Quick start script for the fingerprint effect generator.
Run this script to test the system with sample images.
"""

import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Quick start function"""
    
    print("🎯 Fingerprint Effect Generator - Quick Start")
    print("=" * 50)
    
    # Check if we have sample images
    input_dir = os.path.join(os.path.dirname(__file__), 'input_images')
    if not os.path.exists(input_dir) or not os.listdir(input_dir):
        print("❌ No images found in input_images folder")
        print("Please add some images to input_images/ and try again")
        return
    
    # Run basic example
    print("🚀 Running basic fingerprint generation...")
    
    try:
        from examples.basic_usage import main as run_basic
        run_basic()
        
        print("\n✅ Success! Check the output folder for results.")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install numpy opencv-python pillow scipy")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
