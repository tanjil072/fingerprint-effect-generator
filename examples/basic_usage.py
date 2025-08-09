#!/usr/bin/env python3
"""
Basic usage example for the fingerprint effect generator.
This script processes all images in the input_images folder.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from fingerprintgenerator import generate_fingerprints
from config import cfg
from PIL import Image
import numpy as np
import cv2


def main():
    """
    Basic example usage of the fingerprint effect generator
    """
    
    # Set up paths relative to the project structure
    base_dir = os.path.dirname(os.path.dirname(__file__))
    input_folder = os.path.join(base_dir, "input_images")
    output_folder = os.path.join(base_dir, "output", "processed_images")
    label_folder = os.path.join(base_dir, "output", "label_masks")
    
    # Create output directories if they don't exist
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(label_folder, exist_ok=True)
    
    # Enhanced configuration for better visibility
    enhanced_cfg = cfg.copy()
    enhanced_cfg.update({
        'maxPrints': 3,
        'minPrints': 2,
        'maxSize': 160,
        'minSize': 120,
        'maxIntensity': 1.0,
        'minIntensity': 0.7,
        'brightness_factor': 0.3,
        'contrast_boost': 1.4
    })
    
    print("🎯 Starting Fingerprint Effect Generation")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Label folder: {label_folder}")
    print("-" * 50)
    
    # Check if input folder exists
    if not os.path.exists(input_folder):
        print(f"❌ Input folder not found: {input_folder}")
        print("Please add some images to the input_images folder and try again.")
        return
    
    # Get list of image files
    image_files = [f for f in os.listdir(input_folder) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    if not image_files:
        print(f"❌ No image files found in: {input_folder}")
        print("Supported formats: PNG, JPG, JPEG, BMP, TIFF")
        return
    
    print(f"Found {len(image_files)} image(s) to process:")
    for img_file in image_files:
        print(f"  • {img_file}")
    print()
    
    # Process each image
    processed_count = 0
    for file_name in image_files:
        image_path = os.path.join(input_folder, file_name)
        print(f"🔄 Processing: {file_name}")
        
        try:
            # Generate fingerprint effect
            output_image, output_label = generate_fingerprints(image_path, enhanced_cfg)
            
            # Save the processed image
            output_path = os.path.join(output_folder, file_name)
            output_image.save(output_path)
            
            # Save the label mask
            label_path = os.path.join(label_folder, file_name)
            output_label.save(label_path)
            
            print(f"  ✅ Saved processed image: {output_path}")
            print(f"  ✅ Saved label mask: {label_path}")
            processed_count += 1
            
        except Exception as e:
            print(f"  ❌ Error processing {file_name}: {e}")
        
        print()
    
    print("-" * 50)
    print(f"🎉 Processing complete!")
    print(f"Successfully processed {processed_count}/{len(image_files)} images")
    
    if processed_count > 0:
        print(f"\n📁 Results saved to:")
        print(f"  • Processed images: {output_folder}")
        print(f"  • Label masks: {label_folder}")


if __name__ == "__main__":
    main()
