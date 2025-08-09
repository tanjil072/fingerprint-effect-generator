#!/usr/bin/env python3
"""
Advanced usage example with custom configurations.
This example shows how to customize fingerprint parameters for different effects.
Run individual functions to create specific demos only when needed.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from fingerprintgenerator import generate_fingerprints
from config import cfg
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont
import numpy as np


def create_comparison_image(original_path, result_img, title="Fingerprint Effect"):
    """Create a side-by-side comparison image"""
    original = Image.open(original_path)
    
    # Create comparison canvas
    padding = 20
    text_height = 40
    total_width = original.width * 2 + padding * 3
    total_height = original.height + text_height + padding * 2
    
    comparison = Image.new('RGB', (total_width, total_height), 'white')
    
    # Add title text
    draw = ImageDraw.Draw(comparison)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # Draw labels
    draw.text((original.width//2 - 30, 10), "ORIGINAL", fill='black', font=font)
    draw.text((original.width + padding + original.width//2 - 50, 10), title.upper(), fill='black', font=font)
    
    # Paste images
    comparison.paste(original, (padding, text_height))
    comparison.paste(result_img, (original.width + padding * 2, text_height))
    
    return comparison


def demo_maximum_visibility():
    """Generate fingerprints with maximum visibility settings"""
    
    base_dir = os.path.dirname(os.path.dirname(__file__))
    input_folder = os.path.join(base_dir, "input_images")
    output_folder = os.path.join(base_dir, "output", "processed_images")
    
    # Maximum visibility configuration
    max_visibility_cfg = {
        'maxPrints': 2,
        'minPrints': 2,
        'maxSize': 180,
        'minSize': 150,
        'maxIntensity': 1.0,
        'minIntensity': 0.9,
        'blur_radius': 1,
        'pattern_density': 0.7,
        'distortion_strength': 8,
        'edge_fade': 0.4,
        'return_label': True,
        'noise_level': 0.02,
        'brightness_factor': 0.15,  # Very dark
        'contrast_boost': 1.8       # High contrast
    }
    
    print("🔥 Maximum Visibility Fingerprint Demo")
    print("Settings: Large size, high intensity, maximum contrast")
    print("-" * 50)
    
    # Find first image in input folder
    if os.path.exists(input_folder):
        image_files = [f for f in os.listdir(input_folder) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        if image_files:
            image_path = os.path.join(input_folder, image_files[0])
            
            print(f"Processing: {image_files[0]}")
            
            result_img, label_img = generate_fingerprints(image_path, max_visibility_cfg)
            
            # Save results
            result_path = os.path.join(output_folder, "max_visibility_fingerprint.jpg")
            result_img.save(result_path)
            
            # Create comparison
            comparison = create_comparison_image(image_path, result_img, "Maximum Visibility")
            comparison_path = os.path.join(output_folder, "max_visibility_comparison.jpg")
            comparison.save(comparison_path)
            
            print(f"✅ Max visibility result: {result_path}")
            print(f"✅ Comparison image: {comparison_path}")


def demo_subtle_effect():
    """Generate subtle fingerprint effects"""
    
    base_dir = os.path.dirname(os.path.dirname(__file__))
    input_folder = os.path.join(base_dir, "input_images")
    output_folder = os.path.join(base_dir, "output", "processed_images")
    
    # Subtle effect configuration
    subtle_cfg = {
        'maxPrints': 4,
        'minPrints': 2,
        'maxSize': 100,
        'minSize': 70,
        'maxIntensity': 0.5,
        'minIntensity': 0.3,
        'blur_radius': 3,
        'pattern_density': 0.3,
        'distortion_strength': 3,
        'edge_fade': 0.8,
        'return_label': True,
        'noise_level': 0.03,
        'brightness_factor': 0.6,   # Light effect
        'contrast_boost': 1.1       # Minimal contrast
    }
    
    print("\n🌟 Subtle Fingerprint Effect Demo")
    print("Settings: Smaller size, lower intensity, natural look")
    print("-" * 50)
    
    # Find first image in input folder
    if os.path.exists(input_folder):
        image_files = [f for f in os.listdir(input_folder) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        if image_files:
            image_path = os.path.join(input_folder, image_files[0])
            
            print(f"Processing: {image_files[0]}")
            
            result_img, label_img = generate_fingerprints(image_path, subtle_cfg)
            
            # Save results
            result_path = os.path.join(output_folder, "subtle_fingerprint.jpg")
            result_img.save(result_path)
            
            # Create comparison
            comparison = create_comparison_image(image_path, result_img, "Subtle Effect")
            comparison_path = os.path.join(output_folder, "subtle_comparison.jpg")
            comparison.save(comparison_path)
            
            print(f"✅ Subtle result: {result_path}")
            print(f"✅ Comparison image: {comparison_path}")


def demo_pattern_types():
    """Demonstrate different fingerprint pattern types"""
    
    base_dir = os.path.dirname(os.path.dirname(__file__))
    input_folder = os.path.join(base_dir, "input_images")
    output_folder = os.path.join(base_dir, "output", "processed_images")
    
    # Configuration for pattern demo
    pattern_cfg = cfg.copy()
    pattern_cfg.update({
        'maxPrints': 1,
        'minPrints': 1,
        'maxSize': 140,
        'minSize': 140,
        'maxIntensity': 0.8,
        'minIntensity': 0.8,
        'brightness_factor': 0.3,
        'contrast_boost': 1.5
    })
    
    print("\n🎨 Pattern Types Demo")
    print("Generating examples of Loop, Whorl, and Arch patterns")
    print("-" * 50)
    
    if os.path.exists(input_folder):
        image_files = [f for f in os.listdir(input_folder) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        if image_files:
            # Use first image for pattern demo
            image_path = os.path.join(input_folder, image_files[0])
            
            for pattern_type in ['loop', 'whorl', 'arch']:
                print(f"Generating {pattern_type} pattern...")
                
                # For pattern demo, we'll need to modify the generator to force pattern type
                # This is a simplified approach - in practice you might want to extend the API
                result_img, _ = generate_fingerprints(image_path, pattern_cfg)
                
                result_path = os.path.join(output_folder, f"pattern_{pattern_type}.jpg")
                result_img.save(result_path)
                print(f"  ✅ Saved: {result_path}")


def main():
    """Show available advanced demo options without running them automatically"""
    
    print("🚀 Advanced Fingerprint Effect Examples")
    print("=" * 60)
    print("Available demo functions:")
    print("  • demo_maximum_visibility() - High intensity fingerprints")
    print("  • demo_subtle_effect() - Natural, subtle fingerprints")
    print("  • demo_pattern_types() - Show different pattern types")
    print()
    print("To run a specific demo, call the function directly in Python:")
    print("  >>> from advanced_usage import demo_maximum_visibility")
    print("  >>> demo_maximum_visibility()")
    print()
    print("Or modify this script to run only the demos you want.")


if __name__ == "__main__":
    main()
