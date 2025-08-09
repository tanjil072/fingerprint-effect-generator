#!/bin/bash

# DiffUIR Fingerprint Restoration - Quick Launcher
# This script provides easy access to all functionality

echo "🎯 DiffUIR Fingerprint Restoration"
echo "=================================="
echo ""
echo "Choose an option:"
echo "1. Quick Start Demo (recommended for first run)"
echo "2. Setup Environment Only"
echo "3. Complete Evaluation Pipeline"
echo "4. Single Image Restoration"
echo "5. Benchmark Comparison"
echo "6. Results Visualization"
echo "7. Dataset Preparation"
echo "8. Help & Documentation"
echo ""
read -p "Enter your choice (1-8): " choice

case $choice in
    1)
        echo "🚀 Running Quick Start Demo..."
        python3 quick_start.py
        ;;
    2)
        echo "🔧 Setting up environment..."
        python3 setup.py
        ;;
    3)
        echo "📊 Running complete evaluation..."
        python3 evaluation/run_evaluation.py --mode complete
        ;;
    4)
        read -p "Enter path to input image: " input_image
        read -p "Enter path for output image (optional): " output_image
        if [ -z "$output_image" ]; then
            python3 evaluation/run_evaluation.py --mode single --input-image "$input_image"
        else
            python3 evaluation/run_evaluation.py --mode single --input-image "$input_image" --output-image "$output_image"
        fi
        ;;
    5)
        echo "🏆 Running benchmark comparison..."
        python3 evaluation/benchmark.py --compare-methods
        ;;
    6)
        echo "📈 Creating visualizations..."
        python3 evaluation/visualize_results.py
        ;;
    7)
        read -p "Enter source images directory (optional): " source_dir
        if [ -z "$source_dir" ]; then
            python3 evaluation/run_evaluation.py --prepare-dataset
        else
            python3 evaluation/run_evaluation.py --prepare-dataset --source-images "$source_dir"
        fi
        ;;
    8)
        echo "📚 Documentation:"
        echo "• README.md - Basic usage"
        echo "• PROJECT_OVERVIEW.md - Detailed documentation"
        echo "• config.yaml - Configuration options"
        echo ""
        echo "Key directories:"
        echo "• data/clean/ - Place your original images here"
        echo "• results/ - All evaluation outputs"
        echo "• models/diffuir_weights/ - Model weights location"
        ;;
    *)
        echo "❌ Invalid choice. Please run the script again and choose 1-8."
        ;;
esac
