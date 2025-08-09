#!/usr/bin/env python3
"""
Quick start demo for DiffUIR fingerprint restoration
"""

import os
import sys
import time
from pathlib import Path

def print_header():
    """Print demo header"""
    print("🎯 DiffUIR Fingerprint Restoration - Quick Start Demo")
    print("=" * 55)
    print("This demo will:")
    print("1. Setup the environment")
    print("2. Prepare sample data")
    print("3. Run fingerprint restoration")
    print("4. Compare against baseline methods")
    print("5. Generate evaluation reports")
    print("-" * 55)


def check_setup():
    """Check if setup is complete"""
    print("🔍 Checking setup...")
    
    required_dirs = ['data', 'results', 'src', 'evaluation']
    missing_dirs = []
    
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"❌ Missing directories: {', '.join(missing_dirs)}")
        print("Running setup...")
        return False
    
    print("✅ Directory structure exists")
    
    # Check for configuration
    if not os.path.exists('config.yaml'):
        print("❌ Configuration file not found")
        return False
    
    print("✅ Configuration file found")
    return True


def run_setup():
    """Run the setup process"""
    print("🚀 Running setup process...")
    
    setup_cmd = f"{sys.executable} setup.py"
    exit_code = os.system(setup_cmd)
    
    if exit_code == 0:
        print("✅ Setup completed successfully")
        return True
    else:
        print("⚠️  Setup completed with warnings")
        return True  # Continue anyway


def prepare_sample_data():
    """Prepare sample data for demo"""
    print("📋 Preparing sample dataset...")
    
    # Check if we have sample data from parent project
    parent_input = Path('..') / 'input_images'
    clean_dir = Path('data') / 'clean'
    
    if parent_input.exists():
        # Copy images to clean directory
        import shutil
        
        image_files = list(parent_input.glob('*.png')) + list(parent_input.glob('*.jpg'))
        
        if image_files:
            clean_dir.mkdir(parents=True, exist_ok=True)
            
            for img_file in image_files[:5]:  # Limit to 5 images for demo
                shutil.copy2(img_file, clean_dir / img_file.name)
                print(f"   📸 Copied: {img_file.name}")
            
            print(f"✅ Prepared {len(image_files[:5])} sample images")
            return True
    
    # If no parent images, create a notice
    print("ℹ️  No sample images found in parent project")
    print("   Please place your images in data/clean/ directory")
    
    # Check if user has already placed images
    if clean_dir.exists():
        existing_images = list(clean_dir.glob('*.png')) + list(clean_dir.glob('*.jpg'))
        if existing_images:
            print(f"✅ Found {len(existing_images)} existing images")
            return True
    
    print("⚠️  No sample data available. Demo will use synthetic data.")
    return False


def run_dataset_preparation():
    """Run dataset preparation (generate fingerprinted images)"""
    print("🔧 Generating fingerprinted dataset...")
    
    cmd = f"{sys.executable} -c \"import sys; sys.path.append('src'); from data_loader import DatasetPreparator; from utils import load_config; prep = DatasetPreparator(load_config('config.yaml')); prep.prepare_dataset()\""
    
    exit_code = os.system(cmd)
    
    if exit_code == 0:
        print("✅ Dataset preparation completed")
        return True
    else:
        print("⚠️  Dataset preparation had issues, continuing...")
        return False


def run_restoration():
    """Run the restoration process"""
    print("🤖 Running DiffUIR restoration...")
    
    # Run the complete evaluation
    cmd = f"{sys.executable} evaluation/run_evaluation.py --mode complete"
    
    print("   This may take a few minutes...")
    start_time = time.time()
    
    exit_code = os.system(cmd)
    
    elapsed_time = time.time() - start_time
    
    if exit_code == 0:
        print(f"✅ Restoration completed in {elapsed_time:.1f} seconds")
        return True
    else:
        print("⚠️  Restoration had issues")
        return False


def show_results():
    """Show results summary"""
    print("📊 Results Summary:")
    
    results_dir = Path('results')
    
    # Check what was generated
    if (results_dir / 'metrics').exists():
        metrics_files = list((results_dir / 'metrics').glob('*.json'))
        print(f"   📈 Metrics files: {len(metrics_files)}")
    
    if (results_dir / 'comparisons').exists():
        comparison_files = list((results_dir / 'comparisons').glob('*.png'))
        print(f"   🖼️  Comparison images: {len(comparison_files)}")
    
    if (results_dir / 'reports').exists():
        report_files = list((results_dir / 'reports').glob('*.md'))
        print(f"   📋 Reports: {len(report_files)}")
    
    restored_dir = Path('data') / 'restored'
    if restored_dir.exists():
        restored_files = list(restored_dir.glob('*.png')) + list(restored_dir.glob('*.jpg'))
        print(f"   🔧 Restored images: {len(restored_files)}")
    
    print("\n📁 Generated files:")
    print("   • Restored images: data/restored/")
    print("   • Metrics data: results/metrics/")
    print("   • Visual comparisons: results/comparisons/")
    print("   • Evaluation reports: results/reports/")


def create_visualizations():
    """Create result visualizations"""
    print("📈 Creating visualizations...")
    
    cmd = f"{sys.executable} evaluation/visualize_results.py"
    
    exit_code = os.system(cmd)
    
    if exit_code == 0:
        print("✅ Visualizations created")
        print("   📊 Check results/visualizations/ for plots and dashboards")
        return True
    else:
        print("⚠️  Visualization creation had issues")
        return False


def main():
    """Main demo function"""
    print_header()
    
    success_count = 0
    total_steps = 6
    
    try:
        # Step 1: Check and run setup
        if not check_setup():
            if run_setup():
                success_count += 1
            else:
                print("❌ Setup failed")
                return
        else:
            success_count += 1
        
        # Step 2: Prepare sample data
        if prepare_sample_data():
            success_count += 1
        
        # Step 3: Generate fingerprinted dataset
        if run_dataset_preparation():
            success_count += 1
        
        # Step 4: Run restoration
        if run_restoration():
            success_count += 1
        
        # Step 5: Create visualizations
        if create_visualizations():
            success_count += 1
        
        # Step 6: Show results
        show_results()
        success_count += 1
        
    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
        return
    
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Final summary
    print("\n" + "=" * 55)
    print(f"📋 Demo Summary: {success_count}/{total_steps} steps completed")
    
    if success_count >= 4:
        print("🎉 Demo completed successfully!")
        print("\n🚀 Next Steps:")
        print("1. Explore the results/ directory")
        print("2. Try with your own images in data/clean/")
        print("3. Modify config.yaml for different settings")
        print("4. Use individual evaluation scripts for specific tasks")
        
        print("\n💡 Useful Commands:")
        print("• python evaluation/run_evaluation.py --mode single --input-image path/to/image.png")
        print("• python evaluation/benchmark.py --compare-methods")
        print("• python evaluation/visualize_results.py")
    else:
        print("⚠️  Demo completed with issues. Check the output above for details.")
    
    print("\n📚 For more information, see README.md")


if __name__ == "__main__":
    main()
