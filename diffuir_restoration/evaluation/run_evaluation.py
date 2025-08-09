#!/usr/bin/env python3
"""
Main evaluation script for DiffUIR fingerprint restoration
"""

import sys
import os
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from restoration import FingerprintRestoration


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(
        description='Run DiffUIR fingerprint restoration evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config', 
        default='../config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--mode',
        choices=['single', 'dataset', 'benchmark', 'complete'],
        default='complete',
        help='Evaluation mode to run'
    )
    
    parser.add_argument(
        '--dataset-type',
        choices=['train', 'val', 'test'],
        default='test',
        help='Dataset type to evaluate on'
    )
    
    parser.add_argument(
        '--input-image',
        help='Path to input image (for single mode)'
    )
    
    parser.add_argument(
        '--output-image',
        help='Path to save output image (for single mode)'
    )
    
    parser.add_argument(
        '--skip-baselines',
        action='store_true',
        help='Skip baseline method comparisons'
    )
    
    parser.add_argument(
        '--prepare-dataset',
        action='store_true',
        help='Force dataset preparation even if it exists'
    )
    
    parser.add_argument(
        '--source-images',
        help='Directory containing source images for dataset preparation'
    )
    
    args = parser.parse_args()
    
    print("🎯 DiffUIR Fingerprint Restoration Evaluation")
    print("=" * 50)
    
    try:
        # Initialize restoration pipeline
        print(f"Loading configuration: {args.config}")
        restoration = FingerprintRestoration(args.config)
        
        # Prepare dataset if requested
        if args.prepare_dataset:
            print("📋 Preparing dataset...")
            from data_loader import DatasetPreparator
            preparator = DatasetPreparator(restoration.config)
            preparator.prepare_dataset(args.source_images)
        
        # Run evaluation based on mode
        if args.mode == 'single':
            print("🖼️  Single image restoration mode")
            
            if not args.input_image:
                raise ValueError("--input-image required for single mode")
            
            if not os.path.exists(args.input_image):
                raise FileNotFoundError(f"Input image not found: {args.input_image}")
            
            print(f"Input: {args.input_image}")
            print(f"Output: {args.output_image or 'auto-generated'}")
            
            restored_image, metrics = restoration.restore_single_image(
                args.input_image, 
                args.output_image
            )
            
            print(f"✅ Restoration complete!")
            print(f"   Processing time: {metrics['processing_time']:.3f} seconds")
        
        elif args.mode == 'dataset':
            print(f"📊 Dataset restoration mode ({args.dataset_type})")
            
            results = restoration.restore_dataset(
                args.dataset_type, 
                calculate_metrics=True
            )
            
            print(f"✅ Dataset restoration complete!")
            if results:
                for metric_name, stats in results.items():
                    print(f"   {metric_name.upper()}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        elif args.mode == 'benchmark':
            print(f"🏆 Benchmark comparison mode ({args.dataset_type})")
            
            results = restoration.benchmark_against_baselines(args.dataset_type)
            
            print(f"✅ Benchmark complete!")
            print(f"   Methods compared: {len(results)}")
            
            # Print comparison summary
            if 'diffuir' in results:
                diffuir_metrics = results['diffuir']
                print("\n📊 DiffUIR Results:")
                for metric, stats in diffuir_metrics.items():
                    if isinstance(stats, dict) and 'mean' in stats:
                        print(f"   {metric.upper()}: {stats['mean']:.4f}")
        
        elif args.mode == 'complete':
            print("🚀 Complete evaluation pipeline")
            
            results = restoration.run_complete_evaluation()
            
            print("✅ Complete evaluation finished!")
            print("📁 Check results directory for detailed outputs:")
            print("   - Restored images: data/restored/")
            print("   - Metrics: results/metrics/")
            print("   - Comparisons: results/comparisons/")
            print("   - Reports: results/reports/")
    
    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n🎉 Evaluation completed successfully!")


if __name__ == "__main__":
    main()
