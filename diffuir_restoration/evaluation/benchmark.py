#!/usr/bin/env python3
"""
Benchmark script for comparing DiffUIR against baseline methods
"""

import sys
import os
import argparse
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


def load_benchmark_results(results_file: str) -> Dict:
    """Load benchmark results from JSON file"""
    with open(results_file, 'r') as f:
        return json.load(f)


def create_comparison_plots(results: Dict, output_dir: str):
    """Create comparison plots for benchmark results"""
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Extract data for plotting
    methods = []
    psnr_values = []
    ssim_values = []
    time_values = []
    
    for method_name, metrics in results.items():
        if isinstance(metrics, dict):
            methods.append(method_name.replace('_', ' ').title())
            
            psnr_values.append(metrics.get('psnr', {}).get('mean', 0))
            ssim_values.append(metrics.get('ssim', {}).get('mean', 0))
            time_values.append(metrics.get('processing_time', {}).get('mean', 0))
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Fingerprint Restoration Method Comparison', fontsize=16, fontweight='bold')
    
    # PSNR comparison
    if any(psnr_values):
        axes[0, 0].bar(methods, psnr_values)
        axes[0, 0].set_title('PSNR Comparison (Higher is Better)')
        axes[0, 0].set_ylabel('PSNR (dB)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(psnr_values):
            axes[0, 0].text(i, v + 0.1, f'{v:.2f}', ha='center', va='bottom')
    
    # SSIM comparison
    if any(ssim_values):
        axes[0, 1].bar(methods, ssim_values, color='orange')
        axes[0, 1].set_title('SSIM Comparison (Higher is Better)')
        axes[0, 1].set_ylabel('SSIM')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(ssim_values):
            axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # Processing time comparison
    if any(time_values):
        axes[1, 0].bar(methods, time_values, color='green')
        axes[1, 0].set_title('Processing Time Comparison (Lower is Better)')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(time_values):
            axes[1, 0].text(i, v + max(time_values) * 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # Combined scatter plot (PSNR vs SSIM)
    if any(psnr_values) and any(ssim_values):
        scatter = axes[1, 1].scatter(psnr_values, ssim_values, s=100, alpha=0.7)
        axes[1, 1].set_title('PSNR vs SSIM (Top-right is Better)')
        axes[1, 1].set_xlabel('PSNR (dB)')
        axes[1, 1].set_ylabel('SSIM')
        
        # Add method labels
        for i, method in enumerate(methods):
            axes[1, 1].annotate(method, (psnr_values[i], ssim_values[i]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'benchmark_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Comparison plots saved to: {plot_path}")


def create_performance_table(results: Dict, output_dir: str):
    """Create performance comparison table"""
    
    table_rows = []
    headers = ['Method', 'PSNR (dB)', 'SSIM', 'Processing Time (s)', 'Rank Score']
    
    # Calculate rank scores (higher is better for PSNR/SSIM, lower is better for time)
    method_scores = []
    
    for method_name, metrics in results.items():
        if isinstance(metrics, dict):
            psnr = metrics.get('psnr', {}).get('mean', 0)
            ssim = metrics.get('ssim', {}).get('mean', 0)
            time_val = metrics.get('processing_time', {}).get('mean', float('inf'))
            
            # Simple ranking score (can be improved)
            score = psnr * 10 + ssim * 100 - min(time_val, 10) * 5
            
            method_scores.append({
                'method': method_name.replace('_', ' ').title(),
                'psnr': psnr,
                'ssim': ssim,
                'time': time_val,
                'score': score
            })
    
    # Sort by score (descending)
    method_scores.sort(key=lambda x: x['score'], reverse=True)
    
    # Create table
    table_content = "# Performance Comparison Table\n\n"
    table_content += f"Generated on: {os.popen('date').read().strip()}\n\n"
    table_content += "| " + " | ".join(headers) + " |\n"
    table_content += "|" + "|".join(["---" for _ in headers]) + "|\n"
    
    for i, item in enumerate(method_scores):
        rank_indicator = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else ""
        
        table_content += f"| {item['method']} {rank_indicator} | "
        table_content += f"{item['psnr']:.2f} | "
        table_content += f"{item['ssim']:.4f} | "
        table_content += f"{item['time']:.3f} | "
        table_content += f"{item['score']:.1f} |\n"
    
    table_content += "\n## Ranking Criteria\n"
    table_content += "- **Rank Score**: Weighted combination of PSNR (×10) + SSIM (×100) - Processing Time (×5)\n"
    table_content += "- **Higher scores are better**\n"
    table_content += "- PSNR and SSIM contribute positively, processing time contributes negatively\n"
    
    # Save table
    table_path = os.path.join(output_dir, 'performance_table.md')
    with open(table_path, 'w') as f:
        f.write(table_content)
    
    print(f"📋 Performance table saved to: {table_path}")
    
    # Also print to console
    print("\n📊 Performance Ranking:")
    for i, item in enumerate(method_scores[:5], 1):  # Top 5
        print(f"{i}. {item['method']}: PSNR={item['psnr']:.2f}, SSIM={item['ssim']:.4f}")


def main():
    """Main benchmark function"""
    parser = argparse.ArgumentParser(
        description='Benchmark DiffUIR against baseline methods',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--results-file',
        help='Path to benchmark results JSON file (if not provided, will run benchmark)'
    )
    
    parser.add_argument(
        '--config',
        default='../config.yaml',
        help='Configuration file path'
    )
    
    parser.add_argument(
        '--output-dir',
        default='../results/comparisons',
        help='Directory to save comparison outputs'
    )
    
    parser.add_argument(
        '--dataset-type',
        choices=['train', 'val', 'test'],
        default='test',
        help='Dataset type to benchmark on'
    )
    
    parser.add_argument(
        '--compare-methods',
        action='store_true',
        help='Run benchmark comparison (if no results file provided)'
    )
    
    parser.add_argument(
        '--plot-only',
        action='store_true',
        help='Only create plots from existing results file'
    )
    
    args = parser.parse_args()
    
    print("🏆 DiffUIR Benchmark Comparison")
    print("=" * 40)
    
    try:
        # Load or generate results
        if args.results_file and os.path.exists(args.results_file):
            print(f"📊 Loading existing results: {args.results_file}")
            results = load_benchmark_results(args.results_file)
        
        elif args.compare_methods or not args.plot_only:
            print("🚀 Running benchmark comparison...")
            
            # Import and run restoration
            from restoration import FingerprintRestoration
            
            restoration = FingerprintRestoration(args.config)
            results = restoration.benchmark_against_baselines(args.dataset_type)
        
        else:
            raise ValueError("No results file provided and --compare-methods not specified")
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Generate comparison plots
        print("📈 Creating comparison plots...")
        create_comparison_plots(results, args.output_dir)
        
        # Generate performance table
        print("📋 Creating performance table...")
        create_performance_table(results, args.output_dir)
        
        # Save results if not from file
        if not args.results_file:
            results_output = os.path.join(args.output_dir, f'{args.dataset_type}_benchmark_results.json')
            with open(results_output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"💾 Results saved to: {results_output}")
        
        print("\n✅ Benchmark analysis complete!")
        print(f"📁 Outputs saved to: {args.output_dir}")
    
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
