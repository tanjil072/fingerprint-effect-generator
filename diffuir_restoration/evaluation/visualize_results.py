#!/usr/bin/env python3
"""
Results visualization script for fingerprint restoration evaluation
"""

import sys
import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


def load_results(results_dir: str) -> Dict:
    """Load all available results from directory"""
    results = {}
    
    # Look for JSON files
    results_path = Path(results_dir)
    for json_file in results_path.glob("*metrics.json"):
        method_name = json_file.stem.replace('_metrics', '')
        try:
            with open(json_file, 'r') as f:
                results[method_name] = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")
    
    return results


def create_metric_evolution_plot(results: Dict, output_dir: str):
    """Create plots showing metric evolution across different methods"""
    
    if not results:
        print("No results to plot")
        return
    
    # Extract metrics data
    methods = list(results.keys())
    metrics_data = {}
    
    for method, data in results.items():
        for metric_name, metric_stats in data.items():
            if isinstance(metric_stats, dict) and 'mean' in metric_stats:
                if metric_name not in metrics_data:
                    metrics_data[metric_name] = {'methods': [], 'means': [], 'stds': []}
                
                metrics_data[metric_name]['methods'].append(method)
                metrics_data[metric_name]['means'].append(metric_stats['mean'])
                metrics_data[metric_name]['stds'].append(metric_stats.get('std', 0))
    
    # Create subplots for each metric
    num_metrics = len(metrics_data)
    if num_metrics == 0:
        print("No metrics data found for plotting")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten() if num_metrics > 1 else [axes]
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
    
    for i, (metric_name, metric_data) in enumerate(metrics_data.items()):
        if i >= len(axes):
            break
        
        ax = axes[i]
        
        # Bar plot with error bars
        x_pos = np.arange(len(metric_data['methods']))
        bars = ax.bar(x_pos, metric_data['means'], yerr=metric_data['stds'], 
                     capsize=5, color=colors[:len(metric_data['methods'])])
        
        ax.set_xlabel('Methods')
        ax.set_ylabel(metric_name.upper())
        ax.set_title(f'{metric_name.upper()} Comparison')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(metric_data['methods'], rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, mean_val, std_val in zip(bars, metric_data['means'], metric_data['stds']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std_val,
                   f'{mean_val:.3f}', ha='center', va='bottom', fontsize=8)
        
        ax.grid(axis='y', alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(metrics_data), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'metrics_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Metrics comparison plot saved to: {plot_path}")


def create_summary_dashboard(results: Dict, output_dir: str):
    """Create a comprehensive dashboard with all visualizations"""
    
    if not results:
        print("No results available for dashboard")
        return
    
    # Set up the figure
    fig = plt.figure(figsize=(20, 16))
    
    # Create a grid layout
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle('Fingerprint Restoration Evaluation Dashboard', 
                fontsize=20, fontweight='bold', y=0.95)
    
    # Extract data
    methods = list(results.keys())
    
    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    
    # 1. PSNR comparison (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    psnr_data = []
    psnr_methods = []
    for method, data in results.items():
        if 'psnr' in data and 'mean' in data['psnr']:
            psnr_data.append(data['psnr']['mean'])
            psnr_methods.append(method)
    
    if psnr_data:
        bars1 = ax1.bar(psnr_methods, psnr_data, color=colors[:len(psnr_data)])
        ax1.set_title('PSNR (dB)', fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        for i, v in enumerate(psnr_data):
            ax1.text(i, v + 0.1, f'{v:.2f}', ha='center', va='bottom', fontsize=8)
    
    # 2. SSIM comparison (top-right)
    ax2 = fig.add_subplot(gs[0, 1])
    ssim_data = []
    ssim_methods = []
    for method, data in results.items():
        if 'ssim' in data and 'mean' in data['ssim']:
            ssim_data.append(data['ssim']['mean'])
            ssim_methods.append(method)
    
    if ssim_data:
        bars2 = ax2.bar(ssim_methods, ssim_data, color=colors[:len(ssim_data)])
        ax2.set_title('SSIM', fontweight='bold')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)
        for i, v in enumerate(ssim_data):
            ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 3. Processing time comparison (bottom-left)
    ax3 = fig.add_subplot(gs[0, 2])
    time_data = []
    time_methods = []
    for method, data in results.items():
        if 'processing_time' in data and 'mean' in data['processing_time']:
            time_data.append(data['processing_time']['mean'])
            time_methods.append(method)
    
    if time_data:
        bars3 = ax3.bar(time_methods, time_data, color=colors[:len(time_data)])
        ax3.set_title('Processing Time (s)', fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        for i, v in enumerate(time_data):
            ax3.text(i, v + max(time_data) * 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 4. Combined scatter plot (top-right)
    ax4 = fig.add_subplot(gs[0, 3])
    if psnr_data and ssim_data and len(psnr_data) == len(ssim_data):
        scatter = ax4.scatter(psnr_data, ssim_data, c=colors[:len(psnr_data)], s=100, alpha=0.7)
        ax4.set_xlabel('PSNR (dB)')
        ax4.set_ylabel('SSIM')
        ax4.set_title('Quality Comparison', fontweight='bold')
        
        # Add method labels
        for i, method in enumerate(psnr_methods):
            ax4.annotate(method, (psnr_data[i], ssim_data[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 5. Detailed metrics table (bottom half)
    ax5 = fig.add_subplot(gs[1:3, :])
    ax5.axis('off')
    
    # Create table data
    table_data = []
    headers = ['Method', 'PSNR (dB)', 'SSIM', 'Processing Time (s)', 'Quality Score']
    
    for method, data in results.items():
        psnr_val = data.get('psnr', {}).get('mean', 0)
        ssim_val = data.get('ssim', {}).get('mean', 0)
        time_val = data.get('processing_time', {}).get('mean', 0)
        
        # Simple quality score
        quality_score = (psnr_val / 30) * 0.5 + ssim_val * 0.5
        
        table_data.append([
            method.replace('_', ' ').title(),
            f'{psnr_val:.2f}',
            f'{ssim_val:.4f}',
            f'{time_val:.3f}',
            f'{quality_score:.3f}'
        ])
    
    # Sort by quality score
    table_data.sort(key=lambda x: float(x[-1]), reverse=True)
    
    # Create table
    table = ax5.table(cellText=table_data, colLabels=headers, 
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color the header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color the best row
    if table_data:
        for i in range(len(headers)):
            table[(1, i)].set_facecolor('#92D050')  # Light green for best
    
    ax5.set_title('Detailed Performance Metrics', fontweight='bold', fontsize=14, pad=20)
    
    # 6. Summary statistics (bottom)
    ax6 = fig.add_subplot(gs[3, :])
    ax6.axis('off')
    
    # Calculate summary statistics
    summary_text = f"""
    Evaluation Summary:
    • Total methods compared: {len(methods)}
    • Best PSNR: {max(psnr_data):.2f} dB ({psnr_methods[psnr_data.index(max(psnr_data))]})
    • Best SSIM: {max(ssim_data):.4f} ({ssim_methods[ssim_data.index(max(ssim_data))]})
    • Fastest method: {time_methods[time_data.index(min(time_data))]} ({min(time_data):.3f}s)
    """
    
    ax6.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.7))
    
    # Save dashboard
    dashboard_path = os.path.join(output_dir, 'evaluation_dashboard.png')
    plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Evaluation dashboard saved to: {dashboard_path}")


def main():
    """Main visualization function"""
    parser = argparse.ArgumentParser(
        description='Visualize fingerprint restoration results',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--results-dir',
        default='../results/metrics',
        help='Directory containing results JSON files'
    )
    
    parser.add_argument(
        '--output-dir',
        default='../results/visualizations',
        help='Directory to save visualization outputs'
    )
    
    parser.add_argument(
        '--dashboard-only',
        action='store_true',
        help='Only create the dashboard (skip other plots)'
    )
    
    args = parser.parse_args()
    
    print("📊 Fingerprint Restoration Results Visualization")
    print("=" * 50)
    
    try:
        # Load results
        print(f"📂 Loading results from: {args.results_dir}")
        results = load_results(args.results_dir)
        
        if not results:
            print(f"❌ No results found in {args.results_dir}")
            print("Make sure to run evaluation first to generate results")
            sys.exit(1)
        
        print(f"✅ Loaded results for {len(results)} methods")
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Create visualizations
        if not args.dashboard_only:
            print("📈 Creating metric comparison plots...")
            create_metric_evolution_plot(results, args.output_dir)
        
        print("📊 Creating evaluation dashboard...")
        create_summary_dashboard(results, args.output_dir)
        
        print(f"\n✅ Visualizations complete!")
        print(f"📁 Outputs saved to: {args.output_dir}")
    
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
