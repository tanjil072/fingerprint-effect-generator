"""
Main restoration pipeline for fingerprint removal using DiffUIR
"""

import os
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
import json
from tqdm import tqdm

from .diffuir_model import DiffUIRModel
from .metrics import MetricsCalculator, BaselineMetrics
from .data_loader import DatasetPreparator
from .utils import (
    load_config, setup_logging, setup_device, set_random_seeds,
    load_image, save_image, create_grid_comparison, get_file_list,
    ensure_directory, ProgressTracker
)


class FingerprintRestoration:
    """
    Main class for fingerprint restoration using DiffUIR
    """
    
    def __init__(self, config_path: str):
        """
        Initialize restoration pipeline
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = load_config(config_path)
        
        # Setup logging
        self.logger = setup_logging(self.config)
        self.logger.info("Initializing Fingerprint Restoration Pipeline")
        
        # Set random seeds for reproducibility
        set_random_seeds(self.config['experiment']['random_seed'])
        
        # Setup device
        self.device = setup_device(self.config)
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.metrics_calculator = MetricsCalculator(self.config)
        self.baseline_metrics = BaselineMetrics(self.config)
        
        # Initialize model
        self.model = None
        self._initialize_model()
        
        # Setup output directories
        self._setup_output_directories()
    
    def _initialize_model(self):
        """Initialize the DiffUIR model"""
        try:
            self.model = DiffUIRModel(self.config, self.device)
            model_info = self.model.get_model_info()
            self.logger.info(f"Model initialized: {model_info['model_name']}")
            self.logger.info(f"Total parameters: {model_info['total_parameters']:,}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize model: {e}")
            raise
    
    def _setup_output_directories(self):
        """Setup output directories"""
        base_dir = self.config['output']['results_dir']
        
        self.output_dirs = {
            'metrics': os.path.join(base_dir, self.config['output']['metrics_dir']),
            'comparisons': os.path.join(base_dir, self.config['output']['comparisons_dir']),
            'reports': os.path.join(base_dir, self.config['output']['reports_dir']),
            'restored': self.config['data']['restored_images_path']
        }
        
        for dir_path in self.output_dirs.values():
            ensure_directory(dir_path)
    
    def restore_single_image(self, 
                           image_path: str, 
                           output_path: Optional[str] = None,
                           save_result: bool = True) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Restore a single image
        
        Args:
            image_path: Path to degraded image
            output_path: Path to save restored image
            save_result: Whether to save the result
            
        Returns:
            Tuple of (restored_image, metrics)
        """
        self.logger.info(f"Restoring image: {os.path.basename(image_path)}")
        
        # Load degraded image
        degraded_image = load_image(image_path)
        
        # Restore image
        restored_image, processing_time = self.model.restore_image(degraded_image)
        
        # Calculate metrics if clean image is available
        metrics = {'processing_time': processing_time}
        
        # Save result if requested
        if save_result:
            if output_path is None:
                filename = os.path.basename(image_path)
                output_path = os.path.join(self.output_dirs['restored'], filename)
            
            save_image(restored_image, output_path)
            self.logger.info(f"Saved restored image to: {output_path}")
        
        return restored_image, metrics
    
    def restore_dataset(self, 
                       dataset_type: str = 'test',
                       calculate_metrics: bool = True) -> Dict[str, float]:
        """
        Restore entire dataset and calculate metrics
        
        Args:
            dataset_type: Type of dataset ('train', 'val', 'test')
            calculate_metrics: Whether to calculate quality metrics
            
        Returns:
            Aggregated metrics
        """
        self.logger.info(f"Restoring {dataset_type} dataset")
        
        # Prepare dataset
        preparator = DatasetPreparator(self.config)
        
        # Get image pairs
        if dataset_type == 'test':
            _, _, test_dataset = preparator.get_paired_datasets()
            dataset = test_dataset
        elif dataset_type == 'val':
            _, val_dataset, _ = preparator.get_paired_datasets()
            dataset = val_dataset
        elif dataset_type == 'train':
            train_dataset, _, _ = preparator.get_paired_datasets()
            dataset = train_dataset
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        all_metrics = []
        
        # Progress tracking
        progress = ProgressTracker(len(dataset), f"Restoring {dataset_type} images")
        
        for i in range(len(dataset)):
            # Get data
            data_item = dataset[i]
            filename = data_item['filename']
            
            # Load images as numpy arrays
            clean_path = dataset.clean_images[i]
            fingerprinted_path = dataset.fingerprinted_images[i]
            
            clean_image = load_image(clean_path)
            degraded_image = load_image(fingerprinted_path)
            
            # Restore image
            restored_image, processing_time = self.model.restore_image(degraded_image)
            
            # Calculate metrics if requested
            if calculate_metrics:
                metrics = self.metrics_calculator.calculate_all_metrics(
                    clean_image, restored_image, processing_time
                )
                all_metrics.append(metrics)
            
            # Save restored image
            restored_path = os.path.join(self.output_dirs['restored'], filename)
            save_image(restored_image, restored_path)
            
            # Create comparison if requested
            if self.config['evaluation'].get('save_comparisons', True):
                self._save_comparison(
                    clean_image, degraded_image, restored_image,
                    filename, dataset_type
                )
            
            progress.update()
        
        progress.finish()
        
        # Aggregate metrics
        if calculate_metrics and all_metrics:
            aggregated_metrics = self.metrics_calculator.aggregate_metrics(all_metrics)
            
            # Save metrics
            metrics_filename = f"{dataset_type}_diffuir_metrics.json"
            self.metrics_calculator.save_metrics(
                aggregated_metrics, metrics_filename
            )
            
            # Log results
            self.logger.info(f"Restoration complete for {len(all_metrics)} images")
            for metric_name, stats in aggregated_metrics.items():
                self.logger.info(f"{metric_name.upper()}: {stats['mean']:.4f} ± {stats['std']:.4f}")
            
            return aggregated_metrics
        
        return {}
    
    def _save_comparison(self, 
                        clean: np.ndarray,
                        degraded: np.ndarray, 
                        restored: np.ndarray,
                        filename: str,
                        dataset_type: str):
        """Save side-by-side comparison"""
        try:
            comparison = create_grid_comparison(
                [degraded, restored, clean],
                ['Degraded', 'Restored', 'Clean Ground Truth'],
                rows=1
            )
            
            comparison_filename = f"{dataset_type}_{filename}"
            comparison_path = os.path.join(self.output_dirs['comparisons'], comparison_filename)
            save_image(comparison, comparison_path, denormalize=False)
            
        except Exception as e:
            self.logger.warning(f"Could not save comparison for {filename}: {e}")
    
    def benchmark_against_baselines(self, dataset_type: str = 'test') -> Dict[str, Dict[str, float]]:
        """
        Benchmark DiffUIR against baseline methods
        
        Args:
            dataset_type: Dataset to use for benchmarking
            
        Returns:
            Comparison results
        """
        self.logger.info(f"Running benchmark comparison on {dataset_type} dataset")
        
        # Get DiffUIR results
        diffuir_metrics = self.restore_dataset(dataset_type, calculate_metrics=True)
        
        # Prepare dataset for baseline evaluation
        preparator = DatasetPreparator(self.config)
        
        if dataset_type == 'test':
            _, _, test_dataset = preparator.get_paired_datasets()
            dataset = test_dataset
        else:
            # Handle other dataset types if needed
            _, _, test_dataset = preparator.get_paired_datasets()
            dataset = test_dataset
        
        # Load all images for baseline evaluation
        clean_images = []
        degraded_images = []
        
        for i in range(len(dataset)):
            clean_path = dataset.clean_images[i]
            degraded_path = dataset.fingerprinted_images[i]
            
            clean_images.append(load_image(clean_path))
            degraded_images.append(load_image(degraded_path))
        
        # Evaluate baselines
        baseline_results = {}
        
        if self.config['baselines'].get('enabled', True):
            baseline_methods = self.config['baselines']['methods']
            
            for method_config in baseline_methods:
                method_name = method_config['name']
                method_params = method_config.get('params', {})
                
                self.logger.info(f"Evaluating baseline: {method_name}")
                
                baseline_metrics = self.baseline_metrics.evaluate_baseline(
                    method_name, clean_images, degraded_images, **method_params
                )
                
                baseline_results[method_name] = baseline_metrics
        
        # Combine all results
        all_results = {
            'diffuir': diffuir_metrics,
            **baseline_results
        }
        
        # Save comparison results
        comparison_filename = f"{dataset_type}_benchmark_comparison.json"
        comparison_path = os.path.join(self.output_dirs['metrics'], comparison_filename)
        
        with open(comparison_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Create comparison report
        self._create_benchmark_report(all_results, dataset_type)
        
        return all_results
    
    def _create_benchmark_report(self, results: Dict[str, Dict], dataset_type: str):
        """Create formatted benchmark report"""
        report = f"# Fingerprint Restoration Benchmark Report\n\n"
        report += f"Dataset: {dataset_type}\n"
        report += f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Create comparison table
        if 'psnr' in results.get('diffuir', {}):
            report += "## Performance Comparison\n\n"
            report += "| Method | PSNR (dB) | SSIM | Processing Time (s) |\n"
            report += "|--------|-----------|------|--------------------|\n"
            
            for method_name, metrics in results.items():
                psnr = metrics.get('psnr', {}).get('mean', 0)
                ssim = metrics.get('ssim', {}).get('mean', 0)
                time_val = metrics.get('processing_time', {}).get('mean', 0)
                
                report += f"| {method_name} | {psnr:.2f} | {ssim:.4f} | {time_val:.3f} |\n"
            
            report += "\n"
        
        # Detailed metrics
        for method_name, metrics in results.items():
            report += f"## {method_name.upper()} Detailed Results\n\n"
            
            for metric_name, stats in metrics.items():
                if isinstance(stats, dict) and 'mean' in stats:
                    report += f"**{metric_name.upper()}**\n"
                    report += f"- Mean: {stats['mean']:.4f}\n"
                    report += f"- Std:  {stats['std']:.4f}\n"
                    report += f"- Min:  {stats['min']:.4f}\n"
                    report += f"- Max:  {stats['max']:.4f}\n\n"
        
        # Save report
        report_filename = f"{dataset_type}_benchmark_report.md"
        report_path = os.path.join(self.output_dirs['reports'], report_filename)
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        self.logger.info(f"Benchmark report saved to: {report_path}")
    
    def run_complete_evaluation(self) -> Dict:
        """
        Run complete evaluation pipeline
        
        Returns:
            Complete evaluation results
        """
        self.logger.info("🚀 Starting complete evaluation pipeline")
        
        results = {}
        
        try:
            # 1. Dataset preparation (if needed)
            self.logger.info("Step 1: Preparing dataset")
            preparator = DatasetPreparator(self.config)
            
            # Check if dataset exists, prepare if not
            clean_dir = self.config['data']['clean_images_path']
            if not os.path.exists(clean_dir) or not get_file_list(clean_dir):
                preparator.prepare_dataset()
            
            # 2. Restore test dataset with DiffUIR
            self.logger.info("Step 2: Restoring with DiffUIR")
            diffuir_results = self.restore_dataset('test', calculate_metrics=True)
            results['diffuir'] = diffuir_results
            
            # 3. Benchmark against baselines
            self.logger.info("Step 3: Benchmarking against baselines")
            benchmark_results = self.benchmark_against_baselines('test')
            results['benchmark'] = benchmark_results
            
            # 4. Create final summary
            self.logger.info("Step 4: Creating evaluation summary")
            self._create_evaluation_summary(results)
            
            self.logger.info("✅ Complete evaluation finished successfully")
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            raise
        
        return results
    
    def _create_evaluation_summary(self, results: Dict):
        """Create final evaluation summary"""
        summary_path = os.path.join(self.output_dirs['reports'], 'evaluation_summary.md')
        
        summary = "# Fingerprint Restoration Evaluation Summary\n\n"
        summary += f"Experiment: {self.config['experiment']['name']}\n"
        summary += f"Version: {self.config['experiment']['version']}\n"
        summary += f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Model information
        if self.model:
            model_info = self.model.get_model_info()
            summary += "## Model Information\n"
            for key, value in model_info.items():
                summary += f"- {key}: {value}\n"
            summary += "\n"
        
        # Best results summary
        if 'benchmark' in results:
            benchmark_data = results['benchmark']
            
            summary += "## Performance Summary\n\n"
            
            # Find best PSNR and SSIM
            best_psnr = 0
            best_ssim = 0
            best_psnr_method = ""
            best_ssim_method = ""
            
            for method, metrics in benchmark_data.items():
                if 'psnr' in metrics and 'mean' in metrics['psnr']:
                    psnr_val = metrics['psnr']['mean']
                    if psnr_val > best_psnr:
                        best_psnr = psnr_val
                        best_psnr_method = method
                
                if 'ssim' in metrics and 'mean' in metrics['ssim']:
                    ssim_val = metrics['ssim']['mean']
                    if ssim_val > best_ssim:
                        best_ssim = ssim_val
                        best_ssim_method = method
            
            summary += f"- **Best PSNR**: {best_psnr:.2f} dB ({best_psnr_method})\n"
            summary += f"- **Best SSIM**: {best_ssim:.4f} ({best_ssim_method})\n\n"
        
        summary += "## Files Generated\n"
        summary += "- Restored images: `data/restored/`\n"
        summary += "- Metrics: `results/metrics/`\n"
        summary += "- Comparisons: `results/comparisons/`\n"
        summary += "- Reports: `results/reports/`\n\n"
        
        summary += "---\n"
        summary += "*Generated by DiffUIR Fingerprint Restoration Evaluation*"
        
        with open(summary_path, 'w') as f:
            f.write(summary)
        
        self.logger.info(f"Evaluation summary saved to: {summary_path}")


def main():
    """Main function for CLI usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='DiffUIR Fingerprint Restoration')
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    parser.add_argument('--mode', choices=['single', 'dataset', 'benchmark', 'complete'], 
                       default='complete', help='Evaluation mode')
    parser.add_argument('--input', help='Input image path (for single mode)')
    parser.add_argument('--output', help='Output image path (for single mode)')
    
    args = parser.parse_args()
    
    # Initialize restoration pipeline
    restoration = FingerprintRestoration(args.config)
    
    if args.mode == 'single':
        if not args.input:
            raise ValueError("Input image path required for single mode")
        
        restored_image, metrics = restoration.restore_single_image(
            args.input, args.output
        )
        print(f"Image restored. Processing time: {metrics['processing_time']:.3f}s")
        
    elif args.mode == 'dataset':
        results = restoration.restore_dataset('test')
        print(f"Dataset restoration complete: {len(results)} metrics calculated")
        
    elif args.mode == 'benchmark':
        results = restoration.benchmark_against_baselines('test')
        print(f"Benchmark complete: {len(results)} methods compared")
        
    elif args.mode == 'complete':
        results = restoration.run_complete_evaluation()
        print("Complete evaluation finished successfully")
    
    print("✅ Done!")


if __name__ == "__main__":
    main()
