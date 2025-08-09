"""
Metrics calculation for fingerprint restoration evaluation
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import time
import json
import os


class MetricsCalculator:
    """
    Calculate restoration quality metrics including PSNR and SSIM
    """
    
    def __init__(self, config: Dict):
        """
        Initialize metrics calculator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.evaluation_config = config['evaluation']
        self.metrics_enabled = self.evaluation_config['metrics']
        
        # PSNR settings
        self.psnr_max_val = self.evaluation_config.get('psnr_max_val', 1.0)
        
        # SSIM settings
        self.ssim_window_size = self.evaluation_config.get('ssim_window_size', 11)
        self.ssim_k1 = self.evaluation_config.get('ssim_k1', 0.01)
        self.ssim_k2 = self.evaluation_config.get('ssim_k2', 0.03)
    
    def calculate_psnr(self, clean: np.ndarray, restored: np.ndarray) -> float:
        """
        Calculate Peak Signal-to-Noise Ratio
        
        Args:
            clean: Ground truth image [H, W, C]
            restored: Restored image [H, W, C]
            
        Returns:
            PSNR value in dB
        """
        return float(peak_signal_noise_ratio(
            clean, restored, 
            data_range=self.psnr_max_val
        ))
    
    def calculate_ssim(self, clean: np.ndarray, restored: np.ndarray) -> float:
        """
        Calculate Structural Similarity Index
        
        Args:
            clean: Ground truth image [H, W, C]
            restored: Restored image [H, W, C]
            
        Returns:
            SSIM value (0-1)
        """
        # Handle multi-channel images
        if clean.ndim == 3 and clean.shape[-1] == 3:
            return float(structural_similarity(
                clean, restored,
                win_size=self.ssim_window_size,
                k1=self.ssim_k1,
                k2=self.ssim_k2,
                data_range=self.psnr_max_val,
                channel_axis=-1
            ))
        else:
            return float(structural_similarity(
                clean, restored,
                win_size=self.ssim_window_size,
                k1=self.ssim_k1,
                k2=self.ssim_k2,
                data_range=self.psnr_max_val
            ))
    
    def calculate_lpips(self, clean: np.ndarray, restored: np.ndarray) -> float:
        """
        Calculate LPIPS (Learned Perceptual Image Patch Similarity)
        Note: This is a placeholder - actual LPIPS requires a pre-trained network
        
        Args:
            clean: Ground truth image [H, W, C]
            restored: Restored image [H, W, C]
            
        Returns:
            LPIPS value (lower is better)
        """
        # Placeholder implementation - returns MSE as perceptual proxy
        mse = np.mean((clean - restored) ** 2)
        return float(mse)
    
    def calculate_all_metrics(self, 
                            clean: np.ndarray, 
                            restored: np.ndarray, 
                            processing_time: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate all enabled metrics
        
        Args:
            clean: Ground truth image [H, W, C]
            restored: Restored image [H, W, C]
            processing_time: Time taken for restoration in seconds
            
        Returns:
            Dictionary of metric values
        """
        metrics = {}
        
        if 'psnr' in self.metrics_enabled:
            metrics['psnr'] = self.calculate_psnr(clean, restored)
            
        if 'ssim' in self.metrics_enabled:
            metrics['ssim'] = self.calculate_ssim(clean, restored)
            
        if 'lpips' in self.metrics_enabled:
            metrics['lpips'] = self.calculate_lpips(clean, restored)
            
        if 'processing_time' in self.metrics_enabled and processing_time is not None:
            metrics['processing_time'] = processing_time
        
        return metrics
    
    def aggregate_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Aggregate metrics across multiple images
        
        Args:
            metrics_list: List of metric dictionaries
            
        Returns:
            Aggregated statistics (mean, std, min, max)
        """
        if not metrics_list:
            return {}
        
        aggregated = {}
        
        # Get all metric names
        metric_names = metrics_list[0].keys()
        
        for metric_name in metric_names:
            values = [m[metric_name] for m in metrics_list if metric_name in m]
            
            if values:
                aggregated[metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'count': len(values)
                }
        
        return aggregated
    
    def save_metrics(self, 
                    metrics: Dict, 
                    filename: str, 
                    results_dir: Optional[str] = None):
        """
        Save metrics to JSON file
        
        Args:
            metrics: Metrics dictionary to save
            filename: Output filename
            results_dir: Results directory (uses config if None)
        """
        if results_dir is None:
            results_dir = os.path.join(
                self.config['output']['results_dir'],
                self.config['output']['metrics_dir']
            )
        
        os.makedirs(results_dir, exist_ok=True)
        
        filepath = os.path.join(results_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def create_metrics_report(self, 
                            aggregated_metrics: Dict[str, Dict[str, float]],
                            method_name: str = "DiffUIR") -> str:
        """
        Create a formatted metrics report
        
        Args:
            aggregated_metrics: Aggregated metrics from multiple images
            method_name: Name of the restoration method
            
        Returns:
            Formatted report string
        """
        report = f"# {method_name} Restoration Metrics Report\n\n"
        report += f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        for metric_name, stats in aggregated_metrics.items():
            report += f"## {metric_name.upper()}\n"
            report += f"- Mean: {stats['mean']:.4f}\n"
            report += f"- Std:  {stats['std']:.4f}\n"
            report += f"- Min:  {stats['min']:.4f}\n"
            report += f"- Max:  {stats['max']:.4f}\n"
            report += f"- Count: {stats['count']}\n\n"
        
        return report


class BaselineMetrics:
    """
    Calculate metrics for baseline restoration methods
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.calculator = MetricsCalculator(config)
    
    def gaussian_blur_restoration(self, degraded_image: np.ndarray, **params) -> np.ndarray:
        """Apply Gaussian blur as restoration method"""
        import cv2
        kernel_size = params.get('kernel_size', 5)
        sigma = params.get('sigma', 1.0)
        
        return cv2.GaussianBlur(degraded_image, (kernel_size, kernel_size), sigma)
    
    def median_filter_restoration(self, degraded_image: np.ndarray, **params) -> np.ndarray:
        """Apply median filter as restoration method"""
        import cv2
        kernel_size = params.get('kernel_size', 5)
        
        # Convert to uint8 for cv2.medianBlur
        image_uint8 = (degraded_image * 255).astype(np.uint8)
        restored_uint8 = cv2.medianBlur(image_uint8, kernel_size)
        
        return restored_uint8.astype(np.float32) / 255.0
    
    def bilateral_filter_restoration(self, degraded_image: np.ndarray, **params) -> np.ndarray:
        """Apply bilateral filter as restoration method"""
        import cv2
        d = params.get('d', 9)
        sigma_color = params.get('sigma_color', 75)
        sigma_space = params.get('sigma_space', 75)
        
        # Convert to uint8 for cv2.bilateralFilter
        image_uint8 = (degraded_image * 255).astype(np.uint8)
        restored_uint8 = cv2.bilateralFilter(image_uint8, d, sigma_color, sigma_space)
        
        return restored_uint8.astype(np.float32) / 255.0
    
    def non_local_means_restoration(self, degraded_image: np.ndarray, **params) -> np.ndarray:
        """Apply non-local means denoising as restoration method"""
        import cv2
        h = params.get('h', 10)
        template_window_size = params.get('template_window_size', 7)
        search_window_size = params.get('search_window_size', 21)
        
        # Convert to uint8
        image_uint8 = (degraded_image * 255).astype(np.uint8)
        
        if len(image_uint8.shape) == 3:  # Color image
            restored_uint8 = cv2.fastNlMeansDenoisingColored(
                image_uint8, None, h, h, template_window_size, search_window_size
            )
        else:  # Grayscale
            restored_uint8 = cv2.fastNlMeansDenoising(
                image_uint8, None, h, template_window_size, search_window_size
            )
        
        return restored_uint8.astype(np.float32) / 255.0
    
    def evaluate_baseline(self, 
                         method_name: str, 
                         clean_images: List[np.ndarray],
                         degraded_images: List[np.ndarray],
                         **method_params) -> Dict[str, float]:
        """
        Evaluate a baseline restoration method
        
        Args:
            method_name: Name of the method
            clean_images: List of ground truth images
            degraded_images: List of degraded images
            **method_params: Parameters for the restoration method
            
        Returns:
            Aggregated metrics
        """
        method_func = getattr(self, f"{method_name}_restoration")
        
        all_metrics = []
        
        for clean, degraded in zip(clean_images, degraded_images):
            start_time = time.time()
            restored = method_func(degraded, **method_params)
            processing_time = time.time() - start_time
            
            metrics = self.calculator.calculate_all_metrics(
                clean, restored, processing_time
            )
            all_metrics.append(metrics)
        
        return self.calculator.aggregate_metrics(all_metrics)
