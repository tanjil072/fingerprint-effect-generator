"""
Utility functions for DiffUIR fingerprint restoration
"""

import yaml
import logging
import os
import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple
import cv2
from PIL import Image


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration: {e}")


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured logger
    """
    log_level = getattr(logging, config['output']['log_level'], logging.INFO)
    log_file = os.path.join(config['output']['results_dir'], config['output']['log_file'])
    
    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger('fingerprint_restoration')
    return logger


def setup_device(config: Dict[str, Any]) -> torch.device:
    """
    Setup computation device based on configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        PyTorch device
    """
    device_config = config['model']['device']
    
    if device_config == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_config)
    
    if device.type == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available, falling back to CPU")
        device = torch.device('cpu')
    
    return device


def set_random_seeds(seed: int):
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_directory(path: str):
    """
    Ensure directory exists, create if necessary
    
    Args:
        path: Directory path
    """
    os.makedirs(path, exist_ok=True)


def load_image(image_path: str, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Load and preprocess image
    
    Args:
        image_path: Path to image file
        target_size: Target size as (height, width), None to keep original
        
    Returns:
        Preprocessed image as numpy array [H, W, C] in range [0, 1]
    """
    # Load image
    if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = np.array(Image.open(image_path))
    
    # Resize if needed
    if target_size is not None:
        image = cv2.resize(image, (target_size[1], target_size[0]))
    
    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    return image


def save_image(image: np.ndarray, save_path: str, denormalize: bool = True):
    """
    Save image to file
    
    Args:
        image: Image array [H, W, C]
        save_path: Path to save image
        denormalize: Whether to denormalize from [0,1] to [0,255]
    """
    ensure_directory(os.path.dirname(save_path))
    
    if denormalize:
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    
    if image.shape[-1] == 3:  # RGB
        image_pil = Image.fromarray(image)
        image_pil.save(save_path)
    else:
        cv2.imwrite(save_path, image)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert PyTorch tensor to numpy array
    
    Args:
        tensor: PyTorch tensor [B, C, H, W] or [C, H, W]
        
    Returns:
        Numpy array [H, W, C] or [B, H, W, C]
    """
    if tensor.dim() == 4:  # [B, C, H, W]
        return tensor.permute(0, 2, 3, 1).detach().cpu().numpy()
    elif tensor.dim() == 3:  # [C, H, W]
        return tensor.permute(1, 2, 0).detach().cpu().numpy()
    else:
        return tensor.detach().cpu().numpy()


def numpy_to_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Convert numpy array to PyTorch tensor
    
    Args:
        array: Numpy array [H, W, C] or [B, H, W, C]
        device: Target device
        
    Returns:
        PyTorch tensor [C, H, W] or [B, C, H, W]
    """
    if array.ndim == 4:  # [B, H, W, C]
        tensor = torch.from_numpy(array).permute(0, 3, 1, 2)
    elif array.ndim == 3:  # [H, W, C]
        tensor = torch.from_numpy(array).permute(2, 0, 1)
    else:
        tensor = torch.from_numpy(array)
    
    return tensor.float().to(device)


def create_grid_comparison(images: list, titles: list, rows: int = 1) -> np.ndarray:
    """
    Create a grid comparison of images
    
    Args:
        images: List of images as numpy arrays
        titles: List of titles for each image
        rows: Number of rows in grid
        
    Returns:
        Combined grid image
    """
    import matplotlib.pyplot as plt
    
    cols = len(images) // rows
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    
    for i, (img, title) in enumerate(zip(images, titles)):
        if rows == 1:
            ax = axes[i]
        else:
            ax = axes[i // cols, i % cols]
            
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    
    # Convert to numpy array
    fig.canvas.draw()
    grid_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    grid_image = grid_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    return grid_image


def get_file_list(directory: str, extensions: list = ['.png', '.jpg', '.jpeg']) -> list:
    """
    Get list of image files in directory
    
    Args:
        directory: Directory path
        extensions: Allowed file extensions
        
    Returns:
        List of file paths
    """
    files = []
    for ext in extensions:
        files.extend([
            os.path.join(directory, f) 
            for f in os.listdir(directory) 
            if f.lower().endswith(ext.lower())
        ])
    return sorted(files)


class ProgressTracker:
    """Simple progress tracker for restoration tasks"""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        
    def update(self, increment: int = 1):
        self.current += increment
        progress = self.current / self.total * 100
        print(f"\r{self.description}: {progress:.1f}% ({self.current}/{self.total})", end='')
        
    def finish(self):
        print()  # New line
