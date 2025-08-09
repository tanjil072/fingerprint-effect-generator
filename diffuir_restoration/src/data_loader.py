"""
Data loader for fingerprint restoration experiment
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import shutil
from pathlib import Path

# Add parent directory to path for importing fingerprint generator
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.fingerprintgenerator import FingerprintEffectGenerator
from src.utils import load_image, save_image, get_file_list, ensure_directory


class FingerprintDataset(Dataset):
    """
    PyTorch Dataset for fingerprint restoration
    """
    
    def __init__(self, 
                 clean_images: List[str],
                 fingerprinted_images: List[str],
                 transform=None):
        """
        Initialize dataset
        
        Args:
            clean_images: List of paths to clean images
            fingerprinted_images: List of paths to fingerprinted images
            transform: Optional transforms to apply
        """
        assert len(clean_images) == len(fingerprinted_images), \
            "Number of clean and fingerprinted images must match"
        
        self.clean_images = clean_images
        self.fingerprinted_images = fingerprinted_images
        self.transform = transform
    
    def __len__(self):
        return len(self.clean_images)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
        Get dataset item
        
        Args:
            idx: Index
            
        Returns:
            Dictionary with 'clean', 'fingerprinted', and 'filename'
        """
        clean_path = self.clean_images[idx]
        fingerprinted_path = self.fingerprinted_images[idx]
        
        # Load images
        clean = load_image(clean_path)
        fingerprinted = load_image(fingerprinted_path)
        
        # Apply transforms if provided
        if self.transform:
            clean = self.transform(clean)
            fingerprinted = self.transform(fingerprinted)
        
        # Convert to tensors [C, H, W]
        clean_tensor = torch.from_numpy(clean).permute(2, 0, 1).float()
        fingerprinted_tensor = torch.from_numpy(fingerprinted).permute(2, 0, 1).float()
        
        return {
            'clean': clean_tensor,
            'fingerprinted': fingerprinted_tensor,
            'filename': os.path.basename(clean_path)
        }


class DatasetPreparator:
    """
    Prepare datasets for fingerprint restoration experiment
    """
    
    def __init__(self, config: Dict):
        """
        Initialize dataset preparator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_config = config['data']
        
        # Initialize fingerprint generator
        self.fingerprint_generator = FingerprintEffectGenerator()
        
        # Dataset paths
        self.clean_dir = self.data_config['clean_images_path']
        self.fingerprinted_dir = self.data_config['fingerprinted_images_path']
        self.restored_dir = self.data_config['restored_images_path']
        
        # Create directories
        ensure_directory(self.clean_dir)
        ensure_directory(self.fingerprinted_dir)
        ensure_directory(self.restored_dir)
    
    def copy_source_images(self, source_dir: str):
        """
        Copy images from source directory to clean images directory
        
        Args:
            source_dir: Source directory containing original images
        """
        print(f"Copying images from {source_dir} to {self.clean_dir}")
        
        # Get list of images from source
        source_images = get_file_list(source_dir)
        
        if not source_images:
            print(f"Warning: No images found in {source_dir}")
            return
        
        # Copy images
        for src_path in source_images:
            filename = os.path.basename(src_path)
            dst_path = os.path.join(self.clean_dir, filename)
            shutil.copy2(src_path, dst_path)
        
        print(f"Copied {len(source_images)} images")
    
    def generate_fingerprinted_images(self):
        """
        Generate fingerprinted versions of clean images
        """
        print("Generating fingerprinted images...")
        
        # Get list of clean images
        clean_images = get_file_list(self.clean_dir)
        
        if not clean_images:
            print(f"Warning: No clean images found in {self.clean_dir}")
            return
        
        # Generate fingerprinted versions
        for i, clean_path in enumerate(clean_images):
            filename = os.path.basename(clean_path)
            fingerprinted_path = os.path.join(self.fingerprinted_dir, filename)
            
            # Load clean image
            clean_image = load_image(clean_path)
            
            # Convert to proper format for fingerprint generator (0-255, uint8)
            clean_uint8 = (clean_image * 255).astype(np.uint8)
            
            # Generate fingerprinted image
            fingerprinted_uint8, _ = self.fingerprint_generator.process_image(clean_uint8)
            
            # Convert back to [0, 1] float and save
            fingerprinted_float = fingerprinted_uint8.astype(np.float32) / 255.0
            save_image(fingerprinted_float, fingerprinted_path, denormalize=True)
            
            print(f"\rProcessed {i+1}/{len(clean_images)} images", end='')
        
        print(f"\nGenerated {len(clean_images)} fingerprinted images")
    
    def create_data_splits(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Create train/validation/test splits
        
        Returns:
            Tuple of (train_files, val_files, test_files)
        """
        # Get list of clean images
        clean_images = get_file_list(self.clean_dir)
        
        if not clean_images:
            raise ValueError(f"No images found in {self.clean_dir}")
        
        # Shuffle for random splits
        np.random.seed(self.config['experiment']['random_seed'])
        indices = np.random.permutation(len(clean_images))
        
        # Calculate split sizes
        train_size = int(len(clean_images) * self.data_config['train_ratio'])
        val_size = int(len(clean_images) * self.data_config['val_ratio'])
        
        # Create splits
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        train_files = [clean_images[i] for i in train_indices]
        val_files = [clean_images[i] for i in val_indices]
        test_files = [clean_images[i] for i in test_indices]
        
        print(f"Data splits - Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
        
        return train_files, val_files, test_files
    
    def get_paired_datasets(self) -> Tuple[FingerprintDataset, FingerprintDataset, FingerprintDataset]:
        """
        Get train/val/test datasets with paired clean and fingerprinted images
        
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        # Get data splits
        train_clean, val_clean, test_clean = self.create_data_splits()
        
        # Create corresponding fingerprinted image paths
        def get_fingerprinted_paths(clean_paths):
            return [
                os.path.join(self.fingerprinted_dir, os.path.basename(path))
                for path in clean_paths
            ]
        
        train_fingerprinted = get_fingerprinted_paths(train_clean)
        val_fingerprinted = get_fingerprinted_paths(val_clean)
        test_fingerprinted = get_fingerprinted_paths(test_clean)
        
        # Create datasets
        train_dataset = FingerprintDataset(train_clean, train_fingerprinted)
        val_dataset = FingerprintDataset(val_clean, val_fingerprinted)
        test_dataset = FingerprintDataset(test_clean, test_fingerprinted)
        
        return train_dataset, val_dataset, test_dataset
    
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Get PyTorch data loaders for train/val/test
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        train_dataset, val_dataset, test_dataset = self.get_paired_datasets()
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.data_config['batch_size'],
            shuffle=True,
            num_workers=self.data_config['num_workers']
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.data_config['batch_size'],
            shuffle=False,
            num_workers=self.data_config['num_workers']
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.data_config['batch_size'],
            shuffle=False,
            num_workers=self.data_config['num_workers']
        )
        
        return train_loader, val_loader, test_loader
    
    def prepare_dataset(self, source_images_dir: Optional[str] = None):
        """
        Complete dataset preparation pipeline
        
        Args:
            source_images_dir: Source directory with original images
                              If None, uses existing input_images from parent project
        """
        print("🎯 Preparing fingerprint restoration dataset...")
        
        # Use parent project's input images if no source specified
        if source_images_dir is None:
            parent_input_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 
                'input_images'
            )
            if os.path.exists(parent_input_dir):
                source_images_dir = parent_input_dir
            else:
                raise ValueError("No source images directory specified and parent input_images not found")
        
        # Step 1: Copy source images to clean directory
        self.copy_source_images(source_images_dir)
        
        # Step 2: Generate fingerprinted versions
        self.generate_fingerprinted_images()
        
        # Step 3: Verify dataset
        clean_count = len(get_file_list(self.clean_dir))
        fingerprinted_count = len(get_file_list(self.fingerprinted_dir))
        
        print(f"\n✅ Dataset preparation complete!")
        print(f"   Clean images: {clean_count}")
        print(f"   Fingerprinted images: {fingerprinted_count}")
        
        if clean_count != fingerprinted_count:
            print("⚠️  Warning: Mismatch in clean and fingerprinted image counts")
        
        return clean_count, fingerprinted_count


def prepare_dataset_cli():
    """
    CLI function for dataset preparation
    """
    import argparse
    from .utils import load_config
    
    parser = argparse.ArgumentParser(description='Prepare fingerprint restoration dataset')
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    parser.add_argument('--source-dir', help='Source directory with original images')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create preparator and run
    preparator = DatasetPreparator(config)
    preparator.prepare_dataset(args.source_dir)


if __name__ == "__main__":
    prepare_dataset_cli()
