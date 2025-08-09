"""
DiffUIR Model wrapper for fingerprint restoration
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple
import os
import time
from .utils import tensor_to_numpy, numpy_to_tensor


class SimpleDiffusionModel(nn.Module):
    """
    Simplified diffusion-based restoration model
    This is a placeholder implementation - replace with actual DiffUIR model
    """
    
    def __init__(self, 
                 in_channels: int = 3,
                 out_channels: int = 3, 
                 base_channels: int = 64):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Middle processing
        self.middle = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, out_channels, 7, padding=3),
            nn.Sigmoid()  # Output in [0, 1] range
        )
        
        # Residual connection
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Restored tensor [B, C, H, W]
        """
        residual = self.residual_conv(x)
        
        # Encode
        encoded = self.encoder(x)
        
        # Process
        processed = self.middle(encoded)
        
        # Decode
        decoded = self.decoder(processed)
        
        # Add residual connection
        output = decoded + residual
        
        return torch.clamp(output, 0, 1)


class DiffUIRModel:
    """
    DiffUIR model wrapper for fingerprint restoration
    """
    
    def __init__(self, config: Dict, device: Optional[torch.device] = None):
        """
        Initialize DiffUIR model
        
        Args:
            config: Configuration dictionary
            device: Computation device
        """
        self.config = config
        self.model_config = config['model']
        
        # Setup device
        if device is None:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() and self.model_config['device'] == 'cuda' 
                else 'cpu'
            )
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = self._build_model()
        self.model.to(self.device)
        
        # Load pretrained weights if available
        self._load_weights()
        
        # Set to evaluation mode
        self.model.eval()
        
        # Enable mixed precision if configured
        self.use_amp = self.model_config.get('mixed_precision', False)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
    
    def _build_model(self) -> nn.Module:
        """
        Build the model architecture
        
        Returns:
            PyTorch model
        """
        # For now, use the simplified model
        # TODO: Replace with actual DiffUIR implementation
        model = SimpleDiffusionModel(
            in_channels=self.model_config['in_channels'],
            out_channels=self.model_config['out_channels'],
            base_channels=self.model_config['base_channels']
        )
        
        return model
    
    def _load_weights(self):
        """
        Load pretrained model weights
        """
        checkpoint_path = self.model_config.get('checkpoint_path')
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading model weights from {checkpoint_path}")
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                
                # Handle different checkpoint formats
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                self.model.load_state_dict(state_dict, strict=False)
                print("✅ Model weights loaded successfully")
                
            except Exception as e:
                print(f"⚠️  Warning: Could not load weights: {e}")
                print("Using randomly initialized weights")
        else:
            print("⚠️  No checkpoint found, using randomly initialized weights")
            if checkpoint_path:
                print(f"   Checked path: {checkpoint_path}")
    
    def restore_image(self, degraded_image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Restore a single image
        
        Args:
            degraded_image: Degraded image [H, W, C] in range [0, 1]
            
        Returns:
            Tuple of (restored_image, processing_time)
        """
        start_time = time.time()
        
        with torch.no_grad():
            # Convert to tensor and add batch dimension
            input_tensor = numpy_to_tensor(degraded_image, self.device).unsqueeze(0)
            
            # Forward pass
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    output_tensor = self.model(input_tensor)
            else:
                output_tensor = self.model(input_tensor)
            
            # Convert back to numpy
            restored_image = tensor_to_numpy(output_tensor.squeeze(0))
        
        processing_time = time.time() - start_time
        
        return restored_image, processing_time
    
    def restore_batch(self, degraded_batch: torch.Tensor) -> torch.Tensor:
        """
        Restore a batch of images
        
        Args:
            degraded_batch: Batch tensor [B, C, H, W]
            
        Returns:
            Restored batch tensor [B, C, H, W]
        """
        with torch.no_grad():
            degraded_batch = degraded_batch.to(self.device)
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    restored_batch = self.model(degraded_batch)
            else:
                restored_batch = self.model(degraded_batch)
        
        return restored_batch
    
    def save_model(self, save_path: str, epoch: Optional[int] = None, metrics: Optional[Dict] = None):
        """
        Save model checkpoint
        
        Args:
            save_path: Path to save checkpoint
            epoch: Training epoch (if applicable)
            metrics: Training metrics (if applicable)
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'device': str(self.device),
        }
        
        if epoch is not None:
            checkpoint['epoch'] = epoch
        
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        torch.save(checkpoint, save_path)
        print(f"Model saved to {save_path}")
    
    def get_model_info(self) -> Dict:
        """
        Get model information
        
        Returns:
            Model information dictionary
        """
        def count_parameters():
            return sum(p.numel() for p in self.model.parameters())
        
        def count_trainable_parameters():
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_name': 'DiffUIR (Simplified)',
            'device': str(self.device),
            'total_parameters': count_parameters(),
            'trainable_parameters': count_trainable_parameters(),
            'input_channels': self.model_config['in_channels'],
            'output_channels': self.model_config['out_channels'],
            'mixed_precision': self.use_amp,
        }


class DiffUIRTrainer:
    """
    Trainer for DiffUIR model (if fine-tuning is needed)
    """
    
    def __init__(self, model: DiffUIRModel, config: Dict):
        """
        Initialize trainer
        
        Args:
            model: DiffUIR model instance
            config: Configuration dictionary
        """
        self.model = model
        self.config = config
        self.training_config = config['training']
        
        if not self.training_config.get('enabled', False):
            print("Training is disabled in configuration")
            return
        
        # Setup optimizer
        self.optimizer = torch.optim.Adam(
            self.model.model.parameters(),
            lr=self.training_config['learning_rate'],
            weight_decay=self.training_config.get('weight_decay', 1e-5)
        )
        
        # Setup loss function
        self._setup_loss()
        
        # Setup scheduler
        self._setup_scheduler()
        
        print("✅ Trainer initialized")
    
    def _setup_loss(self):
        """Setup loss function"""
        loss_type = self.training_config.get('loss_type', 'l1')
        
        if loss_type == 'l1':
            self.criterion = nn.L1Loss()
        elif loss_type == 'l2':
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.L1Loss()  # Default fallback
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        scheduler_type = self.training_config.get('scheduler', 'cosine')
        
        if scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.training_config['epochs']
            )
        else:
            self.scheduler = None
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Training metrics
        """
        if not self.training_config.get('enabled', False):
            raise ValueError("Training is not enabled")
        
        self.model.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            clean = batch['clean'].to(self.model.device)
            degraded = batch['fingerprinted'].to(self.model.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            if self.model.use_amp:
                with torch.cuda.amp.autocast():
                    restored = self.model.model(degraded)
                    loss = self.criterion(restored, clean)
                
                # Backward pass
                self.model.scaler.scale(loss).backward()
                self.model.scaler.step(self.optimizer)
                self.model.scaler.update()
            else:
                restored = self.model.model(degraded)
                loss = self.criterion(restored, clean)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # Update scheduler
        if self.scheduler:
            self.scheduler.step()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {'train_loss': avg_loss}
    
    def validate(self, val_loader) -> Dict[str, float]:
        """
        Validate model
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Validation metrics
        """
        self.model.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                clean = batch['clean'].to(self.model.device)
                degraded = batch['fingerprinted'].to(self.model.device)
                
                # Forward pass
                if self.model.use_amp:
                    with torch.cuda.amp.autocast():
                        restored = self.model.model(degraded)
                        loss = self.criterion(restored, clean)
                else:
                    restored = self.model.model(degraded)
                    loss = self.criterion(restored, clean)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {'val_loss': avg_loss}
