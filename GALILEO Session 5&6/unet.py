"""
U-Net Denoiser for Gravity Field Prediction
============================================

Transforms noisy phase measurements to clean gravity field maps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict


class DoubleConv(nn.Module):
    """Double convolution block used in U-Net."""
    
    def __init__(self, in_channels: int, out_channels: int,
                 dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class UNetGravity(nn.Module):
    """
    U-Net architecture for gravity field denoising and prediction.
    
    Architecture:
    - Encoder: Downsampling with skip connections
    - Bottleneck: Deepest representation
    - Decoder: Upsampling with skip connections
    """
    
    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 base_channels: int = 64, depth: int = 4,
                 dropout: float = 0.1):
        """
        Parameters
        ----------
        in_channels : int
            Number of input channels (e.g., 1 for phase)
        out_channels : int
            Number of output channels (e.g., 1 or 3 for gravity components)
        base_channels : int
            Number of channels in first layer
        depth : int
            Depth of U-Net (number of downsampling levels)
        dropout : float
            Dropout probability for uncertainty estimation
        """
        super().__init__()
        
        self.depth = depth
        self.dropout = dropout
        
        # Encoder (downsampling path)
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        channels = base_channels
        in_ch = in_channels
        
        for i in range(depth):
            self.encoders.append(DoubleConv(in_ch, channels, dropout))
            self.pools.append(nn.MaxPool2d(2))
            in_ch = channels
            channels *= 2
        
        # Bottleneck
        self.bottleneck = DoubleConv(in_ch, channels, dropout)
        
        # Decoder (upsampling path)
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        for i in range(depth):
            self.upconvs.append(
                nn.ConvTranspose2d(channels, channels // 2, 2, stride=2)
            )
            self.decoders.append(
                DoubleConv(channels, channels // 2, dropout)
            )
            channels //= 2
        
        # Final output layer
        self.out_conv = nn.Conv2d(channels, out_channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through U-Net.
        
        Parameters
        ----------
        x : torch.Tensor, shape (batch, in_channels, H, W)
            Input phase/noisy field
        
        Returns
        -------
        out : torch.Tensor, shape (batch, out_channels, H, W)
            Predicted clean gravity field
        """
        # Encoder
        skip_connections = []
        
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skip_connections.append(x)
            x = pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        skip_connections = skip_connections[::-1]
        
        for i, (upconv, decoder) in enumerate(zip(self.upconvs, self.decoders)):
            x = upconv(x)
            
            skip = skip_connections[i]
            
            # Handle size mismatch due to padding
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', 
                                 align_corners=True)
            
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)
        
        # Output
        out = self.out_conv(x)
        
        return out
    
    def enable_dropout(self):
        """Enable dropout for uncertainty estimation (MC Dropout)."""
        for m in self.modules():
            if isinstance(m, nn.Dropout2d):
                m.train()


class UNetTrainer:
    """
    Trainer for U-Net gravity field denoiser.
    """
    
    def __init__(self, model: UNetGravity, device: str = 'cuda'):
        """
        Parameters
        ----------
        model : UNetGravity
            U-Net model
        device : str
            Training device
        """
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'psnr': [],
            'ssim': [],
            'mae': []
        }
    
    def train(self, train_loader, val_loader=None,
             epochs: int = 100, lr: float = 1e-3,
             loss_fn: str = 'mse',
             save_best: bool = True,
             checkpoint_path: str = 'best_unet.pth'):
        """
        Train U-Net model.
        
        Parameters
        ----------
        train_loader : DataLoader
            Training data
        val_loader : DataLoader, optional
            Validation data
        epochs : int
            Number of epochs
        lr : float
            Learning rate
        loss_fn : str
            Loss function ('mse', 'mae', 'huber')
        save_best : bool
            Save best model
        checkpoint_path : str
            Path to save checkpoint
        """
        # Loss function
        if loss_fn == 'mse':
            criterion = nn.MSELoss()
        elif loss_fn == 'mae':
            criterion = nn.L1Loss()
        elif loss_fn == 'huber':
            criterion = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss: {loss_fn}")
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=10, factor=0.5
        )
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            self.history['train_loss'].append(train_loss)
            
            # Validation
            if val_loader is not None:
                val_metrics = self.validate(val_loader, criterion)
                val_loss = val_metrics['loss']
                
                self.history['val_loss'].append(val_loss)
                self.history['psnr'].append(val_metrics['psnr'])
                self.history['ssim'].append(val_metrics['ssim'])
                self.history['mae'].append(val_metrics['mae'])
                
                scheduler.step(val_loss)
                
                # Save best model
                if save_best and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(checkpoint_path)
            else:
                scheduler.step(train_loss)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}")
                if val_loader is not None:
                    print(f"  Val Loss: {val_loss:.6f}, "
                          f"PSNR: {val_metrics['psnr']:.2f} dB, "
                          f"SSIM: {val_metrics['ssim']:.4f}")
        
        return self.history
    
    def validate(self, val_loader, criterion) -> Dict[str, float]:
        """
        Validate model and compute metrics.
        
        Returns
        -------
        metrics : dict
            Validation metrics (loss, PSNR, SSIM, MAE)
        """
        self.model.eval()
        
        val_loss = 0
        psnr_total = 0
        ssim_total = 0
        mae_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                
                # Compute metrics
                psnr_total += self.compute_psnr(outputs, targets)
                ssim_total += self.compute_ssim(outputs, targets)
                mae_total += torch.mean(torch.abs(outputs - targets)).item()
        
        n = len(val_loader)
        
        return {
            'loss': val_loss / n,
            'psnr': psnr_total / n,
            'ssim': ssim_total / n,
            'mae': mae_total / n
        }
    
    @staticmethod
    def compute_psnr(pred: torch.Tensor, target: torch.Tensor,
                     max_val: float = 1.0) -> float:
        """
        Compute Peak Signal-to-Noise Ratio (PSNR).
        
        PSNR = 10 * log10(MAX^2 / MSE)
        """
        mse = torch.mean((pred - target)**2)
        if mse == 0:
            return float('inf')
        psnr = 10 * torch.log10(max_val**2 / mse)
        return psnr.item()
    
    @staticmethod
    def compute_ssim(pred: torch.Tensor, target: torch.Tensor,
                     window_size: int = 11) -> float:
        """
        Compute Structural Similarity Index (SSIM).
        
        Simplified version for batch processing.
        """
        C1 = 0.01**2
        C2 = 0.03**2
        
        mu_x = F.avg_pool2d(pred, window_size, stride=1, padding=window_size//2)
        mu_y = F.avg_pool2d(target, window_size, stride=1, padding=window_size//2)
        
        mu_x_sq = mu_x**2
        mu_y_sq = mu_y**2
        mu_xy = mu_x * mu_y
        
        sigma_x_sq = F.avg_pool2d(pred**2, window_size, stride=1, 
                                   padding=window_size//2) - mu_x_sq
        sigma_y_sq = F.avg_pool2d(target**2, window_size, stride=1,
                                   padding=window_size//2) - mu_y_sq
        sigma_xy = F.avg_pool2d(pred * target, window_size, stride=1,
                               padding=window_size//2) - mu_xy
        
        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
                   ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
        
        return torch.mean(ssim_map).item()
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint['history']


class MCDropoutUncertainty:
    """
    Monte Carlo Dropout for uncertainty estimation.
    
    Runs multiple forward passes with dropout enabled to estimate
    prediction uncertainty.
    """
    
    def __init__(self, model: UNetGravity, n_samples: int = 20):
        """
        Parameters
        ----------
        model : UNetGravity
            Trained U-Net model
        n_samples : int
            Number of MC samples
        """
        self.model = model
        self.n_samples = n_samples
    
    def predict_with_uncertainty(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with uncertainty estimation.
        
        Parameters
        ----------
        x : torch.Tensor
            Input data
        
        Returns
        -------
        mean : torch.Tensor
            Mean prediction
        std : torch.Tensor
            Standard deviation (epistemic uncertainty)
        """
        self.model.eval()
        self.model.enable_dropout()  # Enable dropout during inference
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(self.n_samples):
                pred = self.model(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        
        mean = torch.mean(predictions, dim=0)
        std = torch.std(predictions, dim=0)
        
        return mean, std


class EnsembleUncertainty:
    """
    Deep ensemble for uncertainty estimation.
    
    Trains multiple models with different initializations and
    combines predictions.
    """
    
    def __init__(self, n_models: int = 5):
        """
        Parameters
        ----------
        n_models : int
            Number of ensemble members
        """
        self.n_models = n_models
        self.models = []
    
    def train_ensemble(self, model_class, train_loader, val_loader, **train_kwargs):
        """
        Train ensemble of models.
        
        Parameters
        ----------
        model_class : class
            Model class to instantiate
        train_loader : DataLoader
            Training data
        val_loader : DataLoader
            Validation data
        **train_kwargs : dict
            Training arguments
        """
        for i in range(self.n_models):
            print(f"\nTraining ensemble member {i+1}/{self.n_models}")
            
            model = model_class()
            trainer = UNetTrainer(model)
            trainer.train(train_loader, val_loader, **train_kwargs)
            
            self.models.append(model)
    
    def predict_with_uncertainty(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Ensemble prediction with uncertainty.
        
        Parameters
        ----------
        x : torch.Tensor
            Input data
        
        Returns
        -------
        mean : torch.Tensor
            Ensemble mean
        std : torch.Tensor
            Ensemble standard deviation
        """
        predictions = []
        
        with torch.no_grad():
            for model in self.models:
                model.eval()
                pred = model(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        
        mean = torch.mean(predictions, dim=0)
        std = torch.std(predictions, dim=0)
        
        return mean, std
