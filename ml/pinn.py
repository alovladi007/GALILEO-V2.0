"""
Physics-Informed Neural Network (PINN) for Gravity Field Modeling
==================================================================

Enforces the Poisson equation constraint: ∇·g = -4πGρ
where g is gravity field, ρ is density, G is gravitational constant.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Optional
from torch.utils.data import Dataset, DataLoader


class GravityPINN(nn.Module):
    """
    Physics-Informed Neural Network for gravity field prediction.
    
    Network learns to predict gravity field from density distribution
    while enforcing physical constraints.
    """
    
    def __init__(self, hidden_layers: list = [64, 128, 128, 64],
                 activation: str = 'tanh'):
        """
        Parameters
        ----------
        hidden_layers : list
            Sizes of hidden layers
        activation : str
            Activation function ('tanh', 'relu', 'silu')
        """
        super().__init__()
        
        # Input: (x, y, z, ρ) -> Output: (gx, gy, gz)
        layers = []
        input_dim = 4  # coordinates + density
        
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'silu':
                layers.append(nn.SiLU())
            
            prev_dim = hidden_dim
        
        # Output layer: 3 components of gravity
        layers.append(nn.Linear(prev_dim, 3))
        
        self.network = nn.Sequential(*layers)
        
        # Gravitational constant (in SI units: m^3 kg^-1 s^-2)
        self.G = 6.67430e-11
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier initialization for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor, shape (batch, 4)
            Input [x, y, z, ρ]
        
        Returns
        -------
        g : torch.Tensor, shape (batch, 3)
            Predicted gravity field [gx, gy, gz]
        """
        return self.network(x)
    
    def compute_divergence(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute divergence of gravity field using automatic differentiation.
        
        ∇·g = ∂gx/∂x + ∂gy/∂y + ∂gz/∂z
        
        Parameters
        ----------
        x : torch.Tensor
            Input coordinates and density
        
        Returns
        -------
        div_g : torch.Tensor
            Divergence of gravity field
        """
        x.requires_grad_(True)
        
        g = self.forward(x)
        
        # Compute gradients
        gx, gy, gz = g[:, 0], g[:, 1], g[:, 2]
        
        dgx_dx = torch.autograd.grad(gx, x, 
                                     grad_outputs=torch.ones_like(gx),
                                     create_graph=True)[0][:, 0]
        
        dgy_dy = torch.autograd.grad(gy, x,
                                     grad_outputs=torch.ones_like(gy),
                                     create_graph=True)[0][:, 1]
        
        dgz_dz = torch.autograd.grad(gz, x,
                                     grad_outputs=torch.ones_like(gz),
                                     create_graph=True)[0][:, 2]
        
        div_g = dgx_dx + dgy_dy + dgz_dz
        
        return div_g
    
    def physics_loss(self, x: torch.Tensor, lambda_physics: float = 1.0) -> torch.Tensor:
        """
        Compute physics-informed loss enforcing ∇·g = -4πGρ.
        
        Parameters
        ----------
        x : torch.Tensor
            Input [x, y, z, ρ]
        lambda_physics : float
            Weight for physics constraint
        
        Returns
        -------
        loss : torch.Tensor
            Physics constraint violation
        """
        rho = x[:, 3]  # Density
        div_g = self.compute_divergence(x)
        
        # Physics constraint: ∇·g + 4πGρ = 0
        constraint = div_g + 4 * np.pi * self.G * rho
        
        loss = lambda_physics * torch.mean(constraint**2)
        
        return loss


class PINNTrainer:
    """
    Trainer for Physics-Informed Neural Networks.
    
    Combines data loss with physics constraint loss.
    """
    
    def __init__(self, model: GravityPINN, device: str = 'cuda'):
        """
        Parameters
        ----------
        model : GravityPINN
            PINN model to train
        device : str
            Device for training ('cuda' or 'cpu')
        """
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.history = {
            'train_loss': [],
            'data_loss': [],
            'physics_loss': [],
            'val_loss': []
        }
    
    def train(self, train_loader: DataLoader, 
             val_loader: Optional[DataLoader] = None,
             epochs: int = 100,
             lr: float = 1e-3,
             lambda_physics: float = 1.0,
             scheduler_patience: int = 10) -> Dict:
        """
        Train PINN model.
        
        Parameters
        ----------
        train_loader : DataLoader
            Training data loader
        val_loader : DataLoader, optional
            Validation data loader
        epochs : int
            Number of training epochs
        lr : float
            Learning rate
        lambda_physics : float
            Weight for physics loss
        scheduler_patience : int
            Patience for learning rate scheduler
        
        Returns
        -------
        history : dict
            Training history
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=scheduler_patience, factor=0.5
        )
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss_epoch = 0
            data_loss_epoch = 0
            physics_loss_epoch = 0
            
            for batch in train_loader:
                x_batch, g_true = batch
                x_batch = x_batch.to(self.device)
                g_true = g_true.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                g_pred = self.model(x_batch)
                
                # Data loss (MSE)
                data_loss = torch.mean((g_pred - g_true)**2)
                
                # Physics loss
                physics_loss = self.model.physics_loss(x_batch, lambda_physics)
                
                # Total loss
                loss = data_loss + physics_loss
                
                loss.backward()
                optimizer.step()
                
                train_loss_epoch += loss.item()
                data_loss_epoch += data_loss.item()
                physics_loss_epoch += physics_loss.item()
            
            # Average losses
            n_batches = len(train_loader)
            train_loss_epoch /= n_batches
            data_loss_epoch /= n_batches
            physics_loss_epoch /= n_batches
            
            self.history['train_loss'].append(train_loss_epoch)
            self.history['data_loss'].append(data_loss_epoch)
            self.history['physics_loss'].append(physics_loss_epoch)
            
            # Validation
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.history['val_loss'].append(val_loss)
                scheduler.step(val_loss)
            else:
                scheduler.step(train_loss_epoch)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Loss: {train_loss_epoch:.6f} "
                      f"(Data: {data_loss_epoch:.6f}, Physics: {physics_loss_epoch:.6f})")
                if val_loader is not None:
                    print(f"  Val Loss: {val_loss:.6f}")
        
        return self.history
    
    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate model on validation set.
        
        Parameters
        ----------
        val_loader : DataLoader
            Validation data loader
        
        Returns
        -------
        val_loss : float
            Average validation loss
        """
        self.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x_batch, g_true = batch
                x_batch = x_batch.to(self.device)
                g_true = g_true.to(self.device)
                
                g_pred = self.model(x_batch)
                loss = torch.mean((g_pred - g_true)**2)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        return val_loss
    
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


class GravityDataset(Dataset):
    """
    Dataset for gravity field training.
    
    Stores (coordinates, density) -> gravity field pairs.
    """
    
    def __init__(self, coordinates: np.ndarray, densities: np.ndarray,
                 gravity_fields: np.ndarray):
        """
        Parameters
        ----------
        coordinates : ndarray, shape (n_samples, 3)
            Spatial coordinates (x, y, z)
        densities : ndarray, shape (n_samples,)
            Density values
        gravity_fields : ndarray, shape (n_samples, 3)
            True gravity field components
        """
        self.x = torch.FloatTensor(np.column_stack([coordinates, densities]))
        self.g = torch.FloatTensor(gravity_fields)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.g[idx]


def generate_synthetic_gravity_data(n_samples: int = 1000,
                                    grid_size: Tuple[float, float, float] = (100, 100, 50),
                                    n_anomalies: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic gravity field data for training.
    
    Parameters
    ----------
    n_samples : int
        Number of sample points
    grid_size : tuple
        (Lx, Ly, Lz) domain size
    n_anomalies : int
        Number of density anomalies
    
    Returns
    -------
    coords : ndarray, shape (n_samples, 3)
        Sample coordinates
    densities : ndarray, shape (n_samples,)
        Density values
    gravity : ndarray, shape (n_samples, 3)
        Computed gravity field
    """
    Lx, Ly, Lz = grid_size
    
    # Random sample points
    coords = np.random.rand(n_samples, 3) * np.array([Lx, Ly, Lz])
    
    # Generate density field with random anomalies
    densities = np.zeros(n_samples)
    
    for _ in range(n_anomalies):
        # Random anomaly center and size
        center = np.random.rand(3) * np.array([Lx, Ly, Lz])
        size = np.random.rand() * 10 + 5
        amplitude = np.random.randn() * 500 + 2700  # kg/m^3
        
        # Gaussian anomaly
        dist = np.linalg.norm(coords - center, axis=1)
        densities += amplitude * np.exp(-(dist / size)**2)
    
    # Compute gravity field (simplified point source approximation)
    G = 6.67430e-11
    gravity = np.zeros((n_samples, 3))
    
    for i, (pos, rho) in enumerate(zip(coords, densities)):
        # Sum contributions from other points
        for j, (pos_source, rho_source) in enumerate(zip(coords, densities)):
            if i != j:
                r = pos - pos_source
                r_mag = np.linalg.norm(r)
                if r_mag > 1e-6:  # Avoid singularity
                    gravity[i] += -4 * np.pi * G * rho_source * r / r_mag**3
    
    return coords, densities, gravity
