"""
Training Scripts for ML Models
===============================

Scripts for training PINN and U-Net models with synthetic data.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from pathlib import Path

from .pinn import GravityPINN, PINNTrainer, GravityDataset, generate_synthetic_gravity_data
from .unet import UNetGravity, UNetTrainer, MCDropoutUncertainty


class PhaseGravityDataset(Dataset):
    """Dataset for phase -> gravity field mapping."""
    
    def __init__(self, phase_data: np.ndarray, gravity_data: np.ndarray):
        """
        Parameters
        ----------
        phase_data : ndarray, shape (n_samples, H, W)
            Phase measurements
        gravity_data : ndarray, shape (n_samples, H, W)
            Gravity field ground truth
        """
        self.phase = torch.FloatTensor(phase_data).unsqueeze(1)  # Add channel dim
        self.gravity = torch.FloatTensor(gravity_data).unsqueeze(1)
    
    def __len__(self):
        return len(self.phase)
    
    def __getitem__(self, idx):
        return self.phase[idx], self.gravity[idx]


def generate_synthetic_phase_gravity_pairs(n_samples: int = 500,
                                           image_size: int = 128,
                                           noise_level: float = 0.1) -> tuple:
    """
    Generate synthetic phase -> gravity training pairs.
    
    Parameters
    ----------
    n_samples : int
        Number of training samples
    image_size : int
        Size of image (image_size x image_size)
    noise_level : float
        Noise level for phase data
    
    Returns
    -------
    phase_data : ndarray
        Noisy phase measurements
    gravity_data : ndarray
        Clean gravity fields
    """
    phase_data = []
    gravity_data = []
    
    for _ in range(n_samples):
        # Generate random gravity field (sum of Gaussians)
        gravity = np.zeros((image_size, image_size))
        
        n_anomalies = np.random.randint(3, 8)
        for _ in range(n_anomalies):
            # Random Gaussian anomaly
            cx = np.random.randint(0, image_size)
            cy = np.random.randint(0, image_size)
            sigma = np.random.uniform(5, 20)
            amplitude = np.random.uniform(-1, 1)
            
            x, y = np.meshgrid(np.arange(image_size), np.arange(image_size))
            gravity += amplitude * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
        
        # Simulate phase from gravity (simplified)
        phase = np.cumsum(gravity, axis=0)  # Simplified relationship
        
        # Add noise to phase
        phase += noise_level * np.random.randn(*phase.shape) * np.std(phase)
        
        # Normalize
        phase = (phase - np.mean(phase)) / (np.std(phase) + 1e-8)
        gravity = (gravity - np.mean(gravity)) / (np.std(gravity) + 1e-8)
        
        phase_data.append(phase)
        gravity_data.append(gravity)
    
    return np.array(phase_data), np.array(gravity_data)


def train_pinn_model(save_dir: str = './checkpoints', 
                     n_samples: int = 5000,
                     epochs: int = 200,
                     batch_size: int = 64,
                     lr: float = 1e-3,
                     lambda_physics: float = 1.0):
    """
    Train PINN model for gravity field prediction.
    
    Parameters
    ----------
    save_dir : str
        Directory to save checkpoints
    n_samples : int
        Number of training samples
    epochs : int
        Training epochs
    batch_size : int
        Batch size
    lr : float
        Learning rate
    lambda_physics : float
        Weight for physics loss
    """
    print("Generating synthetic gravity data...")
    coords, densities, gravity = generate_synthetic_gravity_data(n_samples=n_samples)
    
    # Create dataset
    dataset = GravityDataset(coords, densities, gravity)
    
    # Train/val split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Create model
    print("Creating PINN model...")
    model = GravityPINN(hidden_layers=[64, 128, 128, 64], activation='tanh')
    
    # Train
    print("Training PINN...")
    trainer = PINNTrainer(model, device='cuda')
    history = trainer.train(
        train_loader, val_loader,
        epochs=epochs, lr=lr,
        lambda_physics=lambda_physics
    )
    
    # Save checkpoint
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_path = f"{save_dir}/pinn_model.pth"
    trainer.save_checkpoint(checkpoint_path)
    print(f"Model saved to {checkpoint_path}")
    
    # Plot training curves
    plot_training_curves(history, save_path=f"{save_dir}/pinn_training_curves.png")
    
    return model, history


def train_unet_model(save_dir: str = './checkpoints',
                     n_samples: int = 500,
                     image_size: int = 128,
                     epochs: int = 150,
                     batch_size: int = 8,
                     lr: float = 1e-3):
    """
    Train U-Net model for phase -> gravity mapping.
    
    Parameters
    ----------
    save_dir : str
        Directory to save checkpoints
    n_samples : int
        Number of training samples
    image_size : int
        Image size
    epochs : int
        Training epochs
    batch_size : int
        Batch size
    lr : float
        Learning rate
    """
    print("Generating synthetic phase-gravity pairs...")
    phase_data, gravity_data = generate_synthetic_phase_gravity_pairs(
        n_samples=n_samples, image_size=image_size
    )
    
    # Create dataset
    dataset = PhaseGravityDataset(phase_data, gravity_data)
    
    # Train/val split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Create model
    print("Creating U-Net model...")
    model = UNetGravity(in_channels=1, out_channels=1, base_channels=64, 
                       depth=4, dropout=0.1)
    
    # Train
    print("Training U-Net...")
    trainer = UNetTrainer(model, device='cuda')
    history = trainer.train(
        train_loader, val_loader,
        epochs=epochs, lr=lr,
        loss_fn='mse',
        save_best=True,
        checkpoint_path=f"{save_dir}/unet_best.pth"
    )
    
    # Save final checkpoint
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_path = f"{save_dir}/unet_final.pth"
    trainer.save_checkpoint(checkpoint_path)
    print(f"Model saved to {checkpoint_path}")
    
    # Plot training curves
    plot_unet_training_curves(history, save_path=f"{save_dir}/unet_training_curves.png")
    
    return model, history


def plot_training_curves(history: dict, save_path: str = None):
    """Plot PINN training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curves
    axes[0].plot(history['train_loss'], label='Total Loss')
    axes[0].plot(history['data_loss'], label='Data Loss')
    axes[0].plot(history['physics_loss'], label='Physics Loss')
    if len(history['val_loss']) > 0:
        axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Curves')
    axes[0].legend()
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)
    
    # Physics loss detail
    axes[1].plot(history['physics_loss'])
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Physics Loss')
    axes[1].set_title('Physics Constraint Violation')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    plt.close()


def plot_unet_training_curves(history: dict, save_path: str = None):
    """Plot U-Net training curves with metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    if len(history['val_loss']) > 0:
        axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    
    # PSNR
    if len(history['psnr']) > 0:
        axes[0, 1].plot(history['psnr'])
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('PSNR (dB)')
        axes[0, 1].set_title('Peak Signal-to-Noise Ratio')
        axes[0, 1].grid(True, alpha=0.3)
    
    # SSIM
    if len(history['ssim']) > 0:
        axes[1, 0].plot(history['ssim'])
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('SSIM')
        axes[1, 0].set_title('Structural Similarity Index')
        axes[1, 0].grid(True, alpha=0.3)
    
    # MAE
    if len(history['mae']) > 0:
        axes[1, 1].plot(history['mae'])
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('MAE')
        axes[1, 1].set_title('Mean Absolute Error')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    plt.close()


def evaluate_uncertainty(model: UNetGravity, test_data: torch.Tensor,
                        method: str = 'mc_dropout', n_samples: int = 20,
                        save_dir: str = './results'):
    """
    Evaluate uncertainty estimation.
    
    Parameters
    ----------
    model : UNetGravity
        Trained model
    test_data : torch.Tensor
        Test input data
    method : str
        'mc_dropout' or 'ensemble'
    n_samples : int
        Number of MC samples
    save_dir : str
        Directory to save results
    """
    if method == 'mc_dropout':
        uncertainty_estimator = MCDropoutUncertainty(model, n_samples=n_samples)
    else:
        raise NotImplementedError("Ensemble method requires multiple models")
    
    mean_pred, std_pred = uncertainty_estimator.predict_with_uncertainty(test_data)
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].imshow(test_data[0, 0].cpu().numpy(), cmap='RdBu_r')
    axes[0].set_title('Input Phase')
    axes[0].axis('off')
    
    axes[1].imshow(mean_pred[0, 0].cpu().numpy(), cmap='RdBu_r')
    axes[1].set_title('Predicted Gravity (Mean)')
    axes[1].axis('off')
    
    axes[2].imshow(std_pred[0, 0].cpu().numpy(), cmap='viridis')
    axes[2].set_title('Uncertainty (Std Dev)')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{save_dir}/uncertainty_estimation.png", dpi=150, bbox_inches='tight')
    print(f"Uncertainty visualization saved to {save_dir}/uncertainty_estimation.png")
    
    plt.close()
    
    return mean_pred, std_pred


if __name__ == "__main__":
    print("=" * 60)
    print("Training Physics-Informed ML Models for Geophysics")
    print("=" * 60)
    
    # Train PINN
    print("\n[1/2] Training PINN...")
    pinn_model, pinn_history = train_pinn_model(
        save_dir='./checkpoints/pinn',
        n_samples=3000,
        epochs=100,
        batch_size=64,
        lambda_physics=1.0
    )
    
    # Train U-Net
    print("\n[2/2] Training U-Net...")
    unet_model, unet_history = train_unet_model(
        save_dir='./checkpoints/unet',
        n_samples=400,
        image_size=128,
        epochs=100,
        batch_size=8
    )
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
