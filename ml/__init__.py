"""
Physics-Informed Machine Learning Module
=========================================

Neural network models for accelerating geophysical inversions.
"""

from .pinn import (
    GravityPINN,
    PINNTrainer,
    GravityDataset,
    generate_synthetic_gravity_data
)

from .unet import (
    UNetGravity,
    UNetTrainer,
    MCDropoutUncertainty,
    EnsembleUncertainty
)

from .train import (
    PhaseGravityDataset,
    generate_synthetic_phase_gravity_pairs,
    train_pinn_model,
    train_unet_model,
    evaluate_uncertainty
)

__all__ = [
    'GravityPINN',
    'PINNTrainer',
    'GravityDataset',
    'generate_synthetic_gravity_data',
    'UNetGravity',
    'UNetTrainer',
    'MCDropoutUncertainty',
    'EnsembleUncertainty',
    'PhaseGravityDataset',
    'generate_synthetic_phase_gravity_pairs',
    'train_pinn_model',
    'train_unet_model',
    'evaluate_uncertainty'
]
