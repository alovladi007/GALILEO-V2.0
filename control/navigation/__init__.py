"""
Navigation and state estimation algorithms.

This module provides filtering and estimation techniques:
- EKF: Extended Kalman Filter for nonlinear systems
- UKF: Unscented Kalman Filter (coming soon)
- Sensor fusion: Multi-sensor integration (coming soon)
"""

from .ekf import (
    compute_jacobian_autodiff,
    ExtendedKalmanFilter,
    OrbitalEKF,
    RelativeNavigationEKF,
    compute_consistency_nis,
    observability_gramian,
)

__all__ = [
    'compute_jacobian_autodiff',
    'ExtendedKalmanFilter',
    'OrbitalEKF',
    'RelativeNavigationEKF',
    'compute_consistency_nis',
    'observability_gramian',
]
