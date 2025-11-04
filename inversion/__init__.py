"""
Geophysical Inversion Package

Implements algorithms for recovering Earth's mass distribution from gravity measurements.
"""

from .algorithms import (
    InversionConfig,
    TikhonovInversion,
    BayesianInversion,
    resolution_matrix,
)

__all__ = [
    'InversionConfig',
    'TikhonovInversion',
    'BayesianInversion',
    'resolution_matrix',
]

__version__ = '0.1.0'
