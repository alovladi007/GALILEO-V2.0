"""
GeoSense Simulation Package

This package contains simulation modules for orbital dynamics and gravity field modeling.
"""

from .gravity import (
    GravityModel,
    SphericalHarmonics,
    load_egm2008_model,
    compute_geoid_height,
)

__all__ = [
    'GravityModel',
    'SphericalHarmonics',
    'load_egm2008_model',
    'compute_geoid_height',
]

__version__ = '0.1.0'
