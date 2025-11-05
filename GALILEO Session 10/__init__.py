"""
Geophysics Module - Earth Reference Models and Background Corrections

This module provides tools for loading and working with Earth reference models
including gravity fields, crustal density priors, hydrology templates, and
terrain corrections for geophysical data processing.
"""

from .gravity_fields import (
    GravityFieldModel,
    load_egm96,
    load_egm2008,
    compute_gravity_anomaly,
)

from .crustal_models import (
    CrustalDensityModel,
    load_crust1,
    terrain_correction,
    bouguer_correction,
    complete_bouguer_anomaly,
    isostatic_correction,
)

from .hydrology import (
    HydrologyModel,
    load_seasonal_water,
    load_groundwater_model,
    hydrological_correction,
)

from .masking import (
    OceanLandMask,
    load_ocean_mask,
    create_region_mask,
)

from .joint_inversion import (
    JointInversionModel,
    setup_joint_inversion,
    integrate_gravity_seismic,
    add_magnetic_data,
    perform_joint_inversion,
    export_for_session5,
    load_from_session5,
)

__version__ = "1.0.0"

__all__ = [
    # Gravity fields
    "GravityFieldModel",
    "load_egm96",
    "load_egm2008",
    "compute_gravity_anomaly",
    
    # Crustal models
    "CrustalDensityModel",
    "load_crust1",
    "terrain_correction",
    "bouguer_correction",
    "complete_bouguer_anomaly",
    "isostatic_correction",
    
    # Hydrology
    "HydrologyModel",
    "load_seasonal_water",
    "load_groundwater_model",
    "hydrological_correction",
    
    # Masking
    "OceanLandMask",
    "load_ocean_mask",
    "create_region_mask",
    
    # Joint inversion
    "JointInversionModel",
    "setup_joint_inversion",
    "integrate_gravity_seismic",
    "add_magnetic_data",
    "perform_joint_inversion",
    "export_for_session5",
    "load_from_session5",
]
