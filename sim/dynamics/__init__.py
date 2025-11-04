"""
Orbital Dynamics Module

Contains implementations of:
- Keplerian (two-body) dynamics
- Orbital perturbations (J2, drag, SRP)
- Relative orbital motion (formation flying)
- Numerical propagators (RK4, adaptive methods)
"""

from .keplerian import (
    GM_EARTH,
    R_EARTH,
    two_body_dynamics,
    mean_motion,
    orbital_period,
    orbital_elements_to_cartesian,
    cartesian_to_orbital_elements,
)

from .perturbations import (
    J2_EARTH,
    OMEGA_EARTH,
    j2_acceleration,
    atmospheric_density,
    atmospheric_drag_acceleration,
    solar_radiation_pressure_acceleration,
    perturbed_dynamics,
)

from .relative import (
    hill_clohessy_wiltshire_dynamics,
    relative_dynamics_nonlinear,
    hill_frame_to_inertial,
    inertial_to_hill_frame,
)

from .propagators import (
    rk4_step,
    propagate_orbit,
    propagate_orbit_jax,
    propagate_relative_orbit,
)

__all__ = [
    # Constants
    'GM_EARTH',
    'R_EARTH',
    'J2_EARTH',
    'OMEGA_EARTH',

    # Keplerian dynamics
    'two_body_dynamics',
    'mean_motion',
    'orbital_period',
    'orbital_elements_to_cartesian',
    'cartesian_to_orbital_elements',

    # Perturbations
    'j2_acceleration',
    'atmospheric_density',
    'atmospheric_drag_acceleration',
    'solar_radiation_pressure_acceleration',
    'perturbed_dynamics',

    # Relative motion
    'hill_clohessy_wiltshire_dynamics',
    'relative_dynamics_nonlinear',
    'hill_frame_to_inertial',
    'inertial_to_hill_frame',

    # Propagators
    'rk4_step',
    'propagate_orbit',
    'propagate_orbit_jax',
    'propagate_relative_orbit',
]

__version__ = '0.1.0'
