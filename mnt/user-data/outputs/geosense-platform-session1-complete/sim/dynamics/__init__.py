"""
Orbital dynamics module for GeoSense platform.

This module implements various orbital dynamics models including:
- Two-body problem (Keplerian orbits)
- J2 perturbation (Earth oblateness)
- Atmospheric drag
- Solar radiation pressure (SRP)
- Relative dynamics (Hill/Clohessy-Wiltshire equations)
- Nonlinear relative dynamics

All implementations use JAX for GPU acceleration and automatic differentiation.
"""

from sim.dynamics.keplerian import (
    two_body_dynamics,
    orbital_elements_to_cartesian,
    cartesian_to_orbital_elements,
    mean_motion,
    orbital_period,
)

from sim.dynamics.perturbations import (
    j2_acceleration,
    atmospheric_drag_acceleration,
    solar_radiation_pressure_acceleration,
    perturbed_dynamics,
)

from sim.dynamics.relative import (
    hill_clohessy_wiltshire_dynamics,
    relative_dynamics_nonlinear,
    hill_frame_to_inertial,
    inertial_to_hill_frame,
)

from sim.dynamics.propagators import (
    rk4_step,
    propagate_orbit,
    propagate_relative_orbit,
)

__all__ = [
    # Keplerian dynamics
    "two_body_dynamics",
    "orbital_elements_to_cartesian",
    "cartesian_to_orbital_elements",
    "mean_motion",
    "orbital_period",
    # Perturbations
    "j2_acceleration",
    "atmospheric_drag_acceleration",
    "solar_radiation_pressure_acceleration",
    "perturbed_dynamics",
    # Relative dynamics
    "hill_clohessy_wiltshire_dynamics",
    "relative_dynamics_nonlinear",
    "hill_frame_to_inertial",
    "inertial_to_hill_frame",
    # Propagators
    "rk4_step",
    "propagate_orbit",
    "propagate_relative_orbit",
]
