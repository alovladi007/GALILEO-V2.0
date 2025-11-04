"""
Laser interferometry phase measurement model.

The phase measurement Δφ(t) is the primary observable in satellite-to-satellite
laser ranging. It's related to the range ρ and range-rate ρ̇ through:

    Δφ(t) = (2π/λ) * 2ρ(t) + integrated_phase
    
    φ̇(t) = (2π/λ) * 2ρ̇(t)

where:
    - λ is the laser wavelength
    - Factor of 2 accounts for round-trip path
    - ρ(t) is the instantaneous range between satellites
    - ρ̇(t) is the range rate
"""

import jax
import jax.numpy as jnp
from jax import Array
from typing import Optional

# Physical constants
C_LIGHT = 299792.458  # Speed of light (km/s)
WAVELENGTH_DEFAULT = 1064e-9 * 1e-6  # 1064 nm Nd:YAG laser (km)


def compute_phase(
    range_km: float,
    wavelength: float = WAVELENGTH_DEFAULT,
    phase_offset: float = 0.0
) -> float:
    """
    Compute phase measurement from instantaneous range.
    
    The phase is proportional to the optical path length. For a round-trip
    measurement (like laser ranging), the path length is twice the range.
    
    Equation:
        Δφ = (2π/λ) * 2ρ + φ₀
    
    where:
        - λ is the laser wavelength
        - ρ is the range (one-way distance)
        - φ₀ is an arbitrary phase offset
    
    Args:
        range_km: Range between satellites (km)
        wavelength: Laser wavelength (km), default is 1064 nm
        phase_offset: Arbitrary phase offset (rad), default 0
    
    Returns:
        Phase measurement (rad)
    
    Note:
        The phase is typically wrapped to [-π, π] or [0, 2π] in practice,
        but this function returns the unwrapped phase for continuous tracking.
    
    Example:
        >>> # 100 km range with 1064 nm laser
        >>> phase = compute_phase(100.0)
        >>> print(f"Phase: {phase:.2e} rad")
    """
    # Two-way path length
    path_length = 2.0 * range_km
    
    # Wave number k = 2π/λ
    k = 2.0 * jnp.pi / wavelength
    
    # Phase
    phase = k * path_length + phase_offset
    
    return phase


def compute_phase_rate(
    range_rate_km_s: float,
    wavelength: float = WAVELENGTH_DEFAULT
) -> float:
    """
    Compute phase rate from instantaneous range rate.
    
    The phase rate (frequency) is proportional to the range rate. This is
    the basis for Doppler velocity measurements.
    
    Equation:
        φ̇ = (2π/λ) * 2ρ̇
    
    where:
        - λ is the laser wavelength
        - ρ̇ is the range rate (relative velocity along line-of-sight)
    
    Args:
        range_rate_km_s: Range rate between satellites (km/s)
        wavelength: Laser wavelength (km)
    
    Returns:
        Phase rate (rad/s)
    
    Note:
        This can be related to the Doppler frequency shift:
        Δf = φ̇ / (2π) = (2 / λ) * ρ̇
    
    Example:
        >>> # 1 m/s relative velocity
        >>> phase_rate = compute_phase_rate(0.001)  # 0.001 km/s
        >>> freq_shift = phase_rate / (2 * jnp.pi)
        >>> print(f"Doppler shift: {freq_shift:.2f} Hz")
    """
    # Two-way Doppler
    doppler_velocity = 2.0 * range_rate_km_s
    
    # Wave number k = 2π/λ
    k = 2.0 * jnp.pi / wavelength
    
    # Phase rate
    phase_rate = k * doppler_velocity
    
    return phase_rate


def range_to_phase(
    ranges: Array,
    wavelength: float = WAVELENGTH_DEFAULT,
    initial_phase: float = 0.0
) -> Array:
    """
    Convert time series of ranges to phase measurements.
    
    This is the forward model for interferometric ranging. It can be used
    to simulate measurements from orbital dynamics.
    
    Args:
        ranges: Array of range measurements (km)
        wavelength: Laser wavelength (km)
        initial_phase: Initial phase offset (rad)
    
    Returns:
        Array of phase measurements (rad)
    
    Example:
        >>> ranges = jnp.linspace(100.0, 105.0, 100)
        >>> phases = range_to_phase(ranges)
    """
    return jax.vmap(
        lambda r: compute_phase(r, wavelength, initial_phase)
    )(ranges)


def phase_to_range(
    phases: Array,
    wavelength: float = WAVELENGTH_DEFAULT,
    unwrap: bool = True
) -> Array:
    """
    Convert phase measurements to ranges (inverse model).
    
    This is the inverse transformation, useful for processing real data
    or validating the forward model.
    
    Equation:
        ρ = (λ / 4π) * Δφ
    
    Args:
        phases: Array of phase measurements (rad)
        wavelength: Laser wavelength (km)
        unwrap: If True, unwrap phase discontinuities
    
    Returns:
        Array of range measurements (km)
    
    Note:
        Phase unwrapping is necessary when the phase wraps around ±π.
        JAX's unwrap function handles this automatically.
    
    Example:
        >>> phases = jnp.array([0.0, jnp.pi, 2*jnp.pi, 3*jnp.pi])
        >>> ranges = phase_to_range(phases)
    """
    if unwrap:
        phases = jnp.unwrap(phases)
    
    # Invert the phase equation: ρ = (λ/4π) * φ
    ranges = (wavelength / (4.0 * jnp.pi)) * phases
    
    return ranges


def compute_phase_from_states(
    r1: Array,
    r2: Array,
    wavelength: float = WAVELENGTH_DEFAULT,
    phase_offset: float = 0.0
) -> float:
    """
    Compute phase measurement from satellite position vectors.
    
    Convenience function that computes range from positions and then
    converts to phase. Useful for generating synthetic measurements
    from orbital dynamics.
    
    Args:
        r1: Position vector of satellite 1 (km)
        r2: Position vector of satellite 2 (km)
        wavelength: Laser wavelength (km)
        phase_offset: Phase offset (rad)
    
    Returns:
        Phase measurement (rad)
    
    Example:
        >>> r1 = jnp.array([7000.0, 0.0, 0.0])
        >>> r2 = jnp.array([7100.0, 0.0, 0.0])
        >>> phase = compute_phase_from_states(r1, r2)
    """
    range_km = jnp.linalg.norm(r2 - r1)
    return compute_phase(range_km, wavelength, phase_offset)


def compute_phase_rate_from_states(
    r1: Array,
    v1: Array,
    r2: Array,
    v2: Array,
    wavelength: float = WAVELENGTH_DEFAULT
) -> float:
    """
    Compute phase rate from satellite state vectors.
    
    Computes the range rate from position and velocity vectors, then
    converts to phase rate.
    
    Equation for range rate:
        ρ̇ = (r₂ - r₁) · (v₂ - v₁) / |r₂ - r₁|
    
    Args:
        r1: Position vector of satellite 1 (km)
        v1: Velocity vector of satellite 1 (km/s)
        r2: Position vector of satellite 2 (km)
        v2: Velocity vector of satellite 2 (km/s)
        wavelength: Laser wavelength (km)
    
    Returns:
        Phase rate (rad/s)
    
    Example:
        >>> r1 = jnp.array([7000.0, 0.0, 0.0])
        >>> v1 = jnp.array([0.0, 7.5, 0.0])
        >>> r2 = jnp.array([7100.0, 0.0, 0.0])
        >>> v2 = jnp.array([0.0, 7.5, 0.0])
        >>> phase_rate = compute_phase_rate_from_states(r1, v1, r2, v2)
    """
    # Relative position and velocity
    delta_r = r2 - r1
    delta_v = v2 - v1
    
    # Range
    range_km = jnp.linalg.norm(delta_r)
    
    # Range rate (projection of relative velocity on line-of-sight)
    range_rate = jnp.dot(delta_r, delta_v) / range_km
    
    return compute_phase_rate(range_rate, wavelength)


# JIT-compile for performance
compute_phase = jax.jit(compute_phase)
compute_phase_rate = jax.jit(compute_phase_rate)
range_to_phase = jax.jit(range_to_phase)
phase_to_range = jax.jit(phase_to_range)
compute_phase_from_states = jax.jit(compute_phase_from_states)
compute_phase_rate_from_states = jax.jit(compute_phase_rate_from_states)
