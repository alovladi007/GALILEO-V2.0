"""
Noise models for laser interferometry.

Implements various noise sources that affect phase measurements:
1. Shot noise - fundamental quantum limit
2. Laser frequency noise - instability in laser frequency
3. Pointing jitter - angular misalignment between satellites
4. Clock jitter - timing errors in measurement
5. Acceleration noise - spacecraft vibrations

All noise models return standard deviations or generate noise realizations
that can be added to ideal phase measurements.
"""

from typing import Optional, Tuple
import jax
import jax.numpy as jnp
from jax import Array
import jax.random as random

# Physical constants
H_PLANCK = 6.62607015e-34  # J⋅s, Planck's constant
C_LIGHT = 299792458.0      # m/s, speed of light
WAVELENGTH_DEFAULT = 1064e-9  # m, Nd:YAG laser wavelength


def shot_noise_std(
    power: float,
    wavelength: float = WAVELENGTH_DEFAULT,
    bandwidth: float = 1.0,
    quantum_efficiency: float = 0.8
) -> float:
    """
    Compute shot noise standard deviation in phase measurement.
    
    Shot noise is the fundamental quantum limit in optical measurements,
    arising from the discrete nature of photons. It sets the lower bound
    on measurement precision.
    
    Equation:
        σ_shot = sqrt(h*f*B / (2*η*P))  [rad/√Hz]
    
    where:
        - h is Planck's constant
        - f is the optical frequency (c/λ)
        - B is the detection bandwidth
        - η is quantum efficiency of detector
        - P is received optical power
    
    Args:
        power: Received optical power (W)
        wavelength: Laser wavelength (m), default 1064 nm
        bandwidth: Detection bandwidth (Hz), default 1 Hz
        quantum_efficiency: Detector quantum efficiency (0-1), default 0.8
    
    Returns:
        Shot noise standard deviation (rad/√Hz)
    
    Note:
        For typical space gravimetry missions:
        - Power: 1-100 pW (after long-distance propagation)
        - Wavelength: 1064 nm (Nd:YAG) or 1550 nm (fiber laser)
        - Bandwidth: 0.1-10 Hz (depends on measurement rate)
        - η: 0.7-0.9 (typical for avalanche photodiodes)
    
    Reference:
        Sheard et al. (2012), "Intersatellite laser ranging instrument
        for the GRACE follow-on mission"
    
    Example:
        >>> # GRACE-FO conditions
        >>> sigma_shot = shot_noise_std(power=10e-12, bandwidth=1.0)
        >>> print(f"Shot noise: {sigma_shot*1e9:.2f} nrad/√Hz")
    """
    # Optical frequency
    frequency = C_LIGHT / wavelength
    
    # Shot noise variance (rad²/Hz)
    # From photon counting statistics
    variance = (H_PLANCK * frequency * bandwidth) / (2 * quantum_efficiency * power)
    
    # Standard deviation (rad/√Hz)
    return jnp.sqrt(variance)


def laser_frequency_noise_std(
    frequency_stability: float,
    range_km: float,
    wavelength: float = WAVELENGTH_DEFAULT
) -> float:
    """
    Compute laser frequency noise contribution to phase measurement.
    
    Laser frequency instability causes phase noise proportional to the
    optical path length. This is typically the dominant noise source
    for ranges > 10 km.
    
    Equation:
        σ_freq = (2π/c) * (δf/f) * ρ  [rad/√Hz]
    
    where:
        - δf/f is the fractional frequency stability
        - ρ is the range (one-way)
        - Factor of 2π/c converts frequency to phase
    
    Args:
        frequency_stability: Fractional frequency stability (δf/f) [1/√Hz]
        range_km: Range between satellites (km)
        wavelength: Laser wavelength (m), not directly used but kept for consistency
    
    Returns:
        Frequency noise standard deviation (rad/√Hz)
    
    Note:
        Typical laser frequency stabilities:
        - Free-running laser: 1e-6 to 1e-8 [1/√Hz]
        - Stabilized laser: 1e-12 to 1e-14 [1/√Hz]
        - Ultrastable laser (cavity): 1e-15 to 1e-16 [1/√Hz]
        
        GRACE-FO uses iodine-stabilized Nd:YAG: ~1e-13 [1/√Hz]
    
    Example:
        >>> # GRACE-FO conditions
        >>> sigma_freq = laser_frequency_noise_std(
        ...     frequency_stability=1e-13,
        ...     range_km=220.0
        ... )
        >>> print(f"Frequency noise: {sigma_freq*1e9:.2f} nrad/√Hz")
    """
    # Convert range to meters
    range_m = range_km * 1000.0
    
    # Phase noise from frequency instability
    # σ_φ = (2π/c) * (δf/f) * ρ
    sigma = (2 * jnp.pi / C_LIGHT) * frequency_stability * range_m
    
    return sigma


def pointing_jitter_noise_std(
    pointing_jitter_rad: float,
    range_km: float,
    beam_divergence_rad: Optional[float] = None
) -> float:
    """
    Compute pointing jitter contribution to phase measurement noise.
    
    Angular misalignment between spacecraft causes optical path length
    variations, which appear as phase noise. This is geometric in nature
    and depends on both the jitter angle and the range.
    
    Equation (small angle):
        σ_pointing = (4π/λ) * ρ * θ  [rad/√Hz]
    
    where:
        - θ is the pointing jitter (rad)
        - ρ is the range
        - λ is the wavelength
    
    For large jitter or considering beam divergence:
        Additional losses occur when beam walks off detector
    
    Args:
        pointing_jitter_rad: RMS pointing jitter (rad/√Hz)
        range_km: Range between satellites (km)
        beam_divergence_rad: Optional beam divergence half-angle (rad)
            If provided, checks if jitter exceeds beam coverage
    
    Returns:
        Pointing noise standard deviation (rad/√Hz)
    
    Note:
        Typical pointing stabilities:
        - Coarse pointing (reaction wheels): ~1-10 μrad/√Hz
        - Fine pointing (piezo mirrors): ~10-100 nrad/√Hz
        - Ultrafine (drag-free): ~1-10 nrad/√Hz
        
        GRACE-FO achieves: ~10 μrad/√Hz with star trackers
    
    Warning:
        If pointing jitter > beam divergence, signal may be lost!
    
    Example:
        >>> sigma_pointing = pointing_jitter_noise_std(
        ...     pointing_jitter_rad=10e-6,  # 10 μrad
        ...     range_km=220.0
        ... )
    """
    # Convert range to meters
    range_m = range_km * 1000.0
    
    # Check if jitter exceeds beam divergence (warning condition)
    if beam_divergence_rad is not None:
        if pointing_jitter_rad > beam_divergence_rad:
            # This is a warning condition but we still compute the noise
            # In practice, the link would be intermittent
            pass
    
    # Path length variation from pointing error
    # For small angles: Δρ ≈ ρ * θ²/2
    # But for phase: Δφ = (4π/λ) * ρ * θ (linear in small angle approx)
    
    wavelength_m = WAVELENGTH_DEFAULT
    sigma = (4 * jnp.pi / wavelength_m) * range_m * pointing_jitter_rad
    
    return sigma


def clock_jitter_noise_std(
    clock_stability: float,
    range_rate_km_s: float,
    wavelength: float = WAVELENGTH_DEFAULT
) -> float:
    """
    Compute clock jitter contribution to phase measurement noise.
    
    Timing errors in the measurement system cause uncertainty in when
    the phase is sampled. Combined with range rate (Doppler), this
    produces phase noise.
    
    Equation:
        σ_clock = (4π/λ) * ρ̇ * τ  [rad/√Hz]
    
    where:
        - ρ̇ is the range rate
        - τ is the timing jitter (Allan deviation at 1 s)
        - Factor includes round-trip (2x) and wavelength conversion
    
    Args:
        clock_stability: Clock stability (s/√Hz or fractional at 1s)
            This is typically the Allan deviation at τ=1s
        range_rate_km_s: Range rate between satellites (km/s)
        wavelength: Laser wavelength (m)
    
    Returns:
        Clock noise standard deviation (rad/√Hz)
    
    Note:
        Typical clock stabilities (Allan deviation at 1s):
        - Crystal oscillator: 1e-9 to 1e-11
        - Rubidium standard: 1e-11 to 1e-12
        - Cesium standard: 1e-12 to 1e-13
        - Hydrogen maser: 1e-13 to 1e-15
        
        GRACE-FO uses USO (Ultra-Stable Oscillator): ~1e-12
    
    Example:
        >>> sigma_clock = clock_jitter_noise_std(
        ...     clock_stability=1e-12,
        ...     range_rate_km_s=0.001  # 1 m/s relative velocity
        ... )
    """
    # Convert range rate to m/s
    range_rate_m_s = range_rate_km_s * 1000.0
    
    # Phase noise from timing jitter
    # Timing error δt causes phase error: δφ = (4π/λ) * ρ̇ * δt
    sigma = (4 * jnp.pi / wavelength) * range_rate_m_s * clock_stability
    
    return sigma


def acceleration_noise_std(
    acceleration_noise: float,
    frequency: float,
    range_km: float,
    wavelength: float = WAVELENGTH_DEFAULT
) -> float:
    """
    Compute spacecraft acceleration noise contribution.
    
    Spacecraft vibrations and disturbances cause acceleration noise that
    appears in the phase measurement through double integration. This is
    important for drag-free satellites and sensitive accelerometers.
    
    Equation:
        σ_accel = (4π/λ) * (a_noise / (2πf)²)  [rad/√Hz]
    
    where:
        - a_noise is acceleration noise PSD (m/s²/√Hz)
        - f is measurement frequency
        - Factor of (2πf)⁻² comes from double integration
    
    Args:
        acceleration_noise: Acceleration noise PSD (m/s²/√Hz)
        frequency: Measurement frequency (Hz)
        range_km: Range between satellites (km) - used for consistency
        wavelength: Laser wavelength (m)
    
    Returns:
        Acceleration noise standard deviation (rad/√Hz)
    
    Note:
        Typical acceleration noise levels:
        - Conventional satellite: 1e-6 to 1e-8 m/s²/√Hz
        - Drag-free satellite: 1e-10 to 1e-13 m/s²/√Hz
        
        GRACE-FO achieves: ~3e-11 m/s²/√Hz at 0.1 Hz
    
    Example:
        >>> sigma_accel = acceleration_noise_std(
        ...     acceleration_noise=3e-11,  # m/s²/√Hz
        ...     frequency=0.1,  # Hz
        ...     range_km=220.0
        ... )
    """
    # Double integration: acceleration -> displacement
    # x(f) = a(f) / (2πf)²
    omega = 2 * jnp.pi * frequency
    displacement_noise = acceleration_noise / (omega ** 2)
    
    # Convert to phase noise
    # Round-trip path length variation -> phase
    sigma = (4 * jnp.pi / wavelength) * displacement_noise
    
    return sigma


def total_phase_noise_std(
    power: float,
    range_km: float,
    range_rate_km_s: float,
    frequency_stability: float = 1e-13,
    pointing_jitter_rad: float = 10e-6,
    clock_stability: float = 1e-12,
    acceleration_noise: float = 3e-11,
    measurement_frequency: float = 0.1,
    bandwidth: float = 1.0,
    wavelength: float = WAVELENGTH_DEFAULT
) -> Tuple[float, dict]:
    """
    Compute total phase measurement noise from all sources.
    
    Combines all noise sources assuming they are uncorrelated (add in
    quadrature). Returns both the total noise and a breakdown by source.
    
    Args:
        power: Received optical power (W)
        range_km: Range between satellites (km)
        range_rate_km_s: Range rate (km/s)
        frequency_stability: Laser frequency stability (1/√Hz)
        pointing_jitter_rad: Pointing jitter (rad/√Hz)
        clock_stability: Clock stability (s/√Hz)
        acceleration_noise: Acceleration noise (m/s²/√Hz)
        measurement_frequency: Measurement frequency (Hz)
        bandwidth: Detection bandwidth (Hz)
        wavelength: Laser wavelength (m)
    
    Returns:
        Tuple of (total_noise, noise_breakdown):
            - total_noise: Total noise std (rad/√Hz)
            - noise_breakdown: Dict with individual noise contributions
    
    Example:
        >>> # GRACE-FO-like conditions
        >>> total, breakdown = total_phase_noise_std(
        ...     power=10e-12,      # 10 pW
        ...     range_km=220.0,    # 220 km
        ...     range_rate_km_s=0.001,  # 1 m/s
        ... )
        >>> print(f"Total noise: {total*1e9:.2f} nrad/√Hz")
        >>> for source, noise in breakdown.items():
        ...     print(f"  {source}: {noise*1e9:.2f} nrad/√Hz")
    """
    # Compute individual noise sources
    shot = shot_noise_std(power, wavelength, bandwidth)
    frequency = laser_frequency_noise_std(frequency_stability, range_km, wavelength)
    pointing = pointing_jitter_noise_std(pointing_jitter_rad, range_km)
    clock = clock_jitter_noise_std(clock_stability, range_rate_km_s, wavelength)
    accel = acceleration_noise_std(
        acceleration_noise, measurement_frequency, range_km, wavelength
    )
    
    # Add in quadrature (assuming uncorrelated)
    total = jnp.sqrt(shot**2 + frequency**2 + pointing**2 + clock**2 + accel**2)
    
    # Create breakdown dictionary
    breakdown = {
        'shot_noise': shot,
        'frequency_noise': frequency,
        'pointing_jitter': pointing,
        'clock_jitter': clock,
        'acceleration_noise': accel,
        'total': total
    }
    
    return total, breakdown


def generate_noise_realization(
    key: Array,
    noise_std: float,
    n_samples: int
) -> Array:
    """
    Generate a noise realization with specified standard deviation.
    
    Creates Gaussian white noise samples that can be added to ideal
    phase measurements to simulate realistic observations.
    
    Args:
        key: JAX random key
        noise_std: Noise standard deviation (rad/√Hz)
        n_samples: Number of samples to generate
    
    Returns:
        Array of noise samples (rad)
    
    Example:
        >>> key = random.PRNGKey(42)
        >>> noise = generate_noise_realization(key, noise_std=1e-9, n_samples=1000)
        >>> # Add to ideal measurements
        >>> phase_measured = phase_ideal + noise
    """
    return noise_std * random.normal(key, shape=(n_samples,))


def noise_equivalent_range(noise_std_rad: float, wavelength: float = WAVELENGTH_DEFAULT) -> float:
    """
    Convert phase noise to equivalent range noise.
    
    Useful for understanding noise in more intuitive units (meters).
    
    Equation:
        σ_range = (λ / 4π) * σ_phase
    
    Args:
        noise_std_rad: Phase noise standard deviation (rad/√Hz)
        wavelength: Laser wavelength (m)
    
    Returns:
        Equivalent range noise (m/√Hz)
    
    Example:
        >>> phase_noise = 1e-9  # 1 nrad/√Hz
        >>> range_noise = noise_equivalent_range(phase_noise)
        >>> print(f"Equivalent range noise: {range_noise*1e9:.2f} nm/√Hz")
    """
    return (wavelength / (4 * jnp.pi)) * noise_std_rad


# JIT-compile for performance
shot_noise_std = jax.jit(shot_noise_std)
laser_frequency_noise_std = jax.jit(laser_frequency_noise_std)
pointing_jitter_noise_std = jax.jit(pointing_jitter_noise_std)
clock_jitter_noise_std = jax.jit(clock_jitter_noise_std)
acceleration_noise_std = jax.jit(acceleration_noise_std)
total_phase_noise_std = jax.jit(total_phase_noise_std)
noise_equivalent_range = jax.jit(noise_equivalent_range)
