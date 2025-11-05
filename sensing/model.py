"""
Optical/RF phase and time-delay measurement model for satellite gravimetry.

This module implements high-precision ranging measurements with realistic noise sources:
- Shot noise (photon counting statistics)
- Frequency/phase noise
- Clock instability (Allan deviation)
- Pointing jitter
- Thermal noise

Supports both inter-satellite ranging and ground-based tracking.
"""

from typing import Tuple, Optional, Dict
import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from enum import Enum


# Physical constants
C_LIGHT = 299792458.0  # m/s - Speed of light
H_PLANCK = 6.62607015e-34  # J·s - Planck constant
K_BOLTZMANN = 1.380649e-23  # J/K - Boltzmann constant


class MeasurementType(Enum):
    """Types of ranging measurements."""
    PHASE = "phase"  # Phase measurement (cycles or radians)
    TIME_DELAY = "time_delay"  # Time-of-flight measurement (seconds)
    RANGE = "range"  # Direct range measurement (meters)
    RANGE_RATE = "range_rate"  # Range rate / Doppler (m/s)


@dataclass
class NoiseParameters:
    """Comprehensive noise parameter specification."""
    
    # Shot noise (photon counting)
    photon_rate: float = 1e9  # photons/s
    quantum_efficiency: float = 0.8  # detector efficiency
    
    # Frequency/phase noise
    frequency_noise_psd: float = 1e-24  # Hz²/Hz at 1 Hz (one-sided)
    phase_noise_floor: float = 1e-6  # rad/√Hz
    
    # Clock stability (Allan deviation)
    allan_dev_coefficients: Dict[str, float] = None  # τ^(-1/2), τ^0, τ^(1/2) terms
    
    # Pointing jitter
    pointing_jitter_rms: float = 1e-6  # rad RMS
    beam_divergence: float = 10e-6  # rad
    
    # Thermal noise
    temperature: float = 300.0  # K
    bandwidth: float = 1e6  # Hz
    
    # Calibration uncertainties
    range_bias: float = 0.0  # m - systematic range offset
    range_bias_drift: float = 1e-9  # m/s - bias drift rate
    scale_factor_error: float = 1e-6  # fractional scale error
    
    def __post_init__(self):
        """Initialize Allan deviation if not provided."""
        if self.allan_dev_coefficients is None:
            # Default: hydrogen maser-class stability
            self.allan_dev_coefficients = {
                'tau_minus_half': 2e-13,  # White frequency noise
                'tau_zero': 5e-15,        # Flicker frequency noise
                'tau_plus_half': 1e-16    # Random walk frequency noise
            }


@dataclass
class OpticalLink:
    """Optical link parameters for ranging."""
    
    wavelength: float = 1064e-9  # m - Nd:YAG laser
    power_transmitted: float = 1.0  # W
    aperture_diameter: float = 0.3  # m
    range: float = 100e3  # m - typical inter-satellite distance
    
    @property
    def frequency(self) -> float:
        """Optical frequency (Hz)."""
        return C_LIGHT / self.wavelength
    
    @property
    def aperture_area(self) -> float:
        """Aperture area (m²)."""
        return np.pi * (self.aperture_diameter / 2)**2


@dataclass
class MeasurementModel:
    """Complete measurement model with noise characterization."""
    
    noise_params: NoiseParameters
    link: OpticalLink
    integration_time: float = 1.0  # s
    
    def geometric_range(
        self,
        pos1: jnp.ndarray,
        pos2: jnp.ndarray
    ) -> float:
        """
        Compute geometric range between two positions.
        
        Args:
            pos1: Position 1 [x, y, z] (m)
            pos2: Position 2 [x, y, z] (m)
        
        Returns:
            Range (m)
        """
        return jnp.linalg.norm(pos2 - pos1)
    
    def geometric_range_rate(
        self,
        pos1: jnp.ndarray,
        vel1: jnp.ndarray,
        pos2: jnp.ndarray,
        vel2: jnp.ndarray
    ) -> float:
        """
        Compute geometric range rate (line-of-sight velocity).
        
        Args:
            pos1, vel1: Position and velocity of satellite 1
            pos2, vel2: Position and velocity of satellite 2
        
        Returns:
            Range rate (m/s)
        """
        delta_pos = pos2 - pos1
        delta_vel = vel2 - vel1
        range_val = jnp.linalg.norm(delta_pos)
        
        # Project relative velocity onto line of sight
        return jnp.dot(delta_vel, delta_pos) / range_val
    
    def time_of_flight(
        self,
        pos1: jnp.ndarray,
        pos2: jnp.ndarray
    ) -> float:
        """
        Compute light travel time.
        
        Args:
            pos1: Transmitter position [x, y, z] (m)
            pos2: Receiver position [x, y, z] (m)
        
        Returns:
            Time of flight (s)
        """
        range_val = self.geometric_range(pos1, pos2)
        return range_val / C_LIGHT
    
    def phase_measurement(
        self,
        pos1: jnp.ndarray,
        pos2: jnp.ndarray
    ) -> float:
        """
        Compute phase measurement (in cycles).
        
        Args:
            pos1: Transmitter position [x, y, z] (m)
            pos2: Receiver position [x, y, z] (m)
        
        Returns:
            Phase in cycles (ambiguous, modulo integer cycles)
        """
        range_val = self.geometric_range(pos1, pos2)
        # Phase = range / wavelength (modulo integer cycles)
        return range_val / self.link.wavelength
    
    def shot_noise_std(self) -> float:
        """
        Compute shot noise standard deviation.
        
        Returns:
            Shot noise std in range equivalent (m)
        
        Notes:
            Shot noise limited ranging precision:
            
            σ_shot = λ / (2π√(2ηNτ))
            
            where:
            - λ: wavelength
            - η: quantum efficiency
            - N: photon rate
            - τ: integration time
        """
        # Photons detected per integration time
        n_photons = (
            self.noise_params.photon_rate * 
            self.noise_params.quantum_efficiency * 
            self.integration_time
        )
        
        # Shot noise in phase (radians)
        phase_noise_rad = 1.0 / jnp.sqrt(2.0 * n_photons)
        
        # Convert to range
        return self.link.wavelength * phase_noise_rad / (2.0 * jnp.pi)
    
    def frequency_noise_std(self) -> float:
        """
        Compute frequency/phase noise contribution.
        
        Returns:
            Phase noise std in range equivalent (m)
        
        Notes:
            Frequency noise contributes to ranging via:
            
            σ_freq = (R/c) * √(Sᵩ(f) / τ)
            
            where Sᵩ(f) is the one-sided phase noise PSD.
        """
        # Phase noise in radians
        phase_noise_rad = self.noise_params.phase_noise_floor / jnp.sqrt(
            self.integration_time
        )
        
        # Convert to range (for two-way ranging, use 2R)
        range_equiv = self.link.range / C_LIGHT
        return range_equiv * phase_noise_rad
    
    def clock_noise_std(self, tau: Optional[float] = None) -> float:
        """
        Compute clock instability contribution using Allan deviation.
        
        Args:
            tau: Averaging time (defaults to integration_time)
        
        Returns:
            Clock noise std in range equivalent (m)
        
        Notes:
            Allan deviation characterizes clock stability:
            
            σ_y(τ) = a₋₁/√τ + a₀ + a₁√τ
            
            Range noise: σ_R = c * σ_y(τ) * t_flight
        """
        if tau is None:
            tau = self.integration_time
        
        coeffs = self.noise_params.allan_dev_coefficients
        
        # Allan deviation
        allan_dev = (
            coeffs['tau_minus_half'] / jnp.sqrt(tau) +
            coeffs['tau_zero'] +
            coeffs['tau_plus_half'] * jnp.sqrt(tau)
        )
        
        # Range uncertainty from clock instability
        # For ranging: multiply by light travel time
        t_flight = self.link.range / C_LIGHT
        return C_LIGHT * allan_dev * t_flight
    
    def pointing_noise_std(self) -> float:
        """
        Compute pointing jitter contribution.
        
        Returns:
            Pointing-induced range noise std (m)
        
        Notes:
            Pointing jitter causes effective range variations:
            
            σ_point ≈ R * θ_jitter
            
            where θ_jitter is the RMS pointing error.
        """
        return self.link.range * self.noise_params.pointing_jitter_rms
    
    def thermal_noise_std(self) -> float:
        """
        Compute thermal (Johnson-Nyquist) noise.
        
        Returns:
            Thermal noise std in signal units
        
        Notes:
            Thermal noise power: P_thermal = kTB
            
            This is typically negligible for optical systems but can
            matter for RF systems or low signal levels.
        """
        thermal_power = (
            K_BOLTZMANN * 
            self.noise_params.temperature * 
            self.noise_params.bandwidth
        )
        
        # Rough conversion to range (system-dependent)
        # This is a simplified model
        snr = self.link.power_transmitted / thermal_power
        phase_noise = 1.0 / jnp.sqrt(snr)
        
        return self.link.wavelength * phase_noise / (2.0 * jnp.pi)
    
    def total_noise_std(self) -> float:
        """
        Compute total measurement noise (RSS combination).
        
        Returns:
            Total noise standard deviation (m)
        
        Notes:
            Assumes independent noise sources, combined in quadrature:
            
            σ_total = √(σ₁² + σ₂² + ... + σₙ²)
        """
        shot = self.shot_noise_std()
        freq = self.frequency_noise_std()
        clock = self.clock_noise_std()
        pointing = self.pointing_noise_std()
        thermal = self.thermal_noise_std()
        
        return jnp.sqrt(shot**2 + freq**2 + clock**2 + pointing**2 + thermal**2)
    
    def noise_budget(self) -> Dict[str, float]:
        """
        Generate complete noise budget breakdown.
        
        Returns:
            Dictionary of noise contributions (m)
        """
        return {
            'shot_noise': float(self.shot_noise_std()),
            'frequency_noise': float(self.frequency_noise_std()),
            'clock_noise': float(self.clock_noise_std()),
            'pointing_noise': float(self.pointing_noise_std()),
            'thermal_noise': float(self.thermal_noise_std()),
            'total_rss': float(self.total_noise_std())
        }
    
    def generate_measurement(
        self,
        pos1: jnp.ndarray,
        pos2: jnp.ndarray,
        key: jax.random.PRNGKey,
        measurement_type: MeasurementType = MeasurementType.RANGE
    ) -> Tuple[float, float]:
        """
        Generate noisy measurement.
        
        Args:
            pos1: Position 1 [x, y, z] (m)
            pos2: Position 2 [x, y, z] (m)
            key: JAX random key
            measurement_type: Type of measurement to generate
        
        Returns:
            measurement: Noisy measurement value
            std: Measurement standard deviation
        """
        # True geometric value
        if measurement_type == MeasurementType.RANGE:
            true_value = self.geometric_range(pos1, pos2)
        elif measurement_type == MeasurementType.TIME_DELAY:
            true_value = self.time_of_flight(pos1, pos2)
        elif measurement_type == MeasurementType.PHASE:
            true_value = self.phase_measurement(pos1, pos2)
        else:
            raise ValueError(f"Unsupported measurement type: {measurement_type}")
        
        # Apply bias and scale factor
        true_value = (
            true_value * (1.0 + self.noise_params.scale_factor_error) +
            self.noise_params.range_bias
        )
        
        # Add noise
        noise_std = self.total_noise_std()
        noise = jax.random.normal(key) * noise_std
        
        measurement = true_value + noise
        
        return measurement, noise_std


def allan_deviation(
    time_series: np.ndarray,
    dt: float,
    max_tau: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Allan deviation of a time series.
    
    Args:
        time_series: Array of measurements
        dt: Sampling interval (s)
        max_tau: Maximum averaging time in samples (default: N/4)
    
    Returns:
        taus: Array of averaging times (s)
        adevs: Allan deviations at each tau
    
    Notes:
        Allan deviation quantifies frequency stability:
        
        σ_y(τ) = √(⟨(ȳₖ₊₁ - ȳₖ)²⟩ / 2)
        
        where ȳₖ is the average over interval k of length τ.
    """
    n = len(time_series)
    
    if max_tau is None:
        max_tau = n // 4
    
    # Logarithmically spaced taus
    taus_samples = np.unique(np.logspace(0, np.log10(max_tau), 50).astype(int))
    taus = taus_samples * dt
    
    adevs = []
    
    for tau_samples in taus_samples:
        # Number of bins
        n_bins = n // tau_samples
        
        if n_bins < 3:
            break
        
        # Compute bin averages
        bins = [
            np.mean(time_series[i*tau_samples:(i+1)*tau_samples])
            for i in range(n_bins)
        ]
        
        # Allan deviation
        diffs = np.diff(bins)
        adev = np.sqrt(0.5 * np.mean(diffs**2))
        adevs.append(adev)
    
    return taus[:len(adevs)], np.array(adevs)


def power_spectral_density(
    time_series: np.ndarray,
    dt: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute one-sided power spectral density.
    
    Args:
        time_series: Array of measurements
        dt: Sampling interval (s)
    
    Returns:
        freqs: Frequency array (Hz)
        psd: One-sided PSD
    """
    n = len(time_series)
    
    # FFT
    fft = np.fft.rfft(time_series - np.mean(time_series))
    
    # One-sided PSD
    psd = 2.0 * dt * np.abs(fft)**2 / n
    
    # Frequency array
    freqs = np.fft.rfftfreq(n, dt)
    
    return freqs, psd


class NoiseGenerator:
    """Generate realistic time-correlated noise sequences."""
    
    @staticmethod
    def white_noise(
        key: jax.random.PRNGKey,
        n_samples: int,
        std: float
    ) -> jnp.ndarray:
        """Generate white Gaussian noise."""
        return jax.random.normal(key, (n_samples,)) * std
    
    @staticmethod
    def random_walk(
        key: jax.random.PRNGKey,
        n_samples: int,
        diffusion_coeff: float,
        dt: float
    ) -> jnp.ndarray:
        """
        Generate random walk (integrated white noise).
        
        Args:
            key: Random key
            n_samples: Number of samples
            diffusion_coeff: Diffusion coefficient (units²/s)
            dt: Time step (s)
        
        Returns:
            Random walk time series
        """
        increments = jax.random.normal(key, (n_samples,)) * np.sqrt(diffusion_coeff * dt)
        return jnp.cumsum(increments)
    
    @staticmethod
    def flicker_noise(
        key: jax.random.PRNGKey,
        n_samples: int,
        dt: float,
        alpha: float = 1.0
    ) -> np.ndarray:
        """
        Generate 1/f (flicker) noise using spectral synthesis.
        
        Args:
            key: Random key
            n_samples: Number of samples
            dt: Time step
            alpha: Power law exponent (1 for pink noise)
        
        Returns:
            Flicker noise time series
        """
        # Generate white noise in frequency domain
        key1, key2 = jax.random.split(key)
        real = jax.random.normal(key1, (n_samples // 2 + 1,))
        imag = jax.random.normal(key2, (n_samples // 2 + 1,))
        
        # Frequency array
        freqs = np.fft.rfftfreq(n_samples, dt)
        freqs[0] = 1.0  # Avoid division by zero
        
        # Apply 1/f^alpha filter
        filter_response = 1.0 / (freqs ** (alpha / 2.0))
        filter_response[0] = 0.0  # Remove DC
        
        # Filtered spectrum
        spectrum = (real + 1j * imag) * filter_response
        
        # Inverse FFT
        time_series = np.fft.irfft(spectrum, n_samples)
        
        # Normalize
        return time_series / np.std(time_series)


# JIT compile for performance
@jax.jit
def _jit_geometric_range(pos1: jnp.ndarray, pos2: jnp.ndarray) -> float:
    """JIT-compiled geometric range calculation."""
    return jnp.linalg.norm(pos2 - pos1)


@jax.jit  
def _jit_range_rate(
    pos1: jnp.ndarray,
    vel1: jnp.ndarray,
    pos2: jnp.ndarray,
    vel2: jnp.ndarray
) -> float:
    """JIT-compiled range rate calculation."""
    delta_pos = pos2 - pos1
    delta_vel = vel2 - vel1
    range_val = jnp.linalg.norm(delta_pos)
    return jnp.dot(delta_vel, delta_pos) / range_val
