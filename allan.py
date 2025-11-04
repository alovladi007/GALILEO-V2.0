"""
Allan deviation and noise characterization tools.

The Allan deviation is a measure of frequency stability widely used in
precision timing and frequency standards. It's particularly useful for
characterizing noise in laser interferometry and clock systems.

Implements:
- Standard Allan deviation
- Overlapping Allan deviation (better statistics)
- Modified Allan deviation
- Power spectral density estimation
- Noise type identification
"""

from typing import Tuple
import jax
import jax.numpy as jnp
from jax import Array


def allan_deviation(
    data: Array,
    sample_rate: float,
    tau_values: Array
) -> Array:
    """
    Compute standard Allan deviation.
    
    The Allan deviation quantifies stability as a function of averaging time.
    It's defined as the RMS of differences between adjacent averages.
    
    Equation:
        σ_y(τ) = sqrt(1/(2(M-1)) * Σ(ȳ_{i+1} - ȳ_i)²)
    
    where:
        - τ is the averaging time
        - ȳ_i is the average over interval i
        - M is the number of intervals
    
    Args:
        data: Time series data (e.g., phase measurements)
        sample_rate: Sampling rate (Hz)
        tau_values: Array of averaging times τ to compute (s)
    
    Returns:
        Array of Allan deviations corresponding to tau_values
    
    Note:
        Standard Allan deviation has limited statistical confidence for
        small datasets. Consider using overlapping_allan_deviation() for
        better statistics.
    
    Reference:
        Allan, D.W. (1966), "Statistics of atomic frequency standards"
    
    Example:
        >>> # Measure clock stability
        >>> phase_data = jnp.array([...])  # Phase measurements
        >>> tau = jnp.logspace(-2, 2, 20)  # 0.01s to 100s
        >>> adev = allan_deviation(phase_data, sample_rate=10.0, tau_values=tau)
        >>> # Plot log-log to identify noise types
    """
    n_samples = len(data)
    sample_period = 1.0 / sample_rate
    
    adev_values = []
    
    for tau in tau_values:
        # Number of samples per averaging interval
        m = int(tau / sample_period)
        
        if m < 1:
            # Tau too small for this sample rate
            adev_values.append(jnp.nan)
            continue
        
        # Number of complete intervals
        n_intervals = n_samples // m
        
        if n_intervals < 2:
            # Need at least 2 intervals
            adev_values.append(jnp.nan)
            continue
        
        # Compute averages for each interval
        averages = []
        for i in range(n_intervals):
            start_idx = i * m
            end_idx = start_idx + m
            avg = jnp.mean(data[start_idx:end_idx])
            averages.append(avg)
        
        averages = jnp.array(averages)
        
        # Compute differences between adjacent averages
        diffs = averages[1:] - averages[:-1]
        
        # Allan variance (two-sample variance)
        allan_var = jnp.mean(diffs ** 2) / 2.0
        
        # Allan deviation
        adev = jnp.sqrt(allan_var)
        adev_values.append(adev)
    
    return jnp.array(adev_values)


def overlapping_allan_deviation(
    data: Array,
    sample_rate: float,
    tau_values: Array
) -> Array:
    """
    Compute overlapping Allan deviation (better statistics).
    
    The overlapping Allan deviation uses all possible overlapping intervals,
    providing better statistical confidence than the standard version,
    especially for limited data.
    
    Algorithm:
        1. For each tau, compute averages over all overlapping intervals
        2. Compute differences between averages separated by tau
        3. Take RMS of differences
    
    Args:
        data: Time series data
        sample_rate: Sampling rate (Hz)
        tau_values: Array of averaging times τ (s)
    
    Returns:
        Array of overlapping Allan deviations
    
    Note:
        Overlapping ADEV converges faster (needs less data) than standard
        ADEV. Use this for most practical applications.
    
    Reference:
        Riley, W.J. (2008), "Handbook of Frequency Stability Analysis"
    
    Example:
        >>> # Better statistics with same data
        >>> oadev = overlapping_allan_deviation(
        ...     phase_data, sample_rate=10.0, tau_values=tau
        ... )
    """
    n_samples = len(data)
    sample_period = 1.0 / sample_rate
    
    oadev_values = []
    
    for tau in tau_values:
        # Number of samples per averaging interval
        m = int(tau / sample_period)
        
        if m < 1:
            oadev_values.append(jnp.nan)
            continue
        
        # Compute all possible overlapping averages
        n_averages = n_samples - m + 1
        
        if n_averages < 2:
            oadev_values.append(jnp.nan)
            continue
        
        # Use convolution for efficient overlapping average computation
        kernel = jnp.ones(m) / m
        averages = jnp.convolve(data, kernel, mode='valid')
        
        # Compute differences between averages separated by m samples
        # (i.e., separated by time tau)
        max_diff_idx = len(averages) - m
        if max_diff_idx < 1:
            oadev_values.append(jnp.nan)
            continue
        
        diffs = averages[m:] - averages[:-m]
        
        # Overlapping Allan variance
        oadev_var = jnp.mean(diffs ** 2) / 2.0
        
        # Overlapping Allan deviation
        oadev = jnp.sqrt(oadev_var)
        oadev_values.append(oadev)
    
    return jnp.array(oadev_values)


def modified_allan_deviation(
    data: Array,
    sample_rate: float,
    tau_values: Array
) -> Array:
    """
    Compute modified Allan deviation.
    
    The modified Allan deviation (MDEV) better distinguishes between
    white phase modulation (WPM) and flicker phase modulation (FPM).
    It uses longer averaging intervals.
    
    Equation:
        Mod σ_y(τ) = sqrt(1/(2n²(m-1)) * Σᵢ Σⱼ (x̄ⱼ,ᵢ₊₁ - x̄ⱼ,ᵢ)²)
    
    where averaging is done over n sub-intervals of length τ₀.
    
    Args:
        data: Time series data
        sample_rate: Sampling rate (Hz)
        tau_values: Array of averaging times τ (s)
    
    Returns:
        Array of modified Allan deviations
    
    Note:
        MDEV is particularly useful for identifying:
        - White PM (slope = -1)
        - Flicker PM (slope = -1)
        - White FM (slope = 0)
        - Flicker FM (slope = +0.5)
    
    Example:
        >>> mdev = modified_allan_deviation(
        ...     phase_data, sample_rate=10.0, tau_values=tau
        ... )
    """
    n_samples = len(data)
    sample_period = 1.0 / sample_rate
    
    mdev_values = []
    
    for tau in tau_values:
        # Number of samples per averaging interval
        m = int(tau / sample_period)
        
        if m < 2:
            mdev_values.append(jnp.nan)
            continue
        
        # Use m sub-intervals of length tau/m
        n_intervals = n_samples // m
        
        if n_intervals < 2:
            mdev_values.append(jnp.nan)
            continue
        
        # Compute averages with sub-averaging
        total_var = 0.0
        count = 0
        
        for i in range(n_intervals - 1):
            start_idx = i * m
            end_idx = start_idx + 2 * m
            
            if end_idx > n_samples:
                break
            
            # First interval
            avg1 = jnp.mean(data[start_idx:start_idx + m])
            
            # Second interval
            avg2 = jnp.mean(data[start_idx + m:end_idx])
            
            total_var += (avg2 - avg1) ** 2
            count += 1
        
        if count < 1:
            mdev_values.append(jnp.nan)
            continue
        
        # Modified Allan variance
        mdev_var = total_var / (2.0 * count)
        
        # Modified Allan deviation
        mdev = jnp.sqrt(mdev_var)
        mdev_values.append(mdev)
    
    return jnp.array(mdev_values)


def identify_noise_type(
    tau_values: Array,
    adev_values: Array
) -> Tuple[str, float]:
    """
    Identify noise type from Allan deviation slope.
    
    Different noise processes have characteristic slopes in log-log plots:
    - White Phase Modulation (WPM): slope = -1
    - Flicker Phase Modulation (FPM): slope = -1
    - White Frequency Modulation (WFM): slope = -0.5
    - Flicker Frequency Modulation (FFM): slope = 0
    - Random Walk Frequency Modulation (RWFM): slope = +0.5
    
    Args:
        tau_values: Averaging times (s)
        adev_values: Allan deviation values
    
    Returns:
        Tuple of (noise_type, slope):
            - noise_type: String describing the dominant noise
            - slope: Fitted slope in log-log space
    
    Note:
        Real data often shows multiple noise types at different tau values.
        This function identifies the dominant noise in the provided range.
    
    Example:
        >>> noise_type, slope = identify_noise_type(tau, adev)
        >>> print(f"Dominant noise: {noise_type} (slope: {slope:.2f})")
    """
    # Remove NaN values
    valid_mask = ~jnp.isnan(adev_values) & (adev_values > 0)
    tau_valid = tau_values[valid_mask]
    adev_valid = adev_values[valid_mask]
    
    if len(tau_valid) < 2:
        return "Unknown (insufficient data)", jnp.nan
    
    # Fit line in log-log space
    log_tau = jnp.log10(tau_valid)
    log_adev = jnp.log10(adev_valid)
    
    # Linear regression: log(ADEV) = slope * log(tau) + intercept
    A = jnp.vstack([log_tau, jnp.ones_like(log_tau)]).T
    slope, intercept = jnp.linalg.lstsq(A, log_adev, rcond=None)[0]
    
    # Classify based on slope
    if slope < -0.75:
        noise_type = "White Phase Modulation (WPM)"
    elif slope < -0.25:
        noise_type = "White Frequency Modulation (WFM)"
    elif slope < 0.25:
        noise_type = "Flicker Frequency Modulation (FFM)"
    elif slope < 0.75:
        noise_type = "Random Walk Frequency Modulation (RWFM)"
    else:
        noise_type = "Drift or other"
    
    return noise_type, float(slope)


def compute_noise_spectrum(
    data: Array,
    sample_rate: float
) -> Tuple[Array, Array]:
    """
    Compute power spectral density (PSD) of noise.
    
    The PSD shows how noise power is distributed across frequencies.
    It's complementary to Allan deviation and useful for identifying
    specific noise sources.
    
    Args:
        data: Time series data
        sample_rate: Sampling rate (Hz)
    
    Returns:
        Tuple of (frequencies, psd):
            - frequencies: Frequency array (Hz)
            - psd: Power spectral density (units²/Hz)
    
    Note:
        For phase measurements, PSD units are rad²/Hz.
        White noise appears flat in PSD.
        1/f noise (flicker) has slope = -1 in log-log plot.
    
    Example:
        >>> freqs, psd = compute_noise_spectrum(phase_data, sample_rate=10.0)
        >>> # Plot log-log to identify noise colors
        >>> import matplotlib.pyplot as plt
        >>> plt.loglog(freqs, psd)
        >>> plt.xlabel('Frequency (Hz)')
        >>> plt.ylabel('PSD (rad²/Hz)')
    """
    # Compute FFT
    fft_data = jnp.fft.rfft(data)
    
    # Compute power spectral density
    # Factor of 2 accounts for one-sided spectrum
    # Normalize by sample rate and number of samples
    n_samples = len(data)
    psd = 2.0 * jnp.abs(fft_data) ** 2 / (sample_rate * n_samples)
    
    # Frequency axis
    freqs = jnp.fft.rfftfreq(n_samples, 1.0 / sample_rate)
    
    return freqs, psd


def noise_floor_from_allan(
    adev_at_1s: float,
    measurement_bandwidth: float
) -> float:
    """
    Estimate noise floor from Allan deviation at 1 second.
    
    Provides a quick estimate of the white noise floor, useful for
    comparing different systems or validating noise models.
    
    Equation (for white noise):
        σ_white ≈ ADEV(τ=1s) * sqrt(2 * BW)
    
    Args:
        adev_at_1s: Allan deviation at tau = 1 second
        measurement_bandwidth: Measurement bandwidth (Hz)
    
    Returns:
        Estimated white noise floor (same units as ADEV)
    
    Example:
        >>> adev_1s = 1e-9  # 1 nrad at 1 second
        >>> noise_floor = noise_floor_from_allan(adev_1s, bandwidth=1.0)
        >>> print(f"Noise floor: {noise_floor*1e9:.2f} nrad/√Hz")
    """
    # For white noise: ADEV(τ) = σ / sqrt(2τ)
    # At τ = 1s: ADEV(1s) = σ / sqrt(2)
    # Therefore: σ = ADEV(1s) * sqrt(2)
    
    # Noise floor per sqrt(Hz)
    white_noise = adev_at_1s * jnp.sqrt(2.0)
    
    return white_noise


# JIT-compile for performance (where applicable)
# Note: Some functions use Python loops which aren't JIT-compatible
# These are kept as regular functions for flexibility
identify_noise_type = jax.jit(identify_noise_type)
compute_noise_spectrum = jax.jit(compute_noise_spectrum)
noise_floor_from_allan = jax.jit(noise_floor_from_allan)
