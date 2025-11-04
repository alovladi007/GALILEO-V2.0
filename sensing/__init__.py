"""
Sensor Data Processing Package

Handles processing of gravimetric sensor data including:
- Allan deviation and noise characterization (allan.py)
- Laser interferometry noise models (noise.py)
- Phase measurement models (phase_model.py)
- Accelerometer and GNSS processing (future)
"""

from .allan import (
    allan_deviation,
    overlapping_allan_deviation,
    modified_allan_deviation,
    identify_noise_type,
    compute_noise_spectrum,
    noise_floor_from_allan,
)

from .noise import (
    shot_noise_std,
    laser_frequency_noise_std,
    pointing_jitter_noise_std,
    clock_jitter_noise_std,
    acceleration_noise_std,
    total_phase_noise_std,
    generate_noise_realization,
    noise_equivalent_range,
)

from .phase_model import (
    compute_phase,
    compute_phase_rate,
    range_to_phase,
    phase_to_range,
    compute_phase_from_states,
    compute_phase_rate_from_states,
)

__all__ = [
    # Allan deviation
    'allan_deviation',
    'overlapping_allan_deviation',
    'modified_allan_deviation',
    'identify_noise_type',
    'compute_noise_spectrum',
    'noise_floor_from_allan',

    # Noise models
    'shot_noise_std',
    'laser_frequency_noise_std',
    'pointing_jitter_noise_std',
    'clock_jitter_noise_std',
    'acceleration_noise_std',
    'total_phase_noise_std',
    'generate_noise_realization',
    'noise_equivalent_range',

    # Phase measurements
    'compute_phase',
    'compute_phase_rate',
    'range_to_phase',
    'phase_to_range',
    'compute_phase_from_states',
    'compute_phase_rate_from_states',
]

__version__ = '0.1.0'
