"""
Laser interferometry module for GeoSense platform.

Implements the phase measurement model for laser ranging between satellites,
including various noise sources:
- Shot noise (fundamental quantum limit)
- Laser frequency noise
- Pointing jitter
- Clock jitter
- Allan deviation calculations

The phase measurement is the fundamental observable for gravimetry missions
like GRACE, GRACE-FO, and future satellite gravimetry concepts.
"""

from interferometry.phase_model import (
    compute_phase,
    compute_phase_rate,
    range_to_phase,
    phase_to_range,
)

from interferometry.noise import (
    shot_noise_std,
    laser_frequency_noise_std,
    pointing_jitter_noise_std,
    clock_jitter_noise_std,
    total_phase_noise_std,
    generate_noise_realization,
)

from interferometry.allan import (
    allan_deviation,
    overlapping_allan_deviation,
    compute_noise_spectrum,
)

__all__ = [
    # Phase model
    "compute_phase",
    "compute_phase_rate",
    "range_to_phase",
    "phase_to_range",
    # Noise models
    "shot_noise_std",
    "laser_frequency_noise_std",
    "pointing_jitter_noise_std",
    "clock_jitter_noise_std",
    "total_phase_noise_std",
    "generate_noise_realization",
    # Allan deviation
    "allan_deviation",
    "overlapping_allan_deviation",
    "compute_noise_spectrum",
]
