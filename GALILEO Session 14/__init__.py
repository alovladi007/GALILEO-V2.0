"""
Optical Bench Emulator Package
Short-baseline interferometer emulation for laboratory testing and demonstrations
"""

from .optical_bench import (
    OpticalBenchEmulator,
    BenchParameters,
    NoiseProfile,
    SignalType
)

__version__ = "1.0.0"
__author__ = "Optical Systems Lab"

__all__ = [
    "OpticalBenchEmulator",
    "BenchParameters",
    "NoiseProfile",
    "SignalType",
]
