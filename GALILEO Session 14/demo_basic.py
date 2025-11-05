#!/usr/bin/env python3
"""
Demo Script 1: Basic Emulator Operation
Demonstrates basic signal generation and data capture
"""

import sys
import time
from optical_bench import OpticalBenchEmulator, BenchParameters, NoiseProfile

def print_header(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")

def demo_basic_operation():
    """Demonstrate basic emulator operation"""
    print_header("DEMO 1: Basic Emulator Operation")
    
    # Initialize emulator with default parameters
    emulator = OpticalBenchEmulator()
    
    print("Initializing optical bench emulator...")
    print(f"  Baseline length: {emulator.params.baseline_length} m")
    print(f"  Wavelength: {emulator.params.wavelength * 1e9:.1f} nm")
    print(f"  Sampling rate: {emulator.params.sampling_rate} Hz")
    print(f"  Temperature: {emulator.params.temperature:.2f} K")
    
    print("\n" + "-" * 60)
    print("Capturing 10 samples...")
    print("-" * 60)
    
    for i in range(10):
        state = emulator.get_full_state()
        
        print(f"\nSample {i+1}:")
        print(f"  Timestamp: {state['timestamp']:.3f} s")
        print(f"  Interference Intensity: {state['interference']['intensity']:.4f}")
        print(f"  Phase: {state['interference']['phase']:.3f} rad")
        print(f"  OPD: {state['interference']['optical_path_diff']:.2f} nm")
        print(f"  Temperature: {state['thermal']['temperature']:.2f} K")
        print(f"  Laser Power: {state['laser']['power_mw']:.3f} mW")
        
        time.sleep(0.1)
    
    print("\n" + "-" * 60)
    print("System Diagnostics:")
    print("-" * 60)
    
    diag = emulator.get_diagnostics()
    for key, value in diag.items():
        print(f"  {key}: {value}")
    
    print("\n✓ Demo 1 completed successfully!")

if __name__ == "__main__":
    try:
        demo_basic_operation()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        sys.exit(1)
