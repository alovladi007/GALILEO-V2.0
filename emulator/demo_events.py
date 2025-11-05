#!/usr/bin/env python3
"""
Demo Script 2: Event Injection
Demonstrates injecting synthetic disturbances and observing system response
"""

import sys
import time
import numpy as np
from optical_bench import OpticalBenchEmulator

def print_header(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")

def print_state_summary(state):
    """Print a concise summary of emulator state"""
    print(f"  t={state['timestamp']:6.2f}s | ", end="")
    print(f"I={state['interference']['intensity']:.3f} | ", end="")
    print(f"φ={state['interference']['phase']:.2f} rad | ", end="")
    print(f"T={state['thermal']['temperature']:.2f} K | ", end="")
    print(f"P={state['laser']['power_mw']:.3f} mW | ", end="")
    print(f"V={state['vibration']['vibration_displacement']:.2f} nm")

def demo_vibration_spike():
    """Demonstrate vibration spike injection"""
    print_header("SCENARIO 1: Vibration Spike")
    
    emulator = OpticalBenchEmulator()
    
    print("Establishing baseline...")
    for i in range(5):
        state = emulator.get_full_state()
        print_state_summary(state)
        time.sleep(0.1)
    
    print("\n⚡ INJECTING VIBRATION SPIKE (magnitude: 10x)")
    emulator.inject_event("vibration_spike", 10.0)
    
    print("\nObserving response...")
    for i in range(10):
        state = emulator.get_full_state()
        print_state_summary(state)
        time.sleep(0.1)
    
    print("\n✓ Vibration event completed\n")

def demo_thermal_jump():
    """Demonstrate thermal disturbance"""
    print_header("SCENARIO 2: Thermal Jump")
    
    emulator = OpticalBenchEmulator()
    
    print("Baseline thermal state...")
    for i in range(5):
        state = emulator.get_full_state()
        print_state_summary(state)
        time.sleep(0.1)
    
    print("\n🌡️  INJECTING THERMAL JUMP (+0.5 K)")
    emulator.inject_event("thermal_jump", 0.5)
    
    print("\nObserving thermal response...")
    for i in range(10):
        state = emulator.get_full_state()
        print_state_summary(state)
        time.sleep(0.1)
    
    print("\n✓ Thermal event completed\n")

def demo_laser_dropout():
    """Demonstrate laser intensity dropout"""
    print_header("SCENARIO 3: Laser Intensity Dropout")
    
    emulator = OpticalBenchEmulator()
    
    print("Normal laser operation...")
    for i in range(5):
        state = emulator.get_full_state()
        print_state_summary(state)
        time.sleep(0.1)
    
    print("\n💡 INJECTING LASER DROPOUT (3x noise)")
    emulator.inject_event("laser_dropout", 3.0)
    
    print("\nObserving laser fluctuations...")
    for i in range(10):
        state = emulator.get_full_state()
        print_state_summary(state)
        time.sleep(0.1)
    
    print("\n✓ Laser dropout event completed\n")

def demo_phase_step():
    """Demonstrate phase step injection"""
    print_header("SCENARIO 4: Phase Step")
    
    emulator = OpticalBenchEmulator()
    
    print("Tracking phase...")
    phases_before = []
    for i in range(5):
        state = emulator.get_full_state()
        phases_before.append(state['interference']['phase'])
        print_state_summary(state)
        time.sleep(0.1)
    
    print("\n📐 INJECTING PHASE STEP (100 mrad)")
    emulator.inject_event("phase_step", 100)
    
    print("\nObserving phase shift...")
    phases_after = []
    for i in range(10):
        state = emulator.get_full_state()
        phases_after.append(state['interference']['phase'])
        print_state_summary(state)
        time.sleep(0.1)
    
    avg_before = np.mean(phases_before)
    avg_after = np.mean(phases_after)
    print(f"\nPhase change: {avg_after - avg_before:.3f} rad")
    print("✓ Phase step event completed\n")

def demo_combined_events():
    """Demonstrate multiple simultaneous events"""
    print_header("SCENARIO 5: Combined Events")
    
    emulator = OpticalBenchEmulator()
    
    print("Stable operation...")
    for i in range(5):
        state = emulator.get_full_state()
        print_state_summary(state)
        time.sleep(0.1)
    
    print("\n⚠️  INJECTING COMBINED DISTURBANCES:")
    print("   • Vibration spike (5x)")
    print("   • Thermal jump (+0.3 K)")
    print("   • Laser dropout (2x)")
    
    emulator.inject_event("vibration_spike", 5.0)
    emulator.inject_event("thermal_jump", 0.3)
    emulator.inject_event("laser_dropout", 2.0)
    
    print("\nObserving system under stress...")
    for i in range(15):
        state = emulator.get_full_state()
        print_state_summary(state)
        time.sleep(0.1)
    
    diag = emulator.get_diagnostics()
    print(f"\nSystem health after events:")
    print(f"  Stability score: {diag['stability_score']:.3f}")
    print(f"  Alignment quality: {diag['alignment_quality']:.3f}")
    print(f"  Data quality: {diag['data_quality']}")
    
    print("\n✓ Combined events scenario completed\n")

def main():
    """Run all event injection demonstrations"""
    print("\n" + "█" * 60)
    print("  OPTICAL BENCH EMULATOR - EVENT INJECTION DEMOS")
    print("█" * 60)
    
    try:
        demo_vibration_spike()
        time.sleep(1)
        
        demo_thermal_jump()
        time.sleep(1)
        
        demo_laser_dropout()
        time.sleep(1)
        
        demo_phase_step()
        time.sleep(1)
        
        demo_combined_events()
        
        print("\n" + "█" * 60)
        print("  ✓ ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("█" * 60 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demos interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
