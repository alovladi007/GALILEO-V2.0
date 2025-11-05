#!/usr/bin/env python3
"""
Demo Script 3: Real-Time Streaming
Demonstrates continuous data streaming for visualization
"""

import sys
import time
import signal
from optical_bench import OpticalBenchEmulator

# Global flag for graceful shutdown
running = True

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global running
    print("\n\n⚠️  Stopping streaming...")
    running = False

def print_header(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")

def demo_continuous_streaming():
    """Stream emulator data continuously"""
    print_header("Real-Time Data Streaming")
    
    emulator = OpticalBenchEmulator()
    
    print("Streaming optical bench data...")
    print("Press Ctrl+C to stop\n")
    print("-" * 60)
    
    # Statistics tracking
    sample_count = 0
    start_time = time.time()
    
    # Print header
    print(f"{'Time':>8} | {'Intensity':>10} | {'Phase':>8} | {'Temp':>8} | {'Laser':>8} | {'Vib':>8}")
    print(f"{'(s)':>8} | {'':>10} | {'(rad)':>8} | {'(K)':>8} | {'(mW)':>8} | {'(nm)':>8}")
    print("-" * 60)
    
    try:
        while running:
            state = emulator.get_full_state()
            
            print(f"{state['timestamp']:8.2f} | "
                  f"{state['interference']['intensity']:10.4f} | "
                  f"{state['interference']['phase']:8.3f} | "
                  f"{state['thermal']['temperature']:8.2f} | "
                  f"{state['laser']['power_mw']:8.3f} | "
                  f"{state['vibration']['vibration_displacement']:8.2f}")
            
            sample_count += 1
            
            # Print stats every 50 samples
            if sample_count % 50 == 0:
                elapsed = time.time() - start_time
                rate = sample_count / elapsed
                print(f"\n  📊 Stats: {sample_count} samples | {rate:.1f} samples/s | {elapsed:.1f}s elapsed\n")
                print(f"{'Time':>8} | {'Intensity':>10} | {'Phase':>8} | {'Temp':>8} | {'Laser':>8} | {'Vib':>8}")
                print(f"{'(s)':>8} | {'':>10} | {'(rad)':>8} | {'(K)':>8} | {'(mW)':>8} | {'(nm)':>8}")
                print("-" * 60)
            
            time.sleep(0.05)  # 20 Hz update rate
    
    except KeyboardInterrupt:
        pass
    
    # Final statistics
    elapsed = time.time() - start_time
    rate = sample_count / elapsed
    
    print("\n" + "-" * 60)
    print(f"Streaming stopped")
    print(f"  Total samples: {sample_count}")
    print(f"  Duration: {elapsed:.2f} s")
    print(f"  Average rate: {rate:.2f} samples/s")
    print(f"  Uptime: {emulator.get_timestamp():.2f} s")
    print("-" * 60)
    
    # Show final diagnostics
    diag = emulator.get_diagnostics()
    print("\nFinal System Diagnostics:")
    for key, value in diag.items():
        print(f"  {key}: {value}")
    
    print("\n✓ Streaming demo completed!")

def demo_high_rate_capture():
    """Demonstrate high-rate data capture"""
    print_header("High-Rate Data Capture")
    
    emulator = OpticalBenchEmulator()
    
    duration = 5.0  # seconds
    samples_per_second = 1000
    
    print(f"Capturing data at {samples_per_second} Hz for {duration} seconds...")
    print(f"Target samples: {int(duration * samples_per_second)}\n")
    
    data_buffer = []
    start_time = time.time()
    sample_count = 0
    
    while time.time() - start_time < duration:
        state = emulator.get_full_state()
        data_buffer.append(state)
        sample_count += 1
        time.sleep(1.0 / samples_per_second)
        
        # Progress indicator
        if sample_count % 1000 == 0:
            print(f"  Captured {sample_count} samples...")
    
    actual_duration = time.time() - start_time
    actual_rate = sample_count / actual_duration
    
    print(f"\n✓ Capture complete!")
    print(f"  Samples captured: {sample_count}")
    print(f"  Actual duration: {actual_duration:.3f} s")
    print(f"  Actual rate: {actual_rate:.1f} Hz")
    print(f"  Buffer size: {len(data_buffer)} entries")
    
    # Analyze captured data
    intensities = [s['interference']['intensity'] for s in data_buffer]
    phases = [s['interference']['phase'] for s in data_buffer]
    
    import numpy as np
    print(f"\nData Statistics:")
    print(f"  Intensity: mean={np.mean(intensities):.4f}, std={np.std(intensities):.4f}")
    print(f"  Phase: mean={np.mean(phases):.4f}, std={np.std(phases):.4f}")

def demo_burst_mode():
    """Demonstrate burst capture mode"""
    print_header("Burst Mode Capture")
    
    emulator = OpticalBenchEmulator()
    
    bursts = 5
    samples_per_burst = 100
    burst_interval = 2.0  # seconds
    
    print(f"Configuration:")
    print(f"  Number of bursts: {bursts}")
    print(f"  Samples per burst: {samples_per_burst}")
    print(f"  Interval between bursts: {burst_interval} s\n")
    
    for burst_num in range(bursts):
        print(f"Burst {burst_num + 1}/{bursts}...")
        
        burst_data = []
        for i in range(samples_per_burst):
            state = emulator.get_full_state()
            burst_data.append(state)
            time.sleep(0.01)  # 100 Hz within burst
        
        # Quick analysis
        avg_intensity = sum(s['interference']['intensity'] for s in burst_data) / len(burst_data)
        print(f"  ✓ Captured {len(burst_data)} samples | Avg intensity: {avg_intensity:.4f}")
        
        if burst_num < bursts - 1:
            print(f"  Waiting {burst_interval}s until next burst...")
            time.sleep(burst_interval)
    
    print(f"\n✓ All bursts completed!")

def main():
    """Run streaming demonstrations"""
    global running
    
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    print("\n" + "█" * 60)
    print("  OPTICAL BENCH EMULATOR - STREAMING DEMOS")
    print("█" * 60)
    
    try:
        print("\n[1/3] Continuous Streaming Demo")
        demo_continuous_streaming()
        
        if running:
            time.sleep(2)
            print("\n[2/3] High-Rate Capture Demo")
            demo_high_rate_capture()
        
        if running:
            time.sleep(2)
            print("\n[3/3] Burst Mode Demo")
            demo_burst_mode()
        
        if running:
            print("\n" + "█" * 60)
            print("  ✓ ALL STREAMING DEMOS COMPLETED!")
            print("█" * 60 + "\n")
        
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
