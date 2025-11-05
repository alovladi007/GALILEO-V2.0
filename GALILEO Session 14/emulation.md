# Optical Bench Emulation System Documentation

## Overview

The Optical Bench Emulator is a comprehensive software system that simulates a short-baseline optical interferometer with realistic signal characteristics, noise profiles, and environmental effects. This emulator is designed for testing, demonstration, and development of data acquisition and analysis systems without requiring physical hardware.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Quick Start](#quick-start)
3. [Installation](#installation)
4. [Components](#components)
5. [Signal Types](#signal-types)
6. [Configuration](#configuration)
7. [API Reference](#api-reference)
8. [Demo Scripts](#demo-scripts)
9. [WebSocket Protocol](#websocket-protocol)
10. [Dashboard](#dashboard)
11. [Advanced Usage](#advanced-usage)
12. [Troubleshooting](#troubleshooting)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Emulator Architecture                   │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌────────────────────┐      ┌──────────────────────┐  │
│  │  Optical Bench     │      │   WebSocket Server   │  │
│  │  Emulator Core     │◄────►│   (Port 8765)        │  │
│  │  - Signal Gen      │      │   - Real-time stream │  │
│  │  - Noise Models    │      │   - Event injection  │  │
│  │  - State Tracking  │      │   - Commands         │  │
│  └────────────────────┘      └──────────────────────┘  │
│           ▲                            │                 │
│           │                            │                 │
│           │                            ▼                 │
│      Direct API              ┌──────────────────────┐  │
│      Access                  │   HTTP Server        │  │
│                              │   (Port 8080)        │  │
│                              │   - Serves Dashboard │  │
│                              └──────────────────────┘  │
│                                       │                 │
│                                       ▼                 │
│                              ┌──────────────────────┐  │
│                              │   Web Dashboard      │  │
│                              │   - Real-time plots  │  │
│                              │   - Controls         │  │
│                              │   - Diagnostics      │  │
│                              └──────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Key Components

1. **Optical Bench Emulator Core** (`optical_bench.py`)
   - Generates synthetic optical signals
   - Simulates environmental effects
   - Maintains system state

2. **WebSocket Server** (`server.py`)
   - Streams real-time data to clients
   - Handles remote control commands
   - Manages multiple client connections

3. **Dashboard** (`dashboard.html`)
   - Real-time visualization
   - Interactive controls
   - System diagnostics display

4. **Demo Scripts**
   - `demo_basic.py` - Basic operation demonstration
   - `demo_events.py` - Event injection scenarios
   - `demo_streaming.py` - Data streaming examples

---

## Quick Start

### 1. Start the Complete System

The fastest way to get started:

```bash
python start_emulator.py
```

This starts both the WebSocket server and dashboard HTTP server. The dashboard will open automatically in your browser at `http://localhost:8080/dashboard.html`.

### 2. Manual Start (Alternative)

Start components individually:

```bash
# Terminal 1: Start WebSocket server
python server.py

# Terminal 2: Start dashboard server
python dashboard_server.py

# Terminal 3: Access dashboard
# Open http://localhost:8080/dashboard.html in browser
```

### 3. Run Demo Scripts

```bash
# Basic operation demo
python demo_basic.py

# Event injection demo
python demo_events.py

# Streaming demo
python demo_streaming.py
```

---

## Installation

### Requirements

- Python 3.8 or higher
- numpy >= 1.24.0
- websockets >= 11.0

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import numpy, websockets; print('Dependencies OK')"

# Make scripts executable (Linux/macOS)
chmod +x demo_*.py server.py dashboard_server.py start_emulator.py
```

---

## Components

### Optical Bench Emulator Core

The core emulator (`optical_bench.py`) simulates a He-Ne laser (632.8 nm) interferometer with a 1-meter baseline.

#### Physical Parameters

```python
from optical_bench import BenchParameters

params = BenchParameters(
    baseline_length=1.0,      # meters
    wavelength=632.8e-9,      # meters (He-Ne laser)
    sampling_rate=1000.0,     # Hz
    temperature=293.15,       # Kelvin (20°C)
    pressure=101325.0,        # Pascal (1 atm)
    humidity=0.45             # relative humidity
)
```

#### Noise Profile

```python
from optical_bench import NoiseProfile

noise = NoiseProfile(
    shot_noise_level=0.01,           # intensity noise
    thermal_noise_level=0.005,       # thermal fluctuation
    vibration_amplitude=1e-9,        # meters
    vibration_frequency=50.0,        # Hz (mains frequency)
    phase_stability=0.1,             # radians RMS
    intensity_fluctuation=0.02       # relative
)
```

### Basic Usage Example

```python
from optical_bench import OpticalBenchEmulator

# Create emulator instance
emulator = OpticalBenchEmulator()

# Get current state
state = emulator.get_full_state()

print(f"Intensity: {state['interference']['intensity']:.4f}")
print(f"Phase: {state['interference']['phase']:.3f} rad")
print(f"Temperature: {state['thermal']['temperature']:.2f} K")
```

---

## Signal Types

### 1. Interference Signal

Simulates the classic two-beam interference pattern:

**Formula**: `I = I₀ × (1 + V × cos(φ))`

Where:
- I₀ = average intensity
- V = visibility (fringe contrast)
- φ = optical phase difference

**Output Parameters**:
- `intensity`: Normalized interference intensity (0-1)
- `phase`: Current phase (radians)
- `optical_path_diff`: Optical path difference (nm)
- `visibility`: Fringe visibility (0-1)

### 2. Thermal Effects

Simulates thermal drift and expansion:

**Output Parameters**:
- `temperature`: System temperature (K)
- `thermal_drift`: Accumulated drift (nm)
- `thermal_expansion`: Length change due to thermal expansion (nm)

**Characteristics**:
- Slow random walk drift
- Periodic temperature variations (5-minute cycle)
- Thermal expansion coefficient: 23×10⁻⁶ K⁻¹ (aluminum)

### 3. Vibration Signal

Multi-mode vibration simulation:

**Output Parameters**:
- `vibration_displacement`: Total displacement (nm)
- `rms_vibration`: RMS vibration level (nm)

**Vibration Sources**:
- 50 Hz: Electrical mains
- 120 Hz: Mechanical pumps
- 17 Hz: Building resonance
- Random: Broadband ambient vibration

### 4. Laser Source

Laser intensity characteristics:

**Output Parameters**:
- `intensity`: Relative intensity
- `power_mw`: Optical power (mW)
- `beam_quality_m2`: Beam quality factor M²
- `intensity_stability`: Stability metric

**Features**:
- Slow drift (10-minute cycle)
- Mode hopping events (rare)
- High-frequency noise

### 5. Fringe Pattern

2D spatial fringe pattern:

**Output Parameters**:
- `pattern`: Array of intensity values (50 points)
- `positions`: Spatial positions (radians)
- `visibility`: Pattern visibility
- `fringe_spacing`: Spatial frequency (μm)

---

## Configuration

### Custom Emulator Configuration

```python
from optical_bench import OpticalBenchEmulator, BenchParameters, NoiseProfile

# Define custom parameters
params = BenchParameters(
    baseline_length=2.0,        # 2-meter baseline
    wavelength=1064e-9,         # 1064 nm (Nd:YAG laser)
    sampling_rate=2000.0,       # 2 kHz sampling
    temperature=298.15          # 25°C
)

# Define custom noise profile
noise = NoiseProfile(
    shot_noise_level=0.005,     # Lower shot noise
    vibration_amplitude=5e-10,  # Better vibration isolation
    phase_stability=0.05        # Better phase stability
)

# Create emulator with custom settings
emulator = OpticalBenchEmulator(params=params, noise=noise)
```

### Runtime Parameter Adjustment

```python
# Modify parameters during operation
emulator.params.temperature = 295.15  # Change temperature
emulator.noise.vibration_amplitude *= 2  # Double vibration

# Reset to defaults
emulator.reset()
```

---

## API Reference

### OpticalBenchEmulator Class

#### Methods

##### `__init__(params=None, noise=None)`

Initialize emulator with optional custom parameters.

**Parameters**:
- `params` (BenchParameters, optional): Physical parameters
- `noise` (NoiseProfile, optional): Noise characteristics

##### `get_full_state(t=None) -> Dict`

Get complete emulator state at specified time.

**Parameters**:
- `t` (float, optional): Time in seconds (uses current time if None)

**Returns**: Dictionary containing:
- `timestamp`: Current simulation time
- `interference`: Interference signal data
- `thermal`: Thermal monitoring data
- `vibration`: Vibration data
- `laser`: Laser source data
- `fringes`: Fringe pattern data
- `parameters`: Current system parameters

##### `inject_event(event_type: str, magnitude: float = 1.0)`

Inject synthetic disturbance events.

**Event Types**:
- `"vibration_spike"`: Increase vibration amplitude
- `"thermal_jump"`: Step change in temperature
- `"laser_dropout"`: Increase laser intensity noise
- `"phase_step"`: Phase discontinuity

**Example**:
```python
# Inject 10x vibration spike
emulator.inject_event("vibration_spike", 10.0)

# Inject +0.5K temperature jump
emulator.inject_event("thermal_jump", 0.5)
```

##### `get_diagnostics() -> Dict`

Get system health metrics.

**Returns**:
- `uptime_seconds`: System runtime
- `stability_score`: Overall stability (0-1)
- `alignment_quality`: Optical alignment metric (0-1)
- `thermal_stable`: Boolean thermal stability flag
- `laser_locked`: Boolean laser lock status
- `fringe_contrast`: Fringe visibility
- `data_quality`: Qualitative assessment ("good"/"marginal"/"poor")

##### `reset()`

Reset emulator to initial state.

---

## Demo Scripts

### demo_basic.py

**Purpose**: Demonstrate basic emulator operation and data capture.

**What it does**:
- Initializes emulator with default parameters
- Captures 10 sequential samples
- Displays key metrics
- Shows system diagnostics

**Run**:
```bash
python demo_basic.py
```

**Output Example**:
```
Sample 1:
  Timestamp: 0.100 s
  Interference Intensity: 0.5234
  Phase: 2.145 rad
  OPD: 125.32 nm
  Temperature: 293.15 K
  Laser Power: 5.023 mW
...
```

### demo_events.py

**Purpose**: Demonstrate event injection and system response.

**Scenarios**:
1. Vibration Spike - 10x vibration increase
2. Thermal Jump - +0.5K temperature step
3. Laser Dropout - 3x intensity noise
4. Phase Step - 100 mrad phase discontinuity
5. Combined Events - Multiple simultaneous disturbances

**Run**:
```bash
python demo_events.py
```

**Key Features**:
- Shows baseline behavior
- Injects controlled disturbances
- Monitors system response
- Evaluates recovery

### demo_streaming.py

**Purpose**: Demonstrate continuous data streaming capabilities.

**Modes**:
1. **Continuous Streaming**: Real-time 20 Hz data stream
2. **High-Rate Capture**: 1 kHz burst capture
3. **Burst Mode**: Periodic high-speed bursts

**Run**:
```bash
python demo_streaming.py
```

**Features**:
- Real-time statistics
- Data quality metrics
- Buffer management
- Rate monitoring

---

## WebSocket Protocol

### Connection

Connect to: `ws://localhost:8765`

### Message Types

#### From Server → Client

##### State Update
```json
{
  "type": "state_update",
  "data": {
    "timestamp": 123.45,
    "interference": {
      "intensity": 0.5234,
      "phase": 2.145,
      "optical_path_diff": 125.32,
      "visibility": 0.95
    },
    "thermal": {...},
    "vibration": {...},
    "laser": {...},
    "fringes": {...},
    "parameters": {...}
  }
}
```

##### Diagnostics
```json
{
  "type": "diagnostics",
  "data": {
    "uptime_seconds": 123.45,
    "stability_score": 0.85,
    "alignment_quality": 0.92,
    "data_quality": "good"
  }
}
```

##### Event Notification
```json
{
  "type": "event",
  "event_type": "event_injected",
  "data": {
    "event_type": "vibration_spike",
    "magnitude": 10.0
  },
  "timestamp": "2024-01-01T12:00:00"
}
```

#### From Client → Server

##### Inject Event
```json
{
  "command": "inject_event",
  "event_type": "vibration_spike",
  "magnitude": 10.0
}
```

##### Reset System
```json
{
  "command": "reset"
}
```

##### Set Update Rate
```json
{
  "command": "set_update_rate",
  "rate": 100
}
```

##### Set Parameters
```json
{
  "command": "set_parameters",
  "parameters": {
    "baseline_length": 2.0,
    "temperature": 295.0
  }
}
```

##### Request Diagnostics
```json
{
  "command": "get_diagnostics"
}
```

### JavaScript Client Example

```javascript
// Connect to emulator
const ws = new WebSocket('ws://localhost:8765');

// Handle connection
ws.onopen = () => {
    console.log('Connected to emulator');
};

// Handle incoming messages
ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    
    if (message.type === 'state_update') {
        const intensity = message.data.interference.intensity;
        console.log(`Intensity: ${intensity}`);
    }
};

// Send command
ws.send(JSON.stringify({
    command: 'inject_event',
    event_type: 'vibration_spike',
    magnitude: 5.0
}));
```

---

## Dashboard

### Features

The web dashboard provides real-time visualization and control:

#### Real-Time Plots
- **Interference Signal**: Intensity vs. time
- **Thermal Monitor**: Temperature trends
- **Vibration Monitor**: Displacement tracking
- **Laser Power**: Power stability
- **Fringe Pattern**: Spatial interference pattern

#### Control Panel
- Inject vibration spike
- Trigger thermal jump
- Simulate laser dropout
- Create phase step
- Reset emulator
- Request diagnostics

#### Metrics Display
- Live intensity, phase, OPD values
- Temperature monitoring
- Vibration RMS levels
- Laser characteristics
- System diagnostics
- Event log

#### System Health
- Stability score indicator
- Alignment quality
- Fringe contrast
- Data quality assessment
- Color-coded health indicators

### Accessing the Dashboard

1. Start the system:
   ```bash
   python start_emulator.py
   ```

2. Open browser to:
   ```
   http://localhost:8080/dashboard.html
   ```

3. Verify connection status (should show "Connected" in green)

### Dashboard Controls

#### Event Injection Buttons
- **Vibration Spike**: Simulates sudden mechanical disturbance
- **Thermal Jump**: Simulates environmental temperature change
- **Laser Dropout**: Simulates laser instability
- **Phase Step**: Simulates mirror adjustment

#### System Buttons
- **Reset Emulator**: Return to initial state
- **Get Diagnostics**: Request full system diagnostics

---

## Advanced Usage

### Custom Signal Processing

```python
import numpy as np
from optical_bench import OpticalBenchEmulator

emulator = OpticalBenchEmulator()

# Collect time series data
duration = 10.0  # seconds
dt = 0.01  # 100 Hz
samples = int(duration / dt)

intensities = []
phases = []

for i in range(samples):
    state = emulator.get_full_state()
    intensities.append(state['interference']['intensity'])
    phases.append(state['interference']['phase'])
    time.sleep(dt)

# Convert to numpy arrays
intensities = np.array(intensities)
phases = np.array(phases)

# Perform FFT analysis
fft = np.fft.fft(intensities)
freqs = np.fft.fftfreq(len(intensities), dt)

# Find dominant frequency
dominant_freq = freqs[np.argmax(np.abs(fft[1:])) + 1]
print(f"Dominant frequency: {dominant_freq:.2f} Hz")
```

### Data Logging

```python
import json
import time
from datetime import datetime
from optical_bench import OpticalBenchEmulator

emulator = OpticalBenchEmulator()

# Log data to file
log_file = f"emulator_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

with open(log_file, 'w') as f:
    for i in range(1000):
        state = emulator.get_full_state()
        json.dump(state, f)
        f.write('\n')
        time.sleep(0.01)

print(f"Logged 1000 samples to {log_file}")
```

### Custom Event Sequences

```python
import time
from optical_bench import OpticalBenchEmulator

def run_experiment_sequence(emulator):
    """Run a predefined experimental sequence"""
    
    # Phase 1: Baseline (30 seconds)
    print("Phase 1: Establishing baseline...")
    time.sleep(30)
    
    # Phase 2: Thermal perturbation
    print("Phase 2: Thermal perturbation...")
    emulator.inject_event("thermal_jump", 1.0)
    time.sleep(60)
    
    # Phase 3: Vibration test
    print("Phase 3: Vibration test...")
    emulator.inject_event("vibration_spike", 5.0)
    time.sleep(30)
    
    # Phase 4: Recovery
    print("Phase 4: Recovery period...")
    time.sleep(60)
    
    # Get final diagnostics
    diag = emulator.get_diagnostics()
    return diag

emulator = OpticalBenchEmulator()
results = run_experiment_sequence(emulator)
print(f"Experiment completed. Final stability: {results['stability_score']:.3f}")
```

### Integration with External Systems

```python
import asyncio
import websockets
import json

async def external_monitor():
    """Connect as external monitoring system"""
    uri = "ws://localhost:8765"
    
    async with websockets.connect(uri) as websocket:
        print("Connected as external monitor")
        
        async for message in websocket:
            data = json.loads(message)
            
            if data['type'] == 'state_update':
                intensity = data['data']['interference']['intensity']
                
                # Trigger alert on anomaly
                if intensity < 0.2 or intensity > 0.8:
                    print(f"⚠️  ANOMALY DETECTED: Intensity = {intensity:.3f}")

# Run external monitor
asyncio.run(external_monitor())
```

---

## Troubleshooting

### Common Issues

#### 1. "Module not found" errors

**Solution**:
```bash
pip install -r requirements.txt
```

#### 2. WebSocket connection fails

**Symptoms**: Dashboard shows "Disconnected"

**Solutions**:
- Verify server is running: `python server.py`
- Check port 8765 is not in use: `lsof -i :8765` (Linux/macOS)
- Check firewall settings
- Try localhost vs. 127.0.0.1

#### 3. Dashboard not loading

**Symptoms**: Blank page or 404 error

**Solutions**:
- Verify dashboard server is running: `python dashboard_server.py`
- Check port 8080 is available
- Navigate to exact URL: `http://localhost:8080/dashboard.html`
- Check browser console for errors (F12)

#### 4. High CPU usage

**Cause**: Update rate too high

**Solution**:
```python
# Reduce update rate via WebSocket
ws.send(JSON.stringify({
    command: 'set_update_rate',
    rate: 20  # 20 Hz instead of default 50 Hz
}))
```

#### 5. Data appears frozen

**Solutions**:
- Check WebSocket connection status
- Verify emulator server is running
- Check for JavaScript errors in browser console
- Try refreshing the dashboard

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from optical_bench import OpticalBenchEmulator
emulator = OpticalBenchEmulator()
```

### Performance Optimization

For high-rate applications:

```python
# Use numpy operations directly
import numpy as np

# Batch generate data
times = np.arange(0, 10, 0.001)  # 10 seconds at 1 kHz
states = [emulator.get_full_state(t) for t in times]
```

---

## System Requirements

### Minimum Requirements
- **CPU**: 1 GHz dual-core processor
- **RAM**: 512 MB available
- **Disk**: 50 MB free space
- **OS**: Windows 10, macOS 10.14, Ubuntu 18.04 or newer
- **Python**: 3.8 or higher
- **Browser**: Chrome 90+, Firefox 88+, Safari 14+, Edge 90+

### Recommended Requirements
- **CPU**: 2 GHz quad-core processor
- **RAM**: 2 GB available
- **Network**: Low-latency local connection

---

## Performance Characteristics

### Timing Performance

- **Update Rate**: Up to 1000 Hz (1 ms sampling)
- **WebSocket Latency**: Typically < 5 ms on localhost
- **Dashboard Refresh**: 20-50 Hz (20-50 ms)
- **Event Response**: < 100 ms

### Resource Usage

- **CPU**: ~5-15% per core at 50 Hz update rate
- **Memory**: ~50-100 MB per emulator instance
- **Network**: ~100 KB/s at 50 Hz (WebSocket)

---

## License and Credits

Developed for laboratory emulation and testing purposes.

For questions, issues, or contributions, please contact the development team.

---

## Appendix: Signal Equations

### Interference Intensity

```
I(t) = I₀ × [1 + V × cos(k × Δℓ(t) + φ_noise(t))]
```

Where:
- I₀ = 0.5 (normalized)
- V = 0.95 (visibility)
- k = 2π/λ (wave number)
- Δℓ(t) = baseline × sin(ω_scan × t) (optical path difference)
- φ_noise ~ N(0, σ_φ²) (phase noise)

### Thermal Expansion

```
Δℓ_thermal = α × ℓ₀ × ΔT
```

Where:
- α = 23×10⁻⁶ K⁻¹ (aluminum)
- ℓ₀ = baseline length
- ΔT = temperature change

### Vibration Model

```
d(t) = Σᵢ Aᵢ × sin(2π × fᵢ × t + φᵢ) + n(t)
```

Where:
- Aᵢ = amplitude of mode i
- fᵢ = frequency of mode i (50, 120, 17 Hz)
- n(t) = white noise

---

**End of Documentation**

Version: 1.0.0  
Last Updated: 2024
