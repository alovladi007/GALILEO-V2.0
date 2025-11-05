# Optical Bench Emulator - Quick Start Guide

## Installation (1 minute)

```bash
cd /home/claude/emulator
pip install -r requirements.txt
```

## Start the System (30 seconds)

### Option 1: Complete System (Recommended)
```bash
python start_emulator.py
```

This automatically:
- Starts WebSocket server on port 8765
- Starts dashboard HTTP server on port 8080  
- Opens dashboard in your browser

### Option 2: Components Separately
```bash
# Terminal 1
python server.py

# Terminal 2  
python dashboard_server.py

# Browser
# Navigate to http://localhost:8080/dashboard.html
```

## Test the System (2 minutes)

### 1. View Live Dashboard
Open: `http://localhost:8080/dashboard.html`

You should see:
- ✅ Green "Connected" status
- 📊 Real-time plots updating
- 📈 Live metrics changing
- 🔍 System diagnostics

### 2. Inject Test Events
Click dashboard buttons:
- "Inject Vibration Spike" - Watch vibration plot spike
- "Thermal Jump" - Observe temperature change
- "Laser Dropout" - See power fluctuations
- "Phase Step" - Notice phase discontinuity

### 3. Run Demo Scripts

**Basic Demo:**
```bash
python demo_basic.py
```
Shows: Basic data capture and system metrics

**Event Demo:**
```bash
python demo_events.py
```
Shows: All event types and system responses

**Streaming Demo:**
```bash
python demo_streaming.py
```
Shows: Continuous data streaming (press Ctrl+C to stop)

## Use the API (1 minute)

```python
from optical_bench import OpticalBenchEmulator

# Create emulator
emulator = OpticalBenchEmulator()

# Get current state
state = emulator.get_full_state()
print(f"Intensity: {state['interference']['intensity']:.3f}")
print(f"Phase: {state['interference']['phase']:.2f} rad")
print(f"Temperature: {state['thermal']['temperature']:.1f} K")

# Inject event
emulator.inject_event("vibration_spike", 10.0)

# Get diagnostics
diag = emulator.get_diagnostics()
print(f"Stability: {diag['stability_score']:.2f}")
```

## Dashboard Features

### Real-Time Plots
- **Interference** - Intensity vs time
- **Thermal** - Temperature trends  
- **Vibration** - Displacement tracking
- **Laser** - Power monitoring
- **Fringe Pattern** - Spatial interference

### Controls
- Event injection buttons
- System reset
- Diagnostics request

### Metrics
- Live signal values
- System health indicators
- Event log
- Performance stats

## Troubleshooting

**Dashboard shows "Disconnected":**
- Verify `server.py` is running
- Check WebSocket on port 8765

**Dashboard won't load:**
- Verify `dashboard_server.py` is running
- Check HTTP server on port 8080
- Navigate to full URL: http://localhost:8080/dashboard.html

**Import errors:**
- Run: `pip install -r requirements.txt`

## Next Steps

1. **Read Full Documentation**: `/docs/emulation.md`
2. **Explore API**: Check `optical_bench.py`
3. **Customize**: Modify parameters and noise profiles
4. **Integrate**: Connect to your own applications

## Common Use Cases

### Testing Data Acquisition
```python
# Collect 1000 samples at 100 Hz
emulator = OpticalBenchEmulator()
data = []
for i in range(1000):
    data.append(emulator.get_full_state())
    time.sleep(0.01)
```

### Remote Monitoring
```javascript
// Connect via WebSocket
const ws = new WebSocket('ws://localhost:8765');
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log(data.data.interference.intensity);
};
```

### Event Sequences
```python
# Automated test sequence
emulator.inject_event("thermal_jump", 0.5)
time.sleep(30)
emulator.inject_event("vibration_spike", 5.0)
time.sleep(30)
diag = emulator.get_diagnostics()
```

## Key Files

- `optical_bench.py` - Core emulator
- `server.py` - WebSocket server
- `dashboard.html` - Web dashboard
- `demo_*.py` - Example scripts
- `/docs/emulation.md` - Full documentation

## Support

For detailed information:
- Full API reference: `/docs/emulation.md`
- WebSocket protocol: `/docs/emulation.md#websocket-protocol`
- Advanced examples: `/docs/emulation.md#advanced-usage`

---

**You're ready to emulate!** 🚀

Total setup time: ~5 minutes
