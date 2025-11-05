# Synthetic Interferometric Data Generator Documentation

## Overview

The Synthetic Data Generator creates realistic interferometric time-series data for testing and validating satellite-based subsurface anomaly detection algorithms. It implements a complete forward model from subsurface density variations through gravitational effects to interferometric phase measurements with realistic noise characteristics.

## Table of Contents

1. [Architecture](#architecture)
2. [Physical Model](#physical-model)
3. [Data Generation Pipeline](#data-generation-pipeline)
4. [Anomaly Types](#anomaly-types)
5. [Output Formats](#output-formats)
6. [Usage Examples](#usage-examples)
7. [Validation & Testing](#validation--testing)
8. [Performance Characteristics](#performance-characteristics)

---

## Architecture

The synthetic data generator consists of several modular components:

```
SyntheticDataGenerator
├── SubsurfaceModel         # Procedural anomaly generation
├── ForwardModel            # Physics simulation
│   ├── density_to_gravity()
│   ├── gravity_to_baseline()
│   ├── baseline_to_phase()
│   └── add_noise()
├── TelemetryGenerator      # Realistic telemetry simulation
└── STACMetadataGenerator   # STAC-compliant metadata
```

### Key Classes

- **`SimulationConfig`**: Defines grid parameters, time steps, and noise levels
- **`SatelliteConfig`**: Satellite constellation parameters
- **`SubsurfaceModel`**: Generates procedural subsurface density fields
- **`ForwardModel`**: Implements physics-based forward modeling
- **`TelemetryGenerator`**: Creates realistic telemetry time series
- **`STACMetadataGenerator`**: Produces STAC-compliant metadata

---

## Physical Model

### Forward Model Chain

The generator implements a complete forward model:

1. **Density Anomalies (Δρ)** → Subsurface density variations
2. **Gravity Field (Δg)** → Gravitational effects at orbital altitude
3. **Baseline Dynamics** → Satellite separation modulation
4. **Interferometric Phase** → Phase measurements
5. **Telemetry + Noise** → Realistic sensor data

### Mathematical Formulation

#### Gravity Calculation
```python
g(x,y) = G ∑ᵢ (ρᵢ - ρ₀) · Vᵢ · zᵢ / r³ᵢ
```
Where:
- G = gravitational constant
- ρᵢ = density at voxel i
- Vᵢ = voxel volume
- zᵢ = depth
- rᵢ = distance to satellite

#### Phase Sensitivity
```python
φ = (2π/λ) · Δb
```
Where:
- λ = radar wavelength (C-band: 5.6 cm)
- Δb = baseline change

#### Noise Model

Three noise components are modeled:
1. **Atmospheric noise**: Spatially correlated (20 grid point correlation length)
2. **Thermal noise**: Uncorrelated white noise
3. **Ionospheric noise**: Time-varying with diurnal cycle

---

## Data Generation Pipeline

### Step 1: Initialize Configuration

```python
from sim.synthetic import SimulationConfig, SatelliteConfig, SyntheticDataGenerator

sim_config = SimulationConfig(
    grid_size=(100, 100, 50),  # (nx, ny, nz)
    grid_spacing=10.0,          # meters
    time_steps=100,             # number of time samples
    seed=42,                    # for reproducibility
    noise_level=0.1             # radians
)

sat_config = SatelliteConfig(
    orbital_height=500e3,       # 500 km
    baseline_nominal=200.0,     # meters
    baseline_variation=10.0     # 1-sigma variation
)
```

### Step 2: Generate Synthetic Data

```python
generator = SyntheticDataGenerator(sim_config, sat_config)
results = generator.generate(output_dir="./data")
```

### Step 3: Access Generated Data

```python
# Load telemetry data
import pandas as pd
telemetry = pd.read_parquet(results['telemetry_path'])

# Load phase arrays
import numpy as np
phase_data = np.load(results['phase_path'])
phases = phase_data['phases']
noisy_phases = phase_data['noisy_phases']
gravity_field = phase_data['gravity_field']
```

---

## Anomaly Types

### 1. Voids/Cavities

**Characteristics:**
- Ellipsoidal shape with surface roughness
- Density contrast: -2000 kg/m³ (air vs rock)
- Size: 20-50m typical dimensions
- Roughness parameter: 0.1-0.3

**Generation:**
```python
model.add_void(
    center=np.array([500, 500, 200]),  # meters
    size=np.array([40, 40, 20])        # meters
)
```

### 2. Tunnels

**Characteristics:**
- Cylindrical geometry
- Density contrast: -1800 kg/m³ (partially filled)
- Radius: 5m typical
- Can span across grid

**Generation:**
```python
model.add_tunnel(
    start=np.array([100, 100, 300]),
    end=np.array([800, 800, 320]),
    radius=5.0
)
```

### 3. Ore Bodies

**Characteristics:**
- Irregular ellipsoidal shape
- Density contrast: +500 to +2000 kg/m³
- Dip and strike orientation
- Size: 30-60m typical dimensions

**Generation:**
```python
model.add_ore_body(
    center=np.array([400, 400, 350]),
    size=np.array([50, 50, 25]),
    density_contrast=1200  # kg/m³
)
```

---

## Output Formats

### 1. Telemetry Data (Parquet)

| Column | Type | Description | Range |
|--------|------|-------------|-------|
| timestamp | datetime | Observation time | - |
| pixel_x | int | X coordinate | [0, nx-1] |
| pixel_y | int | Y coordinate | [0, ny-1] |
| phase | float | Interferometric phase | [-π, π] rad |
| coherence | float | Interferometric coherence | [0, 1] |
| snr | float | Signal-to-noise ratio | [10, 30] dB |
| satellite_id | str | Satellite identifier | SAT1/SAT2 |
| pass_direction | str | Orbit direction | ascending/descending |
| incidence_angle | float | Radar incidence angle | [20°, 45°] |
| quality_flag | int | Data quality indicator | 0=good, 1=warning, 2=bad |

### 2. Phase Arrays (NPZ)

Compressed NumPy archive containing:
- `phases`: Clean interferometric phases [time, x, y]
- `noisy_phases`: Phases with realistic noise [time, x, y]
- `gravity_field`: Gravity anomalies [x, y]
- `baselines`: Satellite baselines [time, x, y]
- `density_field`: Subsurface density model [x, y, z]

### 3. STAC Metadata (JSON)

**Collection Metadata:**
```json
{
  "stac_version": "1.0.0",
  "type": "Collection",
  "id": "synthetic-ifg-20240115_143022",
  "summaries": {
    "anomaly_count": 7,
    "anomaly_types": ["void", "tunnel", "ore"]
  }
}
```

**Item Metadata:**
```json
{
  "type": "Feature",
  "properties": {
    "sar:frequency_band": "C",
    "sar:center_frequency": 5.405,
    "sar:product_type": "interferogram"
  }
}
```

### 4. Dataset Card (JSON)

Comprehensive metadata including:
- Configuration parameters
- Anomaly statistics
- Data schema
- Quality metrics

---

## Usage Examples

### Basic Generation

```python
from sim.synthetic import SyntheticDataGenerator

# Quick generation with defaults
generator = SyntheticDataGenerator()
results = generator.generate()
```

### Custom Anomalies

```python
from sim.synthetic import SimulationConfig, SubsurfaceModel

config = SimulationConfig(grid_size=(150, 150, 75))
model = SubsurfaceModel(config, seed=123)

# Add multiple anomalies
for i in range(5):
    model.add_void()
    
model.add_tunnel(
    start=np.array([100, 100, 400]),
    end=np.array([1200, 1200, 420])
)

for i in range(3):
    model.add_ore_body(density_contrast=1500)
```

### Batch Generation

```python
import numpy as np
from pathlib import Path

# Generate multiple realizations
for seed in range(10):
    config = SimulationConfig(seed=seed)
    generator = SyntheticDataGenerator(config)
    
    output_dir = Path(f"./data/realization_{seed:03d}")
    results = generator.generate(str(output_dir))
    
    print(f"Generated realization {seed} with {len(results['anomalies'])} anomalies")
```

### Analysis Pipeline

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load and analyze generated data
telemetry = pd.read_parquet("data/telemetry_20240115_143022.parquet")
phase_data = np.load("data/phases_20240115_143022.npz")

# Compute phase statistics
phase_std = np.std(phase_data['noisy_phases'], axis=0)
coherence_mean = telemetry.groupby(['pixel_x', 'pixel_y'])['coherence'].mean()

# Identify anomalous regions
anomaly_mask = phase_std > np.percentile(phase_std, 95)
```

---

## Validation & Testing

### Test Coverage

The test suite (`tests/test_synthetic.py`) validates:

1. **Deterministic Output**
   - Fixed seeds produce identical results
   - Reproducibility across runs

2. **Schema Validation**
   - Telemetry DataFrame structure
   - STAC metadata compliance
   - Dataset card format

3. **Data Integrity**
   - Anomaly placement bounds
   - Density field validity
   - Phase continuity
   - Telemetry consistency

4. **Physical Realism**
   - Gravity field magnitudes
   - Baseline variations
   - Coherence distribution

### Running Tests

```bash
# Run all tests
python tests/test_synthetic.py

# Run with pytest (if installed)
pytest tests/test_synthetic.py -v

# Run specific test class
pytest tests/test_synthetic.py::TestSchemaValidation -v
```

### Test Results

Expected output:
```
TEST SUMMARY
============================================================
Tests run: 24
Failures: 0
Errors: 0
Success rate: 100.0%
```

---

## Performance Characteristics

### Computational Complexity

| Operation | Complexity | Typical Time |
|-----------|------------|--------------|
| Density field init | O(nx·ny·nz) | ~0.5s |
| Add anomaly | O(nx·ny·nz) | ~0.2s |
| Gravity calculation | O(nx·ny·nz) | ~2s |
| Phase computation | O(t·nx·ny) | ~0.5s |
| Noise generation | O(t·nx·ny) | ~1s |
| Total pipeline | - | ~5-10s |

*Times for 100×100×50 grid, 100 time steps on modern CPU*

### Memory Requirements

| Grid Size | Memory (approx) |
|-----------|-----------------|
| 50×50×25 | ~100 MB |
| 100×100×50 | ~500 MB |
| 200×200×100 | ~3 GB |

### Scaling Behavior

- **Linear** with number of time steps
- **Cubic** with grid dimension for density field
- **Quadratic** with grid dimension for phase arrays

---

## Example Visualizations

### Density Field Cross-Sections
![Density Field](docs/images/density_field.png)
*Subsurface density model showing procedural anomalies in three orthogonal planes*

### Gravity Field at Satellite Altitude
![Gravity Field](docs/images/gravity_field.png)
*Gravitational anomalies computed at 500 km altitude*

### Interferometric Phase Evolution
![Phase Time Series](docs/images/phase_timeseries.png)
*Clean and noisy phase measurements with temporal evolution*

### Satellite Baseline Dynamics
![Baseline Dynamics](docs/images/baseline_dynamics.png)
*Baseline configuration and distribution statistics*

### Telemetry Data Statistics
![Telemetry Statistics](docs/images/telemetry_statistics.png)
*Statistical distributions of telemetry parameters*

### Anomaly Distribution
![Anomaly Distribution](docs/images/anomaly_distribution.png)
*Summary of procedurally generated subsurface anomalies*

---

## Configuration Reference

### SimulationConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| grid_size | tuple | (100,100,50) | Grid dimensions (nx, ny, nz) |
| grid_spacing | float | 10.0 | Grid cell size in meters |
| time_steps | int | 100 | Number of time samples |
| time_interval | float | 1.0 | Time between samples (days) |
| seed | int/None | None | Random seed for reproducibility |
| noise_level | float | 0.1 | Base noise level (radians) |
| atmospheric_noise | float | 0.05 | Atmospheric noise (radians) |
| thermal_noise | float | 0.03 | Thermal noise (radians) |

### SatelliteConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| num_satellites | int | 2 | Number of satellites |
| orbital_height | float | 500e3 | Orbital altitude (meters) |
| baseline_nominal | float | 200.0 | Nominal baseline (meters) |
| baseline_variation | float | 10.0 | Baseline 1-σ variation (meters) |
| inclination | float | 97.4 | Orbital inclination (degrees) |
| revisit_time | int | 12 | Revisit period (days) |

---

## Troubleshooting

### Common Issues

1. **Memory Error with Large Grids**
   - Reduce grid_size or use smaller chunks
   - Process time steps in batches

2. **Determinism Failures**
   - Ensure seed is set in SimulationConfig
   - Check NumPy version compatibility

3. **Missing Dependencies**
   ```bash
   pip install numpy pandas pyarrow matplotlib seaborn
   ```

4. **STAC Validation Errors**
   - Verify datetime formats are ISO 8601
   - Check bbox coordinate ordering

### Performance Optimization

- Use smaller grids for testing (50×50×25)
- Reduce time_steps for quick iterations
- Disable noise for debugging (noise_level=0)
- Use multiprocessing for batch generation

---

## Future Enhancements

### Planned Features

1. **Additional Anomaly Types**
   - Fault systems
   - Layered strata
   - Fluid reservoirs

2. **Advanced Noise Models**
   - Tropospheric delays
   - Orbital errors
   - Multipath effects

3. **Extended Physics**
   - Elastic deformation
   - Thermal expansion
   - Poroelastic effects

4. **Data Formats**
   - HDF5 support
   - Cloud-optimized GeoTIFF
   - Zarr arrays

5. **Validation Tools**
   - Automatic quality metrics
   - Comparison with real data
   - Statistical validation suite

---

## References

1. **Interferometric SAR**
   - Ferretti, A., et al. (2007). "InSAR Principles"
   - Hanssen, R. (2001). "Radar Interferometry"

2. **Gravity Gradiometry**
   - Rummel, R., et al. (2011). "GOCE gravitational gradiometry"
   - Jekeli, C. (2006). "Airborne gradiometry error analysis"

3. **STAC Specification**
   - [SpatioTemporal Asset Catalog](https://stacspec.org/)

4. **Noise Modeling**
   - Emardson, T.R., et al. (2003). "Neutral atmospheric delay in interferometric SAR"
   - Zebker, H.A., et al. (1997). "Atmospheric effects in interferometric SAR"

---

## License

This synthetic data generator is provided under the MIT License. The generated data can be used for research and development purposes without restrictions.

---

## Contact & Support

For questions, bug reports, or feature requests, please open an issue in the project repository.

**Version:** 1.0.0  
**Last Updated:** January 2024  
**Status:** Production Ready
