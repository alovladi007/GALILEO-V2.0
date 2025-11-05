# Earth Models Documentation

## Overview

The geophysics module provides comprehensive tools for working with Earth reference models in geophysical data processing. This includes gravity field models, crustal density structures, hydrological corrections, and ocean/land masking.

## Module Structure

```
geophysics/
├── __init__.py           # Main module interface
├── gravity_fields.py     # Reference gravity models (EGM96, EGM2008)
├── crustal_models.py     # Crustal density and terrain corrections
├── hydrology.py          # Seasonal water and groundwater models
├── masking.py           # Ocean/land masks and region definitions
└── joint_inversion.py   # Joint inversion with Session 5 integration
```

## Quick Start

### Basic Gravity Field Usage

```python
from geophysics import load_egm96, compute_gravity_anomaly
import numpy as np

# Load reference gravity model
egm96 = load_egm96()

# Define observation points
lat = np.array([40.0, 41.0, 42.0])
lon = np.array([-75.0, -74.0, -73.0])
observed_g = np.array([980200, 980150, 980180])  # mGal

# Compute gravity anomaly
anomaly, components = compute_gravity_anomaly(
    lat, lon, observed_g, egm96,
    correction_type='free_air'
)

print(f"Free-air anomaly: {anomaly} mGal")
```

### Crustal Density Models

```python
from geophysics import load_crust1, terrain_correction

# Load global crustal model
crust = load_crust1()

# Get density at specific locations
density = crust.get_density_at_depth(lat, lon, depth=5000)  # 5 km depth

# Compute terrain correction
elevation = np.array([100, 200, 150])  # meters
dem = np.random.rand(100, 100) * 1000  # Placeholder DEM
dem_lat = np.linspace(39, 43, 100)
dem_lon = np.linspace(-76, -72, 100)

terrain_corr = terrain_correction(
    lat, lon, elevation, dem, dem_lat, dem_lon
)
```

### Hydrological Corrections

```python
from geophysics import load_seasonal_water, hydrological_correction
from datetime import datetime

# Load seasonal water model
hydro = load_seasonal_water(source='GLDAS')

# Compute correction for specific observation time
obs_time = datetime(2024, 6, 15)
hydro_corr = hydrological_correction(lat, lon, obs_time, hydro)

print(f"Hydrological correction: {hydro_corr} mGal")
```

### Ocean/Land Masking

```python
from geophysics import load_ocean_mask, create_region_mask

# Load global ocean/land mask
ocean_mask = load_ocean_mask(resolution=1.0)

# Check if points are on land
is_land = ocean_mask.is_land(lat, lon)

# Create custom region mask
california = create_region_mask(
    lat_range=(32.5, 42.0),
    lon_range=(-124.5, -114.1),
    region_name='california'
)
```

## Gravity Field Models

### Supported Models

1. **EGM96** - Earth Gravitational Model 1996
   - Complete to degree 360
   - ~55 km spatial resolution
   - Global coverage

2. **EGM2008** - Earth Gravitational Model 2008
   - Complete to degree 2190
   - ~9 km spatial resolution
   - Highest resolution global model

### Gravity Anomaly Types

#### Free-Air Anomaly
Corrects for elevation but not mass between observation and reference:

```python
anomaly, components = compute_gravity_anomaly(
    lat, lon, observed_g, model,
    correction_type='free_air',
    elevation=elevation
)
```

Formula: `g_fa = g_obs - g_normal + 0.3086·h`

#### Bouguer Anomaly
Corrects for both elevation and mass of material:

```python
anomaly, components = compute_gravity_anomaly(
    lat, lon, observed_g, model,
    correction_type='bouguer',
    elevation=elevation
)
```

Formula: `g_bouguer = g_fa - 0.1119·ρ·h`

#### Complete Bouguer Anomaly
Includes terrain corrections:

```python
from geophysics.crustal_models import complete_bouguer_anomaly

cba = complete_bouguer_anomaly(
    lat, lon, observed_g, elevation,
    dem, dem_lat, dem_lon
)
```

## Crustal Models

### CRUST1.0 Structure

The CRUST1.0 model provides:
- 7 layers: water, ice, soft sediments, hard sediments, upper/middle/lower crust
- 1° x 1° global coverage
- Layer thicknesses and densities

### Terrain Corrections

Terrain corrections account for gravitational effects of topography:

```python
correction = terrain_correction(
    station_lat, station_lon, station_elev,
    dem, dem_lat, dem_lon,
    density=2670.0,  # kg/m³
    radius=100000.0  # 100 km
)
```

### Isostatic Corrections

Two compensation models are supported:

1. **Airy-Heiskanen**: Variable crustal thickness
```python
correction = isostatic_correction(
    lat, lon, elevation, crustal_model,
    compensation_type='airy'
)
```

2. **Pratt-Hayford**: Variable crustal density
```python
correction = isostatic_correction(
    lat, lon, elevation, crustal_model,
    compensation_type='pratt',
    compensation_depth=35000.0
)
```

## Hydrology Models

### Seasonal Water Storage

Models time-variable water storage from:
- Soil moisture
- Snow cover
- Groundwater
- Surface water

```python
# Load model with temporal data
hydro = load_seasonal_water(
    source='GLDAS',
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31)
)

# Analyze seasonal signal
seasonal = hydro.compute_seasonal_signal(lat, lon)
print(f"Annual amplitude: {seasonal['annual_amplitude']} mm")
```

### Gravity Effect Conversion

Convert between water storage and gravity:

```python
from geophysics.hydrology import compute_grace_equivalent

# Water storage from gravity anomaly
water_mm = compute_grace_equivalent(anomaly, lat_grid, lon_grid)

# Rule of thumb: 1 cm water ≈ 0.42 µGal
```

### Temporal Filtering

Remove seasonal or trending signals:

```python
from geophysics.hydrology import temporal_filtering

# Remove seasonal variations
filtered = temporal_filtering(
    gravity_series, time_stamps,
    filter_type='seasonal'
)

# Extract linear trend
trend = temporal_filtering(
    gravity_series, time_stamps,
    filter_type='trend'
)
```

## Ocean/Land Masking

### Global Masks

```python
# Load with different categories
mask = load_ocean_mask(resolution=1.0, include_lakes=True)

# Categories: 0=ocean, 1=land, 2=ice, 3=lake
categories = mask.categories
```

### Custom Regions

```python
# Rectangular region
region = create_region_mask(
    lat_range=(35, 45),
    lon_range=(-120, -110),
    region_name='western_us'
)

# Polygon region
from geophysics.masking import create_polygon_mask

vertices = [(40, -120), (40, -110), (35, -110), (35, -120)]
polygon = create_polygon_mask(vertices, region_name='my_region')
```

### Mask Operations

```python
from geophysics.masking import combine_masks, mask_statistics

# Combine multiple masks
combined = combine_masks([mask1, mask2], operation='union')

# Compute statistics
stats = mask_statistics(gravity_data, mask, category=1)  # Land only
print(f"Mean: {stats['mean']:.2f} mGal")
print(f"Std: {stats['std']:.2f} mGal")
```

## Joint Inversion

### Integration with Session 5

The joint inversion module provides API for integrating gravity with other datasets:

```python
from geophysics import setup_joint_inversion, integrate_gravity_seismic

# Setup initial model with gravity
joint_model = setup_joint_inversion(
    gravity_data, lat, lon,
    model_name='gravity_seismic'
)

# Add seismic data with petrophysical coupling
joint_model = integrate_gravity_seismic(
    joint_model,
    seismic_velocity,
    seismic_type='velocity',
    coupling_type='petrophysical'
)

# Perform inversion
from geophysics.joint_inversion import perform_joint_inversion

results = perform_joint_inversion(
    joint_model,
    max_iterations=100,
    convergence_tol=1e-6
)

print(f"RMS error: {results['rms_error']:.4f} mGal")
print(f"Converged: {results['converged']}")
```

### Coupling Types

1. **Petrophysical Coupling**: Links density and velocity via empirical relations
   - Gardner's relation: ρ = a·V^b
   - Nafe-Drake curves
   
2. **Structural Coupling**: Ensures boundaries align
   - Gradient matching
   - Interface coherence

3. **Uncoupled**: Independent inversions with shared geometry

### Session 5 Integration

Export/import for Session 5:

```python
from geophysics.joint_inversion import export_for_session5, load_from_session5

# Export to Session 5 format
export_for_session5(joint_model, 'model_for_session5.json')

# Load from Session 5
loaded_model = load_from_session5('session5_output.json')
```

## Best Practices

### Data Preparation

1. **Quality Control**
   - Remove outliers beyond 3σ
   - Check for instrumental drift
   - Validate station elevations

2. **Coordinate Systems**
   - Use WGS84 for consistency
   - Convert to geocentric if needed
   - Handle longitude wrapping (±180°)

3. **Units**
   - Gravity: mGal (1 mGal = 10^-5 m/s²)
   - Density: kg/m³
   - Distance: meters
   - Water storage: mm equivalent

### Correction Order

Apply corrections in this sequence:
1. Instrument corrections (drift, tides)
2. Normal gravity (latitude)
3. Free-air correction (elevation)
4. Bouguer correction (mass)
5. Terrain correction
6. Hydrological correction (if time-series)

### Uncertainty Estimation

```python
# Typical uncertainties
uncertainties = {
    'instrument': 0.01,      # mGal
    'free_air': 0.3086 * elevation_error,
    'bouguer': 0.1119 * density_error * elevation,
    'terrain': 0.05,         # mGal for well-constrained DEM
    'hydrology': 0.02,       # mGal for GLDAS/GRACE
}

total_uncertainty = np.sqrt(sum(u**2 for u in uncertainties.values()))
```

## Performance Optimization

### Large Datasets

For datasets with >10^6 points:

```python
# Use batching
batch_size = 10000
results = []

for i in range(0, len(data), batch_size):
    batch = data[i:i+batch_size]
    result = process_batch(batch)
    results.append(result)

combined = np.concatenate(results)
```

### Memory Management

```python
# Use memory-mapped arrays for large DEMs
import numpy as np

dem = np.memmap('dem.dat', dtype='float32', 
                mode='r', shape=(10000, 10000))
```

## Validation

### Benchmark Datasets

Test against known benchmarks:
- IGSN gravity reference stations
- GRACE satellite validation sites
- Bouguer anomaly maps

### Cross-validation

```python
# Split data for validation
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(
    gravity_data, test_size=0.2, random_state=42
)

# Compute predictions on test set
predicted = model.predict(test_data)
rms_error = np.sqrt(np.mean((test_data - predicted)**2))
```

## References

1. Pavlis, N.K., et al. (2012). "The development and evaluation of the Earth Gravitational Model 2008 (EGM2008)." *Journal of Geophysical Research*, 117(B4).

2. Laske, G., et al. (2013). "Update on CRUST1.0 - A 1-degree Global Model of Earth's Crust." *Geophysical Research Abstracts*, 15, Abstract EGU2013-2658.

3. Rodell, M., et al. (2004). "The Global Land Data Assimilation System." *Bulletin of the American Meteorological Society*, 85(3), 381-394.

4. Tapley, B.D., et al. (2004). "GRACE Measurements of Mass Variability in the Earth System." *Science*, 305(5683), 503-505.

5. Gardner, G.H.F., et al. (1974). "Formation velocity and density—The diagnostic basics for stratigraphic traps." *Geophysics*, 39(6), 770-780.

## Support

For issues or questions:
- Check documentation: `/docs/earth_models.md`
- Review examples: `/examples/geophysics/`
- Contact: support@geophysics.module
