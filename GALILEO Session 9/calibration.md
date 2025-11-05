# Session 9: Calibration and Noise Characterization

## Overview

Session 9 implements comprehensive tools for calibrating orbit determination systems and characterizing measurement noise. These tools are essential for:

- Assessing sensor and measurement quality
- Estimating physical parameters (drag, solar pressure coefficients)
- Validating filter performance
- Constructing error budgets
- Designing optimal calibration maneuvers

## Module Structure

```
sim/
├── calibration.py          # Allan deviation, PSD, whiteness tests
├── system_id.py           # Parameter estimation (CD, CR, empirical models)
├── cal_maneuvers.py       # Synthetic calibration data generation
└── validate_calibration.py # Validation suite with plots
```

## Core Concepts

### 1. Allan Deviation

Allan deviation characterizes the stability of measurements over different averaging times. It's particularly useful for identifying noise types:

**Noise Types and Slopes:**
- **White Noise**: τ^(-1/2) slope - random, uncorrelated
- **Flicker Noise**: τ^(0) slope - 1/f noise, common in electronics
- **Random Walk**: τ^(+1/2) slope - integrated white noise
- **Rate Random Walk**: τ^(+1) slope - drift over time

**Usage:**
```python
from calibration import AllanDeviation

# Compute Allan deviation
adev_calc = AllanDeviation(data, sample_rate, overlapping=True)
taus, adev = adev_calc.compute()

# Identify noise type
noise_id = adev_calc.identify_noise_type(taus, adev)
print(f"Noise type: {noise_id['type']}")
print(f"Slope: {noise_id['slope']:.3f}")
```

### 2. Cross-Spectral Density

Cross-spectral density (CSD) and coherence reveal frequency-domain relationships between measurement channels.

**Key Metrics:**
- **PSD**: Power spectral density - energy distribution vs frequency
- **CSD**: Cross-spectral density - relationship between two signals
- **Coherence**: Normalized correlation (0 to 1) vs frequency
- **Transfer Function**: H(f) = Pyx(f) / Pxx(f)

**Usage:**
```python
from calibration import CrossSpectralDensity

csd_calc = CrossSpectralDensity(nperseg=256)

# Power spectral density
freqs, psd = csd_calc.compute_psd(signal, sample_rate)

# Coherence between channels
freqs, coherence = csd_calc.compute_coherence(signal1, signal2, sample_rate)
```

### 3. Whiteness Tests

Residuals from an optimal filter should be white (uncorrelated) noise. Several tests validate this:

**Statistical Tests:**
- **Ljung-Box**: Tests for autocorrelation at multiple lags
- **Runs Test**: Tests for randomness of positive/negative sequences
- **Durbin-Watson**: Tests for first-order autocorrelation (should be ≈ 2)

**Usage:**
```python
from calibration import WhitenessTest

# Comprehensive whiteness assessment
results = WhitenessTest.comprehensive_test(residuals)

print(f"Ljung-Box p-value: {results['ljung_box']['p_value']:.4f}")
print(f"Overall white: {results['overall_white']}")
```

**Interpretation:**
- p-value > 0.05: Cannot reject null hypothesis of whiteness ✓
- p-value < 0.05: Evidence of autocorrelation (not white) ✗
- DW ≈ 2.0: No first-order autocorrelation ✓

## System Identification

### Drag Coefficient Estimation

Atmospheric drag is modeled as:

```
a_drag = -0.5 * CD * (A/m) * ρ * v_rel * |v_rel|
```

Where:
- CD = drag coefficient (typically 2.0-2.5)
- A/m = area-to-mass ratio (m²/kg)
- ρ = atmospheric density (kg/m³)
- v_rel = relative velocity (m/s)

**Usage:**
```python
from system_id import DragCoefficientEstimator

estimator = DragCoefficientEstimator(
    area_to_mass=0.01,  # m²/kg
    initial_cd=2.2
)

result = estimator.estimate(
    times, residuals, density, velocity
)

print(f"CD: {result.parameters[0]:.4f} ± {result.parameter_uncertainties()[0]:.4f}")
```

### Solar Radiation Pressure Estimation

SRP acceleration:

```
a_srp = -CR * (A/m) * (P_sun / c) * (1 AU / r)² * ŝ
```

Where:
- CR = radiation pressure coefficient (typically 1.0-2.0)
- P_sun = 4.56×10⁻⁶ N/m² (at 1 AU)
- c = speed of light
- r = distance to sun
- ŝ = unit vector away from sun

**Usage:**
```python
from system_id import SolarPressureEstimator

estimator = SolarPressureEstimator(
    area_to_mass=0.01,
    initial_cr=1.5
)

result = estimator.estimate(
    times, residuals, 
    sun_vectors, sun_distances, shadow_factors
)

print(f"CR: {result.parameters[0]:.4f}")
```

### Empirical Acceleration Models

For unmodeled forces, fit empirical models using:
- Polynomial trends (secular drift)
- Harmonic terms (periodic effects)
- Once-per-orbit terms

**Usage:**
```python
from system_id import EmpiricalAccelerationModel

model = EmpiricalAccelerationModel(
    n_harmonics=2,
    polynomial_degree=1
)

result = model.fit(times, residuals, orbital_period)

# Evaluate model
predicted = model.evaluate(times, orbital_period, result.parameters)
```

## Calibration Maneuvers

### Maneuver Types

1. **Translation (Delta-V)**
   - Excites drag and SRP dynamics
   - Tests thrust model
   - Typical: 0.1-1.0 m/s

2. **Rotation (Attitude)**
   - Calibrates center of mass
   - Tests moment of inertia
   - Reveals thrust misalignment

3. **Ballistic Coefficient Changes**
   - Deploys/retracts solar panels
   - Varies cross-sectional area
   - Separates CD from A/m

4. **Coast Arcs**
   - Passive propagation
   - Baseline dynamics
   - Reference for comparison

### Optimal Maneuver Design

**Principles:**
- **Separation**: Space maneuvers 1/4 orbital period apart
- **Diversity**: Vary directions and magnitudes
- **Duration**: Sufficient observability (>1 orbit per configuration)
- **Safety**: Within operational constraints

**Usage:**
```python
from cal_maneuvers import CalibrationManeuverGenerator

generator = CalibrationManeuverGenerator(orbital_period=5400)

# Generate comprehensive sequence
maneuvers = generator.generate_comprehensive_sequence()

# Custom delta-V sequence
dv_maneuvers = generator.generate_delta_v_sequence(
    n_maneuvers=5,
    delta_v_range=(0.1, 1.0),
    spacing='optimal'
)
```

## Synthetic Data Generation

Generate realistic orbit data for testing:

```python
from cal_maneuvers import SyntheticOrbitGenerator

orbit_gen = SyntheticOrbitGenerator()

# Propagate Keplerian orbit
states = orbit_gen.propagate_keplerian(initial_state, times)

# Add perturbations
states = orbit_gen.add_drag_perturbation(
    states, times, cd, area_to_mass, density_model
)

states = orbit_gen.add_srp_perturbation(
    states, times, cr, area_to_mass, sun_vectors
)

# Apply maneuvers
states = orbit_gen.apply_maneuvers(states, times, maneuvers)

# Generate measurements
measurements = orbit_gen.generate_measurements(
    states, measurement_type='range', noise_std=10.0
)
```

## Residual Analysis

Comprehensive residual diagnostics:

```python
from system_id import ResidualAnalyzer

analyzer = ResidualAnalyzer()

# Basic statistics
stats = analyzer.compute_statistics(residuals)
print(f"RMS: {stats['rms']}")
print(f"Mean: {stats['mean']}")

# Outlier detection
outliers = analyzer.detect_outliers(residuals, threshold=3.0)
print(f"Outliers: {np.sum(outliers)} / {len(residuals)}")

# Autocorrelation
acf = analyzer.compute_autocorrelation(residuals, max_lag=50)

# Normalized Innovation Squared (NIS)
nis = analyzer.normalized_innovation_squared(residuals, covariances)
# Should be chi-squared distributed with dim(z) degrees of freedom
```

## Error Budget Analysis

Track error contributions from all sources:

```python
error_sources = {
    'Measurement Noise': 10.0,      # meters
    'Atmospheric Drag': 5.0,
    'Solar Pressure': 3.0,
    'Earth Gravity Model': 2.0,
    'Third Body Perturbations': 1.0,
    'Numerical Integration': 0.5,
    'Station Location': 0.3,
    'Timing Errors': 0.2
}

# Root-sum-squared
rss_total = np.sqrt(sum(v**2 for v in error_sources.values()))

# Contribution percentages
for source, error in error_sources.items():
    contribution = (error / rss_total) * 100
    print(f"{source}: {error:.2f} m ({contribution:.1f}%)")
```

## Validation Workflow

Complete validation process:

```bash
cd /home/claude/orbit_determination/sim
python validate_calibration.py
```

**Validation Tests:**

1. **Whiteness Validation**
   - Verifies white noise passes all tests ✓
   - Confirms autocorrelated noise fails tests ✓
   - Generates ACF plots

2. **Allan Deviation**
   - Tests different noise types
   - Validates slope identification
   - Generates multi-panel plots

3. **System Identification**
   - Estimates CD and CR from synthetic data
   - Checks estimation accuracy
   - Validates post-fit residuals are white

4. **Error Budget**
   - Plots contribution by source
   - Shows cumulative error growth
   - Models time evolution

## Output Files

Validation generates diagnostic plots:

```
validation_results/
├── whiteness_acf.png          # Autocorrelation functions
├── allan_deviation.png        # Allan deviation for noise types
├── parameter_estimation.png   # CD and CR estimation results
└── error_budget.png          # Error budget breakdown
```

## Best Practices

### 1. Data Quality Assessment

**Before filtering:**
- Compute Allan deviation to characterize sensor noise
- Check for unexpected noise types (flicker, random walk)
- Identify appropriate averaging times

**After filtering:**
- Test residuals for whiteness
- Look for systematic patterns (bias, periodicities)
- Check autocorrelation function

### 2. Parameter Estimation

**Drag Coefficient:**
- Requires dense atmosphere (LEO < 800 km)
- Needs accurate density model
- Best estimated during high solar activity

**SRP Coefficient:**
- Requires sunlit arcs (exclude shadow)
- More observable at higher altitudes (>800 km)
- Correlates with area-to-mass ratio

**Best Practice:**
- Estimate CD and CR separately if possible
- Use multiple orbits for stability
- Check parameter correlations

### 3. Calibration Maneuvers

**Design Principles:**
- Start with coast arc (baseline)
- Space maneuvers for observability
- Vary directions to excite all axes
- Include attitude changes if possible
- End with verification coast arc

**Timing:**
- Delta-V: 1/4 orbital period spacing
- Attitude: 1/2 orbital period minimum
- Ballistic changes: Full orbit minimum

### 4. Diagnostics

**Red Flags:**
- Non-white residuals → Model error or unmodeled forces
- Large autocorrelation → Systematic error
- High DW deviation → First-order correlation
- Failed Ljung-Box → Multiple lags correlated

**Remedies:**
- Add empirical acceleration model
- Improve force models (gravity, atmosphere)
- Check for outliers or bad data
- Increase process noise

## Example: Complete Calibration

```python
import numpy as np
from calibration import AllanDeviation, WhitenessTest
from system_id import DragCoefficientEstimator, ResidualAnalyzer
from cal_maneuvers import CalibrationManeuverGenerator

# 1. Characterize sensor noise
print("=== SENSOR CHARACTERIZATION ===")
sensor_data = load_sensor_data()  # Your data loading
adev_calc = AllanDeviation(sensor_data, sample_rate=100.0)
taus, adev = adev_calc.compute()
noise_type = adev_calc.identify_noise_type(taus, adev)
print(f"Sensor noise type: {noise_type['type']}")

# 2. Design calibration maneuvers
print("\n=== MANEUVER DESIGN ===")
generator = CalibrationManeuverGenerator(orbital_period=5400)
maneuvers = generator.generate_comprehensive_sequence()
print(f"Generated {len(maneuvers)} maneuvers")

# 3. Process filter residuals
print("\n=== FILTER VALIDATION ===")
filter_residuals = run_filter()  # Your filter
whiteness = WhitenessTest.comprehensive_test(filter_residuals)
print(f"Residuals white: {whiteness['overall_white']}")

# 4. Estimate parameters
print("\n=== PARAMETER ESTIMATION ===")
cd_estimator = DragCoefficientEstimator(area_to_mass=0.01)
cd_result = cd_estimator.estimate(times, residuals, density, velocity)
print(f"CD = {cd_result.parameters[0]:.4f} ± {cd_result.parameter_uncertainties()[0]:.4f}")

# 5. Validate post-fit residuals
post_fit_white = WhitenessTest.comprehensive_test(cd_result.residuals.flatten())
print(f"Post-fit residuals white: {post_fit_white['overall_white']}")

# 6. Analyze residuals
print("\n=== RESIDUAL ANALYSIS ===")
analyzer = ResidualAnalyzer()
analysis = analyzer.comprehensive_analysis(cd_result.residuals)
print(f"RMS: {analysis['statistics']['rms']}")
print(f"Outliers: {analysis['n_outliers']} ({analysis['outlier_fraction']*100:.1f}%)")
```

## Mathematical Background

### Allan Variance

For averaging time τ:

```
σ²(τ) = (1/2) ⟨(x̄ᵢ₊₁ - x̄ᵢ)²⟩
```

where x̄ᵢ is the average over the i-th interval of length τ.

### Ljung-Box Statistic

```
Q = n(n+2) Σₖ₌₁ʰ (ρ²ₖ / (n-k))
```

where ρₖ is the autocorrelation at lag k. Under H₀ (white noise), Q ~ χ²(h-p) where p is the number of fitted parameters.

### Durbin-Watson Statistic

```
DW = Σₜ₌₂ⁿ (eₜ - eₜ₋₁)² / Σₜ₌₁ⁿ eₜ²
```

Approximately: DW ≈ 2(1 - ρ₁) where ρ₁ is first-order autocorrelation.

### Normalized Innovation Squared

```
NIS(k) = νₖᵀ Sₖ⁻¹ νₖ
```

where νₖ is the innovation and Sₖ is the innovation covariance. Should follow χ²(m) distribution where m = dim(νₖ).

## References

### Allan Deviation
- IEEE Standard 1139-2008: "Standard Definitions of Physical Quantities for Fundamental Frequency and Time Metrology"
- Riley, W.J. (2008): "Handbook of Frequency Stability Analysis"

### System Identification
- Montenbruck & Gill (2000): "Satellite Orbits: Models, Methods, Applications"
- Tapley, Schutz & Born (2004): "Statistical Orbit Determination"

### Calibration
- Markley & Crassidis (2014): "Fundamentals of Spacecraft Attitude Determination and Control"
- Vallado (2013): "Fundamentals of Astrodynamics and Applications"

## Troubleshooting

### Issue: Non-white residuals

**Symptoms:**
- Failed Ljung-Box test
- High autocorrelation
- Systematic patterns in residual plots

**Solutions:**
1. Add empirical acceleration model
2. Improve force models (gravity, drag, SRP)
3. Check for outliers
4. Increase process noise in filter

### Issue: Poor parameter estimates

**Symptoms:**
- Large uncertainties
- Unstable estimates
- High correlation with other parameters

**Solutions:**
1. Increase data span (more orbits)
2. Design better calibration maneuvers
3. Improve observability (geometry, timing)
4. Consider parameter correlations

### Issue: Inconsistent noise characterization

**Symptoms:**
- Allan deviation doesn't match expected
- Unclear noise type identification
- Multiple slopes in different regions

**Solutions:**
1. Collect longer data (10x averaging time)
2. Check for non-stationarity
3. Look for environmental changes
4. Consider mixed noise types

## Next Steps

After Session 9, you'll have:
- ✓ Characterized measurement noise
- ✓ Estimated physical parameters
- ✓ Validated filter residuals
- ✓ Generated error budgets
- ✓ Designed calibration maneuvers

**Integration:**
- Use estimated CD/CR in force models
- Apply whiteness tests to all filters
- Generate error budgets for mission planning
- Design operational calibration sequences

---

*Session 9 complete! Your orbit determination system now includes comprehensive calibration and validation capabilities.*
