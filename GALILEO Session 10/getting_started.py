"""
Getting Started with Geophysics Module

This script provides a simple, interactive introduction to the geophysics module.
Run this to see basic functionality and get started quickly.
"""

import sys
sys.path.insert(0, '/home/claude')

import numpy as np
from datetime import datetime

print("="*60)
print("GEOPHYSICS MODULE - GETTING STARTED")
print("="*60)
print("\nWelcome! This script will demonstrate basic geophysics operations.")
print("Let's process some gravity data together!\n")

# Step 1: Load gravity model
print("Step 1: Loading EGM96 gravity field model...")
from geophysics import load_egm96
egm96 = load_egm96()
print(f"✓ Loaded {egm96.name} (max degree: {egm96.degree_max})")

# Step 2: Define observation points
print("\nStep 2: Defining observation points...")
lat = np.array([40.0, 41.0, 42.0])
lon = np.array([-120.0, -119.0, -118.0])
observed_gravity = np.array([980200, 980150, 980180])  # mGal
elevation = np.array([100, 200, 150])  # meters

print(f"  Locations: {len(lat)} stations")
print(f"  Latitude range: {lat.min():.1f}° to {lat.max():.1f}°")
print(f"  Longitude range: {lon.min():.1f}° to {lon.max():.1f}°")
print(f"  Elevation range: {elevation.min():.0f} to {elevation.max():.0f} m")

# Step 3: Compute gravity anomaly
print("\nStep 3: Computing free-air gravity anomaly...")
from geophysics import compute_gravity_anomaly

anomaly, components = compute_gravity_anomaly(
    lat, lon, observed_gravity, egm96,
    correction_type='free_air',
    elevation=elevation
)

print(f"✓ Free-air anomaly computed")
print(f"  Mean anomaly: {np.mean(anomaly):.2f} mGal")
print(f"  Range: {np.min(anomaly):.2f} to {np.max(anomaly):.2f} mGal")

# Step 4: Load crustal model
print("\nStep 4: Loading CRUST1.0 crustal model...")
from geophysics import load_crust1

crust = load_crust1()
print(f"✓ Loaded {crust.name}")
print(f"  Grid resolution: {len(crust.lat_grid)}x{len(crust.lon_grid)}")
print(f"  Layers: {len(crust.layer_densities)}")

# Query crustal density
density = crust.get_density_at_depth(lat, lon, depth=5000)
print(f"  Crustal density at 5 km depth: {density[0]:.0f} kg/m³")

# Step 5: Bouguer correction
print("\nStep 5: Computing Bouguer correction...")
from geophysics import bouguer_correction

bouguer = bouguer_correction(elevation, density=2670.0)
print(f"✓ Bouguer correction computed")
print(f"  Mean correction: {np.mean(bouguer):.2f} mGal")

# Step 6: Hydrological correction
print("\nStep 6: Computing hydrological correction...")
from geophysics import load_seasonal_water, hydrological_correction

hydro = load_seasonal_water(source='GLDAS')
print(f"✓ Loaded {hydro.name} seasonal water model")
print(f"  Time points: {len(hydro.time_stamps)}")

obs_time = datetime(2024, 6, 15)
hydro_corr = hydrological_correction(lat, lon, obs_time, hydro)
print(f"  Hydrological correction: {np.mean(hydro_corr):.4f} mGal")

# Step 7: Ocean/land masking
print("\nStep 7: Checking ocean/land classification...")
from geophysics import load_ocean_mask

mask = load_ocean_mask(resolution=1.0)
is_land = mask.is_land(lat, lon)
print(f"✓ Loaded {mask.name}")
print(f"  Station classification:")
for i, (lt, ln, land) in enumerate(zip(lat, lon, is_land)):
    status = "Land" if land else "Ocean"
    print(f"    Station {i+1} ({lt:.1f}°, {ln:.1f}°): {status}")

# Step 8: Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"\nProcessed {len(lat)} gravity stations:")
print(f"  ✓ Gravity anomaly computed")
print(f"  ✓ Crustal density queried")
print(f"  ✓ Bouguer correction applied")
print(f"  ✓ Hydrological correction estimated")
print(f"  ✓ Ocean/land classification done")

print("\nCorrected Gravity Values:")
for i in range(len(lat)):
    total_correction = (
        components['free_air_correction'][i] + 
        bouguer[i] + 
        hydro_corr[i]
    )
    corrected = observed_gravity[i] + total_correction
    print(f"  Station {i+1}: {corrected:.2f} mGal (total correction: {total_correction:.2f} mGal)")

print("\n" + "="*60)
print("NEXT STEPS")
print("="*60)
print("\n1. Explore the complete example:")
print("   python /home/claude/examples/complete_geophysics_example.py")
print("\n2. Read the documentation:")
print("   /home/claude/docs/earth_models.md")
print("\n3. Check the quick reference:")
print("   /home/claude/QUICK_REFERENCE.md")
print("\n4. Run the test suite:")
print("   python /home/claude/test_geophysics.py")
print("\n5. Try the benchmarks:")
print("   python /home/claude/benchmarks/background_removal_benchmarks.py")

print("\n" + "="*60)
print("Ready to process your own gravity data!")
print("="*60)
print()
