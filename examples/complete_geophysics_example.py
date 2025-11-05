"""
Complete Example: Gravity Data Processing with Earth Models

This example demonstrates:
1. Loading and using reference gravity models
2. Computing various gravity anomalies
3. Applying crustal and terrain corrections
4. Hydrological corrections
5. Ocean/land masking
6. Joint inversion setup
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# Import geophysics module
import sys
sys.path.insert(0, '/home/claude')

from geophysics import (
    load_egm96, load_egm2008, compute_gravity_anomaly,
    load_crust1, terrain_correction, complete_bouguer_anomaly,
    load_seasonal_water, hydrological_correction,
    load_ocean_mask, create_region_mask,
    setup_joint_inversion, integrate_gravity_seismic,
    perform_joint_inversion
)


def example_1_gravity_fields():
    """Example 1: Working with reference gravity fields."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Reference Gravity Fields")
    print("="*60)
    
    # Load EGM96 model
    print("\nLoading EGM96 gravity field model...")
    egm96 = load_egm96()
    print(f"Model: {egm96.name}")
    print(f"Maximum degree: {egm96.degree_max}")
    print(f"Reference radius: {egm96.reference_radius/1000:.0f} km")
    
    # Define survey area (e.g., California)
    lat = np.linspace(32, 42, 50)
    lon = np.linspace(-125, -114, 50)
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij')
    
    # Compute geoid heights
    print("\nComputing geoid heights...")
    geoid = egm96.compute_geoid_height(lat_grid, lon_grid, max_degree=180)
    
    print(f"Geoid statistics:")
    print(f"  Mean: {np.mean(geoid):.2f} m")
    print(f"  Min: {np.min(geoid):.2f} m")
    print(f"  Max: {np.max(geoid):.2f} m")
    print(f"  Std: {np.std(geoid):.2f} m")
    
    # Simulated gravity observations
    observed_gravity = 980000 + np.random.randn(*lat_grid.shape) * 10
    elevation = np.random.rand(*lat_grid.shape) * 2000  # 0-2000 m
    
    # Compute free-air anomaly
    print("\nComputing free-air anomaly...")
    fa_anomaly, fa_components = compute_gravity_anomaly(
        lat_grid, lon_grid, observed_gravity, egm96,
        correction_type='free_air',
        elevation=elevation
    )
    
    print(f"Free-air anomaly statistics:")
    print(f"  Mean: {np.mean(fa_anomaly):.2f} mGal")
    print(f"  RMS: {np.sqrt(np.mean(fa_anomaly**2)):.2f} mGal")
    
    return lat_grid, lon_grid, observed_gravity, elevation, egm96


def example_2_crustal_models(lat, lon, observed_g, elevation):
    """Example 2: Crustal models and terrain corrections."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Crustal Models and Corrections")
    print("="*60)
    
    # Load CRUST1.0
    print("\nLoading CRUST1.0 crustal model...")
    crust = load_crust1()
    print(f"Model: {crust.name}")
    print(f"Grid resolution: {len(crust.lat_grid)}x{len(crust.lon_grid)}")
    print(f"Layers: {list(crust.layer_densities.keys())}")
    
    # Get crustal density
    print("\nQuerying crustal density...")
    density = crust.get_density_at_depth(lat, lon, depth=5000)
    print(f"Upper crustal density statistics:")
    print(f"  Mean: {np.mean(density):.0f} kg/m³")
    print(f"  Range: {np.min(density):.0f} - {np.max(density):.0f} kg/m³")
    
    # Create synthetic DEM for terrain correction
    print("\nPreparing DEM for terrain correction...")
    dem_size = 200
    dem = np.random.rand(dem_size, dem_size) * 3000  # 0-3000 m
    dem_lat = np.linspace(lat.min() - 2, lat.max() + 2, dem_size)
    dem_lon = np.linspace(lon.min() - 2, lon.max() + 2, dem_size)
    
    # Select a subset of points for terrain correction (computationally intensive)
    print("\nComputing terrain corrections (subset)...")
    subset_idx = np.random.choice(lat.size, size=min(100, lat.size), replace=False)
    lat_subset = lat.flat[subset_idx]
    lon_subset = lon.flat[subset_idx]
    elev_subset = elevation.flat[subset_idx]
    
    terrain_corr = terrain_correction(
        lat_subset, lon_subset, elev_subset,
        dem, dem_lat, dem_lon,
        density=2670.0,
        radius=50000.0  # 50 km
    )
    
    print(f"Terrain correction statistics:")
    print(f"  Mean: {np.mean(terrain_corr):.4f} mGal")
    print(f"  Max: {np.max(np.abs(terrain_corr)):.4f} mGal")
    
    # Complete Bouguer anomaly
    print("\nComputing complete Bouguer anomaly...")
    cba_result = complete_bouguer_anomaly(
        lat_subset, lon_subset,
        observed_g.flat[subset_idx],
        elev_subset,
        dem, dem_lat, dem_lon
    )
    
    print(f"Complete Bouguer anomaly statistics:")
    print(f"  Mean: {np.mean(cba_result['complete_bouguer_anomaly']):.2f} mGal")
    print(f"  RMS: {np.sqrt(np.mean(cba_result['complete_bouguer_anomaly']**2)):.2f} mGal")
    
    return crust, cba_result


def example_3_hydrology(lat, lon):
    """Example 3: Hydrological corrections."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Hydrological Corrections")
    print("="*60)
    
    # Load seasonal water model
    print("\nLoading GLDAS seasonal water model...")
    hydro = load_seasonal_water(source='GLDAS')
    print(f"Model: {hydro.name}")
    print(f"Time points: {len(hydro.time_stamps)}")
    print(f"Components: {list(hydro.components.keys())}")
    
    # Analyze seasonal signal
    print("\nAnalyzing seasonal signal...")
    seasonal = hydro.compute_seasonal_signal(lat, lon)
    
    print(f"Seasonal amplitude statistics:")
    print(f"  Annual mean: {np.mean(seasonal['annual_amplitude']):.1f} mm")
    print(f"  Semi-annual mean: {np.mean(seasonal['semiannual_amplitude']):.1f} mm")
    
    # Compute hydrological correction for specific date
    obs_time = datetime(2024, 6, 15)  # Summer in Northern Hemisphere
    ref_time = datetime(2024, 1, 15)  # Winter reference
    
    print(f"\nComputing hydrological correction...")
    print(f"  Observation time: {obs_time}")
    print(f"  Reference time: {ref_time}")
    
    # Select subset for demonstration
    lat_subset = lat[::5, ::5]
    lon_subset = lon[::5, ::5]
    
    hydro_corr = hydrological_correction(
        lat_subset, lon_subset, obs_time, hydro, ref_time
    )
    
    print(f"Hydrological correction statistics:")
    print(f"  Mean: {np.mean(hydro_corr):.4f} mGal")
    print(f"  Range: {np.min(hydro_corr):.4f} to {np.max(hydro_corr):.4f} mGal")
    print(f"  Equivalent water: {np.mean(hydro_corr) / 0.042:.1f} mm")
    
    return hydro, hydro_corr


def example_4_masking(lat, lon, data):
    """Example 4: Ocean/land masking."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Ocean/Land Masking")
    print("="*60)
    
    # Load global mask
    print("\nLoading global ocean/land mask...")
    ocean_mask = load_ocean_mask(resolution=1.0, include_lakes=True)
    print(f"Mask: {ocean_mask.name}")
    print(f"Resolution: {ocean_mask.resolution}°")
    print(f"Categories: {ocean_mask.categories}")
    
    # Check point classification
    lat_subset = lat[::10, ::10]
    lon_subset = lon[::10, ::10]
    
    is_land = ocean_mask.is_land(lat_subset, lon_subset)
    is_ocean = ocean_mask.is_ocean(lat_subset, lon_subset)
    
    print(f"\nClassification statistics:")
    print(f"  Land points: {np.sum(is_land)}/{is_land.size}")
    print(f"  Ocean points: {np.sum(is_ocean)}/{is_ocean.size}")
    
    # Create custom region
    print("\nCreating custom region (California)...")
    california = create_region_mask(
        lat_range=(32.5, 42.0),
        lon_range=(-124.5, -114.1),
        region_name='california'
    )
    
    # Apply mask to data
    print("\nApplying mask to data...")
    from geophysics.masking import mask_statistics
    
    # For demonstration, create data on the same grid as the mask
    mask_data = np.random.randn(len(ocean_mask.lat_grid), len(ocean_mask.lon_grid)) * 10 + 980000
    
    land_stats = mask_statistics(mask_data, ocean_mask, category=1)
    print(f"Land data statistics:")
    print(f"  Count: {land_stats['count']}")
    print(f"  Mean: {land_stats['mean']:.2f}")
    print(f"  Std: {land_stats['std']:.2f}")
    
    return ocean_mask, california


def example_5_joint_inversion(lat, lon, gravity_data):
    """Example 5: Joint inversion setup."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Joint Inversion")
    print("="*60)
    
    # Setup joint inversion with gravity
    print("\nSetting up joint inversion model...")
    joint_model = setup_joint_inversion(
        gravity_data, lat, lon,
        model_name='california_joint'
    )
    print(f"Model: {joint_model.name}")
    print(f"Data types: {joint_model.data_types}")
    
    # Add synthetic seismic data
    print("\nAdding seismic velocity data...")
    seismic_velocity = 5000 + np.random.randn(*gravity_data.shape) * 500  # m/s
    
    joint_model = integrate_gravity_seismic(
        joint_model,
        seismic_velocity,
        seismic_type='velocity',
        coupling_type='petrophysical'
    )
    
    print(f"Updated data types: {joint_model.data_types}")
    print(f"Coupling: {joint_model.metadata['coupling_description']}")
    
    # Perform inversion
    print("\nPerforming joint inversion...")
    results = perform_joint_inversion(
        joint_model,
        max_iterations=50,
        convergence_tol=1e-4
    )
    
    print(f"\nInversion results:")
    print(f"  Iterations: {results['iterations']}")
    print(f"  Converged: {results['converged']}")
    print(f"  RMS error: {results['rms_error']:.4f} mGal")
    print(f"  Final misfit: {results['misfit']:.2e}")
    
    # Model statistics
    density_model = results['density_model']
    print(f"\nRecovered density model:")
    print(f"  Mean: {np.mean(density_model):.0f} kg/m³")
    print(f"  Range: {np.min(density_model):.0f} - {np.max(density_model):.0f} kg/m³")
    
    return joint_model, results


def create_summary_plots(lat, lon, gravity, geoid, cba_result, hydro_corr, mask):
    """Create summary visualization."""
    print("\n" + "="*60)
    print("Creating summary plots...")
    print("="*60)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Observed gravity
    im1 = axes[0, 0].contourf(lon, lat, gravity, levels=20, cmap='RdBu_r')
    axes[0, 0].set_title('Observed Gravity (mGal)')
    axes[0, 0].set_xlabel('Longitude')
    axes[0, 0].set_ylabel('Latitude')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Plot 2: Geoid height
    im2 = axes[0, 1].contourf(lon, lat, geoid, levels=20, cmap='terrain')
    axes[0, 1].set_title('EGM96 Geoid Height (m)')
    axes[0, 1].set_xlabel('Longitude')
    axes[0, 1].set_ylabel('Latitude')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Plot 3: Ocean/land mask
    im3 = axes[0, 2].contourf(
        mask.lon_grid, mask.lat_grid, mask.mask,
        levels=[0, 0.5, 1.5, 2.5, 3.5],
        colors=['blue', 'green', 'white', 'cyan']
    )
    axes[0, 2].set_title('Ocean/Land Mask')
    axes[0, 2].set_xlabel('Longitude')
    axes[0, 2].set_ylabel('Latitude')
    axes[0, 2].set_xlim(lon.min(), lon.max())
    axes[0, 2].set_ylim(lat.min(), lat.max())
    
    # Plot 4: Complete Bouguer anomaly (subset)
    if cba_result:
        cba = cba_result['complete_bouguer_anomaly']
        axes[1, 0].hist(cba, bins=30, edgecolor='black', alpha=0.7)
        axes[1, 0].set_title('Complete Bouguer Anomaly Distribution')
        axes[1, 0].set_xlabel('Anomaly (mGal)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].axvline(np.mean(cba), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(cba):.2f}')
        axes[1, 0].legend()
    
    # Plot 5: Hydrological correction
    if hydro_corr is not None and hydro_corr.size > 1:
        hydro_flat = hydro_corr.flatten()
        axes[1, 1].hist(hydro_flat, bins=20, edgecolor='black', alpha=0.7, color='cyan')
        axes[1, 1].set_title('Hydrological Correction Distribution')
        axes[1, 1].set_xlabel('Correction (mGal)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].axvline(np.mean(hydro_flat), color='red', linestyle='--',
                          label=f'Mean: {np.mean(hydro_flat):.4f}')
        axes[1, 1].legend()
    
    # Plot 6: Summary statistics
    axes[1, 2].axis('off')
    summary_text = f"""
    PROCESSING SUMMARY
    
    Survey Area:
      Lat: {lat.min():.1f}° to {lat.max():.1f}°
      Lon: {lon.min():.1f}° to {lon.max():.1f}°
      Points: {gravity.size}
    
    Gravity Statistics:
      Mean: {np.mean(gravity):.2f} mGal
      Std: {np.std(gravity):.2f} mGal
    
    Geoid Range:
      {np.min(geoid):.2f} to {np.max(geoid):.2f} m
    
    Corrections Applied:
      ✓ Free-air
      ✓ Bouguer
      ✓ Terrain
      ✓ Hydrological
    """
    axes[1, 2].text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                   verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig('/home/claude/geophysics_example_summary.png', dpi=150, bbox_inches='tight')
    print("Saved summary plot: /home/claude/geophysics_example_summary.png")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("GEOPHYSICS MODULE - COMPLETE DEMONSTRATION")
    print("="*60)
    print("\nThis example demonstrates all features of the geophysics module")
    print("including gravity fields, crustal models, hydrology, masking,")
    print("and joint inversion capabilities.")
    
    # Example 1: Gravity fields
    lat, lon, gravity, elevation, egm96 = example_1_gravity_fields()
    geoid = egm96.compute_geoid_height(lat, lon, max_degree=180)
    
    # Example 2: Crustal models
    crust, cba_result = example_2_crustal_models(lat, lon, gravity, elevation)
    
    # Example 3: Hydrology
    hydro, hydro_corr = example_3_hydrology(lat, lon)
    
    # Example 4: Masking
    ocean_mask, california = example_4_masking(lat, lon, gravity)
    
    # Example 5: Joint inversion
    joint_model, results = example_5_joint_inversion(lat, lon, gravity)
    
    # Create visualization
    create_summary_plots(lat, lon, gravity, geoid, cba_result, hydro_corr, ocean_mask)
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print("\nAll geophysics module features have been demonstrated.")
    print("Check /home/claude/geophysics_example_summary.png for visualization.")
    print("\nFor more information, see: /home/claude/docs/earth_models.md")


if __name__ == '__main__':
    main()
