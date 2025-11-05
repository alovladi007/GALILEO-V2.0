"""
Background Removal Benchmarks

Comprehensive benchmarking suite for evaluating the effectiveness of
Earth model corrections in removing systematic signals from gravity data.

Benchmarks:
1. Temporal stability (seasonal removal)
2. Spatial coherence (terrain correction quality)
3. Model comparison (EGM96 vs EGM2008)
4. Cross-validation
5. Synthetic data tests
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
import json
from typing import Dict, List, Tuple
import time

import sys
sys.path.insert(0, '/home/claude')

from geophysics import (
    load_egm96, load_egm2008, compute_gravity_anomaly,
    load_crust1, complete_bouguer_anomaly,
    load_seasonal_water, hydrological_correction,
    load_ocean_mask
)
from geophysics.hydrology import temporal_filtering, estimate_storage_change


class BenchmarkResults:
    """Container for benchmark results."""
    
    def __init__(self, name: str):
        self.name = name
        self.metrics = {}
        self.timings = {}
        self.plots = {}
    
    def add_metric(self, metric_name: str, value: float):
        """Add a metric to results."""
        self.metrics[metric_name] = value
    
    def add_timing(self, operation: str, duration: float):
        """Add timing information."""
        self.timings[operation] = duration
    
    def save(self, output_dir: Path):
        """Save results to file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_dict = {
            'benchmark_name': self.name,
            'metrics': self.metrics,
            'timings': self.timings,
            'timestamp': datetime.now().isoformat()
        }
        
        output_file = output_dir / f'{self.name}_results.json'
        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"Saved results to: {output_file}")
    
    def print_summary(self):
        """Print summary of results."""
        print(f"\n{'='*60}")
        print(f"BENCHMARK: {self.name}")
        print(f"{'='*60}")
        
        if self.metrics:
            print("\nMetrics:")
            for metric, value in self.metrics.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.6f}")
                else:
                    print(f"  {metric}: {value}")
        
        if self.timings:
            print("\nTimings:")
            for operation, duration in self.timings.items():
                print(f"  {operation}: {duration:.3f} s")


def benchmark_1_temporal_stability(output_dir: Path) -> BenchmarkResults:
    """
    Benchmark 1: Temporal stability of hydrological corrections.
    
    Tests ability to remove seasonal signals from time-series gravity data.
    """
    print("\n" + "="*60)
    print("BENCHMARK 1: Temporal Stability (Seasonal Removal)")
    print("="*60)
    
    results = BenchmarkResults('temporal_stability')
    
    # Create synthetic time series with known seasonal signal
    n_months = 24
    lat = np.array([40.0])
    lon = np.array([-120.0])
    
    # Generate synthetic data: trend + seasonal + noise
    t = np.arange(n_months) / 12.0  # Years
    trend = 0.5 * t  # 0.5 mGal/year drift
    seasonal = 2.0 * np.sin(2 * np.pi * t) + 0.5 * np.sin(4 * np.pi * t)
    noise = np.random.randn(n_months) * 0.1
    
    synthetic_gravity = trend + seasonal + noise + 980000
    
    print(f"\nSynthetic time series generated:")
    print(f"  Duration: {n_months} months")
    print(f"  Trend: 0.5 mGal/year")
    print(f"  Seasonal amplitude: 2.0 mGal (annual)")
    print(f"  Noise level: 0.1 mGal")
    
    # Apply seasonal removal
    time_stamps = [datetime(2024, i+1, 1) for i in range(n_months)]
    
    start_time = time.time()
    filtered = temporal_filtering(
        synthetic_gravity, time_stamps,
        filter_type='seasonal'
    )
    duration = time.time() - start_time
    results.add_timing('seasonal_filtering', duration)
    
    # Compute metrics
    residual = synthetic_gravity - filtered - seasonal
    
    # Original signal statistics
    signal_power = np.var(seasonal)
    noise_power = np.var(noise)
    snr_before = 10 * np.log10(signal_power / noise_power)
    
    # After filtering
    seasonal_removed = np.var(residual - trend - noise)
    snr_after = 10 * np.log10(noise_power / seasonal_removed)
    
    # Metrics
    removal_efficiency = 1 - (seasonal_removed / signal_power)
    rms_residual = np.sqrt(np.mean(residual**2))
    
    results.add_metric('signal_to_noise_before_dB', snr_before)
    results.add_metric('signal_to_noise_after_dB', snr_after)
    results.add_metric('seasonal_removal_efficiency', removal_efficiency)
    results.add_metric('rms_residual_mGal', rms_residual)
    results.add_metric('improvement_dB', snr_after - snr_before)
    
    print(f"\nResults:")
    print(f"  SNR before: {snr_before:.2f} dB")
    print(f"  SNR after: {snr_after:.2f} dB")
    print(f"  Improvement: {snr_after - snr_before:.2f} dB")
    print(f"  Removal efficiency: {removal_efficiency*100:.1f}%")
    print(f"  RMS residual: {rms_residual:.4f} mGal")
    
    # Create plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Original signal
    axes[0].plot(t, synthetic_gravity - 980000, 'b-', label='Observed', linewidth=1.5)
    axes[0].plot(t, trend + seasonal, 'r--', label='True signal', linewidth=1)
    axes[0].set_ylabel('Gravity (mGal)')
    axes[0].set_title('Original Time Series')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Filtered signal
    axes[1].plot(t, filtered - 980000, 'g-', label='Filtered (trend)', linewidth=1.5)
    axes[1].plot(t, trend, 'k--', label='True trend', linewidth=1)
    axes[1].set_ylabel('Gravity (mGal)')
    axes[1].set_title('After Seasonal Removal')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Residuals
    axes[2].plot(t, residual, 'r-', label='Residual', linewidth=1)
    axes[2].axhline(0, color='k', linestyle='--', linewidth=0.5)
    axes[2].fill_between(t, -2*rms_residual, 2*rms_residual, 
                         alpha=0.3, label='±2σ')
    axes[2].set_xlabel('Time (years)')
    axes[2].set_ylabel('Residual (mGal)')
    axes[2].set_title(f'Residuals (RMS: {rms_residual:.4f} mGal)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = output_dir / 'benchmark1_temporal_stability.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {plot_file}")
    plt.close()
    
    return results


def benchmark_2_spatial_coherence(output_dir: Path) -> BenchmarkResults:
    """
    Benchmark 2: Spatial coherence of terrain corrections.
    
    Tests effectiveness of terrain correction in removing topographic effects.
    """
    print("\n" + "="*60)
    print("BENCHMARK 2: Spatial Coherence (Terrain Correction)")
    print("="*60)
    
    results = BenchmarkResults('spatial_coherence')
    
    # Create synthetic topography and gravity
    n = 50
    lat = np.linspace(35, 45, n)
    lon = np.linspace(-125, -115, n)
    lat_2d, lon_2d = np.meshgrid(lat, lon, indexing='ij')
    
    # Synthetic topography with various wavelengths
    elevation = 1000 + 800 * np.sin(2*np.pi*lat_2d/10) * np.cos(2*np.pi*lon_2d/10)
    elevation += 300 * np.random.randn(*elevation.shape)
    elevation = np.maximum(elevation, 0)
    
    # Synthetic gravity with topographic correlation
    # Real Bouguer correction would be ~0.1119 * elevation
    topo_effect = 0.11 * elevation
    background = np.sin(lat_2d/5) * np.cos(lon_2d/5) * 20  # Geological background
    noise = np.random.randn(*lat_2d.shape) * 2
    
    synthetic_gravity = 980000 + topo_effect + background + noise
    
    print(f"\nSynthetic survey generated:")
    print(f"  Grid size: {n}x{n}")
    print(f"  Elevation range: {elevation.min():.0f} - {elevation.max():.0f} m")
    print(f"  Topographic effect: {topo_effect.min():.1f} - {topo_effect.max():.1f} mGal")
    
    # Create DEM (same as elevation for this test)
    dem = elevation.copy()
    dem_lat = lat
    dem_lon = lon
    
    # Compute Complete Bouguer Anomaly
    start_time = time.time()
    
    # Select subset for computation
    subset_size = 100
    idx = np.random.choice(lat_2d.size, size=subset_size, replace=False)
    lat_subset = lat_2d.flat[idx]
    lon_subset = lon_2d.flat[idx]
    g_subset = synthetic_gravity.flat[idx]
    elev_subset = elevation.flat[idx]
    
    cba_result = complete_bouguer_anomaly(
        lat_subset, lon_subset, g_subset, elev_subset,
        dem, dem_lat, dem_lon,
        density=2670.0
    )
    
    duration = time.time() - start_time
    results.add_timing('terrain_correction', duration)
    
    # Compute correlation with elevation before and after
    fa_anomaly = cba_result['free_air_anomaly']
    cba = cba_result['complete_bouguer_anomaly']
    
    corr_before = np.corrcoef(elev_subset, g_subset)[0, 1]
    corr_fa = np.corrcoef(elev_subset, fa_anomaly)[0, 1]
    corr_after = np.corrcoef(elev_subset, cba)[0, 1]
    
    # Compute reduction in correlation
    reduction = abs(corr_after) / abs(corr_before) if corr_before != 0 else 0
    
    results.add_metric('correlation_with_elevation_before', abs(corr_before))
    results.add_metric('correlation_with_elevation_fa', abs(corr_fa))
    results.add_metric('correlation_with_elevation_after', abs(corr_after))
    results.add_metric('correlation_reduction_factor', 1 - reduction)
    results.add_metric('rms_cba_mGal', np.sqrt(np.mean(cba**2)))
    
    print(f"\nResults:")
    print(f"  Correlation with elevation (observed): {abs(corr_before):.4f}")
    print(f"  Correlation with elevation (free-air): {abs(corr_fa):.4f}")
    print(f"  Correlation with elevation (CBA): {abs(corr_after):.4f}")
    print(f"  Reduction factor: {(1-reduction)*100:.1f}%")
    print(f"  RMS CBA: {np.sqrt(np.mean(cba**2)):.2f} mGal")
    
    # Create plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Elevation
    im1 = axes[0, 0].contourf(lon, lat, elevation, levels=20, cmap='terrain')
    axes[0, 0].set_title('Elevation (m)')
    axes[0, 0].set_xlabel('Longitude')
    axes[0, 0].set_ylabel('Latitude')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Observed gravity
    im2 = axes[0, 1].contourf(lon, lat, synthetic_gravity - 980000, 
                              levels=20, cmap='RdBu_r')
    axes[0, 1].set_title('Observed Gravity (mGal)')
    axes[0, 1].set_xlabel('Longitude')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Scatter: elevation vs gravity
    axes[1, 0].scatter(elev_subset, g_subset - 980000, alpha=0.5, s=20)
    axes[1, 0].set_xlabel('Elevation (m)')
    axes[1, 0].set_ylabel('Observed Gravity (mGal)')
    axes[1, 0].set_title(f'Before (r={corr_before:.3f})')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Scatter: elevation vs CBA
    axes[1, 1].scatter(elev_subset, cba, alpha=0.5, s=20, color='green')
    axes[1, 1].set_xlabel('Elevation (m)')
    axes[1, 1].set_ylabel('Complete Bouguer Anomaly (mGal)')
    axes[1, 1].set_title(f'After (r={corr_after:.3f})')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = output_dir / 'benchmark2_spatial_coherence.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {plot_file}")
    plt.close()
    
    return results


def benchmark_3_model_comparison(output_dir: Path) -> BenchmarkResults:
    """
    Benchmark 3: Comparison of gravity field models.
    
    Compares EGM96 and EGM2008 for geoid and gravity anomaly computation.
    """
    print("\n" + "="*60)
    print("BENCHMARK 3: Gravity Field Model Comparison")
    print("="*60)
    
    results = BenchmarkResults('model_comparison')
    
    # Test region
    lat = np.linspace(30, 50, 40)
    lon = np.linspace(-130, -110, 40)
    lat_2d, lon_2d = np.meshgrid(lat, lon, indexing='ij')
    
    # Load models
    print("\nLoading gravity models...")
    start_time = time.time()
    egm96 = load_egm96()
    load_time_96 = time.time() - start_time
    
    start_time = time.time()
    egm2008 = load_egm2008()
    load_time_2008 = time.time() - start_time
    
    results.add_timing('load_egm96', load_time_96)
    results.add_timing('load_egm2008', load_time_2008)
    
    # Compute geoid heights
    print("Computing geoid heights...")
    start_time = time.time()
    geoid_96 = egm96.compute_geoid_height(lat_2d, lon_2d, max_degree=180)
    compute_time_96 = time.time() - start_time
    
    start_time = time.time()
    geoid_2008 = egm2008.compute_geoid_height(lat_2d, lon_2d, max_degree=180)
    compute_time_2008 = time.time() - start_time
    
    results.add_timing('compute_geoid_egm96', compute_time_96)
    results.add_timing('compute_geoid_egm2008', compute_time_2008)
    
    # Compute differences
    geoid_diff = geoid_2008 - geoid_96
    
    results.add_metric('geoid_diff_mean_m', np.mean(geoid_diff))
    results.add_metric('geoid_diff_std_m', np.std(geoid_diff))
    results.add_metric('geoid_diff_max_m', np.max(np.abs(geoid_diff)))
    results.add_metric('geoid_96_range_m', np.ptp(geoid_96))
    results.add_metric('geoid_2008_range_m', np.ptp(geoid_2008))
    
    print(f"\nResults:")
    print(f"  EGM96 geoid range: {np.ptp(geoid_96):.2f} m")
    print(f"  EGM2008 geoid range: {np.ptp(geoid_2008):.2f} m")
    print(f"  Mean difference: {np.mean(geoid_diff):.3f} m")
    print(f"  Std difference: {np.std(geoid_diff):.3f} m")
    print(f"  Max difference: {np.max(np.abs(geoid_diff)):.3f} m")
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # EGM96
    im1 = axes[0, 0].contourf(lon, lat, geoid_96, levels=20, cmap='terrain')
    axes[0, 0].set_title('EGM96 Geoid (m)')
    axes[0, 0].set_ylabel('Latitude')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # EGM2008
    im2 = axes[0, 1].contourf(lon, lat, geoid_2008, levels=20, cmap='terrain')
    axes[0, 1].set_title('EGM2008 Geoid (m)')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Difference
    im3 = axes[1, 0].contourf(lon, lat, geoid_diff, levels=20, cmap='RdBu_r')
    axes[1, 0].set_title('Difference (EGM2008 - EGM96) (m)')
    axes[1, 0].set_xlabel('Longitude')
    axes[1, 0].set_ylabel('Latitude')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Histogram of differences
    axes[1, 1].hist(geoid_diff.flatten(), bins=30, edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Geoid Difference (m)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title(f'Distribution (σ={np.std(geoid_diff):.3f} m)')
    axes[1, 1].axvline(np.mean(geoid_diff), color='red', linestyle='--',
                      label=f'Mean: {np.mean(geoid_diff):.3f}')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = output_dir / 'benchmark3_model_comparison.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {plot_file}")
    plt.close()
    
    return results


def benchmark_4_cross_validation(output_dir: Path) -> BenchmarkResults:
    """
    Benchmark 4: Cross-validation of background removal.
    
    Tests consistency of corrections across different data subsets.
    """
    print("\n" + "="*60)
    print("BENCHMARK 4: Cross-Validation")
    print("="*60)
    
    results = BenchmarkResults('cross_validation')
    
    # Generate synthetic dataset
    n_points = 500
    lat = np.random.uniform(35, 45, n_points)
    lon = np.random.uniform(-125, -115, n_points)
    
    # Synthetic signal
    true_anomaly = 20 * np.sin(lat/5) * np.cos(lon/10)
    background = 50 * (lat - 40) / 10  # Regional trend
    noise = np.random.randn(n_points) * 3
    
    observed = true_anomaly + background + noise + 980000
    
    # K-fold cross-validation
    k_folds = 5
    fold_size = n_points // k_folds
    
    residuals_by_fold = []
    predictions_by_fold = []
    
    print(f"\nPerforming {k_folds}-fold cross-validation...")
    
    for fold in range(k_folds):
        # Split data
        test_idx = slice(fold * fold_size, (fold + 1) * fold_size)
        train_idx = np.concatenate([
            np.arange(0, fold * fold_size),
            np.arange((fold + 1) * fold_size, n_points)
        ])
        
        # Fit regional trend on training data
        train_lat = lat[train_idx]
        train_lon = lon[train_idx]
        train_obs = observed[train_idx]
        
        # Simple planar fit: g = a + b*lat + c*lon
        A = np.column_stack([np.ones(len(train_lat)), train_lat, train_lon])
        coeffs = np.linalg.lstsq(A, train_obs, rcond=None)[0]
        
        # Predict on test data
        test_lat = lat[test_idx]
        test_lon = lon[test_idx]
        test_obs = observed[test_idx]
        
        A_test = np.column_stack([np.ones(len(test_lat)), test_lat, test_lon])
        predicted_background = A_test @ coeffs
        
        # Residuals
        residuals = test_obs - predicted_background
        
        residuals_by_fold.append(residuals)
        predictions_by_fold.append(predicted_background)
        
        print(f"  Fold {fold+1}: RMS residual = {np.sqrt(np.mean(residuals**2)):.3f} mGal")
    
    # Combine all residuals
    all_residuals = np.concatenate(residuals_by_fold)
    
    # Compute cross-validation metrics
    rms_cv = np.sqrt(np.mean(all_residuals**2))
    mae_cv = np.mean(np.abs(all_residuals))
    std_cv = np.std(all_residuals)
    
    results.add_metric('rms_cross_validation_mGal', rms_cv)
    results.add_metric('mae_cross_validation_mGal', mae_cv)
    results.add_metric('std_cross_validation_mGal', std_cv)
    results.add_metric('n_folds', k_folds)
    
    # Compare to noise level
    noise_estimate = np.std(noise)
    results.add_metric('true_noise_level_mGal', noise_estimate)
    results.add_metric('noise_estimation_error', abs(std_cv - noise_estimate))
    
    print(f"\nCross-validation results:")
    print(f"  RMS residual: {rms_cv:.3f} mGal")
    print(f"  MAE: {mae_cv:.3f} mGal")
    print(f"  Std: {std_cv:.3f} mGal")
    print(f"  True noise level: {noise_estimate:.3f} mGal")
    print(f"  Estimation error: {abs(std_cv - noise_estimate):.3f} mGal")
    
    # Create plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Residuals by fold
    fold_positions = []
    fold_data = []
    for i, residuals in enumerate(residuals_by_fold):
        fold_positions.extend([i+1] * len(residuals))
        fold_data.extend(residuals - 980000)
    
    axes[0, 0].violinplot([r - 980000 for r in residuals_by_fold], 
                          positions=range(1, k_folds+1),
                          showmeans=True, showmedians=True)
    axes[0, 0].set_xlabel('Fold')
    axes[0, 0].set_ylabel('Residual (mGal)')
    axes[0, 0].set_title('Residuals by Fold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Residual histogram
    axes[0, 1].hist(all_residuals - 980000, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Residual (mGal)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title(f'Combined Residuals (RMS={rms_cv:.3f})')
    axes[0, 1].axvline(0, color='red', linestyle='--')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(all_residuals - 980000, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Metrics summary
    axes[1, 1].axis('off')
    summary = f"""
    CROSS-VALIDATION SUMMARY
    
    Configuration:
      K-folds: {k_folds}
      Total points: {n_points}
      Points per fold: {fold_size}
    
    Metrics:
      RMS: {rms_cv:.3f} mGal
      MAE: {mae_cv:.3f} mGal
      Std: {std_cv:.3f} mGal
    
    Validation:
      True noise: {noise_estimate:.3f} mGal
      Estimated: {std_cv:.3f} mGal
      Error: {abs(std_cv - noise_estimate):.3f} mGal
    
    Quality: {'PASS' if abs(std_cv - noise_estimate) < 0.5 else 'REVIEW'}
    """
    axes[1, 1].text(0.1, 0.5, summary, fontsize=10, family='monospace',
                   verticalalignment='center')
    
    plt.tight_layout()
    plot_file = output_dir / 'benchmark4_cross_validation.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {plot_file}")
    plt.close()
    
    return results


def benchmark_5_synthetic_recovery(output_dir: Path) -> BenchmarkResults:
    """
    Benchmark 5: Recovery of known synthetic anomalies.
    
    Tests ability to recover true signal after removing backgrounds.
    """
    print("\n" + "="*60)
    print("BENCHMARK 5: Synthetic Anomaly Recovery")
    print("="*60)
    
    results = BenchmarkResults('synthetic_recovery')
    
    # Create synthetic scenario: buried density anomaly
    n = 80
    x = np.linspace(-10, 10, n)  # km
    y = np.linspace(-10, 10, n)  # km
    xx, yy = np.meshgrid(x, y)
    
    # True anomaly: Gaussian body
    anomaly_x, anomaly_y = 0, 0  # Center
    anomaly_amplitude = 50  # mGal
    anomaly_width = 3  # km
    
    true_anomaly = anomaly_amplitude * np.exp(
        -((xx - anomaly_x)**2 + (yy - anomaly_y)**2) / (2 * anomaly_width**2)
    )
    
    # Background signals
    regional_trend = 0.5 * xx + 0.3 * yy  # Regional gradient
    seasonal_effect = 2 * np.sin(2*np.pi*xx/20)  # Seasonal-like pattern
    noise = np.random.randn(n, n) * 1.5
    
    # Observed signal
    observed = true_anomaly + regional_trend + seasonal_effect + noise
    
    print(f"\nSynthetic model:")
    print(f"  Grid: {n}x{n}")
    print(f"  True anomaly amplitude: {anomaly_amplitude} mGal")
    print(f"  Anomaly width: {anomaly_width} km")
    print(f"  Regional trend: Yes")
    print(f"  Seasonal effect: Yes")
    print(f"  Noise level: 1.5 mGal")
    
    # Remove backgrounds
    start_time = time.time()
    
    # Step 1: Remove regional trend (plane fit)
    A = np.column_stack([
        np.ones(xx.size),
        xx.flatten(),
        yy.flatten()
    ])
    coeffs = np.linalg.lstsq(A, observed.flatten(), rcond=None)[0]
    regional_fit = (coeffs[0] + coeffs[1] * xx + coeffs[2] * yy)
    
    after_regional = observed - regional_fit
    
    # Step 2: Remove seasonal (FFT filtering)
    from scipy.fft import fft2, ifft2, fftshift
    
    fft_data = fft2(after_regional)
    fft_shift = fftshift(fft_data)
    
    # High-pass filter (remove long wavelengths)
    center = n // 2
    y_freq, x_freq = np.ogrid[-center:n-center, -center:n-center]
    freq_radius = np.sqrt(x_freq**2 + y_freq**2)
    
    # Keep frequencies > cutoff
    cutoff = n // 10
    high_pass = freq_radius > cutoff
    
    fft_filtered = fft_shift * high_pass
    fft_filtered = np.fft.ifftshift(fft_filtered)
    recovered = np.real(ifft2(fft_filtered))
    
    duration = time.time() - start_time
    results.add_timing('background_removal', duration)
    
    # Compute recovery metrics
    # Normalize for comparison
    true_norm = true_anomaly / np.max(np.abs(true_anomaly))
    recovered_norm = recovered / np.max(np.abs(recovered))
    
    # Correlation
    correlation = np.corrcoef(true_norm.flatten(), recovered_norm.flatten())[0, 1]
    
    # RMS error
    rms_error = np.sqrt(np.mean((recovered - true_anomaly)**2))
    
    # Signal recovery percentage
    signal_power_true = np.sum(true_anomaly**2)
    signal_power_recovered = np.sum(recovered**2)
    recovery_ratio = signal_power_recovered / signal_power_true
    
    results.add_metric('correlation_with_true', correlation)
    results.add_metric('rms_error_mGal', rms_error)
    results.add_metric('signal_recovery_ratio', recovery_ratio)
    results.add_metric('peak_amplitude_true_mGal', np.max(true_anomaly))
    results.add_metric('peak_amplitude_recovered_mGal', np.max(recovered))
    
    print(f"\nRecovery results:")
    print(f"  Correlation with true: {correlation:.4f}")
    print(f"  RMS error: {rms_error:.3f} mGal")
    print(f"  Signal recovery: {recovery_ratio*100:.1f}%")
    print(f"  True peak: {np.max(true_anomaly):.1f} mGal")
    print(f"  Recovered peak: {np.max(recovered):.1f} mGal")
    
    # Create comprehensive plot
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # True anomaly
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.contourf(xx, yy, true_anomaly, levels=20, cmap='RdBu_r')
    ax1.set_title('True Anomaly')
    ax1.set_ylabel('Y (km)')
    plt.colorbar(im1, ax=ax1, label='mGal')
    
    # Observed (with backgrounds)
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.contourf(xx, yy, observed, levels=20, cmap='RdBu_r')
    ax2.set_title('Observed (with backgrounds)')
    plt.colorbar(im2, ax=ax2, label='mGal')
    
    # Recovered
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.contourf(xx, yy, recovered, levels=20, cmap='RdBu_r')
    ax3.set_title(f'Recovered (r={correlation:.3f})')
    plt.colorbar(im3, ax=ax3, label='mGal')
    
    # Regional trend
    ax4 = fig.add_subplot(gs[1, 0])
    im4 = ax4.contourf(xx, yy, regional_fit, levels=20, cmap='viridis')
    ax4.set_title('Regional Trend (Removed)')
    ax4.set_ylabel('Y (km)')
    plt.colorbar(im4, ax=ax4, label='mGal')
    
    # After regional removal
    ax5 = fig.add_subplot(gs[1, 1])
    im5 = ax5.contourf(xx, yy, after_regional, levels=20, cmap='RdBu_r')
    ax5.set_title('After Regional Removal')
    plt.colorbar(im5, ax=ax5, label='mGal')
    
    # Recovery error
    ax6 = fig.add_subplot(gs[1, 2])
    error = recovered - true_anomaly
    im6 = ax6.contourf(xx, yy, error, levels=20, cmap='seismic')
    ax6.set_title(f'Recovery Error (RMS={rms_error:.2f})')
    plt.colorbar(im6, ax=ax6, label='mGal')
    
    # Profile comparison
    ax7 = fig.add_subplot(gs[2, :])
    center_idx = n // 2
    ax7.plot(x, true_anomaly[center_idx, :], 'k-', linewidth=2, label='True')
    ax7.plot(x, observed[center_idx, :], 'r--', linewidth=1.5, label='Observed')
    ax7.plot(x, recovered[center_idx, :], 'b:', linewidth=2, label='Recovered')
    ax7.set_xlabel('X (km)')
    ax7.set_ylabel('Gravity (mGal)')
    ax7.set_title('Central Profile Comparison')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = output_dir / 'benchmark5_synthetic_recovery.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {plot_file}")
    plt.close()
    
    return results


def generate_summary_report(all_results: List[BenchmarkResults], output_dir: Path):
    """Generate comprehensive summary report."""
    print("\n" + "="*60)
    print("GENERATING SUMMARY REPORT")
    print("="*60)
    
    # Compile all metrics
    summary_data = {
        'benchmarks': {},
        'overall_assessment': {},
        'timestamp': datetime.now().isoformat()
    }
    
    for result in all_results:
        summary_data['benchmarks'][result.name] = {
            'metrics': result.metrics,
            'timings': result.timings
        }
    
    # Overall assessment
    # Define quality thresholds
    quality_checks = {
        'temporal_stability': {
            'metric': 'seasonal_removal_efficiency',
            'threshold': 0.90,
            'passed': False
        },
        'spatial_coherence': {
            'metric': 'correlation_reduction_factor',
            'threshold': 0.70,
            'passed': False
        },
        'cross_validation': {
            'metric': 'noise_estimation_error',
            'threshold': 0.5,
            'passed': False
        },
        'synthetic_recovery': {
            'metric': 'correlation_with_true',
            'threshold': 0.85,
            'passed': False
        }
    }
    
    for result in all_results:
        if result.name in quality_checks:
            check = quality_checks[result.name]
            metric_name = check['metric']
            
            if metric_name in result.metrics:
                value = result.metrics[metric_name]
                if result.name == 'cross_validation':
                    check['passed'] = value < check['threshold']
                else:
                    check['passed'] = value >= check['threshold']
    
    summary_data['overall_assessment'] = quality_checks
    
    # Save summary
    summary_file = output_dir / 'benchmark_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\nSummary saved: {summary_file}")
    
    # Print quality assessment
    print("\n" + "="*60)
    print("QUALITY ASSESSMENT")
    print("="*60)
    
    n_passed = sum(1 for check in quality_checks.values() if check['passed'])
    n_total = len(quality_checks)
    
    for name, check in quality_checks.items():
        status = "PASS ✓" if check['passed'] else "REVIEW ✗"
        print(f"  {name:25s}: {status}")
    
    print(f"\nOverall: {n_passed}/{n_total} benchmarks passed")
    
    if n_passed == n_total:
        print("\n✓ All benchmarks PASSED - Background removal is performing well!")
    elif n_passed >= n_total * 0.75:
        print("\n⚠ Most benchmarks passed - Background removal is acceptable")
    else:
        print("\n✗ Several benchmarks failed - Review background removal methods")


def main():
    """Run all benchmarks."""
    print("\n" + "="*60)
    print("GEOPHYSICS MODULE - BACKGROUND REMOVAL BENCHMARKS")
    print("="*60)
    print("\nRunning comprehensive benchmark suite to evaluate")
    print("Earth model corrections and background removal effectiveness.")
    
    # Setup output directory
    output_dir = Path('/home/claude/benchmarks')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run all benchmarks
    all_results = []
    
    try:
        result1 = benchmark_1_temporal_stability(output_dir)
        result1.print_summary()
        result1.save(output_dir)
        all_results.append(result1)
    except Exception as e:
        print(f"Benchmark 1 failed: {e}")
    
    try:
        result2 = benchmark_2_spatial_coherence(output_dir)
        result2.print_summary()
        result2.save(output_dir)
        all_results.append(result2)
    except Exception as e:
        print(f"Benchmark 2 failed: {e}")
    
    try:
        result3 = benchmark_3_model_comparison(output_dir)
        result3.print_summary()
        result3.save(output_dir)
        all_results.append(result3)
    except Exception as e:
        print(f"Benchmark 3 failed: {e}")
    
    try:
        result4 = benchmark_4_cross_validation(output_dir)
        result4.print_summary()
        result4.save(output_dir)
        all_results.append(result4)
    except Exception as e:
        print(f"Benchmark 4 failed: {e}")
    
    try:
        result5 = benchmark_5_synthetic_recovery(output_dir)
        result5.print_summary()
        result5.save(output_dir)
        all_results.append(result5)
    except Exception as e:
        print(f"Benchmark 5 failed: {e}")
    
    # Generate summary report
    if all_results:
        generate_summary_report(all_results, output_dir)
    
    print("\n" + "="*60)
    print("BENCHMARKING COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {output_dir}")
    print("Review plots and JSON files for detailed analysis.")


if __name__ == '__main__':
    main()
