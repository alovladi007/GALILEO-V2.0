#!/usr/bin/env python3
"""
Comprehensive Demonstration of Session 9 Capabilities

This script demonstrates all calibration and system identification tools
in a realistic orbit determination scenario.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from calibration import (AllanDeviation, CrossSpectralDensity, WhitenessTest,
                        plot_allan_deviation)
from system_id import (DragCoefficientEstimator, SolarPressureEstimator,
                       EmpiricalAccelerationModel, ResidualAnalyzer)
from cal_maneuvers import (CalibrationManeuverGenerator, SyntheticOrbitGenerator,
                           create_example_scenario, ManeuverType)


def print_section(title):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(title.center(70))
    print("=" * 70)


def demonstrate_noise_characterization():
    """Demonstrate noise characterization tools."""
    print_section("NOISE CHARACTERIZATION")
    
    np.random.seed(42)
    
    # Simulate sensor with mixed noise
    n = 5000
    rate = 100.0  # Hz
    t = np.arange(n) / rate
    
    # Components
    white = np.random.randn(n) * 0.1
    random_walk = np.cumsum(np.random.randn(n) * 0.01)
    flicker = np.random.randn(n) * 0.05 * np.sqrt(np.arange(1, n+1))
    
    sensor_data = white + random_walk * 0.5
    
    print("\n1. Allan Deviation Analysis")
    print("-" * 70)
    
    adev_calc = AllanDeviation(sensor_data, rate, overlapping=True)
    taus, adev = adev_calc.compute()
    
    noise_id = adev_calc.identify_noise_type(taus, adev)
    print(f"   Noise type: {noise_id['type']}")
    print(f"   Slope: {noise_id['slope']:.3f} (expected: {noise_id['expected_slope']:.1f})")
    
    print("\n2. Power Spectral Density")
    print("-" * 70)
    
    csd_calc = CrossSpectralDensity(nperseg=256)
    freqs, psd = csd_calc.compute_psd(sensor_data, rate)
    
    print(f"   Frequency range: {freqs[1]:.2e} to {freqs[-1]:.2e} Hz")
    print(f"   Peak PSD: {np.max(psd):.2e}")
    print(f"   Mean PSD: {np.mean(psd):.2e}")
    
    print("\n3. Whiteness Assessment")
    print("-" * 70)
    
    white_results = WhitenessTest.comprehensive_test(white)
    print(f"   Pure white noise:")
    print(f"     - Ljung-Box p-value: {white_results['ljung_box']['p_value']:.4f}")
    print(f"     - Runs test p-value: {white_results['runs']['p_value']:.4f}")
    print(f"     - Durbin-Watson: {white_results['durbin_watson']:.4f}")
    print(f"     - Assessment: {'WHITE ✓' if white_results['overall_white'] else 'NOT WHITE ✗'}")
    
    sensor_results = WhitenessTest.comprehensive_test(sensor_data)
    print(f"\n   Mixed sensor noise:")
    print(f"     - Ljung-Box p-value: {sensor_results['ljung_box']['p_value']:.4f}")
    print(f"     - Runs test p-value: {sensor_results['runs']['p_value']:.4f}")
    print(f"     - Durbin-Watson: {sensor_results['durbin_watson']:.4f}")
    print(f"     - Assessment: {'WHITE ✓' if sensor_results['overall_white'] else 'NOT WHITE ✗'}")


def demonstrate_calibration_design():
    """Demonstrate calibration maneuver design."""
    print_section("CALIBRATION MANEUVER DESIGN")
    
    orbital_period = 5400.0  # 90 minutes
    generator = CalibrationManeuverGenerator(orbital_period)
    
    print("\n1. Delta-V Sequence (Drag Calibration)")
    print("-" * 70)
    
    dv_maneuvers = generator.generate_delta_v_sequence(
        n_maneuvers=3,
        delta_v_range=(0.1, 1.0),
        spacing='optimal'
    )
    
    for i, m in enumerate(dv_maneuvers):
        dv_mag = np.linalg.norm(m.parameters['delta_v'])
        print(f"   Maneuver {i+1}:")
        print(f"     Time: {m.start_time/60:.1f} min")
        print(f"     Delta-V: {dv_mag:.3f} m/s")
        print(f"     Direction: {m.parameters['delta_v'] / dv_mag}")
    
    print("\n2. Ballistic Coefficient Changes (Drag/SRP Separation)")
    print("-" * 70)
    
    bc_maneuvers = generator.generate_ballistic_coefficient_changes(n_changes=3)
    
    for i, m in enumerate(bc_maneuvers):
        print(f"   Configuration {i+1}:")
        print(f"     Time: {m.start_time/60:.1f} - {m.end_time/60:.1f} min")
        print(f"     Area factor: {m.parameters['area_factor']:.1f}x")
        print(f"     Name: {m.parameters['configuration']}")
    
    print("\n3. Comprehensive Sequence")
    print("-" * 70)
    
    comprehensive = generator.generate_comprehensive_sequence()
    
    maneuver_types = {}
    for m in comprehensive:
        maneuver_types[m.type] = maneuver_types.get(m.type, 0) + 1
    
    print(f"   Total maneuvers: {len(comprehensive)}")
    print(f"   Total duration: {comprehensive[-1].end_time/3600:.2f} hours")
    for mtype, count in maneuver_types.items():
        print(f"     - {mtype.value}: {count}")


def demonstrate_parameter_estimation():
    """Demonstrate system identification."""
    print_section("PARAMETER ESTIMATION")
    
    np.random.seed(42)
    
    # Create synthetic scenario
    n = 300
    times = np.linspace(0, 2 * 5400, n)  # 3 hours
    
    print("\n1. Drag Coefficient Estimation")
    print("-" * 70)
    
    # True parameters
    true_cd = 2.25
    area_to_mass = 0.01  # m²/kg
    
    # Atmospheric conditions (LEO)
    altitude = 400e3 - 50e3 * np.sin(2 * np.pi * times / 5400)  # Varying altitude
    density = 1e-12 * np.exp(-altitude / 60e3)
    velocity = 7600 + 100 * np.sin(2 * np.pi * times / 5400)
    
    # Generate drag signal
    drag_mag = 0.5 * true_cd * area_to_mass * density * velocity**2
    drag_accel = drag_mag[:, np.newaxis] * np.array([[-1, -0.1, 0.05]])  # Mostly in -x
    
    # Add measurement noise
    noise_level = 2e-8
    residuals = drag_accel + np.random.randn(n, 3) * noise_level
    
    # Estimate CD
    cd_estimator = DragCoefficientEstimator(area_to_mass, initial_cd=2.0)
    cd_result = cd_estimator.estimate(times, residuals, density, velocity)
    
    cd_error = abs(cd_result.parameters[0] - true_cd)
    cd_sigma = cd_result.parameter_uncertainties()[0]
    
    print(f"   True CD: {true_cd:.4f}")
    print(f"   Estimated CD: {cd_result.parameters[0]:.4f} ± {cd_sigma:.4f}")
    print(f"   Error: {cd_error:.4f} ({cd_error/true_cd*100:.2f}%)")
    print(f"   Optimization: {cd_result.success}")
    
    # Analyze post-fit residuals
    post_fit_rms = np.sqrt(np.mean(cd_result.residuals**2))
    improvement = (1 - post_fit_rms / np.sqrt(np.mean(residuals**2))) * 100
    
    print(f"\n   Residual Analysis:")
    print(f"     Pre-fit RMS: {np.sqrt(np.mean(residuals**2)):.2e} m/s²")
    print(f"     Post-fit RMS: {post_fit_rms:.2e} m/s²")
    print(f"     Improvement: {improvement:.1f}%")
    
    # Test whiteness
    whiteness = WhitenessTest.comprehensive_test(cd_result.residuals.flatten())
    print(f"     Whiteness: {whiteness['overall_white']}")
    
    print("\n2. Solar Radiation Pressure Estimation")
    print("-" * 70)
    
    # True parameters
    true_cr = 1.75
    
    # Sun geometry
    sun_vectors = np.random.randn(n, 3)
    sun_vectors /= np.linalg.norm(sun_vectors, axis=1, keepdims=True)
    sun_distances = np.full(n, 1.496e11)  # 1 AU
    
    # Eclipse modeling (simple)
    earth_angle = np.arcsin(6371e3 / np.linalg.norm(times[:, np.newaxis] * [7e3, 0, 0], axis=1))
    sun_angle = np.arccos(np.sum(sun_vectors * [1, 0, 0], axis=1))
    shadow_factors = np.where(sun_angle > earth_angle + 0.1, 1.0, 0.0)
    
    # Generate SRP signal
    P_sun = 4.56e-6
    c = 299792458.0
    srp_mag = true_cr * area_to_mass * (P_sun / c) * shadow_factors
    srp_accel = srp_mag[:, np.newaxis] * (-sun_vectors)
    
    # Add noise
    residuals_srp = srp_accel + np.random.randn(n, 3) * 1e-9
    
    # Estimate CR
    cr_estimator = SolarPressureEstimator(area_to_mass, initial_cr=1.5)
    cr_result = cr_estimator.estimate(times, residuals_srp, sun_vectors,
                                     sun_distances, shadow_factors)
    
    cr_error = abs(cr_result.parameters[0] - true_cr)
    cr_sigma = cr_result.parameter_uncertainties()[0]
    
    print(f"   True CR: {true_cr:.4f}")
    print(f"   Estimated CR: {cr_result.parameters[0]:.4f} ± {cr_sigma:.4f}")
    print(f"   Error: {cr_error:.4f} ({cr_error/true_cr*100:.2f}%)")
    print(f"   Optimization: {cr_result.success}")
    print(f"   Sunlit observations: {np.sum(shadow_factors > 0.5)}/{n}")
    
    print("\n3. Empirical Acceleration Model")
    print("-" * 70)
    
    # Create systematic residuals (unmodeled forces)
    orbital_period = 5400.0
    t_norm = times / orbital_period
    
    true_systematic = np.zeros((n, 3))
    true_systematic[:, 0] = 1e-8 * (0.5 + 0.3 * np.sin(2*np.pi*t_norm))
    true_systematic[:, 1] = 1e-8 * (0.2 * np.cos(2*np.pi*t_norm) + 0.1 * t_norm)
    true_systematic[:, 2] = 1e-8 * (0.1 * np.sin(4*np.pi*t_norm))
    
    residuals_emp = true_systematic + np.random.randn(n, 3) * 5e-10
    
    # Fit empirical model
    emp_model = EmpiricalAccelerationModel(n_harmonics=2, polynomial_degree=1)
    emp_result = emp_model.fit(times, residuals_emp, orbital_period)
    
    print(f"   Model parameters: {len(emp_result.parameters)}")
    print(f"   Pre-fit RMS: {np.sqrt(np.mean(residuals_emp**2)):.2e} m/s²")
    print(f"   Post-fit RMS: {np.sqrt(np.mean(emp_result.residuals**2)):.2e} m/s²")
    print(f"   Variance reduction: {(1 - emp_result.cost / np.sum(residuals_emp**2)) * 100:.1f}%")


def demonstrate_residual_analysis():
    """Demonstrate residual analysis tools."""
    print_section("RESIDUAL ANALYSIS")
    
    np.random.seed(42)
    
    # Create filter residuals (should be white)
    n = 500
    good_residuals = np.random.randn(n, 3) * 0.01
    
    # Add some outliers
    outlier_idx = [50, 150, 300]
    good_residuals[outlier_idx] += np.random.randn(len(outlier_idx), 3) * 0.1
    
    print("\n1. Statistical Summary")
    print("-" * 70)
    
    analyzer = ResidualAnalyzer()
    stats = analyzer.compute_statistics(good_residuals)
    
    print(f"   Mean: {stats['mean']}")
    print(f"   Std: {stats['std']}")
    print(f"   RMS: {stats['rms']}")
    print(f"   Median: {stats['median']}")
    print(f"   MAD: {stats['mad']}")
    
    print("\n2. Outlier Detection")
    print("-" * 70)
    
    outliers = analyzer.detect_outliers(good_residuals, threshold=3.0)
    detected_outliers = np.where(outliers)[0]
    
    print(f"   Total outliers: {np.sum(outliers)} / {n}")
    print(f"   Outlier fraction: {np.sum(outliers)/n*100:.2f}%")
    print(f"   Detected at indices: {detected_outliers[:10]}...")  # First 10
    
    # Check if we found the injected outliers
    found = [idx in detected_outliers for idx in outlier_idx]
    print(f"   Injected outliers found: {sum(found)}/{len(outlier_idx)}")
    
    print("\n3. Autocorrelation Analysis")
    print("-" * 70)
    
    acf = analyzer.compute_autocorrelation(good_residuals[:, 0].flatten(), max_lag=20)
    
    print(f"   ACF values (first 10 lags):")
    for i in range(min(10, len(acf))):
        print(f"     Lag {i}: {acf[i]:+.4f}")
    
    # 95% confidence bands for white noise
    conf_band = 1.96 / np.sqrt(n)
    n_outside = np.sum(np.abs(acf[1:]) > conf_band)
    
    print(f"\n   95% confidence band: ±{conf_band:.4f}")
    print(f"   Values outside band: {n_outside}/{len(acf)-1}")
    
    print("\n4. Comprehensive Analysis")
    print("-" * 70)
    
    analysis = analyzer.comprehensive_analysis(good_residuals)
    
    print(f"   RMS: {analysis['statistics']['rms']}")
    print(f"   Outliers: {analysis['n_outliers']} ({analysis['outlier_fraction']*100:.1f}%)")
    print(f"   Max ACF (lag>0): {np.max(np.abs(analysis['autocorrelation'][1:])):.4f}")


def main():
    """Run complete demonstration."""
    print("\n" + "=" * 70)
    print("SESSION 9: COMPREHENSIVE CALIBRATION DEMONSTRATION".center(70))
    print("=" * 70)
    
    demonstrate_noise_characterization()
    demonstrate_calibration_design()
    demonstrate_parameter_estimation()
    demonstrate_residual_analysis()
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE".center(70))
    print("=" * 70)
    
    print("\nKey Takeaways:")
    print("  ✓ Noise characterization identifies sensor behavior")
    print("  ✓ Allan deviation reveals noise types and stability")
    print("  ✓ Calibration maneuvers enable parameter estimation")
    print("  ✓ CD and CR can be estimated from residuals")
    print("  ✓ Empirical models capture unmodeled forces")
    print("  ✓ Whiteness tests validate filter performance")
    print("  ✓ Residual analysis detects outliers and biases")
    
    print("\nFor more information:")
    print("  - Documentation: docs/calibration.md")
    print("  - Validation: python validate_calibration.py")
    print("  - Examples: python calibration.py, system_id.py, cal_maneuvers.py")


if __name__ == "__main__":
    main()
