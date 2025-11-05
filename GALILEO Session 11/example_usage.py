#!/usr/bin/env python3
"""
Example: Using the Benchmarking Suite
======================================
Demonstrates how to use the verification and benchmarking harness.
"""

import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from bench import (
    RegressionDatasets,
    SpatialResolutionMetrics,
    LocalizationMetrics,
    PerformanceMetrics,
    CoverageAnalyzer
)


def example_1_dataset_generation():
    """Example 1: Generate and verify regression datasets."""
    print("=" * 70)
    print("Example 1: Dataset Generation")
    print("=" * 70 + "\n")
    
    # Create dataset manager
    datasets = RegressionDatasets()
    
    # Generate all datasets
    print("Generating regression datasets...")
    datasets._generate_all_datasets()
    
    # Verify datasets
    print("\nVerifying datasets...")
    if datasets.verify_datasets():
        print("✅ All datasets verified!")
    else:
        print("❌ Some datasets missing!")
    
    # Show dataset info
    info = datasets.get_dataset_info()
    print("\nAvailable dataset categories:")
    for category, dataset_list in info.items():
        print(f"\n{category}:")
        for ds in dataset_list:
            print(f"  - {ds}")


def example_2_spatial_metrics():
    """Example 2: Compute spatial resolution metrics."""
    print("\n" + "=" * 70)
    print("Example 2: Spatial Resolution Metrics")
    print("=" * 70 + "\n")
    
    # Load test data
    datasets = RegressionDatasets()
    point_source = datasets.load_point_source()
    
    # Initialize metrics
    metrics = SpatialResolutionMetrics()
    
    # Compute PSF
    print("Computing Point Spread Function...")
    psf = metrics.compute_psf(point_source)
    print(f"  PSF shape: {psf.shape}")
    
    # Compute FWHM
    fwhm = metrics.compute_fwhm(psf, pixel_size_km=1.0)
    print(f"  FWHM: {fwhm:.2f} km")
    
    # Compute resolution
    resolution = fwhm / 2.355  # Convert to sigma
    print(f"  Resolution: {resolution:.2f} km")
    
    # Compare with gold standard
    gold_psf = datasets.load_gold_output('psf')
    error = np.mean(np.abs(psf[:gold_psf.shape[0], :gold_psf.shape[1]] - gold_psf))
    print(f"  PSF error vs gold: {error:.4f}")
    
    # Evaluate
    if fwhm < 5.0:
        print("  ✅ PASS: Resolution meets < 5 km threshold")
    elif fwhm < 7.5:
        print("  ⚠️  WARN: Resolution acceptable but not optimal")
    else:
        print("  ❌ FAIL: Resolution exceeds threshold")


def example_3_localization_metrics():
    """Example 3: Compute localization metrics."""
    print("\n" + "=" * 70)
    print("Example 3: Localization Metrics")
    print("=" * 70 + "\n")
    
    # Load test data
    datasets = RegressionDatasets()
    anomaly_data = datasets.load_localization_test()
    true_centroids = datasets.load_gold_output('centroids')
    
    # Initialize metrics
    metrics = LocalizationMetrics()
    
    # Compute centroids
    print("Computing anomaly centroids...")
    computed_centroids = metrics.compute_centroids(anomaly_data)
    print(f"  Detected {len(computed_centroids)} anomalies")
    print(f"  Expected {len(true_centroids)} anomalies")
    
    # Compute position errors
    errors = metrics.compute_position_errors(
        computed_centroids, 
        true_centroids, 
        pixel_size_km=1.0
    )
    
    print(f"\nPosition errors:")
    print(f"  Mean error: {np.mean(errors):.2f} km")
    print(f"  Std error:  {np.std(errors):.2f} km")
    print(f"  Max error:  {np.max(errors):.2f} km")
    print(f"  RMS error:  {np.sqrt(np.mean(errors**2)):.2f} km")
    
    # Evaluate
    mean_error = np.mean(errors)
    if mean_error < 2.0:
        print("  ✅ PASS: Localization error < 2 km threshold")
    elif mean_error < 3.0:
        print("  ⚠️  WARN: Localization error acceptable")
    else:
        print("  ❌ FAIL: Localization error exceeds threshold")


def example_4_performance_benchmarks():
    """Example 4: Run performance benchmarks."""
    print("\n" + "=" * 70)
    print("Example 4: Performance Benchmarks")
    print("=" * 70 + "\n")
    
    # Load test data
    datasets = RegressionDatasets()
    
    # Initialize metrics
    metrics = PerformanceMetrics()
    
    # Benchmark forward modeling
    print("Benchmarking forward modeling...")
    model = datasets.load_standard_model()
    runtime_ms = metrics.benchmark_forward_model(model)
    throughput = metrics.compute_throughput(model, runtime_ms)
    
    print(f"  Runtime: {runtime_ms:.2f} ms")
    print(f"  Throughput: {throughput:.0f} cells/sec")
    
    if runtime_ms < 100:
        print("  ✅ PASS: Runtime < 100 ms threshold")
    else:
        print("  ⚠️  WARN: Runtime exceeds 100 ms")
    
    # Benchmark inversion
    print("\nBenchmarking inversion...")
    inv_data = datasets.load_inversion_test()
    inv_runtime, iterations = metrics.benchmark_inversion(inv_data)
    
    print(f"  Runtime: {inv_runtime:.2f} ms")
    print(f"  Iterations: {iterations}")
    print(f"  Time per iteration: {inv_runtime/iterations:.2f} ms")
    
    if inv_runtime < 1000:
        print("  ✅ PASS: Inversion time < 1000 ms threshold")
    else:
        print("  ⚠️  WARN: Inversion time exceeds 1000 ms")
    
    # Benchmark ML inference
    print("\nBenchmarking ML inference...")
    ml_input = datasets.load_ml_test_input()
    ml_runtime, batch_size = metrics.benchmark_ml_inference(ml_input)
    latency = ml_runtime / batch_size
    
    print(f"  Runtime: {ml_runtime:.2f} ms")
    print(f"  Batch size: {batch_size}")
    print(f"  Latency per sample: {latency:.2f} ms")
    print(f"  Throughput: {1000.0/latency:.0f} samples/sec")
    
    if latency < 50:
        print("  ✅ PASS: Latency < 50 ms threshold")
    else:
        print("  ⚠️  WARN: Latency exceeds 50 ms")


def example_5_coverage_analysis():
    """Example 5: Analyze code coverage."""
    print("\n" + "=" * 70)
    print("Example 5: Coverage Analysis")
    print("=" * 70 + "\n")
    
    # Initialize analyzer
    analyzer = CoverageAnalyzer()
    
    print("Running coverage analysis...")
    coverage = analyzer.analyze_coverage()
    
    print(f"\n📊 Overall Coverage: {coverage['overall']:.1f}%\n")
    
    print("Module Coverage:")
    for module, cov in coverage['modules'].items():
        if cov >= 85:
            symbol = "✅"
        elif cov >= 70:
            symbol = "⚠️"
        else:
            symbol = "❌"
        
        print(f"  {symbol} {module:30s}: {cov:5.1f}%")
    
    # Check critical modules
    critical_threshold = 85.0
    critical_modules = analyzer.critical_modules
    critical_pass = all(
        coverage['modules'].get(m, 0) >= critical_threshold 
        for m in critical_modules
    )
    
    print(f"\n{'=' * 70}")
    if critical_pass:
        print(f"✅ All critical modules meet {critical_threshold}% threshold")
    else:
        print(f"⚠️  Some critical modules below {critical_threshold}% threshold")
        below = [m for m in critical_modules 
                if coverage['modules'].get(m, 0) < critical_threshold]
        print(f"   Modules needing attention: {', '.join(below)}")


def example_6_full_benchmark_suite():
    """Example 6: Run full benchmark suite."""
    print("\n" + "=" * 70)
    print("Example 6: Full Benchmark Suite")
    print("=" * 70 + "\n")
    
    print("This would run the complete benchmark suite:")
    print("  python bench.py --suite all --report html --coverage")
    print("\nSuites included:")
    print("  - Spatial Resolution (4 tests)")
    print("  - Localization (4 tests)")
    print("  - Performance (4 tests)")
    print("\nTotal: 12 comprehensive tests")
    print("\nExample output:")
    print("""
📍 Running Spatial Resolution Suite...
----------------------------------------------------------------------
✅ PSF Characterization              |  0.142s | PASS
✅ Frequency Response                |  0.089s | PASS
✅ Resolution Recovery               |  0.234s | PASS
⚠️  Anomaly Separation               |  0.156s | WARN

🎯 Running Localization Suite...
----------------------------------------------------------------------
✅ Centroid Localization             |  0.098s | PASS
✅ Boundary Detection                |  0.145s | PASS
✅ Multi-Target Localization         |  0.187s | PASS
⚠️  Depth Estimation                 |  0.112s | WARN

⚡ Running Performance Suite...
----------------------------------------------------------------------
✅ Forward Modeling                  |  0.045s | PASS
✅ Inversion Speed                   |  0.523s | PASS
✅ ML Inference                      |  0.034s | PASS
✅ Memory Efficiency                 |  0.167s | PASS

======================================================================
📊 BENCHMARK SUMMARY
======================================================================

Total Tests:  12
✅ Passed:    10 (83.3%)
⚠️  Warnings:  2 (16.7%)
❌ Failed:    0 (0.0%)

Total Runtime: 1.73s
    """)


def example_7_custom_test():
    """Example 7: Create custom benchmark test."""
    print("\n" + "=" * 70)
    print("Example 7: Custom Benchmark Test")
    print("=" * 70 + "\n")
    
    print("Creating a custom benchmark test...")
    
    # Generate custom test data
    print("\n1. Generate custom test data:")
    size = 128
    y, x = np.ogrid[:size, :size]
    
    # Create two anomalies at known separation
    separation_km = 8.0  # km
    center1 = (size//2 - separation_km//2, size//2)
    center2 = (size//2 + separation_km//2, size//2)
    
    sigma = 5.0
    anomaly1 = np.exp(-((x - center1[1])**2 + (y - center1[0])**2) / (2 * sigma**2))
    anomaly2 = np.exp(-((x - center2[1])**2 + (y - center2[0])**2) / (2 * sigma**2))
    
    test_data = anomaly1 + anomaly2
    print(f"   Created test data with {separation_km} km separation")
    
    # Run custom test
    print("\n2. Run custom spatial resolution test:")
    metrics = SpatialResolutionMetrics()
    
    # Recover anomalies
    recovered = metrics.recover_anomalies(test_data)
    print("   Recovered anomalies using deconvolution")
    
    # Compute separation
    measured_separation = metrics.compute_separation(recovered)
    print(f"   Measured separation: {measured_separation:.2f} km")
    print(f"   True separation: {separation_km:.2f} km")
    print(f"   Error: {abs(measured_separation - separation_km):.2f} km")
    
    # Evaluate
    error = abs(measured_separation - separation_km)
    if error < 1.0:
        print("   ✅ PASS: Separation error < 1 km")
    else:
        print("   ⚠️  WARN: Separation error > 1 km")


def main():
    """Run all examples."""
    print("\n" + "🔬" * 35)
    print("VERIFICATION & BENCHMARKING SUITE - EXAMPLES")
    print("🔬" * 35 + "\n")
    
    try:
        example_1_dataset_generation()
        example_2_spatial_metrics()
        example_3_localization_metrics()
        example_4_performance_benchmarks()
        example_5_coverage_analysis()
        example_6_full_benchmark_suite()
        example_7_custom_test()
        
        print("\n" + "=" * 70)
        print("✅ All examples completed successfully!")
        print("=" * 70 + "\n")
        
        print("Next steps:")
        print("  1. Run full benchmark: python bench.py --suite all")
        print("  2. Generate reports: python bench.py --report html")
        print("  3. Check coverage: python bench.py --coverage")
        print("  4. Review docs: docs/verification.md")
        print()
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
