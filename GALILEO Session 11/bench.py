#!/usr/bin/env python3
"""
Verification and Benchmarking Harness - bench.py
================================================
Main runner for regression testing, performance benchmarking, and verification
of the geophysical gravity gradiometry processing pipeline.

Usage:
    python bench.py --suite all                    # Run all benchmarks
    python bench.py --suite spatial                # Spatial resolution only
    python bench.py --suite localization           # Localization error only
    python bench.py --suite performance            # Runtime cost only
    python bench.py --coverage                     # Generate coverage report
    python bench.py --report html                  # Generate HTML report
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict

# Local imports
from bench.metrics import (
    SpatialResolutionMetrics,
    LocalizationMetrics,
    PerformanceMetrics,
    CoverageAnalyzer
)
from bench.datasets import RegressionDatasets


@dataclass
class BenchmarkResult:
    """Store results from a single benchmark."""
    name: str
    suite: str
    status: str  # 'PASS', 'FAIL', 'WARN'
    metrics: Dict
    runtime: float
    timestamp: str
    error: Optional[str] = None


class BenchmarkRunner:
    """Main benchmark orchestrator."""
    
    def __init__(self, output_dir: Path = Path("bench/reports")):
        """Initialize benchmark runner."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.datasets = RegressionDatasets()
        self.spatial_metrics = SpatialResolutionMetrics()
        self.localization_metrics = LocalizationMetrics()
        self.performance_metrics = PerformanceMetrics()
        self.coverage_analyzer = CoverageAnalyzer()
        
        self.results: List[BenchmarkResult] = []
        
    def run_suite(self, suite_name: str = "all") -> bool:
        """
        Run specified benchmark suite.
        
        Parameters
        ----------
        suite_name : str
            Suite to run: 'all', 'spatial', 'localization', 'performance'
            
        Returns
        -------
        bool
            True if all tests passed
        """
        print(f"\n{'='*70}")
        print(f"🔬 Geophysics Benchmarking Suite - {suite_name.upper()}")
        print(f"{'='*70}\n")
        
        suites = {
            'spatial': self._run_spatial_suite,
            'localization': self._run_localization_suite,
            'performance': self._run_performance_suite,
        }
        
        if suite_name == 'all':
            success = all(runner() for runner in suites.values())
        elif suite_name in suites:
            success = suites[suite_name]()
        else:
            print(f"❌ Unknown suite: {suite_name}")
            return False
            
        self._generate_summary()
        return success
    
    def _run_spatial_suite(self) -> bool:
        """Run spatial resolution benchmarks."""
        print("📍 Running Spatial Resolution Suite...")
        print("-" * 70)
        
        success = True
        
        # Test 1: Point spread function characterization
        success &= self._run_benchmark(
            "PSF Characterization",
            "spatial",
            self._test_psf_characterization
        )
        
        # Test 2: Spatial frequency response
        success &= self._run_benchmark(
            "Frequency Response",
            "spatial",
            self._test_frequency_response
        )
        
        # Test 3: Resolution recovery
        success &= self._run_benchmark(
            "Resolution Recovery",
            "spatial",
            self._test_resolution_recovery
        )
        
        # Test 4: Anomaly separation
        success &= self._run_benchmark(
            "Anomaly Separation",
            "spatial",
            self._test_anomaly_separation
        )
        
        return success
    
    def _run_localization_suite(self) -> bool:
        """Run localization error benchmarks."""
        print("\n🎯 Running Localization Suite...")
        print("-" * 70)
        
        success = True
        
        # Test 1: Centroid localization
        success &= self._run_benchmark(
            "Centroid Localization",
            "localization",
            self._test_centroid_localization
        )
        
        # Test 2: Boundary detection
        success &= self._run_benchmark(
            "Boundary Detection",
            "localization",
            self._test_boundary_detection
        )
        
        # Test 3: Multi-target localization
        success &= self._run_benchmark(
            "Multi-Target Localization",
            "localization",
            self._test_multi_target_localization
        )
        
        # Test 4: Depth estimation
        success &= self._run_benchmark(
            "Depth Estimation",
            "localization",
            self._test_depth_estimation
        )
        
        return success
    
    def _run_performance_suite(self) -> bool:
        """Run performance/runtime benchmarks."""
        print("\n⚡ Running Performance Suite...")
        print("-" * 70)
        
        success = True
        
        # Test 1: Forward modeling speed
        success &= self._run_benchmark(
            "Forward Modeling",
            "performance",
            self._test_forward_modeling_speed
        )
        
        # Test 2: Inversion convergence
        success &= self._run_benchmark(
            "Inversion Speed",
            "performance",
            self._test_inversion_speed
        )
        
        # Test 3: ML inference speed
        success &= self._run_benchmark(
            "ML Inference",
            "performance",
            self._test_ml_inference_speed
        )
        
        # Test 4: Memory efficiency
        success &= self._run_benchmark(
            "Memory Efficiency",
            "performance",
            self._test_memory_efficiency
        )
        
        return success
    
    def _run_benchmark(self, name: str, suite: str, test_func) -> bool:
        """Run a single benchmark test."""
        start = time.time()
        
        try:
            metrics = test_func()
            runtime = time.time() - start
            
            # Determine status based on metrics
            status = self._evaluate_status(metrics, suite)
            
            result = BenchmarkResult(
                name=name,
                suite=suite,
                status=status,
                metrics=metrics,
                runtime=runtime,
                timestamp=datetime.now().isoformat()
            )
            
            self.results.append(result)
            
            # Print result
            symbol = "✅" if status == "PASS" else "⚠️" if status == "WARN" else "❌"
            print(f"{symbol} {name:30s} | {runtime:6.3f}s | {status}")
            
            return status != "FAIL"
            
        except Exception as e:
            runtime = time.time() - start
            result = BenchmarkResult(
                name=name,
                suite=suite,
                status="FAIL",
                metrics={},
                runtime=runtime,
                timestamp=datetime.now().isoformat(),
                error=str(e)
            )
            self.results.append(result)
            print(f"❌ {name:30s} | {runtime:6.3f}s | FAIL: {e}")
            return False
    
    def _evaluate_status(self, metrics: Dict, suite: str) -> str:
        """Evaluate test status based on metrics."""
        thresholds = {
            'spatial': {
                'resolution_km': 5.0,      # <5km is PASS
                'psf_width': 10.0,          # <10km is PASS
                'frequency_response': 0.8,  # >0.8 is PASS
            },
            'localization': {
                'centroid_error_km': 2.0,   # <2km is PASS
                'boundary_error_km': 3.0,   # <3km is PASS
                'depth_error_pct': 15.0,    # <15% is PASS
            },
            'performance': {
                'runtime_ms': 100.0,        # <100ms is PASS
                'memory_mb': 500.0,         # <500MB is PASS
                'speedup': 5.0,             # >5x is PASS
            }
        }
        
        suite_thresholds = thresholds.get(suite, {})
        
        fails = []
        warns = []
        
        for key, threshold in suite_thresholds.items():
            if key in metrics:
                value = metrics[key]
                
                # Determine if lower or higher is better
                if 'error' in key.lower() or 'runtime' in key.lower() or 'memory' in key.lower():
                    # Lower is better
                    if value > threshold * 1.5:
                        fails.append(key)
                    elif value > threshold:
                        warns.append(key)
                else:
                    # Higher is better
                    if value < threshold * 0.5:
                        fails.append(key)
                    elif value < threshold:
                        warns.append(key)
        
        if fails:
            return "FAIL"
        elif warns:
            return "WARN"
        else:
            return "PASS"
    
    # =========================================================================
    # SPATIAL RESOLUTION TESTS
    # =========================================================================
    
    def _test_psf_characterization(self) -> Dict:
        """Characterize point spread function."""
        # Load test data
        point_source = self.datasets.load_point_source()
        gold_psf = self.datasets.load_gold_output('psf')
        
        # Compute PSF
        computed_psf = self.spatial_metrics.compute_psf(point_source)
        
        # Compare with gold standard
        error = np.mean(np.abs(computed_psf - gold_psf))
        fwhm = self.spatial_metrics.compute_fwhm(computed_psf)
        
        return {
            'psf_error': float(error),
            'psf_width': float(fwhm),  # km
            'resolution_km': float(fwhm / 2.35),  # Convert FWHM to sigma
        }
    
    def _test_frequency_response(self) -> Dict:
        """Test spatial frequency response."""
        # Load synthetic data with known frequencies
        freq_data = self.datasets.load_frequency_test()
        
        # Compute MTF (Modulation Transfer Function)
        mtf = self.spatial_metrics.compute_mtf(freq_data)
        
        # Key metrics
        mtf_50 = self.spatial_metrics.get_mtf_at_frequency(mtf, 0.5)
        nyquist_response = mtf[-1]
        
        return {
            'frequency_response': float(mtf_50),
            'nyquist_response': float(nyquist_response),
            'effective_resolution_km': float(1.0 / mtf_50) if mtf_50 > 0 else 999.0,
        }
    
    def _test_resolution_recovery(self) -> Dict:
        """Test resolution of closely spaced anomalies."""
        # Load twin anomaly test case
        twin_data = self.datasets.load_twin_anomalies()
        gold_output = self.datasets.load_gold_output('twin_anomalies')
        
        # Process and recover anomalies
        recovered = self.spatial_metrics.recover_anomalies(twin_data)
        
        # Compute separation metric
        separation = self.spatial_metrics.compute_separation(recovered)
        recovery_fidelity = np.corrcoef(recovered.flatten(), gold_output.flatten())[0, 1]
        
        return {
            'min_separation_km': float(separation),
            'recovery_fidelity': float(recovery_fidelity),
            'resolution_km': float(separation / 2.0),
        }
    
    def _test_anomaly_separation(self) -> Dict:
        """Test ability to separate overlapping anomalies."""
        # Load overlapping anomalies
        overlap_data = self.datasets.load_overlapping_anomalies()
        gold_separated = self.datasets.load_gold_output('separated_anomalies')
        
        # Perform separation
        separated = self.spatial_metrics.separate_anomalies(overlap_data)
        
        # Compare with gold standard
        rmse = np.sqrt(np.mean((separated - gold_separated)**2))
        cross_talk = self.spatial_metrics.compute_crosstalk(separated, gold_separated)
        
        return {
            'separation_rmse': float(rmse),
            'cross_talk_db': float(cross_talk),
            'separation_quality': float(1.0 / (1.0 + rmse)),
        }
    
    # =========================================================================
    # LOCALIZATION TESTS
    # =========================================================================
    
    def _test_centroid_localization(self) -> Dict:
        """Test centroid localization accuracy."""
        # Load test data with known centroids
        anomaly_data = self.datasets.load_localization_test()
        true_centroids = self.datasets.load_gold_output('centroids')
        
        # Compute centroids
        computed_centroids = self.localization_metrics.compute_centroids(anomaly_data)
        
        # Compute errors
        errors = self.localization_metrics.compute_position_errors(
            computed_centroids, true_centroids
        )
        
        return {
            'centroid_error_km': float(np.mean(errors)),
            'centroid_error_std': float(np.std(errors)),
            'max_error_km': float(np.max(errors)),
            'rms_error_km': float(np.sqrt(np.mean(errors**2))),
        }
    
    def _test_boundary_detection(self) -> Dict:
        """Test boundary detection accuracy."""
        # Load test data with known boundaries
        field_data = self.datasets.load_boundary_test()
        true_boundaries = self.datasets.load_gold_output('boundaries')
        
        # Detect boundaries
        detected_boundaries = self.localization_metrics.detect_boundaries(field_data)
        
        # Compute boundary errors
        boundary_error = self.localization_metrics.compute_boundary_distance(
            detected_boundaries, true_boundaries
        )
        
        return {
            'boundary_error_km': float(np.mean(boundary_error)),
            'boundary_detection_rate': float(len(detected_boundaries) / len(true_boundaries)),
            'false_positive_rate': float(max(0, len(detected_boundaries) - len(true_boundaries)) / len(true_boundaries)),
        }
    
    def _test_multi_target_localization(self) -> Dict:
        """Test localization of multiple targets."""
        # Load multi-target scenario
        multi_data = self.datasets.load_multi_target_test()
        true_positions = self.datasets.load_gold_output('multi_targets')
        
        # Detect and localize targets
        detected_positions = self.localization_metrics.localize_multi_targets(multi_data)
        
        # Match detected to true positions
        matches, errors = self.localization_metrics.match_positions(
            detected_positions, true_positions
        )
        
        return {
            'detection_rate': float(len(matches) / len(true_positions)),
            'mean_position_error_km': float(np.mean(errors)) if len(errors) > 0 else 999.0,
            'false_alarm_rate': float(max(0, len(detected_positions) - len(matches)) / len(true_positions)),
        }
    
    def _test_depth_estimation(self) -> Dict:
        """Test depth estimation accuracy."""
        # Load test data with known depths
        depth_data = self.datasets.load_depth_test()
        true_depths = self.datasets.load_gold_output('depths')
        
        # Estimate depths
        estimated_depths = self.localization_metrics.estimate_depths(depth_data)
        
        # Compute errors
        depth_errors = np.abs(estimated_depths - true_depths) / true_depths * 100  # Percent
        
        return {
            'depth_error_pct': float(np.mean(depth_errors)),
            'depth_error_std': float(np.std(depth_errors)),
            'max_depth_error_pct': float(np.max(depth_errors)),
        }
    
    # =========================================================================
    # PERFORMANCE TESTS
    # =========================================================================
    
    def _test_forward_modeling_speed(self) -> Dict:
        """Test forward modeling performance."""
        # Load standard test case
        model = self.datasets.load_standard_model()
        
        # Run forward modeling and time it
        runtime_ms = self.performance_metrics.benchmark_forward_model(model)
        throughput = self.performance_metrics.compute_throughput(model, runtime_ms)
        
        return {
            'runtime_ms': float(runtime_ms),
            'throughput_cells_per_sec': float(throughput),
            'speedup': float(100.0 / runtime_ms),  # vs 100ms baseline
        }
    
    def _test_inversion_speed(self) -> Dict:
        """Test inversion performance."""
        # Load inversion test case
        data = self.datasets.load_inversion_test()
        
        # Run inversion and measure convergence
        runtime_ms, iterations = self.performance_metrics.benchmark_inversion(data)
        
        return {
            'runtime_ms': float(runtime_ms),
            'iterations': int(iterations),
            'ms_per_iteration': float(runtime_ms / iterations) if iterations > 0 else 999.0,
            'speedup': float(1000.0 / runtime_ms),  # vs 1000ms baseline
        }
    
    def _test_ml_inference_speed(self) -> Dict:
        """Test ML model inference speed."""
        # Load test data
        test_input = self.datasets.load_ml_test_input()
        
        # Run inference and time it
        runtime_ms, batch_size = self.performance_metrics.benchmark_ml_inference(test_input)
        
        return {
            'runtime_ms': float(runtime_ms),
            'samples_per_second': float(batch_size * 1000.0 / runtime_ms),
            'latency_ms': float(runtime_ms / batch_size),
            'speedup': float(50.0 / (runtime_ms / batch_size)),  # vs 50ms baseline
        }
    
    def _test_memory_efficiency(self) -> Dict:
        """Test memory usage."""
        # Run standard pipeline
        memory_stats = self.performance_metrics.measure_memory_usage()
        
        return {
            'peak_memory_mb': float(memory_stats['peak'] / 1024 / 1024),
            'average_memory_mb': float(memory_stats['average'] / 1024 / 1024),
            'memory_efficiency': float(memory_stats['efficiency']),
        }
    
    # =========================================================================
    # REPORTING
    # =========================================================================
    
    def _generate_summary(self):
        """Generate benchmark summary."""
        print(f"\n{'='*70}")
        print("📊 BENCHMARK SUMMARY")
        print(f"{'='*70}\n")
        
        passed = sum(1 for r in self.results if r.status == "PASS")
        warned = sum(1 for r in self.results if r.status == "WARN")
        failed = sum(1 for r in self.results if r.status == "FAIL")
        total = len(self.results)
        
        print(f"Total Tests:  {total}")
        print(f"✅ Passed:    {passed} ({passed/total*100:.1f}%)")
        print(f"⚠️  Warnings:  {warned} ({warned/total*100:.1f}%)")
        print(f"❌ Failed:    {failed} ({failed/total*100:.1f}%)")
        
        total_runtime = sum(r.runtime for r in self.results)
        print(f"\nTotal Runtime: {total_runtime:.2f}s")
        
        # Suite breakdown
        print("\n" + "-" * 70)
        print("Suite Breakdown:")
        for suite in ['spatial', 'localization', 'performance']:
            suite_results = [r for r in self.results if r.suite == suite]
            if suite_results:
                suite_passed = sum(1 for r in suite_results if r.status == "PASS")
                print(f"  {suite:15s}: {suite_passed}/{len(suite_results)} passed")
    
    def generate_report(self, format: str = 'json'):
        """Generate detailed report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == 'json':
            report_file = self.output_dir / f"benchmark_report_{timestamp}.json"
            with open(report_file, 'w') as f:
                json.dump([asdict(r) for r in self.results], f, indent=2)
            print(f"\n📄 JSON report: {report_file}")
            
        elif format == 'html':
            report_file = self.output_dir / f"benchmark_report_{timestamp}.html"
            self._generate_html_report(report_file)
            print(f"\n📄 HTML report: {report_file}")
        
        return report_file
    
    def _generate_html_report(self, output_file: Path):
        """Generate HTML report with visualizations."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Benchmark Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        h1 {{ color: #333; }}
        .summary {{ background: white; padding: 20px; margin: 20px 0; border-radius: 5px; }}
        .results {{ background: white; padding: 20px; margin: 20px 0; border-radius: 5px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #4CAF50; color: white; }}
        .pass {{ color: #4CAF50; font-weight: bold; }}
        .warn {{ color: #FF9800; font-weight: bold; }}
        .fail {{ color: #F44336; font-weight: bold; }}
        .metric {{ margin: 5px 0; }}
    </style>
</head>
<body>
    <h1>🔬 Geophysics Benchmark Report</h1>
    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Total Tests:</strong> {len(self.results)}</p>
        <p><strong>Status:</strong> 
            <span class="pass">{sum(1 for r in self.results if r.status == 'PASS')} PASS</span> | 
            <span class="warn">{sum(1 for r in self.results if r.status == 'WARN')} WARN</span> | 
            <span class="fail">{sum(1 for r in self.results if r.status == 'FAIL')} FAIL</span>
        </p>
    </div>
    <div class="results">
        <h2>Detailed Results</h2>
        <table>
            <tr>
                <th>Test Name</th>
                <th>Suite</th>
                <th>Status</th>
                <th>Runtime</th>
                <th>Key Metrics</th>
            </tr>
"""
        
        for result in self.results:
            status_class = result.status.lower()
            metrics_str = "<br>".join([f"{k}: {v:.3f}" for k, v in list(result.metrics.items())[:3]])
            
            html += f"""
            <tr>
                <td>{result.name}</td>
                <td>{result.suite}</td>
                <td class="{status_class}">{result.status}</td>
                <td>{result.runtime:.3f}s</td>
                <td class="metric">{metrics_str}</td>
            </tr>
"""
        
        html += """
        </table>
    </div>
</body>
</html>
"""
        
        with open(output_file, 'w') as f:
            f.write(html)
    
    def run_coverage_analysis(self) -> float:
        """Run code coverage analysis."""
        print("\n📊 Running Coverage Analysis...")
        print("-" * 70)
        
        coverage = self.coverage_analyzer.analyze_coverage()
        
        print(f"\nOverall Coverage: {coverage['overall']:.1f}%")
        print("\nModule Coverage:")
        for module, cov in coverage['modules'].items():
            symbol = "✅" if cov >= 85 else "⚠️" if cov >= 70 else "❌"
            print(f"  {symbol} {module:30s}: {cov:5.1f}%")
        
        # Check critical modules
        critical_threshold = 85.0
        critical_modules = ['inversion', 'forward_model', 'ml_models']
        critical_pass = all(
            coverage['modules'].get(m, 0) >= critical_threshold 
            for m in critical_modules
        )
        
        if critical_pass:
            print(f"\n✅ All critical modules meet {critical_threshold}% threshold")
        else:
            print(f"\n⚠️  Some critical modules below {critical_threshold}% threshold")
        
        return coverage['overall']


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Geophysics Benchmarking and Verification Suite'
    )
    parser.add_argument(
        '--suite',
        choices=['all', 'spatial', 'localization', 'performance'],
        default='all',
        help='Benchmark suite to run'
    )
    parser.add_argument(
        '--coverage',
        action='store_true',
        help='Run coverage analysis'
    )
    parser.add_argument(
        '--report',
        choices=['json', 'html'],
        help='Generate report in specified format'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('bench/reports'),
        help='Output directory for reports'
    )
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = BenchmarkRunner(output_dir=args.output_dir)
    
    # Run benchmarks
    success = runner.run_suite(args.suite)
    
    # Generate report if requested
    if args.report:
        runner.generate_report(format=args.report)
    
    # Run coverage if requested
    if args.coverage:
        coverage = runner.run_coverage_analysis()
        if coverage < 85.0:
            print(f"\n⚠️  Coverage {coverage:.1f}% below target 85%")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
