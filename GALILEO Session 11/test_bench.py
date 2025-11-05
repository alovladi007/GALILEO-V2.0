"""
Test Suite for Benchmarking Infrastructure
===========================================
Tests the benchmarking suite itself to ensure reliability.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from bench import (
    RegressionDatasets,
    SpatialResolutionMetrics,
    LocalizationMetrics,
    PerformanceMetrics,
    CoverageAnalyzer
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp = tempfile.mkdtemp()
    yield Path(temp)
    shutil.rmtree(temp)


@pytest.fixture
def datasets(temp_dir):
    """Create test datasets instance."""
    return RegressionDatasets(data_dir=temp_dir / "datasets")


class TestRegressionDatasets:
    """Test dataset generation and loading."""
    
    def test_dataset_initialization(self, datasets):
        """Test dataset directory creation."""
        assert datasets.data_dir.exists()
        assert datasets.gold_dir.exists()
    
    def test_point_source_generation(self, datasets):
        """Test point source dataset generation."""
        datasets._generate_point_source()
        
        point_source = datasets.load_point_source()
        assert point_source.shape == (128, 128)
        assert np.max(point_source) > 0
        
        # Check gold PSF exists
        gold_psf = datasets.load_gold_output('psf')
        assert gold_psf.shape == (64, 64)
    
    def test_frequency_test_generation(self, datasets):
        """Test frequency test dataset generation."""
        datasets._generate_frequency_test()
        
        freq_data = datasets.load_frequency_test()
        assert 'frequencies' in freq_data
        assert 'input' in freq_data
        assert 'output' in freq_data
        assert len(freq_data['frequencies']) == 5
    
    def test_localization_test_generation(self, datasets):
        """Test localization dataset generation."""
        datasets._generate_localization_test()
        
        data = datasets.load_localization_test()
        centroids = datasets.load_gold_output('centroids')
        
        assert data.shape == (256, 256)
        assert len(centroids) == 5
        assert centroids.shape[1] == 2  # (y, x) coordinates
    
    def test_all_datasets_generation(self, datasets):
        """Test full dataset generation."""
        datasets._generate_all_datasets()
        assert datasets.verify_datasets()


class TestSpatialResolutionMetrics:
    """Test spatial resolution metric calculations."""
    
    @pytest.fixture
    def metrics(self):
        return SpatialResolutionMetrics()
    
    def test_psf_computation(self, metrics):
        """Test PSF computation."""
        # Create synthetic point source
        size = 64
        y, x = np.ogrid[-size//2:size//2, -size//2:size//2]
        point_source = np.exp(-(x**2 + y**2) / (2 * 5**2))
        
        psf = metrics.compute_psf(point_source)
        
        assert psf.shape[0] > 0
        assert psf.shape[1] > 0
        assert np.max(psf) <= 1.0  # Normalized
    
    def test_fwhm_computation(self, metrics):
        """Test FWHM calculation."""
        # Create Gaussian PSF with known FWHM
        size = 64
        sigma = 5.0
        y, x = np.ogrid[-size//2:size//2, -size//2:size//2]
        psf = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        
        fwhm = metrics.compute_fwhm(psf, pixel_size_km=1.0)
        
        # Expected FWHM ≈ 2.355 * sigma
        expected_fwhm = 2.355 * sigma
        assert abs(fwhm - expected_fwhm) < 2.0  # Within 2 km tolerance
    
    def test_peak_finding(self, metrics):
        """Test 2D peak finding."""
        # Create data with known peaks
        data = np.zeros((100, 100))
        peaks_true = [(20, 20), (50, 50), (80, 80)]
        
        for y, x in peaks_true:
            data[y-5:y+5, x-5:x+5] = np.random.rand(10, 10)
        
        peaks_found = metrics._find_peaks_2d(data, threshold=0.3)
        
        # Should find approximately the right number of peaks
        assert len(peaks_found) >= 2  # At least most of them
    
    def test_crosstalk_computation(self, metrics):
        """Test crosstalk calculation."""
        # Perfect separation: zero crosstalk
        separated = np.random.randn(50, 50)
        gold = separated.copy()
        
        crosstalk = metrics.compute_crosstalk(separated, gold)
        assert crosstalk < -40.0  # Very low crosstalk
        
        # Poor separation: high crosstalk
        noisy_separated = separated + np.random.randn(50, 50) * 0.5
        crosstalk_noisy = metrics.compute_crosstalk(noisy_separated, gold)
        assert crosstalk_noisy > crosstalk  # Higher crosstalk


class TestLocalizationMetrics:
    """Test localization metric calculations."""
    
    @pytest.fixture
    def metrics(self):
        return LocalizationMetrics()
    
    def test_centroid_computation(self, metrics):
        """Test centroid computation."""
        # Create data with known centroid
        size = 100
        y, x = np.ogrid[:size, :size]
        
        center_y, center_x = 50, 50
        data = np.exp(-((x - center_x)**2 + (y - center_y)**2) / 100)
        
        centroids = metrics.compute_centroids(data)
        
        assert len(centroids) >= 1
        # Check if found centroid is close to true centroid
        if len(centroids) > 0:
            found_y, found_x = centroids[0]
            assert abs(found_y - center_y) < 5
            assert abs(found_x - center_x) < 5
    
    def test_position_error_computation(self, metrics):
        """Test position error calculation."""
        # Perfect match
        computed = np.array([[10, 10], [20, 20], [30, 30]])
        true = computed.copy()
        
        errors = metrics.compute_position_errors(computed, true, pixel_size_km=1.0)
        
        assert len(errors) == 3
        assert np.all(errors < 0.01)  # Nearly perfect
        
        # With error
        computed_offset = true + np.array([[1, 1], [2, 2], [3, 3]])
        errors_offset = metrics.compute_position_errors(
            computed_offset, true, pixel_size_km=1.0
        )
        
        assert np.all(errors_offset > 0)  # Non-zero errors
    
    def test_boundary_detection(self, metrics):
        """Test boundary detection."""
        # Create step function with clear boundary
        data = np.zeros((100, 100))
        data[30:70, 30:70] = 1.0
        
        boundaries = metrics.detect_boundaries(data)
        
        # Should detect at least one boundary
        assert len(boundaries) >= 1
    
    def test_depth_estimation(self, metrics):
        """Test depth estimation."""
        # Create anomaly with known depth (width-based estimation)
        size = 100
        y, x = np.ogrid[:size, :size]
        
        depth_km = 10.0
        sigma = depth_km * 0.8  # Width proportional to depth
        
        field = np.exp(-((x - 50)**2 + (y - 50)**2) / (2 * sigma**2))
        
        depth_data = {
            'field': field,
            'positions': np.array([[50, 50]])
        }
        
        estimated_depths = metrics.estimate_depths(depth_data)
        
        assert len(estimated_depths) == 1
        # Rough estimate should be within factor of 2
        assert 5.0 < estimated_depths[0] < 20.0


class TestPerformanceMetrics:
    """Test performance metric calculations."""
    
    @pytest.fixture
    def metrics(self):
        return PerformanceMetrics()
    
    def test_forward_model_benchmark(self, metrics):
        """Test forward modeling benchmark."""
        model = {'size': 50}
        
        runtime = metrics.benchmark_forward_model(model)
        
        assert runtime > 0
        assert runtime < 10000  # Should complete in < 10 seconds
    
    def test_throughput_computation(self, metrics):
        """Test throughput calculation."""
        model = {'size': 50}
        runtime_ms = 100.0
        
        throughput = metrics.compute_throughput(model, runtime_ms)
        
        expected = 50**3 / 0.1  # cells per second
        assert abs(throughput - expected) < expected * 0.01  # 1% tolerance
    
    def test_inversion_benchmark(self, metrics):
        """Test inversion benchmark."""
        data = {'size': 30, 'max_iter': 10}
        
        runtime, iterations = metrics.benchmark_inversion(data)
        
        assert runtime > 0
        assert 1 <= iterations <= 10
    
    def test_ml_inference_benchmark(self, metrics):
        """Test ML inference benchmark."""
        test_input = {'batch_size': 16, 'shape': (32, 32)}
        
        runtime, batch_size = metrics.benchmark_ml_inference(test_input)
        
        assert runtime > 0
        assert batch_size == 16
    
    def test_memory_measurement(self, metrics):
        """Test memory usage measurement."""
        stats = metrics.measure_memory_usage()
        
        assert 'baseline' in stats
        assert 'peak' in stats
        assert 'average' in stats
        assert stats['peak'] >= stats['baseline']


class TestCoverageAnalyzer:
    """Test coverage analysis functionality."""
    
    @pytest.fixture
    def analyzer(self):
        return CoverageAnalyzer()
    
    def test_coverage_simulation(self, analyzer):
        """Test coverage simulation."""
        coverage = analyzer._simulate_coverage()
        
        assert 'overall' in coverage
        assert 'modules' in coverage
        assert 0 <= coverage['overall'] <= 100
        
        # Check critical modules
        for module in analyzer.critical_modules:
            assert module in coverage['modules']
            assert 0 <= coverage['modules'][module] <= 100


class TestIntegration:
    """Integration tests for full benchmark pipeline."""
    
    def test_full_pipeline(self, temp_dir):
        """Test complete benchmarking pipeline."""
        # This would normally import and run the full bench.py,
        # but for unit tests we'll test components
        
        # 1. Create datasets
        datasets = RegressionDatasets(data_dir=temp_dir / "datasets")
        datasets._generate_point_source()
        
        # 2. Load data
        point_source = datasets.load_point_source()
        assert point_source is not None
        
        # 3. Compute metrics
        metrics = SpatialResolutionMetrics()
        psf = metrics.compute_psf(point_source)
        assert psf is not None
        
        # 4. Compute FWHM
        fwhm = metrics.compute_fwhm(psf)
        assert fwhm > 0
    
    def test_benchmark_result_format(self):
        """Test benchmark result data structure."""
        from bench import BenchmarkResult
        
        result = BenchmarkResult(
            name="Test Benchmark",
            suite="spatial",
            status="PASS",
            metrics={'resolution_km': 3.5},
            runtime=0.123,
            timestamp="2025-11-04T10:00:00"
        )
        
        assert result.name == "Test Benchmark"
        assert result.suite == "spatial"
        assert result.status == "PASS"
        assert result.metrics['resolution_km'] == 3.5


# Performance benchmarks (optional, run with --benchmark flag)
class TestBenchmarkPerformance:
    """Performance benchmarks using pytest-benchmark."""
    
    def test_psf_computation_speed(self, benchmark):
        """Benchmark PSF computation speed."""
        metrics = SpatialResolutionMetrics()
        point_source = np.random.randn(128, 128)
        
        result = benchmark(metrics.compute_psf, point_source)
        assert result is not None
    
    def test_centroid_computation_speed(self, benchmark):
        """Benchmark centroid computation speed."""
        metrics = LocalizationMetrics()
        data = np.random.randn(256, 256)
        
        result = benchmark(metrics.compute_centroids, data)
        assert result is not None


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v', '--tb=short'])
