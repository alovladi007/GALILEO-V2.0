

# Verification and Benchmarking Suite

Comprehensive verification, regression testing, and performance benchmarking system for the geophysical gravity gradiometry processing pipeline.

## Overview

The benchmarking suite provides automated testing and verification across three critical domains:

1. **Spatial Resolution**: Point spread function, frequency response, anomaly separation
2. **Localization Error**: Centroid accuracy, boundary detection, depth estimation
3. **Runtime Performance**: Forward modeling, inversion, ML inference, memory efficiency

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Generate test datasets
python -c "from bench.datasets import create_sample_datasets; create_sample_datasets()"
```

### Running Benchmarks

```bash
# Run all benchmark suites
python bench.py --suite all

# Run specific suite
python bench.py --suite spatial
python bench.py --suite localization
python bench.py --suite performance

# Generate reports
python bench.py --suite all --report html
python bench.py --suite all --report json

# Run with coverage analysis
python bench.py --suite all --coverage
```

## Architecture

### Directory Structure

```
bench/
├── datasets/           # Regression test datasets
│   ├── point_source.npy
│   ├── frequency_test.pkl
│   ├── twin_anomalies.npy
│   └── ...
├── gold_outputs/       # Gold standard outputs
│   ├── psf.npy
│   ├── centroids.npy
│   └── ...
├── metrics/            # Metric implementations
│   └── (in metrics.py)
├── reports/            # Generated reports
│   ├── benchmark_report_*.json
│   └── benchmark_report_*.html
└── __init__.py

bench.py                # Main runner
docs/verification.md    # This file
.github/workflows/      # CI integration
    └── benchmark.yml
```

### Component Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      bench.py                               │
│                 (Main Orchestrator)                          │
└─────────────────────────────────────────────────────────────┘
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
    ┌───────────┐   ┌──────────────┐   ┌──────────────┐
    │ Datasets  │   │   Metrics    │   │  Coverage    │
    │  Module   │   │   Module     │   │  Analyzer    │
    └───────────┘   └──────────────┘   └──────────────┘
            │               │                   │
            ▼               ▼                   ▼
    ┌───────────┐   ┌──────────────┐   ┌──────────────┐
    │Regression │   │ Performance  │   │Test Coverage │
    │   Data    │   │   Analysis   │   │   Reports    │
    └───────────┘   └──────────────┘   └──────────────┘
```

## Benchmark Suites

### 1. Spatial Resolution Suite

Tests the system's ability to resolve spatial features and separate nearby anomalies.

#### Tests

##### PSF Characterization
- **Purpose**: Characterize point spread function
- **Metric**: FWHM (Full Width at Half Maximum) in km
- **Target**: < 5 km
- **Method**: Single point source response analysis

##### Frequency Response
- **Purpose**: Test spatial frequency response
- **Metric**: MTF (Modulation Transfer Function) at 0.5 cycles/pixel
- **Target**: > 0.8
- **Method**: Sinusoidal test patterns at multiple frequencies

##### Resolution Recovery
- **Purpose**: Separate closely-spaced twin anomalies
- **Metric**: Minimum separation distance in km
- **Target**: < 5 km with > 0.9 correlation to gold standard
- **Method**: Deconvolution of twin Gaussian sources

##### Anomaly Separation
- **Purpose**: Separate overlapping anomalies
- **Metric**: Crosstalk in dB and RMSE
- **Target**: < -20 dB crosstalk
- **Method**: Blind source separation on 3 overlapping sources

#### Example Output

```
📍 Running Spatial Resolution Suite...
----------------------------------------------------------------------
✅ PSF Characterization              |  0.142s | PASS
✅ Frequency Response                |  0.089s | PASS
✅ Resolution Recovery               |  0.234s | PASS
⚠️  Anomaly Separation               |  0.156s | WARN
```

### 2. Localization Suite

Tests position accuracy for detecting and localizing subsurface anomalies.

#### Tests

##### Centroid Localization
- **Purpose**: Test centroid position accuracy
- **Metric**: Mean position error in km
- **Target**: < 2 km RMS error
- **Method**: 5 Gaussian sources with known positions

##### Boundary Detection
- **Purpose**: Detect boundaries between regions
- **Metric**: Boundary distance error in km
- **Target**: < 3 km mean error
- **Method**: Step function boundaries with smoothing

##### Multi-Target Localization
- **Purpose**: Detect and localize multiple targets
- **Metric**: Detection rate and false alarm rate
- **Target**: > 90% detection rate, < 10% false alarms
- **Method**: 15 randomly positioned targets with noise

##### Depth Estimation
- **Purpose**: Estimate anomaly source depths
- **Metric**: Depth error percentage
- **Target**: < 15% error
- **Method**: Sources at 5, 10, 15, 20 km depths

#### Example Output

```
🎯 Running Localization Suite...
----------------------------------------------------------------------
✅ Centroid Localization             |  0.098s | PASS
✅ Boundary Detection                |  0.145s | PASS
✅ Multi-Target Localization         |  0.187s | PASS
⚠️  Depth Estimation                 |  0.112s | WARN
```

### 3. Performance Suite

Tests computational performance and resource utilization.

#### Tests

##### Forward Modeling Speed
- **Purpose**: Test forward gravity calculation speed
- **Metric**: Runtime in ms, throughput in cells/sec
- **Target**: < 100 ms for 64³ model
- **Method**: FFT-based forward modeling

##### Inversion Speed
- **Purpose**: Test inversion convergence speed
- **Metric**: Runtime in ms, iterations to convergence
- **Target**: < 1000 ms, < 50 iterations
- **Method**: Iterative inversion on 50² grid

##### ML Inference Speed
- **Purpose**: Test neural network inference
- **Metric**: Latency in ms, throughput in samples/sec
- **Target**: < 50 ms latency per sample
- **Method**: Batch inference on 32 samples

##### Memory Efficiency
- **Purpose**: Monitor memory usage
- **Metric**: Peak memory in MB
- **Target**: < 500 MB for standard pipeline
- **Method**: Process monitoring during computation

#### Example Output

```
⚡ Running Performance Suite...
----------------------------------------------------------------------
✅ Forward Modeling                  |  0.045s | PASS
✅ Inversion Speed                   |  0.523s | PASS
✅ ML Inference                      |  0.034s | PASS
✅ Memory Efficiency                 |  0.167s | PASS
```

## Metrics and Thresholds

### Spatial Resolution Metrics

| Metric | Unit | Pass Threshold | Warn Threshold | Fail Threshold |
|--------|------|----------------|----------------|----------------|
| PSF Width (FWHM) | km | < 5.0 | 5.0 - 7.5 | > 7.5 |
| Resolution | km | < 5.0 | 5.0 - 7.5 | > 7.5 |
| Frequency Response | ratio | > 0.8 | 0.6 - 0.8 | < 0.6 |
| Separation RMSE | - | < 0.1 | 0.1 - 0.2 | > 0.2 |
| Crosstalk | dB | < -20 | -20 to -10 | > -10 |

### Localization Metrics

| Metric | Unit | Pass Threshold | Warn Threshold | Fail Threshold |
|--------|------|----------------|----------------|----------------|
| Centroid Error | km | < 2.0 | 2.0 - 3.0 | > 3.0 |
| Boundary Error | km | < 3.0 | 3.0 - 4.5 | > 4.5 |
| Detection Rate | % | > 90 | 70 - 90 | < 70 |
| Depth Error | % | < 15 | 15 - 25 | > 25 |

### Performance Metrics

| Metric | Unit | Pass Threshold | Warn Threshold | Fail Threshold |
|--------|------|----------------|----------------|----------------|
| Forward Model Runtime | ms | < 100 | 100 - 150 | > 150 |
| Inversion Runtime | ms | < 1000 | 1000 - 1500 | > 1500 |
| ML Latency | ms | < 50 | 50 - 75 | > 75 |
| Peak Memory | MB | < 500 | 500 - 750 | > 750 |

## Coverage Analysis

The suite includes code coverage analysis targeting ≥ 85% coverage on critical modules.

### Critical Modules

1. **inversion/** - Inversion algorithms (Target: 85%)
2. **forward_model/** - Forward gravity modeling (Target: 85%)
3. **ml_models/** - Machine learning models (Target: 85%)
4. **preprocessing/** - Data preprocessing (Target: 85%)
5. **validation/** - Data validation (Target: 85%)

### Running Coverage

```bash
# Run with coverage
python bench.py --coverage

# Generate detailed coverage report
pytest --cov=. --cov-report=html --cov-report=term
open htmlcov/index.html
```

### Example Coverage Output

```
📊 Running Coverage Analysis...
----------------------------------------------------------------------

Overall Coverage: 87.5%

Module Coverage:
  ✅ inversion                      :  92.3%
  ✅ forward_model                  :  88.7%
  ✅ ml_models                      :  85.2%
  ✅ preprocessing                  :  91.5%
  ⚠️  validation                    :  83.8%

✅ All critical modules meet 85% threshold
```

## CI Integration

The benchmarking suite is integrated into CI/CD via GitHub Actions.

### Triggers

- **Push** to main or develop branches
- **Pull requests** to main
- **Scheduled**: Daily at 2 AM UTC
- **Manual**: Workflow dispatch

### CI Pipeline

```
┌─────────────┐
│   Checkout  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Install   │
│Dependencies │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Generate   │
│  Datasets   │
└──────┬──────┘
       │
       ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Spatial   │───▶│Localization │───▶│ Performance │
│    Suite    │    │    Suite    │    │    Suite    │
└──────┬──────┘    └──────┬──────┘    └──────┬──────┘
       │                  │                   │
       └──────────────────┴───────────────────┘
                          │
                          ▼
                   ┌─────────────┐
                   │  Coverage   │
                   │  Analysis   │
                   └──────┬──────┘
                          │
                          ▼
                   ┌─────────────┐
                   │   Upload    │
                   │   Results   │
                   └─────────────┘
```

### Matrix Testing

The CI runs benchmarks across multiple Python versions:
- Python 3.9
- Python 3.10
- Python 3.11

### Regression Detection

The CI automatically compares PR performance against main branch and flags:
- **Performance regressions**: > 10% slower
- **Performance improvements**: > 10% faster
- **Test failures**: Any failed tests
- **Coverage drops**: Below 85% on critical modules

### Artifacts

Each CI run produces:
- JSON benchmark results
- HTML reports
- Coverage reports
- Performance comparison charts

## Report Formats

### JSON Report

Machine-readable format for automated analysis:

```json
[
  {
    "name": "PSF Characterization",
    "suite": "spatial",
    "status": "PASS",
    "metrics": {
      "psf_error": 0.023,
      "psf_width": 4.8,
      "resolution_km": 2.04
    },
    "runtime": 0.142,
    "timestamp": "2025-11-04T10:30:15.123456"
  },
  ...
]
```

### HTML Report

Human-readable format with visualizations:

```html
<!DOCTYPE html>
<html>
<head>
    <title>Benchmark Report - 2025-11-04</title>
    ...
</head>
<body>
    <h1>🔬 Geophysics Benchmark Report</h1>
    
    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Total Tests:</strong> 12</p>
        <p><strong>Status:</strong> 
            <span class="pass">10 PASS</span> | 
            <span class="warn">2 WARN</span> | 
            <span class="fail">0 FAIL</span>
        </p>
    </div>
    
    <div class="results">
        <h2>Detailed Results</h2>
        <table>...</table>
    </div>
</body>
</html>
```

## API Reference

### BenchmarkRunner

Main orchestrator class for running benchmarks.

```python
from bench import BenchmarkRunner

runner = BenchmarkRunner(output_dir="bench/reports")

# Run specific suite
success = runner.run_suite('spatial')

# Run all suites
success = runner.run_suite('all')

# Generate reports
runner.generate_report(format='html')
runner.generate_report(format='json')

# Run coverage
coverage = runner.run_coverage_analysis()
```

### RegressionDatasets

Dataset manager for test data.

```python
from bench.datasets import RegressionDatasets

datasets = RegressionDatasets()

# Load test data
point_source = datasets.load_point_source()
twin_anomalies = datasets.load_twin_anomalies()

# Load gold outputs
gold_psf = datasets.load_gold_output('psf')
gold_centroids = datasets.load_gold_output('centroids')

# Verify datasets
datasets.verify_datasets()
```

### Metrics Classes

Individual metric implementations.

```python
from bench.metrics import (
    SpatialResolutionMetrics,
    LocalizationMetrics,
    PerformanceMetrics
)

# Spatial resolution
spatial = SpatialResolutionMetrics()
psf = spatial.compute_psf(point_source)
fwhm = spatial.compute_fwhm(psf)

# Localization
localization = LocalizationMetrics()
centroids = localization.compute_centroids(anomaly_data)
errors = localization.compute_position_errors(computed, true)

# Performance
performance = PerformanceMetrics()
runtime = performance.benchmark_forward_model(model)
```

## Best Practices

### 1. Regular Benchmarking

Run benchmarks:
- **Before releases**: Ensure no regressions
- **After optimizations**: Verify improvements
- **Daily via CI**: Catch issues early
- **On PRs**: Prevent performance degradation

### 2. Threshold Tuning

Adjust thresholds based on:
- Hardware capabilities
- Scientific requirements
- Application constraints
- Historical performance

### 3. Dataset Maintenance

- Keep gold outputs updated
- Add new test cases for edge cases
- Version control datasets
- Document dataset provenance

### 4. Continuous Improvement

- Track performance trends over time
- Identify bottlenecks
- Optimize critical paths
- Update benchmarks for new features

## Troubleshooting

### Common Issues

#### Datasets Not Found

```bash
# Regenerate datasets
python -c "from bench.datasets import create_sample_datasets; create_sample_datasets()"
```

#### Coverage Below Threshold

```bash
# Run detailed coverage analysis
pytest --cov=. --cov-report=term-missing
```

#### CI Failures

```bash
# Run locally first
python bench.py --suite all
# Check specific suite
python bench.py --suite spatial
```

#### Memory Issues

```bash
# Reduce dataset sizes in datasets.py
# Or run suites individually:
python bench.py --suite spatial
python bench.py --suite localization  
python bench.py --suite performance
```

## Future Enhancements

Planned improvements:

1. **Visualization Dashboard**: Real-time performance tracking
2. **Benchmark Database**: Historical performance storage
3. **Automated Tuning**: Threshold optimization
4. **Distributed Benchmarking**: Multi-node testing
5. **GPU Benchmarks**: Accelerator performance tests
6. **Profiling Integration**: Detailed performance analysis

## References

### Documentation
- [Main Project README](../README.md)
- [API Documentation](./api.md)
- [User Guide](./user_guide.md)

### Related Sessions
- Session 5: Inversion Engine
- Session 6: ML Acceleration
- Session 9: Calibration
- Session 10: Earth Models

### External Resources
- [Pytest Documentation](https://docs.pytest.org/)
- [Coverage.py](https://coverage.readthedocs.io/)
- [GitHub Actions](https://docs.github.com/en/actions)

## Support

For issues or questions:
- Open an issue on GitHub
- Contact: support@geophysics-platform.org
- Documentation: https://docs.geophysics-platform.org

---

**Version**: 1.0.0  
**Last Updated**: November 4, 2025  
**Status**: Production Ready ✅
