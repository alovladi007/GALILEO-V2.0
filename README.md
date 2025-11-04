# GeoSense Platform

> 🌍 Space-based geophysical sensing and mapping platform for measuring minute variations in Earth's mass distribution

[![CI](https://github.com/geosense/geosense-platform/workflows/CI/badge.svg)](https://github.com/geosense/geosense-platform/actions)
[![License](https://img.shields.io/badge/License-Proprietary-red.svg)](./compliance/LEGAL.md)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Rust](https://img.shields.io/badge/Rust-1.70+-orange.svg)](https://www.rust-lang.org/)

**⚠️ Research Use Only** - This platform is currently in research phase and designated for scientific and educational purposes only. See [compliance/LEGAL.md](./compliance/LEGAL.md) and [compliance/ETHICS.md](./compliance/ETHICS.md) for details.

## Overview

GeoSense is a comprehensive platform for satellite-based gravimetric sensing, designed to measure and analyze subtle variations in Earth's gravitational field. These measurements enable:

- **Climate Science**: Ice sheet mass balance, sea level contributions
- **Hydrology**: Groundwater storage changes, aquifer dynamics  
- **Solid Earth Geophysics**: Tectonic processes, earthquake precursors
- **Natural Resource Management**: Sustainable resource monitoring
- **Environmental Analysis**: Ecosystem health indicators

### Key Capabilities

- 🛰️ **High-Fidelity Orbit Simulation** - Sub-centimeter orbit determination with full perturbation modeling
- 📡 **Gravimetry Pipeline** - Accelerometer processing and range-rate measurements
- 🔄 **Geophysical Inversion** - Tikhonov and Bayesian methods for mass distribution recovery
- 🤖 **ML-Enhanced Processing** - Neural networks for noise reduction and anomaly detection
- 🌐 **Real-time Visualization** - CesiumJS-based 3D globe with interactive data exploration
- 📊 **Operations Dashboard** - Mission planning, telemetry monitoring, and data management

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     GeoSense Platform                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │   Sim    │  │ Sensing  │  │Inversion │  │    ML    │  │
│  │ (Python) │  │ (Python) │  │ (Python) │  │ (Python) │  │
│  │  JAX/    │  │  NumPy/  │  │   JAX/   │  │  Flax/   │  │
│  │  NumPy   │  │  SciPy   │  │  SciPy   │  │  PyTorch │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Control (Rust)                           │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │  │
│  │  │ Dynamics │  │ Attitude │  │  Power   │          │  │
│  │  │nalgebra  │  │quaternion│  │  mgmt    │          │  │
│  │  └──────────┘  └──────────┘  └──────────┘          │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         UI (Next.js + CesiumJS)                       │  │
│  │  Globe Viewer • Orbit Tracks • Data Layers           │  │
│  │  Time Series • Analysis Tools • Mission Control      │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Operations & Infrastructure                   │  │
│  │  Redis • PostgreSQL • TimescaleDB • Celery           │  │
│  │  Grafana • Prometheus • Jaeger • Docker              │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

See [docs/architecture/](./docs/architecture/) for detailed architecture diagrams.

## Technology Stack

### Core Languages
- **Python 3.11+**: Simulation, sensing, inversion, ML
- **Rust 1.70+**: Real-time control systems, orbit dynamics
- **TypeScript/Next.js 14**: Web-based visualization and UI

### Key Dependencies

**Python:**
- **JAX** - High-performance numerical computing with automatic differentiation
- **NumPy/SciPy** - Scientific computing foundations
- **Flax/PyTorch** - Neural network implementations

**Rust:**
- **nalgebra** - Linear algebra for orbital mechanics
- **tokio** - Async runtime for control systems

**Frontend:**
- **CesiumJS** - 3D globe and geospatial visualization
- **Recharts** - Time series and analytical plots

## Quick Start

### Prerequisites

```bash
# Python 3.11+
python --version

# Rust 1.70+
rustc --version

# Node.js 18+
node --version

# Docker & Docker Compose
docker --version
docker-compose --version
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/geosense/geosense-platform.git
cd geosense-platform
```

2. **Set up Python environment**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev,ml]"
```

3. **Build Rust components**
```bash
cargo build --release
```

4. **Set up UI**
```bash
cd ui
npm install
npm run build
cd ..
```

5. **Start services with Docker Compose**
```bash
docker-compose up -d
```

### Running the Platform

**Start API server:**
```bash
uvicorn ops.api:app --reload --port 8000
```

**Start Celery worker:**
```bash
celery -A ops.worker worker --loglevel=info
```

**Start UI development server:**
```bash
cd ui
npm run dev
```

**Access the platform:**
- UI: http://localhost:3000
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Grafana: http://localhost:3001 (admin/admin)

## Development

### Running Tests

**Python tests:**
```bash
pytest tests/ -v --cov
```

**Rust tests:**
```bash
cargo test --workspace
```

**UI tests:**
```bash
cd ui
npm test
```

### Code Quality

**Python linting:**
```bash
ruff check sim sensing inversion ml ops
black sim sensing inversion ml ops
mypy sim sensing inversion ml ops
```

**Rust linting:**
```bash
cargo fmt --check
cargo clippy -- -D warnings
```

**UI linting:**
```bash
cd ui
npm run lint
npm run type-check
```

### Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

## Project Structure

```
geosense-platform/
├── sim/                    # Simulation (orbit, gravity, sensors)
│   ├── orbit/             # Orbital mechanics
│   ├── gravity/           # Gravity field models
│   └── sensors/           # Sensor simulations
├── control/               # Rust control systems
│   ├── dynamics/          # Orbit dynamics & propagation
│   ├── attitude/          # Attitude determination & control
│   └── power/             # Power management
├── sensing/               # Sensor data processing
│   ├── gravimetry/        # Gravimeter processing
│   ├── accelerometer/     # Accelerometer data
│   └── gnss/              # GNSS processing
├── inversion/             # Geophysical inversion
│   ├── algorithms/        # Inversion methods
│   ├── solvers/           # Numerical solvers
│   └── constraints/       # Regularization
├── ml/                    # Machine learning
│   ├── models/            # Neural network architectures
│   ├── training/          # Training pipelines
│   └── inference/         # Inference engines
├── ops/                   # Operations & orchestration
│   ├── orchestration/     # Task scheduling
│   ├── scheduling/        # Mission planning
│   └── telemetry/         # Telemetry management
├── ui/                    # Next.js web interface
│   ├── src/components/    # React components
│   ├── src/pages/         # Next.js pages
│   └── public/            # Static assets
├── devops/                # Infrastructure as code
│   ├── terraform/         # Cloud infrastructure
│   ├── ansible/           # Configuration management
│   └── k8s/               # Kubernetes manifests
├── docs/                  # Documentation
│   ├── architecture/      # Architecture diagrams
│   ├── api/               # API documentation
│   └── deployment/        # Deployment guides
├── compliance/            # Legal & ethical guidelines
│   ├── ETHICS.md          # Ethical guidelines
│   └── LEGAL.md           # Legal compliance
├── tests/                 # Test suites
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── e2e/               # End-to-end tests
└── scripts/               # Utility scripts
```

## Scientific Background

### Gravimetric Sensing Principles

GeoSense measures Earth's gravitational field using satellite-to-satellite tracking and onboard accelerometry:

1. **Satellite-to-Satellite Tracking (SST)**: Measures range and range-rate between satellites
2. **Accelerometry**: High-precision accelerometers measure non-gravitational forces
3. **Orbit Determination**: Sub-centimeter orbit precision through GNSS
4. **Gradiometry**: Gravity gradient measurements from satellite configurations

### Data Processing Pipeline

```
Raw Measurements → Preprocessing → Level-1 Products → Inversion → Mass Distribution
     ↓                  ↓              ↓                  ↓              ↓
  Range/Rate    Noise Removal    Gravity Field    Regularized       Regional
  Accelerometer    Calibration    Coefficients       Solve          Analysis
  GNSS Position      Filtering      (Spherical      Bayesian         Geoid
                                    Harmonics)      Methods         Heights
```

### Measurement Precision

- **Orbit Determination**: < 1 cm (3D RMS)
- **Accelerometry**: < 10⁻¹⁰ m/s² @ 1 mHz - 0.1 Hz
- **Range-Rate**: < 1 μm/s
- **Gravity Field**: ~100 km spatial resolution, ~1 cm geoid accuracy

## Use Cases

### Climate Science
- Ice sheet mass balance (Greenland, Antarctica)
- Glacial isostatic adjustment
- Sea level budget closure

### Hydrology
- Terrestrial water storage changes
- Groundwater depletion monitoring
- Drought assessment

### Solid Earth
- Co-seismic deformation
- Post-seismic relaxation
- Volcanic mass changes

### Natural Resources
- Aquifer characterization
- Subsurface fluid migration
- Resource sustainability

## Security & Compliance

### Security Features
- 🔐 Multi-factor authentication
- 🔑 Role-based access control (RBAC)
- 📝 Comprehensive audit logging
- 🔒 End-to-end encryption
- 🛡️ Regular security scanning (Trivy, CodeQL)

### Compliance Framework
- ✅ Research use authorization
- ✅ Data protection (GDPR/CCPA consideration)
- ✅ Export control awareness
- ✅ Ethical review process

See [compliance/](./compliance/) for complete guidelines.

## Contributing

Contributions are welcome! Please read our contributing guidelines first.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Standards
- Python: Follow PEP 8, use type hints
- Rust: Follow Rust style guidelines
- TypeScript: Use strict mode
- Documentation: Update docs with code changes
- Tests: Maintain >80% coverage

## License

Proprietary - Research Use Only. See [compliance/LEGAL.md](./compliance/LEGAL.md).

## Citation

If you use GeoSense in your research, please cite:

```bibtex
@software{geosense2025,
  title = {GeoSense: Space-based Geophysical Sensing Platform},
  author = {{GeoSense Team}},
  year = {2025},
  url = {https://github.com/geosense/geosense-platform},
  version = {0.1.0}
}
```

## Support & Contact

- 📧 Email: support@geosense.example
- 🐛 Issues: [GitHub Issues](https://github.com/geosense/geosense-platform/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/geosense/geosense-platform/discussions)
- 📖 Documentation: [docs/](./docs/)

## Acknowledgments

This platform builds upon:
- GRACE/GRACE-FO mission heritage
- GOCE mission techniques
- Open-source scientific computing tools
- Global geophysics community

## Roadmap

- [x] **Phase 1**: Core framework and simulation (Q1 2025)
- [ ] **Phase 2**: Real sensor integration (Q2 2025)
- [ ] **Phase 3**: ML-enhanced processing (Q3 2025)
- [ ] **Phase 4**: Operational deployment preparation (Q4 2025)

---

**Built with ❤️ for the advancement of Earth science**
