# GALILEO V2.0 - GeoSense Platform

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Proprietary-red.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Status](https://img.shields.io/badge/status-Production%20Ready-success.svg)]()

**Enterprise-Grade AI-Enhanced Space-Based Geophysical Sensing Platform**

A comprehensive, production-ready orbital dynamics, guidance/navigation/control, geophysical inversion, and machine learning platform designed for autonomous satellite-based gravimetry missions. Built with JAX for hardware acceleration, featuring complete security/compliance infrastructure, mission trade analysis, and real-time visualization.

---

## 🎯 Overview

GALILEO V2.0 (GeoSense Platform) is a complete end-to-end solution for space-based gravity field measurement and analysis, integrating:

### Core Capabilities

✨ **Orbital Dynamics & Simulation**
- High-precision orbit propagation with perturbations (J2, drag, SRP)
- Formation flying dynamics (Hill-Clohessy-Wiltshire equations)
- Synthetic data generation with procedural anomaly modeling
- Calibration and noise characterization (Allan deviation, system ID)

✨ **Guidance, Navigation & Control**
- LQR/LQG/MPC controllers for formation flying
- Extended Kalman Filter navigation
- ML-enhanced control with safety systems
- Station-keeping and collision avoidance

✨ **Machine Learning & AI**
- Physics-Informed Neural Networks (PINN) for inversion acceleration
- U-Net for noise reduction and uncertainty estimation
- Reinforcement learning for autonomous control
- Synthetic data generation and training infrastructure

✨ **Geophysical Processing**
- Tikhonov and Bayesian inversion algorithms
- Earth models integration (EGM96, EGM2008, CRUST1.0)
- Seasonal hydrology corrections
- Joint inversion with multiple data types
- Background removal and masking

✨ **Mission Design & Analysis**
- Comprehensive trade studies (baseline, orbit, optical, Pareto)
- 1,000+ design configurations evaluated
- Multi-objective optimization and Pareto front identification
- Decision support with risk assessment

✨ **Quality Assurance**
- Comprehensive benchmarking framework (12 tests, 3 suites)
- Automated regression testing with gold standards
- Code coverage analysis (≥85% target)
- CI/CD integration with GitHub Actions

✨ **Security & Compliance**
- Enterprise-grade RBAC authorization
- Cryptographic audit logging with tamper detection
- Encrypted secrets management (AES-128)
- Data retention and legal hold controls
- GDPR, CCPA, HIPAA, SOX, PCI-DSS compliance

✨ **Operations & Deployment**
- FastAPI backend with async task processing (Celery)
- Next.js 14 web UI with CesiumJS 3D visualization
- PostgreSQL + TimescaleDB for time-series data
- Docker orchestration with monitoring (Grafana, Prometheus)
- MinIO object storage for large datasets

---

## 📦 Installation

### Prerequisites

- Python 3.11 or higher
- Node.js 18+ (for UI components)
- Docker (optional, for containerized deployment)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/alovladi007/GALILEO-V2.0.git
cd GALILEO-V2.0

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev,ml,control]"
```

### Optional Dependencies

```bash
# Development tools (pytest, mypy, black, ruff)
pip install -e ".[dev]"

# Machine learning support (PyTorch, Flax)
pip install -e ".[ml]"

# Control systems (cvxpy for MPC)
pip install -e ".[control]"

# All optional dependencies
pip install -e ".[dev,ml,control]"
```

---

## 🌐 Run on Localhost

The platform consists of two components:
1. **FastAPI Backend** (port 5050) - Simulation, computation, and data API
2. **Next.js Frontend** (port 3000) - 3D visualization with CesiumJS

### Step 1: Start the FastAPI Backend

```bash
# Quick start - run the startup script
./start_server.sh

# Or manually with uvicorn
python3 -m uvicorn api.main:app --reload --host 0.0.0.0 --port 5050
```

Backend endpoints:
- **API Documentation**: http://localhost:5050/docs (Interactive Swagger UI)
- **Health Check**: http://localhost:5050/health

### Step 2: Start the Next.js UI

```bash
# Navigate to UI folder
cd ui

# Install dependencies (first time only)
npm install

# Set up Cesium Ion token (required for 3D globe)
cp .env.local.example .env.local
# Edit .env.local and add your Cesium Ion token from https://ion.cesium.com/

# Run development server
npm run dev
```

Frontend:
- **3D Dashboard**: http://localhost:3001 (or http://localhost:3000)
- **Features**: Real-time orbit visualization, gravity anomaly mapping, mission dashboard

---

## 🚀 Quick Examples

### Orbit Propagation with Perturbations

```python
import jax.numpy as jnp
from sim.dynamics import (
    perturbed_dynamics,
    propagate_orbit_jax,
    orbital_elements_to_cartesian,
)

# Define orbital elements (a, e, i, Ω, ω, ν)
oe = jnp.array([7000.0, 0.001, 98.0, 0.0, 0.0, 0.0])  # SSO LEO
state0 = orbital_elements_to_cartesian(oe)

# Propagate with J2, drag, and SRP
times, states = propagate_orbit_jax(
    perturbed_dynamics,
    state0,
    t_span=(0.0, 5400.0),  # 90 minutes
    dt=10.0
)

print(f"Propagated {len(states)} states with perturbations")
```

### Geophysical Inversion with PINN Acceleration

```python
from inversion import TikhonovInversion, InversionConfig
from ml.pinn import PINNInversionAccelerator

# Traditional inversion
config = InversionConfig(
    regularization_parameter=1e-6,
    max_iterations=100,
    tolerance=1e-8
)

inversion = TikhonovInversion(config)
mass_dist = inversion.solve(gravity_data, obs_matrix)

# ML-accelerated inversion (Session 6)
pinn = PINNInversionAccelerator(
    layers=[64, 128, 128, 64],
    activation='tanh'
)
pinn.train(training_data, epochs=1000)
ml_solution = pinn.predict(gravity_data)  # 10-100× faster
```

### Mission Trade Studies

```python
from trades.pareto_analysis import ParetoAnalysis

# Multi-objective optimization
analysis = ParetoAnalysis()
designs, objectives, analyses = analysis.run_pareto_analysis()

# Identify Pareto-optimal configurations
pareto_front = analysis.identify_pareto_front(designs, objectives)
print(f"Found {len(pareto_front)} Pareto-optimal designs")

# Generate visualization
analysis.plot_pareto_fronts(designs, objectives, analyses, 'plots/')
```

### Security & Compliance

```python
from compliance import AuthorizationManager, AuditLogger, SecretsManager

# Authorization
auth_manager = AuthorizationManager()
auth_manager.assign_user_role("researcher", "research_restricted")

# Audit logging
audit = AuditLogger()
audit.log_access(
    user_id="researcher",
    resource="gravity_data",
    action="read",
    granted=True
)

# Secrets management
secrets = SecretsManager()
secret = secrets.create_secret(
    name="api_key",
    value="sk_live_abc123",
    secret_type=SecretType.API_KEY,
    rotation_policy_days=90
)
```

---

## 📁 Repository Structure

```
GALILEO-V2.0/
│
├── 🚀 Core Simulation & Dynamics
│   ├── sim/                          # Orbital simulation
│   │   ├── dynamics/                 # Orbital dynamics (Sessions 0-1)
│   │   │   ├── keplerian.py         # Two-body dynamics (319 lines)
│   │   │   ├── perturbations.py     # J2, drag, SRP (393 lines)
│   │   │   ├── relative.py          # Formation flying (296 lines)
│   │   │   └── propagators.py       # RK4 integration (231 lines)
│   │   ├── gravity.py               # Gravity field modeling (EGM2008)
│   │   ├── synthetic.py             # Synthetic data generation (Session 4)
│   │   ├── calibration.py           # Calibration & noise (Session 9)
│   │   ├── system_id.py             # System identification (Session 9)
│   │   └── cal_maneuvers.py         # Calibration maneuvers (Session 9)
│   │
│   ├── sensing/                      # Sensor processing (Sessions 1-3)
│   │   ├── model.py                 # Measurement models
│   │   ├── allan.py                 # Allan deviation
│   │   ├── noise.py                 # Noise characterization
│   │   └── phase_model.py           # Phase measurements
│   │
├── 🎯 Control & Navigation
│   ├── control/                     # GNC systems (Sessions 2-3)
│   │   ├── controllers/             # Control algorithms
│   │   │   ├── lqr.py              # LQR controller (528 lines)
│   │   │   ├── lqg.py              # LQG with Kalman filter (555 lines)
│   │   │   ├── mpc.py              # Model Predictive Control (630 lines)
│   │   │   ├── mpc_ml.py           # ML-enhanced MPC (476 lines)
│   │   │   ├── station_keeping.py  # Station-keeping (682 lines)
│   │   │   ├── safety_ml.py        # ML safety systems (675 lines)
│   │   │   └── collision_avoidance.py # Collision avoidance (633 lines)
│   │   └── navigation/             # State estimation
│   │       └── ekf.py              # Extended Kalman Filter (636 lines)
│   │
├── 🔬 Geophysical Processing
│   ├── inversion/                    # Inversion algorithms (Session 5)
│   │   ├── solvers.py               # Tikhonov, Bayesian
│   │   └── regularizers.py          # Regularization methods
│   │
│   ├── geophysics/                   # Earth models (Session 10)
│   │   ├── gravity_fields.py        # EGM96, EGM2008
│   │   ├── crustal_models.py        # CRUST1.0
│   │   ├── hydrology.py             # Seasonal water storage
│   │   ├── masking.py               # Ocean/land/ice masks
│   │   └── joint_inversion.py       # Multi-physics inversion
│   │
├── 🤖 Machine Learning
│   ├── ml/                          # ML models (Sessions 3, 6)
│   │   ├── models.py               # Neural architectures (608 lines)
│   │   ├── pinn.py                 # Physics-Informed NN (Session 6)
│   │   ├── unet.py                 # U-Net for noise reduction (Session 6)
│   │   ├── train.py                # Training infrastructure (Session 6)
│   │   ├── reinforcement.py        # RL algorithms (651 lines)
│   │   ├── training.py             # Training infrastructure (685 lines)
│   │   └── inference.py            # Deployment & optimization (651 lines)
│   │
├── 📊 Analysis & Quality
│   ├── bench/                       # Benchmarking (Session 11)
│   │   ├── __init__.py
│   │   ├── metrics.py              # Performance metrics (550 lines)
│   │   └── datasets.py             # Regression datasets (480 lines)
│   ├── bench.py                     # Benchmark runner (580 lines)
│   │
│   ├── trades/                      # Mission trade studies (Session 12)
│   │   ├── baseline_study.py       # Baseline/noise/sensitivity
│   │   ├── orbit_study.py          # Orbit configuration
│   │   ├── optical_study.py        # Optical system design
│   │   └── pareto_analysis.py      # Multi-objective optimization
│   ├── run_trades.py                # Trade study runner
│   │
├── 🔒 Security & Compliance
│   ├── compliance/                  # Security framework (Session 13)
│   │   ├── authorization.py        # RBAC (320 lines)
│   │   ├── audit.py                # Audit logging (340 lines)
│   │   ├── secrets.py              # Secrets management (310 lines)
│   │   └── retention.py            # Data lifecycle (360 lines)
│   ├── security_scan.py             # Security scanner
│   ├── ETHICS.md                    # Ethical guidelines
│   └── LEGAL.md                     # Legal requirements
│   │
├── 🌐 Backend & Operations
│   ├── ops/                         # Backend operations (Session 7)
│   │   ├── tasks.py                # Celery task definitions
│   │   ├── jobs.py                 # Job management
│   │   └── telemetry.py            # Telemetry processing
│   │
│   ├── api/                         # REST API (Session 7)
│   │   ├── main.py                 # FastAPI application
│   │   ├── routes/                 # API endpoints
│   │   └── schemas/                # Pydantic models
│   │
├── 🎨 Frontend & Visualization
│   ├── ui/                          # Next.js 14 UI (Session 8)
│   │   ├── src/
│   │   │   ├── app/                # Next.js app router
│   │   │   ├── components/         # React components
│   │   │   │   ├── GlobeViewer.tsx # CesiumJS 3D globe
│   │   │   │   ├── OrbitViz.tsx    # Orbit visualization
│   │   │   │   └── Dashboard.tsx   # Mission dashboard
│   │   │   └── lib/                # Utilities
│   │   ├── Dockerfile
│   │   └── package.json
│   │
├── 🔬 Laboratory Emulation
│   ├── emulator/                    # Optical bench emulator (Session 14)
│   │   ├── optical_bench.py         # Core emulator (400+ lines)
│   │   ├── server.py                # WebSocket streaming (250+ lines)
│   │   ├── dashboard.html           # Interactive UI (700+ lines)
│   │   ├── dashboard_server.py      # HTTP server
│   │   ├── start_emulator.py        # Master startup script
│   │   ├── demo_basic.py            # Basic operation demo
│   │   ├── demo_events.py           # Event injection demo
│   │   ├── demo_streaming.py        # Streaming demo
│   │   └── QUICKSTART.md            # 5-minute setup guide
│   │
├── 🧪 Testing & Quality
│   ├── tests/
│   │   ├── unit/                   # Unit tests
│   │   ├── integration/            # Integration tests
│   │   ├── security/               # Security tests (Session 13)
│   │   │   └── test_compliance.py # 35 compliance tests
│   │   ├── test_bench.py          # Benchmark tests (Session 11)
│   │   ├── test_inversion.py      # Inversion tests (Session 5)
│   │   ├── test_ml.py             # ML tests (Session 6)
│   │   └── test_geophysics.py     # Geophysics tests (Session 10)
│   │
├── 📖 Documentation
│   ├── docs/
│   │   ├── physics_model.md        # Physics documentation
│   │   ├── calibration.md          # Calibration guide (Session 9)
│   │   ├── earth_models.md         # Earth models (Session 10)
│   │   ├── verification.md         # Benchmarking guide (Session 11)
│   │   ├── security_compliance.md  # Security docs (Session 13)
│   │   ├── emulation.md            # Emulator guide (Session 14)
│   │   ├── decisions/              # Design decisions
│   │   │   └── trade_studies.md   # Trade study memo (Session 12)
│   │   ├── figures/                # Visualizations
│   │   │   ├── allan_deviation_vs_time.png
│   │   │   ├── baseline_trade_study.png (Session 12)
│   │   │   ├── orbit_trade_study.png (Session 12)
│   │   │   ├── optical_trade_study.png (Session 12)
│   │   │   └── pareto_fronts.png (Session 12)
│   │   └── architecture/           # Architecture diagrams
│   │
├── 📋 Examples & Scripts
│   ├── examples/
│   │   ├── complete_geophysics_example.py (Session 10)
│   │   ├── getting_started.py      (Session 10)
│   │   ├── example_usage.py        (Session 11)
│   │   └── demo.py                 (Session 13)
│   │
│   ├── benchmarks/
│   │   └── background_removal_benchmarks.py (Session 10)
│   │
│   ├── scripts/
│   │   └── noise_budget_analysis.py
│   │
├── 🐳 DevOps & Infrastructure
│   ├── .github/
│   │   └── workflows/
│   │       └── benchmark.yml       # CI/CD (Session 11)
│   │
│   ├── docker-compose.yml          # Container orchestration
│   ├── Dockerfile
│   │
├── ⚙️ Configuration
│   ├── pyproject.toml              # Python package config
│   ├── pytest.ini                  # Test configuration
│   ├── setup.py                    # Installation script
│   ├── requirements.txt            # Core dependencies
│   ├── SESSION_*_requirements.txt  # Session-specific deps
│   │
└── 📊 Outputs & Reports
    ├── trade_stats.json            # Trade study results (Session 12)
    ├── index.html                  # Interactive dashboard (Session 12)
    └── SESSION_*_*.md              # Session documentation
```

---

## 🎓 Key Features by Session

### Session 0-1: Physics Foundation & Sensing
- ✅ Keplerian dynamics and perturbations (J2, drag, SRP)
- ✅ Formation flying (Hill-Clohessy-Wiltshire)
- ✅ Laser interferometry and noise characterization
- ✅ Allan deviation analysis

### Session 2-3: GNC & Machine Learning
- ✅ Complete GNC suite (LQR, LQG, MPC, EKF)
- ✅ Neural networks (LSTM, VAE, GNN, Attention)
- ✅ Reinforcement learning (PPO, SAC, multi-agent)
- ✅ ML-enhanced control with safety systems

### Session 4: Synthetic Data Generation
- ✅ Procedural subsurface anomaly generation
- ✅ Forward gravity modeling
- ✅ Telemetry and phase data synthesis

### Session 5-6: Inversion & ML Acceleration
- ✅ Tikhonov and Bayesian inversion
- ✅ Physics-Informed Neural Networks (PINN)
- ✅ U-Net for noise reduction
- ✅ Uncertainty estimation

### Session 7-8: Backend & Web UI
- ✅ FastAPI backend with Celery workers
- ✅ PostgreSQL + TimescaleDB + MinIO
- ✅ Next.js 14 web interface
- ✅ CesiumJS 3D globe visualization

### Session 9: Calibration & Noise Characterization
- ✅ Allan deviation and whiteness tests
- ✅ Drag and solar pressure estimation
- ✅ Calibration maneuver design
- ✅ Validation suite

### Session 10: Earth Models & Geophysics
- ✅ EGM96/EGM2008 gravity fields
- ✅ CRUST1.0 crustal model
- ✅ Seasonal hydrology corrections
- ✅ Joint multi-physics inversion

### Session 11: Verification & Benchmarking
- ✅ 12 comprehensive benchmark tests
- ✅ Automated regression testing
- ✅ Code coverage analysis (≥85%)
- ✅ CI/CD integration

### Session 12: Mission Trade Studies
- ✅ Baseline, orbit, optical trade analyses
- ✅ Pareto front optimization
- ✅ 1,000+ design configurations evaluated
- ✅ Decision support documentation

### Session 13: Security & Compliance
- ✅ Enterprise RBAC authorization
- ✅ Cryptographic audit logging
- ✅ AES-128 secrets management
- ✅ GDPR/CCPA/HIPAA/SOX/PCI-DSS compliance

### Session 14: Laboratory Emulation Mode
- ✅ Short-baseline optical bench emulator (1m, 632.8nm He-Ne)
- ✅ Real-time WebSocket streaming (50-1000 Hz)
- ✅ Interactive web dashboard with Chart.js
- ✅ Synthetic signal injection (thermal, vibration, laser, phase)
- ✅ Environmental effects modeling

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific test suites
pytest tests/unit/ -v                    # Unit tests
pytest tests/integration/ -v             # Integration tests
pytest tests/security/test_compliance.py # Security tests (35 tests)

# Run benchmarks
python bench.py --suite all              # All benchmark suites
python run_trades.py                     # Trade studies

# Security scan
python security_scan.py                  # Automated security analysis
```

---

## 🐳 Docker Deployment

Complete Docker Compose setup for production:

```bash
# Start all services
docker-compose up -d

# Services:
# - api:        FastAPI backend (port 8000)
# - worker:     Celery task queue
# - ui:         Next.js frontend (port 3000)
# - redis:      Cache & message broker
# - postgres:   Metadata storage
# - timescale:  Time-series telemetry
# - minio:      Object storage
# - grafana:    Monitoring dashboard (port 3001)
# - prometheus: Metrics collection (port 9090)
# - jaeger:     Distributed tracing (port 16686)

# View logs
docker-compose logs -f api

# Stop all services
docker-compose down
```

---

## 📊 Performance

Benchmarked on Intel Core i9-12900K, Python 3.11, JAX 0.4.20:

| Operation | Time | Notes |
|-----------|------|-------|
| Two-body propagation (90 min) | ~45 ms | JIT-compiled |
| Perturbed dynamics (J2+drag+SRP) | ~120 ms | JIT-compiled |
| Formation flying (100 min) | ~35 ms | Analytical + RK4 |
| Tikhonov inversion (1000×1000) | ~180 ms | NumPy backend |
| PINN inference | ~5 ms | 10-100× faster than traditional |
| Benchmark suite (12 tests) | ~1.73s | All suites |
| Trade studies (1000 configs) | ~25s | Pareto analysis |

*First run includes JIT compilation overhead (~1-2 seconds)*

---

## 📖 Documentation

### User Guides
- [Physics & Sensing](docs/physics_model.md) - Session 0-1 documentation
- [Calibration Guide](docs/calibration.md) - Session 9 calibration procedures
- [Earth Models](docs/earth_models.md) - Session 10 geophysics guide
- [Verification & Benchmarking](docs/verification.md) - Session 11 testing guide
- [Security & Compliance](docs/security_compliance.md) - Session 13 security framework
- [Laboratory Emulation](docs/emulation.md) - Session 14 emulator guide

### Technical Documentation
- [Trade Studies](docs/decisions/trade_studies.md) - Session 12 design decisions (30 pages)
- [Ethical Guidelines](ETHICS.md) - Research restrictions and ethical framework
- [Legal Requirements](LEGAL.md) - Compliance and legal framework

### Session Documentation
- [SESSION_0_STATUS.md](SESSION_0_STATUS.md) - Architecture setup
- [SESSION_1_README.md](SESSION_1_README.md) - Physics foundation
- [SESSION_2_COMPLETE.md](SESSION_2_COMPLETE.md) - GNC systems
- [SESSIONS_5_6_COMPLETE.md](SESSIONS_5_6_COMPLETE.md) - Inversion & ML
- [SESSION_7_8_README.md](SESSION_7_8_README.md) - Backend & UI
- [SESSION_9_README.md](SESSION_9_README.md) - Calibration
- [SESSION_10_README.md](SESSION_10_README.md) - Geophysics
- [SESSION_11_README.md](SESSION_11_README.md) - Benchmarking
- [SESSION_12_README.md](SESSION_12_README.md) - Trade studies
- [SESSION_13_README.md](SESSION_13_README.md) - Security & compliance

---

## 🛠️ Development

### Code Quality

```bash
# Format code
black . --exclude venv
isort . --skip venv

# Lint
ruff check . --exclude venv

# Type check
mypy sim/ inversion/ ml/ compliance/

# Run security scan
python security_scan.py
```

### Contributing

```bash
# Install with development dependencies
pip install -e ".[dev,ml,control]"

# Install pre-commit hooks (if using)
pre-commit install

# Run tests before committing
pytest tests/ --cov=. --cov-report=term
python bench.py --suite all
```

---

## 📊 Repository Statistics

![Size](https://img.shields.io/github/repo-size/alovladi007/GALILEO-V2.0)
![Files](https://img.shields.io/github/directory-file-count/alovladi007/GALILEO-V2.0)
![Last Commit](https://img.shields.io/github/last-commit/alovladi007/GALILEO-V2.0)

**Current Status**:
- **Sessions Integrated**: 14 (0-14) ✅ **Complete**
- **Total Files**: 114+
- **Total Code**: 31,245+ lines
- **Python Files**: 60+ production modules
- **Tests**: 35+ (compliance) + 25+ (benchmarking) + unit/integration
- **Documentation**: 16,000+ words across all sessions
- **Code Quality**: Type-safe, well-documented, security-scanned
- **Structure**: Production-ready with enterprise security

### Session Breakdown

| Session | Focus | Status | Files | Lines |
|---------|-------|--------|-------|-------|
| 0-1 | Physics & Sensing | ✅ | 8 | 4,018 |
| 2-3 | GNC & ML | ✅ | 3 | 707 |
| 4 | Synthetic Data | ✅ | - | - |
| 5-6 | Inversion & PINN | ✅ | 3 | 1,199 |
| 6-8 | ML, Backend, UI | ✅ | 5 | 541 |
| 9 | Calibration | ✅ | 14 | 4,535 |
| 10 | Geophysics | ✅ | 19 | 6,589 |
| 11 | Benchmarking | ✅ | 16 | 5,898 |
| 12 | Trade Studies | ✅ | 15 | ~1,500 |
| 13 | Security & Compliance | ✅ | 17 | ~2,480 |
| 14 | Laboratory Emulation | ✅ | 14 | ~3,778 |
| **Total** | **All Sessions** | **✅** | **114+** | **31,245+** |

---

## 🔗 Related Projects

- **[JAX](https://github.com/google/jax)**: High-performance numerical computing
- **[CesiumJS](https://cesium.com/platform/cesiumjs/)**: 3D geospatial visualization
- **[FastAPI](https://fastapi.tiangolo.com/)**: Modern Python web framework
- **[Next.js](https://nextjs.org/)**: React framework for production
- **[Orekit](https://www.orekit.org/)**: Space dynamics library (Java)
- **[Poliastro](https://github.com/poliastro/poliastro)**: Python astrodynamics

---

## 📄 License

**Proprietary - Research Use Only**

This software is provided for research and educational purposes. See [LEGAL.md](LEGAL.md) for detailed terms and compliance requirements.

---

## 🙏 Acknowledgments

- **Physics Models**: Based on Curtis (2013), Vallado (2013)
- **Gravity Field**: Uses EGM96/EGM2008 models (NGA)
- **Earth Models**: CRUST1.0, GLDAS hydrology data
- **JAX Team**: For outstanding numerical computing framework
- **Open Source Community**: For tools and libraries

---

## 📞 Contact

**Project**: GALILEO V2.0 (GeoSense Platform)
**Repository**: https://github.com/alovladi007/GALILEO-V2.0
**Issues**: https://github.com/alovladi007/GALILEO-V2.0/issues

For security issues: See [LEGAL.md](LEGAL.md) for contact information.

---

<div align="center">

**Built with ❤️ for Space Science**

**Status**: ✅ Production Ready | **Version**: 2.0 | **Sessions**: 13/13 Complete

[Documentation](docs/) · [Report Bug](https://github.com/alovladi007/GALILEO-V2.0/issues) · [Request Feature](https://github.com/alovladi007/GALILEO-V2.0/issues)

</div>
