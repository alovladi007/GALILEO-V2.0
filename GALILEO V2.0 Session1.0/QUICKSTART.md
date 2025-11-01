# GeoSense Platform - Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### Prerequisites
- Python 3.11+
- Rust 1.70+
- Node.js 20+
- Docker & Docker Compose
- Git

### Initial Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-org/geosense-platform.git
cd geosense-platform

# 2. Set up Python environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev,ml]"

# 3. Set up Rust
cd control
cargo build
cd ..

# 4. Set up UI
cd ui
npm install
cd ..

# 5. Start all services with Docker
docker-compose up -d
```

### Running Tests

```bash
# Python tests
pytest tests/ --cov

# Rust tests
cargo test --workspace

# UI tests
cd ui && npm test
```

### Development Workflow

```bash
# Start development servers
docker-compose up

# In separate terminals:
# Python services (hot reload)
python -m sim.gravity

# Rust services
cd control && cargo run

# UI (hot reload)
cd ui && npm run dev
```

### Access Points

- **UI Dashboard**: http://localhost:3000
- **API Gateway**: http://localhost:8000
- **PostgreSQL**: localhost:5432
- **Redis**: localhost:6379

### Common Commands

```bash
# Format code
black sim sensing inversion ml ops
cargo fmt --all
cd ui && npm run format

# Lint code
ruff check sim sensing inversion ml ops
cargo clippy --all
cd ui && npm run lint

# Type check
mypy sim sensing inversion ml ops
cd ui && npm run type-check

# Run specific tests
pytest tests/unit/test_gravity.py -v
cargo test dynamics
cd ui && npm test -- components
```

### Architecture Diagrams

View the generated architecture diagrams in `/docs/architecture/`:
- `01_context_diagram.png` - System context
- `02_container_diagram.png` - Container architecture
- `03_component_diagram.png` - Component breakdown

### Documentation

- **Full README**: `/README.md`
- **API Docs**: `/docs/api/`
- **Deployment**: `/docs/deployment/`
- **Ethics**: `/compliance/ETHICS.md`
- **Legal**: `/compliance/LEGAL.md`

### Troubleshooting

**Issue**: Python dependencies fail to install
```bash
pip install --upgrade pip setuptools wheel
pip install -e ".[dev]" --no-cache-dir
```

**Issue**: Rust compilation errors
```bash
cargo clean
cargo update
cargo build
```

**Issue**: Docker containers won't start
```bash
docker-compose down -v
docker-compose up --build
```

### Next Steps

1. Read `/SESSION_0_STATUS.md` for detailed status
2. Review `/docs/architecture/` diagrams
3. Check `/compliance/ETHICS.md` for usage guidelines
4. Start implementing core features (see Session 1 in status doc)

### Getting Help

- **Documentation**: `/docs/`
- **Issues**: Check GitHub issues
- **Contributing**: See `CONTRIBUTING.md` in main README

---

Happy coding! 🚀🛰️
