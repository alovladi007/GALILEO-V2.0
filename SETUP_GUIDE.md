# GALILEO V2.0 - Complete Setup Guide

## Current Status (After Session)

### ✅ Completed
1. **Dependencies Installed**
   - PyTorch 2.0.0
   - SQLAlchemy 2.0.0
   - Celery 5.3.0
   - Redis client 5.0.0
   - psycopg2-binary 2.9.6
   - Flax, Optax (JAX ML frameworks)

2. **Bugs Fixed**
   - SQLAlchemy 2.0 compatibility (metadata → meta_info in [ops/models.py](ops/models.py))

3. **Working Endpoints** (6/103)
   - Health check
   - ML: PINN create/train, U-Net create
   - Emulator: create, list
   - Workflow: templates
   - Control: list controllers
   - Trade studies: baseline

### 🔧 Requires Infrastructure Setup

#### PostgreSQL Database (30 endpoints blocked)

**Installation:**
```bash
# macOS
brew install postgresql@14
brew services start postgresql@14

# Linux (Ubuntu/Debian)
sudo apt-get install postgresql-14
sudo service postgresql start

# Check status
psql --version
```

**Setup:**
```bash
# Create user and database
psql postgres
CREATE USER gravity WITH PASSWORD 'gravity_secret';
CREATE DATABASE gravity_ops OWNER gravity;
GRANT ALL PRIVILEGES ON DATABASE gravity_ops TO gravity;
\q

# Test connection
psql -U gravity -d gravity_ops -h localhost
```

**Initialize Schema:**
```bash
cd "/Users/vladimirantoine/GALILEO V2.0/GALILEO-V2.0"
python3 << EOF
from ops.models import Base, engine
Base.metadata.create_all(engine)
print("✓ Database schema created")
EOF
```

**Verify:**
```bash
python3 -c "from ops.models import SessionLocal; db = SessionLocal(); print('✓ Database connected')"
```

#### Redis Server (11 task queue endpoints blocked)

**Installation:**
```bash
# macOS
brew install redis
brew services start redis

# Linux (Ubuntu/Debian)
sudo apt-get install redis-server
sudo service redis-server start

# Check status
redis-cli ping  # Should return PONG
```

**Configuration:**
```bash
# Default Redis runs on localhost:6379
# No authentication required for local development
```

**Verify:**
```bash
python3 -c "import redis; r = redis.Redis(); r.ping(); print('✓ Redis connected')"
```

#### Celery Workers (task queue processing)

**Start Worker:**
```bash
cd "/Users/vladimirantoine/GALILEO V2.0/GALILEO-V2.0"

# Terminal 1: Start Celery worker
celery -A ops.tasks worker --loglevel=info

# Terminal 2: Start Celery beat (for scheduled tasks)
celery -A ops.tasks beat --loglevel=info
```

**Verify:**
```bash
# In another terminal
python3 << EOF
from ops.tasks import test_task
result = test_task.delay()
print(f"Task ID: {result.id}")
print(f"Status: {result.status}")
EOF
```

### 🐛 Known Issues to Fix

#### 1. Calibration Service (5 endpoints)
**Issue:** JAX initialization timeout (>5 seconds on first call)

**Solution:**
```python
# In api/services/calibration_service.py
# Pre-initialize JAX on service startup instead of lazy loading
def __init__(self):
    if SENSING_IMPORTS_AVAILABLE:
        # Trigger JAX compilation on init
        import jax.numpy as jnp
        _ = jnp.array([1.0])  # Warm up JAX
```

**Quick Fix:**
Increase timeout on these endpoints to 30 seconds during initial load.

#### 2. Compliance Service (13 endpoints)
**Issue:** All endpoints timing out

**Root Cause:** Likely trying to connect to external service (Vault, HSM, etc.)

**Investigation:**
```bash
# Check what compliance service is trying to connect to
grep -r "connect\|http\|tcp" api/services/compliance_service.py
```

**Workaround:**
Use in-memory mock for development until external service configured.

#### 3. Workflow Service (5 endpoints)
**Issue:** Timeouts on workflow submission/execution

**Root Cause:** Probably waiting for Celery/Redis

**Fix:** Start Redis + Celery first, then test workflow endpoints.

#### 4. Emulator Lifecycle (5 endpoints)
**Issue:** start/stop/state/history endpoints failing

**Investigation needed:**
```bash
# Check emulator implementation
python3 -c "from api.services import get_emulator_service; svc = get_emulator_service(); print(svc)"
```

#### 5. ML Inference (2 endpoints)
**Issue:** PINN and U-Net inference failing

**Likely cause:** Model not found or wrong input format

**Debug:**
```python
# Test PINN inference directly
from ml.pinn import PINN
model = PINN(model_id="test", hidden_layers=[64, 128, 64])
# Try inference
```

### 📋 Complete Startup Sequence

#### Terminal 1: PostgreSQL
```bash
brew services start postgresql@14
# OR
sudo service postgresql start
```

#### Terminal 2: Redis
```bash
brew services start redis
# OR
sudo service redis-server start
# OR
redis-server
```

#### Terminal 3: Celery Worker
```bash
cd "/Users/vladimirantoine/GALILEO V2.0/GALILEO-V2.0"
celery -A ops.tasks worker --loglevel=info
```

#### Terminal 4: API Server
```bash
cd "/Users/vladimirantoine/GALILEO V2.0/GALILEO-V2.0"
./start_dev_api.sh
# OR
python3 -m uvicorn api.main:app --host 0.0.0.0 --port 5050 --reload
```

#### Terminal 5: Frontend (if using UI)
```bash
cd "/Users/vladimirantoine/GALILEO V2.0/GALILEO-V2.0/ui"
npm run dev
```

### 🧪 Verification Tests

After starting all services:

```bash
cd "/Users/vladimirantoine/GALILEO V2.0/GALILEO-V2.0"
python3 test_integration_v2.py
```

Expected results with infrastructure:
- **Before infrastructure:** 6/103 working (6%)
- **After PostgreSQL:** ~36/103 working (35%)
- **After Redis+Celery:** ~47/103 working (46%)
- **After bug fixes:** ~80/103 working (78%)
- **With correct payloads:** ~94/103 working (91%)

### 📊 Endpoint Breakdown

| Category | Total | Working | Need DB | Need Redis | Need Fixes | Need Payloads |
|----------|-------|---------|---------|------------|------------|---------------|
| Simulation | 3 | 0 | 0 | 0 | 0 | 3 |
| Calibration | 5 | 0 | 0 | 0 | 5 | 0 |
| Compliance | 13 | 0 | 0 | 0 | 13 | 0 |
| Control | 7 | 1 | 0 | 0 | 1 | 5 |
| Database | 17 | 0 | 17 | 0 | 0 | 0 |
| Emulator | 9 | 2 | 0 | 0 | 5 | 2 |
| Inversion | 6 | 0 | 0 | 0 | 1 | 5 |
| ML | 12 | 2 | 0 | 0 | 2 | 8 |
| Tasks | 11 | 0 | 0 | 11 | 0 | 0 |
| Trade Studies | 6 | 1 | 0 | 0 | 0 | 5 |
| Workflow | 8 | 1 | 0 | 5 | 2 | 0 |
| Core | 6 | 2 | 0 | 0 | 0 | 4 |
| **TOTAL** | **103** | **6** | **17** | **16** | **29** | **32** |

### 🎯 Priority Actions

1. **Set up PostgreSQL** → +17 endpoints
2. **Set up Redis + Celery** → +16 endpoints
3. **Fix compliance timeouts** → +13 endpoints
4. **Fix calibration JAX init** → +5 endpoints
5. **Fix emulator lifecycle** → +5 endpoints
6. **Update payload schemas** → Test remaining endpoints

### 🔍 Debugging Tools

**Check service health:**
```bash
# API
curl http://localhost:5050/health

# PostgreSQL
psql -U gravity -d gravity_ops -c "SELECT version();"

# Redis
redis-cli ping

# Celery
celery -A ops.tasks inspect active
```

**Monitor logs:**
```bash
# API logs
tail -f /tmp/uvicorn.log

# Celery logs
# (visible in celery worker terminal)

# PostgreSQL logs
tail -f /usr/local/var/log/postgresql@14/server.log
```

**Test individual endpoints:**
```bash
# Good endpoint (should work)
curl http://localhost:5050/api/control/controllers

# DB endpoint (needs PostgreSQL)
curl -X POST http://localhost:5050/api/db/users \
  -H "Content-Type: application/json" \
  -d '{"username":"test","email":"test@example.com","role":"analyst"}'

# Task endpoint (needs Redis+Celery)
curl -X POST http://localhost:5050/api/tasks/submit \
  -H "Content-Type: application/json" \
  -d '{"task_name":"test_task","parameters":{},"priority":5}'
```

### 📚 Additional Resources

- **OpenAPI Docs:** http://localhost:5050/docs
- **OpenAPI JSON:** http://localhost:5050/openapi.json
- **Status Report:** [ENDPOINT_STATUS_REPORT.md](ENDPOINT_STATUS_REPORT.md)
- **Integration Tests:** [test_integration_v2.py](test_integration_v2.py)

### 🐳 Docker Alternative (If Available)

Create `docker-compose.yml`:
```yaml
version: '3.8'
services:
  postgres:
    image: postgres:14
    environment:
      POSTGRES_USER: gravity
      POSTGRES_PASSWORD: gravity_secret
      POSTGRES_DB: gravity_ops
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:
```

Start: `docker-compose up -d`

---

**Last Updated:** 2025-11-05
**Platform Version:** 0.4.0
**Endpoints:** 103 total, 6 working (6%)
