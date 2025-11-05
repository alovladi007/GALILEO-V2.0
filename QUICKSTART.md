# GALILEO V2.0 - Quick Start Guide

## 🚀 Start All Services (5 minutes)

### Option 1: Docker (Recommended - Easiest)

```bash
# 1. Start infrastructure
docker-compose up -d postgres redis

# 2. Wait for services to be healthy (10 seconds)
docker-compose ps

# 3. Initialize database
python3 scripts/init-database.py

# 4. Start Celery workers
docker-compose up -d celery-worker celery-beat

# 5. Start API server (in separate terminal)
./start_dev_api.sh

# 6. Verify everything is running
curl http://localhost:5050/health
```

**Stop all services:**
```bash
docker-compose down
```

### Option 2: Manual Setup (macOS)

```bash
# 1. Start PostgreSQL
brew services start postgresql@14

# 2. Create database
psql postgres << EOF
CREATE USER gravity WITH PASSWORD 'gravity_secret';
CREATE DATABASE gravity_ops OWNER gravity;
GRANT ALL PRIVILEGES ON DATABASE gravity_ops TO gravity;
\\q
EOF

# 3. Initialize schema
python3 scripts/init-database.py

# 4. Start Redis
brew services start redis

# 5. Start Celery worker (Terminal 1)
celery -A ops.tasks worker --loglevel=info

# 6. Start API server (Terminal 2)
./start_dev_api.sh
```

## ✅ Verification

After starting services, run the integration tests:

```bash
python3 test_integration_v2.py
```

**Expected Results:**
- ✅ PostgreSQL: ~30 database endpoints working
- ✅ Redis/Celery: ~11 task endpoints working
- ✅ Total: ~50/103 endpoints working (48%)

## 📚 Next Steps

1. **Fix remaining bugs** - See ENDPOINT_STATUS_REPORT.md
2. **Add correct test payloads** - See test_integration_v2.py
3. **Configure compliance service** - External vault/HSM setup

---

**Platform Version:** 0.4.0
