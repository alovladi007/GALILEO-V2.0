#!/bin/bash
#
# GALILEO V2.0 - Master Startup Script
# =====================================
#
# Starts all platform components in the correct order:
# 1. Dependency checks and installation
# 2. Configuration initialization
# 3. Backend API (FastAPI)
# 4. Frontend UI (Next.js)
# 5. Optional: Emulator, Database, Monitoring
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# ============================================================================
# Pre-flight Checks
# ============================================================================

print_header "GALILEO V2.0 Startup"
echo ""
echo "Enterprise-grade AI-enhanced space-based geophysical sensing platform"
echo "Version: 2.0.0"
echo ""

# Check Python version
print_header "Checking Prerequisites"

if ! command -v python3 &> /dev/null; then
    print_error "Python 3 not found. Please install Python 3.11+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
print_success "Python $PYTHON_VERSION found"

# Check Node.js version
if ! command -v node &> /dev/null; then
    print_warning "Node.js not found. UI will not be available."
    HAS_NODE=false
else
    NODE_VERSION=$(node --version)
    print_success "Node.js $NODE_VERSION found"
    HAS_NODE=true
fi

# ============================================================================
# Dependency Installation
# ============================================================================

print_header "Installing Dependencies"

# Core dependencies (lightweight, essential)
CORE_DEPS="fastapi uvicorn numpy scipy matplotlib pydantic"

echo "Installing core dependencies..."
pip3 install --break-system-packages --quiet --upgrade pip
pip3 install --break-system-packages --quiet $CORE_DEPS 2>&1 | grep -v "Requirement already satisfied" || true

print_success "Core dependencies installed"

# Optional: Install JAX (can be skipped for testing)
echo ""
read -p "Install JAX for full simulation capabilities? (y/n, default=n): " -n 1 -r INSTALL_JAX
echo ""

if [[ $INSTALL_JAX =~ ^[Yy]$ ]]; then
    print_warning "Installing JAX... this may take a few minutes"
    pip3 install --break-system-packages --quiet "jax[cpu]==0.4.20" jaxlib==0.4.20 || {
        print_warning "JAX installation failed, continuing without it"
    }
fi

# Optional: Install ML dependencies
echo ""
read -p "Install ML dependencies (PyTorch, etc.)? (y/n, default=n): " -n 1 -r INSTALL_ML
echo ""

if [[ $INSTALL_ML =~ ^[Yy]$ ]]; then
    print_warning "Installing ML dependencies... this will take several minutes"
    pip3 install --break-system-packages --quiet torch optax flax || {
        print_warning "ML installation failed, continuing without it"
    }
fi

# ============================================================================
# Configuration Initialization
# ============================================================================

print_header "Initializing Configuration"

# Create directories
mkdir -p data outputs checkpoints logs

# Initialize configuration if it doesn't exist
if [ ! -f "galileo_config.json" ]; then
    echo "Creating default configuration..."
    python3 -c "from config import config; config.save('galileo_config.json')" 2>/dev/null || {
        print_warning "Could not create config file, will use defaults"
    }
fi

print_success "Configuration initialized"

# ============================================================================
# Start Backend API
# ============================================================================

print_header "Starting Backend API"

# Kill existing processes on port 5050
lsof -ti:5050 | xargs kill -9 2>/dev/null || true

echo "Starting FastAPI server on port 5050..."

# Start in background
python3 -m uvicorn api.main:app --host 0.0.0.0 --port 5050 --reload=false > logs/api.log 2>&1 &
API_PID=$!

# Wait for API to be ready
echo "Waiting for API to start..."
for i in {1..30}; do
    if curl -s http://localhost:5050/health > /dev/null 2>&1; then
        print_success "Backend API started (PID: $API_PID)"
        echo "  API Documentation: http://localhost:5050/docs"
        echo "  Health Check: http://localhost:5050/health"
        break
    fi
    sleep 1
    if [ $i -eq 30 ]; then
        print_error "API failed to start. Check logs/api.log"
        cat logs/api.log
        exit 1
    fi
done

# ============================================================================
# Start Frontend UI (if Node.js available)
# ============================================================================

if [ "$HAS_NODE" = true ]; then
    print_header "Starting Frontend UI"

    cd ui

    # Install dependencies if needed
    if [ ! -d "node_modules" ]; then
        echo "Installing Node.js dependencies..."
        npm install --silent || {
            print_error "npm install failed"
            cd ..
            exit 1
        }
    fi

    # Kill existing processes on port 3000
    lsof -ti:3000 | xargs kill -9 2>/dev/null || true

    echo "Starting Next.js development server on port 3000..."

    # Start in background
    npm run dev > ../logs/ui.log 2>&1 &
    UI_PID=$!

    cd ..

    # Wait for UI to be ready
    echo "Waiting for UI to start..."
    for i in {1..60}; do
        if curl -s http://localhost:3000 > /dev/null 2>&1; then
            print_success "Frontend UI started (PID: $UI_PID)"
            echo "  Web Interface: http://localhost:3000"
            break
        fi
        sleep 1
        if [ $i -eq 60 ]; then
            print_warning "UI failed to start. Check logs/ui.log"
        fi
    done
else
    print_warning "Skipping UI startup (Node.js not available)"
fi

# ============================================================================
# Optional: Start Emulator
# ============================================================================

echo ""
read -p "Start laboratory emulator? (y/n, default=n): " -n 1 -r START_EMULATOR
echo ""

if [[ $START_EMULATOR =~ ^[Yy]$ ]]; then
    print_header "Starting Laboratory Emulator"

    # Kill existing processes
    lsof -ti:8765 | xargs kill -9 2>/dev/null || true
    lsof -ti:8080 | xargs kill -9 2>/dev/null || true

    python3 emulator/start_emulator.py > logs/emulator.log 2>&1 &
    EMULATOR_PID=$!

    sleep 2
    print_success "Emulator started (PID: $EMULATOR_PID)"
    echo "  Dashboard: http://localhost:8080/dashboard.html"
    echo "  WebSocket: ws://localhost:8765"
fi

# ============================================================================
# Summary and Instructions
# ============================================================================

print_header "GALILEO V2.0 Started Successfully!"

echo ""
echo "Services Running:"
echo "  ✓ Backend API:  http://localhost:5050/docs"
if [ "$HAS_NODE" = true ]; then
    echo "  ✓ Frontend UI:  http://localhost:3000"
fi
if [[ $START_EMULATOR =~ ^[Yy]$ ]]; then
    echo "  ✓ Emulator:     http://localhost:8080/dashboard.html"
fi

echo ""
echo "Process IDs:"
echo "  API: $API_PID"
if [ "$HAS_NODE" = true ]; then
    echo "  UI:  $UI_PID"
fi
if [[ $START_EMULATOR =~ ^[Yy]$ ]]; then
    echo "  Emulator: $EMULATOR_PID"
fi

echo ""
echo "Logs:"
echo "  API:      tail -f logs/api.log"
if [ "$HAS_NODE" = true ]; then
    echo "  UI:       tail -f logs/ui.log"
fi
if [[ $START_EMULATOR =~ ^[Yy]$ ]]; then
    echo "  Emulator: tail -f logs/emulator.log"
fi

echo ""
echo "To stop all services:"
echo "  ./stop_galileo.sh"
echo ""
echo "To view module status:"
echo "  curl http://localhost:5050/api/modules"
echo ""

# Save PIDs for cleanup
echo "$API_PID" > logs/api.pid
if [ "$HAS_NODE" = true ]; then
    echo "$UI_PID" > logs/ui.pid
fi
if [[ $START_EMULATOR =~ ^[Yy]$ ]]; then
    echo "$EMULATOR_PID" > logs/emulator.pid
fi

print_success "System ready!"

# ============================================================================
# Keep running (optional)
# ============================================================================

echo ""
read -p "Keep this terminal open to monitor logs? (y/n, default=y): " -n 1 -r KEEP_RUNNING
echo ""

if [[ ! $KEEP_RUNNING =~ ^[Nn]$ ]]; then
    echo ""
    echo "Monitoring logs... Press Ctrl+C to stop all services"
    echo ""

    # Trap Ctrl+C to clean shutdown
    trap './stop_galileo.sh' INT

    # Monitor main API log
    tail -f logs/api.log
fi

echo ""
echo "Services are running in the background."
echo "Use ./stop_galileo.sh to stop all services."
