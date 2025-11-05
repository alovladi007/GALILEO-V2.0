#!/bin/bash
#
# GALILEO V2.0 - Shutdown Script
# ================================
#
# Gracefully stops all platform components
#

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

print_header "Stopping GALILEO V2.0"

# Stop by PID files
if [ -f "logs/api.pid" ]; then
    API_PID=$(cat logs/api.pid)
    if ps -p $API_PID > /dev/null 2>&1; then
        kill $API_PID
        print_success "Stopped API (PID: $API_PID)"
    fi
    rm logs/api.pid
fi

if [ -f "logs/ui.pid" ]; then
    UI_PID=$(cat logs/ui.pid)
    if ps -p $UI_PID > /dev/null 2>&1; then
        kill $UI_PID
        print_success "Stopped UI (PID: $UI_PID)"
    fi
    rm logs/ui.pid
fi

if [ -f "logs/emulator.pid" ]; then
    EMULATOR_PID=$(cat logs/emulator.pid)
    if ps -p $EMULATOR_PID > /dev/null 2>&1; then
        kill $EMULATOR_PID
        print_success "Stopped Emulator (PID: $EMULATOR_PID)"
    fi
    rm logs/emulator.pid
fi

# Kill by port (fallback)
echo ""
echo "Cleaning up ports..."

for port in 5050 3000 8080 8765; do
    PID=$(lsof -ti:$port 2>/dev/null)
    if [ ! -z "$PID" ]; then
        kill -9 $PID 2>/dev/null
        print_success "Cleaned up port $port"
    fi
done

print_success "All services stopped"
