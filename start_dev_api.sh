#!/bin/bash
#
# GALILEO V2.0 - Development API Server
# ======================================
#
# Starts the FastAPI server with auto-reload for development.
# Excludes UI directory from file watching to prevent errors.
#

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}GALILEO V2.0 Development API Server${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Kill existing process on port 5050
echo "Checking for existing process on port 5050..."
lsof -ti:5050 | xargs kill -9 2>/dev/null && echo "✓ Stopped existing server" || echo "✓ No existing server found"

echo ""
echo "Starting API server with auto-reload..."
echo "  Host: 0.0.0.0"
echo "  Port: 5050"
echo "  Reload: Enabled (excluding ui/, node_modules/)"
echo ""
echo -e "${GREEN}✓ Server starting...${NC}"
echo ""
echo "Access points:"
echo "  • API Docs:     http://localhost:5050/docs"
echo "  • OpenAPI Spec: http://localhost:5050/openapi.json"
echo "  • Health Check: http://localhost:5050/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start uvicorn with proper exclusions
python3 -m uvicorn api.main:app \
  --host 0.0.0.0 \
  --port 5050 \
  --reload \
  --reload-dir api \
  --reload-dir sim \
  --reload-dir inversion \
  --reload-dir control \
  --reload-dir sensing \
  --reload-dir ml \
  --reload-dir emulator \
  --reload-dir compliance \
  --reload-dir trades \
  --reload-dir geophysics \
  --reload-dir ops
