#!/bin/bash

# GeoSense Platform Server Startup Script
# Starts the FastAPI application on localhost:5050

echo "🛰️  Starting GeoSense Platform API Server..."
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  GALILEO V2.0 - GeoSense Platform"
echo "  Version: 0.4.0"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Warning: No virtual environment detected"
    echo "   Consider activating with: source venv/bin/activate"
    echo ""
fi

# Install dependencies if needed
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "📦 Installing FastAPI dependencies..."
    pip3 install fastapi uvicorn[standard] pydantic -q
    echo "✅ Dependencies installed"
    echo ""
fi

# Start the server
echo "🚀 Server starting at http://localhost:5050"
echo ""
echo "Available endpoints:"
echo "  • Dashboard:     http://localhost:5050"
echo "  • API Docs:      http://localhost:5050/docs"
echo "  • Health Check:  http://localhost:5050/health"
echo ""
echo "Press CTRL+C to stop the server"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Run the server
cd "$(dirname "$0")"
python3 -m uvicorn api.main:app --host 0.0.0.0 --port 5050 --reload
