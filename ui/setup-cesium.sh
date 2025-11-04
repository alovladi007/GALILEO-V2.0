#!/bin/bash
# Copy Cesium static assets to public directory
# This script is run automatically by npm postinstall

echo "Setting up Cesium static assets..."

# Create public directory if it doesn't exist
mkdir -p public

# Copy Cesium assets from node_modules
if [ -d "node_modules/cesium/Build/Cesium" ]; then
    cp -r node_modules/cesium/Build/Cesium/* public/
    echo "✓ Cesium assets copied to public directory"
else
    echo "⚠ Warning: Cesium assets not found in node_modules"
    exit 1
fi
