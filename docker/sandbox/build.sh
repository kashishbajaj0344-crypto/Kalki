#!/bin/bash
# Build script for Kalki simulation sandbox

set -e

echo "Building Kalki Simulation Sandbox..."

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

cd "$SCRIPT_DIR"

# Build the Docker image
docker build -t kalki/simulation-sandbox:latest .

echo "Sandbox image built successfully!"
echo "Image: kalki/simulation-sandbox:latest"
echo ""
echo "To run a test simulation:"
echo "docker run --rm -v \$(pwd)/tmp:/app/tmp kalki/simulation-sandbox:latest"