#!/bin/bash
# Kalki Server Startup Script
# Starts the complete Kalki system with all components

echo "🚀 Starting Kalki Server..."
echo "=================================="

# Set environment
export KALKI_ENV="/Users/kashish/Desktop/Kalki/kalki_env"
export PYTHONPATH="/Users/kashish/Desktop/Kalki:$PYTHONPATH"

# Activate virtual environment
source "$KALKI_ENV/bin/activate"

# Change to Kalki directory
cd "/Users/kashish/Desktop/Kalki"

# Start the unified server
echo "Starting server on http://localhost:8000"
python kalki_unified_server.py