#!/bin/bash
# Launcher script for KALKI API Server

echo "🚀 Starting KALKI Construction Copilot API Server..."
echo "📖 API Documentation: http://localhost:8000/docs"
echo "🔍 Alternative Docs: http://localhost:8000/redoc"
echo "💚 Health Check: http://localhost:8000/health"
echo ""
echo "⏳ Loading KALKI models (this may take 30-60 seconds)..."
echo ""

# Use Python 3.13 with all packages
python3 kalki_api_server.py
