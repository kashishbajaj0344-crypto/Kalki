#!/bin/bash
#
# JARVIS Setup Script
# Installs all dependencies and configures the system
#

set -e  # Exit on error

echo "🤖 JARVIS Setup - Personal AI Assistant"
echo "========================================"
echo ""

# Check Python version
echo "🔍 Checking Python version..."
python3 --version || { echo "❌ Python 3 not found!"; exit 1; }
echo "✅ Python 3 found"
echo ""

# Install core dependencies
echo "📦 Installing core dependencies..."
pip3 install --upgrade pip
pip3 install -r requirements.txt
echo "✅ Core dependencies installed"
echo ""

# Install visualization dependencies
echo "🎨 Installing visualization libraries..."
pip3 install matplotlib pymunk control
echo "✅ Visualization libraries installed"
echo ""

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p data/pdfs
mkdir -p data/vector_db
mkdir -p data/knowledge_db
mkdir -p data/training
mkdir -p output/deliverables
mkdir -p memory/episodic
mkdir -p memory/semantic
echo "✅ Directory structure created"
echo ""

# Test installation
echo "🧪 Testing installation..."
python3 -c "
import sys
try:
    import matplotlib
    import pymunk
    import control
    print('✅ All required modules imported successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"
echo ""

# Display system info
echo "💻 System Information:"
python3 -c "
import platform
import psutil
print(f'   OS: {platform.system()} {platform.release()}')
print(f'   CPU Cores: {psutil.cpu_count()}')
print(f'   RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB')
print(f'   Available Storage: {psutil.disk_usage(\"/\").free / (1024**3):.1f} GB')
"
echo ""

# Check for M-series Mac
echo "🚀 Hardware Optimization:"
if [[ $(uname -m) == "arm64" ]]; then
    echo "✅ Apple Silicon detected!"
    echo "   Your M4 Max is optimized for:"
    echo "   - Local LLM inference"
    echo "   - MLX fine-tuning (install with: pip3 install mlx mlx-lm)"
    echo "   - GPU-accelerated rendering"
else
    echo "⚠️  Not running on Apple Silicon"
    echo "   Some features may be slower"
fi
echo ""

# Test hybrid learning system
echo "🧠 Testing Hybrid Learning System..."
python3 test_complete_system.py 2>&1 | head -20
echo ""

# Test CLI
echo "🖥️  Testing CLI..."
python3 kalki_cli.py learn stats
echo ""

echo "🎉 JARVIS SETUP COMPLETE!"
echo ""
echo "Next steps:"
echo "1. Launch chat: python3 kalki_cli.py chat"
echo "2. Ingest PDFs: python3 kalki_cli.py learn ingest your_file.pdf"
echo "3. Generate apps: python3 kalki_cli.py dev app ios MyApp --monetization iap"
echo ""
echo "📖 Read JARVIS_README.md for complete documentation"
echo ""
echo "Welcome to the future! 🤖✨"
