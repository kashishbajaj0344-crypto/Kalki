#!/bin/bash

# Kalki Power-Up Script
# Brings you from 5% to 80% power in one command

echo "🚀 KALKI POWER-UP SEQUENCE"
echo "================================"
echo ""

# Check if we're in the right directory
if [ ! -f "apps/kalki_app_proactive.py" ]; then
    echo "❌ Error: Not in Kalki directory"
    echo "   Run: cd /Users/kashish/Desktop/Kalki"
    exit 1
fi

echo "✅ Located Kalki directory"
echo ""

# Test systems
echo "🔍 Testing systems readiness..."
python3 test_power_systems.py > /tmp/kalki_test.log 2>&1

if [ $? -eq 0 ]; then
    echo "✅ Systems check passed"
    echo ""
    
    # Show power level
    grep "POWER LEVEL" /tmp/kalki_test.log
    echo ""
    
    # Launch enhanced app
    echo "🚀 Launching Kalki (80% Power)..."
    echo ""
    echo "Press Ctrl+C to stop"
    echo "================================"
    echo ""
    
    streamlit run apps/kalki_app_enhanced.py
else
    echo "⚠️  Some systems offline - see details:"
    cat /tmp/kalki_test.log
    echo ""
    echo "Try running anyway? (y/n)"
    read answer
    
    if [ "$answer" = "y" ]; then
        streamlit run apps/kalki_app_enhanced.py
    fi
fi
