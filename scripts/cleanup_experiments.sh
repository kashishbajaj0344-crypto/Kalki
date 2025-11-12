#!/bin/bash
# Additional Kalki Cleanup - Remove Games and Experiments
echo "🧹 Removing Car Jam games and experimental projects..."

# Create backup for experimental projects
mkdir -p cleanup_experiments/$(date +%Y%m%d_%H%M%S)

echo "🎮 Moving Car Jam games to backup..."
mkdir -p cleanup_experiments/games
mv car-jam-game/ cleanup_experiments/games/ 2>/dev/null || true
mv car-jam-unity/ cleanup_experiments/games/ 2>/dev/null || true

echo "📋 Moving old design reports..."
mkdir -p cleanup_experiments/designs
mv iron_man_design_report_v2.json cleanup_experiments/designs/ 2>/dev/null || true

echo "🔬 Moving experimental files..."
mkdir -p cleanup_experiments/experiments
# Add any other experimental files here if found

echo "✅ Experimental cleanup complete!"
echo ""
echo "📊 What was removed:"
echo "  - car-jam-game/ (Node.js web game)"
echo "  - car-jam-unity/ (Unity game project)"
echo "  - iron_man_design_report_v2.json (old design report)"
echo ""
echo "💡 Files moved to: cleanup_experiments/"
echo "   Delete this directory if you're sure you don't need the games back."