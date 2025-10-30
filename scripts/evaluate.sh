#!/bin/bash
# Kalki Evaluation Script - Phase 20
# Runs the complete test harness and generates evaluation reports

set -e

echo "========================================="
echo "Kalki Multi-Agent System Evaluation"
echo "========================================="
echo ""

# Create reports directory if it doesn't exist
mkdir -p reports

# Get timestamp for report filenames
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
REPORT_DIR="reports/eval_${TIMESTAMP}"
mkdir -p "${REPORT_DIR}"

echo "Report directory: ${REPORT_DIR}"
echo ""

# Run unit tests
echo "Running unit tests..."
python3 -m unittest discover -s tests -p "test_*.py" -v > "${REPORT_DIR}/unit_tests.log" 2>&1
UNIT_TEST_RESULT=$?

if [ $UNIT_TEST_RESULT -eq 0 ]; then
    echo "✓ Unit tests passed"
else
    echo "✗ Unit tests failed (see ${REPORT_DIR}/unit_tests.log)"
fi
echo ""

# Run adversarial tests
echo "Running adversarial test harness..."
python3 scripts/run_adversarial_tests.py --output "${REPORT_DIR}/adversarial_report.json"
ADV_TEST_RESULT=$?

if [ $ADV_TEST_RESULT -eq 0 ]; then
    echo "✓ Adversarial tests completed"
else
    echo "✗ Adversarial tests failed"
fi
echo ""

# Generate performance metrics
echo "Generating performance metrics..."
python3 scripts/generate_metrics.py --output "${REPORT_DIR}/metrics_report.json"
METRICS_RESULT=$?

if [ $METRICS_RESULT -eq 0 ]; then
    echo "✓ Performance metrics generated"
else
    echo "✗ Metrics generation failed"
fi
echo ""

# Create summary report
echo "Creating summary report..."
cat > "${REPORT_DIR}/summary.txt" << EOL
Kalki Multi-Agent System Evaluation Report
Generated: $(date)

Test Results:
-------------
Unit Tests: $([ $UNIT_TEST_RESULT -eq 0 ] && echo "PASSED" || echo "FAILED")
Adversarial Tests: $([ $ADV_TEST_RESULT -eq 0 ] && echo "PASSED" || echo "FAILED")
Metrics Collection: $([ $METRICS_RESULT -eq 0 ] && echo "PASSED" || echo "FAILED")

Report Files:
-------------
- Unit Test Log: unit_tests.log
- Adversarial Report: adversarial_report.json
- Metrics Report: metrics_report.json

All reports saved to: ${REPORT_DIR}
EOL

echo ""
echo "========================================="
echo "Evaluation Complete"
echo "========================================="
echo ""
cat "${REPORT_DIR}/summary.txt"
echo ""

# Exit with error if any tests failed
if [ $UNIT_TEST_RESULT -ne 0 ] || [ $ADV_TEST_RESULT -ne 0 ] || [ $METRICS_RESULT -ne 0 ]; then
    exit 1
fi

exit 0
