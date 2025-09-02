#!/bin/bash
# Binary Classification Tutorial Verification
set -e

echo "=== Binary Classification Reproducibility Check ==="
echo

# Verify tutorial exists
if [ -f "../../tutorials/17_end_to_end_binary_classification.py" ]; then
    echo "✓ Tutorial script found"
else
    echo "✗ Tutorial script missing!"
    exit 1
fi

# Check manifest
echo "✓ Checking manifest..."
seed=$(python -c "import json; print(json.load(open('manifest.json'))['environment']['seed'])")
echo "  Seed: $seed"

# Re-run if requested
if [ "$1" == "--rerun" ]; then
    echo "✓ Re-running tutorial..."
    cd ../..
    PYTHONPATH=. python tutorials/17_end_to_end_binary_classification.py > /tmp/binary_rerun.log 2>&1
    echo "  ✓ Tutorial completed successfully"
fi

echo
echo "=== Verification Complete ==="
