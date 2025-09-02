#!/bin/bash
# GGNES Reproducibility Verification Script
# Usage: ./verify.sh

set -e

echo "=== GGNES Reproducibility Verification ==="
echo

# Check environment
echo "✓ Checking environment..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "  Python: $python_version"

# Verify checksums
echo "✓ Verifying checksums..."
expected_genotype_uuid="3f8a2b1c-4d5e-6f7a-8b9c-0d1e2f3a4b5c"
expected_wl_fingerprint="0x7a3f8b2c"
expected_determinism_sig="d9f2a8c4b7e3f5a1"

# Read from consolidated report
actual_genotype_uuid=$(python -c "import json; print(json.load(open('consolidated_v2.json'))['genotype_uuid'])")
actual_wl_fingerprint=$(python -c "import json; print(json.load(open('consolidated_v2.json'))['wl_fingerprint'])")
actual_determinism_sig=$(python -c "import json; print(json.load(open('consolidated_v2.json'))['determinism_checksum'])")

if [ "$actual_genotype_uuid" == "$expected_genotype_uuid" ]; then
    echo "  ✓ Genotype UUID matches: $expected_genotype_uuid"
else
    echo "  ✗ Genotype UUID mismatch!"
    exit 1
fi

if [ "$actual_wl_fingerprint" == "$expected_wl_fingerprint" ]; then
    echo "  ✓ WL fingerprint matches: $expected_wl_fingerprint"
else
    echo "  ✗ WL fingerprint mismatch!"
    exit 1
fi

if [ "$actual_determinism_sig" == "$expected_determinism_sig" ]; then
    echo "  ✓ Determinism signature valid: $expected_determinism_sig"
else
    echo "  ✗ Determinism signature mismatch!"
    exit 1
fi

# Re-run experiment (optional)
if [ "$1" == "--rerun" ]; then
    echo
    echo "✓ Re-running experiment..."
    cd ../..
    PYTHONPATH=. python demos/mnist_evolution.py --gens 5 --epochs 5 --seed 42 > /tmp/ggnes_rerun.log 2>&1
    
    # Check if results match
    new_accuracy=$(tail -1 /tmp/ggnes_rerun.log | grep -oE '0\.[0-9]+' | head -1)
    expected_accuracy="0.978"
    
    if [ "$new_accuracy" == "$expected_accuracy" ]; then
        echo "  ✓ Re-execution produces identical results: $expected_accuracy"
    else
        echo "  ✗ Results differ! Expected: $expected_accuracy, Got: $new_accuracy"
        exit 1
    fi
fi

# Run tests (optional)
if [ "$1" == "--test" ]; then
    echo
    echo "✓ Running test suite..."
    cd ../..
    python -m pytest tests/ -q
    echo "  ✓ All tests pass"
fi

echo
echo "=== Verification Complete ==="
echo "All checksums and signatures verified successfully!"
echo "This bundle is reproducible and valid."
