# MNIST Evolution Reproducibility Bundle

This bundle contains all artifacts needed to reproduce the MNIST evolution experiment.

## Quick Verification

```bash
./verify.sh
```

## Full Re-run

```bash
./verify.sh --rerun
```

## Bundle Contents

- `manifest.json`: Environment and configuration snapshot
- `genotype.json`: Best evolved architecture (schema v1)
- `consolidated_v2.json`: Complete metrics with determinism signatures
- `fingerprints.txt`: WL hashes for all evolved graphs
- `verify.sh`: One-command verification script

## Expected Results

- Best accuracy: 97.8%
- Generation: 4
- Architecture: 256-neuron hidden layer with skip connection
- Determinism signature: `d9f2a8c4b7e3f5a1`

## Reproduction Steps

1. Install GGNES and dependencies
2. Run `./verify.sh` to validate checksums
3. Run `./verify.sh --rerun` to reproduce results
4. Compare new results with this bundle

## Citation

If you use this bundle in your research:

```bibtex
@software{ggnes2024,
  title={GGNES: Graph Grammar Neuroevolution System},
  author={Sedaghat, Amin},
  year={2024},
  url={https://github.com/Aminsed/GGNES}
}
```
