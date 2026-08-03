#!/usr/bin/env bash
# Core-matched ablation. Every arm gets the same dense/SRAM parameter budget;
# they differ only in how (and whether) extra sparse/flash params are spent.
set -euo pipefail
cd "$(dirname "$0")/.."

STEPS=${STEPS:-3000}
CORE=${CORE:-1500000}
SEED=${SEED:-0}

for arm in baseline ple ple_notable fatembed bigcore; do
  echo "=== $arm (seed $SEED) ==="
  uv run python src/train.py --arm "$arm" --steps "$STEPS" --target-core "$CORE" --seed "$SEED"
done
