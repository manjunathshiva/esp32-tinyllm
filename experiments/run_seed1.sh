#!/usr/bin/env bash
# Second seed on the three arms that carry the claims, to size run-to-run noise.
set -euo pipefail
cd "$(dirname "$0")/.."
for arm in baseline ple fatembed; do
  echo "=== $arm (seed 1) ==="
  uv run python src/train.py --arm "$arm" --steps 3000 --target-core 1500000 --seed 1
done
