#!/usr/bin/env bash
# Does the gain scale with table size? ple_dim=64 already has 2 seeds from the
# main ablation, so only the other points are run here.
#
# table params = vocab(4096) * n_layers(6) * ple_dim
# core overhead = 2 * d_model(128) * ple_dim * n_layers  -- grows with ple_dim,
# so the auto-solver shrinks the FFN to compensate. The sweep therefore measures
# a real tradeoff: more table capacity bought with less compute capacity.
set -euo pipefail
cd "$(dirname "$0")/.."
for pd in 16 32 128 256; do
  echo "=== ple_dim=$pd ==="
  uv run python src/train.py --arm ple --steps 3000 --target-core 1500000 \
      --ple-dim "$pd" --seed 0 --tag "d$pd"
done
