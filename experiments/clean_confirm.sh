#!/usr/bin/env bash
# Clean re-run of everything under ONE consistent accounting (the current
# three-tier param_budget, head excluded from core). The overnight run mixed two
# accountings after param_budget changed mid-flight; this regenerates the main
# ablation and a properly matched deploy config so the whole dataset is coherent.
#
# Kept as-is (already current accounting): the Job 1 corrected sweep (ple*fix-d*).
set -uo pipefail
cd "$(dirname "$0")/.."
log() { echo "[$(date '+%m-%d %H:%M')] $*"; }
run() { log "RUN $*"; uv run python src/train.py "$@" 2>&1 || log "!! FAILED: $*"; }

# ---- Main ablation, vocab 4096, both seeds, current accounting ------------------
log "=== main ablation (vocab 4096, current accounting) ==="
for seed in 0 1; do
  for arm in baseline ple ple_notable fatembed bigcore; do
    run --arm $arm --target-core 1500000 --steps 3000 --seed $seed --tag clean
  done
done

# ---- Deploy validation, vocab 32768, core-matched AND SRAM-fitting --------------
# d=96 L=6 ple_dim=128 -> all arms land ~559k core (273KB@4bit, under 320KB SRAM),
# table 25M. bs16/sl256 keeps 32k cross-entropy off the MPS memory cliff.
log "=== deploy validation (vocab 32768, core-matched ~559k, fits SRAM) ==="
for arm in baseline ple fatembed; do
  run --arm $arm --vocab 32768 --d-model 96 --n-layers 6 --ple-dim 128 \
      --target-core 560000 --batch-size 16 --seq-len 256 --steps 5000 \
      --seed 0 --tag cleandeploy
done

log "=== CLEAN CONFIRM COMPLETE ==="
