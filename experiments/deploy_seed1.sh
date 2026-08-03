#!/usr/bin/env bash
# Second seed on the deploy headline (baseline + ple + fatembed @ vocab 32768,
# core-matched ~559k). Locks the +0.101 nat result against seed noise.
set -uo pipefail
cd "$(dirname "$0")/.."
log() { echo "[$(date '+%m-%d %H:%M')] $*"; }
for arm in baseline ple fatembed; do
  log "RUN $arm seed1"
  uv run python src/train.py --arm $arm --vocab 32768 --d-model 96 --n-layers 6 \
      --ple-dim 128 --target-core 560000 --batch-size 16 --seq-len 256 \
      --steps 5000 --seed 1 --tag cleandeploy 2>&1 || log "!! FAILED $arm"
done
log "=== DEPLOY SEED1 COMPLETE ==="
