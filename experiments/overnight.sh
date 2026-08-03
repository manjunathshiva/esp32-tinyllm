#!/usr/bin/env bash
# Unattended overnight queue. Ordered most-decision-relevant first, so a stall
# late in the night still leaves the important results done. Each run writes its
# own runs/<name>.json, so a single failure loses only that run.
#
# set +e on purpose: a failed run must NOT abort the queue.
set -uo pipefail
cd "$(dirname "$0")/.."

log() { echo "[$(date '+%m-%d %H:%M')] $*"; }
run() { log "RUN $*"; uv run python src/train.py "$@" 2>&1 || log "!! FAILED: $*"; }

# ---- Job 1: corrected table-scaling sweep (THE key result) ---------------------
# Fixed FFN, core floats with ple_dim, so growing the table never cannibalises
# compute (the flaw in the first sweep, where ffn collapsed to 1 at ple_dim=256).
# The table's isolated effect = ple - ple_notable at each ple_dim: identical
# core/ffn/plumbing, differ ONLY by the presence of the lookup table.
# vocab 4096, standard bs32/sl512 (~0.5-0.9 s/step as core grows). Single seed.
log "=== JOB 1: corrected table-scaling sweep (fixed ffn=256, vocab 4096) ==="
for pd in 64 128 256 512; do
  run --arm ple         --fixed-ffn 256 --ple-dim $pd --steps 3000 --seed 0 --tag "fix-d$pd"
  run --arm ple_notable --fixed-ffn 256 --ple-dim $pd --steps 3000 --seed 0 --tag "fix-d$pd"
done

# ---- Job 2: deployable-size validation @ vocab 32768 ---------------------------
# The real headline config: big vocab makes the table cheap, small core fits SRAM.
# bs16/sl256 keeps the 32k-class cross-entropy off the MPS memory cliff (profiled:
# 223 ms/step vs a stall at bs32/sl512). 5000 steps ~= 20M tokens.
# Internally comparable (all vocab 32k); NOT comparable to the vocab-4096 runs.
log "=== JOB 2: deploy validation @ vocab 32768 (bs16 sl256) ==="
for arm in baseline ple fatembed; do
  run --arm $arm --vocab 32768 --ple-dim 152 --target-core 590000 \
      --batch-size 16 --seq-len 256 --steps 5000 --seed 0 --tag "deploy590k"
done
# Reference point at the larger (SRAM-overflowing) core, if the night has room.
for arm in baseline ple fatembed; do
  run --arm $arm --vocab 32768 --ple-dim 152 --target-core 971000 \
      --batch-size 16 --seq-len 256 --steps 5000 --seed 0 --tag "deploy971k"
done

# ---- Job 3: firm up single-seed arms from the main ablation ---------------------
log "=== JOB 3: extra seeds on ple_notable / bigcore (vocab 4096) ==="
run --arm ple_notable --target-core 1500000 --steps 3000 --seed 1
run --arm bigcore     --target-core 1500000 --steps 3000 --seed 1

log "=== OVERNIGHT QUEUE COMPLETE ==="
