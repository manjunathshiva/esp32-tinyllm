# Results

**How to read this document.** It mixes two kinds of number and keeps them
separate on purpose:

- **Measured here** — reproduced on this hardware, from this repo's checkpoint.
  Sections 1 through 4.
- **Inherited from upstream** — results from
  [slvDev/esp32-ai](https://github.com/slvDev/esp32-ai) that justify the
  architecture but were **not re-run in this project**. Section 5. They are
  reported because the design rests on them, not because they were verified
  here. Treat them as cited prior work.

Only one training arm was run for this project: the deployable `ple` config at
seed 0. The comparative ablations (`baseline`, `fatembed`, `ple_notable`,
`bigcore`), the vocabulary and `ple_dim` sweeps, the 4-bit quantization
comparison, and the on-chip bandwidth benchmark were **not** repeated.

---

## 1. The model in this repo

Trained with `src/train.py` on a 74.9M-token TinyStories slice, vocabulary
32,768, on Apple Silicon (MPS).

```
uv run python src/train.py --arm ple --vocab 32768 --d-model 96 --n-layers 6 \
  --ple-dim 128 --target-core 560000 --batch-size 16 --seq-len 256 \
  --steps 5000 --seed 0 --tag cleandeploy
```

| | |
|---|---|
| architecture | d_model 96, 6 layers, 4 heads, ffn_hidden 66, ple_dim 128, seq_len 256 |
| core (SRAM tier) | 558,368 |
| stream (output head) | 3,145,728 |
| table (flash tier) | 25,165,824 |
| **total stored** | **28,869,920** |
| val loss / perplexity | **2.4299 / 11.36** |
| training | 5,000 steps, 20,480,000 tokens, **456 s** |

The three tiers are split by *access pattern*, not size: `core` is dense and read
every token so it must be SRAM-resident; `stream` is the output head, read as one
sequential scan per token, so it costs bandwidth rather than capacity; `table` is
sparse, one row per token, and lives memory-mapped in flash. See `src/budget.py`.

**What "28.9M parameters" means.** Parameters *resident via a memory-hierarchy
split*. It is not a capability claim and must not be quoted as one. This remains
a TinyStories-domain model: it continues text and cannot answer questions, follow
instructions, or recall facts. That ceiling is set by the 559K-parameter dense
core; the flash table buys coherence, not capability.

## 2. Host verification gates

Both run before anything is flashed, and both must pass.

**Numerics.** The portable C runtime (`firmware/common/llm.h`) is compiled on the
host and compared against a PyTorch golden generated from the *dequantized*
weights, which isolates port correctness from quantization error.

```
max abs diff = 0.00002    rms diff = 0.000008
C top token == PyTorch top token
PASS
```

**Tokenizer.** The device tokenizes typed input itself, so its BPE must match the
training tokenizer exactly — a prompt tokenized differently is out of
distribution with nothing in the logs to show for it. `src/tok_check.py` diffs
the C encoder against `tokenizers` over 2,000 corpus lines plus contraction,
digit, punctuation and whitespace edge cases.

```
2000/2000 lines match (100.00%)
PASS
```

## 3. The flash artifact

| | |
|---|---|
| size | 14,912,332 bytes |
| SHA-256 | `dd22df35df128fab39b9e1738af8377bb493a778942e28f1d4a621a40c542a28` |
| format | group-128 ragged int4, fp16 scales |
| partition | `model` at `0x110000`, 15,597,568 bytes (685,236 spare) |

Retraining produces a different checksum; verify against your own export.

## 4. On-chip measurements (ESP32-S3 N16R8)

Boot diagnostics for the artifact above:

```
model: V=32768 D=96 L=6 H=4 F=66 P=128  (mapped 15.6 MB)
head staged int8: 2.54 MB
PSRAM free after alloc: 4328 KB
```

The 1.64MB figure for the head that appears in older notes predates int8
staging; the staged head is 2.54MB, which is where roughly 0.9MB of the
previously-reported free PSRAM went.

### Throughput

Measured over four generation runs in one session:

| run | tokens | tok/s | ms/token | attention |
|---|---:|---:|---:|---:|
| short (context nearly empty) | 42 | 10.40 | 93.1 | 15.7 |
| standard | 199 | 9.49 | 103.2 | 25.9 |
| standard | 173 | 9.52 | 102.0 | 24.6 |
| long, sliding window | 600 | 8.74 | 111.2 | 33.8 |

Per-stage profile, ms/token:

| stage | ms | note |
|---|---:|---|
| output head | **57.6** | identical in every run |
| attention | 15.7 – 34.1 | scales with context position |
| PLE path | 8.5 | constant |
| FFN | 6.9 | constant |
| input | 4.4 | constant |

**The head is constant at 57.6ms across every run while attention tracks context
length.** That is the clearest evidence the measurement is real: the head scans a
fixed 2.43MB of int8 weights per token regardless of position, while attention is
O(pos). At 60.7 MB/s of PSRAM bandwidth the head has a ~40ms read floor inside
its 57.6ms, so it is bandwidth-bound rather than compute-bound. Vectorising
harder buys a bounded ~15%; reducing bytes read (an int4 head with SIMD unpack)
or a smaller/factorised head are the real levers.

### Interactive additions

Three things were added on top of the inherited runtime, all measured above.

**On-device BPE encoding.** The firmware previously decoded only, so prompts had
to be compiled in as literal token ids. `firmware/common/tokenizer.h` plus a
generated 25,096-entry merge table (~197KB) reproduces ByteLevel + BPE including
GPT-2's pre-tokenization split. Merges never cross a chunk boundary, so the split
must match or output diverges even with a correct merge table.

**Sampling.** `llm_sample` implements temperature/top-k, matching `model.py`'s
`generate()` (0.8 / 40). Greedy argmax is available via `/temp 0` and is
deterministic — the same prompt reproduces the same output byte for byte. It
also demonstrates the failure mode it was replaced for: given
`The dragon opened its eyes and`, greedy emits "I will help you" five times in
one block and "You are very kind and kind" twice.

Sampling reduces repetition; it does not remove it. A 559K-parameter core repeats
regardless of decode rule, and sampled runs still show degenerate stretches. The
honest claim is a difference of degree.

**Sliding context.** Generation used to stop at 256 positions. `llm_kv_slide`
drops the oldest half of the KV cache and shifts the rest down. Cached keys carry
RoPE at their original absolute position, so after a shift they would claim
positions they no longer hold; re-priming would cost one full forward pass per
carried token. Because RoPE is a rotation by `pos * freq` and rotations compose,
re-basing a key from `p` to `p-drop` is a single rotation by `-(drop * freq)`.
Positions stay inside `[0, 256)`, so nothing extrapolates past the trained range.

A 600-token run slid three times and stayed coherent across every boundary. The
throughput cost is attention growing with position, not the slide itself. A
rolling window is not a longer memory — the model does forget the opening.

### Application size

829,682 bytes of the 1MB `factory` partition (~79%), the merge tables accounting
for ~197KB. Note that `arduino-cli` reports application size against total flash
rather than the custom partition, so its percentage is misleading.

---

## 5. Inherited results — not reproduced in this project

Everything below is from upstream and justifies why the architecture is shaped
this way. **None of it was re-run here.** Cite it as prior work.

### Core-matched ablation, vocabulary 32,768, two seeds

| arm | core | total | ppl | vs baseline |
|---|---:|---:|---:|---:|
| `baseline` | 559K | 3.7M | 12.58 | — |
| **`ple`** | 558K | **28.9M** | **11.41** | **+0.098 nats / 9.3% ppl** |
| `fatembed` | 559K | 28.9M | 11.94 | +0.052 nats |

PLE beat a same-core, SRAM-fitting baseline by 0.098 nats, roughly 16x the seed
noise, and per-layer injection beat bottom injection by 0.046 nats. This repo's
own `ple` run reaches ppl 11.36, consistent with the 11.41 reported upstream, but
**the comparison arms were not trained here**, so the margin itself is not
independently confirmed by this project.

### What the controls established

- **The table does the work, not the plumbing.** `ple_notable` — all of PLE's
  per-layer adapters and projection with no lookup table — was *worse* than
  baseline at vocabulary 4,096. The flash-resident table is the entire source of
  the gain.
- **Vocabulary size matters.** At vocabulary 4,096 PLE's edge was only +0.025
  nats against +0.098 at 32,768. A large vocabulary makes the table both huge and
  cheap, which is the regime PLE was designed for.
- **PLE is not free capacity.** Doubling the dense core (`bigcore`) gained more
  than PLE did. On a desktop you would simply buy the core; on a microcontroller
  the core is fixed silicon and flash is abundant, which is the entire point.

### 4-bit quantization

Group-wise symmetric int4 PTQ degraded all arms (~ppl +1), but PLE degraded
*less*, retaining 124–128% of its edge — a large redundant lookup table with
per-group scales is inherently more quantization-robust than a small dense model
where every weight is critical. No QAT was needed.

### Bandwidth, measured on an N16R8

| measurement | value |
|---|---|
| PSRAM sequential read | 60.7 MB/s |
| internal SRAM sequential read | 240 MB/s |
| flash random-read, 512B row | 20.3 us |
| per-token table cost (6 random rows) | ~0.12 ms |
| per-token head cost (1.5MB PSRAM scan) | ~17.3 ms |

The 25M-parameter flash table costs well under 1% of per-token memory time —
nearly free, as designed. The output head dominates. This is a bandwidth-only
ceiling, not observed throughput.

---

## 6. Limitations

- **Domain is TinyStories.** No world knowledge, arithmetic, or multi-step
  reasoning. Set by the dense core, unmoved by the table.
- **Context is 256 positions.** The sliding window allows unbounded generation
  but not longer memory. Each prompt starts a fresh context; earlier turns are
  not carried.
- **Repetition persists.** Sampling manages it; the model size causes it.
- **The ESP32-S3's SIMD instructions remain unused.** The head is
  PSRAM-bandwidth-bound, so SIMD alone buys a bounded ~15%.
- **Comparative claims are inherited.** Section 5 was not reproduced here.
- **ASCII assumption in the tokenizer.** GPT-2's split pattern classifies with
  Unicode properties; the C port classifies with ASCII. TinyStories is ASCII, so
  training is unaffected, but typed non-ASCII input may tokenize differently.

## 7. Next

- Record and publish comparative arms locally, so section 5 can move into
  section 1 as reproduced rather than cited.
- Dialogue fine-tuning for a simple-conversation milestone, reusing the same
  32,768 tokenizer so `vocab.h`, the merge tables and every perplexity
  comparison stay valid.
- An int4-in-PSRAM head with SIMD unpack, which attacks the bytes-read floor
  rather than the compute above it.

## Provenance

This project builds on [slvDev/esp32-ai](https://github.com/slvDev/esp32-ai)
(MIT). Its dependencies are the TinyStories dataset (Eldan & Li, Microsoft
Research, [arXiv:2305.07759](https://arxiv.org/abs/2305.07759)) and Google's
published Gemma Per-Layer Embeddings design. llama2.c and DaveBben's esp32-llm
are prior independent work in the same space and appear only as comparison
context; no code or method derives from either.
