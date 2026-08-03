"""Three-tier memory accounting for an ESP32 target.

Deliberately separate from `model.py`. That module's `param_budget()` is what
`make_model()` binary-searches against to match core budgets across arms, so
changing its semantics would silently alter every model built afterwards and
break comparability with runs already completed. This file only *reports*.

The three tiers differ by access pattern, not just speed:

  core    dense, random access, every token.
          -> must be SRAM-resident. The genuinely scarce budget.
  stream  dense, but read as one sequential scan per token. The output head is
          the whole story: you need every logit, so you touch the entire matrix,
          but strictly start to finish.
          -> costs BANDWIDTH, not SRAM capacity. Flash and PSRAM are both decent
             at sequential reads, so this can live off-chip.
  table   sparse: one row per token.
          -> ideal for memory-mapped flash.

Treating `stream` as if it were `core` is what made large vocabularies look
unaffordable, when in fact they are merely slow.
"""

import argparse

# ESP32-S3-DevKitC-1 N16R8, the ordered board.
SRAM_BYTES = 320 * 1024  # 512KB internal, minus IDF/stack/buffers. Conservative.
PSRAM_BYTES = 8 * 1024 * 1024
FLASH_BYTES = 15 * 1024 * 1024  # 16MB minus ~1MB firmware.
# HISTORICAL PLANNING NUMBERS, kept so early sizing decisions stay reproducible.
# Real measurements exist now (RESULTS.md): PSRAM 60.7 MB/s, flash 20.3us/512B
# random read, ~9.5 tok/s end to end. This file is not deployment authority.
PSRAM_BW = 60e6  # bytes/sec
FLASH_BW = 60e6  # bytes/sec, sequential


def tiers(vocab, d_model, n_layers, ple_dim, ffn_hidden, n_heads=4):
    """Parameter counts per tier for the `ple` architecture."""
    per_layer = (
        4 * d_model * d_model  # attn qkv + out proj
        + 3 * d_model * ffn_hidden  # SwiGLU
        + 2 * d_model * ple_dim  # per-layer gate + projection
    )
    core = n_layers * per_layer + d_model * n_layers * ple_dim  # + ple_model_proj
    stream = vocab * d_model  # output head
    table = vocab * n_layers * ple_dim
    return {"core": core, "stream": stream, "table": table}


def report(bits=4, **kw):
    t = tiers(**kw)
    b = bits / 8
    core_b, stream_b, table_b = (t[k] * b for k in ("core", "stream", "table"))

    # Per token: the whole head is scanned; only n_layers rows of the table are read.
    table_row_b = kw["n_layers"] * kw["ple_dim"] * b
    t_head = stream_b / FLASH_BW
    t_table = table_row_b / FLASH_BW

    print(f"  vocab={kw['vocab']} d={kw['d_model']} L={kw['n_layers']} "
          f"ple_dim={kw['ple_dim']} ffn={kw['ffn_hidden']}  @{bits}-bit")
    print(f"    core   {t['core']:>12,} params  {core_b / 1024:>8.0f} KB   "
          f"{'OK' if core_b <= SRAM_BYTES else 'DOES NOT FIT SRAM'}")
    print(f"    stream {t['stream']:>12,} params  {stream_b / 1024:>8.0f} KB   "
          f"{t_head * 1000:.1f} ms/token")
    print(f"    table  {t['table']:>12,} params  {table_b / 1024 / 1024:>8.2f} MB  "
          f"{'OK' if table_b <= FLASH_BYTES else 'EXCEEDS FLASH'}  "
          f"({table_row_b:.0f} B/token, {t_table * 1000:.3f} ms)")
    total = sum(t.values())
    ceiling = 1.0 / (t_head + t_table)
    print(f"    TOTAL  {total:>12,} params   ratio {total * b / SRAM_BYTES:.0f}x SRAM"
          f"   bandwidth ceiling ~{ceiling:.0f} tok/s (est, compute excluded)\n")
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=4)
    args = ap.parse_args()

    print(f"\nESP32-S3 N16R8: {SRAM_BYTES // 1024}KB usable SRAM, "
          f"{FLASH_BYTES // 1024 // 1024}MB usable flash, "
          f"{PSRAM_BYTES // 1024 // 1024}MB PSRAM\n")

    print("[what this repo actually validated]")
    report(bits=args.bits, vocab=4096, d_model=128, n_layers=6, ple_dim=64, ffn_hidden=187)

    print("[candidate deploy configs -- UNVALIDATED, sweep pending]")
    for vocab, d, L, pd, ffn in [
        (4096, 128, 6, 256, 128),
        (8192, 192, 8, 256, 256),
        (16384, 192, 8, 384, 256),
        (32768, 256, 10, 512, 384),
    ]:
        report(bits=args.bits, vocab=vocab, d_model=d, n_layers=L, ple_dim=pd, ffn_hidden=ffn)


if __name__ == "__main__":
    main()
