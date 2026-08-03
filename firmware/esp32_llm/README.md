# ESP32-S3 on-chip inference

This sketch runs the 28.9M-parameter PLE TinyLM on an ESP32-S3 N16R8. The model
lives in the custom `model` flash partition at `0x110000`; the tied
embedding/output head is staged in PSRAM at boot.

## Build and verify

Export the group-128 ragged-int4 model and verify the portable C runtime first:

```bash
cd src
uv run python export.py
uv run python gen_assets.py
cd ..
cc -O3 -o /tmp/esp32-llm-verify firmware/host_verify/verify.c -lm
/tmp/esp32-llm-verify firmware/model/model.bin firmware/model/golden.txt
```

`gen_assets.py` is required, not optional: it writes `vocab.h` (token id -> bytes)
and `bpe.h` (the encode tables), and the sketch will not compile without them.

The device tokenizes typed input itself, so its BPE must agree with the training
tokenizer exactly -- a prompt tokenized differently is out of distribution with
no error to notice. That has its own host gate:

```bash
cc -O3 -o /tmp/tok_test firmware/host_verify/tok_test.c
uv run python src/tok_check.py     # expects 2000/2000 lines to match
```

Build the device firmware with Arduino ESP32 core 3.3.10:

```bash
arduino-cli compile \
  --fqbn 'esp32:esp32:esp32s3:UploadSpeed=921600,USBMode=hwcdc,CDCOnBoot=cdc,UploadMode=default,CPUFreq=240,FlashMode=qio,FlashSize=16M,PartitionScheme=custom,PSRAM=opi,DebugLevel=info' \
  --build-property compiler.optimization_flags=-O3 \
  --build-path /tmp/esp32-llm-build \
  firmware/esp32_llm
```

## Flash and run

Replace the port if the board enumerates under a different device name:

```bash
arduino-cli upload \
  -p /dev/cu.usbmodem2101 \
  --fqbn 'esp32:esp32:esp32s3:UploadSpeed=921600,USBMode=hwcdc,CDCOnBoot=cdc,UploadMode=default,CPUFreq=240,FlashMode=qio,FlashSize=16M,PartitionScheme=custom,PSRAM=opi,DebugLevel=info' \
  --input-dir /tmp/esp32-llm-build \
  firmware/esp32_llm

esptool.py --chip esp32s3 --port /dev/cu.usbmodem2101 --baud 921600 \
  write_flash 0x110000 firmware/model/model.bin

arduino-cli monitor -p /dev/cu.usbmodem2101 --config baudrate=115200
```

The model payload only needs reflashing after a new export. Firmware-only
changes can be uploaded without rewriting the model partition.

The model in this repo (`firmware/model/model.bin`, 14,912,332 bytes, val
perplexity 11.36) has SHA-256:

```text
dd22df35df128fab39b9e1738af8377bb493a778942e28f1d4a621a40c542a28
```

Expected boot diagnostics for that artifact:

```text
model: V=32768 D=96 L=6 H=4 F=66 P=128
head staged int8: 2.54 MB
PSRAM free after alloc: 4328 KB
```

Retraining produces a different checksum, so verify against your own export
rather than this line. Note that the free-PSRAM figure moved once the output
head was staged as int8: the head grew from 1.64MB to 2.54MB, so roughly 0.9MB
of the PSRAM that older notes report as free is now the staged head.

The current runtime measures 102.9ms per model step (9.72 tok/s compute-only);
attached serial runs measure ~9.5 tok/s including output. On-device profile:
57.6ms output head, 25.6ms attention, 8.5ms PLE path, 6.9ms FFN, 4.4ms input.
The head is staged as int8 with int8 activations (host-validated, val perplexity
delta ~0) and is now PSRAM-bandwidth-bound. The fp32 host golden still matches
PyTorch to 1e-5.

## Interactive use

Recommended terminal, because it does not take over any key the REPL needs and
leaves local echo off (the device echoes typed characters itself, so local echo
would double every keystroke):

```bash
uv run --no-project --with pyserial python -m serial.tools.miniterm \
    /dev/cu.usbmodem1101 115200          # exit: Ctrl-]
```

`screen` also works, with two caveats: its command prefix is Ctrl-A, and if the
enclosing terminal emulator has mouse reporting enabled, mouse movement injects
escape sequences into the serial stream whose printable bytes land in the input
line. Turn mouse reporting off, or use miniterm.

All device output is CRLF-terminated, so raw-mode terminals render it correctly.

Type an opening line; the device tokenizes it on-chip and continues the story. It continues text -- it does not answer
questions or follow instructions, which is a property of the TinyStories
training domain and not of the decode path.

```text
> Once upon a time there was a small robot who loved to paint
[13 prompt tokens]

Once upon a time there was a small robot who loved to paint. Every day, the
robot would build towers and houses and birds...

--- 131 tokens in 13.43 s, end of story ---
throughput: 9.76 tok/s   (99.6 ms/token)
```

Commands: `/temp <f>` (0 = greedy), `/topk <n>`, `/tokens <n>`, `/eot <0|1>`,
`/help`.

Sampling defaults to temperature 0.8 / top-k 40, matching `src/model.py`'s
`generate()`. Greedy decoding (`/temp 0`) is available but tends to lock into a
repeated sentence — that is the decode rule, not the weights.

### Getting longer output

Two separate things end a run, and the one people expect is usually not the one
that fires.

**`<|endoftext|>` ends most runs**, not the context limit. TinyStories are short,
so the model reaches "The end." at roughly 130-200 tokens — well inside the
window. `/eot 0` feeds the token through instead of stopping, which rolls
straight into a fresh story and is how you get a long continuous stream.

**The context window is 256 positions**, prompt included. Past that the KV cache
slides: the oldest half is dropped and the rest shifts down, giving a rolling
window. Cached keys carry RoPE at their original absolute position, so sliding
would leave them claiming positions they no longer hold. Re-priming would fix
that at the cost of a full forward pass per carried token — seconds of stall.
Instead, because RoPE is a rotation by `pos * freq` and rotations compose,
re-basing a key from `p` to `p-drop` is one extra rotation by `-(drop * freq)`
(`llm_kv_slide` in `firmware/common/llm.h`). Positions therefore stay inside
`[0, 256)` and nothing extrapolates past the trained range.

The model still forgets the beginning — a rolling window is not a longer memory.

Measured: 600 tokens with `/eot 0` produced 3 slides and 3 story rollovers with
coherent text throughout, at 8.70 tok/s. Throughput drops from ~9.5 because
attention cost grows with position (34.2ms/token with the window consistently
full, against 25.6ms at ~200 and 16.1ms at 40). Each line still starts a fresh
context; earlier lines are not carried.

Adding `bpe.h` puts the application at ~829KB of the 1MB `factory` partition.
Note that `arduino-cli` reports application size against total flash, not the
custom partition, so its percentage is not the number to watch.
