"""Post-training 4-bit quantization check for the deploy models.

The last sim-side unknown: everything trained so far is fp32, but the chip stores
weights in 4-bit. Does PLE's +9.3% edge survive naive group-wise PTQ, or does it
need QAT? We quantize -> dequantize every large weight (group-wise symmetric int4,
the GGUF-Q4 style you'd actually flash) and re-measure val loss.

Crucially this quantizes the 25M-param TABLE too -- the thing we're betting the
flash budget on. If the table is the fragile part under 4-bit, we need to know.

Norms stay fp32 (tiny, and quantizing them is never worth it).
"""

import argparse
import glob
import math
import os

import numpy as np
import torch
import torch.nn as nn

from model import Config, TinyLM

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
RUNS = os.path.join(HERE, "..", "runs")


def quantize_groupwise(w, bits=4, group=64, fp16_scales=False):
    """Symmetric group-wise quant along the last dim; returns dequantized fp32.

    Simulates storage in `bits` with one fp scale per `group` consecutive weights
    of each row -- the standard edge format. qmax = 2^(bits-1)-1.
    """
    orig_shape = w.shape
    x = w.reshape(-1, orig_shape[-1]).float()
    out, cols = x.shape
    pad = (group - cols % group) % group
    if pad:
        x = torch.cat([x, torch.zeros(out, pad, device=x.device)], dim=1)
    x = x.reshape(out, -1, group)  # (out, n_groups, group)
    qmax = 2 ** (bits - 1) - 1  # 7 for 4-bit
    scale = x.abs().amax(dim=2, keepdim=True) / qmax
    scale = scale.clamp_min(1e-8)
    if fp16_scales:
        # Match the flash artifact: round stored group scales to IEEE fp16
        # before quantizing and dequantizing the weights.
        scale = scale.half().float()
    q = torch.clamp(torch.round(x / scale), -qmax, qmax)
    dq = (q * scale).reshape(out, -1)[:, :cols]
    return dq.reshape(orig_shape).to(w.dtype)


def quantize_model(model, bits=4, group=64, quant_table=True, fp16_scales=False):
    n_q = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim < 2:
                continue  # norms, biases -> leave fp32
            if "table" in name and not quant_table:
                continue
            p.copy_(quantize_groupwise(p.data, bits, group, fp16_scales))
            n_q += p.numel()
    return n_q


def load(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = TinyLM(Config(**ck["cfg"])).to(device)
    m.load_state_dict(ck["state"])
    m.eval()
    return m


@torch.no_grad()
def val_loss(model, vocab, device, iters=60, bs=16, sl=256, seed=1234):
    suffix = "" if vocab == 4096 else f"_v{vocab}"
    data = np.memmap(os.path.join(DATA, f"val{suffix}.bin"), dtype=np.uint16, mode="r")
    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(iters):
        ix = rng.integers(0, len(data) - sl - 1, bs)
        x = torch.from_numpy(np.stack([data[i:i+sl] for i in ix]).astype(np.int64)).to(device)
        y = torch.from_numpy(np.stack([data[i+1:i+1+sl] for i in ix]).astype(np.int64)).to(device)
        losses.append(model(x, y)[1].item())
    return sum(losses) / len(losses)


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--tag", default="cleandeploy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fp16-scales", action="store_true",
                    help="round quantization scales to fp16 like the flash format")
    args = ap.parse_args()
    device = get_device()

    rows = {}
    for arm in ["baseline", "ple", "fatembed"]:
        path = os.path.join(RUNS, f"{arm}-{args.tag}-s{args.seed}.pt")
        if not os.path.exists(path):
            continue
        cfg_vocab = torch.load(path, map_location="cpu", weights_only=False)["cfg"]["vocab_size"]

        m = load(path, device)
        fp = val_loss(m, cfg_vocab, device)

        m = load(path, device)  # fresh copy
        nq = quantize_model(m, args.bits, args.group, quant_table=True,
                            fp16_scales=args.fp16_scales)
        q = val_loss(m, cfg_vocab, device)

        rows[arm] = (fp, q, nq)
        print(f"{arm:9s} fp32 val {fp:.4f} (ppl {math.exp(fp):.2f}) | "
              f"{args.bits}-bit val {q:.4f} (ppl {math.exp(q):.2f}) | "
              f"deg {q-fp:+.4f} | quantized {nq/1e6:.1f}M params")

    if "baseline" in rows and "ple" in rows:
        print(f"\n--- does PLE's edge survive {args.bits}-bit? ---")
        fp_gap = rows["baseline"][0] - rows["ple"][0]
        q_gap = rows["baseline"][1] - rows["ple"][1]
        print(f"  PLE vs baseline, fp32   : {fp_gap:+.4f} nats")
        print(f"  PLE vs baseline, {args.bits}-bit  : {q_gap:+.4f} nats")
        print(f"  edge retained           : {100*q_gap/fp_gap:.0f}%")


if __name__ == "__main__":
    main()
