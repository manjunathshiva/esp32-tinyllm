"""Generate text from a trained run, so the loss numbers have something behind them."""

import argparse
import os

import torch
from tokenizers import Tokenizer

from model import Config, TinyLM
from train import get_device

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="path to a runs/*.pt checkpoint")
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--samples", type=int, default=3)
    args = ap.parse_args()

    device = get_device()
    ck = torch.load(args.run, map_location=device, weights_only=False)
    model = TinyLM(Config(**ck["cfg"])).to(device)
    model.load_state_dict(ck["state"])
    model.eval()

    tok = Tokenizer.from_file(os.path.join(HERE, "..", "data", "bpe4096.json"))
    ids = torch.tensor([tok.encode(args.prompt).ids], device=device)

    b = model.param_budget()
    print(f"{os.path.basename(args.run)}  core={b['core']:,} table={b['table']:,}\n")
    for i in range(args.samples):
        out = model.generate(ids, args.tokens, temperature=args.temperature)
        print(f"--- sample {i + 1} ---")
        print(tok.decode(out[0].tolist()).strip(), "\n")


if __name__ == "__main__":
    main()
