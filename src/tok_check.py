"""Prove the device BPE encoder matches the training tokenizer.

The device now tokenizes typed text itself (firmware/common/tokenizer.h). If its
output diverges from `tokenizers`, prompts land out of distribution and the only
symptom is worse generations -- no error, nothing in the logs. So this diffs the
two over real corpus lines, the same way verify.c diffs the C runtime against the
PyTorch golden.

  cc -O3 -o /tmp/tok_test firmware/host_verify/tok_test.c
  uv run python src/tok_check.py
"""

import os
import subprocess
import sys

from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
TOK = os.path.join(ROOT, "data", "bpe32768.json")
CORPUS = os.path.join(ROOT, "data", "tinystories_slice.txt")
BIN = "/tmp/tok_test"
N_LINES = 2000

# Hand-picked cases first: contractions, digits, punctuation runs, repeated and
# leading whitespace, and the empty line -- the places the GPT-2 split pattern is
# easy to get subtly wrong.
EDGE_CASES = [
    "Once upon a time",
    "Once upon a time, there was a little girl named Lily.",
    "She didn't know what to do; she'd never seen it!",
    "I'm sure they're happy... aren't we all?",
    "He said \"hello\" and left.",
    "There were 3 cats and 42 dogs in 2026.",
    "double  space and   triple",
    " leading space",
    "trailing space ",
    "tabs\tand\tmore",
    "UPPER lower MiXeD",
    "a",
    " ",
    "",
    "!!!???...---",
    "cat's dog's bird's",
    "we'll we've we're I'd I'm it's don't",
]


def main():
    if not os.path.exists(BIN):
        sys.exit(f"missing {BIN} -- build it first:\n"
                 f"  cc -O3 -o {BIN} firmware/host_verify/tok_test.c")

    tok = Tokenizer.from_file(TOK)

    lines = list(EDGE_CASES)
    if os.path.exists(CORPUS):
        with open(CORPUS, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.rstrip("\n\r")
                # The harness is line-based, so skip anything with a tab. Also
                # skip the document separator: Python maps <|endoftext|> to the
                # special id 0, while the device encoder only ever sees text a
                # user typed and treats it as literal characters.
                if line and "\t" not in line and "<|endoftext|>" not in line:
                    lines.append(line)
                if len(lines) >= N_LINES:
                    break
    else:
        print(f"note: {CORPUS} absent, running edge cases only")

    proc = subprocess.run(BIN, input="\n".join(lines) + "\n",
                          capture_output=True, text=True)
    got = proc.stdout.split("\n")

    bad = 0
    for i, line in enumerate(lines):
        want = tok.encode(line).ids
        mine = [int(x) for x in got[i].split()] if i < len(got) and got[i].strip() else []
        if want != mine:
            bad += 1
            if bad <= 5:
                print(f"MISMATCH line {i}: {line!r}")
                print(f"  python: {want}")
                print(f"  c     : {mine}")
                print(f"  python decoded: {tok.decode(want)!r}")
                print(f"  c decoded     : {tok.decode(mine)!r}")

    total = len(lines)
    print(f"\n{total - bad}/{total} lines match "
          f"({100.0 * (total - bad) / total:.2f}%)")
    if bad:
        print(f"FAIL: {bad} mismatched")
        return 1
    print("PASS: device encoder matches the training tokenizer")
    return 0


if __name__ == "__main__":
    sys.exit(main())
