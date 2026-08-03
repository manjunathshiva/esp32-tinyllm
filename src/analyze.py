"""Summarise ablation runs and answer the question the experiment was built to ask."""

import glob
import json
import math
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "runs")

LABEL = {
    "baseline": "baseline      (no table)",
    "ple": "ple           (proj + table)",
    "ple_notable": "ple_notable   (proj only)",
    "fatembed": "fatembed      (bottom-only table)",
    "bigcore": "bigcore       (2x core, no table)",
}
ORDER = ["baseline", "ple_notable", "fatembed", "ple", "bigcore"]


def load():
    by_arm = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(RUNS, "*.json"))):
        r = json.load(open(path))
        by_arm[r["arm"]].append(r)
    return by_arm


def mean(xs):
    return sum(xs) / len(xs)


def main():
    by_arm = load()
    if not by_arm:
        print("no runs found")
        return

    print(f"{'arm':38s} {'core':>10s} {'table':>10s} {'val loss':>10s} {'ppl':>9s} {'n':>3s}")
    print("-" * 84)
    vals = {}
    for arm in ORDER:
        if arm not in by_arm:
            continue
        rs = by_arm[arm]
        v = mean([r["final_val"] for r in rs])
        vals[arm] = v
        p = rs[0]["params"]
        spread = ""
        if len(rs) > 1:
            spread = f"  (+/-{(max(r['final_val'] for r in rs) - min(r['final_val'] for r in rs)) / 2:.4f})"
        print(
            f"{LABEL[arm]:38s} {p['core']:>10,} {p['table']:>10,} "
            f"{v:>10.4f} {math.exp(v):>9.2f} {len(rs):>3d}{spread}"
        )

    print("\n--- the comparisons that decide it " + "-" * 47)

    def delta(a, b, question):
        if a not in vals or b not in vals:
            return
        d = vals[b] - vals[a]
        verdict = "YES" if d > 0.01 else ("no" if d < -0.01 else "tie")
        print(f"\n{question}\n  {a} {vals[a]:.4f} vs {b} {vals[b]:.4f} "
              f"-> delta {d:+.4f} nats  [{verdict}]")

    delta("ple", "baseline",
          "Q1. Does PLE beat a plain tiny LM at the same core budget?")
    delta("ple", "ple_notable",
          "Q2. Is the gain from the flash TABLE, or just the per-layer plumbing?")
    delta("ple", "fatembed",
          "Q3. Does injecting PER-LAYER beat the same params injected at the bottom?")
    delta("bigcore", "ple",
          "Q4. How much of a 2x-core model does PLE recover? (positive = core still wins)")

    if all(k in vals for k in ("ple", "bigcore", "baseline")):
        gap = vals["baseline"] - vals["bigcore"]
        got = vals["baseline"] - vals["ple"]
        if gap > 0:
            print(f"\n  PLE recovers {100 * got / gap:.0f}% of the gap between the "
                  f"baseline core and a 2x core, at zero extra SRAM.")


if __name__ == "__main__":
    main()
