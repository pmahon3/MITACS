"""Stable re-summary of scratch/k_loo_score.py's output. The original
script used dB(h) = (LL_gauss - LL_t) / |LL_gauss| matching the
existing M2 protocol, but that ratio is degenerate at h where
baseline LL is near zero — the ratio's denominator dominates and the
day-bootstrap CI blows up. This script reports:
  - mean LL reduction (raw difference in nats) — stable, additive,
    interpretable as "how many nats per forecast does t save vs Gauss"
  - median LL reduction — robust to extreme tail events
  - fraction of days t beats Gauss — frequency-flavoured signal
  - the single worst Gaussian failure per h (max LL_gauss) and t's
    LL on the same day — characterises the catastrophic-event channel

Plus a "robustness" reading: remove the day with the worst Gaussian
LL and see whether the mean reduction survives. This separates the
"Student-t handles routine tails" story from the "Student-t handles
one catastrophic event" story.

PROVENANCE-GRADE: INSPECTION-ONLY.
"""
from __future__ import annotations

import pickle
from pathlib import Path
import numpy as np
import pandas as pd

IN = Path("/tmp/k_loo_dB.pkl")
B = 1000
RNG = np.random.default_rng(20260520)


def day_bootstrap_mean(values: np.ndarray, days: np.ndarray) -> tuple:
    uniq = np.unique(days)
    by_day = {d: np.where(days == d)[0] for d in uniq}
    boots = np.empty(B)
    for b in range(B):
        samp = RNG.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by_day[d] for d in samp])
        boots[b] = values[idx].mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi)


def main() -> None:
    with IN.open("rb") as f:
        blob = pickle.load(f)
    raw = blob["raw"].copy()
    raw["delta_ll"] = raw["ll_gauss"] - raw["ll_t"]

    print("=" * 92)
    print("Option K — STABLE re-summary (raw log-loss difference; "
          "median; worst-case)")
    print("=" * 92)
    print(f"  {'h':>3} {'n':>4} {'mean Δll':>9} {'95% CI':>17} "
          f"{'median Δll':>10} {'t wins':>7} "
          f"{'max LL_g':>9} {'(t LL)':>8}")
    rows = []
    for h, g in raw.groupby("h"):
        g = g.dropna(subset=["delta_ll"])
        if len(g) < 30:
            continue
        mean, lo, hi = day_bootstrap_mean(
            g["delta_ll"].to_numpy(), g["day"].to_numpy()
        )
        median = float(g["delta_ll"].median())
        win_frac = float((g["delta_ll"] > 0).mean())
        # worst Gaussian failure
        worst_idx = g["ll_gauss"].idxmax()
        worst_row = g.loc[worst_idx]
        rows.append({"h": int(h), "mean": mean, "lo": lo, "hi": hi,
                     "median": median, "win_frac": win_frac,
                     "max_llg": float(worst_row["ll_gauss"]),
                     "lt_at_worst": float(worst_row["ll_t"])})
        print(f"  {int(h):>3} {len(g):>4} {mean:>9.4f}  "
              f"[{lo:>+7.4f}, {hi:>+6.4f}] "
              f"{median:>10.4f} {win_frac:>7.1%} "
              f"{worst_row['ll_gauss']:>9.3f} {worst_row['ll_t']:>8.3f}")

    df = pd.DataFrame(rows)

    # Robustness: drop the day with the worst Gaussian LL per h, recompute mean
    print()
    print("=" * 92)
    print("Robustness — repeat with the WORST single Gaussian-LL day "
          "REMOVED at each h:")
    print(f"  {'h':>3} {'n':>4} {'mean Δll':>9} {'95% CI':>17} "
          f"{'median Δll':>10}")
    for h, g in raw.groupby("h"):
        g = g.dropna(subset=["delta_ll"])
        if len(g) < 30:
            continue
        worst_idx = g["ll_gauss"].idxmax()
        g2 = g.drop(worst_idx)
        mean, lo, hi = day_bootstrap_mean(
            g2["delta_ll"].to_numpy(), g2["day"].to_numpy()
        )
        median = float(g2["delta_ll"].median())
        print(f"  {int(h):>3} {len(g2):>4} {mean:>9.4f}  "
              f"[{lo:>+7.4f}, {hi:>+6.4f}] {median:>10.4f}")

    # Headline summary
    print()
    print("=" * 92)
    print("Headline: how many horizons have CI-clean t-advantage on the "
          "raw Δll metric?")
    print("=" * 92)
    df["clean_pos"] = df["lo"] > 0
    print(f"  CI-clean positive at: {df[df['clean_pos']]['h'].tolist()}")
    print(f"  Median across h: mean Δll = "
          f"{df['mean'].median():.4f} nats, "
          f"median Δll = {df['median'].median():.4f} nats")
    avg_winrate = df["win_frac"].mean()
    print(f"  Average win-rate across h: {avg_winrate:.1%}")
    print()
    print("Interpretation key:")
    print("  Δll > 0       => Student-t assigns higher density to actual")
    print("  median Δll    => the typical-day advantage (robust)")
    print("  mean Δll      => includes tail-event advantage (heavy-influenced)")
    print("  win_frac      => coin-flip baseline is 50%")


if __name__ == "__main__":
    main()
