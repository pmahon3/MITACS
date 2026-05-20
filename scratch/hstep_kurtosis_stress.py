"""Stress tests for the kurt(h) finding before it carries Stage-2
load in writeup/tex/qualifier_exhaustion_memo.tex:

  (1) Day-bootstrap CI on excess kurt per horizon.
      Sample kurtosis at n~132 days has wide variance. A point
      estimate of 1.13 at h=3 is not the same as a CI cleanly
      excluding moderate heavy-tailedness.

  (2) Per-day-type kurt(h).
      The rebaseline established kurt ~24-33 PER DAY-TYPE SLAB.
      My pooled cache reads 8.60 at h=1. If the kurt(h)-decays-with-h
      pattern is mostly a POOLING/MIXING effect (mixture of three
      day-types with different drift centres ~looks~ closer to
      Gaussian than each component does), then composition is NOT
      laundering the tail — pooling is. Discriminate by computing
      kurt(h) within each day-type separately.

  (3) Interpret the negative-kurt tail at h>=8.
      Three readings: (i) genuine, recursion variance grows enough
      that residuals saturate; (ii) finite-sample artefact at n=132;
      (iii) over-scaling already ruled out by raw==std agreement.
      Bootstrap CI from (1) directly tests (ii).

Reuses the augmented-scope cache. No estimator reimplementation.
PROVENANCE: INSPECTION-ONLY.
"""
from __future__ import annotations

import pickle
import numpy as np
import pandas as pd

CACHE = "/tmp/bench2122_fc_propagated_augmented.pkl"
B = 2000   # bootstrap replicates
RNG = np.random.default_rng(20260520)
HORIZONS_TO_REPORT = [1, 2, 3, 4, 6, 8, 12, 18, 24]


def daytype_of(date) -> str:
    """Match the project's day-type convention (weekday / saturday /
    sunday). The cache's delivery_date is a date-like; saturday=5,
    sunday=6 in pandas weekday convention."""
    wd = pd.Timestamp(date).weekday()
    if wd == 5:
        return "saturday"
    if wd == 6:
        return "sunday"
    return "weekday"


def kurt(x: np.ndarray) -> float:
    """Fisher excess kurtosis (0 = Gaussian) via pandas (n>=4)."""
    x = x[np.isfinite(x)]
    if len(x) < 4:
        return float("nan")
    return float(pd.Series(x).kurtosis())


def day_bootstrap_kurt(err_df: pd.DataFrame, B: int) -> tuple:
    """Day-blocked bootstrap of kurt on a single horizon's residual
    series. err_df has columns {err, delivery_date}. Returns
    (point_est, lo95, hi95, n_days)."""
    point = kurt(err_df["err"].to_numpy())
    days = err_df["delivery_date"].unique()
    n_days = len(days)
    by_day = {d: err_df[err_df["delivery_date"] == d]["err"].to_numpy()
              for d in days}
    boots = np.empty(B)
    for b in range(B):
        samp = RNG.choice(days, n_days, replace=True)
        concat = np.concatenate([by_day[d] for d in samp])
        boots[b] = kurt(concat)
    boots = boots[np.isfinite(boots)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi), int(n_days)


def main() -> None:
    with open(CACHE, "rb") as f:
        fc = pickle.load(f)

    fc = fc.copy()
    fc["err"] = fc["z_actual"] - fc["z_pred"]
    fc["daytype"] = fc["delivery_date"].apply(daytype_of)

    print(f"cache: n_rows={len(fc)}, "
          f"n_days={fc['delivery_date'].nunique()}, "
          f"daytypes={sorted(fc['daytype'].unique())}")
    print()

    # === (1) Day-bootstrap CI on pooled kurt(h) ===
    print("=" * 78)
    print("(1) Day-bootstrap 95% CI on excess kurt(h) -- pooled across "
          "day-types")
    print("=" * 78)
    print(f"{'h':>3} {'n':>5} {'point':>8} {'lo95':>8} {'hi95':>8}  "
          f"{'reading':<35}")
    rows1 = []
    for h in HORIZONS_TO_REPORT:
        g = fc[fc["horizon_h"] == h][["err", "delivery_date"]]
        if len(g) < 30:
            continue
        point, lo, hi, n_days = day_bootstrap_kurt(g, B)
        # Conservative reading:
        # CI above 1: clearly heavy-tailed.
        # CI below 1: kernel form approximately adequate at this h.
        # CI straddles 1: uncertain — Stage 2 verdict softens.
        if lo > 1.0:
            tag = "heavy-tailed (CI > 1)"
        elif hi < 1.0:
            tag = "approx-Gaussian (CI < 1)"
        else:
            tag = "UNCERTAIN (CI straddles 1)"
        rows1.append({"h": h, "point": point, "lo": lo, "hi": hi,
                      "tag": tag})
        print(f"{h:>3} {len(g):>5} {point:>8.2f} {lo:>8.2f} {hi:>8.2f}  "
              f"{tag:<35}")

    # === (2) Per-day-type kurt(h) — is the decay a pooling artefact? ===
    print()
    print("=" * 78)
    print("(2) Per-day-type excess kurt(h) -- if decay is pooling-driven, "
          "it dissolves here")
    print("=" * 78)
    for dt in ["weekday", "saturday", "sunday"]:
        sub = fc[fc["daytype"] == dt]
        if len(sub) == 0:
            continue
        n_days_dt = sub["delivery_date"].nunique()
        n_caveat = " [n<30; QUALITATIVE only]" if n_days_dt < 30 else ""
        print(f"\n  day-type: {dt}  (n_days = {n_days_dt}){n_caveat}")
        print(f"  {'h':>3} {'n':>5} {'kurt':>10}")
        for h in HORIZONS_TO_REPORT:
            g = sub[sub["horizon_h"] == h]["err"].to_numpy()
            g = g[np.isfinite(g)]
            if len(g) < 10:    # relaxed for sat/sun qualitative read
                continue
            # pd.Series.kurtosis returns NaN for n<4; relaxed bar is n>=10
            print(f"  {h:>3} {len(g):>5} {kurt(g):>10.2f}")

    # === (3) Summary verdict ===
    print()
    print("=" * 78)
    print("(3) Stress-test verdict for the Stage-1 kurt(h) finding")
    print("=" * 78)

    # Pooled finding survives?
    h1 = next(r for r in rows1 if r["h"] == 1)
    h3 = next((r for r in rows1 if r["h"] == 3), None)
    h6 = next((r for r in rows1 if r["h"] == 6), None)
    print(f"h=1 CI: [{h1['lo']:.2f}, {h1['hi']:.2f}] -- {h1['tag']}")
    if h3 is not None:
        print(f"h=3 CI: [{h3['lo']:.2f}, {h3['hi']:.2f}] -- {h3['tag']}")
    if h6 is not None:
        print(f"h=6 CI: [{h6['lo']:.2f}, {h6['hi']:.2f}] -- {h6['tag']}")

    # Pooling check: compare per-day-type max-kurt at h=1 to pooled
    pooled_h1 = h1["point"]
    per_dt_h1 = {}
    for dt in ["weekday", "saturday", "sunday"]:
        g = fc[(fc["daytype"] == dt) & (fc["horizon_h"] == 1)][
            "err"].to_numpy()
        if len(g) >= 30:
            per_dt_h1[dt] = kurt(g)
    print()
    print(f"h=1 kurt pooled: {pooled_h1:.2f}")
    for dt, k in per_dt_h1.items():
        print(f"h=1 kurt {dt}: {k:.2f}")
    max_per_dt = max(per_dt_h1.values()) if per_dt_h1 else float("nan")
    if max_per_dt > 2 * pooled_h1 and pooled_h1 > 0:
        print("=> Per-day-type kurt(h=1) >> pooled: POOLING is "
              "thinning the tail at h=1 (mixing of day-type drift "
              "centres). The h=1 cache reading is a pooling artefact; "
              "the per-day-type residual is much heavier-tailed (and "
              "matches the rebaseline). Whether kurt(h) DECAY is also "
              "pooling-driven needs the per-day-type table above to "
              "show monotone decay within each slab.")
    else:
        print("=> Per-day-type kurt(h=1) ~ pooled: the pooled reading "
              "is honest at h=1.")


if __name__ == "__main__":
    main()
