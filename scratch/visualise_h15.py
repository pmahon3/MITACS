"""Visual inspection of the 14:00->15:00 demand transition anomaly.

The h=15 kurtosis investigation (scratch/investigate_h15.py) found a
robust, predictor-independent, climatology-independent signature in the
14:00->15:00 demand transition: dropping those pairs collapses the
residual kurtosis from 63 to 8 (and the same effect appears in raw
demand differences and the raw z-score first difference). The
*variance* of those pairs is ordinary -- the anomaly is in the
distribution SHAPE, not in the magnitude.

This probe gives a visual sense of that shape without modelling:

  PANEL 1  per-day trajectories D(13:00)..D(16:00), stratified by
           day-type, post-cutoff sample. Each thin line is one delivery
           day; the aggregate mean+/-1sigma envelope is overlaid.
           Visually look for: bimodality, asymmetric ramps, sharp
           direction change between 14:00 and 15:00 not present at
           other transitions.

  PANEL 2  histograms of the THREE adjacent demand differences:
              dD_13->14 = D(14) - D(13)
              dD_14->15 = D(15) - D(14)
              dD_15->16 = D(16) - D(15)
           The 14->15 histogram is the distribution whose kurtosis
           collapsed when removed; the other two are the controls.

PROVENANCE-GRADE: INSPECTION-ONLY.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment._actuals import load_actuals
from experiment.predict import _daytype
from config import load_config

from pathlib import Path
OUT_PNG = str(Path(__file__).resolve().parent / "data" / "h15_visual.png")
SEED = 20260522
N_PER_DAYTYPE = 30           # delivery days to plot, per day-type


def _gather_days(raw: pd.Series, anchor_h: int):
    """Return a dict daytype -> DatetimeIndex of complete delivery days
    in the post-cutoff window, stratified across the full span."""
    cutoff = pd.Timestamp("2024-12-31T23:00:00")
    days = pd.DatetimeIndex(sorted({ts.normalize() for ts in raw.index}))
    days = days[days > cutoff]
    by_dt: dict[str, list] = {"weekday": [], "saturday": [], "sunday": []}
    for D in days:
        # need 13:00..16:00 of this delivery day present
        hours_needed = [D + pd.Timedelta(hours=h) for h in (13, 14, 15, 16)]
        if not all(h in raw.index for h in hours_needed):
            continue
        # day-type at hour 12 (any in-day hour gives the same answer
        # under anchor_h=0; for other anchors, pick a representative)
        dt = _daytype(D + pd.Timedelta(hours=12), anchor_h)
        by_dt[dt].append(D)

    rng = np.random.default_rng(SEED)
    out = {}
    for dt, lst in by_dt.items():
        if len(lst) <= N_PER_DAYTYPE:
            out[dt] = pd.DatetimeIndex(lst)
        else:
            idx = rng.choice(len(lst), N_PER_DAYTYPE, replace=False)
            out[dt] = pd.DatetimeIndex(sorted(np.array(lst)[idx]))
    return out


def main():
    cfg = load_config()
    anchor_h = cfg.data.day_anchor_hours
    raw = load_actuals(cutoff=None)
    sample = _gather_days(raw, anchor_h)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8),
                              gridspec_kw={"hspace": 0.35, "wspace": 0.25})

    hours = [13, 14, 15, 16]
    daytypes = ["weekday", "saturday", "sunday"]
    colours = {"weekday": "#1f77b4", "saturday": "#ff7f0e",
               "sunday": "#2ca02c"}

    # --- PANEL 1: per-day trajectories 13..16 by day-type ---------------
    for ax, dt in zip(axes[0], daytypes):
        days = sample[dt]
        all_levels = []
        for D in days:
            vals = [float(raw.loc[D + pd.Timedelta(hours=h)])
                    for h in hours]
            ax.plot(hours, vals, color=colours[dt], lw=0.7, alpha=0.35)
            all_levels.append(vals)
        arr = np.array(all_levels)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        ax.plot(hours, mean, color="k", lw=2, label="mean")
        ax.fill_between(hours, mean - std, mean + std, color="k",
                        alpha=0.12, label="±1σ")
        ax.set_title(f"{dt}  (n={len(days)} days)")
        ax.set_xlabel("hour of day")
        ax.set_ylabel("demand (MW)")
        ax.set_xticks(hours)
        ax.axvline(14.5, color="red", lw=0.8, ls="--", alpha=0.6)
        ax.axvline(15.5, color="red", lw=0.8, ls=":", alpha=0.4)
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)

    # --- PANEL 2: histogram of adjacent demand differences --------------
    # pool over all sampled days across day-types (the kurtosis effect
    # appears across the whole library; per-daytype slicing would
    # over-fragment the n).
    all_days = pd.DatetimeIndex(sorted(set().union(*sample.values())))
    diffs = {"13→14": [], "14→15": [], "15→16": []}
    for D in all_days:
        d13 = float(raw.loc[D + pd.Timedelta(hours=13)])
        d14 = float(raw.loc[D + pd.Timedelta(hours=14)])
        d15 = float(raw.loc[D + pd.Timedelta(hours=15)])
        d16 = float(raw.loc[D + pd.Timedelta(hours=16)])
        diffs["13→14"].append(d14 - d13)
        diffs["14→15"].append(d15 - d14)
        diffs["15→16"].append(d16 - d15)

    # Use the same x-range so shapes are directly comparable.
    all_v = np.concatenate([np.array(v) for v in diffs.values()])
    lo, hi = np.percentile(all_v, [0.5, 99.5])
    pad = 0.1 * (hi - lo)
    xlim = (lo - pad, hi + pad)
    bins = np.linspace(*xlim, 40)

    from scipy.stats import kurtosis
    for ax, (label, vals) in zip(axes[1], diffs.items()):
        v = np.array(vals)
        kurt = float(kurtosis(v, fisher=True, bias=False))
        ax.hist(v, bins=bins, color="#888", edgecolor="k", lw=0.4)
        ax.set_xlim(*xlim)
        ax.set_title(
            f"ΔD ({label})\n"
            f"mean={v.mean():+.0f}  std={v.std():.0f}  "
            f"excess_kurt={kurt:+.2f}",
            fontsize=10,
        )
        ax.set_xlabel("Δ demand (MW)")
        ax.set_ylabel("count")
        ax.axvline(0, color="red", lw=0.8, alpha=0.5)
        ax.grid(alpha=0.3)

    fig.suptitle(
        "h=15 anomaly: visual inspection (INSPECTION-ONLY)\n"
        "top: per-day 13..16 demand trajectories by day-type; "
        "bottom: ΔD histograms (14→15 is the kurtosis-collapse hour)",
        fontsize=11,
    )
    plt.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT_PNG}")

    # ---- console summary so the visual has a numeric anchor -----------
    print("\nNumeric summary (pooled across day-types, sample n):")
    print(f"  {'transition':>10}  {'n':>5}  {'mean':>8}  "
          f"{'std':>7}  {'min':>7}  {'max':>7}  {'kurt':>7}")
    for label, vals in diffs.items():
        v = np.array(vals)
        k = float(kurtosis(v, fisher=True, bias=False))
        print(f"  {label:>10}  {len(v):>5d}  {v.mean():>+8.0f}  "
              f"{v.std():>7.0f}  {v.min():>+7.0f}  {v.max():>+7.0f}  "
              f"{k:>+7.2f}")


if __name__ == "__main__":
    main()
