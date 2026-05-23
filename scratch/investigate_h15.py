"""Investigation: why does dropping target-hour 15:00 collapse the
global-OLS residual kurtosis 63 -> 8 on Ontario pre-cutoff data?

Stated anomaly (scratch/measure_seam_kurtosis.py): on the 198,740-pair
weekday d=2 library at anchor_h=0, the global-OLS coord-0 residual has
excess kurtosis 63. Dropping the 4.2% of pairs whose TARGET time hits
15:00 brings it to 8. No other clock hour produces an effect of
comparable size. Variance at 15:00 is NOT unusually large.

The four discriminating checks (no story-spinning -- just the data):

  C1  RAW DATA vs FITTED RESIDUAL.
      Compute per-target-hour kurtosis on (a) raw z_{t+1}; (b) first-
      difference z_{t+1} - z_t. If 15:00 is anomalous in raw data, the
      predictor is irrelevant.

  C2  COORD 0 vs COORD 1 (the multivariate residual shape).
      The §2/§3 rank-1 structural finding says coord 1 should be ~0 and
      uninformative. Verify, and check the joint structure at 15:00.

  C3  TARGET-HOUR vs SOURCE-HOUR.
      The original probe masked target == 15:00 (transition INTO 15:00).
      Re-run masking source == 15:00 (transition OUT of 15:00) and the
      symmetric mask (either equals 15:00). The directionality tells us
      whether 15:00 is a sink, a source, or both.

  C4  RAW DEMAND (climatology-free).
      Re-run on raw demand residuals (no z-score), to check whether the
      anomaly is in the dynamics or in the (mu_{m,h}, sigma_{m,h})
      climatology mis-fit at 15:00.

PROVENANCE-GRADE: INSPECTION-ONLY.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis

from config import load_config
from experiment._actuals import (
    load_actuals,
    zscore_params,
    zscore_transform,
)
from scratch.backtest_simplex_theta import _pre_cutoff_embedding


def _kurt(x: np.ndarray) -> float:
    return float(kurtosis(x, fisher=True, bias=False))


def _sweep(name: str, values: np.ndarray, hours: np.ndarray):
    """Print kurtosis of `values` when each clock hour is dropped from
    `hours`. Highlights anomalous hours (|kurt_drop| > 5 vs baseline)."""
    base = _kurt(values)
    print(f"\n  {name}: baseline kurt = {base:+.2f}  (n={len(values)})")
    print(f"    {'h':>3}  {'kept_kurt':>11}  {'delta_vs_base':>14}"
          f"  {'n_dropped':>10}  {'var_dropped':>12}")
    for h in range(24):
        keep = hours != h
        n_drop = int((~keep).sum())
        if keep.sum() == 0 or n_drop == 0:
            continue
        k = _kurt(values[keep])
        var_drop = float(np.var(values[~keep])) if n_drop > 0 else float("nan")
        flag = "  <<<" if abs(k - base) > 5 else ""
        print(f"    {h:>3}  {k:>+11.2f}  {k - base:>+14.2f}"
              f"  {n_drop:>10d}  {var_drop:>12.4e}{flag}")


def main():
    cfg = load_config()
    cutoff = pd.Timestamp("2024-12-31T23:00:00")
    zp = zscore_params(cutoff, method="month_hour")
    raw = load_actuals(cutoff=None)
    raw_pre = raw[raw.index <= cutoff]
    zfull = zscore_transform(raw, zp)
    zpre = zfull[zfull.index <= cutoff]

    print("=" * 72)
    print("INVESTIGATING h=15 KURTOSIS ANOMALY  (INSPECTION-ONLY)")
    print(f"  config anchor_h = {cfg.data.day_anchor_hours}")
    print("=" * 72)

    # --- C1: raw and differenced z-scores --------------------------------
    print("\n----- C1: RAW DATA vs FITTED RESIDUAL -----")
    z = zpre.dropna()
    hr = z.index.hour.to_numpy()
    _sweep("raw z (no predictor)", z.values, hr)

    # first difference: z_{t+1} - z_t, indexed by target time t+1
    dz = z.diff().dropna()
    hr_dz = dz.index.hour.to_numpy()
    _sweep("first-diff z (target hour)", dz.values, hr_dz)

    # --- C2: per-coord residual shape (the weekday d=2 embedding) ---------
    print("\n----- C2: COORD 0 vs COORD 1 (rank-1 check) -----")
    emb = _pre_cutoff_embedding(zpre, d=2)
    block = emb.block.values
    hours = emb.block.index.hour.to_numpy()
    X = block[:-1]; Y = block[1:]
    target_hours = hours[1:]
    source_hours = hours[:-1]
    C = np.linalg.lstsq(X, Y, rcond=None)[0]
    resid = Y - X @ C
    _sweep("global-OLS resid coord 0 (target hour)",
           resid[:, 0], target_hours)
    _sweep("global-OLS resid coord 1 (target hour)",
           resid[:, 1], target_hours)
    # joint kurtosis-equivalent: largest eigenvalue's coord across pairs
    # use Frobenius norm of residual vector as a scalar shape proxy
    rnorm = np.linalg.norm(resid, axis=1)
    _sweep("global-OLS resid Frobenius-norm (target hour)",
           rnorm, target_hours)

    # --- C3: target vs source vs both ------------------------------------
    print("\n----- C3: TARGET vs SOURCE hour -----")
    _sweep("global-OLS resid coord 0 (SOURCE hour)",
           resid[:, 0], source_hours)
    print("\n  COMBINED (drop pairs where EITHER endpoint == h):")
    base = _kurt(resid[:, 0])
    print(f"    {'h':>3}  {'kept_kurt':>11}  {'delta_vs_base':>14}"
          f"  {'n_dropped':>10}")
    for h in range(24):
        keep = (source_hours != h) & (target_hours != h)
        if keep.sum() == 0: continue
        k = _kurt(resid[keep, 0])
        flag = "  <<<" if abs(k - base) > 5 else ""
        n_drop = int((~keep).sum())
        print(f"    {h:>3}  {k:>+11.2f}  {k - base:>+14.2f}"
              f"  {n_drop:>10d}{flag}")

    # --- C4: raw demand (no z-score) ------------------------------------
    print("\n----- C4: RAW DEMAND (climatology-free) -----")
    d_raw = raw_pre.dropna()
    d_raw = d_raw[d_raw.index.isin(zpre.index)]  # same support
    # Just the differenced raw demand by target hour
    dd = d_raw.diff().dropna()
    hr_dd = dd.index.hour.to_numpy()
    _sweep("first-diff D(t+1)-D(t) raw demand (target hour)",
           dd.values, hr_dd)
    # And a global-OLS-equivalent residual in raw-demand space.
    # Use the same d=2 lag structure but on raw demand.
    d_for = pd.DataFrame({"D": d_raw})
    d_for["D_lag1"] = d_for["D"].shift(1)
    d_for["D_target"] = d_for["D"].shift(-1)
    d_for = d_for.dropna()
    Xr = d_for[["D", "D_lag1"]].values
    Yr = d_for["D_target"].values
    Cr = np.linalg.lstsq(Xr, Yr, rcond=None)[0]
    rresid = Yr - Xr @ Cr
    hr_r = d_for.index.to_series().shift(-1).dropna().index.hour.to_numpy()
    # align: rresid is for target = next hour, hour-of-target = d_for.index + 1h
    hr_target_raw = (d_for.index + pd.Timedelta(hours=1)).hour.to_numpy()
    _sweep("raw-demand 2-lag-OLS resid (target hour)",
           rresid, hr_target_raw)

    print("\n" + "=" * 72)
    print("SUMMARY  -- look for the rows flagged '<<<' (|delta|>5)")
    print("If h=15 keeps standing out in C1 (raw z, raw difference)")
    print("  -> anomaly is in the DATA, not the predictor.")
    print("If it appears only in fitted residuals (C2 coord 0)")
    print("  -> property of how the fit responds to 15:00.")
    print("Source-vs-target asymmetry in C3 tells transition direction.")
    print("If it persists in raw demand (C4)")
    print("  -> not a climatology artefact.")
    print("=" * 72)


if __name__ == "__main__":
    main()
