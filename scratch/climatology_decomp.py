"""EXPLORATORY: decompose what's drifting in the climatology.

The W=2y rescore fixed the mid-day bias but introduced a late-night
under-forecast.  Likely cause: different windows pin mu_{m,h} at
different levels per (month, hour) bin, and the post-cutoff actuals
are closer to some windows at some hours, others at others.

This script computes, for each W in {1, 2, 3, 5, 10, "full"} years:

  - mu_{m,h}(W) evaluated on the pre-cutoff window
  - bias_{m,h}(W) = mu_{m,h}(W) - actual_post_mean_{m,h}
  - same for sigma_{m,h}(W)

Then prints the per-W (mu - actual) marginalised over month (giving
the diurnal-bias profile) so we can see which W minimises bias at
each HE.

The hope is that some window minimises the *overall* by-hour bias
better than W=2y; the alternative is that the diurnal shape of the
post-cutoff actuals has shifted in a way no single static climatology
window can match.

PROVENANCE-GRADE: INSPECTION-ONLY.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiment import freeze
from experiment._actuals import load_actuals, mu_at, sigma_at

from scripts.fir.rescore_climatology import _fit_zscore_params_window


def main() -> None:
    spec = freeze.load_verified()
    cutoff = pd.Timestamp(spec["data_cutoff"])
    raw_full = load_actuals(cutoff=None).dropna()
    raw_pre = raw_full[raw_full.index <= cutoff]
    raw_post = raw_full[raw_full.index > cutoff]

    # Reference: post-cutoff actuals' hourly mean (one number per HE)
    actual_by_hour = raw_post.groupby(raw_post.index.hour).mean()

    windows = [1.0, 2.0, 3.0, 5.0, 10.0]

    # Also include "full" using the spec's frozen zscore_params
    from experiment._actuals import zscore_params
    clim_method = spec["predictor"].get("climatology_method", "month_hour")
    zp_full = zscore_params(cutoff, method=clim_method,
                             k_year=spec["predictor"].get("fourier_k_year"),
                             k_day=spec["predictor"].get("fourier_k_day"))

    # For each W: mu(t) and sigma(t) evaluated at every post-cutoff
    # timestamp, then averaged per HE (hour-of-day 0..23 = HE1..HE24)
    print("=== mu_{m,h}(W) - actual_post_mean_{m,h}, averaged over month per HE ===")
    print("    +ve => climatology higher than actuals at that hour")
    print()
    print(f"  {'HE':>3s}  " + "  ".join(f"{'W='+str(int(w))+'y':>7s}" for w in windows) + f"  {'full':>7s}  {'actual':>8s}")
    bias_table = {}
    for W in windows:
        zp = _fit_zscore_params_window(raw_pre, W)
        mu_arr = mu_at(zp, raw_post.index)
        mu_s = pd.Series(mu_arr, index=raw_post.index)
        bias = (mu_s.groupby(mu_s.index.hour).mean() - actual_by_hour)
        bias_table[W] = bias

    mu_full = mu_at(zp_full, raw_post.index)
    mu_full_s = pd.Series(mu_full, index=raw_post.index)
    bias_full = (mu_full_s.groupby(mu_full_s.index.hour).mean() - actual_by_hour)
    bias_table["full"] = bias_full

    for h in range(24):
        vals = [f"{int(round(bias_table[W].iloc[h])):>+7d}" for W in windows]
        vals.append(f"{int(round(bias_table['full'].iloc[h])):>+7d}")
        print(f"  HE{h+1:>2d}  " + "  ".join(vals) + f"  {actual_by_hour.iloc[h]:>8.0f}")

    print()
    print("=== sigma_{m,h}(W), averaged over month per HE ===")
    print(f"  {'HE':>3s}  " + "  ".join(f"{'W='+str(int(w))+'y':>7s}" for w in windows) + f"  {'full':>7s}")
    sigma_table = {}
    for W in windows:
        zp = _fit_zscore_params_window(raw_pre, W)
        sig_arr = sigma_at(zp, raw_post.index)
        sig_s = pd.Series(sig_arr, index=raw_post.index)
        sigma_table[W] = sig_s.groupby(sig_s.index.hour).mean()
    sig_full = sigma_at(zp_full, raw_post.index)
    sig_full_s = pd.Series(sig_full, index=raw_post.index)
    sigma_table["full"] = sig_full_s.groupby(sig_full_s.index.hour).mean()
    for h in range(24):
        vals = [f"{int(round(sigma_table[W].iloc[h])):>7d}" for W in windows]
        vals.append(f"{int(round(sigma_table['full'].iloc[h])):>7d}")
        print(f"  HE{h+1:>2d}  " + "  ".join(vals))

    print()
    print("=== summary: best W per HE (min |bias|) ===")
    best_W = []
    for h in range(24):
        abs_biases = {W: abs(bias_table[W].iloc[h]) for W in windows + ["full"]}
        best = min(abs_biases, key=abs_biases.get)
        best_W.append(best)
        bias = bias_table[best].iloc[h]
        print(f"  HE{h+1:>2d}  best W = {str(best):>5s}  bias = {int(round(bias)):>+6d} MW")

    # Aggregate: is there a single W that's best at most hours?
    from collections import Counter
    print()
    print(f"per-HE winners: {Counter(best_W).most_common()}")


if __name__ == "__main__":
    main()
