"""Quick measurement: per-component residual excess kurtosis under the
day-anchor seam mask, at the current config anchor.

The §A appendix originally cited "133 -> 5" -- measured at anchor=7.
Under the realigned anchor=0, the mask drops different target hours, so
the empirical numbers must be re-measured before citation.

PROVENANCE-GRADE: INSPECTION-ONLY.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis

from config import load_config
from experiment._actuals import load_actuals, zscore_params, zscore_transform
from scratch.backtest_simplex_theta import _pre_cutoff_embedding


def main():
    cfg = load_config()
    anchor_h = cfg.data.day_anchor_hours
    print(f"Measuring residual excess kurtosis at anchor_h={anchor_h}")
    print(f"({'masked' if anchor_h is not None else 'no mask'} vs unmasked)")
    print("-" * 60)

    cutoff = pd.Timestamp("2024-12-31T23:00:00")
    zp = zscore_params(cutoff, method="month_hour")
    zfull = zscore_transform(load_actuals(cutoff=None), zp)
    zpre = zfull[zfull.index <= cutoff]

    # weekday d=2: the §A citation was for the weekday library
    emb = _pre_cutoff_embedding(zpre, d=2)
    block = emb.block.values
    hours = emb.block.index.hour.to_numpy()
    X = block[:-1]
    Y = block[1:]
    target_hours = hours[1:]

    # global OLS fit (deterministic, no LOO-CV noise)
    C = np.linalg.lstsq(X, Y, rcond=None)[0]
    resid = Y - X @ C
    n_total = len(resid)

    # unmasked kurtosis (per coord) -- "what the §A 133 was"
    kurt_unmasked = [
        float(kurtosis(resid[:, j], fisher=True, bias=False))
        for j in range(resid.shape[1])
    ]
    print(f"\nFull library  n={n_total}")
    print(f"  per-coord excess kurtosis: {[f'{k:.1f}' for k in kurt_unmasked]}")

    # masked: drop pairs whose TARGET hour == anchor_h
    keep = target_hours != anchor_h
    n_masked = int(keep.sum())
    pct_dropped = 100 * (1 - keep.mean())
    resid_m = resid[keep]
    kurt_masked = [
        float(kurtosis(resid_m[:, j], fisher=True, bias=False))
        for j in range(resid_m.shape[1])
    ]
    print(f"\nSeam-masked  n={n_masked}  ({pct_dropped:.1f}% of pairs dropped)")
    print(f"  per-coord excess kurtosis: {[f'{k:.1f}' for k in kurt_masked]}")

    print("\n--- summary line for §A ---")
    # coord-0 is the predicted variable; cite that (rank-1: stochastic
    # content is there)
    print(f"  weekday per-component coord-0 residual excess kurtosis: "
          f"{kurt_unmasked[0]:.0f} -> {kurt_masked[0]:.0f} "
          f"with target-hour {anchor_h:02d}:00 mask "
          f"({pct_dropped:.1f}% of pairs dropped)")


if __name__ == "__main__":
    main()
