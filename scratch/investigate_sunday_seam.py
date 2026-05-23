"""Is the sunday d=4 Sigma singularity an embedding/library issue, NOT
a dynamics issue?

Hypothesis (user, 2026-05-22): the sunday d=4 embedded state at anchors
near the start of the sunday window contains LAG coordinates that fall
on Saturday (because lags reach back across the day-type boundary). When
the production sunday-only library is built, the anchor timestamps are
sunday-tagged, but their lag coordinates can be saturday-tagged
underneath -- a quiet contamination the day-type filter doesn't catch.

This probe characterises the contamination without changing production:
  Q1  How many sunday d=4 anchors have at least one lag coord on Saturday?
  Q2  What fraction of the library does this represent?
  Q3  If we EXCLUDE those contaminated anchors from the library, does
      Sigma at the remaining anchors become well-conditioned (without
      the seam mask)? If yes: the mask was a downstream patch for an
      upstream contamination; the right fix is library-side.

PROVENANCE-GRADE: INSPECTION-ONLY.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from edynamics.modelling_tools import Embedding, Lag

from config import load_config
from experiment._actuals import load_actuals, zscore_params, zscore_transform


def main():
    cfg = load_config()
    anchor_h = cfg.data.day_anchor_hours
    print(f"anchor_h = {anchor_h}")

    # Load the day-type-tagged z-series (the same artifact production
    # uses; see processing/clustering/process.py).
    clustered = pd.read_csv(
        cfg.paths.clustered_csv, index_col=0, parse_dates=True
    ).asfreq("h")
    cutoff = pd.Timestamp("2024-12-31T23:00:00")
    pre = clustered[clustered.index <= cutoff].copy()

    sunday_anchors = pre.index[pre["daytype"] == "sunday"]
    print(f"sunday anchors (pre-cutoff): {len(sunday_anchors)}")

    # ----- Q1+Q2: how many sunday anchors have lag coords on Saturday? --
    d = 4
    contaminated_mask = []
    for t in sunday_anchors:
        # lag coordinates: t, t-1, t-2, t-3
        lag_ts = [t - pd.Timedelta(hours=i) for i in range(d)]
        lag_types = []
        for lt in lag_ts:
            if lt in pre.index:
                lag_types.append(pre.loc[lt, "daytype"])
            else:
                lag_types.append(None)
        any_non_sunday = any(
            (lt != "sunday" and lt is not None) for lt in lag_types
        )
        contaminated_mask.append(any_non_sunday)
    contaminated_mask = np.array(contaminated_mask)
    n_contam = int(contaminated_mask.sum())
    n_total = len(contaminated_mask)
    print(f"\nQ1+Q2: sunday d={d} anchors with at least one non-sunday "
          f"lag coord:")
    print(f"  contaminated anchors: {n_contam} / {n_total}  "
          f"({100*n_contam/n_total:.1f}%)")

    # what HOURS do the contaminated anchors fall on?
    contam_anchors = sunday_anchors[contaminated_mask]
    by_hour = pd.Series(contam_anchors).dt.hour.value_counts().sort_index()
    print(f"\n  contaminated anchors by hour-of-day:")
    for h, n in by_hour.items():
        print(f"    {h:>02d}:00  n={n}")

    # ----- Q3: Sigma condition WITHOUT the seam mask, comparing the FULL
    # sunday library vs the CLEAN sunday library (drop contaminated
    # anchors at library-build time too) -----
    from scratch.backtest_simplex_theta import _pre_cutoff_embedding
    from processing.innovations.estimator import local_drift_and_diffusion

    zp = zscore_params(cutoff, method="month_hour")
    zfull = zscore_transform(load_actuals(cutoff=None), zp)
    zpre = zfull[zfull.index <= cutoff]
    emb = _pre_cutoff_embedding(zpre, d=d)
    print(f"\nQ3: Sigma condition at sunday anchors, mask OFF "
          f"(seam mask removed 2026-05-22), 8 random anchors per group")
    rng = np.random.default_rng(0)
    clean_anchors = sunday_anchors[~contaminated_mask]
    if len(clean_anchors) < 8 or len(contam_anchors) < 8:
        print("  (insufficient anchors in one group; skipping)")
        return

    def _report(name, anchors_pool, n=8):
        sample = pd.DatetimeIndex(
            np.sort(rng.choice(anchors_pool, n, replace=False))
        )
        conds = []
        for a in sample:
            try:
                _C, S, _mu, _th, _r = local_drift_and_diffusion(
                    embedding=emb, anchor=a,
                )
                eig = np.linalg.eigvalsh(S)
                eig_pos = eig[eig > 1e-15]
                cond = (eig.max() / eig_pos.min()
                        if len(eig_pos) else float("inf"))
                conds.append(cond)
            except Exception as e:
                print(f"  {name}  {a}: ERROR {e!r}")
        if conds:
            conds = np.array(conds)
            print(f"  {name}  cond(Sigma)  median={np.median(conds):.2e}  "
                  f"min={conds.min():.2e}  max={conds.max():.2e}")

    _report("CONTAMINATED sunday anchors", contam_anchors)
    _report("CLEAN sunday anchors      ", clean_anchors)

    # Also: condition on a SATURDAY anchor (sanity -- saturday has its
    # own d, but if we use d=4 too, does the same contamination story
    # apply?)
    saturday_anchors = pre.index[pre["daytype"] == "saturday"]
    if len(saturday_anchors) >= 8:
        _report("ALL saturday anchors (d=4)", saturday_anchors)


if __name__ == "__main__":
    main()
