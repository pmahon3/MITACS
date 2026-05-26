"""EXPLORATORY: is the +1000 MW post-cutoff climatology gap stationary?

We've shown the post-cutoff `actual - μ_W(t)` gap is ~+1000 MW at every
hour for every recency window (1y..10y, plus full).  Question: is this
gap stable across the 500-day post-cutoff window, or growing with time?

  - If stable: a single ΔL level-shift correction would mostly fix it.
  - If growing: demand is trending up post-cutoff; no static climatology
    correction can fix it.

Plots the gap as a function of delivery_date (smoothed) for full and W=2y.

PROVENANCE-GRADE: INSPECTION-ONLY.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiment import freeze
from experiment._actuals import load_actuals, mu_at, zscore_params
from scripts.fir.rescore_climatology import _fit_zscore_params_window


def main() -> None:
    spec = freeze.load_verified()
    cutoff = pd.Timestamp(spec["data_cutoff"])

    raw_full = load_actuals(cutoff=None).dropna()
    raw_pre = raw_full[raw_full.index <= cutoff]
    raw_post = raw_full[raw_full.index > cutoff]

    zp_full = zscore_params(cutoff, method="month_hour")
    zp_2y = _fit_zscore_params_window(raw_pre, 2.0)

    mu_full = pd.Series(mu_at(zp_full, raw_post.index), index=raw_post.index)
    mu_2y   = pd.Series(mu_at(zp_2y,   raw_post.index), index=raw_post.index)

    gap_full = raw_post - mu_full
    gap_2y   = raw_post - mu_2y

    # By quarter
    print("=" * 60)
    print("gap = actual - μ, averaged per calendar quarter")
    print("=" * 60)
    print(f"{'quarter':>10s}  {'full':>8s}  {'W=2y':>8s}  {'n_rows':>7s}")
    for (yr, q), idx in gap_full.groupby([gap_full.index.year, gap_full.index.quarter]):
        n = len(idx)
        gf = idx.mean()
        g2 = gap_2y.loc[idx.index].mean()
        print(f"  {yr}-Q{q}  {gf:+8.0f}  {g2:+8.0f}  {n:>7d}")

    # By month
    print()
    print("=" * 60)
    print("gap = actual - μ, averaged per calendar month")
    print("=" * 60)
    print(f"{'month':>9s}  {'full':>8s}  {'W=2y':>8s}  {'n_rows':>7s}")
    for (yr, m), idx in gap_full.groupby([gap_full.index.year, gap_full.index.month]):
        n = len(idx)
        gf = idx.mean()
        g2 = gap_2y.loc[idx.index].mean()
        print(f"  {yr}-{m:02d}  {gf:+8.0f}  {g2:+8.0f}  {n:>7d}")

    # End-effect check: is there a trend within the post-cutoff window?
    # Split into halves
    n = len(gap_full)
    mid = gap_full.index[n // 2]
    print()
    print("=" * 60)
    print(f"first half ({gap_full.index.min().date()}..{mid.date()}) vs "
          f"second half ({mid.date()}..{gap_full.index.max().date()})")
    print("=" * 60)
    h1 = gap_full[gap_full.index < mid].mean()
    h2 = gap_full[gap_full.index >= mid].mean()
    print(f"  full:  H1 {h1:+.0f} MW  vs  H2 {h2:+.0f} MW  Δ {h2-h1:+.0f}")
    h1 = gap_2y[gap_2y.index < mid].mean()
    h2 = gap_2y[gap_2y.index >= mid].mean()
    print(f"  W=2y:  H1 {h1:+.0f} MW  vs  H2 {h2:+.0f} MW  Δ {h2-h1:+.0f}")

    # Also: gap vs pre-cutoff "tail" gap to check stationarity
    # i.e. what was actual - μ in the last year of pre-cutoff?
    print()
    print("=" * 60)
    print("pre-cutoff sanity: actual - μ for the LAST year before cutoff")
    print("(should be near 0 for full, near 0 for W=2y by construction)")
    print("=" * 60)
    last_y_start = cutoff - pd.Timedelta(days=365)
    tail = raw_pre[raw_pre.index > last_y_start]
    mu_full_tail = pd.Series(mu_at(zp_full, tail.index), index=tail.index)
    mu_2y_tail   = pd.Series(mu_at(zp_2y,   tail.index), index=tail.index)
    gap_full_tail = (tail - mu_full_tail).mean()
    gap_2y_tail   = (tail - mu_2y_tail).mean()
    print(f"  full:  {gap_full_tail:+.0f} MW")
    print(f"  W=2y:  {gap_2y_tail:+.0f} MW")
    print(f"  (n={len(tail)} rows over {last_y_start.date()}..{cutoff.date()})")


if __name__ == "__main__":
    main()
