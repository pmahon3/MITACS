"""EXPLORATORY: figure: post-cutoff climatology gap is non-stationary.

Two panels:
  (top)    monthly mean of (actual - μ_full(t)) and (actual - μ_W2y(t))
           — both show clear upward drift through 2025-2026 and large
           seasonal swings.  No level-shift can fix this.
  (bottom) the operator decomposition by HE (already shown in
           clim_bias_verify) for context.

PROVENANCE-GRADE: INSPECTION-ONLY.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from experiment import freeze
from experiment._actuals import load_actuals, mu_at, sigma_at, zscore_params
from scripts.fir.rescore_climatology import _fit_zscore_params_window


OUT = Path("writeup/tex/figs/fig_clim_gap_temporal.pdf")


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

    # Also include last 2 years of pre-cutoff as context
    last_2y = raw_pre[raw_pre.index > cutoff - pd.Timedelta(days=730)]
    mu_full_pre = pd.Series(mu_at(zp_full, last_2y.index), index=last_2y.index)
    mu_2y_pre   = pd.Series(mu_at(zp_2y,   last_2y.index), index=last_2y.index)
    gap_full_pre = last_2y - mu_full_pre
    gap_2y_pre   = last_2y - mu_2y_pre

    # Monthly means
    def by_month(s):
        return s.resample("ME").mean()

    fig, ax = plt.subplots(1, 1, figsize=(8, 4.2))
    ax.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax.axvline(cutoff, color="k", lw=0.8, ls="--", alpha=0.5)
    ax.text(cutoff, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 100,
            " data\n cutoff", fontsize=8, va="top", ha="left", alpha=0.6)
    bm_full = by_month(pd.concat([gap_full_pre, gap_full]).sort_index())
    bm_2y   = by_month(pd.concat([gap_2y_pre,   gap_2y]).sort_index())
    ax.plot(bm_full.index, bm_full.values, "o-", label="full-history μ",
            color="C0", lw=1.4, ms=4)
    ax.plot(bm_2y.index,   bm_2y.values,   "s-", label="W=2y μ",
            color="C1", lw=1.4, ms=4)
    ax.set_ylabel("monthly mean (actual − μ)  [MW]")
    ax.set_xlabel("calendar month")
    ax.set_title("Post-cutoff climatology gap is non-stationary "
                 "(no recency window can fix this)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
