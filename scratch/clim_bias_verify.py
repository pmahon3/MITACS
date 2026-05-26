"""EXPLORATORY: verify the per-HE climatology bias numbers reconcile.

climatology_decomp.py reported -700 to -1300 MW μ-vs-actual bias for
every recency window — yet the rescore pickle (cell_B_meaniter_global_W2y)
has mean_signed = +15 MW.  We dig in:

  (1) per-row μ(t) - actual(t),  averaged by HE
  (2) mean μ(t)|_h    - mean actual(t)|_h     (the decomp script's way)
  (3) reconcile against the rescore pickle:
        forecast - actual = (z_next * sigma_mh - (actual - mu_mh))
      should sum to mean_signed.

If (1) and (2) agree, then the -1000 MW μ-bias is real and the +1000 MW
overshoot has to come from z_next * sigma.  Decompose the rescore as

  forecast_signed_err(t) = σ(t)·z_next(t) - (actual(t) - μ(t))
                         = "operator overshoot in MW"  - "climatology gap in MW"

and report each part separately, per HE.  That tells us whether the
late-night bias is mainly climatology-driven (term 2) or operator-driven
(term 1).

PROVENANCE-GRADE: INSPECTION-ONLY.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiment import freeze
from experiment._actuals import load_actuals, mu_at, sigma_at, zscore_params
from scripts.fir.rescore_climatology import _fit_zscore_params_window


def per_hour(s: pd.Series) -> pd.Series:
    """Group by HE (hour-of-day 0..23, label as HE1..HE24)."""
    out = s.groupby(s.index.hour).mean()
    out.index = [f"HE{h+1:>2d}" for h in out.index]
    return out


def main() -> None:
    spec = freeze.load_verified()
    cutoff = pd.Timestamp(spec["data_cutoff"])

    raw_full = load_actuals(cutoff=None).dropna()
    raw_pre = raw_full[raw_full.index <= cutoff]
    raw_post = raw_full[raw_full.index > cutoff]

    # ------- (a) verify climatology_decomp.py's bias number ----------
    print("=" * 75)
    print("(a) per-row μ(t) - actual(t),  averaged per HE  (vs decomp's way)")
    print("=" * 75)

    for W in (1.0, 2.0, 3.0, 5.0, 10.0):
        zp = _fit_zscore_params_window(raw_pre, W)
        mu_arr = mu_at(zp, raw_post.index)
        mu_s = pd.Series(mu_arr, index=raw_post.index)

        # Decomp's way: subtract per-hour means of two separate series
        decomp_way = per_hour(mu_s) - per_hour(raw_post)
        # Per-row way: subtract first, then average
        per_row = per_hour(mu_s - raw_post)

        if not np.allclose(decomp_way.values, per_row.values, atol=1e-6):
            print(f"  W={W}y: DECOMP DOES NOT MATCH PER-ROW")
        else:
            print(f"  W={W}y: decomp-way == per-row-way   (max diff "
                  f"{(decomp_way - per_row).abs().max():.2e})")

    # Quick sanity: full-history climatology
    zp_full = zscore_params(cutoff, method=spec["predictor"].get(
        "climatology_method", "month_hour"))
    mu_arr_full = mu_at(zp_full, raw_post.index)
    mu_full_s = pd.Series(mu_arr_full, index=raw_post.index)
    decomp_full = per_hour(mu_full_s) - per_hour(raw_post)
    per_row_full = per_hour(mu_full_s - raw_post)
    print(f"  full:   decomp-way == per-row-way   (max diff "
          f"{(decomp_full - per_row_full).abs().max():.2e})")

    # ------- (b) compare the two reports of μ-bias -----------------
    print()
    print("=" * 75)
    print("(b) per-HE μ_W(t) - actual(t)   (MW; per-row mean)")
    print("=" * 75)
    print()
    rows = []
    for W in (1.0, 2.0, 3.0, 5.0, 10.0):
        zp = _fit_zscore_params_window(raw_pre, W)
        mu_arr = mu_at(zp, raw_post.index)
        mu_s = pd.Series(mu_arr, index=raw_post.index)
        col = per_hour(mu_s - raw_post)
        col.name = f"W={int(W)}y"
        rows.append(col)
    col_full = per_hour(mu_full_s - raw_post)
    col_full.name = "full"
    rows.append(col_full)
    tab = pd.concat(rows, axis=1)
    print(tab.round(0).astype(int).to_string())

    # ------- (c) rescore pickle reconciliation ---------------------
    print()
    print("=" * 75)
    print("(c) rescore W=2y pickle:  forecast = σ_mh · z_next + μ_mh")
    print("    decompose mean signed error per HE into:")
    print("      A = mean(σ_mh · z_next)        (operator output in MW)")
    print("      B = mean(actual - μ_mh)        (climatology-gap in MW)")
    print("      forecast - actual = z_next·σ - (actual - μ) = A - B")
    print("=" * 75)
    print()

    p_rescore = Path("scratch/data/rescore_W2y/cell_B_meaniter_global_W2y.pkl")
    if not p_rescore.exists():
        # try sweep cache
        p_rescore = Path("scratch/data/rescore_window_sweep/cell_B_W2y.pkl")
    fc = pd.read_pickle(p_rescore)
    print(f"  loaded {p_rescore}  rows={len(fc)}")

    # We need σ_mh(t) and μ_mh(t) at every target_dt.  Refit W=2y climatology.
    zp2 = _fit_zscore_params_window(raw_pre, 2.0)
    target_idx = pd.DatetimeIndex(fc["target_dt"].values)
    mu_t = mu_at(zp2, target_idx)
    sd_t = sigma_at(zp2, target_idx)

    fc = fc.copy()
    fc["mu_mh"]  = mu_t
    fc["sd_mh"]  = sd_t
    fc["z_next"] = (fc["our_forecast_mw"].astype(float) - mu_t) / sd_t

    fc["operator_mw"] = fc["sd_mh"] * fc["z_next"]   # forecast - μ
    fc["clim_gap"]    = fc["actual_mw"].astype(float) - fc["mu_mh"]  # actual - μ
    # forecast - actual = (μ + σ z) - actual = σ z - (actual - μ) = operator - clim_gap
    check = fc["operator_mw"] - fc["clim_gap"]
    actual_signed = fc["our_forecast_mw"].astype(float) - fc["actual_mw"].astype(float)
    diff = (check - actual_signed).abs().max()
    print(f"  sanity: operator_mw - clim_gap = forecast - actual  "
          f"(max diff {diff:.4e})")
    print()

    fc["actual_mw_f"] = fc["actual_mw"].astype(float)
    fc["our_forecast_mw_f"] = fc["our_forecast_mw"].astype(float)
    g = fc.groupby("horizon_h")
    out = pd.DataFrame({
        "A_operator_mw":   g["operator_mw"].mean(),
        "B_clim_gap_mw":   g["clim_gap"].mean(),
        "A_minus_B":       g["operator_mw"].mean() - g["clim_gap"].mean(),
        "signed_err_mw":   g["our_forecast_mw_f"].mean() - g["actual_mw_f"].mean(),
    }).round(0).astype(int)
    out.index = [f"HE{h:>2d}" for h in out.index]
    print(out.to_string())

    # Headline aggregate
    print()
    print(f"  averaged over all HE:")
    print(f"    A (operator output, σ·z_next)         = "
          f"{fc['operator_mw'].mean():+8.1f} MW")
    print(f"    B (climatology gap, actual - μ_mh)    = "
          f"{fc['clim_gap'].mean():+8.1f} MW")
    print(f"    forecast - actual (= A - B)           = "
          f"{(fc['our_forecast_mw'].astype(float) - fc['actual_mw'].astype(float)).mean():+8.1f} MW")


if __name__ == "__main__":
    main()
