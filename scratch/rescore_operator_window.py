"""EXPLORATORY: predictor-2 re-score with the OPERATOR fit on a recent window.

Analogous to scripts/fir/rescore_climatology.py (W=Ny climatology refit)
but applies the recency window to the OPERATOR LIBRARY instead of the
climatology.  Tests the framework-level finding from 2026-05-25:

  - the per-HE bias shape (mid-day over-forecast / late-night under-
    forecast) is NOT a coordinate-transform artifact (Option 3 mu-only
    showed the same shape);
  - hypothesis: the operator's AR coupling is mis-calibrated, learned
    from a 2003-2024 library whose post-2020 dynamics differ from
    pre-2020;
  - if true, refitting (alpha, C, Sigma) on last W years should change
    the per-HE bias shape, not just its level.

Design: hold climatology at the registered spec (full-history,
month_hour).  z = (raw - mu_full) / sigma_full.  Library for the
operator fit = embedding rows whose anchor timestamps are within the
last W years pre-cutoff.  Iteration identical to predictor 2.

PROVENANCE-GRADE: INSPECTION-ONLY.

Usage:

    .venv/bin/python -m scratch.rescore_operator_window \\
        --library-years 2 \\
        --out scratch/data/rescore_op_W2y/cell_B_meaniter_global_opW2y.pkl
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import load_config
from experiment import freeze
from experiment._actuals import (
    load_actuals,
    mu_at,
    sigma_at,
    zscore_params,
    zscore_transform,
)
from experiment.predict import _daytype
from experiment.backtest import _complete_delivery_days


def _build_pre_cutoff_window(z_series: pd.Series,
                              d: int,
                              cutoff: pd.Timestamp,
                              library_years: float) -> tuple:
    """Library (X, Y) on the z-process, restricted to the last
    `library_years` years pre-cutoff.

    The X-rows' anchor timestamp (the last lag in the embedding, i.e. t)
    must be within `(cutoff - library_years*365.25d, cutoff]`; Y must
    exist (so t+1h <= cutoff).
    """
    s = z_series[z_series.index <= cutoff].dropna()
    times = s.index
    vals = s.values.astype(float)
    dt = np.diff(times.view("int64"))
    ONE_HOUR = int(3600e9)
    n = len(vals)
    if n < d + 1:
        raise ValueError(f"series too small: {n} < {d+1}")
    is_hour = (dt == ONE_HOUR).astype(np.int32)
    csum = np.concatenate([[0], np.cumsum(is_hour)])
    window_sums = csum[d:] - csum[:-d]
    ok_start = window_sums == d
    valid_idx = np.where(ok_start)[0]
    # Anchor timestamp of each library row = times[valid_idx + d - 1]
    anchor_times = times[valid_idx + d - 1]
    window_start = cutoff - pd.Timedelta(days=int(library_years * 365.25))
    mask = (anchor_times > window_start) & (anchor_times <= cutoff)
    valid_idx = valid_idx[mask]
    if len(valid_idx) < 100:
        raise ValueError(f"library too small after window restriction: "
                          f"{len(valid_idx)} rows for W={library_years}y, d={d}")
    X = np.stack([vals[valid_idx + d - 1 - j] for j in range(d)], axis=1)
    Y = vals[valid_idx + d].reshape(-1, 1)
    return X, Y


def _global_ols(X: np.ndarray, Y: np.ndarray) -> tuple:
    """Plain OLS (theta=0, uniform weights).  Returns (C, Sigma)."""
    C, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ C
    mu_r = resid.mean(axis=0)
    rc = resid - mu_r
    Sigma = rc.T @ rc / max(len(rc) - 1, 1)
    return C, Sigma


def run_backtest_op_window(zp: dict,
                            raw_full: pd.Series,
                            cutoff: pd.Timestamp,
                            anchor_h: int,
                            dims: dict[str, int],
                            delivery_dates: pd.DatetimeIndex,
                            library_years: float,
                            *,
                            log_every: int = 50) -> pd.DataFrame:
    """Predictor-2 mean-iteration with operator fit restricted to the
    last `library_years` years of pre-cutoff data."""
    ISSUE_HOUR_OFFSET = pd.Timedelta(hours=1)
    z_full = zscore_transform(raw_full, zp)
    lib_cache: dict[int, tuple] = {}
    lib_sizes: dict[int, int] = {}
    rows: list[dict] = []
    t0 = time.time()
    for k, D in enumerate(pd.DatetimeIndex(delivery_dates)):
        D = pd.Timestamp(D.date())
        targets = [D + pd.Timedelta(hours=anchor_h + h) for h in range(24)]
        issue_anchor = targets[0] - ISSUE_HOUR_OFFSET
        dmax = max(int(v) for v in dims.values())
        hist_need = [issue_anchor - pd.Timedelta(hours=i) for i in range(dmax)]
        if not all(h in z_full.index for h in hist_need):
            continue
        if not np.all(np.isfinite(z_full.reindex(hist_need).to_numpy())):
            continue

        hist_window = z_full.loc[
            issue_anchor - pd.Timedelta(hours=dmax) : issue_anchor
        ]
        zhist = dict(zip(hist_window.index, hist_window.values.astype(float)))

        for t in targets:
            dt = _daytype(t, anchor_h)
            d = int(dims[dt])
            if d not in lib_cache:
                X, Y = _build_pre_cutoff_window(
                    z_full[z_full.index <= cutoff], d, cutoff, library_years)
                C, Sigma = _global_ols(X, Y)
                lib_cache[d] = (X, Y, C, Sigma)
                lib_sizes[d] = len(X)
            X, Y, C, Sigma = lib_cache[d]

            prev = t - pd.Timedelta(hours=1)
            lag_times = [prev - pd.Timedelta(hours=i) for i in range(d)]
            try:
                x_query = np.array([zhist[lt] for lt in lag_times], dtype=float)
            except KeyError:
                break
            if not np.all(np.isfinite(x_query)):
                break

            z_next = float(x_query @ (C[:, 0] if C.ndim == 2 else C))
            zhist[t] = z_next

            t_idx = pd.DatetimeIndex([t])
            mu_t = float(mu_at(zp, t_idx)[0])
            sd_t = float(sigma_at(zp, t_idx)[0])
            sd_z = float(np.sqrt(max(Sigma[0, 0], 0.0)))
            rows.append({
                "delivery_date":   D,
                "target_dt":       t,
                "horizon_h":       int((t - targets[0])
                                       / pd.Timedelta(hours=1)) + 1,
                "our_forecast_mw": z_next * sd_t + mu_t,
                "our_pi_lo_mw":   (z_next - sd_z) * sd_t + mu_t,
                "our_pi_hi_mw":   (z_next + sd_z) * sd_t + mu_t,
            })

        if log_every and (k + 1) % log_every == 0:
            elapsed = time.time() - t0
            rate = (k + 1) / elapsed
            eta = (len(delivery_dates) - (k + 1)) / max(rate, 1e-9)
            print(f"  ...  day {k+1:>4d}/{len(delivery_dates)}  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s",
                  flush=True)

    print(f"  library sizes per d: {lib_sizes}", flush=True)
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library-years", type=float, required=True,
                   help="operator library window in years (e.g. 2)")
    p.add_argument("--out", type=Path, required=True,
                   help="output pickle path")
    p.add_argument("--max-days", type=int, default=None)
    args = p.parse_args()

    print(f"  rescore_operator_window  W_op={args.library_years}y  out={args.out}",
          flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    spec = freeze.load_verified()
    cutoff = pd.Timestamp(spec["data_cutoff"])
    anchor_h = int(cfg.data.day_anchor_hours)
    dims = spec["predictor"]["embedding_dims"]
    clim_method = spec["predictor"].get("climatology_method", "month_hour")
    zp = zscore_params(cutoff, method=clim_method,
                        k_year=spec["predictor"].get("fourier_k_year"),
                        k_day=spec["predictor"].get("fourier_k_day"))

    raw_full = load_actuals(cutoff=None).dropna()
    print(f"  cutoff           = {cutoff}", flush=True)
    print(f"  anchor_h         = {anchor_h}", flush=True)
    print(f"  embedding_dims   = {dict(dims)}", flush=True)
    print(f"  climatology      = {clim_method} (full-history, registered spec)", flush=True)
    print(f"  operator window  = last {args.library_years} years pre-cutoff",
          flush=True)

    days = _complete_delivery_days(raw_full, anchor_h)
    days = days[days > cutoff]
    if anchor_h > 0:
        days = days[:-1]
    if args.max_days:
        days = days[:args.max_days]
    print(f"  delivery days    = {len(days)}  "
          f"({days.min().date()} .. {days.max().date()})", flush=True)

    fc = run_backtest_op_window(zp, raw_full, cutoff, anchor_h, dims, days,
                                  args.library_years)
    actual = load_actuals(cutoff=None).dropna()
    fc["actual_mw"] = fc["target_dt"].map(actual).astype("float").round(0).astype("Int64")
    fc = fc.dropna(subset=["actual_mw"])
    err = (fc["our_forecast_mw"] - fc["actual_mw"]).abs()
    pct = err / fc["actual_mw"]
    cov = ((fc["our_pi_lo_mw"] <= fc["actual_mw"])
           & (fc["actual_mw"] <= fc["our_pi_hi_mw"])).mean()
    signed = fc["our_forecast_mw"].astype(float) - fc["actual_mw"].astype(float)
    print(f"\n  [rescore op-window W={args.library_years}y]  "
          f"MAE = {err.mean():7.1f} MW   "
          f"MAPE = {100*pct.mean():5.2f}%   "
          f"interval coverage = {100*cov:5.1f}%   "
          f"mean_signed = {signed.mean():+7.1f} MW   "
          f"rows = {len(fc)}",
          flush=True)

    fc["signed"] = signed
    fc["absol"]  = err.astype(float)
    by_he = fc.groupby("horizon_h").agg(
        signed=("signed", "mean"),
        absol=("absol", "mean"),
        n=("absol", "size"),
    ).round(0).astype(int)
    by_he.index = [f"HE{h:>2d}" for h in by_he.index]
    print()
    print("per-HE signed/absolute error (MW):")
    print(by_he.to_string())

    fc.to_pickle(args.out)
    print(f"\n  wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
