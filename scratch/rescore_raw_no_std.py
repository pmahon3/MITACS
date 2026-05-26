"""EXPLORATORY: predictor-2 re-score with NO standardization.

Tests the framework-level hypothesis (2026-05-25 session): the
`raw -> z = (raw - mu_{m,h}) / sigma_{m,h}` coordinate transform
distorts the short-scale dynamics that we want to estimate, by:

  (a) leaving a non-stationary low-frequency component in z that the
      operator iterates around the wrong center (we documented this
      as the +1000 MW post-cutoff gap, [[mitacs-clim-gap-nonstationary]]);
  (b) reshaping the conditional distribution of x_{t+1} | x_t,...
      via the hour-varying sigma_{m,h}, so the local linear operator
      fit on z is not equivalent to the local linear operator fit on
      raw.

This script removes the transform entirely.  Embedding is in raw MW;
the local fit is on raw MW; the iteration produces raw MW directly.
An intercept term is added to the fit because the embedding column
means are large (~16 GW).

Mirrors scripts.fir.rescore_climatology.run_backtest exactly except for
the coordinate transform layer.  Outputs the same per-row pickle schema.

PROVENANCE-GRADE: INSPECTION-ONLY.  Deviates from registered spec.

Usage:

    .venv/bin/python -m scratch.rescore_raw_no_std \\
        --out scratch/data/rescore_raw/cell_B_meaniter_global_raw.pkl \\
        [--max-days 10]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from edynamics.modelling_tools import Embedding, Lag

from config import load_config
from experiment import freeze
from experiment._actuals import load_actuals
from experiment.predict import _daytype
from experiment.backtest import _complete_delivery_days


def _global_ols_intercept(X: np.ndarray, Y: np.ndarray) -> tuple:
    """Plain OLS with intercept (theta=0, uniform weights).

    Returns (alpha, C, Sigma) where x_next = alpha + x @ C.  Sigma is
    the residual covariance (mu-centred).
    """
    n, d = X.shape
    X_aug = np.hstack([np.ones((n, 1)), X])
    beta, *_ = np.linalg.lstsq(X_aug, Y, rcond=None)
    alpha = beta[0]              # (D,)
    C = beta[1:]                 # (d, D)
    resid = Y - X_aug @ beta
    mu_r = resid.mean(axis=0)
    rc = resid - mu_r
    Sigma = rc.T @ rc / max(len(rc) - 1, 1)
    return alpha, C, Sigma


def _build_pre_cutoff_raw(raw_series: pd.Series,
                           d: int,
                           cutoff: pd.Timestamp) -> tuple:
    """Build the (X, Y) library directly on raw MW (no z-transform).

    Mirrors experiment.predict._build_pre_cutoff but with `raw_series`
    as the input instead of z.  X rows are d-dimensional lag vectors
    in MW; Y rows are the corresponding next-step targets, all in MW.

    Library is restricted to <= cutoff for the leakage guard.
    """
    s = raw_series[raw_series.index <= cutoff].dropna()
    times = s.index
    vals = s.values.astype(float)

    # Build d-dim lag windows for every time t where t, t-1h, ..., t-(d-1)h
    # all exist (consecutive hourly).
    # X[i] = (vals[i], vals[i-1], ..., vals[i-d+1])
    # Y[i] = vals[i+1]
    # We need rows where t to t-d+1 are all present and t+1 exists.
    # Hourly continuity check: time differences exactly 1h between consecutive.
    dt = np.diff(times.view("int64"))
    # nanosec per hour
    ONE_HOUR = int(3600e9)
    # Indices i such that vals[i-d+1..i+1] are all consecutive hours.
    # Equivalently, the diff array from i-d+1 to i (d values) are all ONE_HOUR.
    n = len(vals)
    if n < d + 1:
        raise ValueError(f"library too small: {n} < {d+1}")
    # ok_start[k]: True if vals[k..k+d] is a run of d+1 consecutive hours
    # (d diffs == ONE_HOUR).
    is_hour = (dt == ONE_HOUR).astype(np.int32)
    # Use a sliding-window sum of length d on is_hour.
    csum = np.concatenate([[0], np.cumsum(is_hour)])
    window_sums = csum[d:] - csum[:-d]    # length n - d
    ok_start = window_sums == d           # rows where (k, k+1, ..., k+d) are consecutive
    # For each ok_start[k] = True: X = vals[k+d-1, k+d-2, ..., k], Y = vals[k+d]
    # The "lag-0" coordinate is vals[k+d-1]; lag-1 is vals[k+d-2]; etc.
    valid_idx = np.where(ok_start)[0]
    # X[i, j] = vals[valid_idx[i] + d - 1 - j]
    X = np.stack([vals[valid_idx + d - 1 - j] for j in range(d)], axis=1)
    Y = vals[valid_idx + d].reshape(-1, 1)
    # Sanity: target times = times[valid_idx + d]
    return X, Y


def run_backtest_raw(raw_full: pd.Series,
                      cutoff: pd.Timestamp,
                      anchor_h: int,
                      dims: dict[str, int],
                      delivery_dates: pd.DatetimeIndex,
                      *,
                      log_every: int = 50) -> pd.DataFrame:
    """Predictor-2 mean-iteration on raw MW (no z-transform).

    Returns a long-form DataFrame in the same schema as
    scripts.fir.rescore_climatology.run_backtest, with the prediction
    interval [our_pi_lo_mw, our_pi_hi_mw] derived from the residual
    std on the raw-MW scale (no destandardization).
    """
    ISSUE_HOUR_OFFSET = pd.Timedelta(hours=1)
    raw_pre = raw_full[raw_full.index <= cutoff]
    # Pre-fit one global (alpha, C, Sigma) per embedding dimension.
    lib_cache: dict[int, tuple] = {}
    rows: list[dict] = []
    t0 = time.time()
    for k, D in enumerate(pd.DatetimeIndex(delivery_dates)):
        D = pd.Timestamp(D.date())
        targets = [D + pd.Timedelta(hours=anchor_h + h) for h in range(24)]
        issue_anchor = targets[0] - ISSUE_HOUR_OFFSET
        dmax = max(int(v) for v in dims.values())
        hist_need = [issue_anchor - pd.Timedelta(hours=i) for i in range(dmax)]
        if not all(h in raw_full.index for h in hist_need):
            continue
        if not np.all(np.isfinite(raw_full.reindex(hist_need).to_numpy())):
            continue

        hist_window = raw_full.loc[
            issue_anchor - pd.Timedelta(hours=dmax) : issue_anchor
        ]
        rhist = dict(zip(hist_window.index, hist_window.values.astype(float)))

        for t in targets:
            dt = _daytype(t, anchor_h)
            d = int(dims[dt])
            if d not in lib_cache:
                X, Y = _build_pre_cutoff_raw(raw_pre, d, cutoff)
                alpha, C, Sigma = _global_ols_intercept(X, Y)
                lib_cache[d] = (X, Y, alpha, C, Sigma)
            X, Y, alpha, C, Sigma = lib_cache[d]

            prev = t - pd.Timedelta(hours=1)
            lag_times = [prev - pd.Timedelta(hours=i) for i in range(d)]
            try:
                x_query = np.array([rhist[lt] for lt in lag_times], dtype=float)
            except KeyError:
                break
            if not np.all(np.isfinite(x_query)):
                break

            # Predictor 2: mean iteration via x_next = alpha + x @ C
            raw_next = float(alpha[0] + x_query @ C[:, 0]) if C.ndim == 2 \
                       else float(alpha + x_query @ C)
            rhist[t] = raw_next

            sd_raw = float(np.sqrt(max(Sigma[0, 0], 0.0)))
            rows.append({
                "delivery_date":   D,
                "target_dt":       t,
                "horizon_h":       int((t - targets[0])
                                       / pd.Timedelta(hours=1)) + 1,
                "our_forecast_mw": raw_next,
                "our_pi_lo_mw":    raw_next - sd_raw,
                "our_pi_hi_mw":    raw_next + sd_raw,
            })

        if log_every and (k + 1) % log_every == 0:
            elapsed = time.time() - t0
            rate = (k + 1) / elapsed
            eta = (len(delivery_dates) - (k + 1)) / max(rate, 1e-9)
            print(f"  ...  day {k+1:>4d}/{len(delivery_dates)}  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s",
                  flush=True)
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True,
                   help="output pickle path")
    p.add_argument("--max-days", type=int, default=None,
                   help="smoke-test on first N delivery days only")
    args = p.parse_args()

    print(f"  rescore_raw_no_std  (no z-transform)  out={args.out}",
          flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    spec = freeze.load_verified()
    cutoff = pd.Timestamp(spec["data_cutoff"])
    anchor_h = int(cfg.data.day_anchor_hours)
    dims = spec["predictor"]["embedding_dims"]

    raw_full = load_actuals(cutoff=None).dropna()
    print(f"  cutoff           = {cutoff}", flush=True)
    print(f"  anchor_h         = {anchor_h}", flush=True)
    print(f"  embedding_dims   = {dict(dims)}", flush=True)
    print(f"  raw rows total   = {len(raw_full):,d}", flush=True)
    print(f"  raw pre-cutoff   = {len(raw_full[raw_full.index <= cutoff]):,d}",
          flush=True)

    days = _complete_delivery_days(raw_full, anchor_h)
    days = days[days > cutoff]
    if anchor_h > 0:
        days = days[:-1]
    if args.max_days:
        days = days[:args.max_days]
    print(f"  delivery days    = {len(days)}  "
          f"({days.min().date()} .. {days.max().date()})", flush=True)

    fc = run_backtest_raw(raw_full, cutoff, anchor_h, dims, days)
    actual = load_actuals(cutoff=None).dropna()
    fc["actual_mw"] = fc["target_dt"].map(actual).astype("float").round(0).astype("Int64")
    fc = fc.dropna(subset=["actual_mw"])
    err = (fc["our_forecast_mw"] - fc["actual_mw"]).abs()
    pct = err / fc["actual_mw"]
    cov = ((fc["our_pi_lo_mw"] <= fc["actual_mw"])
           & (fc["actual_mw"] <= fc["our_pi_hi_mw"])).mean()
    signed = fc["our_forecast_mw"].astype(float) - fc["actual_mw"].astype(float)
    print(f"\n  [rescore raw, no std]  "
          f"MAE = {err.mean():7.1f} MW   "
          f"MAPE = {100*pct.mean():5.2f}%   "
          f"interval coverage = {100*cov:5.1f}%   "
          f"mean_signed = {signed.mean():+7.1f} MW   "
          f"rows = {len(fc)}",
          flush=True)

    # Per-HE signed/abs error
    fc["signed"] = signed
    fc["absol"]  = err.astype(float)
    by_he = fc.groupby("horizon_h").agg(
        signed=("signed", "mean"),
        absol=("absol", "mean"),
        n=("absol", "size"),
    ).round(0).astype(int)
    by_he.index = [f"HE{h:>2d}" for h in by_he.index]
    print()
    print("per-HE signed/absolute error (raw MW):")
    print(by_he.to_string())

    fc.to_pickle(args.out)
    print(f"\n  wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
