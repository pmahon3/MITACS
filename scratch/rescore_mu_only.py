"""EXPLORATORY: predictor-2 re-score with mu-only de-mean (no sigma rescaling).

Tests the framework-level hypothesis (2026-05-25):
  - de-mean is necessary (without it AR collapses to library mean,
    confirmed by scratch.rescore_raw_no_std failing catastrophically);
  - sigma-rescaling is the suspected distortion source (the embedding
    lags then span coordinates with hour-varying MW-per-z, mixing scales
    that the WLS local fit can't reason about).

Concretely:

    y(t) = raw(t) - mu_{m,h}(t)       (in MW, not unitless)
    fit AR(d) operator on y           (same pipeline as z-version)
    predict y_next                    (in MW relative to seasonal mean)
    forecast_mw = mu_{m,h}(t) + y_next

No sigma anywhere.  Embedding coordinates are all in MW, so adjacent
lag components have the same units; the WLS local fit sees an isotropic
problem.

Mirrors scripts.fir.rescore_climatology.run_backtest exactly except for
the coordinate transform layer (mu-only, no sigma).  Same intercept-free
fit (the y-process has mean ~0 across the library, so no intercept
needed).  Outputs the same per-row pickle schema.

PROVENANCE-GRADE: INSPECTION-ONLY.

Usage:

    .venv/bin/python -m scratch.rescore_mu_only \\
        --out scratch/data/rescore_mu_only/cell_B_meaniter_global_mu_only.pkl \\
        [--max-days 10]
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
from experiment._actuals import load_actuals, mu_at, zscore_params
from experiment.predict import _daytype
from experiment.backtest import _complete_delivery_days


def _build_pre_cutoff_y(y_series: pd.Series,
                         d: int,
                         cutoff: pd.Timestamp) -> tuple:
    """Library (X, Y) on the de-meaned y-process (raw - mu_{m,h}).

    Identical to experiment.predict._build_pre_cutoff except the input
    is y (MW, mean-zero) rather than z (unitless, std-one).  Library is
    restricted to <= cutoff.
    """
    s = y_series[y_series.index <= cutoff].dropna()
    times = s.index
    vals = s.values.astype(float)
    dt = np.diff(times.view("int64"))
    ONE_HOUR = int(3600e9)
    n = len(vals)
    if n < d + 1:
        raise ValueError(f"library too small: {n} < {d+1}")
    is_hour = (dt == ONE_HOUR).astype(np.int32)
    csum = np.concatenate([[0], np.cumsum(is_hour)])
    window_sums = csum[d:] - csum[:-d]
    ok_start = window_sums == d
    valid_idx = np.where(ok_start)[0]
    X = np.stack([vals[valid_idx + d - 1 - j] for j in range(d)], axis=1)
    Y = vals[valid_idx + d].reshape(-1, 1)
    return X, Y


def _global_ols(X: np.ndarray, Y: np.ndarray) -> tuple:
    """Plain OLS fit (theta=0, uniform weights), no intercept.

    The y-process has mean ~0 by construction, so no intercept term.
    Returns (C, Sigma) where y_next = x @ C; Sigma is the mu-centred
    residual covariance.
    """
    C, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ C
    mu_r = resid.mean(axis=0)
    rc = resid - mu_r
    Sigma = rc.T @ rc / max(len(rc) - 1, 1)
    return C, Sigma


def run_backtest_mu_only(zp: dict,
                          raw_full: pd.Series,
                          cutoff: pd.Timestamp,
                          anchor_h: int,
                          dims: dict[str, int],
                          delivery_dates: pd.DatetimeIndex,
                          *,
                          log_every: int = 50) -> pd.DataFrame:
    """Predictor-2 mean-iteration on y = raw - mu_{m,h}.

    `zp` provides the mu_{m,h} only; sigma is unused.  Iteration is in
    MW (y has MW units); final forecast = mu_{m,h}(t) + y_next.
    """
    ISSUE_HOUR_OFFSET = pd.Timedelta(hours=1)
    mu_arr = mu_at(zp, raw_full.index)
    y_full = pd.Series(raw_full.values - mu_arr,
                       index=raw_full.index, name="y")
    lib_cache: dict[int, tuple] = {}
    rows: list[dict] = []
    t0 = time.time()
    for k, D in enumerate(pd.DatetimeIndex(delivery_dates)):
        D = pd.Timestamp(D.date())
        targets = [D + pd.Timedelta(hours=anchor_h + h) for h in range(24)]
        issue_anchor = targets[0] - ISSUE_HOUR_OFFSET
        dmax = max(int(v) for v in dims.values())
        hist_need = [issue_anchor - pd.Timedelta(hours=i) for i in range(dmax)]
        if not all(h in y_full.index for h in hist_need):
            continue
        if not np.all(np.isfinite(y_full.reindex(hist_need).to_numpy())):
            continue

        hist_window = y_full.loc[
            issue_anchor - pd.Timedelta(hours=dmax) : issue_anchor
        ]
        yhist = dict(zip(hist_window.index, hist_window.values.astype(float)))

        for t in targets:
            dt = _daytype(t, anchor_h)
            d = int(dims[dt])
            if d not in lib_cache:
                X, Y = _build_pre_cutoff_y(
                    y_full[y_full.index <= cutoff], d, cutoff)
                C, Sigma = _global_ols(X, Y)
                lib_cache[d] = (X, Y, C, Sigma)
            X, Y, C, Sigma = lib_cache[d]

            prev = t - pd.Timedelta(hours=1)
            lag_times = [prev - pd.Timedelta(hours=i) for i in range(d)]
            try:
                x_query = np.array([yhist[lt] for lt in lag_times], dtype=float)
            except KeyError:
                break
            if not np.all(np.isfinite(x_query)):
                break

            # Predictor 2: mean iteration via y_next = x @ C
            y_next = float(x_query @ (C[:, 0] if C.ndim == 2 else C))
            yhist[t] = y_next

            t_idx = pd.DatetimeIndex([t])
            mu_t = float(mu_at(zp, t_idx)[0])
            sd_resid = float(np.sqrt(max(Sigma[0, 0], 0.0)))
            rows.append({
                "delivery_date":   D,
                "target_dt":       t,
                "horizon_h":       int((t - targets[0])
                                       / pd.Timedelta(hours=1)) + 1,
                "our_forecast_mw": y_next + mu_t,
                "our_pi_lo_mw":   (y_next - sd_resid) + mu_t,
                "our_pi_hi_mw":   (y_next + sd_resid) + mu_t,
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

    print(f"  rescore_mu_only  (mu de-mean only, no sigma)  out={args.out}",
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
    print(f"  climatology      = {clim_method} (full-history mu only)", flush=True)
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

    fc = run_backtest_mu_only(zp, raw_full, cutoff, anchor_h, dims, days)
    actual = load_actuals(cutoff=None).dropna()
    fc["actual_mw"] = fc["target_dt"].map(actual).astype("float").round(0).astype("Int64")
    fc = fc.dropna(subset=["actual_mw"])
    err = (fc["our_forecast_mw"] - fc["actual_mw"]).abs()
    pct = err / fc["actual_mw"]
    cov = ((fc["our_pi_lo_mw"] <= fc["actual_mw"])
           & (fc["actual_mw"] <= fc["our_pi_hi_mw"])).mean()
    signed = fc["our_forecast_mw"].astype(float) - fc["actual_mw"].astype(float)
    print(f"\n  [rescore mu-only]  "
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
