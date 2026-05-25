"""EXPLORATORY: predictor-2 re-score under a recency-window climatology.

Re-runs predictor 2 (mean-iteration, global-OLS θ=0) over the full
500-day post-cutoff backtest window, but with `(mu_{m,h}, sigma_{m,h})`
refit on the last N years of pre-cutoff data only -- instead of the
full 2003--2024 climatology baked into the registered spec.

The window-length sweep (scratch/climatology_window.py) selected
W = 2 years on three of four metrics (held-out NLL, AIC, residual
diurnal-z amplitude). This script applies that window to the full
backtest and produces the same per-row pickle the existing
predictor 2 produces, in the same schema:

    cell_B_meaniter_global.pkl     (existing, full-history climatology)
    cell_B_meaniter_global_W{N}y.pkl  (this script, N-year climatology)

so that scratch/bias_diagnostics.py's signed-error-by-hour can be
run unchanged on the new pickle.

PROVENANCE-GRADE: INSPECTION-ONLY.  This script DEVIATES from the
registered spec (different climatology) and so cannot emit a
claim-grade artifact.  The change is, structurally, a candidate v2
predictor spec; landing it would require a fresh `experiment.freeze`
registration with the new climatology.

Usage:

    python -m scripts.fir.rescore_climatology \\
        --window-years 2 \\
        --out scratch/data/rescore_W2y/cell_B_meaniter_global_W2y.pkl \\
        [--max-days 10]           # smoke-test on first N days

Designed for SLURM submission on Fir.  Single-CPU; the 500-day
mean-iter loop runs in ~5--10 minutes locally and proportionally
on Fir.  No ray / parallel needed at this scale.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from edynamics.modelling_tools import Embedding, Lag

from config import load_config
from processing.innovations.estimator import _local_fit_at  # only for type-consistency

from experiment import freeze
from experiment._actuals import (
    load_actuals,
    mu_at,
    sigma_at,
    zscore_transform,
)
from experiment.predict import _build_pre_cutoff, _daytype
from experiment.backtest import _complete_delivery_days


# ---------------------------------------------------------------------------
def _fit_zscore_params_window(raw_pre: pd.Series, window_years: float) -> dict:
    """Fit a month×hour-of-day climatology on the last `window_years`
    of `raw_pre`. Returns a zscore_params dict in the same shape that
    `experiment._actuals.zscore_params(method="month_hour")` produces
    (so it can be passed straight to mu_at/sigma_at/zscore_transform).

    The schema (see experiment._actuals.zscore_params:130):
        {"method": "month_hour",
         "mu_mh":    Series indexed by (month, hour),
         "sigma_mh": Series indexed by (month, hour)}
    """
    end = raw_pre.index.max()
    start = end - pd.Timedelta(days=int(window_years * 365.25))
    win = raw_pre[(raw_pre.index > start) & (raw_pre.index <= end)].dropna()
    if len(win) < 24 * 30 * 12:    # at least roughly a year of data
        raise ValueError(
            f"window {window_years}y has too few rows ({len(win)}); "
            f"need at least one full year for all (m, h) bins.")
    df = pd.DataFrame({
        "y": win.values,
        "month": win.index.month,
        "hour":  win.index.hour,
    })
    g = df.groupby(["month", "hour"])
    mu_mh = g["y"].mean()
    sigma_mh = g["y"].std(ddof=0).where(g["y"].std(ddof=0) > 1.0, 1.0)
    return {"method": "month_hour", "mu_mh": mu_mh, "sigma_mh": sigma_mh}


# ---------------------------------------------------------------------------
def _global_ols(X: np.ndarray, Y: np.ndarray) -> tuple:
    """Plain OLS fit (θ=0, uniform weights).  Returns (C, Sigma)."""
    C, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ C
    mu_r = resid.mean(axis=0)
    rc = resid - mu_r
    Sigma = rc.T @ rc / max(len(rc) - 1, 1)
    return C, Sigma


def run_backtest(zp: dict,
                  raw_full: pd.Series,
                  cutoff: pd.Timestamp,
                  anchor_h: int,
                  dims: dict[str, int],
                  delivery_dates: pd.DatetimeIndex,
                  *,
                  log_every: int = 50) -> pd.DataFrame:
    """Predictor-2 mean-iteration over all `delivery_dates` with the
    supplied climatology `zp`.

    Mirrors scratch.backtest_simplex_theta._run(mode='global') exactly,
    except the climatology is whatever zp the caller passes in.

    Returns a long-form DataFrame in the same schema as the existing
    `cell_B_meaniter_global.pkl`.
    """
    ISSUE_HOUR_OFFSET = pd.Timedelta(hours=1)
    z_full = zscore_transform(raw_full, zp)
    # Pre-fit one global C, Σ per embedding dimension (predictor 2's
    # global fit is state-independent: per-d cached).
    lib_cache: dict[int, tuple] = {}
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

        # Recent z-history (lag window) for mean-iteration
        zhist = {
            ts: float(z_full.loc[ts])
            for ts in z_full.index
            if (issue_anchor - pd.Timedelta(hours=dmax)) <= ts <= issue_anchor
        }

        for t in targets:
            dt = _daytype(t, anchor_h)
            d = int(dims[dt])
            if d not in lib_cache:
                X, Y, _emb = _build_pre_cutoff(
                    z_full[z_full.index <= cutoff], d, cutoff)
                C, Sigma = _global_ols(X, Y)
                lib_cache[d] = (X, Y, C, Sigma)
            X, Y, C, Sigma = lib_cache[d]

            prev = t - pd.Timedelta(hours=1)
            lag_times = [prev - pd.Timedelta(hours=i) for i in range(d)]
            try:
                x_query = np.array([zhist[lt] for lt in lag_times],
                                    dtype=float)
            except KeyError:
                break
            if not np.all(np.isfinite(x_query)):
                break

            # Predictor 2: mean iteration via x_next = x @ C
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
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--window-years", type=float, required=True,
                   help="climatology window length in years (e.g. 2)")
    p.add_argument("--out", type=Path, required=True,
                   help="output pickle path")
    p.add_argument("--max-days", type=int, default=None,
                   help="smoke-test on first N delivery days only")
    args = p.parse_args()

    print(f"  rescore_climatology  W={args.window_years}y  out={args.out}",
          flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Load spec and demand series.
    # NOTE on anchor_h: the existing backtests (scratch/backtest_smc.py,
    # scratch/backtest_simplex_theta.py) read anchor_h from
    # cfg.data.day_anchor_hours (the realigned post-refactor value),
    # not from the spec.  The spec on file (anchor_h=7) predates the
    # realignment refactor.  We mirror the backtest convention so the
    # re-scored pickle is directly comparable to cell_B_meaniter_global.
    cfg = load_config()
    spec = freeze.load_verified()
    cutoff = pd.Timestamp(spec["data_cutoff"])
    anchor_h = int(cfg.data.day_anchor_hours)
    dims = spec["predictor"]["embedding_dims"]
    spec_anchor = spec["predictor"]["day_anchor_hour"]
    if spec_anchor != anchor_h:
        print(f"  note: cfg anchor_h={anchor_h} differs from spec "
              f"anchor_h={spec_anchor}.  Using cfg value (matches the "
              f"existing backtest pickle).", flush=True)

    raw_full = load_actuals(cutoff=None).dropna()
    raw_pre = raw_full[raw_full.index <= cutoff]
    print(f"  cutoff           = {cutoff}", flush=True)
    print(f"  anchor_h         = {anchor_h}", flush=True)
    print(f"  embedding_dims   = {dict(dims)}", flush=True)
    print(f"  pre-cutoff rows  = {len(raw_pre):,d}", flush=True)

    # Refit climatology
    zp = _fit_zscore_params_window(raw_pre, args.window_years)
    print(f"  climatology bins = {len(zp['mu_mh'])}", flush=True)

    # Resolve the same 500-day delivery-day population
    days = _complete_delivery_days(raw_full, anchor_h)
    days = days[days > cutoff]
    if anchor_h > 0:
        days = days[:-1]
    if args.max_days:
        days = days[:args.max_days]
    print(f"  delivery days    = {len(days)}  ({days.min().date()} .. {days.max().date()})",
          flush=True)

    # Run backtest
    fc = run_backtest(zp, raw_full, cutoff, anchor_h, dims, days)
    # Attach actuals (same way the existing pickle does)
    actual = load_actuals(cutoff=None).dropna()
    fc["actual_mw"] = fc["target_dt"].map(actual).astype("float").round(0).astype("Int64")
    fc = fc.dropna(subset=["actual_mw"])
    err = (fc["our_forecast_mw"] - fc["actual_mw"]).abs()
    pct = err / fc["actual_mw"]
    cov = ((fc["our_pi_lo_mw"] <= fc["actual_mw"])
           & (fc["actual_mw"] <= fc["our_pi_hi_mw"])).mean()
    print(f"\n  [rescore W={args.window_years}y]  "
          f"MAE = {err.mean():7.1f} MW   "
          f"MAPE = {100*pct.mean():5.2f}%   "
          f"interval coverage = {100*cov:5.1f}%   "
          f"rows = {len(fc)}",
          flush=True)
    fc.to_pickle(args.out)
    print(f"  wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
