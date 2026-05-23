"""EXPLORATORY: SMC multi-step day-ahead backtest, theta x sampling grid.

Tests the Liu-Gao pre-registered predictions (memory mitacs-liu-gao-
prediction). Compares four SMC predictors against the two mean-iteration
baselines already established:

  baselines (mean-iteration, point trajectory):
    A. production-theta (per-query LOO-CV, theta_med ~1.7)
    B. global-OLS       (theta=0 limit, w_i=1)

  SMC cells (M particles, ensemble mean = point forecast):
    C. SMC-global-Gaussian    : theta=0, draw r ~ N(0, Sigma)
    D. SMC-global-empirical   : theta=0, resample r from residuals
    E. SMC-prod-Gaussian      : per-query LOO-CV theta, draw r ~ N(0, Sigma)
    F. SMC-prod-empirical     : per-query LOO-CV theta, empirical r

Pre-registered predictions discriminate as:
  (E vs A) and (F vs A) at same theta -- Gaussian-SMC underperforms,
    empirical-SMC matches/beats. h=24 is the most-discriminating cell
    (locality lever at the daily trough).
  Interval coverage: empirical-SMC ~nominal, Gaussian-SMC undercovers.

Cost note. SMC-global is cheap (C, Sigma, resid are state-independent
constants from one global OLS fit; per-step cost is one matmul + one
RNG draw). SMC-prod is expensive: M*24 LOO-CV calls per day. Start with
the cheap cells; decide on the expensive cells from their numbers.

PROVENANCE-GRADE: INSPECTION-ONLY. The Ontario SMC numbers depend on
the SMC validation gate (scratch/smc_validation.py) which passes for
both variants on a known-ground-truth VAR(1).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from edynamics.modelling_tools import Embedding, Lag

from config import load_config
from processing.innovations.estimator import _local_fit_at

from experiment import freeze
from experiment._actuals import (
    load_actuals,
    mu_at,
    sigma_at,
    zscore_params,
    zscore_transform,
)
from experiment.predict import _build_pre_cutoff, _daytype
from experiment.backtest import _complete_delivery_days

from scratch.simplex_theta import _fit_given_theta
from scratch.smc import smc_trajectory, ensemble_summary

ISSUE_HOUR_OFFSET = pd.Timedelta(hours=1)


def _global_fit(X: np.ndarray, Y: np.ndarray) -> tuple:
    """One global-OLS fit. Returns (C, Sigma, resid) -- state-independent
    so the SMC fit_at(x) closure ignores x."""
    C = np.linalg.lstsq(X, Y, rcond=None)[0]
    resid = Y - X @ C
    mu = resid.mean(axis=0)
    rc = resid - mu[None, :]
    Sigma = rc.T @ rc / len(rc)
    return C, Sigma, resid


def _smc_day_worker(args):
    """Picklable worker: M-particle SMC for one delivery day.

    All inputs are compact and pickle-safe: no globals, no closures,
    no shared caches. Per-worker reproducibility: each day's RNG is
    seeded by base_seed + day_index, so days are independent and
    deterministic regardless of execution order.
    """
    (D, day_idx, base_seed, anchor_h, dims_by_d,
     z_full_index_i8, z_full_values, libs,
     mu_mh, sigma_mh, clim_method, fourier_params,
     cell, variant, M, fixed_theta) = args

    D = pd.Timestamp(pd.Timestamp(D).date())
    targets = [D + pd.Timedelta(hours=anchor_h + h) for h in range(24)]
    issue_anchor = targets[0] - ISSUE_HOUR_OFFSET
    dmax = max(int(v) for v in dims_by_d.values())
    z_full_idx = pd.DatetimeIndex(z_full_index_i8.view("datetime64[ns]"))
    z_full = pd.Series(z_full_values, index=z_full_idx)
    hist_need = [issue_anchor - pd.Timedelta(hours=i) for i in range(dmax)]
    if not all(h in z_full.index for h in hist_need):
        return []
    if not np.all(np.isfinite(z_full.reindex(hist_need).to_numpy())):
        return []

    dt0 = _daytype(targets[0], anchor_h)
    d = int(dims_by_d[dt0])
    X, Y = libs[d]

    lag_times = [issue_anchor - pd.Timedelta(hours=i) for i in range(d)]
    try:
        x_0 = np.array([float(z_full.loc[lt]) for lt in lag_times])
    except KeyError:
        return []

    if cell == "global":
        C_g, S_g, r_g = _global_fit(X, Y)
        fit_at = lambda _x: (C_g, S_g, r_g)
    elif cell == "fixed":
        d_ = d
        def fit_at(x):
            C, Sigma = _fit_given_theta(X, Y, x, d_, fixed_theta)
            resid = Y - X @ C
            return C, Sigma, resid
    elif cell == "prod":
        d_ = d
        def fit_at(x):
            C, Sigma, _mu, _theta, resid = _local_fit_at(X, Y, x, d_)
            return C, Sigma, resid
    else:
        raise ValueError(f"unknown cell {cell!r}")

    rng = np.random.default_rng(base_seed + day_idx)
    H = 24
    traj = smc_trajectory(x_0, H, fit_at, variant, M, rng)
    summ = ensemble_summary(traj)

    # rebuild a minimal zscore_params for the de-z-score
    if clim_method == "month_hour":
        zp = {"method": "month_hour", "mu_mh": mu_mh, "sigma_mh": sigma_mh}
    else:
        zp = {"method": "fourier", "fourier": fourier_params}

    rows = []
    for h, t in enumerate(targets):
        t_idx = pd.DatetimeIndex([t])
        mu_t = float(mu_at(zp, t_idx)[0])
        sd_t = float(sigma_at(zp, t_idx)[0])
        z_mean = float(summ["mean"][h, 0])
        z_lo = float(summ["q_lo"][h])
        z_hi = float(summ["q_hi"][h])
        rows.append({
            "delivery_date": D,
            "target_dt": t,
            "horizon_h": h + 1,
            "our_forecast_mw": z_mean * sd_t + mu_t,
            "our_pi_lo_mw": z_lo * sd_t + mu_t,
            "our_pi_hi_mw": z_hi * sd_t + mu_t,
            "daytype": dt0,
            "embedding_dim": d,
            "cell": cell,
            "variant": variant,
        })
    return rows


def _run_smc(delivery_dates, cell, variant, M, fixed_theta=None,
             seed=20260522, pool=None):
    """Run the SMC backtest across all delivery dates.

    `pool`: a ray.util.multiprocessing.Pool (or any .map-capable pool).
    None = serial. The expensive 'prod' cell (per-query LOO-CV per
    particle per horizon) is the main reason to parallelise -- per-day
    parallelism is the established pattern (cf. simplex_theta field
    build, commit 4f4c877).
    """
    spec = freeze.load_verified()
    spec_cutoff = pd.Timestamp(spec["data_cutoff"])
    anchor_h = load_config().data.day_anchor_hours
    dims = spec["predictor"]["embedding_dims"]
    clim_method = spec["predictor"].get("climatology_method", "month_hour")
    zp = zscore_params(spec_cutoff, method=clim_method)
    z_full = zscore_transform(load_actuals(cutoff=None), zp)

    # Pre-build the per-d pre-cutoff library ONCE (workers receive
    # arrays; no Embedding objects cross the IPC boundary).
    libs: dict[int, tuple] = {}
    dims_by_d = {dt: int(dims[dt]) for dt in dims}
    for d in sorted({int(v) for v in dims.values()}):
        X, Y, _emb = _build_pre_cutoff(
            z_full[z_full.index <= spec_cutoff], d, spec_cutoff
        )
        libs[d] = (X, Y)

    # zp pieces needed by mu_at/sigma_at in workers
    mu_mh = zp.get("mu_mh")
    sigma_mh = zp.get("sigma_mh")
    fourier_params = zp.get("fourier")

    # Pack z_full as int8 ns ts + values so the worker pickles cleanly.
    z_idx_i8 = np.array(z_full.index.asi8, copy=False)
    z_vals = z_full.to_numpy()

    days = pd.DatetimeIndex(delivery_dates)
    tasks = [
        (D, i, seed, anchor_h, dims_by_d,
         z_idx_i8, z_vals, libs, mu_mh, sigma_mh, clim_method,
         fourier_params, cell, variant, M, fixed_theta)
        for i, D in enumerate(days)
    ]

    all_rows = []
    if pool is None:
        for i, t in enumerate(tasks):
            all_rows.extend(_smc_day_worker(t))
            if (i + 1) % 50 == 0:
                print(f"  ... {i+1}/{len(days)} days", flush=True)
    else:
        done = 0
        for rows in pool.imap_unordered(_smc_day_worker, tasks):
            all_rows.extend(rows)
            done += 1
            if done % 50 == 0:
                print(f"  ... {done}/{len(days)} days", flush=True)
    return pd.DataFrame(all_rows)


def _mae(pred, act):
    e = (pred - act).abs().dropna()
    return float(e.mean()) if len(e) else np.nan


def _mape(pred, act):
    m = act != 0
    e = ((pred[m] - act[m]).abs() / act[m]).dropna()
    return float(e.mean() * 100) if len(e) else np.nan


def _metrics(fc, actual, label, save_dir=None, cell_tag=None):
    """Compute aggregate + per-horizon metrics for a single cell.

    `save_dir` + `cell_tag`: if both set, the joined per-row frame is
    written to {save_dir}/cell_{cell_tag}.parquet (with row columns
    target_dt, delivery_date, horizon_h, daytype, our_forecast_mw,
    our_pi_lo_mw, our_pi_hi_mw, actual_mw). The on-disk frames support
    paired-day-bootstrap hypothesis testing of the Liu-Gao predictions
    via scratch/bootstrap_smc.py (the per-row data is what summaries
    discard; saving it is what enables 95% CIs on MAE differences).
    """
    fc = fc.copy()
    fc["actual_mw"] = fc["target_dt"].map(actual)
    fc = fc.dropna(subset=["actual_mw"])
    mae = _mae(fc["our_forecast_mw"], fc["actual_mw"])
    mape = _mape(fc["our_forecast_mw"], fc["actual_mw"])
    # interval coverage (Liu-Gao prediction 4)
    inside = (
        (fc["actual_mw"] >= fc["our_pi_lo_mw"])
        & (fc["actual_mw"] <= fc["our_pi_hi_mw"])
    )
    cov_pct = float(inside.mean() * 100)
    # analytical std error for coverage (binomial, n hourly forecasts)
    n = len(fc)
    cov_se = float(100 * np.sqrt((cov_pct / 100) * (1 - cov_pct / 100) / n))
    per_h = {}
    for h, g in fc.groupby("horizon_h"):
        per_h[int(h)] = {
            "mae": _mae(g["our_forecast_mw"], g["actual_mw"]),
            "mape": _mape(g["our_forecast_mw"], g["actual_mw"]),
            "n": int(len(g)),
        }
    print(f"  [{label:>22}]  MAE = {mae:7.1f} MW   MAPE = {mape:5.2f}%   "
          f"interval coverage = {cov_pct:5.1f}% (±{cov_se:.2f})   "
          f"rows={n}")
    if save_dir is not None and cell_tag is not None:
        path = Path(save_dir) / f"cell_{cell_tag}.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        # core columns the bootstrap script needs. Optional metadata
        # columns (daytype, embedding_dim, theta_star, etc.) are
        # included when present but not required (mean-iter cells
        # produce a thinner frame than the SMC cells). Pickle rather
        # than parquet to avoid the pyarrow dependency.
        core = ["target_dt", "delivery_date", "horizon_h",
                "our_forecast_mw", "our_pi_lo_mw", "our_pi_hi_mw",
                "actual_mw"]
        optional = ["daytype", "embedding_dim", "theta_star",
                    "cell", "variant"]
        cols = core + [c for c in optional if c in fc.columns]
        fc[cols].to_pickle(path)
    return {"mae": mae, "mape": mape, "per_h": per_h,
            "cov_pct": cov_pct, "cov_se": cov_se, "fc": fc}


def _per_horizon_delta(prod, other, label):
    print(f"\n  PER-HORIZON dMAE ({label} - production):")
    print(f"    {'h':>3} {'prod_MAE':>9} {'oth_MAE':>9} {'dMAE':>8} "
          f"{'dMAPE':>8}")
    for h in range(1, 25):
        p = prod["per_h"].get(h)
        s = other["per_h"].get(h)
        if p is None or s is None:
            continue
        print(f"    {h:>3} {p['mae']:>9.1f} {s['mae']:>9.1f} "
              f"{s['mae'] - p['mae']:>+8.1f} "
              f"{s['mape'] - p['mape']:>+8.2f}")


def main_grid(M: int = 200, max_days: int | None = None, cheap_only=False,
              parallel: bool = True, save_dir: str | None = None):
    """Run the SMC backtest grid.

    cheap_only=True runs only the global cells (state-independent fits,
    fast); the expensive prod cells are skipped. Useful for a first look.

    parallel=True wraps the per-day SMC in a ray Pool -- the established
    pattern (cf. simplex_theta field build, commit 4f4c877). Required to
    make the prod cells (M*24 LOO-CV/day) tractable.
    """
    from scratch.backtest_simplex_theta import _run as _mean_run

    actual = load_actuals(cutoff=None)
    cutoff = pd.Timestamp("2024-12-31T23:00:00")
    anchor_h = load_config().data.day_anchor_hours
    days = _complete_delivery_days(actual, anchor_h)
    days = days[days > cutoff]
    if anchor_h > 0:
        days = days[:-1]
    if max_days:
        days = days[:max_days]

    print("=" * 70)
    print(f"SMC backtest grid  (INSPECTION-ONLY)  M={M}  "
          f"anchor_h={anchor_h}  parallel={parallel}")
    print(f"post-cutoff delivery days: {len(days)}  "
          f"({days.min().date()}..{days.max().date()})")
    print("=" * 70)

    pool = None
    if parallel:
        import multiprocessing
        import ray
        from ray.util.multiprocessing import Pool
        if not ray.is_initialized():
            ray.init(log_to_driver=False)
        pool = Pool(multiprocessing.cpu_count())
        print(f"ray pool: {multiprocessing.cpu_count()} workers")

    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        print(f"per-row arrays will be saved to {save_dir}/cell_*.parquet")

    print("\nA. mean-iteration baselines:")
    prod = _metrics(_mean_run(days, mode="production"), actual,
                    "A. mean-iter production", save_dir, "A_meaniter_prod")
    glob_mi = _metrics(_mean_run(days, mode="global"), actual,
                       "B. mean-iter global-OLS", save_dir,
                       "B_meaniter_global")

    print("\nC. SMC-global-Gaussian:", flush=True)
    smc_gG = _metrics(
        _run_smc(days, cell="global", variant="gaussian", M=M, pool=pool),
        actual, "C. SMC-global Gaussian", save_dir, "C_smc_global_gauss")

    print("\nD. SMC-global-empirical:", flush=True)
    smc_gE = _metrics(
        _run_smc(days, cell="global", variant="empirical", M=M, pool=pool),
        actual, "D. SMC-global empirical", save_dir, "D_smc_global_emp")

    if not cheap_only:
        print("\nE. SMC-prod-Gaussian (expensive: M*24 LOO-CV/day) ...",
              flush=True)
        smc_pG = _metrics(
            _run_smc(days, cell="prod", variant="gaussian", M=M, pool=pool),
            actual, "E. SMC-prod Gaussian", save_dir, "E_smc_prod_gauss")
        print("\nF. SMC-prod-empirical:", flush=True)
        smc_pE = _metrics(
            _run_smc(days, cell="prod", variant="empirical", M=M, pool=pool),
            actual, "F. SMC-prod empirical", save_dir, "F_smc_prod_emp")

    # per-horizon deltas vs production baseline (A)
    print("\n" + "-" * 70)
    print("LIU-GAO predictions check (memory mitacs-liu-gao-prediction):")
    print(f"  P1  SMC-Gaussian   should UNDERPERFORM mean-iter at same theta")
    print(f"  P2  SMC-empirical  should MATCH OR BEAT mean-iter at same theta")
    print(f"  P3  empirical-SMC  LESS theta-sensitive than Gaussian-SMC")
    print(f"  P4  empirical-SMC  interval coverage ~ nominal (~68%);")
    print(f"      Gaussian-SMC   should undercover")
    print("-" * 70)

    _per_horizon_delta(prod, glob_mi, "B mean-iter global-OLS")
    _per_horizon_delta(prod, smc_gG, "C SMC-global Gaussian")
    _per_horizon_delta(prod, smc_gE, "D SMC-global empirical")
    if not cheap_only:
        _per_horizon_delta(prod, smc_pG, "E SMC-prod Gaussian")
        _per_horizon_delta(prod, smc_pE, "F SMC-prod empirical")

    print("=" * 70)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--M", type=int, default=200,
                    help="number of SMC particles (default 200)")
    ap.add_argument("--max-days", type=int, default=None,
                    help="cap delivery days (quick mechanics check)")
    ap.add_argument("--cheap-only", action="store_true",
                    help="run only the global (state-independent) SMC "
                         "cells; skip the expensive prod cells")
    ap.add_argument("--save-dir", type=str, default=None,
                    help="if set, save each cell's per-row joined "
                         "frame to {save_dir}/cell_*.parquet for "
                         "paired-day-bootstrap analysis")
    args = ap.parse_args()
    main_grid(M=args.M, max_days=args.max_days,
              cheap_only=args.cheap_only,
              save_dir=args.save_dir)
