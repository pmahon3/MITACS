"""EXPLORATORY: day-ahead backtest with the simplex-interpolated theta field.

A side-by-side of two predictors on the SAME post-cutoff delivery days:

  production : theta re-selected per query by `_theta_loo_cv`
               (this is experiment/predict_multistep.day_ahead verbatim)
  simplex    : theta read from the frozen theta-field via the (d+1)-simplex
               (scratch/simplex_theta.fit_at_simplex_theta)

Everything else is identical -- same frozen spec, same <=cutoff library,
same z-score climatology, same iterated-24h structure, same leakage
guard. The ONLY difference is where theta comes from, so the MAE/MAPE
delta isolates the simplex-theta strategy.

Context. The gate probe (scratch/diagnose_theta_fibre.py, repaired) found
the per-anchor theta field statistically indistinguishable from a
constant field (Gate-1 ratios <= 1.17 vs a d-matched VAR(d) null; Gate-2
permutation-null z ~ 0). The prediction this makes is explicit: the
simplex backtest should land on top of production, because interpolating
a ~constant field returns ~the same theta everywhere. This probe is the
direct head-to-head test of that prediction.

PROVENANCE-GRADE: INSPECTION-ONLY -- exploratory predictor variant;
NOT registered, NOT claim-grade, MUST NOT be cited as a result. The
production backtest number (786 MW, CLAIM-grade artifact e405a121...)
is unaffected and remains the authoritative figure.
"""
from __future__ import annotations

import pickle
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

from scratch.simplex_theta import (
    build_theta_field,
    fit_at_fixed_theta,
    fit_at_simplex_theta,
)

ISSUE_HOUR_OFFSET = pd.Timedelta(hours=1)
# theta-field is expensive (one production LOO-CV per library point);
# cache per (dimension, cutoff, climatology) so re-runs are instant.
FIELD_CACHE = Path("/tmp/simplex_theta_fields.pkl")


def _run(delivery_dates, mode, theta_fields=None, fixed_theta=None):
    """Iterated 24h day-ahead forecast for each date.

    mode='production' : _local_fit_at (theta via _theta_loo_cv per query)
    mode='simplex'    : fit_at_simplex_theta (theta from the frozen field)
    mode='fixed'      : fit_at_fixed_theta (a single constant `fixed_theta`)
    """
    spec = freeze.load_verified()
    cutoff = pd.Timestamp(spec["data_cutoff"])
    anchor_h = spec["predictor"]["day_anchor_hour"]
    dims = spec["predictor"]["embedding_dims"]
    clim_method = spec["predictor"].get("climatology_method", "month_hour")
    zp = zscore_params(cutoff, method=clim_method)

    raw_full = load_actuals(cutoff=None)
    z_full = zscore_transform(raw_full, zp)

    lib_cache: dict[int, tuple] = {}
    rows = []

    for D in pd.DatetimeIndex(delivery_dates):
        D = pd.Timestamp(D.date())
        targets = [D + pd.Timedelta(hours=h) for h in range(24)]
        issue_anchor = targets[0] - ISSUE_HOUR_OFFSET
        dmax = max(int(v) for v in dims.values())
        hist_need = [issue_anchor - pd.Timedelta(hours=i) for i in range(dmax)]
        if not all(h in z_full.index for h in hist_need):
            continue
        if not np.all(np.isfinite(z_full.reindex(hist_need).to_numpy())):
            continue

        zhist = {
            ts: float(z_full.loc[ts])
            for ts in z_full.index
            if (issue_anchor - pd.Timedelta(hours=dmax)) <= ts <= issue_anchor
        }

        for t in targets:
            dt = _daytype(t, anchor_h)
            d = int(dims[dt])
            if d not in lib_cache:
                lib_cache[d] = _build_pre_cutoff(
                    z_full[z_full.index <= cutoff], d, cutoff)
            X, Y, _emb = lib_cache[d]

            prev = t - pd.Timedelta(hours=1)
            lag_times = [prev - pd.Timedelta(hours=i) for i in range(d)]
            try:
                x_query = np.array([zhist[lt] for lt in lag_times],
                                   dtype=float)
            except KeyError:
                break
            if not np.all(np.isfinite(x_query)):
                break

            if mode == "production":
                C, Sigma, _mu, theta, _ = _local_fit_at(X, Y, x_query, d)
            elif mode == "simplex":
                C, Sigma, theta = fit_at_simplex_theta(
                    theta_fields[d], x_query)
            elif mode == "fixed":
                C, Sigma, theta = fit_at_fixed_theta(
                    X, Y, x_query, d, fixed_theta)
            else:
                raise ValueError(f"unknown mode {mode!r}")

            z_next = float(x_query @ (C[:, 0] if C.ndim == 2 else C))
            zhist[t] = z_next

            t_idx = pd.DatetimeIndex([t])
            mu_t = float(mu_at(zp, t_idx)[0])
            sd_t = float(sigma_at(zp, t_idx)[0])
            sd_z = float(np.sqrt(max(Sigma[0, 0], 0.0)))
            rows.append({
                "delivery_date": D,
                "target_dt": t,
                "horizon_h": int((t - targets[0]) / pd.Timedelta(hours=1)) + 1,
                "our_forecast_mw": z_next * sd_t + mu_t,
                "our_pi_lo_mw": (z_next - sd_z) * sd_t + mu_t,
                "our_pi_hi_mw": (z_next + sd_z) * sd_t + mu_t,
                "theta": theta,
            })
    return pd.DataFrame(rows)


def _pre_cutoff_embedding(z_pre, d):
    """Embedding over pre-cutoff z-scores with d lags -- the field's
    anchor set is this embedding's library (same construction as
    experiment.predict._build_pre_cutoff, but the Embedding is kept so
    its anchor timestamps drive the production theta* lookup)."""
    df = z_pre.to_frame("zscore").asfreq("h")
    lags = [Lag(variable_name="zscore", tau=-i) for i in range(d)]
    lib = df.index[(df.index <= df.index.max())][d:-1]
    emb = Embedding(data=df, observers=lags, library_times=lib)
    emb.compile()
    return emb


def _build_fields(dims, anchor_h):
    """One frozen theta-field per embedding dimension. Each field's theta*
    is the PRODUCTION theta* (read from local_drift_and_diffusion) at every
    pre-cutoff library point -- disk-cached, since that is one LOO-CV per
    point."""
    spec = freeze.load_verified()
    cutoff = pd.Timestamp(spec["data_cutoff"])
    clim_method = spec["predictor"].get("climatology_method", "month_hour")
    cache_key = (clim_method, cutoff.isoformat(),
                 tuple(sorted({int(v) for v in dims.values()})))
    if FIELD_CACHE.exists():
        cached = pickle.loads(FIELD_CACHE.read_bytes())
        if cached.get("key") == cache_key:
            print(f"  loaded cached theta-field(s) from {FIELD_CACHE}")
            return cached["fields"]

    zp = zscore_params(cutoff, method=clim_method)
    z_full = zscore_transform(load_actuals(cutoff=None), zp)
    z_pre = z_full[z_full.index <= cutoff]

    # ray pool: the per-anchor LOO-CV is embarrassingly parallel
    import multiprocessing

    import ray
    from ray.util.multiprocessing import Pool
    if not ray.is_initialized():
        ray.init(log_to_driver=False)
    pool = Pool(multiprocessing.cpu_count())
    print(f"  ray pool: {multiprocessing.cpu_count()} workers")

    fields = {}
    for d in sorted({int(v) for v in dims.values()}):
        emb = _pre_cutoff_embedding(z_pre, d)
        anchors = emb.block.index
        print(f"  building theta-field d={d}: {len(anchors)} library "
              f"points (production LOO-CV each, parallel) ...", flush=True)
        fields[d] = build_theta_field(
            embedding=emb, anchors=anchors, day_anchor_hour=anchor_h,
            pool=pool)
    FIELD_CACHE.write_bytes(
        pickle.dumps({"key": cache_key, "fields": fields}))
    print(f"  cached theta-field(s) -> {FIELD_CACHE}")
    return fields


def _mae(pred, act):
    e = (pred - act).abs().dropna()
    return float(e.mean()) if len(e) else np.nan


def _mape(pred, act):
    m = act != 0
    e = ((pred[m] - act[m]).abs() / act[m]).dropna()
    return float(e.mean() * 100) if len(e) else np.nan


def _metrics(fc, actual, label):
    """Overall AND per-horizon MAE/MAPE.

    The per-horizon view is the load-bearing one for the simplex-theta
    question: theta-localisation could be inert one-step (queries sit in
    the data-dense core) yet matter multi-step, because the iterated
    24-step trajectory can drift to the sparse manifold fringe where the
    bandwidth is decisive. An overall delta would average that away --
    so the delta is reported per horizon h=1..24.
    """
    fc = fc.copy()
    fc["actual_mw"] = fc["target_dt"].map(actual)
    fc = fc.dropna(subset=["actual_mw"])
    mae = _mae(fc["our_forecast_mw"], fc["actual_mw"])
    mape = _mape(fc["our_forecast_mw"], fc["actual_mw"])
    per_h = {}
    for h, g in fc.groupby("horizon_h"):
        per_h[int(h)] = {
            "mae": _mae(g["our_forecast_mw"], g["actual_mw"]),
            "mape": _mape(g["our_forecast_mw"], g["actual_mw"]),
            "theta_med": float(g["theta"].median()),
            "n": int(len(g)),
        }
    print(f"  [{label:>10}]  overall MAE = {mae:7.1f} MW   "
          f"MAPE = {mape:5.2f}%   rows={len(fc)}   "
          f"theta[min/med/max]={fc['theta'].min():.3f}/"
          f"{fc['theta'].median():.3f}/{fc['theta'].max():.3f}")
    return {"mae": mae, "mape": mape, "per_h": per_h, "fc": fc}


def _per_horizon_table(prod, other, other_label):
    """Print the per-horizon dMAE/dMAPE/theta table for `other` vs prod."""
    print(f"\n  PER-HORIZON delta ({other_label} - production):")
    print(f"    {'h':>3} {'prod_MAE':>9} {'oth_MAE':>9} {'dMAE':>8} "
          f"{'dMAPE':>8} {'th_prod':>8} {'th_oth':>8}")
    for h in range(1, 25):
        p = prod["per_h"].get(h)
        s = other["per_h"].get(h)
        if p is None or s is None:
            continue
        print(f"    {h:>3} {p['mae']:>9.1f} {s['mae']:>9.1f} "
              f"{s['mae'] - p['mae']:>+8.1f} "
              f"{s['mape'] - p['mape']:>+8.2f} "
              f"{p['theta_med']:>8.3f} {s['theta_med']:>8.3f}")


def fixed_theta_test(max_days=None,
                     thetas=(1.7, 3.0, 4.2, 6.0)):
    """FALSIFICATION CONTROL for the simplex-theta multi-step gain.

    The per-horizon backtest showed the simplex predictor beats production
    by up to ~29 MW at mid horizons -- while running at a uniformly higher
    theta (median ~4.2 vs ~1.7). Hypothesis: the gain is a bandwidth-LEVEL
    effect (a wider, more global theta is more robust once the iterated
    trajectory has accumulated error), NOT a localisation or field effect.

    Test: run the iterated day-ahead backtest with theta pinned to a single
    constant -- no field, no per-query selection. If a fixed theta ~4
    recovers the simplex predictor's per-horizon gain, the simplex/field
    machinery is an unnecessary route to 'use a wider theta multi-step'.

    PROVENANCE-GRADE: INSPECTION-ONLY.
    """
    actual = load_actuals(cutoff=None)
    cutoff = pd.Timestamp("2024-12-31T23:00:00")
    days = _complete_delivery_days(actual)
    days = days[days > cutoff]
    if max_days:
        days = days[:max_days]

    print("=" * 70)
    print("FIXED-THETA falsification control  (INSPECTION-ONLY)")
    print(f"post-cutoff delivery days: {len(days)}  "
          f"({days.min().date()}..{days.max().date()})")
    print(f"fixed theta grid: {list(thetas)}")
    print("=" * 70)

    print("\nrunning backtests:")
    prod = _metrics(_run(days, mode="production"), actual, "production")
    for th in thetas:
        fc = _run(days, mode="fixed", fixed_theta=th)
        m = _metrics(fc, actual, f"fixed={th}")
        print(f"  -> overall dMAE vs production: "
              f"{m['mae'] - prod['mae']:+.1f} MW")

    # full per-horizon table for the theta closest to the simplex median
    th_match = min(thetas, key=lambda t: abs(t - 4.2))
    fc_match = _run(days, mode="fixed", fixed_theta=th_match)
    m_match = _metrics(fc_match, actual, f"fixed={th_match}")
    _per_horizon_table(prod, m_match, f"fixed theta={th_match}")

    print("\n  READING (no verdict baked in):")
    print("    fixed theta~4 RECOVERS the simplex per-horizon gain")
    print("       -> the gain is a bandwidth-LEVEL effect; the simplex /")
    print("          field machinery is unnecessary. Clean spec change:")
    print("          'multi-step forecasting uses a wider/global theta'.")
    print("    fixed theta does NOT recover it")
    print("       -> the simplex scheme does something beyond a level")
    print("          shift; investigate what.")
    print("=" * 70)


def main(max_days=None):
    cfg = load_config()
    actual = load_actuals(cutoff=None)
    cutoff = pd.Timestamp("2024-12-31T23:00:00")
    days = _complete_delivery_days(actual)
    days = days[days > cutoff]
    if max_days:
        days = days[:max_days]

    print("=" * 70)
    print("SIMPLEX-THETA day-ahead backtest  (INSPECTION-ONLY)")
    print(f"post-cutoff delivery days: {len(days)}  "
          f"({days.min().date()}..{days.max().date()})")
    print("=" * 70)

    spec = freeze.load_verified()
    dims = spec["predictor"]["embedding_dims"]
    anchor_h = spec["predictor"]["day_anchor_hour"]

    print("\nfreezing theta-field(s):")
    fields = _build_fields(dims, anchor_h)

    print("\nrunning backtests:")
    fc_prod = _run(days, mode="production")
    prod = _metrics(fc_prod, actual, "production")

    fc_simp = _run(days, mode="simplex", theta_fields=fields)
    simp = _metrics(fc_simp, actual, "simplex")

    print("\n" + "-" * 70)
    print(f"  OVERALL delta (simplex - production):  "
          f"MAE {simp['mae'] - prod['mae']:+.1f} MW   "
          f"MAPE {simp['mape'] - prod['mape']:+.2f} pp")

    # Per-horizon delta -- the load-bearing view. theta-localisation
    # could be inert one-step but bite multi-step as the iterated
    # trajectory drifts to the sparse manifold fringe; a horizon-growing
    # delta is the signature of that. A flat delta across all 24 falsifies
    # it and makes the uniform-regularity finding stronger.
    print("\n  PER-HORIZON delta (h == hour-of-day in a day-ahead forecast):")
    print(f"    {'h':>3} {'prod_MAE':>9} {'simp_MAE':>9} {'dMAE':>8} "
          f"{'dMAPE':>8} {'th_prod':>8} {'th_simp':>8}")
    for h in range(1, 25):
        p = prod["per_h"].get(h)
        s = simp["per_h"].get(h)
        if p is None or s is None:
            continue
        print(f"    {h:>3} {p['mae']:>9.1f} {s['mae']:>9.1f} "
              f"{s['mae'] - p['mae']:>+8.1f} "
              f"{s['mape'] - p['mape']:>+8.2f} "
              f"{p['theta_med']:>8.3f} {s['theta_med']:>8.3f}")

    print("\n  READING (no verdict baked in):")
    print("    flat dMAE across all h  -> theta-localisation inert even")
    print("       multi-step; uniform-regularity finding STRENGTHENED.")
    print("    dMAE growing with h     -> theta bites as the iterated")
    print("       trajectory leaves the dense core; a real lever at the")
    print("       long-horizon fringe -- changes the writeup / spec.")
    print("    gate-probe prediction: flat ~0 (theta field ~ constant).")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-days", type=int, default=None,
                    help="cap delivery days (quick mechanics check)")
    ap.add_argument("--fixed-theta-test", action="store_true",
                    help="run the fixed-theta falsification control "
                         "(no field build) instead of the simplex backtest")
    args = ap.parse_args()
    if args.fixed_theta_test:
        fixed_theta_test(max_days=args.max_days)
    else:
        main(max_days=args.max_days)
