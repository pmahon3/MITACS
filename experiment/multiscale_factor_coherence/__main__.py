"""Multiscale factor coherence experiment runner.

Implements §6.1 (direct multi-scale factor estimation) and §6.2
(iterated-vs-direct comparison) of
``notes/seeds/multiscale_factor_coherence.md``, under the
preregistration in
``notes/preregistrations/2026-05-26_multiscale-direct-h-step/``.

Calls ``processing.innovations.estimator.global_ols_fit`` (the
production function added for this experiment) and the existing
``experiment._actuals`` / ``experiment.freeze`` / ``experiment.predict``
production functions. /audit code-path should return PRODUCTION-PATH on
this script.

Metrics produced::

    M1(h, dt) = || C_h - C_1^h ||_F / || C_h ||_F        for h in {2,6,12,24}
    M2        = Sigma_h[0,0] / iterated_Sigma_h[0,0]     at (h=12, weekday)

with paired-day-bootstrap 95% CI on each.

Plus a synthetic VAR(1) consistency gate (Phase A baseline.secondary):
on a synthetic VAR(1) where the strict semigroup holds by construction,
the same pipeline must yield M1 small AND M2 close to 1. If it doesn't,
the discrepancy in the Ontario numbers is a methodology artifact.
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import PROJECT_ROOT, load_config
from experiment import freeze
from experiment._actuals import (
    load_pre_cutoff_actuals,
    zscore_params,
    zscore_transform,
)
from experiment.predict import _build_pre_cutoff, _daytype
from processing.innovations.estimator import global_ols_fit


# Horizons specified in Phase A §metric/§stopping_criterion.
HORIZONS = (1, 2, 6, 12, 24)
H_M2 = 12              # v1 M2 + v2 D_RATIO both evaluated at this horizon
DT_M2 = "weekday"      # v1 M2 + v2 D_RATIO both evaluated at this day type
N_BOOTSTRAP = 1000

# v2 precondition gate: kl_synthetic at the headline horizon (h=H_M2)
# must be below this threshold. Above it, the metric is too noisy to
# discriminate strict-semigroup from non-strict and the experiment
# cannot answer the question. Registered in
# notes/preregistrations/2026-05-26_multiscale-direct-h-step-v2/phase_a.yaml.
V2_SYNTHETIC_NOISE_FLOOR_MAX = 0.20    # nats


# ---------------------------------------------------------------------------
# Library construction
# ---------------------------------------------------------------------------


def _h_step_pairs(
    z: pd.Series,
    d: int,
    cutoff: pd.Timestamp,
    h: int,
    anchor_h: int,
    day_type: str,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Build the (X, Y_h) library of h-step pairs filtered to a day type.

    ``X[i] = (z(t_i - 1h), ..., z(t_i - d h))`` (d-dim lag vector at t_i),
    ``Y_h[i] = z(t_i + (h-1) h)`` shifted one full lag-window past the
    anchor end (mirrors the registered predictor's targets, see
    ``experiment.predict._build_pre_cutoff``).

    All pairs satisfy:
      - ``t_i`` is pre-cutoff
      - ``t_i + (h-1) h`` is pre-cutoff (leakage guard)
      - ``_daytype(t_i + (h-1) h, anchor_h) == day_type``  (day-type tagged at TARGET, mirrors production)
      - the lag vector and the target are both finite.

    Returns ``(X, Y_h, target_times)``.
    """
    # Reuse the production library builder for the d-dim X structure.
    # It returns X = blk.iloc[:-1], Y_1 = blk.iloc[1:]. We can derive
    # Y_h by shifting the embedding block index by (h-1) extra rows.
    X_blk, _Y1_blk, emb = _build_pre_cutoff(z, d, cutoff)
    blk = emb.block            # full block: index = t_i (anchor times)
    # X_blk is blk.iloc[:-1].values, so anchor times for X_blk:
    anchor_times = blk.index[:-1]
    target_times = anchor_times + pd.Timedelta(hours=h)
    # Trim to pairs whose target is in the block (i.e. pre-cutoff and embedded)
    in_block_mask = target_times.isin(blk.index)
    X = X_blk[in_block_mask]
    target_times = target_times[in_block_mask]
    # Lookup Y values via the block
    Y_h = blk.loc[target_times].values
    # Filter by day type of the TARGET (mirrors the production
    # convention -- see experiment.predict._daytype usage in predict.py
    # and scratch/rescore_*.py where dt is computed at target t).
    dt_mask = np.array(
        [_daytype(t, anchor_h) == day_type for t in target_times],
        dtype=bool,
    )
    X = X[dt_mask]
    Y_h = Y_h[dt_mask]
    target_times = target_times[dt_mask]
    # Finiteness guard
    finite = np.isfinite(X).all(axis=1) & np.isfinite(Y_h).all(axis=1)
    return X[finite], Y_h[finite], target_times[finite]


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _matrix_power(C: np.ndarray, h: int) -> np.ndarray:
    """Integer matrix power; h=0 returns identity."""
    if h == 0:
        return np.eye(C.shape[0])
    out = C.copy()
    for _ in range(h - 1):
        out = out @ C
    return out


def _iterated_sigma(C1: np.ndarray, Sigma1: np.ndarray, h: int) -> np.ndarray:
    """Discrete-time semigroup-iterated covariance:

      Sigma_h_iter = sum_{j=0..h-1} (C1^j) Sigma1 (C1^j)^T

    Mirrors eq (★) of the seed in the discrete-time analogue
    (multiscale_factor_coherence.md §3).
    """
    out = np.zeros_like(Sigma1)
    Cj = np.eye(C1.shape[0])
    for _ in range(h):
        out = out + Cj @ Sigma1 @ Cj.T
        Cj = Cj @ C1
    return out


def _frobenius_relative(A: np.ndarray, B: np.ndarray) -> float:
    """|| A - B ||_F / || A ||_F ; returns inf if ||A||_F == 0."""
    na = np.linalg.norm(A, "fro")
    if na == 0:
        return float("inf")
    return float(np.linalg.norm(A - B, "fro") / na)


def _kl_gaussian_1d(
    mu_p: np.ndarray, mu_q: np.ndarray,
    var_p: float, var_q: float,
) -> np.ndarray:
    """Closed-form 1D Gaussian KL( N(mu_p, var_p) || N(mu_q, var_q) ),
    elementwise on (mu_p, mu_q) arrays of equal shape.

      KL = log(sigma_q/sigma_p) + (var_p + (mu_p - mu_q)^2) / (2*var_q) - 0.5

    Stable as long as var_p > 0 and var_q > 0. v2 phase_a §metric pins
    P = direct, Q = iterated: forward KL penalises the iterated picture
    for missing mass where the direct says it should be.
    """
    if var_p <= 0 or var_q <= 0:
        raise ValueError(
            f"_kl_gaussian_1d requires positive variances; got "
            f"var_p={var_p}, var_q={var_q}"
        )
    return (
        0.5 * np.log(var_q / var_p)
        + (var_p + (mu_p - mu_q) ** 2) / (2.0 * var_q)
        - 0.5
    )


# ---------------------------------------------------------------------------
# Single-run factor estimation
# ---------------------------------------------------------------------------


def _build_all_libraries(
    z: pd.Series,
    cutoff: pd.Timestamp,
    anchor_h: int,
    dims: dict,
    horizons: tuple[int, ...] = HORIZONS,
) -> dict:
    """Build (X, Y_h, target_dates) for every (day_type, h) ONCE.

    Bootstrap then just row-filters the cached arrays per resampled
    date list; the expensive embedding construction happens once.
    """
    libs: dict = {}
    for dt in ("weekday", "saturday", "sunday"):
        d = int(dims[dt])
        libs[dt] = {}
        for h in horizons:
            X, Y, targets = _h_step_pairs(z, d, cutoff, h, anchor_h, dt)
            target_dates = pd.DatetimeIndex(targets).normalize()
            libs[dt][h] = {"X": X, "Y": Y, "target_dates": target_dates}
    return libs


def _fit_factors(
    z: pd.Series,
    cutoff: pd.Timestamp,
    anchor_h: int,
    dims: dict,
    horizons: tuple[int, ...] = HORIZONS,
    *,
    day_filter: pd.DatetimeIndex | None = None,
    libs: dict | None = None,
) -> dict:
    """For each (day_type, h), fit (C_h, Sigma_h) via global_ols_fit.

    ``day_filter`` (optional) restricts which delivery-day TARGETS are
    used (set during bootstrap to a resample of days). When None, uses
    all days available in z.

    ``libs`` (optional) is the pre-built library cache from
    :func:`_build_all_libraries`. When provided, skip the embedding
    construction (the expensive step) and just row-filter the cached
    arrays. The bootstrap passes this for ~100x speedup.

    Returns nested dict::

        {day_type: {h: {"C": ..., "Sigma": ..., "n": int}}}
    """
    if libs is None:
        libs = _build_all_libraries(z, cutoff, anchor_h, dims, horizons)
    out: dict = {}
    for dt in ("weekday", "saturday", "sunday"):
        out[dt] = {}
        for h in horizons:
            cell = libs[dt][h]
            X, Y, target_dates = cell["X"], cell["Y"], cell["target_dates"]
            if day_filter is not None:
                mask = target_dates.isin(day_filter)
                X, Y = X[mask], Y[mask]
            if len(X) < 50:
                out[dt][h] = {"C": None, "Sigma": None, "n": int(len(X))}
                continue
            C, Sigma, _mu, _resid = global_ols_fit(X, Y)
            out[dt][h] = {"C": C, "Sigma": Sigma, "n": int(len(X))}
    return out


# ---------------------------------------------------------------------------
# Aggregate metrics M1, M2 from one factor-fit pass
# ---------------------------------------------------------------------------


def _aggregate(factors: dict) -> dict:
    """Compute M1, M2, and the full per-(dt,h) tables."""
    drift_rel: dict = {}    # (dt, h) -> ||C_h - C_1^h||_F / ||C_h||_F
    diffusion_ratio: dict = {}    # (dt, h) -> Sigma_h[0,0] / iterated_Sigma_h[0,0]

    for dt in factors:
        C1 = factors[dt][1]["C"]
        Sigma1 = factors[dt][1]["Sigma"]
        if C1 is None or Sigma1 is None:
            continue
        for h in factors[dt]:
            if h == 1:
                continue
            cell = factors[dt][h]
            if cell["C"] is None:
                continue
            Ch = cell["C"]
            Sh = cell["Sigma"]
            Ch_iter = _matrix_power(C1, h)
            drift_rel[(dt, h)] = _frobenius_relative(Ch, Ch_iter)
            Sh_iter = _iterated_sigma(C1, Sigma1, h)
            denom = Sh_iter[0, 0]
            diffusion_ratio[(dt, h)] = float(Sh[0, 0] / denom) if denom != 0 else float("nan")

    # Aggregate metrics
    drift_vals = list(drift_rel.values())
    M1 = float(np.nanmax(drift_vals)) if drift_vals else float("nan")
    M2 = diffusion_ratio.get((DT_M2, H_M2), float("nan"))

    return {
        "M1": M1,
        "M2": M2,
        "drift_rel": drift_rel,
        "diffusion_ratio": diffusion_ratio,
    }


# ---------------------------------------------------------------------------
# Aggregate KL metric (v2 — distributional divergence on coord-0)
# ---------------------------------------------------------------------------


def _aggregate_kl(factors: dict, libs: dict) -> dict:
    """Compute kl_ontario(h, dt) per v2 phase_a §metric.

    For each (day_type, h>1):
      c_direct[i] = (X[i] @ C_h)[0]
      c_iter[i]   = (X[i] @ C_1^h)[0]
      var_direct  = Sigma_h[0, 0]
      var_iter    = iter_Sigma_h[0, 0]   where iter_Sigma_h = sum_{j=0..h-1} C_1^j Sigma_1 (C_1^j)^T
      kl(h, dt)   = mean over rows i of KL( N(c_direct[i], var_direct) || N(c_iter[i], var_iter) )

    ``libs`` provides the (X, Y) library per (dt, h) — required because
    the expectation is over empirical embedding states.

    Returns ``{"kl": {(dt, h): float}, "var_direct": ..., "var_iter": ...,
                "n": {(dt, h): int}}``.
    """
    kl: dict = {}
    var_direct: dict = {}
    var_iter: dict = {}
    n: dict = {}
    for dt in factors:
        C1 = factors[dt][1]["C"]
        Sigma1 = factors[dt][1]["Sigma"]
        if C1 is None or Sigma1 is None:
            continue
        for h in factors[dt]:
            if h == 1:
                continue
            cell = factors[dt][h]
            if cell["C"] is None:
                continue
            Ch = cell["C"]
            Sh = cell["Sigma"]
            Ch_iter = _matrix_power(C1, h)
            Sh_iter = _iterated_sigma(C1, Sigma1, h)
            var_p = float(Sh[0, 0])
            var_q = float(Sh_iter[0, 0])
            X = libs[dt][h]["X"]
            if len(X) == 0 or var_p <= 0 or var_q <= 0:
                kl[(dt, h)] = float("nan")
                continue
            c_direct = (X @ Ch)[:, 0]
            c_iter = (X @ Ch_iter)[:, 0]
            kl_per_row = _kl_gaussian_1d(c_direct, c_iter, var_p, var_q)
            kl[(dt, h)] = float(kl_per_row.mean())
            var_direct[(dt, h)] = var_p
            var_iter[(dt, h)] = var_q
            n[(dt, h)] = int(len(X))
    return {
        "kl": kl,
        "var_direct": var_direct,
        "var_iter": var_iter,
        "n": n,
    }


# ---------------------------------------------------------------------------
# Paired-day bootstrap
# ---------------------------------------------------------------------------


def _all_delivery_dates(z: pd.Series, anchor_h: int) -> pd.DatetimeIndex:
    """Unique calendar dates of post-anchor times in z (the unit of
    resampling for the bootstrap)."""
    shifted = z.index - pd.Timedelta(hours=anchor_h)
    return pd.DatetimeIndex(np.unique(shifted.normalize()))


def _bootstrap_metrics(
    z: pd.Series,
    cutoff: pd.Timestamp,
    anchor_h: int,
    dims: dict,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = 0,
) -> dict:
    """Paired-day bootstrap on (M1, M2). Resamples delivery dates with
    replacement at the (date) level; preserves cross-horizon pairing
    automatically because all h share the same date filter.

    Returns ``{"M1": (low, point, high), "M2": (low, point, high)}``.
    """
    rng = np.random.default_rng(seed)
    all_dates = _all_delivery_dates(z, anchor_h)
    # only pre-cutoff dates as the resample population
    all_dates = all_dates[all_dates <= cutoff]
    n_dates = len(all_dates)
    # Build libraries ONCE; bootstrap iterations only re-row-filter.
    libs = _build_all_libraries(z, cutoff, anchor_h, dims)
    M1s, M2s = [], []
    kl_samples: dict[tuple, list] = {}        # (dt, h) -> list of kl values
    for b in range(n_bootstrap):
        idx = rng.integers(0, n_dates, size=n_dates)
        sampled_dates = pd.DatetimeIndex(all_dates[idx].unique())
        factors_b = _fit_factors(
            z, cutoff, anchor_h, dims,
            day_filter=sampled_dates, libs=libs,
        )
        agg_b = _aggregate(factors_b)
        M1s.append(agg_b["M1"])
        M2s.append(agg_b["M2"])
        # v2: per-bootstrap KL aggregation uses the SAME factors fit. We
        # pass the original libs (not row-filtered) because the
        # expectation E_x[KL] in v2 phase_a is over the (h-step library
        # X rows) -- the X distribution is part of the population, not
        # the resampled unit. Bootstrap resamples the FIT (the C_h,
        # Sigma_h), not the evaluation distribution. This matches the
        # paired-day-bootstrap convention: the day population is the
        # sampling unit for the fit; the X grid is the integration
        # measure.
        agg_kl_b = _aggregate_kl(factors_b, libs)
        for key, val in agg_kl_b["kl"].items():
            kl_samples.setdefault(key, []).append(val)
    M1s = np.array(M1s)
    M2s = np.array(M2s)
    kl_samples_arr = {k: np.array(v) for k, v in kl_samples.items()}
    return {
        "M1": (float(np.nanpercentile(M1s, 2.5)),
               float(np.nanpercentile(M1s, 50)),
               float(np.nanpercentile(M1s, 97.5))),
        "M2": (float(np.nanpercentile(M2s, 2.5)),
               float(np.nanpercentile(M2s, 50)),
               float(np.nanpercentile(M2s, 97.5))),
        "M1_samples": M1s,
        "M2_samples": M2s,
        "kl_samples": kl_samples_arr,
    }


# ---------------------------------------------------------------------------
# Synthetic VAR(1) gate (Phase A baseline.secondary)
# ---------------------------------------------------------------------------


def _synthetic_gate(d: int = 2, n: int = 50_000, seed: int = 1) -> dict:
    """Run the same fit + comparison pipeline on synthetic VAR(1)
    where the strict semigroup holds by construction.

    Returns M1, M2 (per the Phase A definitions). Both must be small:
    M1 close to 0 (drift composes exactly when L is well-defined),
    M2 close to 1 (iterated Sigma equals direct Sigma in the limit
    of infinite library). If either deviates substantially, the
    pipeline has a methodological artifact and Ontario results are
    untrustworthy.
    """
    rng = np.random.default_rng(seed)
    # ground-truth L (small d, well-conditioned, eigenvalues < 1)
    L_true = np.array([[0.6, 0.2], [0.1, 0.5]])[:d, :d]
    Q_true = np.eye(d) * 0.3
    L_chol = np.linalg.cholesky(Q_true)
    z = np.zeros((n, d))
    for t in range(1, n):
        z[t] = L_true @ z[t-1] + L_chol @ rng.standard_normal(d)
    # Build X, Y_h for h = 1, 2, 6, 12, 24 (no day-type)
    out = {}
    Xs: dict[int, np.ndarray] = {}
    for h in HORIZONS:
        X = z[:-h]
        Y = z[h:]
        C, Sigma, _mu, _ = global_ols_fit(X, Y)
        out[h] = {"C": C, "Sigma": Sigma}
        Xs[h] = X
    C1 = out[1]["C"]
    Sigma1 = out[1]["Sigma"]
    drift = {}
    diff_ratio = {}
    kl_per_h: dict[int, float] = {}
    var_p_per_h: dict[int, float] = {}
    var_q_per_h: dict[int, float] = {}
    for h in HORIZONS:
        if h == 1:
            continue
        Ch_iter = _matrix_power(C1, h)
        drift[h] = _frobenius_relative(out[h]["C"], Ch_iter)
        Sh_iter = _iterated_sigma(C1, Sigma1, h)
        denom = Sh_iter[0, 0]
        diff_ratio[h] = float(out[h]["Sigma"][0, 0] / denom) if denom != 0 else float("nan")
        # v2 KL on the scalar innovation coordinate
        Ch = out[h]["C"]
        var_p = float(out[h]["Sigma"][0, 0])
        var_q = float(Sh_iter[0, 0])
        X_h = Xs[h]
        c_direct = (X_h @ Ch)[:, 0]
        c_iter = (X_h @ Ch_iter)[:, 0]
        kl_per_h[h] = float(_kl_gaussian_1d(c_direct, c_iter, var_p, var_q).mean())
        var_p_per_h[h] = var_p
        var_q_per_h[h] = var_q
    return {
        "M1": float(max(drift.values())) if drift else float("nan"),
        "M2": diff_ratio.get(H_M2, float("nan")),
        "drift": drift,
        "diff_ratio": diff_ratio,
        "kl": kl_per_h,
        "var_direct": var_p_per_h,
        "var_iter": var_q_per_h,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path,
                   default=Path("scratch/data/multiscale_factor/factors.pkl"))
    p.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    p.add_argument("--skip-bootstrap", action="store_true",
                   help="point estimates only (for fast iteration)")
    args = p.parse_args(argv)

    cfg = load_config()
    spec = freeze.load_verified()
    cutoff = pd.Timestamp(spec["data_cutoff"])
    anchor_h = int(cfg.data.day_anchor_hours)
    dims = spec["predictor"]["embedding_dims"]
    clim_method = spec["predictor"].get("climatology_method", "month_hour")
    k_year = spec["predictor"].get("fourier_k_year")
    k_day = spec["predictor"].get("fourier_k_day")

    print(f"# multiscale_factor_coherence")
    print(f"  cutoff           = {cutoff}")
    print(f"  anchor_h         = {anchor_h}")
    print(f"  embedding_dims   = {dict(dims)}")
    print(f"  climatology      = {clim_method}")
    print(f"  horizons         = {HORIZONS}")
    print(f"  M2 evaluated at  = (h={H_M2}, day_type={DT_M2})")
    print()

    # ---- data prep (production-path) ----
    raw_pre = load_pre_cutoff_actuals(cutoff).dropna()
    zp = zscore_params(cutoff, method=clim_method,
                       k_year=k_year, k_day=k_day)
    z = zscore_transform(raw_pre, zp)

    # ---- synthetic gate FIRST (v2 precondition; gates Ontario reporting) ----
    print(f"# synthetic VAR(1) consistency gate ...")
    t0 = time.time()
    syn = _synthetic_gate()
    print(f"  v1 metrics:   M1 = {syn['M1']:.4f}     M2 = {syn['M2']:.4f}")
    print(f"  v2 KL nats:   per-h = {syn['kl']}")
    print(f"  per-h drift:       {syn['drift']}")
    print(f"  per-h diff ratio:  {syn['diff_ratio']}")
    print(f"  ({time.time()-t0:.1f}s)")
    print()

    # ---- v2 PRECONDITION GATE ----
    syn_kl_headline = syn["kl"].get(H_M2, float("inf"))
    print(f"# v2 precondition gate: kl_synthetic(h={H_M2}) "
          f"= {syn_kl_headline:.4f} nats")
    if syn_kl_headline >= V2_SYNTHETIC_NOISE_FLOOR_MAX:
        print(f"  FAIL: noise floor exceeds threshold "
              f"{V2_SYNTHETIC_NOISE_FLOOR_MAX} nats.")
        print(f"  The KL metric itself is degenerate at this h; the "
              f"experiment CANNOT answer the question. Aborting before "
              f"reporting Ontario numbers (per v2 phase_a §metric).")
        print(f"  Next step: redesign metric, re-preregister, rerun.")
        return 2
    print(f"  PASS: under threshold {V2_SYNTHETIC_NOISE_FLOOR_MAX} nats. "
          f"Ontario reporting authorized.")
    print()

    # ---- Ontario point estimate (v1 metrics + v2 KL) ----
    print(f"# Ontario direct-h-step factor fit ...")
    t0 = time.time()
    libs = _build_all_libraries(z, cutoff, anchor_h, dims)
    factors = _fit_factors(z, cutoff, anchor_h, dims, libs=libs)
    point = _aggregate(factors)
    kl_point = _aggregate_kl(factors, libs)
    print(f"  v1: M1 (point) = {point['M1']:.4f}")
    print(f"  v1: M2 (point) = {point['M2']:.4f}")
    kl_headline = kl_point["kl"].get((DT_M2, H_M2), float("nan"))
    D_RATIO_point = kl_headline / syn_kl_headline if syn_kl_headline > 0 else float("nan")
    print(f"  v2: kl_ontario({DT_M2}, h={H_M2}) = {kl_headline:.4f} nats")
    print(f"  v2: D_RATIO (point) = {D_RATIO_point:.4f}")
    print()
    print("  v2 per-(dt, h) kl_ontario [nats]:")
    for (dt, h), v in sorted(kl_point["kl"].items()):
        print(f"    {dt:>8s}  h={h:>2d}  {v:.4f}")
    print()
    print("  v1 per-(dt, h) drift_rel:")
    for (dt, h), v in sorted(point["drift_rel"].items()):
        print(f"    {dt:>8s}  h={h:>2d}  {v:.4f}")
    print()
    print("  v1 per-(dt, h) diffusion_ratio Sigma_h[0,0] / iter_Sigma_h[0,0]:")
    for (dt, h), v in sorted(point["diffusion_ratio"].items()):
        print(f"    {dt:>8s}  h={h:>2d}  {v:.4f}")
    print(f"  ({time.time()-t0:.1f}s)")
    print()

    # ---- bootstrap CI ----
    if args.skip_bootstrap:
        print("# bootstrap SKIPPED")
        boot = None
        D_RATIO_ci = None
    else:
        print(f"# paired-day bootstrap (n={args.n_bootstrap}) ...")
        t0 = time.time()
        boot = _bootstrap_metrics(z, cutoff, anchor_h, dims, args.n_bootstrap)
        print(f"  v1 M1: 95% CI [{boot['M1'][0]:.4f}, {boot['M1'][2]:.4f}]  "
              f"median {boot['M1'][1]:.4f}")
        print(f"  v1 M2: 95% CI [{boot['M2'][0]:.4f}, {boot['M2'][2]:.4f}]  "
              f"median {boot['M2'][1]:.4f}")
        # v2 D_RATIO CI: divide each bootstrap KL sample by syn_kl_headline
        # (syn_kl is computed once, not bootstrapped — phase_a says the
        # synthetic gate is the noise-floor REFERENCE unit, not itself a
        # variable to bootstrap; the Ontario bootstrap CI captures the
        # uncertainty in the numerator).
        kl_headline_samples = boot["kl_samples"].get((DT_M2, H_M2))
        if kl_headline_samples is None:
            D_RATIO_ci = None
            print(f"  v2 D_RATIO: no samples at (h={H_M2}, dt={DT_M2})")
        else:
            d_ratio_samples = kl_headline_samples / syn_kl_headline
            D_RATIO_ci = (
                float(np.nanpercentile(d_ratio_samples, 2.5)),
                float(np.nanpercentile(d_ratio_samples, 50)),
                float(np.nanpercentile(d_ratio_samples, 97.5)),
            )
            print(f"  v2 D_RATIO: 95% CI [{D_RATIO_ci[0]:.4f}, "
                  f"{D_RATIO_ci[2]:.4f}]  median {D_RATIO_ci[1]:.4f}")
        print(f"  ({time.time()-t0:.1f}s)")
        print()
        # ---- v2 verdict against pre-registered criteria ----
        print(f"# v2 verdict (pre-registered criteria):")
        print(f"  D_RATIO point = {D_RATIO_point:.4f}")
        if D_RATIO_ci:
            print(f"  D_RATIO 95% CI = [{D_RATIO_ci[0]:.4f}, "
                  f"{D_RATIO_ci[2]:.4f}]")
        if D_RATIO_point > 5.0:
            print(f"  D_RATIO > 5.0 → CORROBORATION criterion fires "
                  f"(seed picture C supported)")
        elif D_RATIO_point <= 2.0:
            print(f"  D_RATIO <= 2.0 → FALSIFICATION criterion fires "
                  f"(strict-semigroup picture A adequate)")
        else:
            print(f"  D_RATIO in (2.0, 5.0] → AMBIGUOUS region "
                  f"(report and characterise; do not claim)")
        print()

    # ---- pickle everything ----
    out = args.out
    if not out.is_absolute():
        out = (Path.cwd() / out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "factors": factors,
        "point": point,
        "kl_point": kl_point,
        "D_RATIO_point": D_RATIO_point,
        "D_RATIO_ci": D_RATIO_ci,
        "synthetic": syn,
        "syn_kl_headline": syn_kl_headline,
        "bootstrap": boot,
        "config": {
            "cutoff": cutoff,
            "anchor_h": anchor_h,
            "embedding_dims": dict(dims),
            "horizons": HORIZONS,
            "H_M2": H_M2,
            "DT_M2": DT_M2,
            "v2_synthetic_noise_floor_max": V2_SYNTHETIC_NOISE_FLOOR_MAX,
            "n_bootstrap": args.n_bootstrap if not args.skip_bootstrap else 0,
        },
    }
    with out.open("wb") as f:
        pickle.dump(payload, f)
    try:
        rel = out.relative_to(PROJECT_ROOT)
        print(f"  wrote {rel}")
    except ValueError:
        print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
