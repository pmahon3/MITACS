"""Q1 runner -- Student-t vs Gaussian on marginal PIT + D_RATIO_t.

Implements phase_a of
``notes/preregistrations/2026-05-26_q1-student-t-vs-gaussian/``.
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import t as student_t_dist

from config import PROJECT_ROOT, load_config
from experiment import freeze
from experiment._actuals import (
    load_actuals,
    load_pre_cutoff_actuals,
    zscore_params,
    zscore_transform,
)
from experiment.backtest import _complete_delivery_days
from experiment.predict import _build_pre_cutoff, _daytype
from processing.innovations.estimator import (
    global_ols_fit,
    student_t_mle_fit,
)


# Pre-registered constants (phase_a)
N_PARTICLES = 200          # SMC particle count for iteration
NU_BOUNDS = (2.5, 300.0)   # per phase_a R-D
NU_PATHOLOGICAL_LOW = 2.5
NU_PATHOLOGICAL_HIGH = 200.0  # within 1% of upper bound
N_BOOTSTRAP = 1000
N_PIT_BINS = 10
HEADLINE_H = 12
HEADLINE_DAYTYPE = "weekday"
SYNTHETIC_NU_TRUE = 5.0
SYNTHETIC_N = 50_000


# ---------------------------------------------------------------------------
# M0 — Gaussian baseline (analytic PIT from v2 factors)
# ---------------------------------------------------------------------------


def _iterated_drift_and_var_gauss(
    C1: np.ndarray, Sigma1: np.ndarray, x_state: np.ndarray, h_max: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form iterated mean and variance under Gaussian (M0).

    Same computation P1 (PIT-calendar-fingerprint) used. Returns
    ``(mu_iter[1..h_max], var_iter[1..h_max])`` arrays for the
    scalar coord-0 prediction.
    """
    h_max = int(h_max)
    mu = np.zeros(h_max)
    var = np.zeros(h_max)
    Cj = np.eye(C1.shape[0])
    sum_cov = np.zeros_like(Sigma1)
    x = x_state.copy()
    for h in range(1, h_max + 1):
        sum_cov = sum_cov + Cj @ Sigma1 @ Cj.T
        var[h - 1] = sum_cov[0, 0]
        x = C1 @ x
        mu[h - 1] = x[0]
        Cj = Cj @ C1
    return mu, var


def _pit_M0(
    z_full: pd.Series,
    factors: dict,
    cutoff: pd.Timestamp,
    anchor_h: int,
    dims: dict,
) -> pd.DataFrame:
    """Build the M0 PIT table -- analytic Gaussian PIT for every
    post-cutoff (delivery_day, hour-of-day)."""
    days = _complete_delivery_days(z_full, anchor_h)
    days = days[days > cutoff]
    rows = []
    for D in days:
        D = pd.Timestamp(D.date())
        dt = _daytype(D + pd.Timedelta(hours=anchor_h), anchor_h)
        d = int(dims[dt])
        C1 = factors[dt][1]["C"]
        S1 = factors[dt][1]["Sigma"]
        issue_anchor = D + pd.Timedelta(hours=anchor_h - 1)
        lag_times = [issue_anchor - pd.Timedelta(hours=i) for i in range(d)]
        if not all(t in z_full.index for t in lag_times):
            continue
        x_state = z_full.reindex(lag_times).to_numpy()
        if not np.all(np.isfinite(x_state)):
            continue
        mu_iter, var_iter = _iterated_drift_and_var_gauss(C1, S1, x_state, h_max=24)
        target_dts = [
            D + pd.Timedelta(hours=anchor_h + (h - 1)) for h in range(1, 25)
        ]
        actuals = z_full.reindex(target_dts).to_numpy()
        if not np.all(np.isfinite(actuals)):
            continue
        for h in range(1, 25):
            i = h - 1
            std = np.sqrt(var_iter[i]) if var_iter[i] > 0 else np.nan
            if not np.isfinite(std) or std <= 0:
                continue
            u = float(ndtr((actuals[i] - mu_iter[i]) / std))
            rows.append({
                "delivery_day": D,
                "h": h,
                "day_type": dt,
                "u_PIT": u,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# M1 / M2 — Student-t kernel; SMC propagation for iteration
# ---------------------------------------------------------------------------


def _fit_t_kernels_per_daytype(
    z_pre: pd.Series,
    dims: dict,
    cutoff: pd.Timestamp,
) -> dict:
    """For each day-type, refit (C, s, nu) on the pre-cutoff library
    via Student-t MLE on coord-0 of the residual. Returns
    ``{day_type: {"C": ndarray (d, 1 — coord-0 only), "s": float,
    "nu": float, "d": int}}``.
    """
    out = {}
    for dt in ("weekday", "saturday", "sunday"):
        d = int(dims[dt])
        # Use the production library builder so X has the d-lag structure
        # matching what v2/eigenmodes/P1 used.
        X, Y, _emb = _build_pre_cutoff(z_pre.to_frame("zscore").asfreq("h")["zscore"]
                                        if False else z_pre, d, cutoff)
        # Filter to day-type at the anchor time of each X-row. The
        # _build_pre_cutoff function gives all anchors; we restrict to dt.
        # Simpler: refit on all-anchor data is the same as v2's approach
        # (the per-day-type split is downstream at the kernel application
        # step). For Student-t MLE, we want the residual law per day-type,
        # which means we need to subset.
        # For simplicity here: refit per day-type by re-building the
        # library with a day-type filter inline. We do this by reusing
        # global_ols_fit's library (which is all anchors) and filtering
        # Y/X rows by day-type at the anchor index. But _build_pre_cutoff
        # doesn't expose anchor times. We work around by recomputing
        # anchors from z_pre directly.
        d_max = max(int(v) for v in dims.values())
        idx = z_pre.index
        valid = []
        for i in range(d, len(idx) - 1):
            anchor = idx[i]
            target = idx[i + 1]
            if target - anchor != pd.Timedelta(hours=1):
                continue
            if _daytype(anchor, 0) != dt:
                continue
            # build lag vector
            ok = True
            lag = np.zeros(d)
            for j in range(d):
                if idx[i - j] != anchor - pd.Timedelta(hours=j):
                    ok = False
                    break
                lag[j] = z_pre.iloc[i - j]
            if not ok or not np.all(np.isfinite(lag)):
                continue
            y = z_pre.iloc[i + 1]
            if not np.isfinite(y):
                continue
            valid.append((lag, y))
        if len(valid) < 200:
            out[dt] = None
            continue
        X_dt = np.array([v[0] for v in valid])
        Y_dt = np.array([v[1] for v in valid])
        C_dt, s_dt, nu_dt, _ = student_t_mle_fit(X_dt, Y_dt)
        out[dt] = {
            "C": C_dt,           # (d, 1); coord-0 of the next-state
            "s": s_dt,
            "nu": nu_dt,
            "d": d,
            "n": int(len(valid)),
        }
    return out


def _smc_iterate(
    C1_col: np.ndarray,    # (d, 1) coord-0 row of C (the only stochastic one)
    s_1: float,
    nu_1: float,
    x_state_d: np.ndarray, # (d,) initial state (z_{t-1}, z_{t-2}, ..., z_{t-d})
    h_max: int,
    M: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """SMC particle propagation under the Student-t kernel.

    At each step h, propagate M particles:
      coord-0 of the next state is drawn as
        u_{j, h+1} = C1_col[:, 0] @ x_{j, h} + s_1 * T_j  with T_j ~ t_{nu_1}(0, 1)
      coords 1..d-1 shift by one (deterministic, per rank-1 structure).

    Returns ``z_h`` of shape ``(M, h_max)``: the coord-0 particle
    values at each scale h = 1..h_max.
    """
    d = x_state_d.shape[0]
    # Particles: each particle is a d-dim state. Initial: M copies of x_state_d.
    particles = np.tile(x_state_d, (M, 1))  # (M, d)
    out = np.zeros((M, h_max))
    for h in range(1, h_max + 1):
        # Student-t noise
        T_draws = student_t_dist.rvs(df=nu_1, size=M, random_state=rng)
        # next-state coord-0
        next_z = (particles @ C1_col).ravel() + s_1 * T_draws
        out[:, h - 1] = next_z
        # shift particles: new state is (next_z, particles[:, :d-1])
        particles = np.column_stack([next_z, particles[:, :d - 1]])
    return out


def _pit_M1(
    z_full: pd.Series,
    factors_M0: dict,
    t_kernels: dict,
    cutoff: pd.Timestamp,
    anchor_h: int,
    dims: dict,
    rng: np.random.Generator,
    M: int = N_PARTICLES,
) -> pd.DataFrame:
    """Build the M1 PIT table: SMC-iterated Student-t kernel forecasts.

    PIT via randomized mid-rank on the particle distribution at each
    (delivery_day, hour-of-day), matching the distributional_smc
    convention.
    """
    days = _complete_delivery_days(z_full, anchor_h)
    days = days[days > cutoff]
    rows = []
    for D in days:
        D = pd.Timestamp(D.date())
        dt = _daytype(D + pd.Timedelta(hours=anchor_h), anchor_h)
        ker = t_kernels.get(dt)
        if ker is None:
            continue
        d = int(dims[dt])
        issue_anchor = D + pd.Timedelta(hours=anchor_h - 1)
        lag_times = [issue_anchor - pd.Timedelta(hours=i) for i in range(d)]
        if not all(t in z_full.index for t in lag_times):
            continue
        x_state = z_full.reindex(lag_times).to_numpy()
        if not np.all(np.isfinite(x_state)):
            continue
        particles_h = _smc_iterate(
            ker["C"], ker["s"], ker["nu"], x_state, h_max=24, M=M, rng=rng,
        )
        target_dts = [
            D + pd.Timedelta(hours=anchor_h + (h - 1)) for h in range(1, 25)
        ]
        actuals = z_full.reindex(target_dts).to_numpy()
        if not np.all(np.isfinite(actuals)):
            continue
        for h in range(1, 25):
            i = h - 1
            samp = particles_h[:, i]
            # Randomized mid-rank PIT (matches scratch/distributional_smc.py)
            less = float(np.sum(samp < actuals[i]))
            eq = float(np.sum(samp == actuals[i]))
            U = rng.random()
            u = (less + U * (eq + 1.0)) / (M + 1.0)
            rows.append({
                "delivery_day": D,
                "h": h,
                "day_type": dt,
                "u_PIT": float(u),
                "nu_used": ker["nu"],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# M2 — full Student-t commitment (climatology refit)
# ---------------------------------------------------------------------------


def _fit_t_climatology(raw_pre: pd.Series) -> dict:
    """Fit Student-t (mu, s, nu) per (month, hour-of-day) bin on
    pre-cutoff raw demand."""
    out = {}
    for (m, h), vals in raw_pre.groupby([raw_pre.index.month, raw_pre.index.hour]):
        y = vals.values.astype(float)
        y = y[np.isfinite(y)]
        if len(y) < 50:
            out[(int(m), int(h))] = None
            continue
        # 1D location-scale-nu: model as y = mu + s * t_{nu}(0, 1)
        # Use student_t_mle_fit with X = ones (intercept-only regression)
        X1 = np.ones((len(y), 1))
        C_fit, s_fit, nu_fit, _ = student_t_mle_fit(X1, y)
        out[(int(m), int(h))] = {
            "mu": float(C_fit[0, 0]),
            "s": float(s_fit),
            "nu": float(nu_fit),
            "n": int(len(y)),
        }
    return out


def _z_under_t_climatology(raw: pd.Series, clim: dict) -> pd.Series:
    """Compute z = (raw - mu_{m,h}) / s_{m,h} using the Student-t climatology."""
    out = np.full(len(raw), np.nan)
    for i, t in enumerate(raw.index):
        key = (t.month, t.hour)
        params = clim.get(key)
        if params is None:
            continue
        out[i] = (raw.iloc[i] - params["mu"]) / params["s"]
    return pd.Series(out, index=raw.index, name="z_t")


# ---------------------------------------------------------------------------
# Marginal PIT chi^2
# ---------------------------------------------------------------------------


def _marginal_pit_chi2(u: np.ndarray, n_bins: int = N_PIT_BINS) -> tuple[float, np.ndarray]:
    """10-bin chi^2 vs uniform on (0, 1)."""
    u = u[np.isfinite(u)]
    bins = np.linspace(0, 1, n_bins + 1)
    hist, _ = np.histogram(u, bins=bins)
    expected = len(u) / n_bins
    chi2 = float(((hist - expected) ** 2 / expected).sum())
    return chi2, hist


# ---------------------------------------------------------------------------
# D_RATIO_t at headline cell — directly-fit Student-t at h=12 vs SMC-iterated
# ---------------------------------------------------------------------------


def _fit_t_h_step_for_daytype(
    z_pre: pd.Series,
    cutoff: pd.Timestamp,
    h: int,
    day_type: str,
    d: int,
) -> dict | None:
    """Fit Student-t directly at horizon h for one day-type.

    Library: (X, Y_h) pairs where X is the d-lag embedding at anchor t
    and Y_h is the coord-0 value at t + h. Day-type filter at the
    anchor.
    """
    idx = z_pre.index
    valid = []
    for i in range(d, len(idx) - h):
        anchor = idx[i]
        target = idx[i + h]
        # contiguous embedding window + target available
        # anchor's day-type:
        if _daytype(anchor, 0) != day_type:
            continue
        # contiguity check
        ok = True
        lag = np.zeros(d)
        for j in range(d):
            if idx[i - j] != anchor - pd.Timedelta(hours=j):
                ok = False
                break
            lag[j] = z_pre.iloc[i - j]
        if not ok or not np.all(np.isfinite(lag)):
            continue
        if target - anchor != pd.Timedelta(hours=h):
            continue
        y = z_pre.iloc[i + h]
        if not np.isfinite(y):
            continue
        valid.append((lag, y))
    if len(valid) < 100:
        return None
    X = np.array([v[0] for v in valid])
    Y = np.array([v[1] for v in valid])
    C_fit, s_fit, nu_fit, _ = student_t_mle_fit(X, Y)
    return {"C": C_fit, "s": s_fit, "nu": nu_fit, "n": len(valid)}


def _wasserstein2_1d(samples_a: np.ndarray, samples_b: np.ndarray) -> float:
    """1D Wasserstein-2 squared distance between two empirical
    distributions via L^2 distance of sorted samples (Hoeffding 1940)."""
    # Resample to equal size by sorting; the L^2 distance of sorted
    # samples is the 1D W_2^2 for equal n; for unequal n, interpolate.
    a = np.sort(samples_a[np.isfinite(samples_a)])
    b = np.sort(samples_b[np.isfinite(samples_b)])
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    # quantile-interpolate to a common grid
    n = max(len(a), len(b))
    grid = (np.arange(n) + 0.5) / n
    qa = np.interp(grid, (np.arange(len(a)) + 0.5) / len(a), a)
    qb = np.interp(grid, (np.arange(len(b)) + 0.5) / len(b), b)
    return float(np.mean((qa - qb) ** 2))


def _compute_d_ratio_t(
    z_pre: pd.Series, z_full: pd.Series,
    factors_M0: dict, t_kernels: dict,
    cutoff: pd.Timestamp, anchor_h: int, dims: dict,
    h: int = HEADLINE_H, day_type: str = HEADLINE_DAYTYPE,
    M: int = N_PARTICLES, n_states: int = 200,
    rng: np.random.Generator | None = None,
) -> dict:
    """Student-t analogue of v2's D_RATIO at headline cell.

    For up to n_states post-cutoff weekday issue-anchor states x:
      iter = SMC particles of Student-t kernel iterated h steps
      direct = samples from the directly-fit Student-t at h
      W_2 = 1D Wasserstein-2 between the two empirical samples
    Aggregate by averaging W_2^2 over states.

    Synthetic gate uses a known Student-t-VAR system.
    """
    if rng is None:
        rng = np.random.default_rng(7)

    # Directly-fit Student-t at horizon h
    d = int(dims[day_type])
    direct = _fit_t_h_step_for_daytype(z_pre, cutoff, h, day_type, d)
    if direct is None:
        return {"empirical_w2": float("nan"), "synthetic_w2": float("nan"),
                "D_RATIO_t": float("nan"), "note": "direct fit failed"}

    # Iterate M1 kernel for n_states post-cutoff states
    ker = t_kernels[day_type]
    days = _complete_delivery_days(z_full, anchor_h)
    days = days[days > cutoff]
    # take only days whose anchor is the requested day_type
    day_state_pairs = []
    for D in days:
        D = pd.Timestamp(D.date())
        dt_here = _daytype(D + pd.Timedelta(hours=anchor_h), anchor_h)
        if dt_here != day_type:
            continue
        issue_anchor = D + pd.Timedelta(hours=anchor_h - 1)
        lag_times = [issue_anchor - pd.Timedelta(hours=i) for i in range(d)]
        if not all(t in z_full.index for t in lag_times):
            continue
        x_state = z_full.reindex(lag_times).to_numpy()
        if not np.all(np.isfinite(x_state)):
            continue
        day_state_pairs.append((D, x_state))
    if len(day_state_pairs) > n_states:
        idx = rng.choice(len(day_state_pairs), size=n_states, replace=False)
        day_state_pairs = [day_state_pairs[i] for i in sorted(idx)]
    if not day_state_pairs:
        return {"empirical_w2": float("nan"), "synthetic_w2": float("nan"),
                "D_RATIO_t": float("nan"), "note": "no post-cutoff states for this daytype"}

    w2_list = []
    state_day_list = []
    n_direct_samples = 1000
    for D_state, x_state in day_state_pairs:
        particles_h = _smc_iterate(
            ker["C"], ker["s"], ker["nu"], x_state, h_max=h, M=M, rng=rng,
        )
        iter_samp = particles_h[:, h - 1]
        # direct samples: mean = (x_state @ direct["C"])[0]; scale = direct["s"]
        # nu = direct["nu"]; draw T's and scale+shift.
        mu_direct = float((x_state.reshape(1, d) @ direct["C"]).ravel()[0])
        T = student_t_dist.rvs(df=direct["nu"], size=n_direct_samples, random_state=rng)
        direct_samp = mu_direct + direct["s"] * T
        w2 = _wasserstein2_1d(iter_samp, direct_samp)
        w2_list.append(w2)
        state_day_list.append(D_state)
    empirical_w2 = float(np.mean(w2_list))

    # Synthetic gate: Student-t-VAR
    syn_rng = np.random.default_rng(13)
    L_syn = np.array([[0.6, 0.2], [0.1, 0.5]])
    s_syn = 0.3
    nu_syn = SYNTHETIC_NU_TRUE
    n_syn = SYNTHETIC_N
    z_syn = np.zeros((n_syn, 2))
    for t in range(1, n_syn):
        T_draw = student_t_dist.rvs(df=nu_syn, size=2, random_state=syn_rng)
        z_syn[t] = L_syn @ z_syn[t-1] + s_syn * T_draw

    # Direct-h fit on synthetic
    X_syn = z_syn[:-h]
    Y_syn = z_syn[h:, 0]
    C_syn, s_syn_fit, nu_syn_fit, _ = student_t_mle_fit(X_syn, Y_syn)
    # 1-step fit for SMC propagation
    X_syn1 = z_syn[:-1]
    Y_syn1 = z_syn[1:, 0]
    C_syn1, s_syn1, nu_syn1, _ = student_t_mle_fit(X_syn1, Y_syn1)

    # Sample n_states synthetic states; compute W2 per state; average
    state_idx = syn_rng.choice(len(X_syn) - 10, size=min(n_states, len(X_syn) - 10), replace=False)
    w2_syn_list = []
    for si in state_idx:
        xs = z_syn[si]
        particles_syn = _smc_iterate(
            C_syn1, s_syn1, nu_syn1, xs, h_max=h, M=M, rng=syn_rng,
        )
        iter_samp_syn = particles_syn[:, h - 1]
        mu_direct_syn = float(xs.reshape(1, -1) @ C_syn)
        T_syn = student_t_dist.rvs(df=nu_syn_fit, size=n_direct_samples,
                                    random_state=syn_rng)
        direct_samp_syn = mu_direct_syn + s_syn_fit * T_syn
        w2_syn = _wasserstein2_1d(iter_samp_syn, direct_samp_syn)
        w2_syn_list.append(w2_syn)
    synthetic_w2 = float(np.mean(w2_syn_list))

    d_ratio = empirical_w2 / synthetic_w2 if synthetic_w2 > 0 else float("nan")
    return {
        "empirical_w2": empirical_w2,
        "synthetic_w2": synthetic_w2,
        "D_RATIO_t": d_ratio,
        "n_states_empirical": len(day_state_pairs),
        "n_states_synthetic": len(state_idx),
        "direct_nu": direct["nu"],
        "direct_n": direct["n"],
        "synthetic_nu_fitted_1step": nu_syn1,
        "synthetic_nu_fitted_h_step": nu_syn_fit,
        # Exposed for paired-day bootstrap by caller:
        "per_state_w2": w2_list,
        "per_state_day": state_day_list,
        "synthetic_w2_per_state": w2_syn_list,
    }


# ---------------------------------------------------------------------------
# Synthetic gate -- separate check that M1 fits a Student-t generator correctly
# ---------------------------------------------------------------------------


def _synthetic_gate(
    nu_true: float = SYNTHETIC_NU_TRUE, n: int = SYNTHETIC_N, seed: int = 17,
) -> dict:
    """Generate a Student-t-VAR(1), fit Student-t MLE under M0/M1
    equivalents, check that M1 recovers calibration.

    Diagnostic only (not part of the verdict criteria).
    """
    rng = np.random.default_rng(seed)
    L_true = np.array([[0.7, 0.0], [0.0, 0.0]])
    s_true = 0.5
    z = np.zeros((n, 2))
    for t in range(1, n):
        T = student_t_dist.rvs(df=nu_true, size=1, random_state=rng)
        z[t, 0] = L_true[0, 0] * z[t-1, 0] + s_true * T[0]
        z[t, 1] = z[t-1, 0]   # delay shift

    # fit Student-t kernel
    X_syn = z[:-1]
    Y_syn = z[1:, 0]
    C_fit, s_fit, nu_fit, _ = student_t_mle_fit(X_syn, Y_syn)
    return {
        "nu_true": float(nu_true),
        "nu_fit": float(nu_fit),
        "s_true": float(s_true),
        "s_fit": float(s_fit),
        "C_true_first_row": [float(L_true[0, 0]), 0.0],
        "C_fit_first_col": list(C_fit.ravel()),
        "recovery_OK": (
            abs(nu_fit - nu_true) / nu_true < 0.20
            and abs(s_fit - s_true) / s_true < 0.20
            and abs(C_fit[0, 0] - L_true[0, 0]) < 0.05
        ),
    }


# ---------------------------------------------------------------------------
# Outcome evaluation
# ---------------------------------------------------------------------------


def _evaluate_outcome(
    chi2_M0: float, chi2_M1: float, chi2_M2: float,
    t_kernels: dict, t_clim: dict | None,
) -> dict:
    """Evaluate pre-registered R-A / R-B / R-C / R-D outcomes per
    phase_a §shape_outcomes."""
    # R-D check first: any pathological nu in M1 or M2?
    pathological = False
    pathology_details = []
    for dt, ker in t_kernels.items():
        if ker is None:
            pathological = True
            pathology_details.append(f"M1 {dt}: fit failed")
            continue
        nu = ker["nu"]
        if nu < NU_PATHOLOGICAL_LOW or nu > NU_PATHOLOGICAL_HIGH:
            pathological = True
            pathology_details.append(
                f"M1 {dt}: nu = {nu:.2f} outside [{NU_PATHOLOGICAL_LOW}, {NU_PATHOLOGICAL_HIGH}]"
            )
    if t_clim is not None:
        n_pathological_clim = sum(
            1 for v in t_clim.values()
            if v is not None and (v["nu"] < NU_PATHOLOGICAL_LOW
                                  or v["nu"] > NU_PATHOLOGICAL_HIGH)
        )
        if n_pathological_clim > 0:
            pathological = True
            pathology_details.append(
                f"M2 climatology: {n_pathological_clim} bins with pathological nu"
            )
    if pathological:
        return {
            "verdict": "R-D",
            "reason": "; ".join(pathology_details),
        }

    # Best (= lowest chi^2) of M1, M2
    best = min(chi2_M1, chi2_M2)
    if best <= 30:
        verdict = "R-A"
        reason = f"min(chi2_M1={chi2_M1:.1f}, chi2_M2={chi2_M2:.1f}) = {best:.1f} <= 30; Student-t fixes the marginal pathology"
    elif best <= 500:
        verdict = "R-B"
        reason = f"min(chi2_M1={chi2_M1:.1f}, chi2_M2={chi2_M2:.1f}) = {best:.1f} in (30, 500]; Student-t helps but doesn't fully calibrate"
    else:
        verdict = "R-C"
        reason = f"min(chi2_M1={chi2_M1:.1f}, chi2_M2={chi2_M2:.1f}) = {best:.1f} > 500; Student-t does not absorb the marginal pathology"
    return {"verdict": verdict, "reason": reason}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="in_path", type=Path,
                   default=Path("scratch/data/multiscale_factor/factors_v2_final.pkl"))
    p.add_argument("--out", type=Path,
                   default=Path("scratch/data/distributional_class_q1/q1.pkl"))
    p.add_argument("--n-particles", type=int, default=N_PARTICLES)
    p.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    p.add_argument("--skip-bootstrap", action="store_true",
                   help="point estimates only (for fast iteration)")
    p.add_argument("--skip-m2", action="store_true",
                   help="skip M2 (full Student-t climatology refit); M0+M1 only")
    args = p.parse_args(argv)

    cfg = load_config()
    spec = freeze.load_verified()
    cutoff = pd.Timestamp(spec["data_cutoff"])
    anchor_h = int(cfg.data.day_anchor_hours)
    dims = spec["predictor"]["embedding_dims"]
    clim_method = spec["predictor"].get("climatology_method", "month_hour")
    k_year = spec["predictor"].get("fourier_k_year")
    k_day = spec["predictor"].get("fourier_k_day")

    print(f"# Q1: Student-t vs Gaussian")
    print(f"  cutoff:       {cutoff}")
    print(f"  anchor_h:     {anchor_h}")
    print(f"  embedding:    {dict(dims)}")
    print(f"  N_PARTICLES:  {args.n_particles}")
    print(f"  N_BOOTSTRAP:  {args.n_bootstrap}")
    print()

    rng = np.random.default_rng(42)

    # ---- M0 setup: read v2 factors ----
    with args.in_path.open("rb") as f:
        v2 = pickle.load(f)
    factors_M0 = v2["factors"]

    raw_full = load_actuals(cutoff=None).dropna()
    zp = zscore_params(cutoff, method=clim_method, k_year=k_year, k_day=k_day)
    z_full = zscore_transform(raw_full, zp)
    raw_pre = load_pre_cutoff_actuals(cutoff).dropna()
    z_pre = zscore_transform(raw_pre, zp)

    print(f"# synthetic gate (Student-t generator known) ...")
    t0 = time.time()
    syn = _synthetic_gate()
    print(f"  nu_true = {syn['nu_true']}, nu_fit = {syn['nu_fit']:.3f}; "
          f"recovery_OK = {syn['recovery_OK']}  ({time.time()-t0:.1f}s)")
    print()

    print(f"# M0: reading v2 factors; building PIT table ...")
    t0 = time.time()
    pit_M0 = _pit_M0(z_full, factors_M0, cutoff, anchor_h, dims)
    chi2_M0, hist_M0 = _marginal_pit_chi2(pit_M0["u_PIT"].to_numpy())
    print(f"  rows={len(pit_M0)}  chi^2={chi2_M0:.1f}  ({time.time()-t0:.1f}s)")
    print()

    print(f"# M1: Student-t kernel MLE per day-type ...")
    t0 = time.time()
    t_kernels = _fit_t_kernels_per_daytype(z_pre, dims, cutoff)
    for dt, ker in t_kernels.items():
        if ker is None:
            print(f"  {dt}: fit failed")
        else:
            print(f"  {dt}: nu={ker['nu']:.3f}  s={ker['s']:.4f}  n={ker['n']}")
    print(f"  ({time.time()-t0:.1f}s)")
    print()

    print(f"# M1: SMC propagation for post-cutoff PIT ...")
    t0 = time.time()
    pit_M1 = _pit_M1(z_full, factors_M0, t_kernels, cutoff, anchor_h, dims,
                       rng=rng, M=args.n_particles)
    chi2_M1, hist_M1 = _marginal_pit_chi2(pit_M1["u_PIT"].to_numpy())
    print(f"  rows={len(pit_M1)}  chi^2={chi2_M1:.1f}  ({time.time()-t0:.1f}s)")
    print()

    # M2: optional (heavy)
    if args.skip_m2:
        print(f"# M2: SKIPPED")
        t_clim = None
        pit_M2 = None
        chi2_M2 = float("nan")
    else:
        print(f"# M2: Student-t climatology refit per (month, hour) ...")
        t0 = time.time()
        t_clim = _fit_t_climatology(raw_pre)
        n_bins_total = sum(1 for v in t_clim.values() if v is not None)
        nu_vals = [v["nu"] for v in t_clim.values() if v is not None]
        if nu_vals:
            print(f"  {n_bins_total} bins fit; nu range [{min(nu_vals):.2f}, {max(nu_vals):.2f}]; "
                  f"median {float(np.median(nu_vals)):.2f}  ({time.time()-t0:.1f}s)")
        else:
            print(f"  all bins failed; M2 unavailable")
        print()

        print(f"# M2: re-z under Student-t climatology, refit kernel, SMC ...")
        t0 = time.time()
        z_full_M2 = _z_under_t_climatology(raw_full, t_clim).dropna()
        z_pre_M2 = z_full_M2[z_full_M2.index <= cutoff]
        t_kernels_M2 = _fit_t_kernels_per_daytype(z_pre_M2, dims, cutoff)
        pit_M2 = _pit_M1(z_full_M2, factors_M0, t_kernels_M2, cutoff, anchor_h, dims,
                          rng=np.random.default_rng(99), M=args.n_particles)
        chi2_M2, hist_M2 = _marginal_pit_chi2(pit_M2["u_PIT"].to_numpy())
        print(f"  rows={len(pit_M2)}  chi^2={chi2_M2:.1f}  ({time.time()-t0:.1f}s)")
        print()

    # ---- Bootstrap CI on the headline numbers ----
    if args.skip_bootstrap:
        print(f"# bootstrap SKIPPED (--skip-bootstrap)")
        boot = None
    else:
        print(f"# paired-day bootstrap (n={args.n_bootstrap}) on chi^2 per model ...")
        t0 = time.time()
        # Bootstrap resamples post-cutoff delivery days; for each
        # resample, re-aggregate the PIT values from the already-
        # computed pit_M0/M1/M2 tables (no refitting; the bootstrap
        # captures resampling uncertainty in the marginal PIT
        # statistic, not refit uncertainty). The full re-fit
        # bootstrap is much more expensive; per phase_a's
        # paired_day_bootstrap convention, day-level resampling on
        # the aggregated PIT is the standard estimator.
        all_days = pd.DatetimeIndex(sorted(set(pit_M0["delivery_day"])))
        n_days = len(all_days)
        boot_chi2 = {"M0": [], "M1": [], "M2": []}
        rng_boot = np.random.default_rng(31)
        # pre-index by delivery_day for fast lookup
        grouped_M0 = {d: g for d, g in pit_M0.groupby("delivery_day", sort=False)}
        grouped_M1 = {d: g for d, g in pit_M1.groupby("delivery_day", sort=False)}
        grouped_M2 = (
            {d: g for d, g in pit_M2.groupby("delivery_day", sort=False)}
            if pit_M2 is not None else None
        )
        for b in range(args.n_bootstrap):
            idx = rng_boot.integers(0, n_days, size=n_days)
            sampled_days = all_days[idx]
            u0 = pd.concat([grouped_M0[d] for d in sampled_days if d in grouped_M0],
                            ignore_index=True)["u_PIT"].to_numpy()
            u1 = pd.concat([grouped_M1[d] for d in sampled_days if d in grouped_M1],
                            ignore_index=True)["u_PIT"].to_numpy()
            c0, _ = _marginal_pit_chi2(u0)
            c1, _ = _marginal_pit_chi2(u1)
            boot_chi2["M0"].append(c0)
            boot_chi2["M1"].append(c1)
            if grouped_M2 is not None:
                u2 = pd.concat([grouped_M2[d] for d in sampled_days if d in grouped_M2],
                                ignore_index=True)["u_PIT"].to_numpy()
                c2, _ = _marginal_pit_chi2(u2)
                boot_chi2["M2"].append(c2)
            else:
                boot_chi2["M2"].append(float("nan"))
        boot_chi2 = {k: np.array(v) for k, v in boot_chi2.items()}
        for k in ("M0", "M1", "M2"):
            arr = boot_chi2[k]
            if np.all(np.isnan(arr)):
                print(f"  chi^2 {k}: SKIPPED")
            else:
                lo, med, hi = np.nanpercentile(arr, [2.5, 50, 97.5])
                print(f"  chi^2 {k}: median {med:.1f}  CI [{lo:.1f}, {hi:.1f}]")
        boot = {"chi2": {k: v.tolist() for k, v in boot_chi2.items()}}
        print(f"  ({time.time()-t0:.1f}s)")
        print()

    print(f"# D_RATIO_t at (h={HEADLINE_H}, {HEADLINE_DAYTYPE}) under M1 ...")
    # D_RATIO_t bootstrap: resample states (post-cutoff weekday issue-
    # anchors), re-aggregate per_state_w2 list; synthetic_w2 is a
    # population constant, not re-bootstrapped (it's the noise-floor
    # unit; resampling it would conflate two sources of uncertainty).
    # Same convention as v2 and eigenmodes-v1.
    t0 = time.time()
    d_ratio_t = _compute_d_ratio_t(
        z_pre, z_full, factors_M0, t_kernels,
        cutoff, anchor_h, dims,
        h=HEADLINE_H, day_type=HEADLINE_DAYTYPE,
        M=args.n_particles, n_states=200,
        rng=np.random.default_rng(101),
    )
    print(f"  empirical_w2  = {d_ratio_t['empirical_w2']:.4e}")
    print(f"  synthetic_w2  = {d_ratio_t['synthetic_w2']:.4e}")
    print(f"  D_RATIO_t     = {d_ratio_t['D_RATIO_t']:.3f}")
    print(f"  ({time.time()-t0:.1f}s)")
    print()

    # ---- D_RATIO_t bootstrap CI ----
    if not args.skip_bootstrap:
        print(f"# D_RATIO_t paired-day bootstrap (n={args.n_bootstrap}) ...")
        t0 = time.time()
        w2_arr = np.array(d_ratio_t["per_state_w2"])
        day_arr = pd.DatetimeIndex(d_ratio_t["per_state_day"])
        syn_w2_const = d_ratio_t["synthetic_w2"]
        all_w2_days = pd.DatetimeIndex(sorted(set(day_arr)))
        # Group W2 values by day
        per_day = {d: [] for d in all_w2_days}
        for v, d_ in zip(w2_arr, day_arr):
            per_day[d_].append(float(v))
        per_day = {d: np.array(v) for d, v in per_day.items()}
        n_days_w2 = len(all_w2_days)
        d_ratio_boot = []
        rng_b2 = np.random.default_rng(53)
        for b in range(args.n_bootstrap):
            idx = rng_b2.integers(0, n_days_w2, size=n_days_w2)
            samp_days = all_w2_days[idx]
            vals = []
            for d_ in samp_days:
                vals.extend(per_day[d_].tolist())
            if not vals:
                d_ratio_boot.append(float("nan"))
                continue
            emp_w2_b = float(np.mean(vals))
            d_ratio_boot.append(emp_w2_b / syn_w2_const if syn_w2_const > 0 else float("nan"))
        d_ratio_boot = np.array(d_ratio_boot)
        lo, med, hi = np.nanpercentile(d_ratio_boot, [2.5, 50, 97.5])
        print(f"  D_RATIO_t: median {med:.3f}  CI [{lo:.3f}, {hi:.3f}]  ({time.time()-t0:.1f}s)")
        if boot is None:
            boot = {}
        boot["D_RATIO_t"] = d_ratio_boot.tolist()
        print()

    # ---- Outcome ----
    outcome = _evaluate_outcome(chi2_M0, chi2_M1, chi2_M2, t_kernels,
                                  t_clim if not args.skip_m2 else None)
    print(f"# pre-registered outcome: {outcome['verdict']}")
    print(f"  reason: {outcome['reason']}")
    print()

    # ---- Pickle ----
    out = args.out
    if not out.is_absolute():
        out = (Path.cwd() / out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "chi2_M0": chi2_M0,
        "chi2_M1": chi2_M1,
        "chi2_M2": chi2_M2,
        "hist_M0": list(hist_M0.astype(int)),
        "hist_M1": list(hist_M1.astype(int)),
        "hist_M2": list(hist_M2.astype(int)) if not args.skip_m2 else None,
        "t_kernels_M1": t_kernels,
        "t_climatology_M2": t_clim,
        "pit_M0": pit_M0,
        "pit_M1": pit_M1,
        "pit_M2": pit_M2,
        "d_ratio_t": d_ratio_t,
        "synthetic_gate": syn,
        "bootstrap": boot,
        "outcome": outcome,
        "config": {
            "n_particles": args.n_particles,
            "n_bootstrap": args.n_bootstrap if not args.skip_bootstrap else 0,
            "skipped_m2": args.skip_m2,
            "headline_h": HEADLINE_H,
            "headline_daytype": HEADLINE_DAYTYPE,
            "nu_bounds": NU_BOUNDS,
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
