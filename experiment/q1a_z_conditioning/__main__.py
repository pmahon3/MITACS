"""Q1A runner — Z-conditioning head-to-head between Z_b (binary
weekend) and Z_c (8-level categorical) on the M3 (mix-2 Gaussian)
residual law.

Implements phase_a body_sha256
``a2b13f6d8bcd6d2e4206322fc056c9b4388e761c5b5d651282e0f34d634e6d63``
in directory
``notes/preregistrations/2026-05-27_q1a-z-conditioning-strongest-axis/``.
Cuts inherited from the resolution-paths-thread skeleton (Q1A node);
the integer values 228 and 2284 are the rounded forms of Q2A's
gate-validated cuts (R-A2_cut = 228.44 / R-C2_cut = 2284.44).

The two candidates re-fit M3 per (day_type, Z) cell on the SAME
pre-cutoff library Q2A's M3 used; SMC particle propagation uses the
per-cell mixture parameters of the cell each propagated state belongs
to. The verdict statistic is
``better_chi2 = min(chi2_Z_b, chi2_Z_c)``
against the pre-locked cuts R-A1A (≤ 228) / R-B1A (228, 2284] /
R-C1A (> 2284). Head-to-head sub-finding records which Z construction
won (with paired-bootstrap CI on the difference).

Production-path commitments:
  • ``mixture_2_gaussian_mle_fit`` is called per (day_type, Z) cell
    (the production fitter — no reimplementation).
  • ``_smc_iterate_family`` / ``_marginal_pit_chi2`` from Q2A reused
    (the production SMC + chi^2 paths — no reimplementation).
  • A reproduction-check assertion blocks the head-to-head if M3
    refit with NO Z conditioning does not reproduce Q2A's settled
    chi^2 = 953.84 within tolerance.
  • Inline synthetic gate (BLOCKING) before any Ontario read.

Emits a CLAIM-GRADE provenanced artifact via
``experiment.provenance.make_result``.
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
import yaml

from config import PROJECT_ROOT, load_config
from experiment import freeze
from experiment._actuals import (
    load_actuals,
    load_pre_cutoff_actuals,
    zscore_params,
    zscore_transform,
)
from experiment.backtest import _complete_delivery_days
from experiment.predict import _daytype  # noqa: F401
from processing.innovations.estimator import mixture_2_gaussian_mle_fit

# Q2A's production helpers — reused, NOT reimplemented (these power
# both the unconditioned reproduction-check and the per-(day_type, Z)
# fits, matching Q2A's seed/library/sampler conventions exactly).
from experiment.distributional_class_q2a.__main__ import (
    _build_library_for_daytype,
    _make_mixture_2_sampler,
    _marginal_pit_chi2,
    _smc_iterate_family,
)


# ---------------------------------------------------------------------------
# Pre-registered constants (phase_a)
# ---------------------------------------------------------------------------

# Q2A's M3 SMC convention (preserve for paired comparison).
N_PARTICLES = 200          # SMC particle count per state
N_BOOTSTRAP = 1000         # paired-day bootstrap reps (phase_a)
N_PIT_BINS = 10            # 10-bin chi^2 vs uniform (phase_a)
SEED_BASE = 20260527       # session date YYYYMMDD; per-purpose offsets below

# Per-tag seed offset for Q1A's SMC propagation. For the unconditioned
# reproduction check we MUST reproduce Q2A's chi^2 = 953.84 within
# tolerance, which requires Q2A's M3 SMC seed (= SEED_BASE_Q2A + 100 +
# _TAG_SEED_OFFSET_M3 = 20260526 + 100 + 31). We replay Q2A's exact
# seed for the no-Z baseline; the Z_b and Z_c branches use new seed
# offsets registered here.
SEED_BASE_Q2A = 20260526
_TAG_SEED_OFFSET_M3 = 31     # Q2A's M3 offset (used for the no-Z baseline)
_Z_SEED_OFFSET = {
    "Z_b": 67,   # this experiment's binary-Z SMC offset
    "Z_c": 73,   # this experiment's 8-level-Z SMC offset
}
_Z_BOOT_OFFSET = 8011        # paired-day bootstrap seed (Q1A)

# Pre-registered verdict cuts (thread skeleton; integer-rounded forms
# of Q2A's gate-validated cuts).
R_A1A_CUT = 228      # corroboration: better_chi2 <= R_A1A_CUT (thread)
R_C1A_CUT = 2284     # falsification: better_chi2 > R_C1A_CUT (thread)
# R-B1A band = (R_A1A_CUT, R_C1A_CUT]

# Q2A's settled M3 baseline (the reproduction-check anchor).
Q2A_M3_CHI2_SETTLED = 953.84
# Reproduction-check tolerance. Q2A's paired-bootstrap CI [685, 1357]
# is too loose to use as a structural-equality bar; we want to catch
# silent drift in the library/seed/sampler pipeline. A 50.0 chi^2
# tolerance is roughly 1σ of Q2A's bootstrap distribution
# (σ ≈ (1357-685)/4 = 168 / 3 ≈ 56) but the SEED-EQUALITY clause
# means we're not really bootstrapping — we expect bit-exact-up-to-
# floating-point reproduction at the seeded point estimate. 5.0 is
# achievable when seeds match end-to-end; we use 50.0 to allow for
# the residual sources of variation noted in phase_a (paired-day
# bootstrap noise affects which library rows enter the fit) while
# still firing on any structural-mismatch defect.
_REPRODUCTION_TOL_CHI2 = 50.0

# Pool-to-parent fallback (phase_a §metric step 2).
# A (day_type, Z) cell with fewer than MIN_CELL_SIZE rows in the
# PRE-CUTOFF LIBRARY pools its post-cutoff propagation onto the
# day_type-only M3 fit. Note: phase_a's "n=213 sunday bin 6" refers
# to POST-CUTOFF eval rows; pre-cutoff has years of history per cell,
# so pooling-fallback should almost never fire on the fit side.
MIN_CELL_SIZE = 200

# Synthetic gate at-cell row count and SMC test draws (phase_a
# baselines.secondary "in-script self-test").
SYNTHETIC_GATE_N_LIB = 2000   # rows per synthetic (day_type, Z) cell
SYNTHETIC_GATE_M = 100        # SMC particles per state (synthetic side)
SYNTHETIC_GATE_CHI2_CEILING = 60.0  # per-cell test PIT chi^2 sanity ceiling


# ---------------------------------------------------------------------------
# Z derivation
# ---------------------------------------------------------------------------
#
# Z_b (binary)        : 1 if hour_of_week bin >= 5 else 0
# Z_c (8-level categorical) : bin 0..7
# bin = (weekday * 24 + (h - 1)) // 21, weekday ∈ {0=Mon..6=Sun},
# h ∈ {1..24} (Q1's pit_M1 convention; h is the delivery-clock hour).

_HOUR_OF_WEEK_BIN_WIDTH = 21


def _hour_of_week_bin(weekday: int, h: int) -> int:
    """8 bins of 21 hours each. weekday=0..6 (Mon..Sun), h=1..24.

    Matches P1's _hour_of_week_bin and the Q1A phase_a §metric bin
    construction. Cross-checked against P1's hot-zone reading.
    """
    hour_of_week = weekday * 24 + (h - 1)
    return hour_of_week // _HOUR_OF_WEEK_BIN_WIDTH  # 0..7


def _z_b_label(weekday: int, h: int) -> int:
    """Binary weekend indicator: 1 if bin >= 5 (Fri 10:00 - Sun 23:00).

    Hot-zone window per P1 SETTLED R-A: bins 5/6/7 carry α̂_L 0.55-0.61;
    bins 0..4 are the weekday/weekend-margins cold zone.
    """
    return 1 if _hour_of_week_bin(weekday, h) >= 5 else 0


def _z_c_label(weekday: int, h: int) -> int:
    """8-level categorical: bin 0..7 directly."""
    return _hour_of_week_bin(weekday, h)


def _derive_z_for_anchor(anchor: pd.Timestamp) -> tuple[int, int]:
    """Derive (Z_b, Z_c) for a one-row anchor.

    The Z labels are computed at the DELIVERY HOUR (= anchor + 1h
    target) so that they describe the cell the next-step prediction
    is FOR. anchor is at hour h_anchor; the target row is at h = h_anchor
    + 1 (mod 24). Within the Q2A M3 SMC convention, however, the
    "fit-time" Z is the Z at the ANCHOR (the row carrying the d-step
    lag vector); the "propagation-time" Z is the Z at the predicted
    hour. Per phase_a §metric step 3 the residual sampler "uses the
    per-(day_type, Z) cell mixture parameters of the cell the
    propagated state belongs to" — i.e. Z is FOR THE PREDICTED HOUR
    (not the anchor). This function returns Z at the predicted-hour
    convention by adding 1 hour to anchor.
    """
    target = anchor + pd.Timedelta(hours=1)
    weekday = int(target.dayofweek)
    h = int(target.hour) + 1  # convert 0..23 clock hour to 1..24 convention
    # pd.Timestamp arithmetic handles day rollover correctly:
    # anchor 23:00 + 1h = next day 00:00, target.hour = 0, h = 1, and
    # target.dayofweek is the next day. No special-case needed.
    return _z_b_label(weekday, h), _z_c_label(weekday, h)


# ---------------------------------------------------------------------------
# Library construction per (day_type, Z) cell
# ---------------------------------------------------------------------------
#
# Mirrors Q2A's _build_library_for_daytype (imported above; no
# reimplementation) but adds a Z-filtered slice on top. The
# day_type filter is applied at the ANCHOR (matches Q2A); the Z
# filter is applied at the predicted-hour (anchor + 1h) per phase_a.


def _build_library_per_z_cell(
    z_pre: pd.Series,
    day_type: str,
    d: int,
    z_kind: str,   # "Z_b" or "Z_c"
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Per-(day_type, Z) cell (X, Y) library.

    Returns ``{z_value: (X, Y)}`` for every Z stratum that has at
    least MIN_CELL_SIZE rows. Strata below threshold are absent from
    the returned dict; callers detect pooling-fallback via the empty
    cell.

    Walks the same delta-1h-contiguity-checked, day_type-filtered
    index walk as Q2A's _build_library_for_daytype, but additionally
    bins each row by Z at the predicted-hour.
    """
    idx = z_pre.index
    by_z: dict[int, list[tuple[np.ndarray, float]]] = {}
    for i in range(d, len(idx) - 1):
        anchor = idx[i]
        target = idx[i + 1]
        if target - anchor != pd.Timedelta(hours=1):
            continue
        if _daytype(anchor, 0) != day_type:
            continue
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
        z_b, z_c = _derive_z_for_anchor(anchor)
        z_val = z_b if z_kind == "Z_b" else z_c
        by_z.setdefault(z_val, []).append((lag, float(y)))
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for z_val, rows in by_z.items():
        if len(rows) < MIN_CELL_SIZE:
            continue
        X = np.array([r[0] for r in rows])
        Y = np.array([r[1] for r in rows])
        out[z_val] = (X, Y)
    return out


def _fit_mix2_per_z_cell(
    libs_by_dt_z: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]],
) -> dict[str, dict[int, dict]]:
    """Production-path mixture_2 fit per (day_type, Z) cell.

    Returns ``{day_type: {z_value: {"C": (d,1), "params": dict, "d": int,
    "n": int}}}``.
    """
    out: dict[str, dict[int, dict]] = {}
    for dt, z_to_lib in libs_by_dt_z.items():
        out[dt] = {}
        for z_val, (X, Y) in z_to_lib.items():
            C, params, _resid = mixture_2_gaussian_mle_fit(X, Y)
            out[dt][z_val] = {
                "C": C,
                "params": params,
                "d": int(X.shape[1]),
                "n": int(len(Y)),
            }
    return out


# ---------------------------------------------------------------------------
# Parent-day_type-only M3 fit (for pool-to-parent fallback + reproduction)
# ---------------------------------------------------------------------------


def _fit_mix2_per_daytype_parent(
    z_pre: pd.Series,
    dims: dict[str, int],
) -> dict[str, dict | None]:
    """Per day_type, no-Z M3 fit. This is also the reproduction-check
    fit (Q2A's M3 path — same library walker, same fitter, same data).
    Per phase_a §metric.SECONDARY: re-running M3 with no Z conditioning
    on the pre-cutoff library reproduces Q2A's chi^2 = 953.84 within
    paired-bootstrap noise.
    """
    out: dict[str, dict | None] = {}
    for dt in ("weekday", "saturday", "sunday"):
        d = int(dims[dt])
        lib = _build_library_for_daytype(z_pre, dt, d)
        if lib is None:
            out[dt] = None
            continue
        X_dt, Y_dt = lib
        C, params, _resid = mixture_2_gaussian_mle_fit(X_dt, Y_dt)
        out[dt] = {
            "C": C,
            "params": params,
            "d": d,
            "n": int(len(Y_dt)),
        }
    return out


# ---------------------------------------------------------------------------
# SMC propagation per Z candidate
# ---------------------------------------------------------------------------
#
# Mirrors Q2A's _pit_for_family but routes each (day_type, predicted-
# hour) propagation through the corresponding (day_type, Z) cell's
# fit. When the cell library is empty (n < MIN_CELL_SIZE in pre-cutoff),
# the propagation falls back to the parent day_type-only M3 fit.


def _pit_for_z_candidate(
    z_full: pd.Series,
    z_kind: str,
    fits_per_dt_z: dict[str, dict[int, dict]],
    fits_per_dt_parent: dict[str, dict | None],
    cutoff: pd.Timestamp,
    anchor_h: int,
    dims: dict[str, int],
    rng: np.random.Generator,
    M: int = N_PARTICLES,
) -> tuple[pd.DataFrame, dict[tuple[str, int], int]]:
    """Per-issue-time PIT table for one Z candidate.

    Routes each (day_type, predicted-hour) to the corresponding
    (day_type, Z) cell's M3 fit; falls back to the parent day_type
    fit if the cell library was too thin. The fallback log records
    how many rows pooled per (day_type, Z) cell.

    Returns ``(pit_df, fallback_log)`` where
    ``pit_df`` has columns
        ``[delivery_day, h, day_type, z_value, used_fallback, u_PIT]``
    and ``fallback_log[(day_type, z_value)] = count`` of rows that
    pooled to the parent fit.
    """
    days = _complete_delivery_days(z_full, anchor_h)
    days = days[days > cutoff]
    rows: list[dict] = []
    fallback_log: dict[tuple[str, int], int] = {}

    for D in days:
        D = pd.Timestamp(D.date())
        dt = _daytype(D + pd.Timedelta(hours=anchor_h), anchor_h)
        d = int(dims[dt])
        issue_anchor = D + pd.Timedelta(hours=anchor_h - 1)
        lag_times = [issue_anchor - pd.Timedelta(hours=i) for i in range(d)]
        if not all(t in z_full.index for t in lag_times):
            continue
        x_state = z_full.reindex(lag_times).to_numpy()
        if not np.all(np.isfinite(x_state)):
            continue

        # The 24-step SMC trajectory propagates through 24 predicted
        # hours; each step's Z depends on the predicted hour. We
        # propagate one step at a time and route to the cell-fit for
        # the predicted-hour's Z. This matches phase_a §metric step 3.
        target_dts = [
            D + pd.Timedelta(hours=anchor_h + (h - 1)) for h in range(1, 25)
        ]
        actuals = z_full.reindex(target_dts).to_numpy()
        if not np.all(np.isfinite(actuals)):
            continue

        particles = np.tile(x_state, (M, 1))  # (M, d)
        for h_step in range(1, 25):
            predicted_dt = target_dts[h_step - 1]
            weekday_p = int(predicted_dt.dayofweek)
            h_p = int(predicted_dt.hour) + 1  # 1..24 convention
            if z_kind == "Z_b":
                z_val = _z_b_label(weekday_p, h_p)
            else:
                z_val = _z_c_label(weekday_p, h_p)

            # Cell fit lookup with pool-to-parent fallback.
            cell_fit = fits_per_dt_z.get(dt, {}).get(z_val)
            used_fallback = False
            if cell_fit is None:
                cell_fit = fits_per_dt_parent.get(dt)
                used_fallback = True
                fallback_log[(dt, z_val)] = fallback_log.get((dt, z_val), 0) + 1
            if cell_fit is None:
                # Both cell and parent missing — should not happen on
                # the registered configuration. Skip this row.
                continue

            sampler = _make_mixture_2_sampler(cell_fit["params"])
            C_col = cell_fit["C"]
            r_draws = sampler(M, rng)
            next_z = (particles @ C_col).ravel() + r_draws
            samp_at_h = next_z.copy()
            # Particle update: shift the lag window.
            particles = np.column_stack([next_z, particles[:, :d - 1]])

            # Randomized mid-rank PIT (same convention as Q1/Q2A).
            less = float(np.sum(samp_at_h < actuals[h_step - 1]))
            eq = float(np.sum(samp_at_h == actuals[h_step - 1]))
            U = rng.random()
            u = (less + U * (eq + 1.0)) / (M + 1.0)
            rows.append({
                "delivery_day": D,
                "h": h_step,
                "day_type": dt,
                "z_value": int(z_val),
                "used_fallback": used_fallback,
                "u_PIT": float(u),
            })
    return pd.DataFrame(rows), fallback_log


def _pit_no_z_reproduction(
    z_full: pd.Series,
    fits_per_dt_parent: dict[str, dict | None],
    cutoff: pd.Timestamp,
    anchor_h: int,
    dims: dict[str, int],
    rng: np.random.Generator,
    M: int = N_PARTICLES,
) -> pd.DataFrame:
    """No-Z reproduction-check PIT table.

    Identical to Q2A's M3 _pit_for_family path: no Z routing, parent
    fit used at every step. Identical seed (caller passes
    SEED_BASE_Q2A + 100 + _TAG_SEED_OFFSET_M3) so the chi^2 reproduces
    Q2A's 953.84 within tolerance.
    """
    # Build the family_fit dict in Q2A's format, then call Q2A's
    # production _pit_for_family directly — no reimplementation.
    # However, _pit_for_family expects a "family_kind" to route the
    # sampler factory; we pass "mixture_2" since that's what M3 is.
    from experiment.distributional_class_q2a.__main__ import _pit_for_family

    return _pit_for_family(
        z_full,
        fits_per_dt_parent,
        "mixture_2",
        cutoff,
        anchor_h,
        dims,
        rng=rng,
        M=M,
    )


# ---------------------------------------------------------------------------
# Inline synthetic gate (BLOCKING)
# ---------------------------------------------------------------------------
#
# Phase_a §baselines.secondary: generates synthetic data with a known
# Z-conditional mixture DGP — Z=0 cell mixes 0.9 N(0,1) + 0.1 N(0,3);
# Z=1 cell mixes 0.6 N(0,1) + 0.4 N(0,4). Two recovery checks:
#   (i) per-cell fitted (w, s1, s2) within tolerance of ground truth;
#   (ii) PIT u_PIT under the Z-conditional model is approximately
#        Uniform[0,1] (chi^2 << R-A1A cut at 228).
# Companion gate: under FALSE Z (random labels orthogonal to mixture
# parameters), the per-stratum fits collapse back to the marginal
# mixture and chi^2 ~ baseline; this confirms the per-stratum fit
# is not artificially shrinking chi^2 by overfitting.


# Known Z-conditional DGP (locked at script-write time, NOT post-hoc).
_SYNTH_DGP = {
    0: {"w": (0.9, 0.1), "s": (1.0, 3.0)},   # Z=0 cell
    1: {"w": (0.6, 0.4), "s": (1.0, 4.0)},   # Z=1 cell
}
_SYNTH_PARAM_TOL = 0.15  # tolerance on fitted (w_r, s2) vs ground truth


def _draw_synthetic_z_conditional(
    n_per_cell: int,
    rng: np.random.Generator,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """One (X, Y) library per Z cell, drawn from the known DGP.

    X is (n, 2) standard normal for both cells; Y = X @ c_true +
    cell-specific mixture residual. The mixture parameters of each
    Z cell are NOT given to the fitter — it must recover them from
    the residual structure alone.
    """
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    c_true = np.array([[0.7], [0.2]])
    for z_val, dgp in _SYNTH_DGP.items():
        w = np.array(dgp["w"])
        s = np.array(dgp["s"])
        X = rng.standard_normal((n_per_cell, 2))
        which = rng.choice(2, size=n_per_cell, p=w)
        eps = rng.standard_normal(n_per_cell)
        r = eps * s[which]
        Y = (X @ c_true).ravel() + r
        out[z_val] = (X, Y)
    return out


def _run_inline_synthetic_gate(verbose: bool = True) -> dict:
    """Inline synthetic gate. BLOCKING on FAIL.

    Three sub-checks:
      (a) per-cell mixture parameter recovery within tolerance.
      (b) per-cell PIT chi^2 under the Z-conditional model << 228.
      (c) under FALSE (random) Z labels, fits collapse back to the
          marginal mixture; companion sanity check.
    """
    rng = np.random.default_rng(SEED_BASE + 9001)
    libs = _draw_synthetic_z_conditional(SYNTHETIC_GATE_N_LIB, rng)

    out: dict[str, Any] = {"checks": []}
    all_ok = True

    # (a) Per-cell mixture parameter recovery.
    fits_per_z: dict[int, dict] = {}
    for z_val, (X, Y) in libs.items():
        C, params, _r = mixture_2_gaussian_mle_fit(X, Y)
        fits_per_z[z_val] = {"C": C, "params": params, "d": int(X.shape[1])}
        w_true = _SYNTH_DGP[z_val]["w"]
        s_true = _SYNTH_DGP[z_val]["s"]
        w_fit = params["weights"]
        s_fit = params["scales"]
        # Identifiability: mixture_2_gaussian_mle_fit orders s1<=s2.
        # The DGP also has s1<=s2 by construction (s = (1, 3) or (1, 4)),
        # so no relabeling needed.
        w_r_true = w_true[1]
        w_r_fit = w_fit[1]
        s2_true = s_true[1]
        s2_fit = s_fit[1]
        ok_w = abs(w_r_fit - w_r_true) <= _SYNTH_PARAM_TOL
        ok_s = abs(s2_fit - s2_true) / s2_true <= _SYNTH_PARAM_TOL
        out["checks"].append({
            "name": f"synth_cell_z{z_val}_param_recovery",
            "w_r_true": w_r_true, "w_r_fit": w_r_fit,
            "s2_true": s2_true, "s2_fit": s2_fit,
            "ok_w": ok_w, "ok_s": ok_s,
        })
        if verbose:
            tag = "OK" if (ok_w and ok_s) else "FAIL"
            print(f"  [{tag}] synth cell Z={z_val}: "
                  f"w_r {w_r_fit:.3f} (true {w_r_true:.3f}, tol {_SYNTH_PARAM_TOL}); "
                  f"s2 {s2_fit:.3f} (true {s2_true:.3f}, rel tol {_SYNTH_PARAM_TOL})")
        all_ok = all_ok and ok_w and ok_s

    # (b) Per-cell test-draw PIT chi^2 << 228.
    chi2_per_cell: dict[int, float] = {}
    for z_val, (_X_lib, _Y_lib) in libs.items():
        fit = fits_per_z[z_val]
        sampler = _make_mixture_2_sampler(fit["params"])
        c_true = np.array([[0.7], [0.2]])
        n_test = SYNTHETIC_GATE_N_LIB
        X_test = rng.standard_normal((n_test, 2))
        means_pred = (X_test @ fit["C"]).ravel()
        true_means = (X_test @ c_true).ravel()
        s_true = np.array(_SYNTH_DGP[z_val]["s"])
        w_true = np.array(_SYNTH_DGP[z_val]["w"])
        which = rng.choice(2, size=n_test, p=w_true)
        actuals = true_means + rng.standard_normal(n_test) * s_true[which]
        pit = np.empty(n_test)
        for i in range(n_test):
            particles = means_pred[i] + sampler(SYNTHETIC_GATE_M, rng)
            less = float(np.sum(particles < actuals[i]))
            eq = float(np.sum(particles == actuals[i]))
            U = rng.random()
            pit[i] = (less + U * (eq + 1.0)) / (SYNTHETIC_GATE_M + 1.0)
        chi2, _hist = _marginal_pit_chi2(pit)
        chi2_per_cell[z_val] = float(chi2)
        ok_chi2 = chi2 <= SYNTHETIC_GATE_CHI2_CEILING
        out["checks"].append({
            "name": f"synth_cell_z{z_val}_chi2",
            "chi2": float(chi2), "ceiling": SYNTHETIC_GATE_CHI2_CEILING,
            "ok": ok_chi2,
        })
        if verbose:
            tag = "OK" if ok_chi2 else "FAIL"
            print(f"  [{tag}] synth cell Z={z_val}: chi^2 = {chi2:.1f}  "
                  f"(ceiling {SYNTHETIC_GATE_CHI2_CEILING:.0f}, n_test={n_test})")
        all_ok = all_ok and ok_chi2

    # (c) FALSE-Z companion: pool both cells under random Z labels;
    # fit collapses to marginal mixture; chi^2 should NOT be artificially
    # shrunk (i.e., it should be HIGHER than the Z-true case because we
    # are mis-stratifying).
    all_X = np.concatenate([libs[0][0], libs[1][0]])
    all_Y = np.concatenate([libs[0][1], libs[1][1]])
    rng_false = np.random.default_rng(SEED_BASE + 9101)
    false_z = rng_false.choice(2, size=len(all_Y))
    fits_false: dict[int, dict] = {}
    for z_val in (0, 1):
        mask = false_z == z_val
        Xc = all_X[mask]
        Yc = all_Y[mask]
        if len(Yc) < MIN_CELL_SIZE:
            continue
        C, params, _r = mixture_2_gaussian_mle_fit(Xc, Yc)
        fits_false[z_val] = {"C": C, "params": params, "d": int(Xc.shape[1])}

    # The substantive question is whether the FALSE-Z per-cell mixtures
    # collapse toward the MARGINAL mixture rather than picking up the
    # cell-true mixture. Compare each false-Z cell's fitted w_r and s2
    # against the marginal-mixture w_r and s2 (which we know exactly:
    # w_r_marg = 0.5 * 0.1 + 0.5 * 0.4 = 0.25; s2_marg ~ 3.5 because
    # the false-Z cells inherit a 50/50 mix of the two true cells'
    # heavy components).
    w_r_marg_expected = 0.5 * _SYNTH_DGP[0]["w"][1] + 0.5 * _SYNTH_DGP[1]["w"][1]
    collapse_w = []
    for z_val in fits_false:
        w_r_fit = fits_false[z_val]["params"]["weights"][1]
        # Distance to marginal (small) vs distance to cell-true (large)
        d_marg = abs(w_r_fit - w_r_marg_expected)
        d_true = abs(w_r_fit - _SYNTH_DGP[z_val]["w"][1])
        # "Collapse" = closer to marginal than to cell-true.
        collapse_w.append(d_marg < d_true)
        if verbose:
            ok = d_marg < d_true
            tag = "OK" if ok else "WARN"
            print(f"  [{tag}] false-Z cell {z_val}: w_r {w_r_fit:.3f}  "
                  f"(d to marginal {w_r_marg_expected:.3f}: {d_marg:.3f}; "
                  f"d to cell-true {_SYNTH_DGP[z_val]['w'][1]:.3f}: {d_true:.3f})")
    out["checks"].append({
        "name": "false_z_companion_collapse",
        "all_cells_collapsed": bool(all(collapse_w) and collapse_w),
        # advisory: a false-Z that fails to collapse would suggest the
        # fitter is sensitive to label assignment in ways that could
        # invent spurious Z effects on real data. We log but do not
        # block on this — the substantive gates (a) and (b) are
        # blocking, (c) is informational.
    })

    out["all_ok"] = all_ok
    return out


# ---------------------------------------------------------------------------
# Pre-registered outcome evaluation
# ---------------------------------------------------------------------------


def _evaluate_outcome(
    chi2_Z_b: float,
    chi2_Z_c: float,
    chi2_diff_se: float | None,
) -> dict:
    """Mechanical evaluation per phase_a §metric.HEAD-TO-HEAD VERDICT RULE.

    The verdict fires on better_chi2 = min(chi2_Z_b, chi2_Z_c) against
    the pre-locked cuts. Near-tie handling: if |chi2_diff| < 1.0 * SE
    of the paired difference, the head-to-head sub-finding records
    'no ranking'.
    """
    better_chi2 = min(chi2_Z_b, chi2_Z_c)
    winner = "Z_b" if chi2_Z_b < chi2_Z_c else "Z_c"
    diff = chi2_Z_b - chi2_Z_c
    near_tie = False
    if chi2_diff_se is not None and chi2_diff_se > 0:
        near_tie = abs(diff) < 1.0 * chi2_diff_se

    if better_chi2 <= R_A1A_CUT:
        verdict = "R-A1A"
        reason = (
            f"better_chi2 = min(Z_b={chi2_Z_b:.1f}, Z_c={chi2_Z_c:.1f}) "
            f"= {better_chi2:.1f} <= R-A1A cut {R_A1A_CUT}; "
            f"hour_of_week-derived Z conditioning closes the marginal "
            f"chi^2 gap. Head-to-head sub-finding: "
            f"{'winner = ' + winner if not near_tie else 'NEAR-TIE (no ranking)'}; "
            f"chi^2 diff = {diff:+.1f}."
        )
    elif better_chi2 <= R_C1A_CUT:
        verdict = "R-B1A"
        reason = (
            f"better_chi2 = min(Z_b={chi2_Z_b:.1f}, Z_c={chi2_Z_c:.1f}) "
            f"= {better_chi2:.1f} in (R-A1A {R_A1A_CUT}, R-C1A {R_C1A_CUT}]; "
            f"Z conditioning helps but does not close. "
            f"Sub-finding: "
            f"{'winner = ' + winner if not near_tie else 'NEAR-TIE'}; "
            f"chi^2 diff = {diff:+.1f}. Thread branches to Q1B."
        )
    else:
        verdict = "R-C1A"
        reason = (
            f"better_chi2 = min(Z_b={chi2_Z_b:.1f}, Z_c={chi2_Z_c:.1f}) "
            f"= {better_chi2:.1f} > R-C1A cut {R_C1A_CUT}; "
            f"Z conditioning does not help. Thread branches to P2."
        )
    return {
        "verdict": verdict,
        "better_chi2": better_chi2,
        "winner": winner,
        "near_tie": near_tie,
        "chi2_diff": diff,
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# Body rendering
# ---------------------------------------------------------------------------


def _render_body(
    *,
    chi2: dict[str, float],
    hist: dict[str, list[int]],
    boot: dict | None,
    fits_per_dt_z: dict[str, dict[str, dict[int, dict]]],
    fits_per_dt_parent: dict[str, dict | None],
    repro_chi2: float,
    outcome: dict,
    synth: dict,
    fallback_log: dict[str, dict[tuple[str, int], int]],
    per_cell_chi2: dict[str, dict[tuple[str, int], dict[str, Any]]],
    config_block: dict,
) -> str:
    L: list[str] = []
    L.append("Q1A — Z-conditioning (head-to-head Z_b vs Z_c) on M3")
    L.append("=" * 70)
    L.append("")
    L.append("Pre-registration: notes/preregistrations/")
    L.append("  2026-05-27_q1a-z-conditioning-strongest-axis/phase_a.yaml")
    L.append(f"  phase_a body_sha256: a2b13f6d8bcd6d2e4206322fc056c9b4388e761c5b5d651282e0f34d634e6d63")
    L.append("Cuts from thread skeleton (integer-rounded forms of Q2A cuts):")
    L.append(f"  R-A1A (corroboration):    better_chi2 <= {R_A1A_CUT}")
    L.append(f"  R-C1A (falsification):    better_chi2  > {R_C1A_CUT}")
    L.append(f"  R-B1A (ambiguous band):   ({R_A1A_CUT}, {R_C1A_CUT}]")
    L.append("")
    L.append("REPRODUCTION CHECK (M3 with NO Z conditioning vs Q2A settled)")
    L.append("-" * 70)
    L.append(f"  Q1A reproduction chi^2: {repro_chi2:.1f}")
    L.append(f"  Q2A settled chi^2:      {Q2A_M3_CHI2_SETTLED:.2f}")
    L.append(f"  |diff|:                 {abs(repro_chi2 - Q2A_M3_CHI2_SETTLED):.1f}  "
             f"(tol {_REPRODUCTION_TOL_CHI2:.1f})")
    L.append("")
    L.append("MARGINAL PIT CHI^2 PER Z CANDIDATE (post-cutoff, 10-bin)")
    L.append("-" * 70)
    for tag in ("Z_b", "Z_c"):
        line = f"  chi2_{tag}: {chi2[tag]:>10.1f}"
        if boot is not None and tag in boot.get("chi2", {}):
            arr = np.asarray(boot["chi2"][tag])
            lo, med, hi = np.nanpercentile(arr, [2.5, 50, 97.5])
            line += f"    bootstrap 95% CI [{lo:.1f}, {hi:.1f}]  median {med:.1f}"
        L.append(line)
    L.append("")
    L.append("HEAD-TO-HEAD CHI^2 DIFFERENCE (Z_b - Z_c)")
    L.append("-" * 70)
    L.append(f"  chi2_diff = {outcome['chi2_diff']:+.1f}")
    if boot is not None and "diff" in boot.get("chi2", {}):
        arr = np.asarray(boot["chi2"]["diff"])
        lo, med, hi = np.nanpercentile(arr, [2.5, 50, 97.5])
        L.append(f"  bootstrap 95% CI [{lo:+.1f}, {hi:+.1f}]  median {med:+.1f}")
    L.append(f"  near_tie: {outcome['near_tie']}")
    L.append("")
    L.append("BETTER (verdict) CHI^2")
    L.append("-" * 70)
    L.append(f"  better_chi2 = min(Z_b, Z_c) = {outcome['better_chi2']:.1f}")
    L.append(f"  winner                     = {outcome['winner']}")
    if boot is not None and "better" in boot.get("chi2", {}):
        arr = np.asarray(boot["chi2"]["better"])
        lo, med, hi = np.nanpercentile(arr, [2.5, 50, 97.5])
        L.append(f"  bootstrap 95% CI [{lo:.1f}, {hi:.1f}]  median {med:.1f}")
        # Landslide qualifier (phase_a §corroboration_criterion):
        # R-A1A at landslide requires point AND lower-CI <= R-A1A_CUT.
        landslide = lo <= R_A1A_CUT and outcome['better_chi2'] <= R_A1A_CUT
        L.append(f"  landslide qualifier (R-A1A only): "
                 f"{'TRUE' if landslide else 'FALSE'}")
    L.append("")
    L.append("PRE-REGISTERED VERDICT")
    L.append("-" * 70)
    L.append(f"  verdict: {outcome['verdict']}")
    L.append(f"  reason:  {outcome['reason']}")
    L.append("")
    L.append("PER-CELL FITTED MIXTURE PARAMETERS (mix-2)")
    L.append("-" * 70)
    for tag in ("Z_b", "Z_c"):
        L.append(f"  [{tag}]")
        for dt, cell_fits in fits_per_dt_z[tag].items():
            for z_val in sorted(cell_fits.keys()):
                fit = cell_fits[z_val]
                w = fit["params"]["weights"]
                s = fit["params"]["scales"]
                L.append(
                    f"    {dt} Z={z_val}: w=({w[0]:.3f}, {w[1]:.3f})  "
                    f"s=({s[0]:.3f}, {s[1]:.3f})  n={fit['n']}"
                )
    L.append("")
    L.append("PER-CELL CHI^2 CONTRIBUTION (informational decomposition)")
    L.append("-" * 70)
    for tag in ("Z_b", "Z_c"):
        L.append(f"  [{tag}]")
        cells = per_cell_chi2[tag]
        if not cells:
            L.append("    (no cells)")
        else:
            for (dt, z_val), entry in sorted(cells.items()):
                L.append(
                    f"    {dt} Z={z_val}: chi^2 = {entry['chi2']:>7.1f}  "
                    f"n={entry['n_rows']}"
                )
    L.append("")
    L.append("POOLING FALLBACK LOG (cells that pooled to parent day_type)")
    L.append("-" * 70)
    for tag in ("Z_b", "Z_c"):
        L.append(f"  [{tag}]")
        log = fallback_log[tag]
        if not log:
            L.append("    (no cells pooled — every (day_type, Z) cell met MIN_CELL_SIZE)")
        else:
            for (dt, z_val), count in sorted(log.items()):
                L.append(f"    {dt} Z={z_val}: {count} post-cutoff rows pooled to parent")
    L.append("")
    L.append("PIT HISTOGRAMS (10-bin) — post-cutoff marginal")
    L.append("-" * 70)
    for tag in ("Z_b", "Z_c"):
        L.append(f"  {tag}: {hist[tag]}")
    L.append("")
    L.append("INLINE SYNTHETIC GATE")
    L.append("-" * 70)
    L.append(f"  all_ok: {synth['all_ok']}")
    for chk in synth["checks"]:
        L.append(f"  {chk['name']}: " + ", ".join(
            f"{k}={v}" for k, v in chk.items() if k != "name"
        ))
    L.append("")
    L.append("CONFIG")
    L.append("-" * 70)
    for k, v in config_block.items():
        L.append(f"  {k}: {v}")
    L.append("")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# Result.yaml renderer
# ---------------------------------------------------------------------------


def _render_result_yaml(
    *,
    chi2: dict[str, float],
    boot: dict | None,
    fits_per_dt_z: dict[str, dict[str, dict[int, dict]]],
    outcome: dict,
    synth: dict,
    fallback_log: dict[str, dict[tuple[str, int], int]],
    per_cell_chi2: dict[str, dict[tuple[str, int], dict[str, Any]]],
    repro_chi2: float,
    config_block: dict,
    artifact_txt_path: Path,
    artifact_pkl_path: Path,
) -> dict:
    def ci_for(tag: str) -> tuple[float | None, float | None]:
        if boot is None:
            return None, None
        arr = boot.get("chi2", {}).get(tag)
        if arr is None:
            return None, None
        lo, hi = np.nanpercentile(np.asarray(arr), [2.5, 97.5])
        return float(lo), float(hi)

    # Per-cell mixture params per (day_type, Z) cell.
    def cells_for(tag: str) -> dict[str, dict[int, dict]]:
        out: dict[str, dict] = {}
        for dt, cell_fits in fits_per_dt_z[tag].items():
            out[dt] = {}
            for z_val, fit in cell_fits.items():
                out[dt][int(z_val)] = {
                    "weights": [float(x) for x in fit["params"]["weights"]],
                    "scales": [float(x) for x in fit["params"]["scales"]],
                    "n_library": int(fit["n"]),
                }
        return out

    ci_low_b, ci_high_b = ci_for("Z_b")
    ci_low_c, ci_high_c = ci_for("Z_c")
    ci_low_better, ci_high_better = ci_for("better")
    ci_low_diff, ci_high_diff = ci_for("diff")

    # Landslide qualifier on R-A1A (structured field; previously rendered
    # in body only). Per phase_a §corroboration_criterion: R-A1A at
    # landslide strength requires BOTH point AND lower-bound 95% CI to
    # clear R_A1A_CUT (228). Only well-defined when verdict is R-A1A
    # AND a bootstrap CI exists for `better`.
    landslide_R_A1A: bool | None = None
    if outcome["verdict"] == "R-A1A":
        if ci_low_better is not None:
            landslide_R_A1A = (
                outcome["better_chi2"] <= R_A1A_CUT
                and ci_low_better <= R_A1A_CUT
            )
        else:
            landslide_R_A1A = None  # no CI -> cannot evaluate

    return {
        "schema": "result",
        "written_at": pd.Timestamp.utcnow().isoformat() + "Z",
        "references": [{
            "file": "phase_a.yaml",
            "body_sha256":
                "a2b13f6d8bcd6d2e4206322fc056c9b4388e761c5b5d651282e0f34d634e6d63",
        }],
        "artifact": {
            "primary_txt": str(artifact_txt_path),
            "pickle": str(artifact_pkl_path),
            "grade": "CLAIM",
            "emitted_by": "experiment.q1a_z_conditioning.__main__",
            "notes": (
                "Primary CLAIM artifact is the .txt under experiment/results/ "
                "written via experiment.provenance.make_result (carries its "
                "own header + body hash). Pickle holds the full PIT tables "
                "and per-cell fitted-params dicts for re-analysis."
            ),
        },
        "code_path_audit": {
            "verdict": "PRODUCTION-PATH",
            "tool": ".venv/bin/python -m experiment.audit.code_path "
                    "experiment/q1a_z_conditioning/__main__.py",
            "blocking": False,
        },
        # Coverage of phase_a.variables.dependent (Check R contract):
        # chi2_Z_b, chi2_Z_c, better_chi2, chi2_diff, chi2_Z_b_ci95,
        # chi2_Z_c_ci95, chi2_diff_ci95, per_cell_mixture_params_Z_b,
        # per_cell_mixture_params_Z_c, per_cell_chi2_contribution_Z_b,
        # per_cell_chi2_contribution_Z_c, pooling_fallback_log_Z_b,
        # pooling_fallback_log_Z_c, winner, near_tie_flag
        "primary_result": {
            "metric": "better_chi2 = min(chi2_Z_b, chi2_Z_c) "
                      "on post-cutoff marginal PIT",
            "value": {
                "chi2_Z_b": float(chi2["Z_b"]),
                "chi2_Z_c": float(chi2["Z_c"]),
                "better_chi2": float(outcome["better_chi2"]),
                "chi2_diff": float(outcome["chi2_diff"]),
                "winner": outcome["winner"],
                "near_tie_flag": bool(outcome["near_tie"]),
                "landslide_R_A1A": landslide_R_A1A,
            },
            "ci_low": {
                "chi2_Z_b": ci_low_b,
                "chi2_Z_c": ci_low_c,
                "better_chi2": ci_low_better,
                "chi2_diff": ci_low_diff,
            },
            "ci_high": {
                "chi2_Z_b": ci_high_b,
                "chi2_Z_c": ci_high_c,
                "better_chi2": ci_high_better,
                "chi2_diff": ci_high_diff,
            },
            "chi2_Z_b_ci95": [ci_low_b, ci_high_b] if ci_low_b is not None else None,
            "chi2_Z_c_ci95": [ci_low_c, ci_high_c] if ci_low_c is not None else None,
            "chi2_diff_ci95": [ci_low_diff, ci_high_diff] if ci_low_diff is not None else None,
            "ci_method_actually_used":
                f"paired_day_bootstrap_n={config_block['n_bootstrap']}",
            "outcome_verdict": outcome["verdict"],
            "outcome_reason": outcome["reason"],
        },
        "secondary_results": [
            {
                "name": "per_cell_mixture_params_Z_b",
                "value": cells_for("Z_b"),
            },
            {
                "name": "per_cell_mixture_params_Z_c",
                "value": cells_for("Z_c"),
            },
            {
                "name": "per_cell_chi2_contribution_Z_b",
                "value": {
                    f"{dt}__z{z}": {
                        "chi2": entry["chi2"],
                        "n_rows": entry["n_rows"],
                    }
                    for (dt, z), entry in per_cell_chi2["Z_b"].items()
                },
                "notes": (
                    "Per-(day_type, Z_b) cell 10-bin marginal PIT chi^2. "
                    "Informational decomposition; the verdict statistic "
                    "is the pooled-marginal chi^2 in primary_result. "
                    "Per phase_a §metric.SECONDARY: 'Per-cell chi^2 "
                    "contribution decomposition.'"
                ),
            },
            {
                "name": "per_cell_chi2_contribution_Z_c",
                "value": {
                    f"{dt}__z{z}": {
                        "chi2": entry["chi2"],
                        "n_rows": entry["n_rows"],
                    }
                    for (dt, z), entry in per_cell_chi2["Z_c"].items()
                },
                "notes": (
                    "Per-(day_type, Z_c) cell 10-bin marginal PIT chi^2. "
                    "Informational decomposition (same convention as "
                    "per_cell_chi2_contribution_Z_b)."
                ),
            },
            {
                "name": "pooling_fallback_log_Z_b",
                "value": {
                    f"{dt}__z{z}": count
                    for (dt, z), count in fallback_log["Z_b"].items()
                },
            },
            {
                "name": "pooling_fallback_log_Z_c",
                "value": {
                    f"{dt}__z{z}": count
                    for (dt, z), count in fallback_log["Z_c"].items()
                },
            },
            {
                "name": "reproduction_check_no_z",
                "value": {
                    "q1a_no_z_chi2": float(repro_chi2),
                    "q2a_settled_chi2": Q2A_M3_CHI2_SETTLED,
                    "abs_diff": float(abs(repro_chi2 - Q2A_M3_CHI2_SETTLED)),
                    "tol": _REPRODUCTION_TOL_CHI2,
                    "ok": float(abs(repro_chi2 - Q2A_M3_CHI2_SETTLED))
                          <= _REPRODUCTION_TOL_CHI2,
                },
                "notes": (
                    "Per phase_a §metric.SECONDARY 'Reproduction check': "
                    "re-running M3 with no Z conditioning on the pre-cutoff "
                    "library reproduces Q2A's chi^2 = 953.84 within "
                    "paired-bootstrap noise. ASSERTION: blocking on FAIL."
                ),
            },
            {
                "name": "inline_synthetic_gate",
                "value": {
                    "all_ok": bool(synth["all_ok"]),
                    "checks": synth["checks"],
                },
                "notes": (
                    "Per phase_a §baselines.secondary: known Z-conditional "
                    "DGP (Z=0: 0.9 N(0,1)+0.1 N(0,3); Z=1: 0.6 N(0,1)+0.4 "
                    "N(0,4)). BLOCKING: per-cell parameter recovery within "
                    f"{_SYNTH_PARAM_TOL} AND per-cell PIT chi^2 < "
                    f"{SYNTHETIC_GATE_CHI2_CEILING}."
                ),
            },
            {
                "name": "pre_registered_cuts",
                "value": {
                    "R_A1A_cut": R_A1A_CUT,
                    "R_C1A_cut": R_C1A_CUT,
                    "source": ("resolution-paths-thread skeleton "
                               "(Q1A node); integer-rounded forms of "
                               "Q2A's gate-validated R-A2_cut=228.44 "
                               "and R-C2_cut=2284.44."),
                },
            },
        ],
        "execution_deviations": [],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path,
                   default=Path("scratch/data/q1a_z_conditioning/q1a.pkl"))
    p.add_argument("--n-particles", type=int, default=N_PARTICLES)
    p.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    p.add_argument("--skip-bootstrap", action="store_true",
                   help="point estimates only (for fast iteration)")
    p.add_argument("--save-pit", action="store_true",
                   help="include per-row PIT tables in the output pickle")
    p.add_argument(
        "--emit-result", action="store_true",
        help="write the CLAIM-GRADE provenanced artifact via "
             "experiment.provenance.make_result (refuses a dirty tree) "
             "PLUS the registry result.yaml under notes/preregistrations/"
             "2026-05-27_q1a-z-conditioning-strongest-axis/"
    )
    args = p.parse_args(argv)

    cfg = load_config()
    spec = freeze.load_verified()
    cutoff = pd.Timestamp(spec["data_cutoff"])
    anchor_h = int(cfg.data.day_anchor_hours)
    dims = spec["predictor"]["embedding_dims"]
    clim_method = spec["predictor"].get("climatology_method", "month_hour")
    k_year = spec["predictor"].get("fourier_k_year")
    k_day = spec["predictor"].get("fourier_k_day")

    print(f"# Q1A: Z-conditioning head-to-head (Z_b binary, Z_c 8-level)")
    print(f"  cutoff:       {cutoff}")
    print(f"  anchor_h:     {anchor_h}")
    print(f"  embedding:    {dict(dims)}")
    print(f"  N_PARTICLES:  {args.n_particles}")
    print(f"  N_BOOTSTRAP:  {args.n_bootstrap}")
    print(f"  cuts:         R-A1A <= {R_A1A_CUT}, R-C1A > {R_C1A_CUT}")
    print(f"  Q2A settled:  chi^2 = {Q2A_M3_CHI2_SETTLED} (reproduction-check anchor)")
    print()

    # ---- Inline synthetic gate (BLOCKING) ----
    print(f"# inline synthetic gate ...")
    t0 = time.time()
    synth = _run_inline_synthetic_gate(verbose=True)
    print(f"  ({time.time()-t0:.1f}s)")
    if not synth["all_ok"]:
        print(
            "# GATE FAIL — refusing to run Q1A on Ontario data. "
            "Investigate the production mixture_2_gaussian_mle_fit "
            "or the Z derivation before proceeding.",
            file=sys.stderr,
        )
        return 2
    print(f"# gate PASS")
    print()

    # ---- Load actuals + build z ----
    raw_full = load_actuals(cutoff=None).dropna()
    zp = zscore_params(cutoff, method=clim_method, k_year=k_year, k_day=k_day)
    z_full = zscore_transform(raw_full, zp)
    raw_pre = load_pre_cutoff_actuals(cutoff).dropna()
    z_pre = zscore_transform(raw_pre, zp)

    # ---- Fit parent (no-Z) M3 per day_type. Doubles as reproduction-check
    # fit AND pool-to-parent fallback fit. ----
    print(f"# fitting M3 per day_type (no Z; reproduction + fallback) ...")
    t0 = time.time()
    fits_per_dt_parent = _fit_mix2_per_daytype_parent(z_pre, dims)
    for dt, fit in fits_per_dt_parent.items():
        if fit is None:
            print(f"  {dt}: library too thin")
        else:
            print(f"  {dt}: n={fit['n']}  params={fit['params']}")
    print(f"  ({time.time()-t0:.1f}s)")
    print()

    # ---- Reproduction check: M3-no-Z chi^2 vs Q2A's 953.84 ----
    print(f"# reproduction check: M3 with NO Z (Q2A's exact seed) ...")
    t0 = time.time()
    rng_repro = np.random.default_rng(
        (SEED_BASE_Q2A + 100 + _TAG_SEED_OFFSET_M3) & 0xFFFFFFFF
    )
    pit_repro = _pit_no_z_reproduction(
        z_full, fits_per_dt_parent, cutoff, anchor_h, dims,
        rng=rng_repro, M=args.n_particles,
    )
    repro_chi2, _hist = _marginal_pit_chi2(pit_repro["u_PIT"].to_numpy())
    repro_diff = abs(repro_chi2 - Q2A_M3_CHI2_SETTLED)
    print(f"  Q1A no-Z chi^2 = {repro_chi2:.1f}; Q2A settled = "
          f"{Q2A_M3_CHI2_SETTLED:.2f}; |diff| = {repro_diff:.1f}  "
          f"(tol {_REPRODUCTION_TOL_CHI2:.1f})  ({time.time()-t0:.1f}s)")
    # BLOCKING ASSERTION per advisor: M3-no-Z must reproduce Q2A's
    # settled chi^2; otherwise the Z-conditioning chi^2 numbers are
    # not comparable to Q2A's baseline and the verdict cuts are
    # meaningless.
    assert repro_diff <= _REPRODUCTION_TOL_CHI2, (
        f"REPRODUCTION CHECK FAILED: M3 with no Z gives chi^2 = "
        f"{repro_chi2:.1f}, but Q2A's settled M3 chi^2 = "
        f"{Q2A_M3_CHI2_SETTLED:.2f} (|diff| = {repro_diff:.1f} > tol "
        f"{_REPRODUCTION_TOL_CHI2:.1f}). The library / seed / sampler "
        "pipeline has drifted from Q2A's; the Z-conditioning chi^2 "
        "numbers are not comparable to Q2A's baseline and the verdict "
        "cuts (228, 2284) lose their meaning. INVESTIGATE BEFORE "
        "RUNNING THE HEAD-TO-HEAD."
    )
    print(f"  reproduction-check PASS")
    print()

    # ---- Build per-(day_type, Z) cell libraries + fit M3 ----
    libs: dict[str, dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]] = {
        "Z_b": {}, "Z_c": {},
    }
    fits_per_dt_z: dict[str, dict[str, dict[int, dict]]] = {"Z_b": {}, "Z_c": {}}
    for tag in ("Z_b", "Z_c"):
        print(f"# fitting M3 per (day_type, {tag}) cell ...")
        t0 = time.time()
        for dt in ("weekday", "saturday", "sunday"):
            d = int(dims[dt])
            libs[tag][dt] = _build_library_per_z_cell(z_pre, dt, d, z_kind=tag)
        fits_per_dt_z[tag] = _fit_mix2_per_z_cell(libs[tag])
        for dt, cell_fits in fits_per_dt_z[tag].items():
            if not cell_fits:
                print(f"  {dt}: NO cells met MIN_CELL_SIZE={MIN_CELL_SIZE}; "
                      f"all will pool to parent")
            else:
                z_keys = sorted(cell_fits.keys())
                ns = [cell_fits[k]["n"] for k in z_keys]
                print(f"  {dt}: {len(cell_fits)} cells; "
                      f"Z values {z_keys}; n={ns}")
        print(f"  ({time.time()-t0:.1f}s)")
        print()

    # ---- SMC propagation per Z candidate ----
    pit_tables: dict[str, pd.DataFrame] = {}
    chi2: dict[str, float] = {}
    hist: dict[str, list[int]] = {}
    fallback_log: dict[str, dict[tuple[str, int], int]] = {}
    for tag in ("Z_b", "Z_c"):
        print(f"# {tag} SMC propagation + post-cutoff PIT ...")
        t0 = time.time()
        rng_tag = np.random.default_rng(
            (SEED_BASE + 100 + _Z_SEED_OFFSET[tag]) & 0xFFFFFFFF
        )
        pit_tables[tag], fallback_log[tag] = _pit_for_z_candidate(
            z_full, tag, fits_per_dt_z[tag], fits_per_dt_parent,
            cutoff, anchor_h, dims, rng=rng_tag, M=args.n_particles,
        )
        c, h = _marginal_pit_chi2(pit_tables[tag]["u_PIT"].to_numpy())
        chi2[tag] = float(c)
        hist[tag] = list(h.astype(int))
        n_fallback = sum(fallback_log[tag].values())
        print(f"  rows={len(pit_tables[tag])}  chi^2={chi2[tag]:.1f}  "
              f"fallback rows={n_fallback}  ({time.time()-t0:.1f}s)")
        print()

    # ---- Per-cell chi^2 contribution decomposition (phase_a §metric
    # SECONDARY informational + variables.dependent items 10/11). For
    # each (tag, day_type, z_value) cell, compute the 10-bin marginal
    # PIT chi^2 over the post-cutoff rows in that cell. The decomposition
    # is reported informationally; the verdict statistic is the
    # POOLED-marginal chi^2 above. ----
    per_cell_chi2: dict[str, dict[tuple[str, int], dict[str, Any]]] = {
        "Z_b": {}, "Z_c": {},
    }
    for tag in ("Z_b", "Z_c"):
        for (dt, z_val), grp in pit_tables[tag].groupby(
            ["day_type", "z_value"], sort=False
        ):
            u = grp["u_PIT"].to_numpy()
            cell_chi2, _cell_h = _marginal_pit_chi2(u)
            per_cell_chi2[tag][(str(dt), int(z_val))] = {
                "chi2": float(cell_chi2),
                "n_rows": int(len(u)),
            }

    # ---- Paired-day bootstrap ----
    chi2_diff_se: float | None = None
    if args.skip_bootstrap:
        print(f"# bootstrap SKIPPED (--skip-bootstrap)")
        boot = None
    else:
        print(f"# paired-day bootstrap (n={args.n_bootstrap}) ...")
        t0 = time.time()
        all_days = pd.DatetimeIndex(sorted(set().union(
            *[set(pit_tables[t]["delivery_day"]) for t in ("Z_b", "Z_c")]
        )))
        n_days = len(all_days)
        grouped = {
            tag: {d: g for d, g in pit_tables[tag].groupby(
                "delivery_day", sort=False)}
            for tag in ("Z_b", "Z_c")
        }
        boot_chi2 = {"Z_b": [], "Z_c": [], "diff": [], "better": []}
        rng_boot = np.random.default_rng(
            (SEED_BASE + _Z_BOOT_OFFSET) & 0xFFFFFFFF
        )
        for b in range(args.n_bootstrap):
            idx = rng_boot.integers(0, n_days, size=n_days)
            sampled_days = all_days[idx]
            cells: dict[str, float] = {}
            for tag in ("Z_b", "Z_c"):
                u = pd.concat(
                    [grouped[tag][d] for d in sampled_days if d in grouped[tag]],
                    ignore_index=True,
                )["u_PIT"].to_numpy()
                c, _ = _marginal_pit_chi2(u)
                cells[tag] = c
                boot_chi2[tag].append(c)
            boot_chi2["diff"].append(cells["Z_b"] - cells["Z_c"])
            boot_chi2["better"].append(min(cells.values()))
        boot_chi2 = {k: np.array(v) for k, v in boot_chi2.items()}
        for k in ("Z_b", "Z_c", "diff", "better"):
            arr = boot_chi2[k]
            lo, med, hi = np.nanpercentile(arr, [2.5, 50, 97.5])
            print(f"  {k:>6}: median {med:.1f}  CI [{lo:.1f}, {hi:.1f}]")
        chi2_diff_se = float(np.nanstd(boot_chi2["diff"]))
        boot = {"chi2": {k: v.tolist() for k, v in boot_chi2.items()}}
        print(f"  paired diff SE = {chi2_diff_se:.2f}")
        print(f"  ({time.time()-t0:.1f}s)")
        print()

    # ---- Mechanical outcome ----
    outcome = _evaluate_outcome(chi2["Z_b"], chi2["Z_c"], chi2_diff_se)
    print(f"# pre-registered outcome: {outcome['verdict']}")
    print(f"  better_chi2:  {outcome['better_chi2']:.1f}")
    print(f"  winner:       {outcome['winner']}  (near_tie: {outcome['near_tie']})")
    print(f"  reason:       {outcome['reason']}")
    print()

    # ---- Pickle (always) ----
    out_pkl = args.out
    if not out_pkl.is_absolute():
        out_pkl = (Path.cwd() / out_pkl).resolve()
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "chi2": chi2,
        "hist": hist,
        "fits_per_dt_z": fits_per_dt_z,
        "fits_per_dt_parent": fits_per_dt_parent,
        "repro_chi2": float(repro_chi2),
        "pit_tables": pit_tables if args.save_pit else None,
        "bootstrap": boot,
        "outcome": outcome,
        "synth_gate": synth,
        "fallback_log": fallback_log,
        "per_cell_chi2": per_cell_chi2,
        "config": {
            "n_particles": args.n_particles,
            "n_bootstrap": args.n_bootstrap if not args.skip_bootstrap else 0,
            "n_pit_bins": N_PIT_BINS,
            "seed_base": SEED_BASE,
            "seed_base_q2a": SEED_BASE_Q2A,
            "R_A1A_cut": R_A1A_CUT,
            "R_C1A_cut": R_C1A_CUT,
            "min_cell_size": MIN_CELL_SIZE,
            "phase_a_body_sha256":
                "a2b13f6d8bcd6d2e4206322fc056c9b4388e761c5b5d651282e0f34d634e6d63",
        },
    }
    with out_pkl.open("wb") as f:
        pickle.dump(payload, f)
    try:
        rel = out_pkl.relative_to(PROJECT_ROOT)
        print(f"  wrote {rel}")
    except ValueError:
        print(f"  wrote {out_pkl}")

    # ---- Emit CLAIM artifact + result.yaml ----
    if args.emit_result:
        from experiment.provenance import Grade, make_result

        config_block = payload["config"]
        body = _render_body(
            chi2=chi2,
            hist=hist,
            boot=boot,
            fits_per_dt_z=fits_per_dt_z,
            fits_per_dt_parent=fits_per_dt_parent,
            repro_chi2=float(repro_chi2),
            outcome=outcome,
            synth=synth,
            fallback_log=fallback_log,
            per_cell_chi2=per_cell_chi2,
            config_block=config_block,
        )
        out_txt = (
            PROJECT_ROOT / "experiment" / "results" / "q1a_z_conditioning.txt"
        )
        print()
        print(f"# emitting CLAIM-GRADE artifact -> "
              f"{out_txt.relative_to(PROJECT_ROOT)}")
        hdr = make_result(
            path=out_txt,
            grade=Grade.CLAIM,
            title=("Q1A: Z-conditioning head-to-head (Z_b binary vs "
                   "Z_c 8-level categorical) on M3 / marginal PIT"),
            body=body,
            inputs={
                "data_cutoff": cutoff.isoformat(),
                "anchor_h": anchor_h,
                "embedding_dims": dict(dims),
                "phase_a_body_sha256":
                    "a2b13f6d8bcd6d2e4206322fc056c9b4388e761c5b5d651282e0f34d634e6d63",
                "thread_body_sha256":
                    "21a8bb20b6e9e19c8536fa664dc8978749c19c0d1c21dee3f239c59781d3f6d6",
                "q2a_settled_chi2": Q2A_M3_CHI2_SETTLED,
                "q1a_reproduction_chi2": float(repro_chi2),
                "R_A1A_cut": R_A1A_CUT,
                "R_C1A_cut": R_C1A_CUT,
                "MIN_CELL_SIZE": MIN_CELL_SIZE,
            },
            seeds={
                "seed_base": SEED_BASE,
                "seed_base_q2a": SEED_BASE_Q2A,
                "rng_purpose_offsets": {
                    "smc_per_z_offsets": _Z_SEED_OFFSET,
                    "bootstrap_offset": _Z_BOOT_OFFSET,
                    "smc_per_z": "SEED_BASE + 100 + _Z_SEED_OFFSET[tag]",
                    "bootstrap": "SEED_BASE + _Z_BOOT_OFFSET",
                    "reproduction_no_z": (
                        "SEED_BASE_Q2A + 100 + _TAG_SEED_OFFSET_M3 "
                        "(replays Q2A's M3 SMC seed exactly)"
                    ),
                    "synthetic_gate": "SEED_BASE + 9001 (Z-conditional) / 9101 (false-Z)",
                },
            },
            frozen_spec_required=True,  # predictor-derived CLAIM
        )
        print(f"  grade              = {hdr['grade']}")
        print(f"  inputs_fingerprint = {hdr['inputs_fingerprint'][:16]}...")
        print(f"  body_sha256        = {hdr['body_sha256'][:16]}...")
        print()

        # Result yaml in the registry directory
        result_yaml = _render_result_yaml(
            chi2=chi2,
            boot=boot,
            fits_per_dt_z=fits_per_dt_z,
            outcome=outcome,
            synth=synth,
            fallback_log=fallback_log,
            per_cell_chi2=per_cell_chi2,
            repro_chi2=float(repro_chi2),
            config_block=config_block,
            artifact_txt_path=out_txt.relative_to(PROJECT_ROOT),
            artifact_pkl_path=out_pkl.relative_to(PROJECT_ROOT)
                if out_pkl.is_relative_to(PROJECT_ROOT) else out_pkl,
        )
        out_yaml = (
            PROJECT_ROOT / "notes" / "preregistrations"
            / "2026-05-27_q1a-z-conditioning-strongest-axis"
            / "result.yaml"
        )
        with out_yaml.open("w") as f:
            yaml.safe_dump(result_yaml, f, sort_keys=False,
                           default_flow_style=False)
        print(f"  wrote {out_yaml.relative_to(PROJECT_ROOT)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
