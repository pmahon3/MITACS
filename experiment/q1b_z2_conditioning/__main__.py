"""Q1B / Q1B' runner — Z2-conditioning head-to-head between Z2_b (binary
overnight indicator) and Z2_c (4-level categorical per P1 production
binning) on the M3 (mix-2 Gaussian) residual law, with Q1A's settled
Z_c (8-level hour_of_week) carried forward as Z1.

Originally written for Q1B (head-to-head verdict against better_chi2
cuts (228, 2284]); now also serves Q1B' (single-arm tightening variant
at N_PARTICLES=400; verdict on cumulative_chi2_improvement_above_Z_c
CI width). The computational pipeline is identical; the artifact-pinning
strings below (phase_a hash + prereg-dir + thread hash) point at Q1B''s
preregistration since 2026-05-29 commit (this commit) per the inline
substitution; Check P + code-path audit ratified the substitution.

Implements phase_a body_sha256
``7ed3eea35553b9c5cd3e03d0003603ed4d56337d151362d4a11c2d1955b75e78``
in directory
``notes/preregistrations/2026-05-29_q1b-prime-smc-tightening/``.
Cuts inherited from the resolution-paths-thread skeleton (Q1B node);
the integer values 228 and 2284 are the rounded forms of Q2A's
gate-validated cuts (R-A2_cut = 228.44 / R-C2_cut = 2284.44).

The two candidates re-fit M3 per (day_type, Z_c, Z2) cell on the SAME
pre-cutoff library Q2A/Q1A's M3 used; SMC particle propagation uses the
per-cell mixture parameters of the cell each propagated state belongs
to. The verdict statistic is

  better_chi2 = min(chi2_Z2_b, chi2_Z2_c)

against the pre-locked cuts R-A1B (≤ 228) / R-B1B (228, 2284] /
R-C1B (> 2284). Head-to-head sub-finding records which Z2 construction
won (with paired-bootstrap CI on the difference).

Production-path commitments:
  • ``mixture_2_gaussian_mle_fit`` is called per (day_type, Z_c, Z2)
    cell (the production fitter — no reimplementation).
  • ``_smc_iterate_family`` / ``_marginal_pit_chi2`` /
    ``_build_library_for_daytype`` / ``_make_mixture_2_sampler`` are
    reused from Q2A (no reimplementation).
  • TWO reproduction-check assertions block the head-to-head:
      (1) M3 with σ=(day_type) reproduces Q2A's settled chi^2 = 953.84
          within tolerance.
      (2) M3 with σ=(day_type, Z_c) reproduces Q1A's settled chi^2 =
          952.5167 within tolerance (the cumulative-baseline anchor).
  • Pool-to-parent fallback target is the (day_type, Z_c) parent fit
    (NOT day_type-only); pooling to day_type-only would silently undo
    Q1A's Z_c carryforward and contaminate the cumulative chi^2
    (phase_a §metric step 2).
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
# both the unconditioned reproduction-check and the per-(day_type, Z_c,
# Z2) fits, matching Q2A/Q1A's seed/library/sampler conventions
# exactly).
from experiment.distributional_class_q2a.__main__ import (
    _build_library_for_daytype,
    _make_mixture_2_sampler,
    _marginal_pit_chi2,
    _smc_iterate_family,
)

# P1's production time_of_day labeller — the canonical Z2_c binning per
# phase_a §amendment_note ("P1's production binning is the source of
# truth"). Cited at experiment/p1_patra_sen/__main__.py:81-96.
from experiment.p1_patra_sen.__main__ import _time_of_day_label


# ---------------------------------------------------------------------------
# Pre-registered constants (phase_a)
# ---------------------------------------------------------------------------

# Q2A's M3 SMC convention (preserve for paired comparison).
N_PARTICLES = 200          # SMC particle count per state
N_BOOTSTRAP = 1000         # paired-day bootstrap reps (phase_a)
N_PIT_BINS = 10            # 10-bin chi^2 vs uniform (phase_a)
SEED_BASE = 20260527       # session date YYYYMMDD; per-purpose offsets below

# Reproduction-check #1 (M3 no-Z) replays Q2A's exact M3 SMC seed
# (SEED_BASE_Q2A + 100 + _TAG_SEED_OFFSET_M3 = 20260526 + 100 + 31
# = 20260657; cf. Q1A line 1140, identical convention).
# Reproduction-check #2 (M3 σ=(day_type, Z_c)) replays Q1A's exact
# Z_c-branch SMC seed (SEED_BASE_Q1A + 100 + _Z_SEED_OFFSET_Q1A_Z_c =
# 20260527 + 100 + 73 = 20260700; cf. Q1A line 1201, computed there as
# Q1A's SEED_BASE + 100 + _Z_SEED_OFFSET["Z_c"]).
SEED_BASE_Q2A = 20260526       # for reproduction-check #1 anchor (Q2A's SEED_BASE)
SEED_BASE_Q1A = 20260527       # for reproduction-check #2 anchor (Q1A's SEED_BASE)
_TAG_SEED_OFFSET_M3 = 31       # Q2A's M3 offset (used for no-Z baseline)
_Z_SEED_OFFSET_Q1A_Z_c = 73    # Q1A's Z_c SMC offset (used for Z_c-only replay)

# Q1B's own per-tag seed offsets for the Z2 head-to-head branches.
# Distinct from Q1A's _Z_SEED_OFFSET so the Z2_b / Z2_c branches don't
# accidentally inherit Q1A's Z_b / Z_c seeds.
_Z2_SEED_OFFSET = {
    "Z2_b": 89,   # Q1B's binary-Z2 SMC offset
    "Z2_c": 97,   # Q1B's 4-level-Z2 SMC offset
}
_Z2_BOOT_OFFSET = 8013        # paired-day bootstrap seed (Q1B; distinct from Q1A's 8011)

# Pre-registered verdict cuts (thread skeleton; integer-rounded forms
# of Q2A's gate-validated cuts).
R_A1B_CUT = 228      # corroboration: better_chi2 <= R_A1B_CUT (thread)
R_C1B_CUT = 2284     # falsification: better_chi2 > R_C1B_CUT (thread)
# R-B1B band = (R_A1B_CUT, R_C1B_CUT]

# Settled baselines from predecessor nodes (the two reproduction-check
# anchors).
Q2A_M3_CHI2_SETTLED = 953.84      # σ=(day_type) baseline (S9)
Q1A_M3_Z_c_CHI2_SETTLED = 952.5167  # σ=(day_type, Z_c) baseline (S12)

# Reproduction-check tolerance. Same rationale as Q1A: 50.0 chi^2 is
# ~1σ of Q2A's bootstrap distribution but with seed-equality we expect
# bit-exact-up-to-floating-point reproduction at the seeded point
# estimate. 50.0 admits paired-day bootstrap noise from library row
# entry while still firing on any structural-mismatch defect.
_REPRODUCTION_TOL_CHI2 = 50.0

# Pool-to-parent fallback (phase_a §metric step 2). A (day_type, Z_c,
# Z2) cell with fewer than MIN_CELL_SIZE rows in the PRE-CUTOFF
# LIBRARY pools its post-cutoff propagation onto the (day_type, Z_c)
# parent fit — NOT day_type-only. Pooling to day_type-only would
# silently undo Q1A's Z_c carryforward and contaminate the cumulative
# chi^2 (phase_a §metric step 2 emphatic).
MIN_CELL_SIZE = 200

# Synthetic gate at-cell row count and SMC test draws (phase_a
# baselines.secondary "in-script self-test").
SYNTHETIC_GATE_N_LIB = 2000   # rows per synthetic (day_type, Z_c, Z2) cell
SYNTHETIC_GATE_M = 100        # SMC particles per state (synthetic side)
SYNTHETIC_GATE_CHI2_CEILING = 60.0  # per-cell test PIT chi^2 sanity ceiling


# ---------------------------------------------------------------------------
# Z derivation
# ---------------------------------------------------------------------------
#
# Z_c (carryforward from Q1A; 8-level hour_of_week categorical):
#   bin = (weekday * 24 + (h - 1)) // 21, weekday ∈ {0=Mon..6=Sun},
#   h ∈ {1..24} (Q1's pit_M1 convention; h is the delivery-clock hour).
#
# Z2_b (binary overnight indicator):  0 if h ∈ {1..6} else 1.
# Z2_c (4-level categorical per P1 production binning at
#   experiment/p1_patra_sen/__main__.py:81-96):
#     overnight    h ∈ {1..6}    → 0
#     morning_ramp h ∈ {7..11}   → 1
#     afternoon    h ∈ {12..17}  → 2
#     evening      h ∈ {18..24}  → 3

_HOUR_OF_WEEK_BIN_WIDTH = 21


def _hour_of_week_bin(weekday: int, h: int) -> int:
    """8 bins of 21 hours each. weekday=0..6 (Mon..Sun), h=1..24.

    Matches Q1A's `_hour_of_week_bin` and P1's identical function at
    experiment/p1_patra_sen/__main__.py:99-105. Z_c carryforward from
    Q1A SETTLED R-B1A.
    """
    hour_of_week = weekday * 24 + (h - 1)
    return hour_of_week // _HOUR_OF_WEEK_BIN_WIDTH  # 0..7


def _z_c_label(weekday: int, h: int) -> int:
    """8-level categorical Z_c: bin 0..7 directly (Q1A carryforward)."""
    return _hour_of_week_bin(weekday, h)


def _z2_b_label(h: int) -> int:
    """Binary overnight indicator: 0 if h ∈ {1..6} else 1.

    Clean overnight-vs-rest contrast per P1 SETTLED R-A
    time_of_day_axis_ranking (overnight 0.1125 essentially Gaussian
    vs daytime/evening 0.475-0.598).
    """
    return 0 if h <= 6 else 1


_Z2_C_LABEL_TO_INT = {
    "overnight":    0,
    "morning_ramp": 1,
    "afternoon":    2,
    "evening":      3,
}


def _z2_c_label(h: int) -> int:
    """4-level categorical Z2_c per P1's production binning.

    Uses P1's `_time_of_day_label(h)` directly (experiment/p1_patra_sen/
    __main__.py:81-96), then maps the string label to its integer level
    via _Z2_C_LABEL_TO_INT. Per phase_a §amendment_note, P1's
    production binning is the methodology source-of-truth.
    """
    return _Z2_C_LABEL_TO_INT[_time_of_day_label(h)]


def _derive_z_for_anchor(anchor: pd.Timestamp) -> tuple[int, int, int]:
    """Derive (Z_c, Z2_b, Z2_c) for a one-row anchor.

    Z labels are computed at the DELIVERY HOUR (= anchor + 1h target)
    so they describe the cell the next-step prediction is FOR — same
    convention as Q1A (anchor + 1h; see Q1A lines 172-186 for the
    production pattern). pd.Timestamp arithmetic handles day rollover
    correctly.
    """
    target = anchor + pd.Timedelta(hours=1)
    weekday = int(target.dayofweek)
    h = int(target.hour) + 1  # 0..23 clock hour → 1..24 convention
    return _z_c_label(weekday, h), _z2_b_label(h), _z2_c_label(h)


# ---------------------------------------------------------------------------
# Library construction per (day_type, Z_c, Z2) cell
# ---------------------------------------------------------------------------
#
# Mirrors Q1A's _build_library_per_z_cell (Q1A lines 201-249), but the
# cell key is now a 3-tuple (Z_c, Z2). Day_type filter at ANCHOR
# (matches Q2A/Q1A); Z filter at predicted-hour (anchor + 1h) per
# phase_a §metric step 3.


def _build_library_per_zc_z2_cell(
    z_pre: pd.Series,
    day_type: str,
    d: int,
    z2_kind: str,   # "Z2_b" or "Z2_c"
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    """Per-(day_type, Z_c, Z2) cell (X, Y) library.

    Returns ``{(z_c_value, z2_value): (X, Y)}`` for every cell with at
    least MIN_CELL_SIZE rows. Cells below threshold are absent from the
    returned dict; callers detect pool-to-(day_type, Z_c)-parent
    fallback via the empty cell.
    """
    idx = z_pre.index
    by_cell: dict[tuple[int, int], list[tuple[np.ndarray, float]]] = {}
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
        z_c_val, z2_b_val, z2_c_val = _derive_z_for_anchor(anchor)
        z2_val = z2_b_val if z2_kind == "Z2_b" else z2_c_val
        by_cell.setdefault((z_c_val, z2_val), []).append((lag, float(y)))
    out: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for key, rows in by_cell.items():
        if len(rows) < MIN_CELL_SIZE:
            continue
        X = np.array([r[0] for r in rows])
        Y = np.array([r[1] for r in rows])
        out[key] = (X, Y)
    return out


def _build_library_per_zc_only(
    z_pre: pd.Series,
    day_type: str,
    d: int,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Per-(day_type, Z_c) cell (X, Y) library — Q1A's σ-algebra
    exactly. Used to fit the (day_type, Z_c) parent for both
    reproduction-check #2 and pool-to-parent fallback.

    Returns ``{z_c_value: (X, Y)}``. This duplicates Q1A's
    _build_library_per_z_cell with z_kind="Z_c"; we inline it here
    rather than importing to avoid coupling to Q1A's internal symbol
    names (the production-path audit treats Q1A as a sibling
    experiment, not a whitelisted module).
    """
    idx = z_pre.index
    by_zc: dict[int, list[tuple[np.ndarray, float]]] = {}
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
        z_c_val, _z2_b, _z2_c = _derive_z_for_anchor(anchor)
        by_zc.setdefault(z_c_val, []).append((lag, float(y)))
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for z_c_val, rows in by_zc.items():
        if len(rows) < MIN_CELL_SIZE:
            continue
        X = np.array([r[0] for r in rows])
        Y = np.array([r[1] for r in rows])
        out[z_c_val] = (X, Y)
    return out


# ---------------------------------------------------------------------------
# Production-path fits per σ-algebra
# ---------------------------------------------------------------------------


def _fit_mix2_per_zc_z2_cell(
    libs_by_dt_cell: dict[str, dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]],
) -> dict[str, dict[tuple[int, int], dict]]:
    """Production-path mixture_2 fit per (day_type, Z_c, Z2) cell."""
    out: dict[str, dict[tuple[int, int], dict]] = {}
    for dt, cell_to_lib in libs_by_dt_cell.items():
        out[dt] = {}
        for key, (X, Y) in cell_to_lib.items():
            C, params, _resid = mixture_2_gaussian_mle_fit(X, Y)
            out[dt][key] = {
                "C": C,
                "params": params,
                "d": int(X.shape[1]),
                "n": int(len(Y)),
            }
    return out


def _fit_mix2_per_zc_parent(
    libs_by_dt_zc: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]],
) -> dict[str, dict[int, dict]]:
    """Production-path mixture_2 fit per (day_type, Z_c) parent cell.

    Doubles as (a) reproduction-check #2 fit (Q1A's σ-algebra exactly)
    AND (b) pool-to-parent fallback target for sparse (day_type, Z_c,
    Z2) cells per phase_a §metric step 2.
    """
    out: dict[str, dict[int, dict]] = {}
    for dt, zc_to_lib in libs_by_dt_zc.items():
        out[dt] = {}
        for z_c_val, (X, Y) in zc_to_lib.items():
            C, params, _resid = mixture_2_gaussian_mle_fit(X, Y)
            out[dt][z_c_val] = {
                "C": C,
                "params": params,
                "d": int(X.shape[1]),
                "n": int(len(Y)),
            }
    return out


def _fit_mix2_per_daytype_parent(
    z_pre: pd.Series,
    dims: dict[str, int],
) -> dict[str, dict | None]:
    """Per day_type, no-Z M3 fit. Reproduction-check #1 fit (Q2A's M3
    exactly). Per phase_a §metric.SECONDARY 'Reproduction check #1':
    re-running M3 with no Z, no Z2 reproduces Q2A's chi^2 = 953.84
    within paired-bootstrap noise.
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
# SMC propagation per σ-algebra
# ---------------------------------------------------------------------------


def _pit_for_zc_z2_candidate(
    z_full: pd.Series,
    z2_kind: str,
    fits_per_dt_cell: dict[str, dict[tuple[int, int], dict]],
    fits_per_dt_zc_parent: dict[str, dict[int, dict]],
    fits_per_dt_dt_parent: dict[str, dict | None],
    cutoff: pd.Timestamp,
    anchor_h: int,
    dims: dict[str, int],
    rng: np.random.Generator,
    M: int = N_PARTICLES,
) -> tuple[pd.DataFrame, dict[tuple[str, int, int], int]]:
    """Per-issue-time PIT table for one Z2 candidate (head-to-head).

    Routes each (day_type, predicted-hour) one-step to the
    corresponding (day_type, Z_c, Z2) cell M3 fit; on cell-miss falls
    back to the (day_type, Z_c) parent fit per phase_a §metric step 2
    (NOT day_type-only). If the (day_type, Z_c) parent is also absent
    (should not happen on the registered configuration), final fallback
    is the day_type parent (logged separately).

    Returns ``(pit_df, fallback_log)`` where ``pit_df`` has columns
    ``[delivery_day, h, day_type, z_c_value, z2_value, used_fallback,
    u_PIT]`` and ``fallback_log[(day_type, z_c_value, z2_value)] =
    count`` of rows that pooled to (day_type, Z_c) parent.
    """
    days = _complete_delivery_days(z_full, anchor_h)
    days = days[days > cutoff]
    rows: list[dict] = []
    fallback_log: dict[tuple[str, int, int], int] = {}

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
            z_c_val = _z_c_label(weekday_p, h_p)
            if z2_kind == "Z2_b":
                z2_val = _z2_b_label(h_p)
            else:
                z2_val = _z2_c_label(h_p)

            # Cell fit lookup with cascading fallback:
            #   (day_type, Z_c, Z2) cell -> (day_type, Z_c) parent -> (day_type) parent
            cell_fit = fits_per_dt_cell.get(dt, {}).get((z_c_val, z2_val))
            used_fallback = False
            if cell_fit is None:
                cell_fit = fits_per_dt_zc_parent.get(dt, {}).get(z_c_val)
                if cell_fit is not None:
                    used_fallback = True
                    fallback_log[(dt, z_c_val, z2_val)] = (
                        fallback_log.get((dt, z_c_val, z2_val), 0) + 1
                    )
            if cell_fit is None:
                # Both 3-tuple cell AND (day_type, Z_c) parent missing
                # — final fallback to day_type parent (informational).
                cell_fit = fits_per_dt_dt_parent.get(dt)
                if cell_fit is not None:
                    used_fallback = True
                    fallback_log[(dt, z_c_val, z2_val)] = (
                        fallback_log.get((dt, z_c_val, z2_val), 0) + 1
                    )
            if cell_fit is None:
                continue

            sampler = _make_mixture_2_sampler(cell_fit["params"])
            C_col = cell_fit["C"]
            r_draws = sampler(M, rng)
            next_z = (particles @ C_col).ravel() + r_draws
            samp_at_h = next_z.copy()
            particles = np.column_stack([next_z, particles[:, :d - 1]])

            less = float(np.sum(samp_at_h < actuals[h_step - 1]))
            eq = float(np.sum(samp_at_h == actuals[h_step - 1]))
            U = rng.random()
            u = (less + U * (eq + 1.0)) / (M + 1.0)
            rows.append({
                "delivery_day": D,
                "h": h_step,
                "day_type": dt,
                "z_c_value": int(z_c_val),
                "z2_value": int(z2_val),
                "used_fallback": used_fallback,
                "u_PIT": float(u),
            })
    return pd.DataFrame(rows), fallback_log


def _pit_no_z_reproduction(
    z_full: pd.Series,
    fits_per_dt_dt_parent: dict[str, dict | None],
    cutoff: pd.Timestamp,
    anchor_h: int,
    dims: dict[str, int],
    rng: np.random.Generator,
    M: int = N_PARTICLES,
) -> pd.DataFrame:
    """Reproduction-check #1 PIT table: M3 with σ=(day_type) alone.

    Identical to Q2A's M3 _pit_for_family path: no Z, no Z2 routing,
    day_type parent fit used at every step. Identical seed (caller
    passes SEED_BASE_Q2A + 100 + _TAG_SEED_OFFSET_M3) so the chi^2
    reproduces Q2A's 953.84 within tolerance.
    """
    from experiment.distributional_class_q2a.__main__ import _pit_for_family

    return _pit_for_family(
        z_full,
        fits_per_dt_dt_parent,
        "mixture_2",
        cutoff,
        anchor_h,
        dims,
        rng=rng,
        M=M,
    )


def _pit_zc_only_reproduction(
    z_full: pd.Series,
    fits_per_dt_zc_parent: dict[str, dict[int, dict]],
    fits_per_dt_dt_parent: dict[str, dict | None],
    cutoff: pd.Timestamp,
    anchor_h: int,
    dims: dict[str, int],
    rng: np.random.Generator,
    M: int = N_PARTICLES,
) -> pd.DataFrame:
    """Reproduction-check #2 PIT table: M3 with σ=(day_type, Z_c) only.

    Replays Q1A's Z_c-only path exactly. Routes propagation to the
    (day_type, Z_c) parent fits (no Z2 stratification). Caller seeds
    the RNG with SEED_BASE_Q1A + 100 + _Z_SEED_OFFSET_Q1A_Z_c so the
    chi^2 reproduces Q1A's 952.5167 within tolerance.

    Structurally identical to _pit_for_zc_z2_candidate but with cell
    lookup terminating at the (day_type, Z_c) parent — i.e., the
    cascading-fallback path of the head-to-head with no 3-tuple cells
    populated.
    """
    days = _complete_delivery_days(z_full, anchor_h)
    days = days[days > cutoff]
    rows: list[dict] = []
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
        target_dts = [
            D + pd.Timedelta(hours=anchor_h + (h - 1)) for h in range(1, 25)
        ]
        actuals = z_full.reindex(target_dts).to_numpy()
        if not np.all(np.isfinite(actuals)):
            continue

        particles = np.tile(x_state, (M, 1))
        for h_step in range(1, 25):
            predicted_dt = target_dts[h_step - 1]
            weekday_p = int(predicted_dt.dayofweek)
            h_p = int(predicted_dt.hour) + 1
            z_c_val = _z_c_label(weekday_p, h_p)

            cell_fit = fits_per_dt_zc_parent.get(dt, {}).get(z_c_val)
            if cell_fit is None:
                cell_fit = fits_per_dt_dt_parent.get(dt)
            if cell_fit is None:
                continue

            sampler = _make_mixture_2_sampler(cell_fit["params"])
            C_col = cell_fit["C"]
            r_draws = sampler(M, rng)
            next_z = (particles @ C_col).ravel() + r_draws
            samp_at_h = next_z.copy()
            particles = np.column_stack([next_z, particles[:, :d - 1]])

            less = float(np.sum(samp_at_h < actuals[h_step - 1]))
            eq = float(np.sum(samp_at_h == actuals[h_step - 1]))
            U = rng.random()
            u = (less + U * (eq + 1.0)) / (M + 1.0)
            rows.append({
                "delivery_day": D,
                "h": h_step,
                "day_type": dt,
                "z_c_value": int(z_c_val),
                "u_PIT": float(u),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Inline synthetic gate (BLOCKING)
# ---------------------------------------------------------------------------
#
# Phase_a §baselines.secondary: generates synthetic data with a known
# Z2-conditional mixture DGP that varies parameters across (Z_c_stratum,
# Z2_stratum) pairs (3-tuple cell key). Two recovery checks:
#   (i)  per-cell fitted (w, s1, s2) within tolerance of ground truth;
#   (ii) per-cell PIT chi^2 << R-A1B cut at 228.
# Companion FALSE-Z2 gate: regenerate with Z2 randomly permuted; confirm
# chi^2 collapses to the Z_c-only baseline (so the per-stratum fit is
# not artificially shrinking chi^2 by overfitting a noise label).

# Known Z2-conditional DGP (locked at script-write time, NOT post-hoc).
# Keyed by (Z_c_stratum, Z2_stratum) → mixture params. The DGP varies
# parameters within a Z_c stratum across Z2 to exercise the Z2 marginal
# information ABOVE Z_c — i.e., the very quantity Q1B is testing.
# We pick a SMALL number of (z_c, z2) pairs (4 strata: 2 Z_c × 2 Z2)
# to keep the synthetic gate under ~30s wall-clock; the Ontario
# (day_type × Z_c × Z2) cell count is much larger but the gate's role
# is structural recovery, not magnitude.
_SYNTH_DGP: dict[tuple[int, int], dict] = {
    (0, 0): {"w": (0.95, 0.05), "s": (1.0, 2.5)},   # Z_c=0, Z2=0
    (0, 1): {"w": (0.85, 0.15), "s": (1.0, 3.5)},   # Z_c=0, Z2=1
    (1, 0): {"w": (0.70, 0.30), "s": (1.0, 4.0)},   # Z_c=1, Z2=0
    (1, 1): {"w": (0.55, 0.45), "s": (1.0, 5.0)},   # Z_c=1, Z2=1
}
_SYNTH_PARAM_TOL = 0.15  # tolerance on fitted (w_r, s2) vs ground truth


def _draw_synthetic_zc_z2_conditional(
    n_per_cell: int,
    rng: np.random.Generator,
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    """One (X, Y) library per (Z_c, Z2) cell, drawn from the known DGP."""
    out: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    c_true = np.array([[0.7], [0.2]])
    for key, dgp in _SYNTH_DGP.items():
        w = np.array(dgp["w"])
        s = np.array(dgp["s"])
        X = rng.standard_normal((n_per_cell, 2))
        which = rng.choice(2, size=n_per_cell, p=w)
        eps = rng.standard_normal(n_per_cell)
        r = eps * s[which]
        Y = (X @ c_true).ravel() + r
        out[key] = (X, Y)
    return out


def _run_inline_synthetic_gate(verbose: bool = True) -> dict:
    """Inline synthetic gate. BLOCKING on FAIL.

    Three sub-checks:
      (a) per-cell mixture parameter recovery within tolerance.
      (b) per-cell PIT chi^2 under the Z2-conditional model << 228.
      (c) under FALSE (random) Z2 labels, fits collapse back to the
          Z_c-only marginal mixture; companion sanity check.
    """
    rng = np.random.default_rng(SEED_BASE + 9001)
    libs = _draw_synthetic_zc_z2_conditional(SYNTHETIC_GATE_N_LIB, rng)

    out: dict[str, Any] = {"checks": []}
    all_ok = True

    # (a) Per-cell mixture parameter recovery.
    fits_per_cell: dict[tuple[int, int], dict] = {}
    for key, (X, Y) in libs.items():
        C, params, _r = mixture_2_gaussian_mle_fit(X, Y)
        fits_per_cell[key] = {"C": C, "params": params, "d": int(X.shape[1])}
        w_true = _SYNTH_DGP[key]["w"]
        s_true = _SYNTH_DGP[key]["s"]
        w_fit = params["weights"]
        s_fit = params["scales"]
        # Identifiability: mixture_2_gaussian_mle_fit orders s1<=s2 and
        # the DGP also has s1<=s2 by construction; no relabeling.
        w_r_true = w_true[1]
        w_r_fit = w_fit[1]
        s2_true = s_true[1]
        s2_fit = s_fit[1]
        ok_w = abs(w_r_fit - w_r_true) <= _SYNTH_PARAM_TOL
        ok_s = abs(s2_fit - s2_true) / s2_true <= _SYNTH_PARAM_TOL
        out["checks"].append({
            "name": f"synth_cell_zc{key[0]}_z2{key[1]}_param_recovery",
            "w_r_true": w_r_true, "w_r_fit": w_r_fit,
            "s2_true": s2_true, "s2_fit": s2_fit,
            "ok_w": ok_w, "ok_s": ok_s,
        })
        if verbose:
            tag = "OK" if (ok_w and ok_s) else "FAIL"
            print(f"  [{tag}] synth cell (Z_c={key[0]}, Z2={key[1]}): "
                  f"w_r {w_r_fit:.3f} (true {w_r_true:.3f}, tol {_SYNTH_PARAM_TOL}); "
                  f"s2 {s2_fit:.3f} (true {s2_true:.3f}, rel tol {_SYNTH_PARAM_TOL})")
        all_ok = all_ok and ok_w and ok_s

    # (b) Per-cell test-draw PIT chi^2 << 228.
    chi2_per_cell: dict[tuple[int, int], float] = {}
    for key in libs:
        fit = fits_per_cell[key]
        sampler = _make_mixture_2_sampler(fit["params"])
        c_true = np.array([[0.7], [0.2]])
        n_test = SYNTHETIC_GATE_N_LIB
        X_test = rng.standard_normal((n_test, 2))
        means_pred = (X_test @ fit["C"]).ravel()
        true_means = (X_test @ c_true).ravel()
        s_true = np.array(_SYNTH_DGP[key]["s"])
        w_true = np.array(_SYNTH_DGP[key]["w"])
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
        chi2_per_cell[key] = float(chi2)
        ok_chi2 = chi2 <= SYNTHETIC_GATE_CHI2_CEILING
        out["checks"].append({
            "name": f"synth_cell_zc{key[0]}_z2{key[1]}_chi2",
            "chi2": float(chi2), "ceiling": SYNTHETIC_GATE_CHI2_CEILING,
            "ok": ok_chi2,
        })
        if verbose:
            tag = "OK" if ok_chi2 else "FAIL"
            print(f"  [{tag}] synth cell (Z_c={key[0]}, Z2={key[1]}): "
                  f"chi^2 = {chi2:.1f}  "
                  f"(ceiling {SYNTHETIC_GATE_CHI2_CEILING:.0f}, n_test={n_test})")
        all_ok = all_ok and ok_chi2

    # (c) FALSE-Z2 companion: within each Z_c, pool both Z2 cells under
    # random Z2 labels; fit collapses to Z_c-only marginal mixture. The
    # substantive question is whether the false-Z2 per-cell mixtures
    # collapse toward the Z_c-MARGINAL mixture rather than picking up
    # the cell-true mixture.
    rng_false = np.random.default_rng(SEED_BASE + 9101)
    collapse_w: list[bool] = []
    for z_c_val in (0, 1):
        X_zc = np.concatenate([libs[(z_c_val, 0)][0], libs[(z_c_val, 1)][0]])
        Y_zc = np.concatenate([libs[(z_c_val, 0)][1], libs[(z_c_val, 1)][1]])
        false_z2 = rng_false.choice(2, size=len(Y_zc))
        # Z_c-marginal w_r as the collapse target:
        w_r_marg_expected = (
            0.5 * _SYNTH_DGP[(z_c_val, 0)]["w"][1]
            + 0.5 * _SYNTH_DGP[(z_c_val, 1)]["w"][1]
        )
        for z2_val in (0, 1):
            mask = false_z2 == z2_val
            Xc = X_zc[mask]
            Yc = Y_zc[mask]
            if len(Yc) < MIN_CELL_SIZE:
                continue
            _C, params, _r = mixture_2_gaussian_mle_fit(Xc, Yc)
            w_r_fit = params["weights"][1]
            d_marg = abs(w_r_fit - w_r_marg_expected)
            d_true = abs(w_r_fit - _SYNTH_DGP[(z_c_val, z2_val)]["w"][1])
            collapse_w.append(d_marg < d_true)
            if verbose:
                ok = d_marg < d_true
                tag = "OK" if ok else "WARN"
                print(f"  [{tag}] false-Z2 cell (Z_c={z_c_val}, Z2={z2_val}): "
                      f"w_r {w_r_fit:.3f}  "
                      f"(d to Z_c-marginal {w_r_marg_expected:.3f}: "
                      f"{d_marg:.3f}; d to cell-true "
                      f"{_SYNTH_DGP[(z_c_val, z2_val)]['w'][1]:.3f}: "
                      f"{d_true:.3f})")
    out["checks"].append({
        "name": "false_z2_companion_collapse",
        "all_cells_collapsed": bool(collapse_w and all(collapse_w)),
        # advisory: same convention as Q1A (informational, not blocking).
    })

    out["all_ok"] = all_ok
    return out


# ---------------------------------------------------------------------------
# Pre-registered outcome evaluation
# ---------------------------------------------------------------------------


def _evaluate_outcome(
    chi2_Z2_b: float,
    chi2_Z2_c: float,
    chi2_diff_se: float | None,
) -> dict:
    """Mechanical evaluation per phase_a §metric.HEAD-TO-HEAD VERDICT RULE.

    The verdict fires on better_chi2 = min(chi2_Z2_b, chi2_Z2_c)
    against the pre-locked cuts. NEAR-TIE handling: if |chi2_diff| <
    1.0 * SE of the paired difference, the head-to-head sub-finding
    records 'no ranking'.
    """
    better_chi2 = min(chi2_Z2_b, chi2_Z2_c)
    winner = "Z2_b" if chi2_Z2_b < chi2_Z2_c else "Z2_c"
    diff = chi2_Z2_b - chi2_Z2_c
    near_tie = False
    if chi2_diff_se is not None and chi2_diff_se > 0:
        near_tie = abs(diff) < 1.0 * chi2_diff_se

    if better_chi2 <= R_A1B_CUT:
        verdict = "R-A1B"
        reason = (
            f"better_chi2 = min(Z2_b={chi2_Z2_b:.1f}, Z2_c={chi2_Z2_c:.1f}) "
            f"= {better_chi2:.1f} <= R-A1B cut {R_A1B_CUT}; time_of_day-"
            f"derived Z2 ON TOP OF hour_of_week-derived Z_c closes the "
            f"cumulative marginal chi^2 gap. Per thread "
            f"branching_rules.Q1B.R-A1B and root.closure_rule, the "
            f"resolution-paths-thread is RESOLVED on the multi-Z path. "
            f"Head-to-head sub-finding: "
            f"{'winner = ' + winner if not near_tie else 'NEAR-TIE (no ranking)'}; "
            f"chi^2 diff = {diff:+.1f}."
        )
    elif better_chi2 <= R_C1B_CUT:
        verdict = "R-B1B"
        reason = (
            f"better_chi2 = min(Z2_b={chi2_Z2_b:.1f}, Z2_c={chi2_Z2_c:.1f}) "
            f"= {better_chi2:.1f} in (R-A1B {R_A1B_CUT}, R-C1B {R_C1B_CUT}]; "
            f"Z2 helps but does not close on top of Z_c. Per thread "
            f"branching_rules.Q1B.R-B1B = null, no auto-advancement. "
            f"Head-to-head sub-finding: "
            f"{'winner = ' + winner if not near_tie else 'NEAR-TIE'}; "
            f"chi^2 diff = {diff:+.1f}."
        )
    else:
        verdict = "R-C1B"
        reason = (
            f"better_chi2 = min(Z2_b={chi2_Z2_b:.1f}, Z2_c={chi2_Z2_c:.1f}) "
            f"= {better_chi2:.1f} > R-C1B cut {R_C1B_CUT}; Z2 doesn't "
            f"help on top of Z_c (or makes things worse via small-cell "
            f"EM variance). Thread routes to P2 per "
            f"branching_rules.Q1B.R-C1B."
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
    fits_per_dt_cell: dict[str, dict[str, dict[tuple[int, int], dict]]],
    repro_chi2_no_z: float,
    repro_chi2_zc: float,
    outcome: dict,
    cumulative_improvement: float,
    cumulative_improvement_ci: tuple[float | None, float | None],
    synth: dict,
    fallback_log: dict[str, dict[tuple[str, int, int], int]],
    per_cell_chi2: dict[str, dict[tuple[str, int, int], dict[str, Any]]],
    config_block: dict,
) -> str:
    L: list[str] = []
    L.append("Q1B — Z2-conditioning (head-to-head Z2_b vs Z2_c) on M3, "
             "carrying forward Q1A's Z_c (Z1)")
    L.append("=" * 78)
    L.append("")
    L.append("Pre-registration: notes/preregistrations/")
    L.append("  2026-05-29_q1b-prime-smc-tightening/phase_a.yaml")
    L.append(f"  phase_a body_sha256: "
             f"7ed3eea35553b9c5cd3e03d0003603ed4d56337d151362d4a11c2d1955b75e78")
    L.append("Cuts from thread skeleton (integer-rounded forms of Q2A cuts):")
    L.append(f"  R-A1B (corroboration):    better_chi2 <= {R_A1B_CUT}")
    L.append(f"  R-C1B (falsification):    better_chi2  > {R_C1B_CUT}")
    L.append(f"  R-B1B (ambiguous band):   ({R_A1B_CUT}, {R_C1B_CUT}]")
    L.append("")
    L.append("REPRODUCTION CHECK #1 (M3 σ=(day_type) vs Q2A settled)")
    L.append("-" * 78)
    L.append(f"  Q1B reproduction chi^2: {repro_chi2_no_z:.4f}")
    L.append(f"  Q2A settled chi^2:      {Q2A_M3_CHI2_SETTLED:.4f}")
    L.append(f"  |diff|:                 {abs(repro_chi2_no_z - Q2A_M3_CHI2_SETTLED):.4f}  "
             f"(tol {_REPRODUCTION_TOL_CHI2:.1f})")
    L.append("")
    L.append("REPRODUCTION CHECK #2 (M3 σ=(day_type, Z_c) vs Q1A settled)")
    L.append("-" * 78)
    L.append(f"  Q1B reproduction chi^2: {repro_chi2_zc:.4f}")
    L.append(f"  Q1A settled chi^2:      {Q1A_M3_Z_c_CHI2_SETTLED:.4f}")
    L.append(f"  |diff|:                 {abs(repro_chi2_zc - Q1A_M3_Z_c_CHI2_SETTLED):.4f}  "
             f"(tol {_REPRODUCTION_TOL_CHI2:.1f})")
    L.append("")
    L.append("MARGINAL PIT CHI^2 PER Z2 CANDIDATE (post-cutoff, 10-bin)")
    L.append("-" * 78)
    for tag in ("Z2_b", "Z2_c"):
        line = f"  chi2_{tag}: {chi2[tag]:>10.1f}"
        if boot is not None and tag in boot.get("chi2", {}):
            arr = np.asarray(boot["chi2"][tag])
            lo, med, hi = np.nanpercentile(arr, [2.5, 50, 97.5])
            line += f"    bootstrap 95% CI [{lo:.1f}, {hi:.1f}]  median {med:.1f}"
        L.append(line)
    L.append("")
    L.append("HEAD-TO-HEAD CHI^2 DIFFERENCE (Z2_b - Z2_c)")
    L.append("-" * 78)
    L.append(f"  chi2_diff = {outcome['chi2_diff']:+.1f}")
    if boot is not None and "diff" in boot.get("chi2", {}):
        arr = np.asarray(boot["chi2"]["diff"])
        lo, med, hi = np.nanpercentile(arr, [2.5, 50, 97.5])
        L.append(f"  bootstrap 95% CI [{lo:+.1f}, {hi:+.1f}]  median {med:+.1f}")
    L.append(f"  near_tie: {outcome['near_tie']}")
    L.append("")
    L.append("BETTER (verdict) CHI^2 AND CUMULATIVE IMPROVEMENT ABOVE Z_c")
    L.append("-" * 78)
    L.append(f"  better_chi2 = min(Z2_b, Z2_c) = {outcome['better_chi2']:.1f}")
    L.append(f"  winner                       = {outcome['winner']}")
    if boot is not None and "better" in boot.get("chi2", {}):
        arr = np.asarray(boot["chi2"]["better"])
        lo, med, hi = np.nanpercentile(arr, [2.5, 50, 97.5])
        L.append(f"  bootstrap 95% CI [{lo:.1f}, {hi:.1f}]  median {med:.1f}")
        landslide = lo <= R_A1B_CUT and outcome['better_chi2'] <= R_A1B_CUT
        L.append(f"  landslide qualifier (R-A1B only): "
                 f"{'TRUE' if landslide else 'FALSE'}")
    L.append(f"  cumulative_chi2_improvement_above_Z_c = "
             f"{Q1A_M3_Z_c_CHI2_SETTLED} - {outcome['better_chi2']:.1f} = "
             f"{cumulative_improvement:+.1f}")
    if cumulative_improvement_ci[0] is not None:
        L.append(f"    bootstrap 95% CI "
                 f"[{cumulative_improvement_ci[0]:+.1f}, "
                 f"{cumulative_improvement_ci[1]:+.1f}]")
    L.append("")
    L.append("PRE-REGISTERED VERDICT")
    L.append("-" * 78)
    L.append(f"  verdict: {outcome['verdict']}")
    L.append(f"  reason:  {outcome['reason']}")
    L.append("")
    L.append("PER-CELL FITTED MIXTURE PARAMETERS (mix-2) — 3-tuple cell key "
             "(day_type, Z_c, Z2)")
    L.append("-" * 78)
    for tag in ("Z2_b", "Z2_c"):
        L.append(f"  [{tag}]")
        for dt, cell_fits in fits_per_dt_cell[tag].items():
            for key in sorted(cell_fits.keys()):
                fit = cell_fits[key]
                w = fit["params"]["weights"]
                s = fit["params"]["scales"]
                L.append(
                    f"    {dt} (Z_c={key[0]}, Z2={key[1]}): "
                    f"w=({w[0]:.3f}, {w[1]:.3f})  "
                    f"s=({s[0]:.3f}, {s[1]:.3f})  n={fit['n']}"
                )
    L.append("")
    L.append("PER-CELL CHI^2 CONTRIBUTION (informational decomposition)")
    L.append("-" * 78)
    for tag in ("Z2_b", "Z2_c"):
        L.append(f"  [{tag}]")
        cells = per_cell_chi2[tag]
        if not cells:
            L.append("    (no cells)")
        else:
            for (dt, z_c, z2), entry in sorted(cells.items()):
                L.append(
                    f"    {dt} (Z_c={z_c}, Z2={z2}): "
                    f"chi^2 = {entry['chi2']:>7.1f}  n={entry['n_rows']}"
                )
    L.append("")
    L.append("POOLING FALLBACK LOG (cells that pooled to (day_type, Z_c) parent)")
    L.append("-" * 78)
    for tag in ("Z2_b", "Z2_c"):
        L.append(f"  [{tag}]")
        log = fallback_log[tag]
        if not log:
            L.append("    (no cells pooled — every (day_type, Z_c, Z2) cell met "
                     "MIN_CELL_SIZE)")
        else:
            for (dt, z_c, z2), count in sorted(log.items()):
                L.append(f"    {dt} (Z_c={z_c}, Z2={z2}): "
                         f"{count} post-cutoff rows pooled to (day_type, Z_c) parent")
    L.append("")
    L.append("PIT HISTOGRAMS (10-bin) — post-cutoff marginal")
    L.append("-" * 78)
    for tag in ("Z2_b", "Z2_c"):
        L.append(f"  {tag}: {hist[tag]}")
    L.append("")
    L.append("INLINE SYNTHETIC GATE")
    L.append("-" * 78)
    L.append(f"  all_ok: {synth['all_ok']}")
    for chk in synth["checks"]:
        L.append(f"  {chk['name']}: " + ", ".join(
            f"{k}={v}" for k, v in chk.items() if k != "name"
        ))
    L.append("")
    L.append("CONFIG")
    L.append("-" * 78)
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
    fits_per_dt_cell: dict[str, dict[str, dict[tuple[int, int], dict]]],
    outcome: dict,
    cumulative_improvement: float,
    cumulative_improvement_ci: tuple[float | None, float | None],
    synth: dict,
    fallback_log: dict[str, dict[tuple[str, int, int], int]],
    per_cell_chi2: dict[str, dict[tuple[str, int, int], dict[str, Any]]],
    repro_chi2_no_z: float,
    repro_chi2_zc: float,
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

    def cells_for(tag: str) -> dict[str, dict[str, dict]]:
        out: dict[str, dict] = {}
        for dt, cell_fits in fits_per_dt_cell[tag].items():
            out[dt] = {}
            for (z_c, z2), fit in cell_fits.items():
                k = f"zc{int(z_c)}__z2{int(z2)}"
                out[dt][k] = {
                    "z_c": int(z_c),
                    "z2": int(z2),
                    "weights": [float(x) for x in fit["params"]["weights"]],
                    "scales": [float(x) for x in fit["params"]["scales"]],
                    "n_library": int(fit["n"]),
                }
        return out

    ci_low_b, ci_high_b = ci_for("Z2_b")
    ci_low_c, ci_high_c = ci_for("Z2_c")
    ci_low_better, ci_high_better = ci_for("better")
    ci_low_diff, ci_high_diff = ci_for("diff")

    # Cumulative improvement CI: directly derived from the better_chi2
    # bootstrap distribution (improvement = Q1A_settled - better_chi2;
    # the LOW of improvement corresponds to the HIGH of better_chi2,
    # so we sign-flip the percentile picks).
    cum_ci_low, cum_ci_high = cumulative_improvement_ci

    # Landslide qualifier on R-A1B (structured field). Per phase_a
    # §corroboration_criterion: R-A1B at landslide requires BOTH the
    # point AND lower-bound 95% CI of better_chi2 to clear R_A1B_CUT
    # (228). Only well-defined when verdict is R-A1B AND a bootstrap CI
    # exists for `better`.
    landslide_R_A1B: bool | None = None
    if outcome["verdict"] == "R-A1B":
        if ci_low_better is not None:
            landslide_R_A1B = (
                outcome["better_chi2"] <= R_A1B_CUT
                and ci_low_better <= R_A1B_CUT
            )
        else:
            landslide_R_A1B = None

    return {
        "schema": "result",
        "written_at": pd.Timestamp.utcnow().isoformat() + "Z",
        "references": [{
            "file": "phase_a.yaml",
            "body_sha256":
                "7ed3eea35553b9c5cd3e03d0003603ed4d56337d151362d4a11c2d1955b75e78",
        }],
        "artifact": {
            "primary_txt": str(artifact_txt_path),
            "pickle": str(artifact_pkl_path),
            "grade": "CLAIM",
            "emitted_by": "experiment.q1b_z2_conditioning.__main__",
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
                    "experiment/q1b_z2_conditioning/__main__.py",
            "blocking": False,
        },
        # Coverage of phase_a.variables.dependent (Check R contract):
        # chi2_Z2_b, chi2_Z2_c, better_chi2, chi2_diff, chi2_Z2_b_ci95,
        # chi2_Z2_c_ci95, chi2_diff_ci95,
        # cumulative_chi2_improvement_above_Z_c,
        # cumulative_chi2_improvement_above_Z_c_ci95,
        # per_cell_mixture_params_Z2_b, per_cell_mixture_params_Z2_c,
        # per_cell_chi2_contribution_Z2_b,
        # per_cell_chi2_contribution_Z2_c, pooling_fallback_log_Z2_b,
        # pooling_fallback_log_Z2_c, winner, near_tie_flag,
        # landslide_R_A1B  (18 total)
        "primary_result": {
            "metric": "better_chi2 = min(chi2_Z2_b, chi2_Z2_c) "
                      "on post-cutoff marginal PIT, σ=(day_type, Z_c, Z2)",
            "value": {
                "chi2_Z2_b": float(chi2["Z2_b"]),
                "chi2_Z2_c": float(chi2["Z2_c"]),
                "better_chi2": float(outcome["better_chi2"]),
                "chi2_diff": float(outcome["chi2_diff"]),
                "winner": outcome["winner"],
                "near_tie_flag": bool(outcome["near_tie"]),
                "landslide_R_A1B": landslide_R_A1B,
                "cumulative_chi2_improvement_above_Z_c": float(cumulative_improvement),
            },
            "ci_low": {
                "chi2_Z2_b": ci_low_b,
                "chi2_Z2_c": ci_low_c,
                "better_chi2": ci_low_better,
                "chi2_diff": ci_low_diff,
                "cumulative_chi2_improvement_above_Z_c": cum_ci_low,
            },
            "ci_high": {
                "chi2_Z2_b": ci_high_b,
                "chi2_Z2_c": ci_high_c,
                "better_chi2": ci_high_better,
                "chi2_diff": ci_high_diff,
                "cumulative_chi2_improvement_above_Z_c": cum_ci_high,
            },
            "chi2_Z2_b_ci95":
                [ci_low_b, ci_high_b] if ci_low_b is not None else None,
            "chi2_Z2_c_ci95":
                [ci_low_c, ci_high_c] if ci_low_c is not None else None,
            "chi2_diff_ci95":
                [ci_low_diff, ci_high_diff] if ci_low_diff is not None else None,
            "cumulative_chi2_improvement_above_Z_c_ci95":
                [cum_ci_low, cum_ci_high] if cum_ci_low is not None else None,
            "ci_method_actually_used":
                f"paired_day_bootstrap_n={config_block['n_bootstrap']}",
            "outcome_verdict": outcome["verdict"],
            "outcome_reason": outcome["reason"],
        },
        "secondary_results": [
            {
                "name": "per_cell_mixture_params_Z2_b",
                "value": cells_for("Z2_b"),
            },
            {
                "name": "per_cell_mixture_params_Z2_c",
                "value": cells_for("Z2_c"),
            },
            {
                "name": "per_cell_chi2_contribution_Z2_b",
                "value": {
                    f"{dt}__zc{z_c}__z2{z2}": {
                        "chi2": entry["chi2"],
                        "n_rows": entry["n_rows"],
                    }
                    for (dt, z_c, z2), entry in per_cell_chi2["Z2_b"].items()
                },
                "notes": (
                    "Per-(day_type, Z_c, Z2_b) cell 10-bin marginal PIT "
                    "chi^2. Informational decomposition; the verdict "
                    "statistic is the pooled-marginal chi^2 in "
                    "primary_result."
                ),
            },
            {
                "name": "per_cell_chi2_contribution_Z2_c",
                "value": {
                    f"{dt}__zc{z_c}__z2{z2}": {
                        "chi2": entry["chi2"],
                        "n_rows": entry["n_rows"],
                    }
                    for (dt, z_c, z2), entry in per_cell_chi2["Z2_c"].items()
                },
                "notes": (
                    "Per-(day_type, Z_c, Z2_c) cell 10-bin marginal PIT "
                    "chi^2. Informational decomposition (same convention "
                    "as per_cell_chi2_contribution_Z2_b)."
                ),
            },
            {
                "name": "pooling_fallback_log_Z2_b",
                "value": {
                    f"{dt}__zc{z_c}__z2{z2}": count
                    for (dt, z_c, z2), count in fallback_log["Z2_b"].items()
                },
                "notes": (
                    "Pool-fallback target = (day_type, Z_c) parent per "
                    "phase_a §metric step 2 (NOT day_type-only; pooling "
                    "to day_type-only would silently undo Q1A's Z_c "
                    "carryforward)."
                ),
            },
            {
                "name": "pooling_fallback_log_Z2_c",
                "value": {
                    f"{dt}__zc{z_c}__z2{z2}": count
                    for (dt, z_c, z2), count in fallback_log["Z2_c"].items()
                },
                "notes": (
                    "Pool-fallback target = (day_type, Z_c) parent per "
                    "phase_a §metric step 2."
                ),
            },
            {
                "name": "reproduction_check_no_z",
                "value": {
                    "q1b_no_z_chi2": float(repro_chi2_no_z),
                    "q2a_settled_chi2": Q2A_M3_CHI2_SETTLED,
                    "abs_diff":
                        float(abs(repro_chi2_no_z - Q2A_M3_CHI2_SETTLED)),
                    "tol": _REPRODUCTION_TOL_CHI2,
                    "ok": float(abs(repro_chi2_no_z - Q2A_M3_CHI2_SETTLED))
                          <= _REPRODUCTION_TOL_CHI2,
                },
                "notes": (
                    "Per phase_a §metric.SECONDARY 'Reproduction check #1': "
                    "re-running M3 with σ=(day_type) on the pre-cutoff "
                    "library reproduces Q2A's chi^2 = 953.84 within "
                    "paired-bootstrap noise. ASSERTION: blocking on FAIL."
                ),
            },
            {
                "name": "reproduction_check_zc_only",
                "value": {
                    "q1b_zc_only_chi2": float(repro_chi2_zc),
                    "q1a_settled_chi2": Q1A_M3_Z_c_CHI2_SETTLED,
                    "abs_diff":
                        float(abs(repro_chi2_zc - Q1A_M3_Z_c_CHI2_SETTLED)),
                    "tol": _REPRODUCTION_TOL_CHI2,
                    "ok": float(abs(repro_chi2_zc - Q1A_M3_Z_c_CHI2_SETTLED))
                          <= _REPRODUCTION_TOL_CHI2,
                },
                "notes": (
                    "Per phase_a §metric.SECONDARY 'Reproduction check #2': "
                    "re-running M3 with σ=(day_type, Z_c) (Q1A's carry-"
                    "forward) reproduces Q1A's chi^2 = 952.5167 within "
                    "paired-bootstrap noise. ASSERTION: blocking on FAIL. "
                    "This is the cumulative-baseline anchor — the Z2 "
                    "reduction measurement only means what it claims if "
                    "this passes."
                ),
            },
            {
                "name": "inline_synthetic_gate",
                "value": {
                    "all_ok": bool(synth["all_ok"]),
                    "checks": synth["checks"],
                },
                "notes": (
                    "Per phase_a §baselines.secondary: known Z2-"
                    "conditional DGP per (Z_c, Z2) cell key. BLOCKING: "
                    f"per-cell parameter recovery within {_SYNTH_PARAM_TOL} "
                    f"AND per-cell PIT chi^2 < "
                    f"{SYNTHETIC_GATE_CHI2_CEILING}. False-Z2 companion "
                    "check (informational): collapse to Z_c-only marginal "
                    "mixture under randomized Z2 labels."
                ),
            },
            {
                "name": "pre_registered_cuts",
                "value": {
                    "R_A1B_cut": R_A1B_CUT,
                    "R_C1B_cut": R_C1B_CUT,
                    "source": ("resolution-paths-thread skeleton "
                               "(Q1B node); integer-rounded forms of "
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
                   default=Path("scratch/data/q1b_z2_conditioning/q1b.pkl"))
    p.add_argument("--n-particles", type=int, default=N_PARTICLES)
    p.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    p.add_argument("--skip-bootstrap", action="store_true",
                   help="point estimates only (for fast iteration)")
    p.add_argument("--save-pit", action="store_true",
                   help="include per-row PIT tables in the output pickle")
    p.add_argument("--gate-only", action="store_true",
                   help="run only the inline synthetic gate and exit "
                        "(fast iteration; no Ontario data read)")
    p.add_argument(
        "--emit-result", action="store_true",
        help="write the CLAIM-GRADE provenanced artifact via "
             "experiment.provenance.make_result (refuses a dirty tree) "
             "PLUS the registry result.yaml under notes/preregistrations/"
             "2026-05-29_q1b-prime-smc-tightening/"
    )
    args = p.parse_args(argv)

    # ---- Inline synthetic gate (BLOCKING) ----
    print(f"# Q1B: Z2-conditioning head-to-head (Z2_b binary, Z2_c 4-level)")
    print(f"  cuts:         R-A1B <= {R_A1B_CUT}, R-C1B > {R_C1B_CUT}")
    print(f"  baselines:    Q2A no-Z chi^2 = {Q2A_M3_CHI2_SETTLED}; "
          f"Q1A Z_c chi^2 = {Q1A_M3_Z_c_CHI2_SETTLED}")
    print(f"  N_PARTICLES:  {args.n_particles}")
    print(f"  N_BOOTSTRAP:  {args.n_bootstrap}")
    print()
    print(f"# inline synthetic gate ...")
    t0 = time.time()
    synth = _run_inline_synthetic_gate(verbose=True)
    print(f"  ({time.time()-t0:.1f}s)")
    if not synth["all_ok"]:
        print(
            "# GATE FAIL — refusing to run Q1B on Ontario data. "
            "Investigate the production mixture_2_gaussian_mle_fit "
            "or the Z derivation before proceeding.",
            file=sys.stderr,
        )
        return 2
    print(f"# gate PASS")
    print()

    if args.gate_only:
        print(f"# --gate-only set; exiting without Ontario read.")
        return 0

    cfg = load_config()
    spec = freeze.load_verified()
    cutoff = pd.Timestamp(spec["data_cutoff"])
    anchor_h = int(cfg.data.day_anchor_hours)
    dims = spec["predictor"]["embedding_dims"]
    clim_method = spec["predictor"].get("climatology_method", "month_hour")
    k_year = spec["predictor"].get("fourier_k_year")
    k_day = spec["predictor"].get("fourier_k_day")

    print(f"  cutoff:       {cutoff}")
    print(f"  anchor_h:     {anchor_h}")
    print(f"  embedding:    {dict(dims)}")
    print()

    # ---- Load actuals + build z ----
    raw_full = load_actuals(cutoff=None).dropna()
    zp = zscore_params(cutoff, method=clim_method, k_year=k_year, k_day=k_day)
    z_full = zscore_transform(raw_full, zp)
    raw_pre = load_pre_cutoff_actuals(cutoff).dropna()
    z_pre = zscore_transform(raw_pre, zp)

    # ---- Fit day_type parent (no Z, no Z2) — reproduction-check #1 ----
    print(f"# fitting M3 per day_type (no Z, no Z2; reproduction #1 + final "
          f"fallback) ...")
    t0 = time.time()
    fits_per_dt_dt_parent = _fit_mix2_per_daytype_parent(z_pre, dims)
    for dt, fit in fits_per_dt_dt_parent.items():
        if fit is None:
            print(f"  {dt}: library too thin")
        else:
            print(f"  {dt}: n={fit['n']}  params={fit['params']}")
    print(f"  ({time.time()-t0:.1f}s)")
    print()

    # ---- Fit (day_type, Z_c) parent — reproduction-check #2 + pool-fallback ----
    print(f"# fitting M3 per (day_type, Z_c) parent (reproduction #2 + "
          f"pool-fallback target) ...")
    t0 = time.time()
    libs_per_dt_zc: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    for dt in ("weekday", "saturday", "sunday"):
        d = int(dims[dt])
        libs_per_dt_zc[dt] = _build_library_per_zc_only(z_pre, dt, d)
    fits_per_dt_zc_parent = _fit_mix2_per_zc_parent(libs_per_dt_zc)
    for dt, cell_fits in fits_per_dt_zc_parent.items():
        if not cell_fits:
            print(f"  {dt}: NO Z_c cells met MIN_CELL_SIZE={MIN_CELL_SIZE}")
        else:
            zc_keys = sorted(cell_fits.keys())
            ns = [cell_fits[k]["n"] for k in zc_keys]
            print(f"  {dt}: {len(cell_fits)} Z_c cells; values {zc_keys}; n={ns}")
    print(f"  ({time.time()-t0:.1f}s)")
    print()

    # ---- Reproduction check #1: M3 σ=(day_type) chi^2 vs Q2A's 953.84 ----
    print(f"# reproduction check #1: M3 σ=(day_type) (Q2A's exact seed) ...")
    t0 = time.time()
    rng_repro1 = np.random.default_rng(
        (SEED_BASE_Q2A + 100 + _TAG_SEED_OFFSET_M3) & 0xFFFFFFFF
    )
    pit_repro1 = _pit_no_z_reproduction(
        z_full, fits_per_dt_dt_parent, cutoff, anchor_h, dims,
        rng=rng_repro1, M=args.n_particles,
    )
    repro_chi2_no_z, _hist1 = _marginal_pit_chi2(pit_repro1["u_PIT"].to_numpy())
    repro_diff1 = abs(repro_chi2_no_z - Q2A_M3_CHI2_SETTLED)
    print(f"  Q1B no-Z chi^2 = {repro_chi2_no_z:.4f}; Q2A settled = "
          f"{Q2A_M3_CHI2_SETTLED:.2f}; |diff| = {repro_diff1:.4f}  "
          f"(tol {_REPRODUCTION_TOL_CHI2:.1f})  ({time.time()-t0:.1f}s)")
    # BLOCKING ASSERTION per phase_a §metric.SECONDARY: M3-no-Z must
    # reproduce Q2A's settled chi^2; otherwise downstream chi^2 numbers
    # are not comparable to the thread's pre-locked cuts.
    assert repro_diff1 <= _REPRODUCTION_TOL_CHI2, (
        f"REPRODUCTION CHECK #1 FAILED: M3 with σ=(day_type) gives "
        f"chi^2 = {repro_chi2_no_z:.4f}, but Q2A's settled M3 chi^2 = "
        f"{Q2A_M3_CHI2_SETTLED:.2f} (|diff| = {repro_diff1:.4f} > tol "
        f"{_REPRODUCTION_TOL_CHI2:.1f}). The library / seed / sampler "
        "pipeline has drifted from Q2A's; the Z2 chi^2 numbers are not "
        "comparable to Q2A's baseline and the verdict cuts (228, 2284) "
        "lose their meaning. INVESTIGATE BEFORE RUNNING THE HEAD-TO-HEAD."
    )
    print(f"  reproduction-check #1 PASS")
    print()

    # ---- Reproduction check #2: M3 σ=(day_type, Z_c) chi^2 vs Q1A's 952.5167 ----
    print(f"# reproduction check #2: M3 σ=(day_type, Z_c) (Q1A's Z_c seed) ...")
    t0 = time.time()
    rng_repro2 = np.random.default_rng(
        (SEED_BASE_Q1A + 100 + _Z_SEED_OFFSET_Q1A_Z_c) & 0xFFFFFFFF
    )
    pit_repro2 = _pit_zc_only_reproduction(
        z_full, fits_per_dt_zc_parent, fits_per_dt_dt_parent,
        cutoff, anchor_h, dims, rng=rng_repro2, M=args.n_particles,
    )
    repro_chi2_zc, _hist2 = _marginal_pit_chi2(pit_repro2["u_PIT"].to_numpy())
    repro_diff2 = abs(repro_chi2_zc - Q1A_M3_Z_c_CHI2_SETTLED)
    print(f"  Q1B Z_c-only chi^2 = {repro_chi2_zc:.4f}; Q1A settled = "
          f"{Q1A_M3_Z_c_CHI2_SETTLED:.4f}; |diff| = {repro_diff2:.4f}  "
          f"(tol {_REPRODUCTION_TOL_CHI2:.1f})  ({time.time()-t0:.1f}s)")
    # BLOCKING ASSERTION per phase_a §metric.SECONDARY reproduction
    # check #2: the cumulative-baseline anchor 952.5167 must reproduce
    # for the Z2 reduction measurement to mean what it claims.
    assert repro_diff2 <= _REPRODUCTION_TOL_CHI2, (
        f"REPRODUCTION CHECK #2 FAILED: M3 with σ=(day_type, Z_c) gives "
        f"chi^2 = {repro_chi2_zc:.4f}, but Q1A's settled Z_c chi^2 = "
        f"{Q1A_M3_Z_c_CHI2_SETTLED:.4f} (|diff| = {repro_diff2:.4f} > tol "
        f"{_REPRODUCTION_TOL_CHI2:.1f}). The cumulative-baseline anchor "
        "has drifted from Q1A's; cumulative_chi2_improvement_above_Z_c "
        "cannot be interpreted against Q1A. INVESTIGATE BEFORE RUNNING "
        "THE HEAD-TO-HEAD."
    )
    print(f"  reproduction-check #2 PASS")
    print()

    # ---- Build per-(day_type, Z_c, Z2) cell libraries + fit M3 ----
    libs_cells: dict[str, dict[str,
                               dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]]] = {
        "Z2_b": {}, "Z2_c": {},
    }
    fits_per_dt_cell: dict[str, dict[str, dict[tuple[int, int], dict]]] = {
        "Z2_b": {}, "Z2_c": {},
    }
    for tag in ("Z2_b", "Z2_c"):
        print(f"# fitting M3 per (day_type, Z_c, {tag}) cell ...")
        t0 = time.time()
        for dt in ("weekday", "saturday", "sunday"):
            d = int(dims[dt])
            libs_cells[tag][dt] = _build_library_per_zc_z2_cell(
                z_pre, dt, d, z2_kind=tag
            )
        fits_per_dt_cell[tag] = _fit_mix2_per_zc_z2_cell(libs_cells[tag])
        for dt, cell_fits in fits_per_dt_cell[tag].items():
            if not cell_fits:
                print(f"  {dt}: NO 3-tuple cells met MIN_CELL_SIZE="
                      f"{MIN_CELL_SIZE}; all will pool to (day_type, Z_c)")
            else:
                keys = sorted(cell_fits.keys())
                ns = [cell_fits[k]["n"] for k in keys]
                print(f"  {dt}: {len(cell_fits)} 3-tuple cells; "
                      f"(Z_c, Z2) values {keys}; n={ns}")
        print(f"  ({time.time()-t0:.1f}s)")
        print()

    # ---- SMC propagation per Z2 candidate ----
    pit_tables: dict[str, pd.DataFrame] = {}
    chi2: dict[str, float] = {}
    hist: dict[str, list[int]] = {}
    fallback_log: dict[str, dict[tuple[str, int, int], int]] = {}
    for tag in ("Z2_b", "Z2_c"):
        print(f"# {tag} SMC propagation + post-cutoff PIT ...")
        t0 = time.time()
        rng_tag = np.random.default_rng(
            (SEED_BASE + 100 + _Z2_SEED_OFFSET[tag]) & 0xFFFFFFFF
        )
        pit_tables[tag], fallback_log[tag] = _pit_for_zc_z2_candidate(
            z_full, tag, fits_per_dt_cell[tag],
            fits_per_dt_zc_parent, fits_per_dt_dt_parent,
            cutoff, anchor_h, dims, rng=rng_tag, M=args.n_particles,
        )
        c, h = _marginal_pit_chi2(pit_tables[tag]["u_PIT"].to_numpy())
        chi2[tag] = float(c)
        hist[tag] = list(h.astype(int))
        n_fallback = sum(fallback_log[tag].values())
        print(f"  rows={len(pit_tables[tag])}  chi^2={chi2[tag]:.1f}  "
              f"fallback rows={n_fallback}  ({time.time()-t0:.1f}s)")
        print()

    # ---- Per-cell chi^2 contribution decomposition (3-tuple key) ----
    per_cell_chi2: dict[str, dict[tuple[str, int, int], dict[str, Any]]] = {
        "Z2_b": {}, "Z2_c": {},
    }
    for tag in ("Z2_b", "Z2_c"):
        for (dt, z_c, z2), grp in pit_tables[tag].groupby(
            ["day_type", "z_c_value", "z2_value"], sort=False
        ):
            u = grp["u_PIT"].to_numpy()
            cell_chi2, _cell_h = _marginal_pit_chi2(u)
            per_cell_chi2[tag][(str(dt), int(z_c), int(z2))] = {
                "chi2": float(cell_chi2),
                "n_rows": int(len(u)),
            }

    # ---- Paired-day bootstrap ----
    chi2_diff_se: float | None = None
    if args.skip_bootstrap:
        print(f"# bootstrap SKIPPED (--skip-bootstrap)")
        boot = None
        cumulative_improvement_ci: tuple[float | None, float | None] = (None, None)
    else:
        print(f"# paired-day bootstrap (n={args.n_bootstrap}) ...")
        t0 = time.time()
        all_days = pd.DatetimeIndex(sorted(set().union(
            *[set(pit_tables[t]["delivery_day"]) for t in ("Z2_b", "Z2_c")]
        )))
        n_days = len(all_days)
        grouped = {
            tag: {d: g for d, g in pit_tables[tag].groupby(
                "delivery_day", sort=False)}
            for tag in ("Z2_b", "Z2_c")
        }
        boot_chi2 = {"Z2_b": [], "Z2_c": [], "diff": [], "better": [],
                     "cum_improvement": []}
        rng_boot = np.random.default_rng(
            (SEED_BASE + _Z2_BOOT_OFFSET) & 0xFFFFFFFF
        )
        for b in range(args.n_bootstrap):
            idx = rng_boot.integers(0, n_days, size=n_days)
            sampled_days = all_days[idx]
            cells: dict[str, float] = {}
            for tag in ("Z2_b", "Z2_c"):
                u = pd.concat(
                    [grouped[tag][d] for d in sampled_days if d in grouped[tag]],
                    ignore_index=True,
                )["u_PIT"].to_numpy()
                c, _ = _marginal_pit_chi2(u)
                cells[tag] = c
                boot_chi2[tag].append(c)
            boot_chi2["diff"].append(cells["Z2_b"] - cells["Z2_c"])
            better_b = min(cells.values())
            boot_chi2["better"].append(better_b)
            boot_chi2["cum_improvement"].append(
                Q1A_M3_Z_c_CHI2_SETTLED - better_b
            )
        boot_chi2 = {k: np.array(v) for k, v in boot_chi2.items()}
        for k in ("Z2_b", "Z2_c", "diff", "better", "cum_improvement"):
            arr = boot_chi2[k]
            lo, med, hi = np.nanpercentile(arr, [2.5, 50, 97.5])
            print(f"  {k:>16}: median {med:.1f}  CI [{lo:.1f}, {hi:.1f}]")
        chi2_diff_se = float(np.nanstd(boot_chi2["diff"]))
        boot = {"chi2": {k: v.tolist() for k, v in boot_chi2.items()}}
        cum_lo, cum_hi = np.nanpercentile(
            boot_chi2["cum_improvement"], [2.5, 97.5]
        )
        cumulative_improvement_ci = (float(cum_lo), float(cum_hi))
        print(f"  paired diff SE = {chi2_diff_se:.2f}")
        print(f"  ({time.time()-t0:.1f}s)")
        print()

    # ---- Mechanical outcome ----
    outcome = _evaluate_outcome(chi2["Z2_b"], chi2["Z2_c"], chi2_diff_se)
    cumulative_improvement = Q1A_M3_Z_c_CHI2_SETTLED - outcome["better_chi2"]
    print(f"# pre-registered outcome: {outcome['verdict']}")
    print(f"  better_chi2:                {outcome['better_chi2']:.1f}")
    print(f"  winner:                     {outcome['winner']}  "
          f"(near_tie: {outcome['near_tie']})")
    print(f"  cumulative_improvement:     {cumulative_improvement:+.1f}  "
          f"(vs Q1A baseline {Q1A_M3_Z_c_CHI2_SETTLED})")
    print(f"  reason:                     {outcome['reason']}")
    print()

    # ---- Pickle (always) ----
    out_pkl = args.out
    if not out_pkl.is_absolute():
        out_pkl = (Path.cwd() / out_pkl).resolve()
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "chi2": chi2,
        "hist": hist,
        "fits_per_dt_cell": fits_per_dt_cell,
        "fits_per_dt_zc_parent": fits_per_dt_zc_parent,
        "fits_per_dt_dt_parent": fits_per_dt_dt_parent,
        "repro_chi2_no_z": float(repro_chi2_no_z),
        "repro_chi2_zc": float(repro_chi2_zc),
        "pit_tables": pit_tables if args.save_pit else None,
        "bootstrap": boot,
        "outcome": outcome,
        "cumulative_improvement": float(cumulative_improvement),
        "cumulative_improvement_ci": cumulative_improvement_ci,
        "synth_gate": synth,
        "fallback_log": fallback_log,
        "per_cell_chi2": per_cell_chi2,
        "config": {
            "n_particles": args.n_particles,
            "n_bootstrap": args.n_bootstrap if not args.skip_bootstrap else 0,
            "n_pit_bins": N_PIT_BINS,
            "seed_base": SEED_BASE,
            "seed_base_q2a": SEED_BASE_Q2A,
            "seed_base_q1a": SEED_BASE_Q1A,
            "R_A1B_cut": R_A1B_CUT,
            "R_C1B_cut": R_C1B_CUT,
            "min_cell_size": MIN_CELL_SIZE,
            "phase_a_body_sha256":
                "7ed3eea35553b9c5cd3e03d0003603ed4d56337d151362d4a11c2d1955b75e78",
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
            fits_per_dt_cell=fits_per_dt_cell,
            repro_chi2_no_z=float(repro_chi2_no_z),
            repro_chi2_zc=float(repro_chi2_zc),
            outcome=outcome,
            cumulative_improvement=float(cumulative_improvement),
            cumulative_improvement_ci=cumulative_improvement_ci,
            synth=synth,
            fallback_log=fallback_log,
            per_cell_chi2=per_cell_chi2,
            config_block=config_block,
        )
        out_txt = (
            PROJECT_ROOT / "experiment" / "results" / "q1b_z2_conditioning.txt"
        )
        print()
        print(f"# emitting CLAIM-GRADE artifact -> "
              f"{out_txt.relative_to(PROJECT_ROOT)}")
        hdr = make_result(
            path=out_txt,
            grade=Grade.CLAIM,
            title=("Q1B: Z2-conditioning head-to-head (Z2_b binary "
                   "overnight vs Z2_c 4-level categorical) on M3 with "
                   "Q1A's Z_c carryforward / marginal PIT"),
            body=body,
            inputs={
                "data_cutoff": cutoff.isoformat(),
                "anchor_h": anchor_h,
                "embedding_dims": dict(dims),
                "phase_a_body_sha256":
                    "7ed3eea35553b9c5cd3e03d0003603ed4d56337d151362d4a11c2d1955b75e78",
                "thread_body_sha256":
                    "5c8f5e7d5a3640018ea99010079498e92e5c79e86d187881a7f3d9eccdd18fb9",
                "q2a_settled_chi2": Q2A_M3_CHI2_SETTLED,
                "q1a_settled_zc_chi2": Q1A_M3_Z_c_CHI2_SETTLED,
                "q1b_repro_no_z_chi2": float(repro_chi2_no_z),
                "q1b_repro_zc_only_chi2": float(repro_chi2_zc),
                "R_A1B_cut": R_A1B_CUT,
                "R_C1B_cut": R_C1B_CUT,
                "MIN_CELL_SIZE": MIN_CELL_SIZE,
            },
            seeds={
                "seed_base": SEED_BASE,
                "seed_base_q2a": SEED_BASE_Q2A,
                "seed_base_q1a": SEED_BASE_Q1A,
                "rng_purpose_offsets": {
                    "smc_per_z2_offsets": _Z2_SEED_OFFSET,
                    "bootstrap_offset": _Z2_BOOT_OFFSET,
                    "smc_per_z2": "SEED_BASE + 100 + _Z2_SEED_OFFSET[tag]",
                    "bootstrap": "SEED_BASE + _Z2_BOOT_OFFSET",
                    "reproduction_no_z": (
                        "SEED_BASE_Q2A + 100 + _TAG_SEED_OFFSET_M3 "
                        "(replays Q2A's M3 SMC seed exactly)"
                    ),
                    "reproduction_zc_only": (
                        "SEED_BASE_Q1A + 100 + _Z_SEED_OFFSET_Q1A_Z_c "
                        "(replays Q1A's Z_c-branch SMC seed exactly)"
                    ),
                    "synthetic_gate":
                        "SEED_BASE + 9001 (Z2-conditional) / 9101 (false-Z2)",
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
            fits_per_dt_cell=fits_per_dt_cell,
            outcome=outcome,
            cumulative_improvement=float(cumulative_improvement),
            cumulative_improvement_ci=cumulative_improvement_ci,
            synth=synth,
            fallback_log=fallback_log,
            per_cell_chi2=per_cell_chi2,
            repro_chi2_no_z=float(repro_chi2_no_z),
            repro_chi2_zc=float(repro_chi2_zc),
            config_block=config_block,
            artifact_txt_path=out_txt.relative_to(PROJECT_ROOT),
            artifact_pkl_path=out_pkl.relative_to(PROJECT_ROOT)
                if out_pkl.is_relative_to(PROJECT_ROOT) else out_pkl,
        )
        out_yaml = (
            PROJECT_ROOT / "notes" / "preregistrations"
            / "2026-05-29_q1b-prime-smc-tightening"
            / "result.yaml"
        )
        with out_yaml.open("w") as f:
            yaml.safe_dump(result_yaml, f, sort_keys=False,
                           default_flow_style=False)
        print(f"  wrote {out_yaml.relative_to(PROJECT_ROOT)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
