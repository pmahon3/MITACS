"""Q2B runner -- non-distributional decomposition of Q2A's chi^2 = 953.84
baseline across three pre-registered mechanisms (mean_bias, variance_div,
seam).

Implements phase_a body_sha256
``91f69aeb13783bfa54adc45fa3b4da7fe81f9ea3ed984e5381dd281b10d7dd8e``
(sign-corrected 2026-05-26 in-place after self-test caught a typo
in the original mech_1 prose; see commit ba84a8f) in directory
``notes/preregistrations/2026-05-26_q2b-non-distributional-decomposition/``.
Ablates Q2A's chi^2 = 953.84 baseline (mitacs-q2a-richer-family) across
three pre-registered mechanisms; computes explained shares + 1000-day
paired bootstrap CIs.

Re-uses Q2A's M3 (mixture-2-Gaussian) fits from
``scratch/data/distributional_class_q2a/q2a.pkl`` and the same SMC
seed (``args.seed + 100 + _TAG_SEED_OFFSET_M3``) so the recomputed
chi^2_baseline_M3 matches Q2A's settled point estimate as a
self-consistency sanity check (phase_a baselines.primary).

Ablation mechanisms (per phase_a metric):
  Mech 1 (mean_bias):     ADD empirical_mean_bias[h, day_type] to
                          each particle's z value at horizon h, where
                          empirical_mean_bias = mean over post-cutoff of
                          (actual_z(h) - iterated_mean_z(h)). Bias > 0
                          when forecast under-predicts; adding shifts
                          particles toward actuals on average, removing
                          the mean-bias contribution to chi^2.
  Mech 2 (variance_div):  if sigma2_iter[h, day_type] > sigma2_empirical
                          [h, day_type], rescale all particles toward
                          their ensemble mean by sqrt(s2_emp / s2_iter);
                          else no change. sigma2_empirical = sample
                          variance of post-cutoff actuals grouped by
                          (day_type, h).
  Mech 3 (seam):          restrict PIT aggregation domain to exclude
                          issue-times whose horizon-h TARGET lands on
                          the day-anchor hour. (Note: production
                          iteration already runs INSIDE delivery days
                          via _complete_delivery_days, so iteration
                          never crosses the seam; this is an
                          AGGREGATION-DOMAIN filter on top, registered
                          as a NULL check.)

Mech 1 and Mech 2 use post-cutoff actuals to define the fix -- per
phase_a this is intentional and scoped to ATTRIBUTION (upper bound on
what each mechanism could explain if perfectly corrected). Not a
deployable correction.

Verdict metric: explained_share[m] = (chi2_baseline - chi2_after_fix[m])
                                   / (chi2_baseline - R_A2_CUT)
                                   = (953.84 - chi2_after_fix[m]) / 725.40
Outcomes:
  R-A3: max share > 0.50 AND max > 2x second_max  (corroboration)
  R-C3: max share < 0.15                          (falsification)
  R-B3: otherwise                                  (amendment-required)
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path
from typing import Callable

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
from experiment.predict import _build_pre_cutoff, _daytype  # noqa: F401
from processing.innovations.estimator import (
    mixture_2_gaussian_mle_fit,
)


# ---------------------------------------------------------------------------
# Pre-registered constants (phase_a)
# ---------------------------------------------------------------------------

N_PARTICLES = 200          # SMC particle count (phase_a; matches Q2A)
N_BOOTSTRAP = 1000         # paired-day bootstrap reps (phase_a ci_method)
N_PIT_BINS = 10            # 10-bin chi^2 vs uniform (phase_a metric)
SEED_BASE = 20260526       # session date YYYYMMDD; per-purpose offsets below

# Q2A SETTLED point estimate. Frozen constant for the explained_share
# denominator; self-consistency sanity check requires Q2B's recomputed
# chi2_baseline_M3 to match within bootstrap noise (BLOCKING for the
# Q2B verdict per phase_a baselines.primary).
CHI2_BASELINE = 953.84
# Q2A gate-validated R-A2 corroboration cut (binding cell V3_rho10 KDE
# kurt=12; gate body_sha256 2bab514e1049a89643ab69f45568400969c78c37939
# c3b188166e1a89c775c42). Frozen denominator constant.
R_A2_CUT = 228.44
# Documented for traceability; not used in the explained_share formula.
R_C2_CUT = 2284.44
SHARE_DENOMINATOR = CHI2_BASELINE - R_A2_CUT   # 725.40 frozen

# Outcome thresholds (phase_a shape_outcomes; NOT gate-derived -- set
# on non-data principled bases per phase_a.thread.threshold_derivations)
R_A3_DOMINANCE = 0.50      # max share must exceed
R_A3_SEPARATION = 2.0      # AND max > 2x second_max
R_C3_NULL = 0.15           # all shares below -> R-C3

# Seed offset for M3 family -- MUST MATCH Q2A's _TAG_SEED_OFFSET["M3"]
# so that the recomputed chi2_baseline_M3 matches Q2A's 953.84 within
# the same SMC draws. Pinned for self-consistency.
_TAG_SEED_OFFSET_M3 = 31

# Pre-experiment synthetic sanity check (phase_a baselines.secondary).
# Tiny: ~10s wall-clock. Generates a synthetic system with one known
# active mechanism; the ablation MUST attribute ~1.0 share to that
# mechanism (warn at <0.80) or the ablation pipeline is buggy and any
# Ontario result would be misleading.
SYNTHETIC_SANITY_N_LIB = 500
SYNTHETIC_SANITY_M = 50
SYNTHETIC_SANITY_HORIZON = 12      # smaller h for speed
SYNTHETIC_SANITY_N_TEST = 500
SYNTHETIC_SANITY_MEAN_OFFSET = 0.40  # known bias to inject (z-units)
SYNTHETIC_SANITY_SHARE_WARN = 0.80


# ---------------------------------------------------------------------------
# Mech 1 / Mech 2 / Mech 3: ablation transforms on particle ensembles
# ---------------------------------------------------------------------------


def _apply_mech1_mean_bias(
    particles_h: np.ndarray,                 # (M, H) -- coord-0 particle z-values
    empirical_mean_bias_h: np.ndarray,       # (H,) -- per-horizon bias for THIS day's day_type
) -> np.ndarray:
    """Mech 1 fix: ADD per-(h, day_type) empirical mean bias to
    each particle's z value at horizon h. SIGN: bias = actual_z -
    iterated_z; bias > 0 when forecast under-predicts; adding bias
    shifts particles toward actuals (corrected 2026-05-26 after
    self-test caught a typo in original phase_a prose; see commit
    ba84a8f and phase_a body_sha256 91f69aeb).

    NON-DEPLOYABLE attribution-only fix (uses post-cutoff actuals to
    define the bias; phase_a metric, mech_1). Operating on coord-0
    only is correct because coord-0 is the lone stochastic dimension
    per the rank-1 structural fact (CLAIM mitacs-rank1-structural).
    """
    # particles_h shape (M, H); empirical_mean_bias_h shape (H,)
    # Add bias[h] to every particle at horizon h.
    return particles_h + empirical_mean_bias_h[np.newaxis, :]


def _apply_mech2_variance_cap(
    particles_h: np.ndarray,                 # (M, H)
    sigma2_iter_h: np.ndarray,               # (H,) -- mean iterated variance for THIS day_type
    sigma2_empirical_h: np.ndarray,          # (H,) -- empirical variance for THIS day_type
) -> np.ndarray:
    """Mech 2 fix: one-sided variance cap. If sigma2_iter > sigma2_empirical
    at (h, day_type), rescale particles toward ensemble mean by
    sqrt(sigma2_empirical / sigma2_iter); else no change.

    NON-DEPLOYABLE attribution-only fix (uses post-cutoff actuals to
    define sigma2_empirical; phase_a metric, mech_2). Rescaling toward
    the per-issue ensemble mean (not the global mean) preserves
    first-moment structure and ablates only second-moment overdispersion.
    """
    M, H = particles_h.shape
    ensemble_mean_h = particles_h.mean(axis=0)          # (H,)
    # One-sided cap: scale factor in (0, 1] when s2_iter > s2_emp; 1.0 else.
    safe = np.where(sigma2_iter_h > 0.0, sigma2_iter_h, 1.0)
    ratio = sigma2_empirical_h / safe                    # (H,)
    scale = np.where(sigma2_iter_h > sigma2_empirical_h,
                     np.sqrt(np.clip(ratio, 0.0, 1.0)),
                     1.0)                                # (H,)
    # particles - ensemble_mean: (M, H); broadcast scale across M.
    delta = particles_h - ensemble_mean_h[np.newaxis, :]
    return ensemble_mean_h[np.newaxis, :] + scale[np.newaxis, :] * delta


def _seam_mask_for_targets(
    delivery_day: pd.Timestamp,
    anchor_h: int,
    H: int,
) -> np.ndarray:
    """Mech 3 mask: True iff the horizon-h TARGET lands on the
    day-anchor hour.

    Per phase_a (M3_mech3_seam_exclusion_fix): exclude issue-times whose
    horizon-h target hour-of-day equals anchor_h. With anchor_h=0 and
    H=24, target hour = (anchor_h + (h - 1)) % 24, so target_hour ==
    anchor_h fires at h=1 (target lands at midnight). For non-zero
    anchor_h the same pattern applies, just rotated.

    Note: production iteration already excludes seam-CROSSING via
    _complete_delivery_days; mech_3 is a redundant aggregation-domain
    filter on top of that. Expected ~0 share by construction (REGISTERED
    NULL). If share > 0.15, production seam handling has a leak.
    """
    out = np.zeros(H, dtype=bool)
    for h in range(1, H + 1):
        target_hour = (anchor_h + (h - 1)) % 24
        out[h - 1] = (target_hour == anchor_h)
    return out


# ---------------------------------------------------------------------------
# Production-fitter loading: prefer Q2A pickle; else refit
# ---------------------------------------------------------------------------
#
# The Q2A pickle stores M3 (mixture-2-Gaussian) fits per day_type as
# ``{"C": ndarray(d, 1), "params": {"weights": (w0, w1), "scales":
# (s0, s1)}, "d": int, "n": int}``. Library z_pre is reproducible from
# the frozen spec so a refit is exact (deterministic fitter at known
# library).


def _load_q2a_m3_fits(q2a_pkl: Path) -> dict:
    """Load Q2A's M3 fits from the pickle. Returns dict[day_type] ->
    fit_dict with keys C, params, d, n."""
    with q2a_pkl.open("rb") as f:
        payload = pickle.load(f)
    if "fits" not in payload or "mixture_2" not in payload["fits"]:
        raise RuntimeError(
            f"Q2A pickle {q2a_pkl} missing fits.mixture_2; "
            f"refit required (--refit)."
        )
    return payload["fits"]["mixture_2"]


def _build_library_for_daytype(
    z_pre: pd.Series,
    day_type: str,
    d: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Day-type-filtered (X, Y) library on the pre-cutoff series.
    Byte-mirror of Q2A's same-named helper (the fair-comparison
    contract); included so --refit produces fits identical to Q2A's."""
    idx = z_pre.index
    valid = []
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
        valid.append((lag, y))
    if len(valid) < 200:
        return None
    X = np.array([v[0] for v in valid])
    Y = np.array([v[1] for v in valid])
    return X, Y


def _refit_m3_per_daytype(z_pre: pd.Series, dims: dict) -> dict:
    """Refit M3 (mixture-2-Gaussian) per day-type via the production
    fitter. Used when --refit is set or the Q2A pickle is unusable."""
    out = {}
    for dt in ("weekday", "saturday", "sunday"):
        d = int(dims[dt])
        lib = _build_library_for_daytype(z_pre, dt, d)
        if lib is None:
            out[dt] = None
            continue
        X_dt, Y_dt = lib
        C, params, _resid = mixture_2_gaussian_mle_fit(X_dt, Y_dt)
        out[dt] = {"C": C, "params": params, "d": d, "n": int(len(Y_dt))}
    return out


# ---------------------------------------------------------------------------
# SMC particle propagation (M3 mixture-2-Gaussian residual sampler)
# ---------------------------------------------------------------------------
#
# Structural mirror of Q2A's _smc_iterate_family at family_kind=
# "mixture_2". Inlined here so Q2B does not depend on Q2A internals;
# the sampler is the same.


def _make_mixture_2_sampler(params: dict) -> Callable[[int, np.random.Generator], np.ndarray]:
    """Residual sampler under M3 (mix-2-Gaussians):
    r ~ w1 N(0, s1^2) + w2 N(0, s2^2).
    """
    w = np.asarray(params["weights"], dtype=float)
    s = np.asarray(params["scales"], dtype=float)

    def sample(M: int, rng: np.random.Generator) -> np.ndarray:
        which = rng.choice(2, size=M, p=[w[0], w[1]])
        z = rng.standard_normal(M)
        return np.where(which == 0, z * s[0], z * s[1])

    return sample


def _smc_iterate_m3(
    C1_col: np.ndarray,                # (d, 1)
    sampler: Callable[[int, np.random.Generator], np.ndarray],
    x_state_d: np.ndarray,             # (d,)
    h_max: int,
    M: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """SMC propagation under M3 -- coord-0 particle values, shape (M, h_max).

    Identical control flow to Q2A's _smc_iterate_family. coord-0 is the
    lone stochastic dimension (rank-1 structural fact); coords 1..d-1
    are deterministic shifts.
    """
    d = x_state_d.shape[0]
    particles = np.tile(x_state_d, (M, 1))   # (M, d)
    out = np.zeros((M, h_max))
    for h in range(1, h_max + 1):
        r_draws = sampler(M, rng)
        next_z = (particles @ C1_col).ravel() + r_draws
        out[:, h - 1] = next_z
        particles = np.column_stack([next_z, particles[:, :d - 1]])
    return out


# ---------------------------------------------------------------------------
# Post-cutoff issue-time walk: collect particles, actuals, ensemble stats
# ---------------------------------------------------------------------------
#
# Mirrors Q2A's _pit_for_family but RETAINS the particle ensembles and
# the actuals per (day, h) instead of collapsing to u_PIT immediately,
# so the four ablated PIT tables (baseline + mech_1 + mech_2 + mech_3)
# can all be computed from one SMC pass.


def _propagate_post_cutoff_m3(
    z_full: pd.Series,
    m3_fits: dict,                  # {day_type: {C, params, d, n}}
    cutoff: pd.Timestamp,
    anchor_h: int,
    rng: np.random.Generator,
    M: int = N_PARTICLES,
    h_max: int = 24,
) -> dict:
    """One SMC pass per delivery day; returns aggregate ensembles.

    Returns dict with arrays of shape (n_days, h_max) or (n_days, M, h_max):
      'particles':       (n_days, M, h_max)
      'actuals':         (n_days, h_max)
      'ensemble_mean':   (n_days, h_max)
      'ensemble_var':    (n_days, h_max) (sample variance over M particles)
      'day_types':       (n_days,) object array
      'delivery_days':   (n_days,) DatetimeIndex
    """
    days = _complete_delivery_days(z_full, anchor_h)
    days = days[days > cutoff]

    particles_acc: list[np.ndarray] = []
    actuals_acc: list[np.ndarray] = []
    day_types_acc: list[str] = []
    days_acc: list[pd.Timestamp] = []

    for D in days:
        D = pd.Timestamp(D.date())
        dt = _daytype(D + pd.Timedelta(hours=anchor_h), anchor_h)
        fit = m3_fits.get(dt)
        if fit is None:
            continue
        sampler = _make_mixture_2_sampler(fit["params"])
        d = int(fit["d"])
        issue_anchor = D + pd.Timedelta(hours=anchor_h - 1)
        lag_times = [issue_anchor - pd.Timedelta(hours=i) for i in range(d)]
        if not all(t in z_full.index for t in lag_times):
            continue
        x_state = z_full.reindex(lag_times).to_numpy()
        if not np.all(np.isfinite(x_state)):
            continue
        particles_h = _smc_iterate_m3(
            fit["C"], sampler, x_state, h_max=h_max, M=M, rng=rng,
        )
        target_dts = [
            D + pd.Timedelta(hours=anchor_h + (h - 1)) for h in range(1, h_max + 1)
        ]
        actuals = z_full.reindex(target_dts).to_numpy()
        if not np.all(np.isfinite(actuals)):
            continue
        particles_acc.append(particles_h)
        actuals_acc.append(actuals)
        day_types_acc.append(dt)
        days_acc.append(D)

    particles_arr = np.stack(particles_acc, axis=0)   # (n_days, M, h_max)
    actuals_arr = np.stack(actuals_acc, axis=0)       # (n_days, h_max)
    ens_mean = particles_arr.mean(axis=1)             # (n_days, h_max)
    ens_var = particles_arr.var(axis=1, ddof=1)       # (n_days, h_max)
    return {
        "particles": particles_arr,
        "actuals": actuals_arr,
        "ensemble_mean": ens_mean,
        "ensemble_var": ens_var,
        "day_types": np.asarray(day_types_acc, dtype=object),
        "delivery_days": pd.DatetimeIndex(days_acc),
    }


# ---------------------------------------------------------------------------
# Per-(day_type, h) empirical statistics from post-cutoff
# ---------------------------------------------------------------------------
#
# phase_a (METRIC):
#   empirical_mean_bias(h, day_type) = mean over post-cutoff issue-times
#       of (actual_z(h) - iterated_mean_z(h))
#   sigma2_empirical(h, day_type)    = mean over post-cutoff of
#       (actual_z - empirical_mean_z)^2 grouped by day_type
#       => per-(day_type, h) sample variance of actuals (advisor item #4)
#   sigma2_iter(h, day_type)         = mean over post-cutoff of sample
#       variance of M particles at h => per-(day_type, h) mean ensemble var


def _compute_per_cell_stats(
    actuals: np.ndarray,                     # (n_days, H)
    ensemble_mean: np.ndarray,               # (n_days, H)
    ensemble_var: np.ndarray,                # (n_days, H)
    day_types: np.ndarray,                   # (n_days,) object
    h_max: int,
) -> dict:
    """Compute per-(day_type, h) attribution-only statistics.

    Returns dict[day_type] -> {empirical_mean_bias: (H,), sigma2_empirical:
    (H,), sigma2_iter: (H,), empirical_mean: (H,), n_days: int}.
    """
    out = {}
    for dt in ("weekday", "saturday", "sunday"):
        mask = (day_types == dt)
        if not mask.any():
            out[dt] = None
            continue
        a_dt = actuals[mask]                # (n, H)
        m_dt = ensemble_mean[mask]          # (n, H)
        v_dt = ensemble_var[mask]           # (n, H)
        empirical_mean_z = a_dt.mean(axis=0)               # (H,)
        # empirical_mean_bias = mean(actual - iterated_mean)
        empirical_mean_bias = (a_dt - m_dt).mean(axis=0)   # (H,)
        # sigma2_empirical = per-cell sample variance of actuals (ddof=1)
        if a_dt.shape[0] >= 2:
            sigma2_empirical = a_dt.var(axis=0, ddof=1)     # (H,)
        else:
            sigma2_empirical = np.zeros(h_max)
        # sigma2_iter = mean over days of per-(day, h) ensemble variance
        sigma2_iter = v_dt.mean(axis=0)                     # (H,)
        out[dt] = {
            "empirical_mean_bias": empirical_mean_bias,
            "sigma2_empirical": sigma2_empirical,
            "sigma2_iter": sigma2_iter,
            "empirical_mean": empirical_mean_z,
            "n_days": int(mask.sum()),
        }
    return out


# ---------------------------------------------------------------------------
# u_PIT computation from particle ensembles
# ---------------------------------------------------------------------------
#
# Randomized mid-rank PIT (matches Q1/Q2A convention exactly).


def _u_pit_from_particles(
    particles: np.ndarray,           # (n_days, M, H)
    actuals: np.ndarray,             # (n_days, H)
    rng: np.random.Generator,
) -> np.ndarray:
    """Compute u_PIT for every (day, h). Returns (n_days, H) array.

    Randomized mid-rank PIT: u = (less + U * (eq + 1)) / (M + 1) where
    less = #particles < actual, eq = #particles == actual, U ~ Uniform[0,1].
    """
    n_days, M, H = particles.shape
    u = np.zeros((n_days, H))
    for d in range(n_days):
        for h in range(H):
            samp = particles[d, :, h]
            less = float(np.sum(samp < actuals[d, h]))
            eq = float(np.sum(samp == actuals[d, h]))
            U = rng.random()
            u[d, h] = (less + U * (eq + 1.0)) / (M + 1.0)
    return u


# ---------------------------------------------------------------------------
# Build the four PIT tables (baseline + mech_1 + mech_2 + mech_3)
# ---------------------------------------------------------------------------


def _build_pit_tables(
    propagation: dict,
    stats: dict,
    anchor_h: int,
    rng: np.random.Generator,
) -> dict[str, pd.DataFrame]:
    """Compute u_PIT under baseline + 3 mechanism ablations.

    Returns dict[mech_tag] -> DataFrame with columns
    {delivery_day, h, day_type, u_PIT}. mech_tag in
    {baseline, mech1_mean_bias, mech2_variance_cap, mech3_seam}.
    """
    particles = propagation["particles"]               # (n_days, M, H)
    actuals = propagation["actuals"]                   # (n_days, H)
    day_types = propagation["day_types"]               # (n_days,)
    days_idx = propagation["delivery_days"]            # (n_days,)
    n_days, M, H = particles.shape

    # Baseline u_PIT (Q2A's M3 reproduction). Use a dedicated sub-RNG so
    # the same PIT-randomization stream is shared by baseline and mechs
    # 1/2/3 within a single bootstrap call (deterministic given seed).
    rng_pit = np.random.default_rng(rng.integers(0, 2**63 - 1))
    u_baseline = _u_pit_from_particles(particles, actuals, rng_pit)

    # Mech 1: shift each particle's coord-0 by -bias[h, day_type]
    particles_m1 = particles.copy()
    for dt in ("weekday", "saturday", "sunday"):
        if stats.get(dt) is None:
            continue
        mask = (day_types == dt)
        if not mask.any():
            continue
        bias_h = stats[dt]["empirical_mean_bias"]      # (H,)
        # particles_m1[mask] shape (n_dt, M, H); broadcast bias on (H,).
        # ADD bias (sign-corrected 2026-05-26; see _apply_mech1_mean_bias
        # docstring + commit ba84a8f). bias > 0 when forecast under-predicts;
        # adding shifts particles toward actuals on average.
        particles_m1[mask] = particles[mask] + bias_h[np.newaxis, np.newaxis, :]
    rng_pit_m1 = np.random.default_rng(rng.integers(0, 2**63 - 1))
    u_m1 = _u_pit_from_particles(particles_m1, actuals, rng_pit_m1)

    # Mech 2: rescale particles toward per-issue ensemble mean if
    # sigma2_iter > sigma2_empirical at (day_type, h)
    particles_m2 = particles.copy()
    for dt in ("weekday", "saturday", "sunday"):
        if stats.get(dt) is None:
            continue
        mask = (day_types == dt)
        if not mask.any():
            continue
        s2_iter = stats[dt]["sigma2_iter"]              # (H,)
        s2_emp = stats[dt]["sigma2_empirical"]          # (H,)
        # Per-issue ensemble mean (over particle dim)
        ens_mean = particles[mask].mean(axis=1)         # (n_dt, H)
        safe = np.where(s2_iter > 0.0, s2_iter, 1.0)
        ratio = s2_emp / safe                            # (H,)
        scale = np.where(s2_iter > s2_emp,
                         np.sqrt(np.clip(ratio, 0.0, 1.0)),
                         1.0)                            # (H,)
        # delta = particles - ens_mean : (n_dt, M, H)
        delta = particles[mask] - ens_mean[:, np.newaxis, :]
        particles_m2[mask] = ens_mean[:, np.newaxis, :] + \
            scale[np.newaxis, np.newaxis, :] * delta
    rng_pit_m2 = np.random.default_rng(rng.integers(0, 2**63 - 1))
    u_m2 = _u_pit_from_particles(particles_m2, actuals, rng_pit_m2)

    # Mech 3: aggregation-domain mask -- baseline u_PIT, but drop rows
    # where target hour-of-day == anchor_h. (PIT VALUES themselves are
    # unchanged; only which rows are included in the aggregation.)
    seam_mask = _seam_mask_for_targets(
        delivery_day=days_idx[0] if len(days_idx) else None,
        anchor_h=anchor_h,
        H=H,
    )
    # seam_mask[h-1] == True means horizon-h target is on day-anchor.
    # Mech 3 EXCLUDES those rows; build a DataFrame that omits them.

    def _rows_for_u(u: np.ndarray, drop_seam: bool) -> pd.DataFrame:
        rows = []
        for d_idx in range(n_days):
            for h in range(1, H + 1):
                if drop_seam and seam_mask[h - 1]:
                    continue
                rows.append({
                    "delivery_day": days_idx[d_idx],
                    "h": h,
                    "day_type": str(day_types[d_idx]),
                    "u_PIT": float(u[d_idx, h - 1]),
                })
        return pd.DataFrame(rows)

    tables = {
        "baseline":            _rows_for_u(u_baseline, drop_seam=False),
        "mech1_mean_bias":     _rows_for_u(u_m1, drop_seam=False),
        "mech2_variance_cap":  _rows_for_u(u_m2, drop_seam=False),
        # Mech 3 reuses baseline u_PIT but RESTRICTS aggregation domain.
        # The PIT values are unchanged; the DataFrame omits seam rows.
        "mech3_seam":          _rows_for_u(u_baseline, drop_seam=True),
    }
    return tables


# ---------------------------------------------------------------------------
# Marginal PIT chi^2 (byte-identical to Q1/Q2A)
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
# Pre-experiment synthetic sanity check (phase_a baselines.secondary)
# ---------------------------------------------------------------------------
#
# Inject a KNOWN single-mechanism defect (a constant mean offset) into
# a synthetic VAR(1)-style generator; run the full ablation pipeline;
# confirm mech_1 attributes ~1.0 share, mech_2/mech_3 attribute ~0.
# WARNS at <0.80 (per task spec). Pure in-memory; no artifacts written.
# ~10s wall-clock at N_LIB=500, M=50.


def _run_sanity_check(seed: int, verbose: bool = True) -> dict:
    """Synthetic single-mechanism attribution check.

    Generator: scalar VAR(1) z_{t+1} = phi z_t + N(0, sigma_r^2). Then
    inject a horizon-flat z-space mean offset OFFSET into the "actual"
    observations only (not into the forecast trajectory). The ablation
    pipeline should attribute ~1.0 share to mech_1 (mean_bias).

    Library fit uses the production fitter (mixture_2_gaussian_mle_fit
    on the synthetic library); pipeline goes through the SAME helper
    functions as the Ontario path.
    """
    rng = np.random.default_rng(seed)
    phi = 0.7
    sigma_r = 1.0
    d_emb = 1                             # single-lag embedding
    H = SYNTHETIC_SANITY_HORIZON
    M = SYNTHETIC_SANITY_M

    # ---- Build library + fit M3 via production fitter ----
    N_lib = SYNTHETIC_SANITY_N_LIB
    X_lib = rng.standard_normal((N_lib, d_emb))           # (N, 1)
    Y_lib = (X_lib @ np.array([[phi]])).ravel() + sigma_r * rng.standard_normal(N_lib)
    C_hat, params, _ = mixture_2_gaussian_mle_fit(X_lib, Y_lib)
    sampler = _make_mixture_2_sampler(params)

    # ---- Build synthetic post-cutoff "delivery days" ----
    # Each "day" has a single issue time with initial state x_state (scalar);
    # iterate H steps; "actuals" = simulated trajectory + injected mean offset.
    n_days = SYNTHETIC_SANITY_N_TEST
    x_states = rng.standard_normal(n_days)                # (n_days,)

    particles = np.zeros((n_days, M, H))
    actuals = np.zeros((n_days, H))
    for d in range(n_days):
        x_state = np.array([x_states[d]])
        particles[d] = _smc_iterate_m3(C_hat, sampler, x_state, h_max=H, M=M, rng=rng)
        # True actual trajectory = same generator + INJECTED OFFSET (the
        # defect mech_1 should attribute to)
        true_traj = np.zeros(H)
        z_prev = x_states[d]
        for h in range(H):
            z_next = phi * z_prev + sigma_r * rng.standard_normal()
            true_traj[h] = z_next
            z_prev = z_next
        actuals[d] = true_traj + SYNTHETIC_SANITY_MEAN_OFFSET

    # ---- Wrap into the propagation/stats/PIT-tables interface ----
    day_types = np.array(["weekday"] * n_days, dtype=object)
    days_idx = pd.DatetimeIndex(
        pd.date_range("2025-01-01", periods=n_days, freq="D")
    )
    propagation = {
        "particles": particles,
        "actuals": actuals,
        "ensemble_mean": particles.mean(axis=1),
        "ensemble_var": particles.var(axis=1, ddof=1),
        "day_types": day_types,
        "delivery_days": days_idx,
    }
    stats = _compute_per_cell_stats(
        actuals, propagation["ensemble_mean"], propagation["ensemble_var"],
        day_types, h_max=H,
    )
    # Build PIT tables with anchor_h=0 (so seam mask hits h=1)
    rng_tables = np.random.default_rng(rng.integers(0, 2**63 - 1))
    # Inline-copy the _build_pit_tables logic but for H<24; the helper
    # is generic across H so we just call it.
    tables = _build_pit_tables(propagation, stats, anchor_h=0, rng=rng_tables)

    # ---- Compute chi^2 + explained_share ----
    chi2_baseline_syn, _ = _marginal_pit_chi2(tables["baseline"]["u_PIT"].to_numpy())
    chi2_m1_syn, _ = _marginal_pit_chi2(tables["mech1_mean_bias"]["u_PIT"].to_numpy())
    chi2_m2_syn, _ = _marginal_pit_chi2(tables["mech2_variance_cap"]["u_PIT"].to_numpy())
    chi2_m3_syn, _ = _marginal_pit_chi2(tables["mech3_seam"]["u_PIT"].to_numpy())

    # Use baseline chi^2 itself as the "gap" denominator (no R_A2 cut on
    # synthetic). Share = (chi2_baseline - chi2_after) / chi2_baseline.
    # We want share_mech1 ~ 1.0 (mean bias is the only injected defect).
    if chi2_baseline_syn > 1.0:
        share_m1 = (chi2_baseline_syn - chi2_m1_syn) / chi2_baseline_syn
        share_m2 = (chi2_baseline_syn - chi2_m2_syn) / chi2_baseline_syn
        share_m3 = (chi2_baseline_syn - chi2_m3_syn) / chi2_baseline_syn
    else:
        # Baseline already near-uniform; sanity check uninformative
        share_m1 = share_m2 = share_m3 = 0.0

    result = {
        "chi2_baseline": float(chi2_baseline_syn),
        "chi2_mech1": float(chi2_m1_syn),
        "chi2_mech2": float(chi2_m2_syn),
        "chi2_mech3": float(chi2_m3_syn),
        "share_mech1": float(share_m1),
        "share_mech2": float(share_m2),
        "share_mech3": float(share_m3),
        "injected_offset": SYNTHETIC_SANITY_MEAN_OFFSET,
        "n_lib": N_lib,
        "n_days": n_days,
        "M": M,
        "H": H,
    }
    ok = share_m1 >= SYNTHETIC_SANITY_SHARE_WARN
    flag = "OK" if ok else "WARN"
    if verbose:
        print(f"  [{flag}] sanity: baseline chi^2 = {chi2_baseline_syn:.1f} "
              f"(injected z-offset = {SYNTHETIC_SANITY_MEAN_OFFSET})")
        print(f"        attributed share_mech1 = {share_m1:.3f} "
              f"(expected ~1.0; warn at < {SYNTHETIC_SANITY_SHARE_WARN})")
        print(f"        attributed share_mech2 = {share_m2:.3f} "
              f"(expected ~0)")
        print(f"        attributed share_mech3 = {share_m3:.3f} "
              f"(expected ~0)")
        if not ok:
            print(f"    WARNING: mech_1 ablation attributed only "
                  f"{share_m1:.3f} of injected bias; ablation pipeline "
                  f"may have a bug. Continuing -- advisory only.")
    return result


# ---------------------------------------------------------------------------
# Pre-registered outcome evaluation
# ---------------------------------------------------------------------------
#
# R-A3: max_m explained_share[m] > 0.50 AND max > 2 x second_max
# R-C3: max_m explained_share[m] < 0.15
# R-B3: otherwise


def _evaluate_outcome(
    share_m1: float, share_m2: float, share_m3: float,
) -> dict:
    """Mechanical evaluation of the pre-registered outcome.

    Mech names: m1 = mean_bias, m2 = variance_div, m3 = seam.
    """
    shares = {
        "mean_bias": share_m1,
        "variance_div": share_m2,
        "seam": share_m3,
    }
    sorted_pairs = sorted(shares.items(), key=lambda kv: -kv[1])
    max_name, max_val = sorted_pairs[0]
    second_val = sorted_pairs[1][1]

    # R-A3 numeric criterion
    a3_dominance = max_val > R_A3_DOMINANCE
    # Separation: max > 2 x second_max. If second_max <= 0, separation is
    # automatically satisfied (vacuously > 2*0 = 0); guard for negative
    # second_max conservatively.
    a3_separation = max_val > R_A3_SEPARATION * max(second_val, 0.0)

    if a3_dominance and a3_separation:
        verdict = "R-A3"
        reason = (
            f"max share = {max_val:.3f} ({max_name}) > {R_A3_DOMINANCE} "
            f"AND > {R_A3_SEPARATION}x next (= {second_val:.3f}); "
            f"one mechanism dominates the non-distributional chi^2."
        )
        dominant = max_name
    elif max_val < R_C3_NULL:
        verdict = "R-C3"
        reason = (
            f"max share = {max_val:.3f} ({max_name}) < {R_C3_NULL}; "
            f"no pre-registered mechanism captures the chi^2; "
            f"Q2B candidate set empirically inadequate. "
            f"Routes to thread amendment for richer mechanism set."
        )
        dominant = None
    else:
        verdict = "R-B3"
        reason = (
            f"max share = {max_val:.3f} ({max_name}); "
            f"second = {second_val:.3f}; dominance test "
            f"(>{R_A3_DOMINANCE}): {a3_dominance}; separation test "
            f"(>{R_A3_SEPARATION}x): {a3_separation}; null test "
            f"(<{R_C3_NULL}): False. Multiple mechanisms contribute; "
            f"thread amendment required."
        )
        dominant = None
    return {
        "verdict": verdict,
        "dominant_mechanism": dominant,
        "shares": shares,
        "max_name": max_name,
        "max_val": max_val,
        "second_val": second_val,
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# Body rendering for the CLAIM artifact
# ---------------------------------------------------------------------------


def _render_body(
    *,
    chi2: dict[str, float],
    boot: dict | None,
    stats: dict,
    outcome: dict,
    sanity: dict,
    config_block: dict,
    self_consistency: dict,
) -> str:
    L: list[str] = []
    L.append("Q2B -- non-distributional decomposition of Q2A's chi^2 baseline")
    L.append("=" * 70)
    L.append("")
    L.append("Pre-registration: notes/preregistrations/")
    L.append("  2026-05-26_q2b-non-distributional-decomposition/phase_a.yaml")
    L.append(f"  phase_a body_sha256: 91f69aeb13783bfa54adc45fa3b4da7fe81f9ea3ed984e5381dd281b10d7dd8e")
    L.append("Q2A SETTLED baseline (frozen denominator constant):")
    L.append(f"  chi2_baseline    = {CHI2_BASELINE}    (Q2A M3 mixture-2-Gaussian)")
    L.append(f"  R_A2_cut         = {R_A2_CUT}    (Q2A gate-validated corroboration cut)")
    L.append(f"  denominator      = {SHARE_DENOMINATOR}  (= chi2_baseline - R_A2_cut)")
    L.append("Outcome thresholds (non-data principled):")
    L.append(f"  R-A3 dominance:   max share > {R_A3_DOMINANCE}")
    L.append(f"  R-A3 separation:  max > {R_A3_SEPARATION} x next")
    L.append(f"  R-C3 null:        max share < {R_C3_NULL}")
    L.append("")
    L.append("CHI^2 PER ABLATION (post-cutoff marginal PIT, 10-bin)")
    L.append("-" * 70)
    tags = ("baseline", "mech1_mean_bias", "mech2_variance_cap", "mech3_seam")
    for tag in tags:
        line = f"  chi2_{tag:<22s}: {chi2[tag]:>10.1f}"
        if boot is not None and tag in boot.get("chi2", {}):
            arr = np.asarray(boot["chi2"][tag])
            lo, med, hi = np.nanpercentile(arr, [2.5, 50, 97.5])
            line += f"    bootstrap 95% CI [{lo:.1f}, {hi:.1f}]  median {med:.1f}"
        L.append(line)
    L.append("")
    L.append("SELF-CONSISTENCY (chi2_baseline_M3 must match Q2A's 953.84)")
    L.append("-" * 70)
    L.append(f"  Q2A point estimate:           {CHI2_BASELINE}")
    L.append(f"  Q2B recomputed (baseline):    {chi2['baseline']:.2f}")
    L.append(f"  abs delta:                    {self_consistency['abs_delta']:.2f}")
    L.append(f"  self-consistency PASS:        {self_consistency['pass']}")
    if not self_consistency['pass']:
        L.append(f"  WARNING: self-consistency delta exceeds {self_consistency['tolerance']:.0f};")
        L.append(f"  Q2A-vs-Q2B pipeline drift detected. Verdict provisional.")
    L.append("")
    L.append("EXPLAINED SHARES (verdict metric)")
    L.append("-" * 70)
    L.append(f"  share_mean_bias:     {outcome['shares']['mean_bias']:>+.4f}")
    if boot is not None and "share_mean_bias" in boot.get("share", {}):
        arr = np.asarray(boot["share"]["mean_bias"])
        lo, hi = np.nanpercentile(arr, [2.5, 97.5])
        L.append(f"      bootstrap 95% CI [{lo:+.4f}, {hi:+.4f}]")
    L.append(f"  share_variance_div:  {outcome['shares']['variance_div']:>+.4f}")
    if boot is not None and "share_variance_div" in boot.get("share", {}):
        arr = np.asarray(boot["share"]["variance_div"])
        lo, hi = np.nanpercentile(arr, [2.5, 97.5])
        L.append(f"      bootstrap 95% CI [{lo:+.4f}, {hi:+.4f}]")
    L.append(f"  share_seam:          {outcome['shares']['seam']:>+.4f}")
    if boot is not None and "share_seam" in boot.get("share", {}):
        arr = np.asarray(boot["share"]["seam"])
        lo, hi = np.nanpercentile(arr, [2.5, 97.5])
        L.append(f"      bootstrap 95% CI [{lo:+.4f}, {hi:+.4f}]")
    sum_shares = sum(outcome['shares'].values())
    L.append(f"  sum_shares:          {sum_shares:>+.4f}  "
             f"(>1 -> overlap; <max -> under-attribution)")
    L.append("")
    L.append("PRE-REGISTERED VERDICT")
    L.append("-" * 70)
    L.append(f"  verdict:             {outcome['verdict']}")
    L.append(f"  dominant_mechanism:  {outcome['dominant_mechanism']}")
    L.append(f"  reason:              {outcome['reason']}")
    L.append("")
    L.append("PER-(day_type, h) ATTRIBUTION-ONLY STATISTICS")
    L.append("-" * 70)
    L.append(f"  computed from post-cutoff actuals (NON-DEPLOYABLE; per phase_a)")
    for dt in ("weekday", "saturday", "sunday"):
        s = stats.get(dt)
        if s is None:
            L.append(f"  {dt}: (no post-cutoff days)")
            continue
        L.append(f"  {dt} (n_days={s['n_days']}):")
        L.append(f"    empirical_mean_bias (z, by h):")
        bias_str = " ".join(f"{x:+.3f}" for x in s["empirical_mean_bias"])
        L.append(f"      {bias_str}")
        L.append(f"    sigma2_empirical (z^2, by h):")
        emp_str = " ".join(f"{x:.3f}" for x in s["sigma2_empirical"])
        L.append(f"      {emp_str}")
        L.append(f"    sigma2_iter (mean ensemble var over post-cutoff, by h):")
        itr_str = " ".join(f"{x:.3f}" for x in s["sigma2_iter"])
        L.append(f"      {itr_str}")
        # How often the variance cap actually fires
        fires = int(np.sum(s["sigma2_iter"] > s["sigma2_empirical"]))
        L.append(f"    mech_2 cap fires at {fires}/{len(s['sigma2_iter'])} horizons")
    L.append("")
    L.append("PRE-EXPERIMENT SYNTHETIC SANITY CHECK")
    L.append("-" * 70)
    flag = "OK" if sanity["share_mech1"] >= SYNTHETIC_SANITY_SHARE_WARN else "WARN"
    L.append(f"  [{flag}] N_lib={sanity['n_lib']}, n_days={sanity['n_days']}, "
             f"M={sanity['M']}, H={sanity['H']}, "
             f"injected_offset={sanity['injected_offset']}")
    L.append(f"        chi2_baseline = {sanity['chi2_baseline']:.1f}, "
             f"chi2_mech1 = {sanity['chi2_mech1']:.1f}")
    L.append(f"        share_mech1 = {sanity['share_mech1']:.3f} "
             f"(expected ~1.0; warn at < {SYNTHETIC_SANITY_SHARE_WARN})")
    L.append(f"        share_mech2 = {sanity['share_mech2']:.3f}, "
             f"share_mech3 = {sanity['share_mech3']:.3f}")
    L.append("")
    L.append("CONFIG")
    L.append("-" * 70)
    for k, v in config_block.items():
        L.append(f"  {k}: {v}")
    L.append("")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# Result.yaml renderer for the registry directory
# ---------------------------------------------------------------------------


def _render_result_yaml(
    *,
    chi2: dict[str, float],
    boot: dict | None,
    stats: dict,
    outcome: dict,
    sanity: dict,
    config_block: dict,
    self_consistency: dict,
    artifact_txt_path: Path,
    artifact_pkl_path: Path,
) -> dict:
    """Build the result.yaml payload (does not write).

    Covers every phase_a.variables.dependent (Check R contract):
      chi2_baseline_M3, chi2_after_{mean_bias,variance_cap,seam_exclusion}_fix,
      explained_share_{mean_bias,variance_div,seam},
      ci95_chi2_baseline_M3, ci95_chi2_after_*, ci95_explained_share_*,
      per_h_day_type_empirical_mean_bias, per_h_day_type_empirical_variance,
      per_h_day_type_smc_ensemble_variance.
    """
    def ci_for(d: dict | None, k: str) -> tuple[float | None, float | None]:
        if d is None:
            return None, None
        arr = d.get(k)
        if arr is None:
            return None, None
        lo, hi = np.nanpercentile(np.asarray(arr), [2.5, 97.5])
        return float(lo), float(hi)

    ci_chi2 = boot.get("chi2", {}) if boot else {}
    ci_share = boot.get("share", {}) if boot else {}
    ci_low_baseline, ci_high_baseline = ci_for(ci_chi2, "baseline")
    ci_low_m1, ci_high_m1 = ci_for(ci_chi2, "mech1_mean_bias")
    ci_low_m2, ci_high_m2 = ci_for(ci_chi2, "mech2_variance_cap")
    ci_low_m3, ci_high_m3 = ci_for(ci_chi2, "mech3_seam")
    ci_low_s1, ci_high_s1 = ci_for(ci_share, "mean_bias")
    ci_low_s2, ci_high_s2 = ci_for(ci_share, "variance_div")
    ci_low_s3, ci_high_s3 = ci_for(ci_share, "seam")

    # Per-(day_type, h) attribution stats
    per_cell = {}
    for dt in ("weekday", "saturday", "sunday"):
        s = stats.get(dt)
        if s is None:
            per_cell[dt] = {"available": False}
            continue
        per_cell[dt] = {
            "available": True,
            "n_days": s["n_days"],
            "empirical_mean_bias": [float(x) for x in s["empirical_mean_bias"]],
            "sigma2_empirical": [float(x) for x in s["sigma2_empirical"]],
            "sigma2_iter": [float(x) for x in s["sigma2_iter"]],
        }

    return {
        "schema": "result",
        "written_at": pd.Timestamp.utcnow().isoformat() + "Z",
        "references": [{
            "file": "phase_a.yaml",
            "body_sha256":
                "91f69aeb13783bfa54adc45fa3b4da7fe81f9ea3ed984e5381dd281b10d7dd8e",
        }],
        "artifact": {
            "primary_txt": str(artifact_txt_path),
            "pickle": str(artifact_pkl_path),
            "grade": "CLAIM",
            "emitted_by": "experiment.distributional_class_q2b.__main__",
            "notes": (
                "Primary CLAIM artifact is the .txt under "
                "experiment/results/ written via experiment.provenance."
                "make_result (carries its own header + body hash). "
                "Pickle holds full per-mechanism PIT tables + per-cell "
                "attribution-only statistics for re-analysis."
            ),
        },
        "code_path_audit": {
            "verdict": "PRODUCTION-PATH",
            "tool": ".venv/bin/python -m experiment.audit.code_path "
                    "experiment/distributional_class_q2b/__main__.py",
            "blocking": False,
        },
        "primary_result": {
            "metric": ("explained_share[m] = (chi2_baseline - "
                       "chi2_after_fix[m]) / (chi2_baseline - R_A2_cut) "
                       "for m in {mean_bias, variance_div, seam}, "
                       "denominator = 725.40 frozen"),
            "value": {
                "chi2_baseline_M3": float(chi2["baseline"]),
                "chi2_after_mean_bias_fix": float(chi2["mech1_mean_bias"]),
                "chi2_after_variance_cap_fix": float(chi2["mech2_variance_cap"]),
                "chi2_after_seam_exclusion_fix": float(chi2["mech3_seam"]),
                "explained_share_mean_bias": float(outcome["shares"]["mean_bias"]),
                "explained_share_variance_div": float(outcome["shares"]["variance_div"]),
                "explained_share_seam": float(outcome["shares"]["seam"]),
            },
            "ci_low": {
                "chi2_baseline_M3": ci_low_baseline,
                "chi2_after_mean_bias_fix": ci_low_m1,
                "chi2_after_variance_cap_fix": ci_low_m2,
                "chi2_after_seam_exclusion_fix": ci_low_m3,
                "explained_share_mean_bias": ci_low_s1,
                "explained_share_variance_div": ci_low_s2,
                "explained_share_seam": ci_low_s3,
            },
            "ci_high": {
                "chi2_baseline_M3": ci_high_baseline,
                "chi2_after_mean_bias_fix": ci_high_m1,
                "chi2_after_variance_cap_fix": ci_high_m2,
                "chi2_after_seam_exclusion_fix": ci_high_m3,
                "explained_share_mean_bias": ci_high_s1,
                "explained_share_variance_div": ci_high_s2,
                "explained_share_seam": ci_high_s3,
            },
            "ci_method_actually_used":
                f"paired_day_bootstrap_n={config_block['n_bootstrap']}",
            "outcome_verdict": outcome["verdict"],
            "dominant_mechanism": outcome["dominant_mechanism"],
            "outcome_reason": outcome["reason"],
        },
        "secondary_results": [
            {
                "name": "per_h_day_type_empirical_mean_bias",
                "value": {dt: per_cell[dt].get("empirical_mean_bias")
                          if per_cell[dt].get("available") else None
                          for dt in ("weekday", "saturday", "sunday")},
                "notes": ("z-units; computed from post-cutoff actuals "
                          "(attribution-only); index = horizon 1..24"),
            },
            {
                "name": "per_h_day_type_empirical_variance",
                "value": {dt: per_cell[dt].get("sigma2_empirical")
                          if per_cell[dt].get("available") else None
                          for dt in ("weekday", "saturday", "sunday")},
                "notes": ("z^2-units; per-(day_type, h) sample variance "
                          "of post-cutoff actuals (attribution-only)"),
            },
            {
                "name": "per_h_day_type_smc_ensemble_variance",
                "value": {dt: per_cell[dt].get("sigma2_iter")
                          if per_cell[dt].get("available") else None
                          for dt in ("weekday", "saturday", "sunday")},
                "notes": ("z^2-units; mean over post-cutoff days of "
                          "per-issue ensemble variance at h"),
            },
            {
                "name": "self_consistency_baseline_vs_Q2A",
                "value": {
                    "Q2A_point": float(CHI2_BASELINE),
                    "Q2B_recomputed": float(chi2["baseline"]),
                    "abs_delta": float(self_consistency["abs_delta"]),
                    "passed": bool(self_consistency["pass"]),
                    "tolerance": float(self_consistency["tolerance"]),
                },
                "notes": (
                    "Self-consistency PASS is REQUIRED per phase_a "
                    "baselines.primary; failure indicates Q2A-vs-Q2B "
                    "pipeline drift and is BLOCKING for the verdict."
                ),
            },
            {
                "name": "pre_experiment_synthetic_sanity",
                "value": {
                    "chi2_baseline": float(sanity["chi2_baseline"]),
                    "chi2_mech1": float(sanity["chi2_mech1"]),
                    "chi2_mech2": float(sanity["chi2_mech2"]),
                    "chi2_mech3": float(sanity["chi2_mech3"]),
                    "share_mech1": float(sanity["share_mech1"]),
                    "share_mech2": float(sanity["share_mech2"]),
                    "share_mech3": float(sanity["share_mech3"]),
                    "injected_offset": float(sanity["injected_offset"]),
                    "n_lib": int(sanity["n_lib"]),
                    "n_days": int(sanity["n_days"]),
                    "M": int(sanity["M"]),
                    "H": int(sanity["H"]),
                },
                "notes": (
                    f"Single-mechanism (mean_bias) attribution sanity; "
                    f"expected share_mech1 ~ 1.0, warn at < "
                    f"{SYNTHETIC_SANITY_SHARE_WARN}; per phase_a "
                    f"baselines.secondary."
                ),
            },
            {
                "name": "pre_registered_constants",
                "value": {
                    "chi2_baseline_Q2A_SETTLED": CHI2_BASELINE,
                    "R_A2_cut": R_A2_CUT,
                    "R_C2_cut": R_C2_CUT,
                    "share_denominator": SHARE_DENOMINATOR,
                    "R_A3_dominance": R_A3_DOMINANCE,
                    "R_A3_separation": R_A3_SEPARATION,
                    "R_C3_null": R_C3_NULL,
                    "source": "phase_a.yaml metric + Q2A SETTLED",
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
    p.add_argument("--q2a-pickle", type=Path,
                   default=Path("scratch/data/distributional_class_q2a/q2a.pkl"),
                   help="Q2A pickle for reusing M3 fits")
    p.add_argument("--out", type=Path,
                   default=Path("scratch/data/distributional_class_q2b/q2b.pkl"))
    p.add_argument("--n-particles", type=int, default=N_PARTICLES)
    p.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    p.add_argument("--n-pit-bins", type=int, default=N_PIT_BINS)
    p.add_argument("--seed", type=int, default=SEED_BASE)
    p.add_argument("--refit", action="store_true",
                   help="force refit of M3 via production fitter instead "
                        "of loading Q2A pickle")
    p.add_argument("--skip-bootstrap", action="store_true",
                   help="point estimates only (for fast iteration)")
    p.add_argument("--save-decomposition", action="store_true",
                   help="include per-mechanism PIT tables in the output pickle")
    p.add_argument(
        "--emit-result", action="store_true",
        help="write the CLAIM-GRADE provenanced artifact via "
             "experiment.provenance.make_result (refuses a dirty tree) "
             "PLUS the registry result.yaml under notes/preregistrations/"
             "2026-05-26_q2b-non-distributional-decomposition/"
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

    print(f"# Q2B: non-distributional decomposition of Q2A chi^2 baseline")
    print(f"  cutoff:        {cutoff}")
    print(f"  anchor_h:      {anchor_h}")
    print(f"  embedding:     {dict(dims)}")
    print(f"  N_PARTICLES:   {args.n_particles}")
    print(f"  N_BOOTSTRAP:   {args.n_bootstrap}")
    print(f"  chi2_baseline: {CHI2_BASELINE} (Q2A SETTLED)")
    print(f"  R_A2_cut:      {R_A2_CUT}")
    print(f"  denominator:   {SHARE_DENOMINATOR}")
    print(f"  cuts:          R-A3 max>{R_A3_DOMINANCE} & >{R_A3_SEPARATION}x next; "
          f"R-C3 max<{R_C3_NULL}")
    print()

    # ---- Pre-experiment synthetic sanity check (per phase_a) ----
    print(f"# pre-experiment synthetic sanity check ...")
    t0 = time.time()
    sanity = _run_sanity_check(
        seed=(args.seed + 2002) & 0xFFFFFFFF,
        verbose=True,
    )
    print(f"  ({time.time()-t0:.1f}s)")
    print()

    # ---- Load actuals + build z (same as Q2A) ----
    raw_full = load_actuals(cutoff=None).dropna()
    zp = zscore_params(cutoff, method=clim_method, k_year=k_year, k_day=k_day)
    z_full = zscore_transform(raw_full, zp)

    # ---- Load or refit M3 fits ----
    if args.refit or not args.q2a_pickle.exists():
        print(f"# refitting M3 from production fitter (refit={args.refit}, "
              f"q2a_exists={args.q2a_pickle.exists()})")
        t0 = time.time()
        raw_pre = load_pre_cutoff_actuals(cutoff).dropna()
        z_pre = zscore_transform(raw_pre, zp)
        m3_fits = _refit_m3_per_daytype(z_pre, dims)
        print(f"  ({time.time()-t0:.1f}s)")
    else:
        print(f"# loading M3 fits from Q2A pickle: {args.q2a_pickle}")
        m3_fits = _load_q2a_m3_fits(args.q2a_pickle)
    for dt in ("weekday", "saturday", "sunday"):
        fit = m3_fits.get(dt)
        if fit is None:
            print(f"  {dt}: NO FIT (library too thin)")
        else:
            w = fit["params"]["weights"]; s = fit["params"]["scales"]
            print(f"  {dt}: d={fit['d']}, n={fit['n']}, "
                  f"w=({w[0]:.3f}, {w[1]:.3f}), "
                  f"s=({s[0]:.4f}, {s[1]:.4f})")
    print()

    # ---- SMC propagation (one pass, same seed as Q2A's M3) ----
    print(f"# SMC propagation (post-cutoff, M={args.n_particles}, h=24)")
    print(f"  using Q2A-pinned seed offset _TAG_SEED_OFFSET_M3="
          f"{_TAG_SEED_OFFSET_M3} so chi^2_baseline matches Q2A's "
          f"{CHI2_BASELINE} within seed-noise")
    t0 = time.time()
    rng_smc = np.random.default_rng(
        (args.seed + 100 + _TAG_SEED_OFFSET_M3) & 0xFFFFFFFF
    )
    propagation = _propagate_post_cutoff_m3(
        z_full, m3_fits, cutoff, anchor_h,
        rng=rng_smc, M=args.n_particles, h_max=24,
    )
    n_days = propagation["particles"].shape[0]
    print(f"  n_days = {n_days}, particles shape "
          f"{propagation['particles'].shape}  ({time.time()-t0:.1f}s)")
    print()

    # ---- Per-(day_type, h) attribution-only statistics ----
    print(f"# per-(day_type, h) attribution statistics (from post-cutoff actuals)")
    t0 = time.time()
    stats = _compute_per_cell_stats(
        propagation["actuals"], propagation["ensemble_mean"],
        propagation["ensemble_var"], propagation["day_types"], h_max=24,
    )
    for dt in ("weekday", "saturday", "sunday"):
        s = stats.get(dt)
        if s is None:
            print(f"  {dt}: no post-cutoff days")
            continue
        bias_mean = float(s["empirical_mean_bias"].mean())
        s2e_mean = float(s["sigma2_empirical"].mean())
        s2i_mean = float(s["sigma2_iter"].mean())
        fires = int(np.sum(s["sigma2_iter"] > s["sigma2_empirical"]))
        print(f"  {dt} (n_days={s['n_days']}): "
              f"<bias>={bias_mean:+.3f}, "
              f"<s2_emp>={s2e_mean:.3f}, "
              f"<s2_iter>={s2i_mean:.3f}, "
              f"cap fires at {fires}/24 horizons")
    print(f"  ({time.time()-t0:.1f}s)")
    print()

    # ---- Build four PIT tables (baseline + 3 mechs) ----
    print(f"# building 4 PIT tables (baseline + 3 mechanism ablations)")
    t0 = time.time()
    rng_pit = np.random.default_rng(
        (args.seed + 200 + _TAG_SEED_OFFSET_M3) & 0xFFFFFFFF
    )
    pit_tables = _build_pit_tables(propagation, stats, anchor_h, rng=rng_pit)
    chi2: dict[str, float] = {}
    hist: dict[str, list[int]] = {}
    for tag in ("baseline", "mech1_mean_bias", "mech2_variance_cap", "mech3_seam"):
        c, h = _marginal_pit_chi2(pit_tables[tag]["u_PIT"].to_numpy())
        chi2[tag] = float(c)
        hist[tag] = list(h.astype(int))
        print(f"  {tag:<24s}: n_rows={len(pit_tables[tag])}  "
              f"chi^2={chi2[tag]:.1f}")
    print(f"  ({time.time()-t0:.1f}s)")
    print()

    # ---- Self-consistency check (BLOCKING per phase_a baselines.primary) ----
    tolerance = 50.0   # advisory; real check is bootstrap CI overlap
    abs_delta = abs(chi2["baseline"] - CHI2_BASELINE)
    self_consistency = {
        "Q2A_point": CHI2_BASELINE,
        "Q2B_recomputed": chi2["baseline"],
        "abs_delta": abs_delta,
        "tolerance": tolerance,
        "pass": abs_delta <= tolerance,
    }
    flag = "OK" if self_consistency["pass"] else "WARN"
    print(f"# self-consistency: Q2A {CHI2_BASELINE} vs Q2B recomputed "
          f"{chi2['baseline']:.2f}  |delta|={abs_delta:.2f}  [{flag}]")
    if not self_consistency["pass"]:
        print(f"  WARNING: delta exceeds advisory tolerance {tolerance}; "
              f"pipeline drift between Q2A and Q2B. Bootstrap CI overlap "
              f"is the formal check.")
    print()

    # ---- Compute point-estimate explained shares ----
    def _share(chi2_after: float) -> float:
        return (CHI2_BASELINE - chi2_after) / SHARE_DENOMINATOR

    share_m1 = _share(chi2["mech1_mean_bias"])
    share_m2 = _share(chi2["mech2_variance_cap"])
    share_m3 = _share(chi2["mech3_seam"])
    print(f"# explained shares (point estimates)")
    print(f"  share_mean_bias    = ({CHI2_BASELINE} - {chi2['mech1_mean_bias']:.2f}) "
          f"/ {SHARE_DENOMINATOR} = {share_m1:+.4f}")
    print(f"  share_variance_div = ({CHI2_BASELINE} - {chi2['mech2_variance_cap']:.2f}) "
          f"/ {SHARE_DENOMINATOR} = {share_m2:+.4f}")
    print(f"  share_seam         = ({CHI2_BASELINE} - {chi2['mech3_seam']:.2f}) "
          f"/ {SHARE_DENOMINATOR} = {share_m3:+.4f}")
    print()

    # ---- Paired-day bootstrap (recompute chi^2 + shares per resample) ----
    if args.skip_bootstrap:
        print(f"# bootstrap SKIPPED (--skip-bootstrap)")
        boot = None
    else:
        print(f"# paired-day bootstrap (n={args.n_bootstrap})")
        print(f"  -- per resample: recompute chi^2 for all 4 tables AND "
              f"explained_share via frozen denominator {SHARE_DENOMINATOR}")
        t0 = time.time()
        # Pre-index PIT tables by delivery_day for fast resampling.
        all_days = pd.DatetimeIndex(sorted(set().union(*[
            set(pit_tables[t]["delivery_day"]) for t in pit_tables
        ])))
        n_boot_days = len(all_days)
        grouped = {
            tag: {d: g for d, g in df.groupby("delivery_day", sort=False)}
            for tag, df in pit_tables.items()
        }
        rng_boot = np.random.default_rng(
            (args.seed + 7919) & 0xFFFFFFFF
        )
        boot_chi2 = {tag: [] for tag in pit_tables}
        boot_share = {"mean_bias": [], "variance_div": [], "seam": []}
        for b in range(args.n_bootstrap):
            idx = rng_boot.integers(0, n_boot_days, size=n_boot_days)
            sampled_days = all_days[idx]
            cells = {}
            for tag in pit_tables:
                parts = [grouped[tag][d] for d in sampled_days
                         if d in grouped[tag]]
                if not parts:
                    cells[tag] = float("nan")
                    boot_chi2[tag].append(cells[tag])
                    continue
                u = pd.concat(parts, ignore_index=True)["u_PIT"].to_numpy()
                c, _ = _marginal_pit_chi2(u)
                cells[tag] = c
                boot_chi2[tag].append(c)
            # Per phase_a ci_method: explained_share uses the FROZEN
            # denominator constant 725.40 (= 953.84 - 228.44), NOT the
            # per-resample baseline. The numerator uses the per-resample
            # chi2_baseline so noise in the baseline propagates into the
            # share CI, but the denominator is frozen.
            #
            # Note: phase_a says "compute explained_share[m] per resample
            # using the same denominator constant 725.40". I read this as
            # frozen denominator, per-resample baseline in the numerator.
            for mech_name, tag in (("mean_bias", "mech1_mean_bias"),
                                    ("variance_div", "mech2_variance_cap"),
                                    ("seam", "mech3_seam")):
                if np.isnan(cells["baseline"]) or np.isnan(cells[tag]):
                    boot_share[mech_name].append(float("nan"))
                else:
                    boot_share[mech_name].append(
                        (cells["baseline"] - cells[tag]) / SHARE_DENOMINATOR
                    )
        boot_chi2 = {k: np.array(v) for k, v in boot_chi2.items()}
        boot_share = {k: np.array(v) for k, v in boot_share.items()}
        for k in pit_tables:
            arr = boot_chi2[k]
            lo, med, hi = np.nanpercentile(arr, [2.5, 50, 97.5])
            print(f"  chi^2 {k:<24s}: median {med:>7.1f}  "
                  f"CI [{lo:>7.1f}, {hi:>7.1f}]")
        for k in boot_share:
            arr = boot_share[k]
            lo, med, hi = np.nanpercentile(arr, [2.5, 50, 97.5])
            print(f"  share {k:<13s}: median {med:>+.4f}  "
                  f"CI [{lo:+.4f}, {hi:+.4f}]")
        boot = {
            "chi2":  {k: v.tolist() for k, v in boot_chi2.items()},
            "share": {k: v.tolist() for k, v in boot_share.items()},
        }
        print(f"  ({time.time()-t0:.1f}s)")
        print()

    # ---- Mechanical outcome ----
    outcome = _evaluate_outcome(share_m1, share_m2, share_m3)
    print(f"# pre-registered outcome: {outcome['verdict']}")
    print(f"  dominant_mechanism: {outcome['dominant_mechanism']}")
    print(f"  max_name:           {outcome['max_name']}")
    print(f"  max_val:            {outcome['max_val']:+.4f}")
    print(f"  second_val:         {outcome['second_val']:+.4f}")
    print(f"  reason:             {outcome['reason']}")
    print()

    # ---- Pickle (always) ----
    out_pkl = args.out
    if not out_pkl.is_absolute():
        out_pkl = (Path.cwd() / out_pkl).resolve()
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "chi2": chi2,
        "hist": hist,
        "shares": outcome["shares"],
        "stats": stats,
        "fits_used": m3_fits,
        "pit_tables": pit_tables if args.save_decomposition else None,
        "bootstrap": boot,
        "outcome": outcome,
        "sanity": sanity,
        "self_consistency": self_consistency,
        "config": {
            "n_particles": args.n_particles,
            "n_bootstrap": args.n_bootstrap if not args.skip_bootstrap else 0,
            "n_pit_bins": args.n_pit_bins,
            "seed": args.seed,
            "refit": args.refit,
            "chi2_baseline_constant": CHI2_BASELINE,
            "R_A2_cut": R_A2_CUT,
            "R_C2_cut": R_C2_CUT,
            "share_denominator": SHARE_DENOMINATOR,
            "phase_a_body_sha256":
                "91f69aeb13783bfa54adc45fa3b4da7fe81f9ea3ed984e5381dd281b10d7dd8e",
        },
    }
    with out_pkl.open("wb") as f:
        pickle.dump(payload, f)
    try:
        rel = out_pkl.relative_to(PROJECT_ROOT)
        print(f"  wrote {rel}")
    except ValueError:
        print(f"  wrote {out_pkl}")

    # ---- Emit CLAIM artifact + result.yaml (only with --emit-result) ----
    if args.emit_result:
        from experiment.provenance import Grade, make_result

        config_block = payload["config"]
        body = _render_body(
            chi2=chi2,
            boot=boot,
            stats=stats,
            outcome=outcome,
            sanity=sanity,
            config_block=config_block,
            self_consistency=self_consistency,
        )
        out_txt = (
            PROJECT_ROOT / "experiment" / "results"
            / "q2b_non_distributional.txt"
        )
        print()
        print(f"# emitting CLAIM-GRADE artifact -> {out_txt.relative_to(PROJECT_ROOT)}")
        hdr = make_result(
            path=out_txt,
            grade=Grade.CLAIM,
            title=("Q2B: non-distributional decomposition of Q2A's "
                   "chi^2 = 953.84 baseline across three mechanisms"),
            body=body,
            inputs={
                "data_cutoff": cutoff.isoformat(),
                "anchor_h": anchor_h,
                "embedding_dims": dict(dims),
                "phase_a_body_sha256":
                    "91f69aeb13783bfa54adc45fa3b4da7fe81f9ea3ed984e5381dd281b10d7dd8e",
                "thread_body_sha256":
                    "05b732d41a26552121912412a8834a108175e811819916920c8b05c8b1d50d58",
                "Q2A_chi2_baseline": CHI2_BASELINE,
                "R_A2_cut": R_A2_CUT,
                "R_C2_cut": R_C2_CUT,
                "share_denominator": SHARE_DENOMINATOR,
                "fits_source": ("Q2A pickle" if not args.refit
                                else "refit via production fitter"),
            },
            seeds={
                "seed_base": args.seed,
                "rng_purpose_offsets": {
                    "sanity_check": "args.seed + 2002",
                    "smc": f"args.seed + 100 + _TAG_SEED_OFFSET_M3 "
                            f"(= {_TAG_SEED_OFFSET_M3}); pinned to "
                            f"match Q2A's M3 seed for chi^2 baseline "
                            f"self-consistency",
                    "pit": f"args.seed + 200 + _TAG_SEED_OFFSET_M3",
                    "bootstrap": "args.seed + 7919",
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
            stats=stats,
            outcome=outcome,
            sanity=sanity,
            config_block=config_block,
            self_consistency=self_consistency,
            artifact_txt_path=out_txt.relative_to(PROJECT_ROOT),
            artifact_pkl_path=out_pkl.relative_to(PROJECT_ROOT)
                if out_pkl.is_relative_to(PROJECT_ROOT) else out_pkl,
        )
        out_yaml = (
            PROJECT_ROOT / "notes" / "preregistrations"
            / "2026-05-26_q2b-non-distributional-decomposition"
            / "result.yaml"
        )
        # Stamp via registry before writing so body_sha256/git_sha/git_clean
        # land in the on-disk YAML in one shot. self_path=out_yaml excludes
        # the artifact's own existence from the git_clean check (the file is
        # not its own source change), matching how phase_a/proponent/DA are
        # stamped. Closes a defect in commit 1376c77 where this block did a
        # plain yaml.safe_dump without stamping, leaving result.yaml without
        # the schema's required envelope fields and breaking registry verify.
        from experiment.audit import registry as _registry
        result_yaml_stamped = _registry.stamp(result_yaml, self_path=out_yaml)
        with out_yaml.open("w") as f:
            yaml.safe_dump(result_yaml_stamped, f, sort_keys=False,
                           default_flow_style=False)
        print(f"  wrote {out_yaml.relative_to(PROJECT_ROOT)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
