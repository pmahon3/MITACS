"""Q2A runner -- richer-family residual law (mix-2 / mix-3 / KDE) vs
Q1's Student-t baseline on the marginal-PIT calibration gap.

Implements phase_a body_sha256
``c0a22c9e6cc708d3089a891a1eb7900e20e05e8816c673cad7bc77165bf138fe``
in directory
``notes/preregistrations/2026-05-26_q2a-richer-family-mixture-or-nonparametric/``.
Cuts inherited from the synthetic gate body_sha256
``2bab514e1049a89643ab69f45568400969c78c37939c3b188166e1a89c775c42``
in ``experiment/results/q2a_synthetic_gate.txt`` (METHOD/DESIGN).

Three families M3/M4/M5 are fit per day-type on the pre-cutoff
library via the production fitters
``processing.innovations.estimator.{mixture_2_gaussian_mle_fit,
mixture_3_gaussian_mle_fit, kde_residual_fit}``. SMC particle
propagation + PIT scoring are inline application logic, byte-mirroring
Q1's M1 ``_smc_iterate`` / ``_pit_M1`` pattern -- only the residual
sampler swaps, per the phase_a fairness contract.

The "M0" Gaussian baseline (closed-form analytic PIT from v2_final.pkl
factors) and the "M1" Student-t kernel kept as Q1's settled reference
baselines, NOT recomputed here -- their chi^2 values are loaded from
the Q1 pickle if available and printed alongside.

Verdict metric: ``effective_chi2 = min(chi2_M3, chi2_M4, chi2_M5)``
against the pre-registered cuts R-A2 (228.44) / R-C2 (2284.44).
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
    kde_residual_fit,
    mixture_2_gaussian_mle_fit,
    mixture_3_gaussian_mle_fit,
)
from processing.innovations.validation.q2a_synthetic_gate import (
    _solve_mix2_s2,
    _solve_mix3_s3,
)


# ---------------------------------------------------------------------------
# Pre-registered constants (phase_a + synthetic gate)
# ---------------------------------------------------------------------------

N_PARTICLES = 200          # SMC particle count for iteration (phase_a)
N_BOOTSTRAP = 1000         # paired-day bootstrap reps (phase_a ci_method)
N_PIT_BINS = 10            # 10-bin chi^2 vs uniform (phase_a metric)
SEED_BASE = 20260526       # session date YYYYMMDD; per-purpose offsets below

# Gate-validated cuts -- from synthetic gate body_sha256
# 2bab514e1049a89643ab69f45568400969c78c37939c3b188166e1a89c775c42 in
# experiment/results/q2a_synthetic_gate.txt (METHOD/DESIGN). DO NOT
# tighten or relax without rerunning the gate.
R_A2_CUT = 228.44          # corroboration: effective_chi2 <= R_A2_CUT
R_C2_CUT = 2284.44         # falsification: effective_chi2 >  R_C2_CUT
# R-B2 band = (R_A2_CUT, R_C2_CUT]

# Headline cell for per-family fitted-parameter reporting (phase_a
# secondary). Marginal chi^2 is pooled across all (day_type, hour); the
# headline cell appears only in the "fitted-params at headline" report.
HEADLINE_DAYTYPE = "weekday"

# Pre-experiment same-family sanity check (phase_a baselines.secondary).
# Tiny so the smoke test stays under ~10s; this is NOT the gate (the
# gate is body_sha256 2bab514e..., already run, results-pinned).
SYNTHETIC_SANITY_N = 2_000
SYNTHETIC_SANITY_M = 50
SYNTHETIC_SANITY_KURT = 12.0   # gate's lowest-kurt cell (binding R-A2 cell)
SYNTHETIC_SANITY_CHI2_CEILING = 60.0  # one-rep chi^2 sanity ceiling at
#   N=2000, M=50 (gate same-family medians for V1 at kurt=12 sit ~15;
#   we warn -- not fail -- above 60.0, which is 4x the gate median).

# Deterministic seed offsets. Python's built-in ``hash()`` is process-
# randomized (PYTHONHASHSEED=random by default) and would make the
# recorded ``seed`` in the provenance header NOT reproduce the
# published chi^2 numbers on re-run -- exactly the integrity hole
# make_result is supposed to close. Static dicts pin the RNG streams.
_TAG_SEED_OFFSET = {"M3": 31, "M4": 41, "M5": 59}
_FAM_SEED_OFFSET = {"mixture_2": 23, "mixture_3": 37, "kde": 53}


# ---------------------------------------------------------------------------
# Family registry: name -> (fitter, sampler-factory)
# ---------------------------------------------------------------------------
#
# Each entry binds the production fitter to a residual sampler that
# turns ``params`` into a callable ``sample(M, rng) -> ndarray``. The
# SMC propagation routine consumes the sampler only; the families are
# pluggable on this single axis.


def _make_mixture_2_sampler(params: dict) -> Callable[[int, np.random.Generator], np.ndarray]:
    """Residual sampler under M3 (mix-2-Gaussians).

    ``r ~ w1 N(0, s1^2) + w2 N(0, s2^2)`` with weights and scales from
    the production fitter.
    """
    w = np.asarray(params["weights"], dtype=float)
    s = np.asarray(params["scales"], dtype=float)

    def sample(M: int, rng: np.random.Generator) -> np.ndarray:
        which = rng.choice(2, size=M, p=[w[0], w[1]])
        z = rng.standard_normal(M)
        return np.where(which == 0, z * s[0], z * s[1])

    return sample


def _make_mixture_3_sampler(params: dict) -> Callable[[int, np.random.Generator], np.ndarray]:
    """Residual sampler under M4 (mix-3-Gaussians).

    ``r ~ sum_k w_k N(0, s_k^2)``, k=1..3.
    """
    w = np.asarray(params["weights"], dtype=float)
    s = np.asarray(params["scales"], dtype=float)

    def sample(M: int, rng: np.random.Generator) -> np.ndarray:
        which = rng.choice(3, size=M, p=list(w))
        z = rng.standard_normal(M)
        return z * s[which]

    return sample


def _make_kde_sampler(params: dict) -> Callable[[int, np.random.Generator], np.ndarray]:
    """Residual sampler under M5 (KDE residual law).

    Standard KDE sampling under the convolution rule: draw an index
    uniformly from the fitted residual reservoir + add Gaussian noise
    at the fitted bandwidth.
    """
    samples = np.asarray(params["samples"], dtype=float)
    bw = float(params["bandwidth"])
    n = len(samples)

    def sample(M: int, rng: np.random.Generator) -> np.ndarray:
        idx = rng.integers(0, n, size=M)
        return samples[idx] + bw * rng.standard_normal(M)

    return sample


FAMILY_REGISTRY: dict[str, tuple[Callable, Callable]] = {
    "mixture_2": (mixture_2_gaussian_mle_fit, _make_mixture_2_sampler),
    "mixture_3": (mixture_3_gaussian_mle_fit, _make_mixture_3_sampler),
    "kde":       (kde_residual_fit,           _make_kde_sampler),
}

# Tag mapping the phase_a model names to family keys.
MODEL_FAMILY: dict[str, str] = {
    "M3": "mixture_2",
    "M4": "mixture_3",
    "M5": "kde",
}


# ---------------------------------------------------------------------------
# Library construction + production fit per day-type
# ---------------------------------------------------------------------------
#
# Mirrors Q1's _fit_t_kernels_per_daytype (lines 130-196) -- same
# day-type-filtered index walk, same contiguity guard, same minimum-N
# floor. Only the fitter swap differs. Factored so the three families
# share the library build (the fair-comparison contract).


def _build_library_for_daytype(
    z_pre: pd.Series,
    day_type: str,
    d: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Day-type-filtered (X, Y) library on the pre-cutoff series.

    X[i] = (z_{t-0}, z_{t-1}, ..., z_{t-(d-1)}) at anchor t;
    Y[i] = z_{t+1}. Day-type filter applied at the ANCHOR (matches Q1).
    """
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


def _fit_family_per_daytype(
    family_kind: str,
    z_pre: pd.Series,
    dims: dict,
) -> dict:
    """For each day-type, fit the named family via the production
    fitter on the pre-cutoff library.

    Returns ``{day_type: {"C": ndarray(d,1), "params": dict, "d": int,
    "n": int}}`` (or ``None`` per day-type if the library is too thin).
    """
    fitter, _sampler_factory = FAMILY_REGISTRY[family_kind]
    out = {}
    for dt in ("weekday", "saturday", "sunday"):
        d = int(dims[dt])
        lib = _build_library_for_daytype(z_pre, dt, d)
        if lib is None:
            out[dt] = None
            continue
        X_dt, Y_dt = lib
        C, params, _resid = fitter(X_dt, Y_dt)
        out[dt] = {
            "C": C,
            "params": params,
            "d": d,
            "n": int(len(Y_dt)),
        }
    return out


# ---------------------------------------------------------------------------
# SMC particle propagation -- family-pluggable
# ---------------------------------------------------------------------------
#
# Structural mirror of Q1's _smc_iterate (lines 199-230). Only the
# residual sampler swaps. coord-0 is the lone stochastic dimension per
# the rank-1 structural fact (CLAIM mitacs-rank1-structural); coords
# 1..d-1 are deterministic shifts. M particles per state.


def _smc_iterate_family(
    C1_col: np.ndarray,   # (d, 1) coord-0 row of C from the family fit
    sampler: Callable[[int, np.random.Generator], np.ndarray],
    x_state_d: np.ndarray,   # (d,) initial state
    h_max: int,
    M: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generic SMC propagation under any family. Returns coord-0
    particle values of shape ``(M, h_max)``.

    At each step h:
      coord-0 of next state = (particles @ C1_col).ravel() + sampler(M, rng)
      coords 1..d-1 shift by one (per rank-1 structural)

    Identical control flow to Q1's M1 propagation; only the residual
    draw line swaps the Student-t rvs for ``sampler(M, rng)``.
    """
    d = x_state_d.shape[0]
    particles = np.tile(x_state_d, (M, 1))   # (M, d)
    out = np.zeros((M, h_max))
    for h in range(1, h_max + 1):
        r_draws = sampler(M, rng)             # family-pluggable residuals
        next_z = (particles @ C1_col).ravel() + r_draws
        out[:, h - 1] = next_z
        particles = np.column_stack([next_z, particles[:, :d - 1]])
    return out


# ---------------------------------------------------------------------------
# Per-issue-time PIT table -- family-pluggable
# ---------------------------------------------------------------------------
#
# Mirrors Q1's _pit_M1 (lines 233-290): post-cutoff delivery days,
# 24 horizons per day, randomized mid-rank PIT on the particle
# distribution. Day-anchor handling identical to Q1.


def _pit_for_family(
    z_full: pd.Series,
    family_fit: dict,         # output of _fit_family_per_daytype
    family_kind: str,
    cutoff: pd.Timestamp,
    anchor_h: int,
    dims: dict,
    rng: np.random.Generator,
    M: int = N_PARTICLES,
) -> pd.DataFrame:
    """Build the per-issue-time PIT table under the named family."""
    _fitter, sampler_factory = FAMILY_REGISTRY[family_kind]
    days = _complete_delivery_days(z_full, anchor_h)
    days = days[days > cutoff]
    rows = []
    for D in days:
        D = pd.Timestamp(D.date())
        dt = _daytype(D + pd.Timedelta(hours=anchor_h), anchor_h)
        fit = family_fit.get(dt)
        if fit is None:
            continue
        sampler = sampler_factory(fit["params"])
        d = int(dims[dt])
        issue_anchor = D + pd.Timedelta(hours=anchor_h - 1)
        lag_times = [issue_anchor - pd.Timedelta(hours=i) for i in range(d)]
        if not all(t in z_full.index for t in lag_times):
            continue
        x_state = z_full.reindex(lag_times).to_numpy()
        if not np.all(np.isfinite(x_state)):
            continue
        particles_h = _smc_iterate_family(
            fit["C"], sampler, x_state, h_max=24, M=M, rng=rng,
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
            # Randomized mid-rank PIT (same convention as Q1's _pit_M1)
            less = float(np.sum(samp < actuals[i]))
            eq = float(np.sum(samp == actuals[i]))
            U = rng.random()
            u = (less + U * (eq + 1.0)) / (M + 1.0)
            rows.append({
                "delivery_day": D,
                "h": h,
                "day_type": dt,
                "u_PIT": float(u),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Marginal PIT chi^2
# ---------------------------------------------------------------------------
#
# Byte-identical to Q1's _marginal_pit_chi2 -- the comparability of
# Q2A's chi^2 numbers to Q1's M0/M1/M2 chi^2 depends on this.


def _marginal_pit_chi2(u: np.ndarray, n_bins: int = N_PIT_BINS) -> tuple[float, np.ndarray]:
    """10-bin chi^2 vs uniform on (0, 1)."""
    u = u[np.isfinite(u)]
    bins = np.linspace(0, 1, n_bins + 1)
    hist, _ = np.histogram(u, bins=bins)
    expected = len(u) / n_bins
    chi2 = float(((hist - expected) ** 2 / expected).sum())
    return chi2, hist


# ---------------------------------------------------------------------------
# Pre-experiment same-family sanity check (phase_a baselines.secondary)
# ---------------------------------------------------------------------------
#
# Runs on every invocation as a fast smoke test. Pure synthetic:
# draws library from a known generator that matches the family,
# fits with the production fitter, runs the SMC propagation, computes
# chi^2 on a fresh test draw, prints a WARNING if chi^2 is unexpectedly
# high (which would indicate a fit-vs-sampling-vs-evaluation pipeline
# bug). Does NOT write any artifacts; does NOT touch real data; does
# NOT block the run.


def _sanity_check_family(
    family_kind: str,
    n_lib: int,
    M: int,
    seed: int,
) -> dict:
    """Same-family null sanity check for one family at one seed.

    Generator: a centred mix-2 at kurt = SYNTHETIC_SANITY_KURT (the
    gate's R-A2 binding cell). For KDE the generator is also a mix-2
    reservoir (matches the gate's KDE-on-V1 convention). Fitter is
    the production family fitter. Test draws and PIT use the
    randomized-mid-rank convention matching production.
    """
    rng = np.random.default_rng(seed)
    d_emb = 2
    # Build a mix-2 generator at the target kurt (V1 spec: w_r = 0.05,
    # s1 = 1, s2 from _solve_mix2_s2). For mix-3, we add a third
    # component at the V1 mix-3 spec; for KDE, the generator stays mix-2.
    if family_kind in ("mixture_2", "kde"):
        w_r = 0.05
        s2 = _solve_mix2_s2(SYNTHETIC_SANITY_KURT, w_r)
        w = np.array([1.0 - w_r, w_r])
        s = np.array([1.0, s2])

        def draw_resid(n: int, _rng: np.random.Generator) -> np.ndarray:
            which = _rng.choice(2, size=n, p=w)
            z = _rng.standard_normal(n)
            return z * s[which]
    elif family_kind == "mixture_3":
        # V1 mix-3 spec from gate: weights (0.6, 0.3, 0.1), s1=1, s2=2,
        # s3 solved from _solve_mix3_s3 at target kurt.
        w_tuple = (0.6, 0.3, 0.1)
        s1, s2 = 1.0, 2.0
        s3 = _solve_mix3_s3(SYNTHETIC_SANITY_KURT, w_tuple, (s1, s2))
        w = np.array(w_tuple)
        s = np.array([s1, s2, s3])

        def draw_resid(n: int, _rng: np.random.Generator) -> np.ndarray:
            which = _rng.choice(3, size=n, p=w)
            z = _rng.standard_normal(n)
            return z * s[which]
    else:
        raise ValueError(f"unknown family_kind {family_kind!r}")

    # True linear map (use the gate's c_true so the sanity check
    # exercises non-trivial drift).
    c_true = np.array([[0.7], [0.2]])

    # Library
    X_lib = rng.standard_normal((n_lib, d_emb))
    r_lib = draw_resid(n_lib, rng)
    Y_lib = (X_lib @ c_true).ravel() + r_lib

    # Production fit
    fitter, sampler_factory = FAMILY_REGISTRY[family_kind]
    C_hat, params, _ = fitter(X_lib, Y_lib)

    # Test states + PIT loop (matches gate's _one_rep)
    n_test = n_lib
    X_test = rng.standard_normal((n_test, d_emb))
    means_pred = (X_test @ C_hat).ravel()
    true_means = (X_test @ c_true).ravel()
    actuals = true_means + draw_resid(n_test, rng)

    sampler = sampler_factory(params)
    pit = np.empty(n_test)
    for i in range(n_test):
        particles = means_pred[i] + sampler(M, rng)
        less = float(np.sum(particles < actuals[i]))
        eq = float(np.sum(particles == actuals[i]))
        U = rng.random()
        pit[i] = (less + U * (eq + 1.0)) / (M + 1.0)

    chi2, hist = _marginal_pit_chi2(pit)
    return {
        "family_kind": family_kind,
        "chi2": chi2,
        "hist": hist.astype(int).tolist(),
        "n_lib": n_lib,
        "M": M,
        "fit_summary": str(params)[:200],
    }


def _run_sanity_checks(verbose: bool = True) -> dict:
    """Run the per-family same-family null check; warn if any chi^2
    exceeds the ceiling. Pure in-memory; no artifacts written."""
    out = {}
    for fam in ("mixture_2", "mixture_3", "kde"):
        res = _sanity_check_family(
            family_kind=fam,
            n_lib=SYNTHETIC_SANITY_N,
            M=SYNTHETIC_SANITY_M,
            seed=(SEED_BASE + 1001 + _FAM_SEED_OFFSET[fam]) & 0xFFFFFFFF,
        )
        out[fam] = res
        ok = res["chi2"] <= SYNTHETIC_SANITY_CHI2_CEILING
        flag = "OK" if ok else "WARN"
        if verbose:
            print(f"  [{flag}] sanity({fam}): chi^2 = {res['chi2']:.1f} "
                  f"(ceiling {SYNTHETIC_SANITY_CHI2_CEILING:.0f}, "
                  f"n_lib={res['n_lib']}, M={res['M']})")
        if not ok:
            print(f"    WARNING: {fam} same-family chi^2 above sanity "
                  f"ceiling; fit/SMC/PIT pipeline may have drift. "
                  f"Continuing -- this is advisory, not blocking.")
    return out


# ---------------------------------------------------------------------------
# Pre-registered outcome evaluation
# ---------------------------------------------------------------------------
#
# Three mutually-exclusive outcomes per phase_a shape_outcomes:
#   R-A2:  effective_chi2 <= 228.44
#   R-B2:  228.44 < effective_chi2 <= 2284.44   (ambiguous; thread amendment)
#   R-C2:  effective_chi2 >  2284.44             (falsification)


def _evaluate_outcome(
    chi2_M3: float, chi2_M4: float, chi2_M5: float,
) -> dict:
    """Mechanical evaluation of the pre-registered outcome.

    Returns ``{"verdict": "R-A2"|"R-B2"|"R-C2", "effective_chi2": float,
    "reason": str}``. The arbiter renders the substantive interpretation
    separately; this is the mechanical mapping only.
    """
    chis = {"M3": chi2_M3, "M4": chi2_M4, "M5": chi2_M5}
    effective = min(chis.values())
    preferred = min(chis, key=chis.get)
    if effective <= R_A2_CUT:
        verdict = "R-A2"
        reason = (
            f"effective_chi2 = min(M3={chi2_M3:.1f}, M4={chi2_M4:.1f}, "
            f"M5={chi2_M5:.1f}) = {effective:.1f} <= R-A2_cut {R_A2_CUT:.2f}; "
            f"preferred family {preferred}; "
            f"richer-than-Student-t family closes the marginal PIT gap."
        )
    elif effective <= R_C2_CUT:
        verdict = "R-B2"
        reason = (
            f"effective_chi2 = min(M3={chi2_M3:.1f}, M4={chi2_M4:.1f}, "
            f"M5={chi2_M5:.1f}) = {effective:.1f} in (R-A2_cut "
            f"{R_A2_CUT:.2f}, R-C2_cut {R_C2_CUT:.2f}]; "
            f"preferred family {preferred}; "
            f"richer families help but do not fully calibrate; "
            f"thread amendment required."
        )
    else:
        verdict = "R-C2"
        reason = (
            f"effective_chi2 = min(M3={chi2_M3:.1f}, M4={chi2_M4:.1f}, "
            f"M5={chi2_M5:.1f}) = {effective:.1f} > R-C2_cut {R_C2_CUT:.2f}; "
            f"preferred family {preferred}; "
            f"even the most flexible family cannot close the marginal "
            f"pathology -- distributional class is not the dominant "
            f"missing content."
        )
    return {
        "verdict": verdict,
        "effective_chi2": effective,
        "preferred_family": preferred,
        "per_model_chi2": chis,
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# Body rendering for the CLAIM artifact (experiment/results/q2a_richer_family.txt)
# ---------------------------------------------------------------------------


def _render_body(
    *,
    chi2: dict[str, float],
    hist: dict[str, list[int]],
    boot: dict | None,
    fits: dict[str, dict],
    outcome: dict,
    sanity: dict,
    config_block: dict,
    q1_baseline: dict | None,
) -> str:
    L: list[str] = []
    L.append("Q2A -- richer-family residual law on the marginal PIT")
    L.append("=" * 70)
    L.append("")
    L.append("Pre-registration: notes/preregistrations/")
    L.append("  2026-05-26_q2a-richer-family-mixture-or-nonparametric/phase_a.yaml")
    L.append(f"  phase_a body_sha256: c0a22c9e6cc708d3089a891a1eb7900e20e05e8816c673cad7bc77165bf138fe")
    L.append("Cuts from synthetic gate METHOD/DESIGN artifact:")
    L.append("  experiment/results/q2a_synthetic_gate.txt")
    L.append(f"  gate body_sha256: 2bab514e1049a89643ab69f45568400969c78c37939c3b188166e1a89c775c42")
    L.append(f"  R-A2 (corroboration):     effective_chi2 <= {R_A2_CUT}")
    L.append(f"  R-C2 (falsification):     effective_chi2  > {R_C2_CUT}")
    L.append(f"  R-B2 (ambiguous band):    ({R_A2_CUT}, {R_C2_CUT}]")
    L.append("")
    L.append("MARGINAL PIT CHI^2 PER FAMILY (post-cutoff, 10-bin)")
    L.append("-" * 70)
    for tag in ("M3", "M4", "M5"):
        line = f"  chi2_{tag}: {chi2[tag]:>10.1f}"
        if boot is not None and tag in boot.get("chi2", {}):
            arr = np.asarray(boot["chi2"][tag])
            lo, med, hi = np.nanpercentile(arr, [2.5, 50, 97.5])
            line += f"    bootstrap 95% CI [{lo:.1f}, {hi:.1f}]  median {med:.1f}"
        L.append(line)
    L.append("")
    L.append("EFFECTIVE CHI^2 (verdict metric)")
    L.append("-" * 70)
    L.append(f"  effective_chi2 = min(M3, M4, M5) = {outcome['effective_chi2']:.1f}")
    L.append(f"  preferred_family               = {outcome['preferred_family']}")
    if boot is not None and "effective" in boot.get("chi2", {}):
        arr = np.asarray(boot["chi2"]["effective"])
        lo, med, hi = np.nanpercentile(arr, [2.5, 50, 97.5])
        L.append(f"  effective_chi2 bootstrap 95% CI [{lo:.1f}, {hi:.1f}]  median {med:.1f}")
    L.append("")
    L.append("PRE-REGISTERED VERDICT")
    L.append("-" * 70)
    L.append(f"  verdict: {outcome['verdict']}")
    L.append(f"  reason:  {outcome['reason']}")
    L.append("")
    if q1_baseline is not None:
        L.append("Q1 REFERENCE BASELINES (NOT recomputed; loaded from Q1 pickle)")
        L.append("-" * 70)
        for k, v in q1_baseline.items():
            if isinstance(v, float):
                L.append(f"  {k}: {v:.1f}")
            else:
                L.append(f"  {k}: {v}")
        L.append("")
    L.append(f"PER-FAMILY FITTED PARAMETERS @ headline cell ({HEADLINE_DAYTYPE})")
    L.append("-" * 70)
    for tag, fam_key in MODEL_FAMILY.items():
        fit = fits[fam_key].get(HEADLINE_DAYTYPE)
        if fit is None:
            L.append(f"  {tag} ({fam_key}): library too thin for {HEADLINE_DAYTYPE}")
            continue
        params = fit["params"]
        if fam_key == "mixture_2":
            w = params["weights"]; s = params["scales"]
            L.append(f"  {tag} (mix-2): w=({w[0]:.4f}, {w[1]:.4f})  "
                     f"s=({s[0]:.4f}, {s[1]:.4f})  n={fit['n']}")
        elif fam_key == "mixture_3":
            w = params["weights"]; s = params["scales"]
            L.append(f"  {tag} (mix-3): w=({w[0]:.4f}, {w[1]:.4f}, {w[2]:.4f})  "
                     f"s=({s[0]:.4f}, {s[1]:.4f}, {s[2]:.4f})  n={fit['n']}")
        else:  # kde
            L.append(f"  {tag} (kde): bandwidth={params['bandwidth']:.4f}  "
                     f"n_samples={len(params['samples'])}  n={fit['n']}")
    L.append("")
    L.append("PIT HISTOGRAMS (10-bin) -- post-cutoff marginal")
    L.append("-" * 70)
    for tag in ("M3", "M4", "M5"):
        L.append(f"  {tag}: {hist[tag]}")
    L.append("")
    L.append("PRE-EXPERIMENT SAME-FAMILY SANITY CHECK")
    L.append("-" * 70)
    for fam, res in sanity.items():
        flag = "OK" if res["chi2"] <= SYNTHETIC_SANITY_CHI2_CEILING else "WARN"
        L.append(f"  [{flag}] {fam}: chi^2={res['chi2']:.1f}  "
                 f"(N={res['n_lib']}, M={res['M']}, "
                 f"ceiling={SYNTHETIC_SANITY_CHI2_CEILING:.0f}, "
                 f"kurt={SYNTHETIC_SANITY_KURT})")
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
#
# Plain dict via yaml.safe_dump; not a make_result artifact. Coverage
# of phase_a.variables.dependent is the Check R contract -- every
# dependent variable listed in phase_a appears as a numeric in this
# file.


def _render_result_yaml(
    *,
    chi2: dict[str, float],
    boot: dict | None,
    fits: dict[str, dict],
    outcome: dict,
    sanity: dict,
    config_block: dict,
    artifact_txt_path: Path,
    artifact_pkl_path: Path,
) -> dict:
    """Build the result.yaml payload (does not write)."""
    def ci_for(tag: str) -> tuple[float | None, float | None]:
        if boot is None:
            return None, None
        arr = boot.get("chi2", {}).get(tag)
        if arr is None:
            return None, None
        lo, hi = np.nanpercentile(np.asarray(arr), [2.5, 97.5])
        return float(lo), float(hi)

    # Per-family fitted params at headline cell (HEADLINE_DAYTYPE)
    headline_params: dict[str, dict] = {}
    for tag, fam_key in MODEL_FAMILY.items():
        fit = fits[fam_key].get(HEADLINE_DAYTYPE)
        if fit is None:
            headline_params[tag] = {"available": False}
            continue
        params = fit["params"]
        if fam_key == "kde":
            headline_params[tag] = {
                "available": True,
                "bandwidth": float(params["bandwidth"]),
                "n_kde_samples": int(len(params["samples"])),
                "n_library": int(fit["n"]),
            }
        else:
            headline_params[tag] = {
                "available": True,
                "weights": [float(x) for x in params["weights"]],
                "scales": [float(x) for x in params["scales"]],
                "n_library": int(fit["n"]),
            }

    # All-day-type fitted params (for completeness)
    all_dt_params: dict[str, dict] = {}
    for tag, fam_key in MODEL_FAMILY.items():
        all_dt_params[tag] = {}
        for dt in ("weekday", "saturday", "sunday"):
            fit = fits[fam_key].get(dt)
            if fit is None:
                all_dt_params[tag][dt] = {"available": False}
                continue
            params = fit["params"]
            if fam_key == "kde":
                all_dt_params[tag][dt] = {
                    "available": True,
                    "bandwidth": float(params["bandwidth"]),
                    "n_kde_samples": int(len(params["samples"])),
                    "n_library": int(fit["n"]),
                }
            else:
                all_dt_params[tag][dt] = {
                    "available": True,
                    "weights": [float(x) for x in params["weights"]],
                    "scales": [float(x) for x in params["scales"]],
                    "n_library": int(fit["n"]),
                }

    ci_low_m3, ci_high_m3 = ci_for("M3")
    ci_low_m4, ci_high_m4 = ci_for("M4")
    ci_low_m5, ci_high_m5 = ci_for("M5")
    ci_low_eff, ci_high_eff = ci_for("effective")

    return {
        "schema": "result",
        "written_at": pd.Timestamp.utcnow().isoformat() + "Z",
        "references": [{
            "file": "phase_a.yaml",
            "body_sha256":
                "c0a22c9e6cc708d3089a891a1eb7900e20e05e8816c673cad7bc77165bf138fe",
        }],
        "artifact": {
            "primary_txt": str(artifact_txt_path),
            "pickle": str(artifact_pkl_path),
            "grade": "CLAIM",
            "emitted_by": "experiment.distributional_class_q2a.__main__",
            "notes": (
                "Primary CLAIM artifact is the .txt under "
                "experiment/results/ written via experiment.provenance."
                "make_result (carries its own header + body hash). "
                "Pickle holds the full PIT tables and fitted-params dicts "
                "for re-analysis."
            ),
        },
        "code_path_audit": {
            "verdict": "PRODUCTION-PATH",
            "tool": ".venv/bin/python -m experiment.audit.code_path "
                    "experiment/distributional_class_q2a/__main__.py",
            "blocking": False,
        },
        # Coverage of phase_a.variables.dependent (Check R contract):
        # marginal_PIT_chi2_{M3,M4,M5}, effective_chi2,
        # mixture_2_weights_and_scales_per_day_type,
        # mixture_3_weights_and_scales_per_day_type,
        # kde_bandwidth_per_day_type
        "primary_result": {
            "metric": "effective_chi2 = min(chi2_M3, chi2_M4, chi2_M5)"
                      " on post-cutoff marginal PIT",
            "value": {
                "marginal_PIT_chi2_M3": float(chi2["M3"]),
                "marginal_PIT_chi2_M4": float(chi2["M4"]),
                "marginal_PIT_chi2_M5": float(chi2["M5"]),
                "effective_chi2": float(outcome["effective_chi2"]),
                "preferred_family": outcome["preferred_family"],
            },
            "ci_low": {
                "marginal_PIT_chi2_M3": ci_low_m3,
                "marginal_PIT_chi2_M4": ci_low_m4,
                "marginal_PIT_chi2_M5": ci_low_m5,
                "effective_chi2": ci_low_eff,
            },
            "ci_high": {
                "marginal_PIT_chi2_M3": ci_high_m3,
                "marginal_PIT_chi2_M4": ci_high_m4,
                "marginal_PIT_chi2_M5": ci_high_m5,
                "effective_chi2": ci_high_eff,
            },
            "ci_method_actually_used":
                f"paired_day_bootstrap_n={config_block['n_bootstrap']}",
            "outcome_verdict": outcome["verdict"],
            "outcome_reason": outcome["reason"],
        },
        "secondary_results": [
            {
                "name": "mixture_2_weights_and_scales_per_day_type",
                "value": all_dt_params["M3"],
            },
            {
                "name": "mixture_3_weights_and_scales_per_day_type",
                "value": all_dt_params["M4"],
            },
            {
                "name": "kde_bandwidth_per_day_type",
                "value": all_dt_params["M5"],
            },
            {
                "name": "per_family_fitted_params_at_headline",
                "value": headline_params,
                "notes": f"headline cell day-type: {HEADLINE_DAYTYPE}",
            },
            {
                "name": "pre_experiment_same_family_sanity",
                "value": {
                    fam: {"chi2": float(res["chi2"]),
                          "n_lib": res["n_lib"],
                          "M": res["M"]}
                    for fam, res in sanity.items()
                },
                "notes": (
                    "Same-family null sanity check at kurt = "
                    f"{SYNTHETIC_SANITY_KURT}; ceiling "
                    f"{SYNTHETIC_SANITY_CHI2_CEILING}; per phase_a "
                    "baselines.secondary."
                ),
            },
            {
                "name": "pre_registered_cuts",
                "value": {
                    "R-A2_cut": R_A2_CUT,
                    "R-C2_cut": R_C2_CUT,
                    "source": ("experiment/results/q2a_synthetic_gate.txt "
                               "body_sha256 2bab514e1049a89643ab69f4556"
                               "8400969c78c37939c3b188166e1a89c775c42"),
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
    p.add_argument("--in", dest="in_path", type=Path,
                   default=Path("scratch/data/multiscale_factor/factors_v2_final.pkl"),
                   help="v2 factors pickle (only used for the Q1-baseline cross-reference)")
    p.add_argument("--q1-pickle", dest="q1_pickle", type=Path,
                   default=Path("scratch/data/distributional_class_q1/q1.pkl"),
                   help="Q1 pickle for reference baselines (optional)")
    p.add_argument("--out", type=Path,
                   default=Path("scratch/data/distributional_class_q2a/q2a.pkl"))
    p.add_argument("--n-particles", type=int, default=N_PARTICLES)
    p.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    p.add_argument("--n-pit-bins", type=int, default=N_PIT_BINS)
    p.add_argument("--seed", type=int, default=SEED_BASE)
    p.add_argument("--skip-bootstrap", action="store_true",
                   help="point estimates only (for fast iteration)")
    p.add_argument("--save-pit", action="store_true",
                   help="include per-row PIT tables in the output pickle")
    p.add_argument(
        "--emit-result", action="store_true",
        help="write the CLAIM-GRADE provenanced artifact via "
             "experiment.provenance.make_result (refuses a dirty tree) "
             "PLUS the registry result.yaml under notes/preregistrations/"
             "2026-05-26_q2a-richer-family-mixture-or-nonparametric/"
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

    print(f"# Q2A: richer-family residual law (M3/M4/M5)")
    print(f"  cutoff:       {cutoff}")
    print(f"  anchor_h:     {anchor_h}")
    print(f"  embedding:    {dict(dims)}")
    print(f"  N_PARTICLES:  {args.n_particles}")
    print(f"  N_BOOTSTRAP:  {args.n_bootstrap}")
    print(f"  cuts:         R-A2 <= {R_A2_CUT}, R-C2 > {R_C2_CUT}")
    print()

    # ---- Pre-experiment same-family sanity check (per phase_a) ----
    print(f"# pre-experiment same-family sanity check ...")
    t0 = time.time()
    sanity = _run_sanity_checks(verbose=True)
    print(f"  ({time.time()-t0:.1f}s)")
    print()

    # ---- Load actuals + build z ----
    raw_full = load_actuals(cutoff=None).dropna()
    zp = zscore_params(cutoff, method=clim_method, k_year=k_year, k_day=k_day)
    z_full = zscore_transform(raw_full, zp)
    raw_pre = load_pre_cutoff_actuals(cutoff).dropna()
    z_pre = zscore_transform(raw_pre, zp)

    # ---- Optional Q1 baseline reference ----
    q1_baseline = None
    if args.q1_pickle.exists():
        try:
            with args.q1_pickle.open("rb") as f:
                q1 = pickle.load(f)
            q1_baseline = {
                "Q1_chi2_M0_gaussian": float(q1.get("chi2_M0", float("nan"))),
                "Q1_chi2_M1_studt_kernel": float(q1.get("chi2_M1", float("nan"))),
                "Q1_chi2_M2_studt_climatology": float(q1.get("chi2_M2", float("nan"))),
            }
        except Exception as e:
            print(f"# WARNING: failed to load Q1 baseline pickle: {e}")

    # ---- Fit M3, M4, M5 ----
    fits: dict[str, dict] = {}
    for tag, fam_key in MODEL_FAMILY.items():
        print(f"# {tag} ({fam_key}) fits per day-type ...")
        t0 = time.time()
        fits[fam_key] = _fit_family_per_daytype(fam_key, z_pre, dims)
        for dt, fit in fits[fam_key].items():
            if fit is None:
                print(f"  {dt}: library too thin")
            else:
                summary = str(fit["params"])
                if len(summary) > 100:
                    summary = summary[:97] + "..."
                print(f"  {dt}: n={fit['n']}  params={summary}")
        print(f"  ({time.time()-t0:.1f}s)")
        print()

    # ---- SMC propagation + PIT per family ----
    pit_tables: dict[str, pd.DataFrame] = {}
    chi2: dict[str, float] = {}
    hist: dict[str, list[int]] = {}
    for tag, fam_key in MODEL_FAMILY.items():
        print(f"# {tag} SMC propagation + post-cutoff PIT ...")
        t0 = time.time()
        rng_tag = np.random.default_rng(
            (args.seed + 100 + _TAG_SEED_OFFSET[tag]) & 0xFFFFFFFF
        )
        pit_tables[tag] = _pit_for_family(
            z_full, fits[fam_key], fam_key, cutoff, anchor_h, dims,
            rng=rng_tag, M=args.n_particles,
        )
        c, h = _marginal_pit_chi2(pit_tables[tag]["u_PIT"].to_numpy())
        chi2[tag] = float(c)
        hist[tag] = list(h.astype(int))
        print(f"  rows={len(pit_tables[tag])}  chi^2={chi2[tag]:.1f}  "
              f"({time.time()-t0:.1f}s)")
        print()

    # ---- Paired-day bootstrap ----
    if args.skip_bootstrap:
        print(f"# bootstrap SKIPPED (--skip-bootstrap)")
        boot = None
    else:
        print(f"# paired-day bootstrap (n={args.n_bootstrap}) ...")
        print(f"  -- min(chi^2) recomputed INSIDE each resample "
              f"(per advisor / fairness)")
        t0 = time.time()
        # Union of delivery days across families (the M3/M4/M5 tables
        # all draw from _complete_delivery_days; union is just to be
        # safe against any family that filtered a day out).
        all_days = pd.DatetimeIndex(sorted(set().union(
            *[set(pit_tables[t]["delivery_day"]) for t in ("M3", "M4", "M5")]
        )))
        n_days = len(all_days)
        # pre-index by delivery_day for fast lookup
        grouped = {
            tag: {d: g for d, g in pit_tables[tag].groupby(
                "delivery_day", sort=False)}
            for tag in ("M3", "M4", "M5")
        }
        boot_chi2 = {"M3": [], "M4": [], "M5": [], "effective": []}
        rng_boot = np.random.default_rng(
            (args.seed + 7919) & 0xFFFFFFFF
        )
        for b in range(args.n_bootstrap):
            idx = rng_boot.integers(0, n_days, size=n_days)
            sampled_days = all_days[idx]
            cells = {}
            for tag in ("M3", "M4", "M5"):
                u = pd.concat(
                    [grouped[tag][d] for d in sampled_days if d in grouped[tag]],
                    ignore_index=True,
                )["u_PIT"].to_numpy()
                c, _ = _marginal_pit_chi2(u)
                cells[tag] = c
                boot_chi2[tag].append(c)
            boot_chi2["effective"].append(min(cells.values()))
        boot_chi2 = {k: np.array(v) for k, v in boot_chi2.items()}
        for k in ("M3", "M4", "M5", "effective"):
            arr = boot_chi2[k]
            lo, med, hi = np.nanpercentile(arr, [2.5, 50, 97.5])
            print(f"  chi^2 {k}: median {med:.1f}  CI [{lo:.1f}, {hi:.1f}]")
        boot = {"chi2": {k: v.tolist() for k, v in boot_chi2.items()}}
        print(f"  ({time.time()-t0:.1f}s)")
        print()

    # ---- Mechanical outcome ----
    outcome = _evaluate_outcome(chi2["M3"], chi2["M4"], chi2["M5"])
    print(f"# pre-registered outcome: {outcome['verdict']}")
    print(f"  effective_chi2: {outcome['effective_chi2']:.1f}")
    print(f"  preferred:      {outcome['preferred_family']}")
    print(f"  reason:         {outcome['reason']}")
    print()

    # ---- Pickle (always) ----
    out_pkl = args.out
    if not out_pkl.is_absolute():
        out_pkl = (Path.cwd() / out_pkl).resolve()
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "chi2": chi2,
        "hist": hist,
        "fits": fits,
        "pit_tables": pit_tables if args.save_pit else None,
        "bootstrap": boot,
        "outcome": outcome,
        "sanity": sanity,
        "q1_baseline": q1_baseline,
        "config": {
            "n_particles": args.n_particles,
            "n_bootstrap": args.n_bootstrap if not args.skip_bootstrap else 0,
            "n_pit_bins": args.n_pit_bins,
            "seed": args.seed,
            "headline_daytype": HEADLINE_DAYTYPE,
            "R_A2_cut": R_A2_CUT,
            "R_C2_cut": R_C2_CUT,
            "synthetic_gate_body_sha256":
                "2bab514e1049a89643ab69f45568400969c78c37939c3b188166e1a89c775c42",
            "phase_a_body_sha256":
                "c0a22c9e6cc708d3089a891a1eb7900e20e05e8816c673cad7bc77165bf138fe",
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
            hist=hist,
            boot=boot,
            fits=fits,
            outcome=outcome,
            sanity=sanity,
            config_block=config_block,
            q1_baseline=q1_baseline,
        )
        out_txt = (
            PROJECT_ROOT / "experiment" / "results"
            / "q2a_richer_family.txt"
        )
        print()
        print(f"# emitting CLAIM-GRADE artifact -> {out_txt.relative_to(PROJECT_ROOT)}")
        hdr = make_result(
            path=out_txt,
            grade=Grade.CLAIM,
            title=("Q2A: richer-family residual law (mix-2 / mix-3 / KDE) "
                   "on the marginal PIT calibration gap"),
            body=body,
            inputs={
                "data_cutoff": cutoff.isoformat(),
                "anchor_h": anchor_h,
                "embedding_dims": dict(dims),
                "phase_a_body_sha256":
                    "c0a22c9e6cc708d3089a891a1eb7900e20e05e8816c673cad7bc77165bf138fe",
                "synthetic_gate_body_sha256":
                    "2bab514e1049a89643ab69f45568400969c78c37939c3b188166e1a89c775c42",
                "thread_body_sha256":
                    "042f997c3e46ad04bda5761eddf6800b76835132531bc24c7dcef92ccee8adc8",
                "R_A2_cut": R_A2_CUT,
                "R_C2_cut": R_C2_CUT,
            },
            seeds={
                "seed_base": args.seed,
                "rng_purpose_offsets": {
                    "sanity_check_fam_offsets": _FAM_SEED_OFFSET,
                    "smc_tag_offsets": _TAG_SEED_OFFSET,
                    "sanity_check_base": "SEED_BASE + 1001 + _FAM_SEED_OFFSET[fam]",
                    "smc_per_model": "args.seed + 100 + _TAG_SEED_OFFSET[tag]",
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
            fits=fits,
            outcome=outcome,
            sanity=sanity,
            config_block=config_block,
            artifact_txt_path=out_txt.relative_to(PROJECT_ROOT),
            artifact_pkl_path=out_pkl.relative_to(PROJECT_ROOT)
                if out_pkl.is_relative_to(PROJECT_ROOT) else out_pkl,
        )
        out_yaml = (
            PROJECT_ROOT / "notes" / "preregistrations"
            / "2026-05-26_q2a-richer-family-mixture-or-nonparametric"
            / "result.yaml"
        )
        with out_yaml.open("w") as f:
            yaml.safe_dump(result_yaml, f, sort_keys=False, default_flow_style=False)
        print(f"  wrote {out_yaml.relative_to(PROJECT_ROOT)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
