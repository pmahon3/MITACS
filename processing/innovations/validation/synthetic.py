"""Synthetic ground-truth recovery tests for production estimators.

Three gates, each calling production functions directly (NOT
reimplementing them — the discipline rule of `CLAUDE.md`):

  1. VAR(1) recovery — ``build_local_gaussian_semigroup`` on known
     (A, Q) → per-anchor drift / diffusion within tolerance.

  2. Non-Gaussianity — ``innovation_diagnostics`` on Gaussian-noise
     VAR(1) reads ~Gaussian; Student-t-noise VAR(1) flags heavy-
     tailed clearly.

  3. Patra–Sen mixture-fraction recovery — ``patra_sen_fit`` on
     known α_true ∈ {0.0, 0.50} samples returns α̂_L within the
     paper's guaranteed band. P1 of the resolution-paths thread
     depends on this estimator; this gate is its precondition.

Generates a stable VAR(1) process

    x_{t+1} = A x_t + eps_t,   eps_t ~ N(0, Q)

with a *known* drift ``A`` (spectral radius < 1) and a *known* SPD diffusion
``Q``, flows it through the **exact production code path** --
``estimator.build_local_gaussian_semigroup`` (per-anchor true-LOO-CV drift
bandwidth + plain mu-centred residual covariance, NO residual kernel) -- and
asserts that the recovered per-anchor drift and diffusion match ground truth.

Convention note
---------------
The drift is ``C = lstsq(WX, Wy)`` with forecast ``x_next = x @ C``
(row-vector / right-multiply convention). For the column-convention VAR(1)
``x_{t+1} = A x_t`` this means the recovered ``C`` estimates ``A.T``. The
checks below compare ``C_hat`` against ``A.T`` accordingly.

Diffusion note
--------------
Diffusion is the plain mu-centred covariance of the locally-fitted residuals
``Y - X@C`` (no residual kernel -- see ``estimator.py`` and memory note
``mitacs-theta-rail-pinning`` for why the kernel was dropped). For VAR(1)
the innovations are Gaussian iid, so this plain covariance is the *exact*
maximum-likelihood estimator of ``Q`` -- the sharpest possible check that
the production estimator is unbiased.

Run standalone::

    python -m processing.innovations.validation.synthetic
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from edynamics.modelling_tools import Embedding, Lag

# Validate the *production* estimator code path, not a copy.
from processing.innovations.estimator import (
    build_local_gaussian_semigroup,
    patra_sen_fit,
)


# ──────────────────────────────────────────────────────────────────────────
# Ground-truth process
# ──────────────────────────────────────────────────────────────────────────
def make_var1_params(d: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A stable drift ``A`` (spectral radius ~0.5) and an SPD diffusion ``Q``."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d))
    # rescale to a comfortably stable spectral radius
    radius = max(abs(np.linalg.eigvals(A)))
    A = A * (0.5 / radius)
    M = rng.standard_normal((d, d))
    Q = M @ M.T + d * np.eye(d)  # SPD, well-conditioned
    return A, Q


def simulate_var1(
    A: np.ndarray,
    Q: np.ndarray,
    n: int,
    burn: int,
    seed: int,
) -> np.ndarray:
    """Simulate ``n`` retained samples of x_{t+1} = A x_t + N(0, Q)."""
    d = A.shape[0]
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(Q)
    T = n + burn
    X = np.empty((T + 1, d))
    X[0] = rng.standard_normal(d)
    for t in range(T):
        X[t + 1] = A @ X[t] + L @ rng.standard_normal(d)
    return X[burn:]


def simulate_var1_t(
    A: np.ndarray,
    Q: np.ndarray,
    n: int,
    burn: int,
    seed: int,
    df: int = 5,
) -> np.ndarray:
    """VAR(1) with Student-t innovations (KNOWN heavy-tailed).

    Innovations are t_df scaled to unit variance then coloured by
    ``chol(Q)``. For df=5 the per-coordinate excess kurtosis is exactly
    ``6/(df-4) = 6.0`` -- a finite, unambiguous non-Gaussian target the
    diagnostic must clearly flag (vs ~0 for the Gaussian simulator).
    """
    d = A.shape[0]
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(Q)
    scale = np.sqrt((df - 2) / df)  # t_df variance = df/(df-2); unit-ise
    T = n + burn
    X = np.empty((T + 1, d))
    X[0] = rng.standard_normal(d)
    for t in range(T):
        eps = rng.standard_t(df, size=d) * scale
        X[t + 1] = A @ X[t] + L @ eps
    return X[burn:]


def build_embedding(X: np.ndarray) -> tuple[Embedding, pd.DatetimeIndex]:
    """Wrap the VAR(1) path as an hourly DataFrame with d Lag(tau=0) observers.

    Using one variable per coordinate with tau=0 makes the embedding block
    columns the raw state, so the recovered C is exactly the (transposed)
    VAR matrix -- the cleanest possible ground-truth check.
    """
    d = X.shape[1]
    idx = pd.date_range("2000-01-01", periods=len(X), freq="h")
    cols = [f"x{i}" for i in range(d)]
    df = pd.DataFrame(X, index=idx, columns=cols).asfreq("h")
    observers = [Lag(variable_name=c, tau=0) for c in cols]
    embedding = Embedding(data=df, observers=observers, library_times=idx)
    embedding.compile()
    return embedding, idx


# ──────────────────────────────────────────────────────────────────────────
# Recovery
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class RecoveryResult:
    drift_rel_err: float       # median_j ||C_hat_j - A.T||_F / ||A.T||_F
    diffusion_rel_err: float   # median_j ||Sigma_hat_j - Q||_F / ||Q||_F
    eig_rel_err: float         # median_j relative error of sorted eig(Sigma) vs eig(Q)
    n_anchors: int


def recover(
    *,
    d: int = 3,
    n: int = 4000,
    burn: int = 500,
    seed: int = 7,
    n_anchors: int = 60,
) -> tuple[RecoveryResult, np.ndarray, np.ndarray]:
    A, Q = make_var1_params(d, seed)
    X = simulate_var1(A, Q, n=n, burn=burn, seed=seed + 1)
    embedding, idx = build_embedding(X)

    # anchors: a sample of interior times (skip ends so neighbours exist)
    rng = np.random.default_rng(seed + 2)
    interior = idx[d + 1 : -2]
    sel = np.sort(
        rng.choice(len(interior), size=min(n_anchors, len(interior)), replace=False)
    )
    anchors = interior[sel]

    # Validate the EXACT production code path: build_local_gaussian_semigroup
    # (true-LOO-CV theta + plain residual covariance, no residual kernel).
    est = build_local_gaussian_semigroup(
        embedding=embedding, anchors=pd.DatetimeIndex(anchors)
    )

    C_hat = est.coefficients          # (N, d, d) -- estimates A.T
    Sig_hat = est.covariances         # (N, d, d) -- estimates Q

    A_T = A.T
    eig_Q = np.sort(np.linalg.eigvalsh(Q))

    drift_errs, diff_errs, eig_errs = [], [], []
    for Ci, Si in zip(C_hat, Sig_hat):
        if not (np.all(np.isfinite(Ci)) and np.all(np.isfinite(Si))):
            continue
        drift_errs.append(np.linalg.norm(Ci - A_T) / np.linalg.norm(A_T))
        diff_errs.append(np.linalg.norm(Si - Q) / np.linalg.norm(Q))
        eig_Si = np.sort(np.linalg.eigvalsh(Si))
        eig_errs.append(np.linalg.norm(eig_Si - eig_Q) / np.linalg.norm(eig_Q))

    result = RecoveryResult(
        drift_rel_err=float(np.median(drift_errs)),
        diffusion_rel_err=float(np.median(diff_errs)),
        eig_rel_err=float(np.median(eig_errs)),
        n_anchors=len(drift_errs),
    )
    return result, A, Q


# ──────────────────────────────────────────────────────────────────────────
# Test entry points
# ──────────────────────────────────────────────────────────────────────────
# Tolerances: WLS drift on a linear-Gaussian system is unbiased, so the drift
# bar is tight. The kernel-reweighted covariance is a consistent but
# finite-sample estimator of Q, so the diffusion bar is looser.
DRIFT_TOL = 0.10
DIFFUSION_TOL = 0.25
EIG_TOL = 0.25


def test_recovers_var1() -> None:
    result, _, _ = recover()
    assert result.n_anchors > 0, "no finite anchors recovered"
    assert result.drift_rel_err < DRIFT_TOL, (
        f"drift rel err {result.drift_rel_err:.4f} >= {DRIFT_TOL}"
    )
    assert result.diffusion_rel_err < DIFFUSION_TOL, (
        f"diffusion rel err {result.diffusion_rel_err:.4f} >= {DIFFUSION_TOL}"
    )
    assert result.eig_rel_err < EIG_TOL, (
        f"eig rel err {result.eig_rel_err:.4f} >= {EIG_TOL}"
    )


def _innov_kurt_over_anchors(X: np.ndarray, n_anchors: int, seed: int):
    """Median (excess_kurt, tail_ratio) from the PRODUCTION
    ``innovation_diagnostics`` over anchors of a VAR(1) path."""
    from processing.innovations.estimator import innovation_diagnostics

    embedding, idx = build_embedding(X)
    d = X.shape[1]
    rng = np.random.default_rng(seed)
    interior = idx[d + 1 : -2]
    sel = np.sort(
        rng.choice(len(interior), min(n_anchors, len(interior)), replace=False)
    )
    eks, trs = [], []
    for t in interior[sel]:
        di = innovation_diagnostics(embedding=embedding, anchor=t)  # no mask
        if np.isfinite(di["excess_kurt"]):
            eks.append(di["excess_kurt"])
            trs.append(di["tail_ratio"])
    return float(np.median(eks)), float(np.median(trs))


# Gaussian innovations -> excess kurt ~0, tail ratio ~1.
# Student-t(df=5) -> per-coord excess kurt 6.0 (the WLS fit attenuates it
# somewhat; observed ~4.7). Thresholds calibrated from observed clean-vs-
# heavy behaviour (G: kurt~-0.05 tail~0.98 ; t: kurt~4.7 tail~1.43). The
# tail_ratio is deliberately outlier-RESISTANT so it moves less than
# kurtosis -- the binding assertions are (a) no false positive on
# Gaussian and (b) clear SEPARATION, not large absolute t values.
GAUSS_KURT_MAX = 0.5      # Gaussian must read ~0 (no false positive)
GAUSS_TAIL_MAX = 1.10
T_KURT_MIN = 3.0          # t kurtosis must register clearly elevated
T_TAIL_MIN = 1.20         # t tail_ratio clearly above the Gaussian ~0.98
KURT_SEP_MIN = 2.0        # t must exceed Gaussian kurt by a clear margin
TAIL_SEP_MIN = 0.25       # and tail_ratio by a clear margin


def test_innovation_nongaussianity() -> None:
    A, Q = make_var1_params(d=3, seed=11)
    Xg = simulate_var1(A, Q, n=6000, burn=500, seed=12)
    Xt = simulate_var1_t(A, Q, n=6000, burn=500, seed=12, df=5)
    gk, gt = _innov_kurt_over_anchors(Xg, 40, seed=1)
    tk, tt = _innov_kurt_over_anchors(Xt, 40, seed=1)
    assert gk < GAUSS_KURT_MAX, f"Gaussian excess_kurt {gk:.2f} not ~0"
    assert gt < GAUSS_TAIL_MAX, f"Gaussian tail_ratio {gt:.2f} not ~1"
    assert tk > T_KURT_MIN, f"t excess_kurt {tk:.2f} not flagged heavy"
    assert tt > T_TAIL_MIN, f"t tail_ratio {tt:.2f} not flagged heavy"
    assert tk - gk > KURT_SEP_MIN, (
        f"kurt can't separate t ({tk:.2f}) from G ({gk:.2f})"
    )
    assert tt - gt > TAIL_SEP_MIN, (
        f"tail_ratio can't separate t ({tt:.2f}) from G ({gt:.2f})"
    )


# ──────────────────────────────────────────────────────────────────────────
# Multi-step composition gate (Task 23) -- the THEORY-touching validation.
#
# Iterating the one-step Pi_Delta H steps is the explicitly-unbuilt
# semigroup (Chapman-Kolmogorov never established for this estimator;
# theory-correspondence qualifier 3). For VAR(1) the H-step map has a
# CLOSED FORM, so any deviation of the composed estimate is a real
# composition defect, not model misspecification:
#   true H-step drift      : A^H        (row-conv: C_true = (A^H).T = (A.T)^H)
#   true H-step diffusion  : sum_{k=0}^{H-1} A^k Q (A^k).T
# The gate composes the PRODUCTION one-step estimate and checks recovery,
# AND reports the error-growth curve (compounding is EXPECTED -- shown,
# never averaged away).
# ──────────────────────────────────────────────────────────────────────────
_HORIZONS = (1, 2, 4, 8, 12, 24)
# Composition compounds estimation error geometrically; tolerances widen
# with H accordingly. These bracket "sound composition", not "perfect".
_MS_DRIFT_TOL = {1: 0.10, 2: 0.15, 4: 0.25, 8: 0.40, 12: 0.55, 24: 0.90}
_MS_DIFF_TOL = {1: 0.25, 2: 0.30, 4: 0.40, 8: 0.55, 12: 0.70, 24: 1.10}


def _true_multistep(A: np.ndarray, Q: np.ndarray, H: int):
    """Closed-form VAR(1) H-step drift A^H and accumulated diffusion."""
    Ah = np.linalg.matrix_power(A, H)
    S = np.zeros_like(Q)
    Ak = np.eye(A.shape[0])
    for _ in range(H):
        S = S + Ak @ Q @ Ak.T
        Ak = Ak @ A
    return Ah, S


def multistep_recovery(d: int = 3, seed: int = 7, n_anchors: int = 60):
    """Compose the production one-step estimate H steps; compare to the
    closed-form VAR(1) H-step truth. Returns per-horizon rel-errors."""
    A, Q = make_var1_params(d, seed)
    X = simulate_var1(A, Q, n=4000, burn=500, seed=seed + 1)
    embedding, idx = build_embedding(X)
    rng = np.random.default_rng(seed + 2)
    interior = idx[d + 1 : -2]
    sel = np.sort(
        rng.choice(len(interior), size=min(n_anchors, len(interior)), replace=False)
    )
    est = build_local_gaussian_semigroup(
        embedding=embedding, anchors=pd.DatetimeIndex(interior[sel])
    )
    # one-step estimate, aggregated over anchors (median = robust point est)
    C1 = np.median(est.coefficients, axis=0)   # ~ A.T  (row convention)
    S1 = np.median(est.covariances, axis=0)    # ~ Q

    rows = []
    for H in _HORIZONS:
        # compose drift: C1^H estimates (A^H).T ; compare to (A^H).T
        CH = np.linalg.matrix_power(C1, H)
        # compose diffusion via the same VAR recursion using the ESTIMATE:
        # Sigma_H = sum_{k=0}^{H-1} (C1.T)^k S1 ((C1.T)^k).T  -- estimator-only
        SH = np.zeros_like(S1)
        Ak = np.eye(d)
        for _ in range(H):
            SH = SH + Ak @ S1 @ Ak.T
            Ak = Ak @ C1.T
        Ah_t, S_true = _true_multistep(A, Q, H)
        CH_true = Ah_t.T
        de = np.linalg.norm(CH - CH_true) / np.linalg.norm(CH_true)
        se = np.linalg.norm(SH - S_true) / np.linalg.norm(S_true)
        rows.append((H, float(de), float(se)))
    return rows


def test_multistep_composition() -> None:
    rows = multistep_recovery()
    for H, de, se in rows:
        assert de < _MS_DRIFT_TOL[H], (
            f"H={H}: drift composition rel err {de:.3f} >= {_MS_DRIFT_TOL[H]} "
            f"-- iterated Pi_Delta does NOT recover the VAR(1) H-step map"
        )
        assert se < _MS_DIFF_TOL[H], (
            f"H={H}: diffusion composition rel err {se:.3f} >= {_MS_DIFF_TOL[H]}"
        )
    # error must be monotone-ish non-decreasing in H (compounding is real)
    des = [de for _, de, _ in rows]
    assert des[-1] >= des[0], "drift error should grow with horizon"


# ──────────────────────────────────────────────────────────────────────────
# Patra–Sen mixture-fraction recovery gate (added 2026-05-27 for the
# resolution-paths thread's P1 node). The production ``patra_sen_fit``
# is exercised on two known-α_true cells:
#
#   Cell A: α_true = 0.0 (pure standard-normal). Theorem 5 of Patra & Sen
#           (2016) GUARANTEES P(α̂_L = 0) = 1 − β = 0.95 in this case
#           (the asymptotic CvM quantile is exact at α_0 = 0). In
#           practice the grid resolution 1/n_grid ≈ 0.0025 is the
#           effective floor — we test α̂_L ≤ 0.05 (well below the
#           R-C cut of 0.10).
#   Cell B: α_true = 0.50 mixture of N(0,1) and N(3,1). Phase_a's
#           expected band is α̂_L ∈ [0.45, 0.55] for n ≈ 2000.
#
# The synthetic data is iid (no "days"), so the gate's CIs are over
# multiple seeds rather than paired-day bootstraps — that resampling
# scheme is specific to the Ontario script. Mirror this in the
# docstring so the next reader doesn't misread it as a bug.
# ──────────────────────────────────────────────────────────────────────────
# Gate criteria (over a multi-seed distribution; see patra_sen_recovery
# for the resampling design). Cell A: Theorem 5 guarantees
# ``P(α̂_L = 0) = 1 − β = 0.95`` when α_0 = 0, so we require the
# multi-seed FRACTION that fires exactly at 0 to be ≥ 0.90 (a small
# slack on the 0.95 paper guarantee for n=2000) AND p95 ≤ 0.05 (the
# rare positive readings must remain well below the R-C cut of 0.10).
# Cell B: α̂_L is a LOWER bound on α_true=0.50, so the gate is that
# the multi-seed MEDIAN of α̂_L sits in [0.42, 0.52]: high enough that
# the diagnostic finds the signal cleanly, low enough that it
# respects the lower-bound property (≤ α_true with finite-sample
# slack). Phase_a's stated band "0.45-0.55" was the writer's nominal
# guess — the empirical multi-seed distribution at n=2000 is what
# actually constrains the gate; phase-fidelity Check P should look
# at the empirical distribution before P1 fires.
_PATRA_SEN_NULL_P95_MAX = 0.05      # cells A / U-A: 95th percentile of α̂_L
_PATRA_SEN_NULL_ZERO_FRAC = 0.90    # cells A / U-A: P(α̂_L = 0) must be ≥ this
_PATRA_SEN_HALF_MED_LO = 0.42       # cell B (gaussian): median α̂_L lower bound
_PATRA_SEN_HALF_MED_HI = 0.52       # cell B (gaussian): median α̂_L upper bound
_PATRA_SEN_HALF_U_MED_LO = 0.40     # cell U-B (uniform): median α̂_L lower bound
_PATRA_SEN_HALF_U_MED_HI = 0.50     # cell U-B (uniform): median α̂_L upper bound

# Number of seeds for the multi-seed characterization. 50 is enough
# to estimate p5/p50/p95 cleanly at the precision we gate on
# (~0.02-0.03 standard error on quantiles) and runs in a few seconds.
_PATRA_SEN_N_SEEDS = 50


def patra_sen_recovery(
    *,
    n: int = 2000,
    n_seeds: int = _PATRA_SEN_N_SEEDS,
    base_seed: int = 21,
) -> dict[str, float]:
    """Multi-seed Patra–Sen recovery test on known α_true cells.

    Four cells across two F_b settings:

      Cell A   — F_b=N(0,1); α_true = 0; pure standard normal.
      Cell B   — F_b=N(0,1); α_true = 0.5; 50/50 N(0,1) + N(3,1) mixture.
      Cell U-A — F_b=Uniform[0,1]; α_true = 0; pure Uniform[0,1].
      Cell U-B — F_b=Uniform[0,1]; α_true = 0.5; 50/50 Uniform[0,1] +
                 Beta(0.5, 5) mixture (Beta concentrated near 0 — clearly
                 non-uniform signal).

    Theorem 5 of Patra & Sen guarantees ``P(α̂_L = 0) = 1 − β = 0.95``
    at α_0 = 0 regardless of F_b, so cells A and U-A use the same gate
    (zero-fraction ≥ 0.90, p95 ≤ 0.05). α̂_L is a LOWER confidence
    bound, so cells B / U-B expect α̂_L ≲ 0.5 with finite-sample slack.
    The uniform-F_b cells are the precondition for P1
    (``2026-05-27_p1-patra-sen-per-stratum-localization``), whose
    methodology amendment switched from F_b=N(0,1) on standardized
    residuals to F_b=Uniform[0,1] on PIT values (Patra-Sen 2016
    Theorem 1 equivalence; matches Q1's measurement convention).

    The synthetic data is iid (no "days"). The Ontario P1 script uses
    paired-day bootstrap CIs at the outer loop — that resampling
    scheme is specific to the time-series structure of the residuals
    and is NOT applied here.

    Returns a flat dict carrying both per-seed point estimates (for
    diagnostic plotting) and the quantile summaries the gate uses.
    """
    rng_master = np.random.default_rng(base_seed)
    seeds = rng_master.integers(0, 2**31, size=n_seeds)

    alpha_L_nulls = np.empty(n_seeds)
    alpha_L_halves = np.empty(n_seeds)
    alpha_L_u_nulls = np.empty(n_seeds)
    alpha_L_u_halves = np.empty(n_seeds)
    heur_null = np.empty(n_seeds)
    heur_half = np.empty(n_seeds)

    for k, s in enumerate(seeds):
        rng = np.random.default_rng(int(s))
        # Cell A: pure standard normal (α_true = 0).
        r_null = rng.standard_normal(n)
        fit_null = patra_sen_fit(r_null)
        alpha_L_nulls[k] = fit_null["alpha_L"]
        heur_null[k] = fit_null["alpha_hat_heuristic"]

        # Cell B: 50/50 mixture of N(0, 1) and N(3, 1).
        mask = rng.uniform(size=n) < 0.5
        r_half = np.where(
            mask, rng.standard_normal(n) + 3.0, rng.standard_normal(n)
        )
        fit_half = patra_sen_fit(r_half)
        alpha_L_halves[k] = fit_half["alpha_L"]
        heur_half[k] = fit_half["alpha_hat_heuristic"]

        # Cell U-A: pure Uniform[0,1] (α_true = 0).
        u_null = rng.uniform(size=n)
        fit_u_null = patra_sen_fit(u_null, F_b="uniform")
        alpha_L_u_nulls[k] = fit_u_null["alpha_L"]

        # Cell U-B: 50/50 mixture of Uniform[0,1] and Beta(0.5, 5).
        # Beta(0.5, 5) concentrates near 0 — clean non-uniform alternative.
        mask_u = rng.uniform(size=n) < 0.5
        u_half = np.where(mask_u, rng.beta(0.5, 5, size=n), rng.uniform(size=n))
        fit_u_half = patra_sen_fit(u_half, F_b="uniform")
        alpha_L_u_halves[k] = fit_u_half["alpha_L"]

    # Quantile summaries. p5/p50/p95 follow the same convention as the
    # P1 phase_a's "alpha_L_marginal" sanity-check expects.
    pct = lambda a, q: float(np.percentile(a, q))  # noqa: E731

    return {
        "n": int(n),
        "n_seeds": int(n_seeds),
        # Cell A summaries (F_b=N(0,1), α_true=0):
        "alpha_L_null_zero_frac": float(np.mean(alpha_L_nulls == 0.0)),
        "alpha_L_null_p50": pct(alpha_L_nulls, 50),
        "alpha_L_null_p95": pct(alpha_L_nulls, 95),
        "alpha_L_null_max": float(np.max(alpha_L_nulls)),
        # Cell B summaries (F_b=N(0,1), α_true=0.5):
        "alpha_L_half_p5": pct(alpha_L_halves, 5),
        "alpha_L_half_p50": pct(alpha_L_halves, 50),
        "alpha_L_half_p95": pct(alpha_L_halves, 95),
        # Cell U-A summaries (F_b=Uniform, α_true=0):
        "alpha_L_u_null_zero_frac": float(np.mean(alpha_L_u_nulls == 0.0)),
        "alpha_L_u_null_p50": pct(alpha_L_u_nulls, 50),
        "alpha_L_u_null_p95": pct(alpha_L_u_nulls, 95),
        "alpha_L_u_null_max": float(np.max(alpha_L_u_nulls)),
        # Cell U-B summaries (F_b=Uniform, α_true=0.5):
        "alpha_L_u_half_p5": pct(alpha_L_u_halves, 5),
        "alpha_L_u_half_p50": pct(alpha_L_u_halves, 50),
        "alpha_L_u_half_p95": pct(alpha_L_u_halves, 95),
        # Heuristic medians (diagnostic only):
        "alpha_hat_heuristic_null_p50": pct(heur_null, 50),
        "alpha_hat_heuristic_half_p50": pct(heur_half, 50),
        # c_n from the last fit (same across seeds and F_b; just for the report):
        "c_n": float(fit_null["c_n"]),
    }


def test_patra_sen_recovery() -> None:
    res = patra_sen_recovery()
    # Cell A: zero-fraction must respect Theorem 5's 0.95 guarantee
    # (with small finite-sample slack at n=2000).
    assert res["alpha_L_null_zero_frac"] >= _PATRA_SEN_NULL_ZERO_FRAC, (
        f"Cell A zero-fraction {res['alpha_L_null_zero_frac']:.2f} < "
        f"{_PATRA_SEN_NULL_ZERO_FRAC}. Theorem 5 says α̂_L = 0 w.p. ≥ 0.95 "
        "when α_0 = 0; a lower empirical fraction indicates the c_n "
        "threshold (0.6792) or the eq.9 limit at γ=0 is wrong."
    )
    assert res["alpha_L_null_p95"] <= _PATRA_SEN_NULL_P95_MAX, (
        f"Cell A p95 α̂_L = {res['alpha_L_null_p95']:.4f} > "
        f"{_PATRA_SEN_NULL_P95_MAX}. Even the rare positive readings on "
        "pure-Gaussian data must stay well below the R-C cut (0.10)."
    )
    # Cell B: median α̂_L in the empirical band derived from the
    # Theorem 5 lower-bound property.
    assert (
        _PATRA_SEN_HALF_MED_LO <= res["alpha_L_half_p50"]
        <= _PATRA_SEN_HALF_MED_HI
    ), (
        f"Cell B median α̂_L = {res['alpha_L_half_p50']:.4f} outside "
        f"[{_PATRA_SEN_HALF_MED_LO}, {_PATRA_SEN_HALF_MED_HI}]. "
        "α̂_L is a LOWER bound on α_true=0.5, so the median should sit "
        "slightly below 0.5; large deviations indicate the PAVA "
        "projection or the c_n threshold is wrong."
    )
    # Cell U-A: same Theorem-5 guarantee as Cell A, but with F_b=Uniform.
    assert res["alpha_L_u_null_zero_frac"] >= _PATRA_SEN_NULL_ZERO_FRAC, (
        f"Cell U-A zero-fraction {res['alpha_L_u_null_zero_frac']:.2f} < "
        f"{_PATRA_SEN_NULL_ZERO_FRAC}. Theorem 5 holds for any F_b under "
        "F continuous; failure here indicates the uniform-F_b branch in "
        "patra_sen_fit is wrong."
    )
    assert res["alpha_L_u_null_p95"] <= _PATRA_SEN_NULL_P95_MAX, (
        f"Cell U-A p95 α̂_L = {res['alpha_L_u_null_p95']:.4f} > "
        f"{_PATRA_SEN_NULL_P95_MAX}. Uniform null must give α̂_L ≈ 0 "
        "with the same finite-sample tightness as the Gaussian null."
    )
    # Cell U-B: median α̂_L for Uniform[0,1] + Beta(0.5, 5) mixture.
    assert (
        _PATRA_SEN_HALF_U_MED_LO <= res["alpha_L_u_half_p50"]
        <= _PATRA_SEN_HALF_U_MED_HI
    ), (
        f"Cell U-B median α̂_L = {res['alpha_L_u_half_p50']:.4f} outside "
        f"[{_PATRA_SEN_HALF_U_MED_LO}, {_PATRA_SEN_HALF_U_MED_HI}]. "
        "α̂_L is a LOWER bound on α_true=0.5; Beta(0.5,5) is concentrated "
        "near 0 so the mixture is well-separated from uniform and α̂_L "
        "should sit just below 0.5 with finite-sample slack."
    )


# Seed manifest for the C4 provenance artifact: every RNG seed that
# determines this gate's numbers. Hardcoded here AND in the calls below
# (single source would obscure the gate logic); kept in lockstep.
GATE_SEEDS = {
    "non_gaussianity_var1_params": 11,
    "non_gaussianity_simulate": 12,
    "non_gaussianity_anchors": 1,
    "non_gaussianity_student_t_df": 5,
    "patra_sen_recovery": 21,
    "recover_and_multistep": "see recover()/multistep_recovery() "
    "internal seeds (fixed, deterministic)",
}


def report() -> tuple[str, bool]:
    """Run all three gates, returning (human-readable text, all_pass).

    ONE renderer: the stdout view and the provenanced C4 artifact are
    byte-identical by construction. The gate computations are unchanged
    — only routed through a buffer instead of bare ``print``."""
    out: list[str] = []
    p = lambda *a: out.append(" ".join(str(x) for x in a))

    result, A, Q = recover()
    p("VAR(1) recovery:")
    p(f"  anchors evaluated  : {result.n_anchors}")
    p(f"  drift  rel err     : {result.drift_rel_err:.4f}  (tol {DRIFT_TOL})")
    p(f"  diff   rel err     : {result.diffusion_rel_err:.4f}  (tol {DIFFUSION_TOL})")
    p(f"  eig(Σ) rel err     : {result.eig_rel_err:.4f}  (tol {EIG_TOL})")
    ok = (
        result.n_anchors > 0
        and result.drift_rel_err < DRIFT_TOL
        and result.diffusion_rel_err < DIFFUSION_TOL
        and result.eig_rel_err < EIG_TOL
    )
    p("RESULT: " + ("PASS" if ok else "FAIL"))

    p("")
    p("Non-Gaussianity gate (production innovation_diagnostics):")
    A2, Q2 = make_var1_params(d=3, seed=11)
    Xg = simulate_var1(A2, Q2, n=6000, burn=500, seed=12)
    Xt = simulate_var1_t(A2, Q2, n=6000, burn=500, seed=12, df=5)
    gk, gt = _innov_kurt_over_anchors(Xg, 40, seed=1)
    tk, tt = _innov_kurt_over_anchors(Xt, 40, seed=1)
    p(f"  Gaussian noise : excess_kurt={gk:6.3f}  tail_ratio={gt:5.3f}")
    p(f"  Student-t(df=5): excess_kurt={tk:6.3f}  tail_ratio={tt:5.3f}")
    ng_ok = (
        gk < GAUSS_KURT_MAX
        and gt < GAUSS_TAIL_MAX
        and tk > T_KURT_MIN
        and tt > T_TAIL_MIN
        and (tk - gk) > KURT_SEP_MIN
        and (tt - gt) > TAIL_SEP_MIN
    )
    p("RESULT: " + ("PASS" if ng_ok else "FAIL"))

    p("")
    p("Multi-step composition gate (iterated Pi_Delta vs VAR(1) "
      "closed-form; error growth shown, not hidden):")
    ms = multistep_recovery()
    p(f"  {'H':>3} {'drift_relerr':>13} {'(tol)':>7} "
      f"{'diff_relerr':>12} {'(tol)':>7}")
    ms_ok = True
    for H, de, se in ms:
        dt_, st_ = _MS_DRIFT_TOL[H], _MS_DIFF_TOL[H]
        row_ok = de < dt_ and se < st_
        ms_ok &= row_ok
        p(f"  {H:>3} {de:>13.4f} {dt_:>7.2f} {se:>12.4f} {st_:>7.2f}"
          f"  {'ok' if row_ok else 'FAIL'}")
    p("RESULT: " + ("PASS" if ms_ok else "FAIL"))
    p("  (drift error compounds with horizon BY CONSTRUCTION -- this "
      "is the honest error-growth characterization, not a defect "
      "unless it breaches tolerance)")

    p("")
    p("Patra–Sen mixture-fraction recovery gate (production "
      "patra_sen_fit; preregistered for P1 of resolution-paths thread):")
    ps = patra_sen_recovery()
    p(f"  multi-seed characterization at n={ps['n']}, n_seeds={ps['n_seeds']}:")
    p(f"  α_true = 0.00 (pure standard-normal):")
    p(f"    P(α̂_L = 0)  = {ps['alpha_L_null_zero_frac']:.2f}   "
      f"(tol ≥ {_PATRA_SEN_NULL_ZERO_FRAC:.2f}; Theorem 5 says 0.95)")
    p(f"    α̂_L  p50    = {ps['alpha_L_null_p50']:.4f}")
    p(f"    α̂_L  p95    = {ps['alpha_L_null_p95']:.4f}   "
      f"(tol ≤ {_PATRA_SEN_NULL_P95_MAX:.2f})")
    p(f"    α̂_L  max    = {ps['alpha_L_null_max']:.4f}")
    p(f"    α̃_0  p50    = {ps['alpha_hat_heuristic_null_p50']:.4f}  "
      "(§5 heuristic, diagnostic only)")
    p(f"  α_true = 0.50 (50/50 N(0,1) + N(3,1)):")
    p(f"    α̂_L  p5     = {ps['alpha_L_half_p5']:.4f}")
    p(f"    α̂_L  p50    = {ps['alpha_L_half_p50']:.4f}   "
      f"(tol [{_PATRA_SEN_HALF_MED_LO:.2f}, {_PATRA_SEN_HALF_MED_HI:.2f}])")
    p(f"    α̂_L  p95    = {ps['alpha_L_half_p95']:.4f}")
    p(f"    α̃_0  p50    = {ps['alpha_hat_heuristic_half_p50']:.4f}  "
      "(§5 heuristic, diagnostic only)")
    p("  --- F_b = Uniform[0,1] (P1 methodology precondition) ---")
    p(f"  α_true = 0.00 (pure Uniform[0,1]):")
    p(f"    P(α̂_L = 0)  = {ps['alpha_L_u_null_zero_frac']:.2f}   "
      f"(tol ≥ {_PATRA_SEN_NULL_ZERO_FRAC:.2f}; Theorem 5 same as Cell A)")
    p(f"    α̂_L  p50    = {ps['alpha_L_u_null_p50']:.4f}")
    p(f"    α̂_L  p95    = {ps['alpha_L_u_null_p95']:.4f}   "
      f"(tol ≤ {_PATRA_SEN_NULL_P95_MAX:.2f})")
    p(f"    α̂_L  max    = {ps['alpha_L_u_null_max']:.4f}")
    p(f"  α_true = 0.50 (50/50 Uniform[0,1] + Beta(0.5, 5)):")
    p(f"    α̂_L  p5     = {ps['alpha_L_u_half_p5']:.4f}")
    p(f"    α̂_L  p50    = {ps['alpha_L_u_half_p50']:.4f}   "
      f"(tol [{_PATRA_SEN_HALF_U_MED_LO:.2f}, {_PATRA_SEN_HALF_U_MED_HI:.2f}])")
    p(f"    α̂_L  p95    = {ps['alpha_L_u_half_p95']:.4f}")
    p(f"  c_n = {ps['c_n']:.4f} (Cramér–von Mises 95% asymptotic quantile)")
    ps_ok = (
        ps["alpha_L_null_zero_frac"] >= _PATRA_SEN_NULL_ZERO_FRAC
        and ps["alpha_L_null_p95"] <= _PATRA_SEN_NULL_P95_MAX
        and _PATRA_SEN_HALF_MED_LO <= ps["alpha_L_half_p50"] <= _PATRA_SEN_HALF_MED_HI
        and ps["alpha_L_u_null_zero_frac"] >= _PATRA_SEN_NULL_ZERO_FRAC
        and ps["alpha_L_u_null_p95"] <= _PATRA_SEN_NULL_P95_MAX
        and _PATRA_SEN_HALF_U_MED_LO <= ps["alpha_L_u_half_p50"] <= _PATRA_SEN_HALF_U_MED_HI
    )
    p("RESULT: " + ("PASS" if ps_ok else "FAIL"))

    return "\n".join(out) + "\n", bool(ok and ng_ok and ms_ok and ps_ok)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="synthetic validation gates")
    ap.add_argument(
        "--emit-result", action="store_true",
        help="write the provenanced METHOD-grade artifact (C4) via "
             "experiment.provenance.make_result (refuses a dirty tree)",
    )
    args = ap.parse_args()

    text, all_pass = report()
    print(text, end="")

    if args.emit_result:
        from config import PROJECT_ROOT
        from experiment.provenance import Grade, make_result

        out_path = (
            PROJECT_ROOT / "experiment" / "results"
            / "synthetic_validation_gates.txt"
        )
        hdr = make_result(
            path=out_path,
            grade=Grade.METHOD,
            title="Estimator validation gates: VAR(1) recovery + "
                  "non-Gaussianity + multi-step composition "
                  "(production-path)",
            body=text,
            inputs={"all_gates_pass": all_pass},
            seeds=GATE_SEEDS,
            frozen_spec_required=False,  # synthetic VAR(1), not the
            #                              frozen Ontario predictor
        )
        print(f"\nwrote provenanced C4 artifact -> {out_path}")
        print(f"  all_gates_pass     = {all_pass}")
        print(f"  inputs_fingerprint = {hdr['inputs_fingerprint'][:16]}…")
        print(f"  body_sha256        = {hdr['body_sha256'][:16]}…")
        raise SystemExit(0 if all_pass else 1)
