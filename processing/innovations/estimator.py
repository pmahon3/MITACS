"""Local Gaussian predictive-semigroup estimator.

For each anchor time the estimator fits a local linear drift ``C_j`` by
weighted least squares whose Gaussian drift bandwidth ``theta_j`` is chosen
per anchor by **true leave-one-out cross-validation** on one-step forecast
error, and a local diffusion ``Sigma_j`` = the mu-centred *plain* covariance
of the locally-fitted residuals ``Y - X @ C_j`` (NO residual kernel).

Why this design (see memory ``mitacs-theta-rail-pinning`` for the full
evidence trail):

  - The edynamics ``LocalGLSelector`` is degenerate for the normalized
    Gaussian kernel: its GL score has no interior optimum (sigma* pins at
    any grid ceiling; score monotone-decreasing as bandwidth -> inf).
    Verified empirically. The Resolvent_Framework programme prescribes no
    bandwidth-selection convention to defer to (verified). So the rule is
    a settled applied-statistics choice, not a theory question.
  - true-LOO-CV is the only candidate that is well-posed (genuinely
    U-shaped objective), non-degenerate, per-anchor adaptive, and free of
    extra hyperparameters.
  - Dropping the residual kernel: diffusion is still residual-based and
    still spatially local (theta localizes which points enter the fit);
    only the residual-magnitude reweighting is removed. The kernel
    systematically shrank Sigma toward the small-residual core (the
    collapse pathology in mild form); the plain covariance is the
    unbiased local second moment -- the honest ``kappa_Q`` 2nd moment,
    not a robustified proxy. Empirically, CV-likelihood when free to pick
    sigma reproduces the plain covariance within ~10%, and the headline
    spectral structure (r_hat) is invariant across all bandwidth rules.

Together ``(C_j, Sigma_j)`` parameterise the per-anchor Gaussian Markov
kernel ``x' | x ~ N(x @ C_j, Sigma_j)`` on the delay-embedding state space.

Theoretical correspondence to the Resolvent_Framework programme (Paper II),
stated precisely with its qualifiers (see the project memory note
``mitacs-theory-correspondence`` for the full mapping):

  This is a finite-sample, parametric, SINGLE-STEP, *locally-Gaussian*
  estimator of the programme's conditional-regularity (disintegration)
  kernel ``kappa_Q`` at the embedding layer -- the layer it targets. From
  ``(C_j, Sigma_j)`` the operator pair ``K_Delta`` / ``P_Delta*`` is
  recoverable; ``Sigma_j != 0`` is the programme's non-Dirac regime, and
  ``Sigma_j -> 0`` reproduces the classical-Koopman collapse.

  Qualifiers (do not drop): (1) register -- the programme *discloses*
  ``kappa_Q`` from the measure; this *constructs* a Gaussian proxy and
  fits it (same target, opposite register: "estimator of", not "instance
  of"); (2) Gaussianity is imposed, exact only where the local
  conditional law is Gaussian; (3) single-step only -- Chapman-Kolmogorov
  / the ``{Pi_t}`` semigroup is neither constructed nor verified, so this
  is ``Pi_Delta`` (the generator/slice), not the semigroup.

No novelty/operator-naming claims are made here; terminology defers to
the Resolvent_Framework programme.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from edynamics.modelling_tools import Embedding
from edynamics.modelling_tools.kernels import Gaussian


@dataclass
class SemigroupEstimate:
    """Per-anchor local Gaussian semigroup parameters.

    Shapes (``N`` anchors, ``d`` embedding dimension):
        coefficients : (N, d, d)  -- drift ``C_j`` (``x_next = x @ C_j``)
        covariances  : (N, d, d)  -- diffusion ``Sigma_j`` (SPD)
        resid_means  : (N, d)     -- residual mean ``mu_j``
        eigvals      : (N, d)     -- ascending eigenvalues of ``Sigma_j``
        theta_star   : (N,)       -- per-anchor true-LOO-CV drift bandwidth
        sigma_star   : (N,)       -- OBSOLETE: residual kernel was dropped;
                                     filled with NaN, retained only so the
                                     save/load interface contract is stable
        anchor_times : (N,) int64 -- nanosecond anchor timestamps
    """

    coefficients: np.ndarray
    covariances: np.ndarray
    resid_means: np.ndarray
    eigvals: np.ndarray
    theta_star: np.ndarray
    sigma_star: np.ndarray
    anchor_times: np.ndarray


def _gauss_w(dist: np.ndarray, theta: float, dim: int) -> np.ndarray:
    """Normalized Gaussian kernel weights (matches edynamics ``Gaussian``)."""
    norm = (2.0 * np.pi) ** (-dim / 2) * (1.0 / theta**dim)
    return norm * np.exp(-0.5 * (dist / theta) ** 2)


def _theta_loo_cv(
    dists: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    d: int,
    n_grid: int = 18,
    sub: int = 150,
    seed: int = 0,
) -> float:
    """Per-anchor drift bandwidth by TRUE leave-one-out one-step CV.

    True LOO (refit C excluding the held-out row) is genuinely U-shaped in
    theta; an *in-sample* weighted residual is monotone and rail-pins. LOO
    is O(m^2) per theta, so it is evaluated exactly on a bounded random
    subsample of ``sub`` library points (no convention-sensitive
    hat-matrix shortcut -- a buggy shortcut was the reason this is naive).
    """
    n = len(dists)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, sub, replace=False) if n > sub else np.arange(n)
    Xs, Ys, ds = X[idx], Y[idx], dists[idx]
    m = len(idx)
    grid = np.geomspace(max(ds.min(), 1e-3), ds.max(), n_grid)
    best = (np.inf, grid[len(grid) // 2])
    for th in grid:
        w = _gauss_w(ds, th, d)
        if w.sum() <= 0:
            continue
        tot = 0.0
        for i in range(m):
            keep = np.arange(m) != i
            wi = w[keep]
            Ci = np.linalg.lstsq(
                wi[:, None] * Xs[keep], wi[:, None] * Ys[keep], rcond=None
            )[0]
            tot += np.sum((Ys[i] - Xs[i] @ Ci) ** 2)
        if tot / m < best[0]:
            best = (tot / m, th)
    return float(best[1])


def local_drift_and_diffusion(
    *,
    embedding: Embedding,
    anchor: pd.Timestamp,
    day_anchor_hour: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """Per-anchor ``(C, Sigma, mu, theta*, resid)``.

    ``resid`` is the production residual array ``Y - X@C`` (after any
    day-anchor masking) -- returned so diagnostics can read it without
    reimplementing the fit.

    Drift ``C``: WLS with Gaussian kernel at the true-LOO-CV bandwidth
    ``theta*``. Diffusion ``Sigma``: mu-centred *plain* covariance of the
    locally-fitted residuals ``Y - X @ C`` -- NO residual kernel. Spatial
    locality is carried entirely by ``theta*`` (which points enter the
    fit); the diffusion is the unbiased local second moment of those
    residuals, not a kernel-shrunk proxy. Leave-one-out (drop the anchor
    row) mirrors the old ``project(leave_out=True)``.

    ``day_anchor_hour``: if set, transitions whose one-step *target*
    timestamp falls on that hour are dropped from the fit and the
    residuals. These are the day-type rollover seam: the one-step map
    spans the day-anchor boundary, producing artificial discontinuous
    transitions that otherwise dominate ``Sigma`` (Ontario weekday
    per-component residual excess kurtosis 133 -> 5 when masked; ~4% of
    steps; see memory ``mitacs-dayanchor-seam``). ``None`` (default) =
    no masking -- correct for data with no day-anchor structure (e.g.
    the VAR(1) validation gate).
    """
    d = embedding.block.shape[1]
    block = embedding.block
    blk = block.loc[block.index != anchor]
    X_df = blk.iloc[:-1]
    Y_df = blk.iloc[1:]

    if day_anchor_hour is not None:
        keep = Y_df.index.hour != day_anchor_hour
        X_df, Y_df = X_df[keep], Y_df[keep]

    X = X_df.values
    Y = Y_df.values
    x0 = block.loc[anchor].values
    dists = np.linalg.norm(X - x0, axis=1)

    theta = _theta_loo_cv(dists, X, Y, d)
    w = _gauss_w(dists, theta, d)
    C = np.linalg.lstsq(w[:, None] * X, w[:, None] * Y, rcond=None)[0]

    resid = Y - X @ C
    mu = resid.mean(axis=0)
    rc = resid - mu[None, :]
    Sigma = rc.T @ rc / len(rc)              # plain (unweighted) covariance
    return C, Sigma, mu, theta, resid


def innovation_diagnostics(
    *,
    embedding: Embedding,
    anchor: pd.Timestamp,
    day_anchor_hour: int | None = None,
) -> dict:
    """Univariate non-Gaussianity of the scalar one-step innovation.

    For a single-variable delay embedding, ``Sigma_j`` is rank-1 by
    construction (coords 2..d of the image are deterministic shifts of the
    input -- see memory ``mitacs-rank1-structural``). The ONLY stochastic
    content is the coordinate-0 residual r0 = z_{t+1} - (X@C)_0. So
    non-Gaussianity is a univariate question on r0, not a multivariate
    (Mardia) one -- the multivariate machinery was the source of earlier
    inflated/degenerate numbers.

    Reuses :func:`local_drift_and_diffusion` for the production fit and the
    exact production residuals (no reimplementation); reads only r0.

    Returns ``innov_var`` (= Sigma[0,0], the one scalar diffusion
    content), ``excess_kurt`` (Fisher; 0 = Gaussian), and ``tail_ratio``
    (robust, outlier-resistant: the r0 (0.5, 99.5) inter-quantile range
    over its IQR, divided by the same ratio for a standard normal so
    Gaussian ~ 1.0 -- cross-checks the fragile kurtosis).
    """
    _C, Sigma, _mu, _theta, resid = local_drift_and_diffusion(
        embedding=embedding, anchor=anchor, day_anchor_hour=day_anchor_hour
    )
    r0 = resid[:, 0]
    r0 = r0 - r0.mean()

    n = r0.size
    s = r0.std()
    excess_kurt = float(np.mean(r0**4) / (s**4) - 3.0) if s > 0 else float("nan")

    q005, q25, q75, q995 = np.quantile(r0, [0.005, 0.25, 0.75, 0.995])
    iqr = q75 - q25
    # standard-normal reference for (q99.5-q0.5)/(q75-q25); derived from
    # scipy (= 3.818930) so the constant can't silently drift.
    from scipy.stats import norm as _norm

    _NORM_REF = (_norm.ppf(0.995) - _norm.ppf(0.005)) / (
        _norm.ppf(0.75) - _norm.ppf(0.25)
    )
    tail_ratio = (
        float(((q995 - q005) / iqr) / _NORM_REF) if iqr > 0 else float("nan")
    )

    return {
        "innov_var": float(Sigma[0, 0]),
        "excess_kurt": excess_kurt,
        "tail_ratio": tail_ratio,
        "n": int(n),
    }


def build_local_gaussian_semigroup(
    *,
    embedding: Embedding,
    anchors: pd.DatetimeIndex,
    day_anchor_hour: int | None = None,
    **_legacy,
) -> SemigroupEstimate:
    """Fit per-anchor local Gaussian semigroup parameters.

    Drift bandwidth per anchor by true-LOO-CV; diffusion = plain residual
    covariance (no residual kernel). ``day_anchor_hour`` (if set) masks
    day-type-rollover-seam transitions -- see
    :func:`local_drift_and_diffusion`. ``**_legacy`` swallows the
    now-unused ``theta_grid``/``sigma_grid``/``gl_penalty_C`` kwargs so
    existing callers keep working without change.
    """
    anchors = pd.DatetimeIndex(anchors)
    d = embedding.block.shape[1]
    N = len(anchors)

    C_all = np.empty((N, d, d), dtype=float)
    Sigma_all = np.empty((N, d, d), dtype=float)
    mu_all = np.empty((N, d), dtype=float)
    eig_all = np.empty((N, d), dtype=float)
    theta_star = np.empty(N, dtype=float)

    for i, anchor_t in enumerate(anchors):
        C, Sigma, mu, theta, _resid = local_drift_and_diffusion(
            embedding=embedding,
            anchor=anchor_t,
            day_anchor_hour=day_anchor_hour,
        )
        C_all[i] = C
        Sigma_all[i] = Sigma
        mu_all[i] = mu
        eig_all[i] = np.linalg.eigvalsh(Sigma)  # ascending, symmetric
        theta_star[i] = theta

    return SemigroupEstimate(
        coefficients=C_all,
        covariances=Sigma_all,
        resid_means=mu_all,
        eigvals=eig_all,
        theta_star=theta_star,
        sigma_star=np.full(N, np.nan),  # obsolete; interface stability only
        anchor_times=anchors.asi8.astype(np.int64),
    )
