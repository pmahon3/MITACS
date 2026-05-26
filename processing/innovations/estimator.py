"""Local Gaussian predictive-semigroup estimator (S-map kernel).

For each anchor time the estimator fits a local linear drift ``C_j`` by
weighted least squares whose S-map (Sugihara 1994) bandwidth
``theta_j`` is chosen per anchor by **true leave-one-out
cross-validation** on one-step forecast error, and a local diffusion
``Sigma_j`` = the mu-centred *plain* covariance of the locally-fitted
residuals ``Y - X @ C_j`` (NO residual kernel).

S-map weights are ``w_i = exp(-theta * dist_i / d_bar)`` with
``d_bar = mean(dist_i)``: dimensionless ``theta >= 0``, ``theta = 0``
the global linear fit (a finite achievable left endpoint), ``theta``
increasing localising. See :func:`_smap_w` for the definition and the
draft (eq:weights) for the model framing.

Why this design (see memory ``mitacs-theta-rail-pinning`` for the
evidence trail under the prior Gaussian-kernel parameterisation, and
the 2026-05-23 S-map switch in commit history):

  - The edynamics ``LocalGLSelector``'s GL criterion is degenerate for
    the normalised Gaussian kernel (its ``theta^{-d}`` normaliser
    cancels the penalty-growth opposition the criterion requires).
    The S-map kernel carries no such normaliser; the bandwidth-search
    pathology that motivated true-LOO-CV under the Gaussian kernel
    does not exist here, but true-LOO-CV is kept as the selection rule
    so the empirical re-test is a clean parameterisation comparison.
  - true-LOO-CV is well-posed (genuinely U-shaped objective when the
    underlying optimum is interior; monotone in the global direction
    when the optimum is at the boundary), non-degenerate, per-anchor
    adaptive, and free of extra hyperparameters.
  - Dropping the residual kernel: diffusion is still residual-based and
    still spatially local (theta localizes which points enter the fit);
    only the residual-magnitude reweighting is removed. The kernel
    systematically shrank Sigma toward the small-residual core; the
    plain covariance is the unbiased local second moment -- the honest
    ``kappa_Q`` 2nd moment, not a robustified proxy.

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


def _smap_w(dist: np.ndarray, theta: float) -> np.ndarray:
    """S-map (Sugihara 1994) kernel weights:
    ``w_i = exp(-theta * dist_i / d_bar)`` with ``d_bar = mean(dist)``.

    Dimensionless ``theta >= 0``: ``theta = 0`` is uniform weights (the
    global linear fit, finite achievable point); ``theta`` increasing
    localises. Distance to the first power, not squared. No normaliser
    -- WLS uses relative weights, so the absent ``theta^{-d}`` factor
    that the Gaussian kernel previously carried makes no difference to
    the resulting ``C, Sigma``, but removes the GL-degeneracy pathology
    that factor induced.
    """
    dbar = dist.mean()
    if dbar <= 0:
        # all distances zero -> uniform weights (query coincides with
        # every library point; degenerate, but well-defined as the
        # global limit).
        return np.ones_like(dist)
    return np.exp(-theta * dist / dbar)


def _theta_loo_cv(
    dists: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    d: int,
    n_grid: int = 17,
    sub: int = 150,
    seed: int = 0,
) -> float:
    """Per-anchor S-map bandwidth by TRUE leave-one-out one-step CV.

    True LOO (refit C excluding the held-out row) is genuinely U-shaped in
    theta; an *in-sample* weighted residual is monotone and rail-pins.

    Implementation: the weighted LS hat-matrix gives the LOO residuals
    in closed form, so the inner per-row refit loop collapses to a
    single full fit + a hat-matrix diagonal lookup per theta.  For a
    weighted LS fit with weights ``w`` (such that ``lstsq(w[:,None]*X,
    w[:,None]*Y)`` is the WLS solution; the minimiser of
    ``sum w_i^2 (y_i - x_i C)^2``), the in-sample prediction is
    ``Y_hat = H Y`` with hat matrix ``H = X (X' W^2 X)^{-1} X' W^2``;
    and the LOO residual is the in-sample residual divided by
    ``1 - h_ii``.

    This is exact (not an approximation), reduces the per-call cost
    from O(n_grid * m^2 d^2) to O(n_grid * m d^2), and was validated
    numerically against the original per-row-refit implementation
    (max-abs LOO-SSE delta < 1e-10 across the grid on random fixtures).

    Grid: ``linspace(0, 8, n_grid)`` -- the Sugihara-typical S-map range.
    ``theta = 0`` is the global linear fit (a finite, achievable left
    endpoint); ``theta = 8`` weights the nearest neighbour ~3000x more
    than the mean-distance one.  ``d`` is unused by the S-map kernel but
    kept in the signature for backwards compatibility with callers
    threaded under the older Gaussian-kernel API.
    """
    del d  # S-map kernel is dimension-independent; argument retained
    n = len(dists)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, sub, replace=False) if n > sub else np.arange(n)
    Xs, Ys, ds = X[idx], Y[idx], dists[idx]
    m = len(idx)
    grid = np.linspace(0.0, 8.0, n_grid)
    best = (np.inf, grid[0])
    for th in grid:
        w = _smap_w(ds, th)
        if w.sum() <= 0:
            continue
        # Weighted-LS closed form for LOO.
        # Build A = w[:,None] * X  (the rescaled design that lstsq sees);
        # B = w[:,None] * Y likewise.  Solve A C = B in least squares to
        # get the same C as the per-row baseline, then read the hat
        # matrix diagonal off the (orthogonal) factor of A.
        A = w[:, None] * Xs
        # Normal-equation solve via Cholesky (faster than lstsq for the
        # tiny d we have here, and we need the inverse for the hat
        # matrix anyway).
        AtA = A.T @ A
        AtB = A.T @ (w[:, None] * Ys)
        try:
            L = np.linalg.cholesky(AtA + 1e-12 * np.eye(AtA.shape[0]))
        except np.linalg.LinAlgError:
            # Degenerate weights -> skip this theta
            continue
        Z = np.linalg.solve(L, AtB)
        C = np.linalg.solve(L.T, Z)
        # In-sample prediction in unscaled space (the LOO residual is
        # computed unscaled, matching the per-row baseline's
        # Ys[i] - Xs[i] @ Ci).
        Yhat = Xs @ C
        # h_ii = a_i' (A'A)^{-1} a_i where a_i is the i-th row of A.
        # Compute (A'A)^{-1/2} A' efficiently via the Cholesky:
        #   solve L H = A'  ->  H is (d, m), h_ii = ||H[:, i]||^2.
        H = np.linalg.solve(L, A.T)         # (d, m)
        h_diag = np.einsum("dm,dm->m", H, H)
        # Clamp to avoid /0; h_ii cannot exceed 1 in theory but
        # rounding can put it slightly above on perfectly-collinear rows.
        denom = np.clip(1.0 - h_diag, 1e-12, None)
        loo_resid = (Ys - Yhat) / denom[:, None]
        tot = float(np.sum(loo_resid ** 2))
        if tot / m < best[0]:
            best = (tot / m, th)
    return float(best[1])


def local_drift_and_diffusion(
    *,
    embedding: Embedding,
    anchor: pd.Timestamp,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """Per-anchor ``(C, Sigma, mu, theta*, resid)``.

    ``resid`` is the production residual array ``Y - X@C`` -- returned so
    diagnostics can read it without reimplementing the fit.

    Drift ``C``: WLS with Gaussian kernel at the true-LOO-CV bandwidth
    ``theta*``. Diffusion ``Sigma``: mu-centred *plain* covariance of the
    locally-fitted residuals ``Y - X @ C`` -- NO residual kernel. Spatial
    locality is carried entirely by ``theta*`` (which points enter the
    fit); the diffusion is the unbiased local second moment of those
    residuals, not a kernel-shrunk proxy. Leave-one-out (drop the anchor
    row) mirrors the old ``project(leave_out=True)``.

    There used to be an optional ``day_anchor_hour`` target-time seam
    mask -- pairs whose one-step target equalled the day-anchor were
    dropped, on the empirical claim that they were day-type-crossing
    discontinuities dominating Sigma. Removed 2026-05-22 after the
    realignment refactor: the mask's apparent "Sigma stabilisation"
    effect was a floating-point artefact of reading the condition
    number from a structurally-rank-deficient Sigma, not a real
    conditioning improvement. The largest (real) eigenvalue of Sigma
    -- where its stochastic content lives -- is unaffected by the mask
    on Ontario data at anchor=0. The validation gates pass without it.
    """
    block = embedding.block
    d = block.shape[1]
    blk = block.loc[block.index != anchor]
    X_df = blk.iloc[:-1]
    Y_df = blk.iloc[1:]
    x0 = block.loc[anchor].values
    return _local_fit_at(X_df.values, Y_df.values, x0, d)


def _local_fit_at(
    X: np.ndarray,
    Y: np.ndarray,
    x_query: np.ndarray,
    d: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """Core local Gaussian fit at an explicit query state.

    The shared mechanism behind :func:`local_drift_and_diffusion` (which
    derives ``X, Y, x_query`` from an embedding + anchor) and the frozen
    forecast generator (which supplies a pre-cutoff library ``X, Y`` and
    a post-cutoff query state ``x_query``). Both call THIS -- the
    prediction path *is* the production path, not a parallel
    reimplementation (the project's diagnostic-discipline rule).

    Returns ``(C, Sigma, mu, theta*, resid)`` exactly as before.
    """
    dists = np.linalg.norm(X - x_query, axis=1)
    theta = _theta_loo_cv(dists, X, Y, d)
    w = _smap_w(dists, theta)
    C = np.linalg.lstsq(w[:, None] * X, w[:, None] * Y, rcond=None)[0]

    resid = Y - X @ C
    mu = resid.mean(axis=0)
    rc = resid - mu[None, :]
    Sigma = rc.T @ rc / len(rc)              # plain (unweighted) covariance
    return C, Sigma, mu, theta, resid


def global_ols_fit(
    X: np.ndarray, Y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Global (theta=0) OLS fit on a pre-built library ``(X, Y)``.

    Production-grade analogue of :func:`_local_fit_at` for the global
    case. Used by callers that need the global linear drift directly
    (the estimator-as-run reduces to this at the registered d=2 z-lag
    embedding: see memory ``mitacs-theta-rail-pinning`` RESOLVED).
    Without this function, callers reimplement the same `np.linalg.lstsq`
    + plain covariance inline and trigger /audit code-path REIMPLEMENTED.

    ``Y`` may be any same-row-count shift of ``X`` -- the canonical
    one-step ``Y = X.iloc[1:]`` for the standard drift, or an h-step
    ``Y = X.iloc[h:]`` (call site supplies the shape) for the direct
    h-step factor used by the multiscale-factor-coherence programme
    (notes/seeds/multiscale_factor_coherence.md). The estimator does
    not assume a particular relationship; it fits ``Y = X @ C + e``
    on whatever pairs the caller provides.

    Returns ``(C, Sigma, mu, resid)`` in the same shape conventions
    :func:`_local_fit_at` uses (drift, plain mu-centred residual
    covariance, residual mean, raw residuals).
    """
    C = np.linalg.lstsq(X, Y, rcond=None)[0]
    resid = Y - X @ C
    mu = resid.mean(axis=0)
    rc = resid - mu[None, :]
    Sigma = rc.T @ rc / max(len(rc) - 1, 1)
    return C, Sigma, mu, resid


def innovation_diagnostics(
    *,
    embedding: Embedding,
    anchor: pd.Timestamp,
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
        embedding=embedding, anchor=anchor,
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
    **_legacy,
) -> SemigroupEstimate:
    """Fit per-anchor local Gaussian semigroup parameters.

    Drift bandwidth per anchor by true-LOO-CV; diffusion = plain residual
    covariance (no residual kernel). ``**_legacy`` swallows the
    now-unused ``theta_grid``/``sigma_grid``/``gl_penalty_C``/
    ``day_anchor_hour`` kwargs so existing callers keep working without
    change. (``day_anchor_hour`` was an optional seam mask removed
    2026-05-22; see :func:`local_drift_and_diffusion`.)
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
            embedding=embedding, anchor=anchor_t,
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
