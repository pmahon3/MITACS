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


def student_t_mle_fit(
    X: np.ndarray, Y: np.ndarray,
    *,
    nu_bounds: tuple[float, float] = (2.5, 300.0),
    max_irls_iters: int = 100,
    tol: float = 1e-7,
) -> tuple[np.ndarray, float, float, np.ndarray]:
    """Maximum-likelihood fit of a univariate Student-t linear-regression
    model ``Y[:, 0] = X @ C[:, 0] + r`` with ``r ~ Student-t(0, s, nu)``.

    Production-grade function for the distributional-class line of
    inquiry (notes/preregistrations/2026-05-26_distributional-class-thread/).
    Without it, the Student-t MLE would be reimplemented inline in each
    script that needs it and trigger /audit code-path REIMPLEMENTED.

    Mathematical specification (see writeup/tex/missing_content_memo.tex
    §4.4):

    * Log-likelihood (eq. t-loglik). The IRLS regression update (eq.
      t-irls) uses Student-t-induced weights
      ``w_i = (nu + 1) / (nu + r_i^2 / s^2)``; the scale equation
      (eq. t-scale) is implicit and solved by fixed-point iteration;
      nu is updated by scalar optimization on the profile
      log-likelihood (Brent's method via :func:`scipy.optimize.minimize_scalar`).

    * ``nu_bounds`` constrains the degrees-of-freedom search. Boundary
      hits (within 1% of either edge) are flagged via a returned
      ``info['nu_at_boundary']`` field; callers (e.g.\\ the
      distributional-class Q1 R-D criterion) check this to detect
      pathological fits.

    * Only the coord-0 residual is modelled. Higher coordinates of the
      embedding are deterministic shifts under the single-variable
      delay structure (mitacs-rank1-structural); the rank-1 innovation
      is the only stochastic content.

    Returns ``(C, s, nu, resid)`` where ``C`` is the (d, 1) drift
    matrix from the IRLS regression on coord-0, ``s`` is the
    Student-t scale parameter (not the standard deviation; the
    variance is ``s^2 * nu / (nu - 2)`` for ``nu > 2``), ``nu`` the
    fitted degrees of freedom, and ``resid`` the coord-0 residual
    vector after the converged fit.
    """
    from scipy.optimize import minimize_scalar
    from scipy.special import digamma, gammaln

    n, d = X.shape
    if Y.ndim == 1:
        y = Y.astype(float)
    else:
        y = Y[:, 0].astype(float)
    X = X.astype(float)

    # initialize with OLS (the Gaussian limit nu -> infinity)
    C_col, *_ = np.linalg.lstsq(X, y, rcond=None)
    C = C_col.reshape(d, 1)
    resid = y - (X @ C).ravel()
    s2 = max(float(np.mean(resid ** 2)), 1e-12)
    nu = 5.0   # central starting point; small enough for heavy tails, not at boundary

    def _profile_neg_log_lik(log_nu: float) -> float:
        """profile -log-lik in nu, with C, s held fixed at current."""
        nu_local = float(np.exp(log_nu))
        nu_local = max(nu_bounds[0], min(nu_bounds[1], nu_local))
        s_local = float(np.sqrt(s2))
        ll = (
            n * gammaln((nu_local + 1) / 2)
            - n * gammaln(nu_local / 2)
            - 0.5 * n * np.log(nu_local * np.pi)
            - n * np.log(s_local)
            - ((nu_local + 1) / 2) * float(
                np.sum(np.log1p(resid ** 2 / (nu_local * s2)))
            )
        )
        return -ll

    for _ in range(max_irls_iters):
        # weights from Student-t score (eq. t-irls)
        w = (nu + 1.0) / (nu + resid ** 2 / max(s2, 1e-12))
        # weighted least squares update for C
        Xw = X * w[:, None]
        XtX = X.T @ Xw
        Xty = X.T @ (w * y)
        try:
            C_new_col = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            C_new_col = np.linalg.lstsq(XtX, Xty, rcond=None)[0]
        C_new = C_new_col.reshape(d, 1)
        resid_new = y - (X @ C_new).ravel()
        # scale update via the implicit Student-t equation (eq. t-scale)
        s2_new = float(
            np.mean(
                (nu + 1.0) * resid_new ** 2
                / (nu + resid_new ** 2 / max(s2, 1e-12))
            )
        )
        s2_new = max(s2_new, 1e-12)
        # nu update via Brent on log nu (1D)
        res_brent = minimize_scalar(
            _profile_neg_log_lik,
            bounds=(np.log(nu_bounds[0]), np.log(nu_bounds[1])),
            method="bounded",
        )
        nu_new = float(np.exp(res_brent.x))
        # convergence
        rel_change = (
            np.linalg.norm(C_new - C) / max(np.linalg.norm(C), 1e-12)
            + abs(s2_new - s2) / max(s2, 1e-12)
            + abs(nu_new - nu) / max(nu, 1e-12)
        )
        C = C_new
        resid = resid_new
        s2 = s2_new
        nu = nu_new
        if rel_change < tol:
            break

    return C, float(np.sqrt(s2)), float(nu), resid


def mixture_2_gaussian_mle_fit(
    X: np.ndarray, Y: np.ndarray,
    *,
    max_em_iters: int = 200,
    tol: float = 1e-7,
    seed: int = 0,
) -> tuple[np.ndarray, dict, np.ndarray]:
    """MLE fit of ``Y[:, 0] = X @ C + r`` with ``r`` distributed as a
    symmetric centred mixture of two zero-mean Gaussians.

    Production-grade fitter for the distributional-class line of
    inquiry's Q2A node ("richer-family-mixture-or-nonparametric";
    notes/preregistrations/2026-05-26_distributional-class-thread/).
    Without it, the mixture MLE would be reimplemented inline in each
    Q2A script and trigger /audit code-path REIMPLEMENTED.

    Model: ``r ~ w1 * N(0, s1^2) + w2 * N(0, s2^2)`` with ``w1 + w2 = 1``.
    Both components share location 0 (symmetric / centred) so the
    mixture is identifiable from a single scalar residual stream with
    the regression slope absorbing any non-zero mean.

    Algorithm: OLS warm-start for the drift (the Gaussian limit of the
    family); EM on residuals alternating posterior-responsibility E-step
    and weighted-variance M-step (McLachlan & Peel 2000, "Finite Mixture
    Models", §2.8 — the textbook two-component centred-Normal-mixture
    EM). Plain numpy throughout — no sklearn — so the EM is fully under
    audit control. Deterministic given the seed (the EM init is
    deterministic from the OLS residuals; the seed is reserved for any
    future random init that callers might enable).

    Returns ``(C, params, resid)`` matching the ``(X, Y) -> (C, family_params,
    resid)`` calling convention used by :func:`student_t_mle_fit`:

    * ``C`` is the (d, 1) drift matrix (coord-0 only; OLS estimator
      under the centred-mixture model — the symmetric components mean
      OLS on coord-0 *is* the MLE for the location parameter).
    * ``params`` = ``{"weights": (w1, w2), "scales": (s1, s2)}``.
    * ``resid`` is the (n,) coord-0 residual after the converged fit.
    """
    del seed   # reserved for a future random-init variant; current init
    #            is deterministic from the OLS residual quantiles.
    n, d = X.shape
    if Y.ndim == 1:
        y = Y.astype(float)
    else:
        y = Y[:, 0].astype(float)
    X = X.astype(float)

    # OLS for the drift (centred mixture: OLS on coord-0 is MLE of the
    # location since each component has mean 0).
    C_col, *_ = np.linalg.lstsq(X, y, rcond=None)
    C = C_col.reshape(d, 1)
    resid = y - (X @ C).ravel()

    # Initialize EM: split the residual variance between a "core" (s1)
    # and a "tail" (s2) by quantile inspection of |r|. The bottom-50%
    # |r| determines s1; the top-10% |r| determines s2 (sqrt of the
    # corresponding 2nd moments).
    r = resid - resid.mean()
    abs_r = np.abs(r)
    q50, q90 = np.quantile(abs_r, [0.50, 0.90])
    s1 = max(q50 / 0.6745, 1e-6)   # 0.6745 = Phi^-1(0.75) so q50(|N|) = 0.6745 sigma
    s2 = max(q90 / 1.6449, s1 * 1.5)  # 0.95-quantile factor for half-normal
    w1, w2 = 0.7, 0.3
    log2pi = float(np.log(2 * np.pi))

    def _log_norm_pdf(x: np.ndarray, sigma: float) -> np.ndarray:
        return -0.5 * (x * x) / (sigma * sigma) - np.log(sigma) - 0.5 * log2pi

    prev_ll = -np.inf
    for _ in range(max_em_iters):
        # E-step: posterior responsibilities (log-sum-exp stabilized)
        log_p1 = np.log(max(w1, 1e-300)) + _log_norm_pdf(r, s1)
        log_p2 = np.log(max(w2, 1e-300)) + _log_norm_pdf(r, s2)
        log_max = np.maximum(log_p1, log_p2)
        log_denom = log_max + np.log(np.exp(log_p1 - log_max) + np.exp(log_p2 - log_max))
        gamma1 = np.exp(log_p1 - log_denom)
        gamma2 = 1.0 - gamma1
        # M-step: closed-form for centred Normal mixture
        n1 = gamma1.sum()
        n2 = gamma2.sum()
        if n1 < 1.0 or n2 < 1.0:
            # one component collapsed; clamp to keep both alive
            n1 = max(n1, 1.0)
            n2 = max(n2, 1.0)
        w1_new = n1 / n
        w2_new = 1.0 - w1_new
        s1_new = float(np.sqrt(max((gamma1 * r * r).sum() / n1, 1e-12)))
        s2_new = float(np.sqrt(max((gamma2 * r * r).sum() / n2, 1e-12)))
        # ordered: keep s1 <= s2 (identifiability)
        if s1_new > s2_new:
            s1_new, s2_new = s2_new, s1_new
            w1_new, w2_new = w2_new, w1_new
        # log-likelihood for convergence
        ll = float(log_denom.sum())
        if abs(ll - prev_ll) < tol * max(abs(prev_ll), 1.0):
            w1, w2, s1, s2 = w1_new, w2_new, s1_new, s2_new
            break
        w1, w2, s1, s2 = w1_new, w2_new, s1_new, s2_new
        prev_ll = ll

    params = {
        "weights": (float(w1), float(w2)),
        "scales": (float(s1), float(s2)),
    }
    return C, params, resid


def mixture_3_gaussian_mle_fit(
    X: np.ndarray, Y: np.ndarray,
    *,
    max_em_iters: int = 200,
    tol: float = 1e-7,
    seed: int = 0,
) -> tuple[np.ndarray, dict, np.ndarray]:
    """MLE fit of ``Y[:, 0] = X @ C + r`` with ``r`` distributed as a
    symmetric centred mixture of three zero-mean Gaussians.

    Production-grade fitter for the distributional-class line of
    inquiry's Q2A node (notes/preregistrations/2026-05-26_distributional-class-thread/).

    Model: ``r ~ w1 * N(0, s1^2) + w2 * N(0, s2^2) + w3 * N(0, s3^2)``
    with ``w1 + w2 + w3 = 1`` and all components centred at 0. The
    three-component centred mixture buys finer tail control than mix-2
    (a "core / shoulder / tail" decomposition) at the cost of a richer
    EM landscape — initialisation by residual-quantile splits keeps the
    EM in a well-posed basin in our regime.

    Algorithm: OLS warm-start for the drift; EM on residuals (McLachlan &
    Peel 2000, §2.8 generalised to K=3). Plain numpy. Deterministic.

    Returns ``(C, params, resid)`` matching the
    :func:`student_t_mle_fit` calling convention:

    * ``params`` = ``{"weights": (w1, w2, w3), "scales": (s1, s2, s3)}``,
      ordered ``s1 <= s2 <= s3``.
    """
    del seed
    K = 3
    n, d = X.shape
    if Y.ndim == 1:
        y = Y.astype(float)
    else:
        y = Y[:, 0].astype(float)
    X = X.astype(float)

    C_col, *_ = np.linalg.lstsq(X, y, rcond=None)
    C = C_col.reshape(d, 1)
    resid = y - (X @ C).ravel()
    r = resid - resid.mean()
    abs_r = np.abs(r)
    q33, q66, q95 = np.quantile(abs_r, [0.33, 0.66, 0.95])
    # half-normal quantile factors (scale = q/factor): Phi^-1((1+p)/2)
    # for the |N| quantile. For p in (0.33, 0.66, 0.95):
    # Phi^-1(0.665)=0.4263; Phi^-1(0.83)=0.9542; Phi^-1(0.975)=1.9600
    s1 = max(q33 / 0.4263, 1e-6)
    s2 = max(q66 / 0.9542, s1 * 1.2)
    s3 = max(q95 / 1.9600, s2 * 1.5)
    w = np.array([0.6, 0.3, 0.1])
    scales = np.array([s1, s2, s3])
    log2pi = float(np.log(2 * np.pi))

    def _log_norm_pdf(x: np.ndarray, sigma: float) -> np.ndarray:
        return -0.5 * (x * x) / (sigma * sigma) - np.log(sigma) - 0.5 * log2pi

    prev_ll = -np.inf
    for _ in range(max_em_iters):
        # E-step: (n, K) log-responsibilities
        log_p = np.empty((n, K))
        for k in range(K):
            log_p[:, k] = np.log(max(w[k], 1e-300)) + _log_norm_pdf(r, scales[k])
        log_max = log_p.max(axis=1)
        log_denom = log_max + np.log(np.exp(log_p - log_max[:, None]).sum(axis=1))
        gamma = np.exp(log_p - log_denom[:, None])
        # M-step
        Nk = gamma.sum(axis=0)
        Nk = np.maximum(Nk, 1.0)        # guard against component collapse
        w_new = Nk / n
        w_new = w_new / w_new.sum()
        scales_new = np.sqrt(
            np.maximum((gamma * (r ** 2)[:, None]).sum(axis=0) / Nk, 1e-12)
        )
        # order by scale (identifiability)
        order = np.argsort(scales_new)
        w_new = w_new[order]
        scales_new = scales_new[order]
        ll = float(log_denom.sum())
        if abs(ll - prev_ll) < tol * max(abs(prev_ll), 1.0):
            w, scales = w_new, scales_new
            break
        w, scales = w_new, scales_new
        prev_ll = ll

    params = {
        "weights": tuple(float(x) for x in w),
        "scales": tuple(float(x) for x in scales),
    }
    return C, params, resid


def kde_residual_fit(
    X: np.ndarray, Y: np.ndarray,
    *,
    bandwidth: str | float | np.ndarray = "silverman",
) -> tuple[np.ndarray, dict, np.ndarray]:
    """OLS drift + non-parametric Gaussian-kernel KDE on the residual.

    Production-grade fitter for the distributional-class line of
    inquiry's Q2A node (notes/preregistrations/2026-05-26_distributional-class-thread/),
    representing the non-parametric end of the family ladder.

    Model: ``Y[:, 0] = X @ C + r`` where ``r`` has unspecified density
    estimated by Gaussian-kernel KDE on the OLS residual sample. The
    KDE density is

      f(t) = (1 / (n h)) sum_i K((t - r_i) / h),     K = standard Normal pdf

    with ``h`` set by Silverman's rule of thumb (Silverman 1986,
    *Density Estimation for Statistics and Data Analysis*, eq. 3.31)
    ``h = 0.9 * min(std(r), IQR(r) / 1.34) * n^{-1/5}``. Sampling from
    the estimated density uses the convolution rule: draw one of the
    stored samples uniformly and add ``h * N(0, 1)`` noise (this is
    exact: the KDE *is* the law of "uniform-pick + Gaussian-jitter
    with scale h").

    Plain numpy (no ``scipy.stats.gaussian_kde``) so the KDE
    construction is fully under audit control.

    Returns ``(C, params, resid)`` matching the
    :func:`student_t_mle_fit` calling convention:

    * ``params`` = ``{"bandwidth": float, "samples": np.ndarray,
      "normalization": float}`` — the (h, r_samples, 1/(n*h)) triple
      that downstream code needs to evaluate pdf, draw from the law,
      or roundtrip to a downstream evaluator.
    """
    n, d = X.shape
    if Y.ndim == 1:
        y = Y.astype(float)
    else:
        y = Y[:, 0].astype(float)
    X = X.astype(float)

    C_col, *_ = np.linalg.lstsq(X, y, rcond=None)
    C = C_col.reshape(d, 1)
    resid = y - (X @ C).ravel()
    r = resid - resid.mean()

    if isinstance(bandwidth, str):
        if bandwidth != "silverman":
            raise ValueError(
                f"unknown bandwidth rule {bandwidth!r}; use 'silverman' "
                f"or a numeric value"
            )
        std = float(np.std(r, ddof=1)) if n > 1 else 1.0
        q25, q75 = np.quantile(r, [0.25, 0.75])
        iqr = float(q75 - q25)
        spread = min(std, iqr / 1.34) if iqr > 0 else std
        h = 0.9 * max(spread, 1e-9) * (n ** (-1.0 / 5.0))
    elif isinstance(bandwidth, np.ndarray):
        h = float(np.asarray(bandwidth).item())
    else:
        h = float(bandwidth)
    h = max(h, 1e-9)

    params = {
        "bandwidth": float(h),
        "samples": r.astype(float),
        "normalization": float(1.0 / (n * h)),
    }
    return C, params, resid


# Cramér–von Mises 95% asymptotic quantile (Patra & Sen 2016, line above
# their Theorem 6: "the asymptotic 95% quantile of G_n is 0.6792, and is
# used in our data analysis"). This is THE paper's recommended value for
# β = 0.05, distribution-free under the F-continuous assumption, exact
# when α_0 = 0. Hardcoded with the paper page reference so the constant
# can't silently drift.
_PATRA_SEN_CVM_Q95 = 0.6792


def patra_sen_fit(
    residuals: np.ndarray,
    *,
    F_b: str = "gaussian",
    confidence: float = 0.95,
    n_grid: int = 401,
) -> dict:
    """Patra–Sen (2016) two-component mixture-fraction diagnostic.

    Estimates ``α_0`` and the honest lower confidence bound ``α̂_L`` for
    the mixture ``F = α F_s + (1 − α) F_b``, where ``F_b`` is known
    (standard normal here) and ``F_s`` is an unknown "signal" / "non-null"
    distribution. The identifiability convention is the paper's eq. (4):
    ``α_0`` is the SMALLEST ``α`` for which ``(F − (1−α)F_b)/α`` is a
    valid CDF — so ``α_0 = 0`` exactly when ``F = F_b`` (no non-Gaussian
    contamination).

    Production-grade fitter for the resolution-paths line of inquiry's
    P1 node ("patra-sen-per-stratum-localization";
    notes/preregistrations/2026-05-27_p1-patra-sen-per-stratum-localization/).
    Without it, the per-stratum α̂_L would be reimplemented inline in
    the P1 script and trigger /audit code-path REIMPLEMENTED.

    Reference: Patra, R. K. & Sen, B. (2016), "Estimation of a
    Two-Component Mixture Model with Applications to Multiple Testing",
    *JRSS-B* **78**(4); arXiv:1204.5488.

    Algorithm
    ---------
    For each ``γ`` on a uniform grid of ``[0, 1]``:

    1. Compute the naive "signal" CDF at the sample points (eq. 2)::

         F̂_s^γ(X_i) = (F_n(X_i) − (1 − γ) F_b(X_i)) / γ

    2. Project onto the cone of non-decreasing functions clipped to
       [0, 1] via PAVA — this is ``F̌_s^γ`` from eq. (3) / Lemma 1.

    3. Compute the criterion (eq. 7, eq. 8)::

         T(γ) = √n · γ · d_n(F̂_s^γ, F̌_s^γ)
              = √n · d_n(F_n, γ F̌_s^γ + (1 − γ) F_b)

       where ``d_n`` is the ``L^2(F_n)`` distance, i.e. the root-mean-
       square deviation evaluated at the n sample points.

    Then by Theorem 5 (with ``c_n`` taken as the (1−β) Cramér–von Mises
    quantile, asymptotically equivalent to the exact ``H_n^{-1}(1−β)``
    per Theorem 6 for n ≳ 500):

      α̂_L^{1-β} = inf {γ ∈ (0, 1] : T(γ) ≤ c_n}.

    For 95% (β = 0.05), the paper's recommended ``c_n = 0.6792`` is
    used (line preceding Theorem 6). For arbitrary β, the asymptotic
    Cramér–von Mises quantile is looked up via scipy.stats.cramervonmises.

    Identifiability and boundary behaviour
    --------------------------------------
    By Lemma 7 the set ``{γ : T(γ) ≤ c_n}`` is convex and contains 1,
    so this inf is well-defined; if T(1) > c_n (extremely rare; F_n
    farther from any one-component law than the CvM 95% bound), we
    return ``α̂_L = 1.0`` as a degenerate upper-pin. If the inf is
    not found because T(γ) > c_n on the whole grid, that's the same
    "α_0 = 1" case.

    Theorem 5 guarantees ``P(α̂_L = 0) = 1 − β`` when α_0 = 0, so on
    pure-F_b data the diagnostic returns exactly 0 with probability
    95%, not a positive-biased "near-zero" — this is THE property
    that distinguishes Patra–Sen from naive plug-in estimators.

    Parameters
    ----------
    residuals
        1-D array of i.i.d. samples from the mixture F. For our use
        case these are STANDARDIZED residuals ``u_i = (z − z_iter) /
        s_iter`` from a fitted Student-t kernel predictor (M1 from Q1),
        so ``F_b = N(0, 1)`` is the right background under the null of
        "M1 is correctly calibrated."
    F_b
        Background CDF. Currently only ``"gaussian"`` (standard normal)
        is implemented; a user-supplied callable could be plugged in
        if a different known background is needed.
    confidence
        Lower-bound confidence level (1 − β). Default 0.95.
    n_grid
        Number of γ values to sweep on [0, 1]. The grid is uniform.
        401 gives a 0.0025 resolution on α̂_L — finer than the
        ranges (R-A: 0.20 / R-C: 0.10) we evaluate against.

    Returns
    -------
    dict with keys:
        alpha_L : float
            Honest lower confidence bound on α_0 (Theorem 5).
        alpha_hat_heuristic : float
            The tuning-parameter-free "elbow" estimator (§5, α̃_0) —
            the γ at which the second numerical derivative of T(γ)/√n
            is maximised. Reported for diagnostic plotting; α̂_L is
            the load-bearing quantity.
        n : int
            Sample size.
        confidence : float
            Confidence level used.
        c_n : float
            The Cramér–von Mises quantile used as the threshold.
    """
    _ALLOWED_FB = {"gaussian", "uniform"}
    if F_b not in _ALLOWED_FB:
        raise NotImplementedError(
            f"F_b={F_b!r}: only {sorted(_ALLOWED_FB)} are implemented. "
            "Add a branch below if a different known background is "
            "needed."
        )

    r = np.asarray(residuals, dtype=float).ravel()
    r = r[np.isfinite(r)]
    n = r.size
    if n < 20:
        raise ValueError(
            f"patra_sen_fit needs at least 20 finite samples; got {n}. "
            "Stratum is too small for a meaningful α̂_L."
        )
    if F_b == "uniform":
        # Uniform[0,1] is the natural F_b when the input is a PIT — by
        # Patra & Sen 2016 Theorem 1 (monotone invariance of α_0), the
        # mixture-fraction estimator on u_PIT against Uniform[0,1] is
        # equal to the estimator on u_std against Student-t(ν) with the
        # same ν used in the PIT transform. Reject out-of-range inputs:
        # u_PIT must lie in [0, 1].
        if r.min() < 0.0 - 1e-9 or r.max() > 1.0 + 1e-9:
            raise ValueError(
                f"F_b='uniform' expects PIT values in [0, 1]; got "
                f"min={r.min():.4f}, max={r.max():.4f}. Pass standardized "
                "residuals with F_b='gaussian' instead, or the PIT scale "
                "with F_b='uniform'."
            )
        # Numerical clamp into [0, 1] to handle FP noise at the boundary.
        r = np.clip(r, 0.0, 1.0)

    # Threshold c_n for the confidence level. 95% is the paper's
    # explicit recommendation (line preceding Theorem 6: 0.6792). For
    # arbitrary β, defer to scipy's asymptotic CvM CDF.
    beta = 1.0 - confidence
    if abs(beta - 0.05) < 1e-12:
        c_n = _PATRA_SEN_CVM_Q95
    else:
        # Asymptotic Cramér–von Mises CDF: scipy doesn't expose the
        # one-sample-CvM-statistic-distribution quantile function
        # directly, so we invert via a Brent root-find on the survival
        # function of the cramervonmises statistic over a large
        # synthetic-uniform sample (Theorem 6 says this is the right
        # limit; n_ref = 5000 is comfortably "moderately large").
        from scipy.stats import cramervonmises  # noqa: PLC0415
        from scipy.optimize import brentq       # noqa: PLC0415

        rng_ref = np.random.default_rng(0)
        n_ref = 5000
        n_mc = 2000
        Ts = np.empty(n_mc)
        for k in range(n_mc):
            u = rng_ref.uniform(size=n_ref)
            # statistic n * omega^2; we want n * omega^2's quantile
            # at (1-β), so this gives c_n directly.
            Ts[k] = cramervonmises(u, "uniform").statistic
        # n * omega^2 -> sqrt of that approximates √n * d(F_n, F)
        # (eq. before Theorem 6); √(T) gives the c_n on the right scale.
        c_n = float(np.quantile(np.sqrt(Ts), 1.0 - beta))

    # Step 1: empirical CDF at the (sorted) sample points, and F_b at
    # the same points. F_n(X_(i)) = i/n for i = 1..n (right-continuous).
    r_sorted = np.sort(r)
    Fn_sorted = np.arange(1, n + 1, dtype=float) / n
    # Background CDF F_b at the sorted sample points. For 'uniform' the
    # CDF on [0, 1] is the identity F_b(x) = x; for 'gaussian' it's the
    # standard normal CDF Φ(x).
    if F_b == "uniform":
        Fb_sorted = r_sorted.copy()
    else:  # F_b == "gaussian"
        from scipy.stats import norm as _norm  # noqa: PLC0415
        Fb_sorted = _norm.cdf(r_sorted)

    # Grid of γ on [0, 1]. We include γ = 0 explicitly because the
    # paper's eq. (9) gives the well-defined limit
    #     lim_{γ→0+} γ · d_n(F̂_s^γ, F̌_s^γ) = d_n(F_n, F_b),
    # and Theorem 5's "P(α̂_L = 0) = 1 − β when α_0 = 0" property
    # requires the inf to be ACHIEVABLE at γ = 0 when the data are
    # pure F_b — otherwise the smallest grid value (e.g. 1/n_grid)
    # becomes a hard lower floor. We ALSO include γ = 1 because there
    # F̂_s^1 = F_n is already a CDF, so T(1) ≈ 0 — guaranteeing the
    # inf set is non-empty (Lemma 7 then says the set is convex and
    # equals [α̂_L, 1]).
    gamma_grid = np.linspace(0.0, 1.0, n_grid)

    # scipy.optimize.isotonic_regression is the O(n) PAVA (scipy ≥ 1.12);
    # used here for the L^2 monotone-increasing projection of eq. (3).
    # Equivalent to sklearn's IsotonicRegression but keeps us inside the
    # "plain scipy/numpy, no sklearn" discipline that the Q2A fitters
    # (mixture_2/3_gaussian_mle_fit, kde_residual_fit) all follow.
    from scipy.optimize import isotonic_regression  # noqa: PLC0415

    T = np.empty(n_grid)
    # T(0) = √n · d_n(F_n, F_b) per eq. (9). At γ = 0 the naive
    # F̂_s^γ is undefined (divide-by-zero), so we use the paper's
    # explicit limit instead of running the PAVA branch with γ = 0.
    T[0] = np.sqrt(n) * float(np.sqrt(np.mean((Fn_sorted - Fb_sorted) ** 2)))
    for i in range(1, n_grid):
        gamma = gamma_grid[i]
        # Naive F̂_s^γ at sample points (eq. 2).
        F_hat_s = (Fn_sorted - (1.0 - gamma) * Fb_sorted) / gamma
        # PAVA: monotone non-decreasing projection. Then clip to [0, 1]
        # per Lemma 1 (F̌_s^γ = min(max(F̃_s^γ, 0), 1)).
        F_check_s = isotonic_regression(F_hat_s, increasing=True).x
        F_check_s = np.clip(F_check_s, 0.0, 1.0)
        # L^2(F_n) distance = sqrt(mean of squared diffs at sample pts).
        # d_n(F̂_s^γ, F̌_s^γ).
        d_n = float(np.sqrt(np.mean((F_hat_s - F_check_s) ** 2)))
        T[i] = np.sqrt(n) * gamma * d_n

    # α̂_L = inf{γ : T(γ) ≤ c_n}. By Lemma 7 the set {γ : T(γ) ≤ c_n}
    # is convex = [α̂_L, 1], so the inf is the FIRST grid point at
    # which T drops below c_n. T is non-increasing in γ for γ ≥ α_0
    # (Lemma 6 / 9) — strictly decreasing as γ decreases past α_0.
    below = T <= c_n
    if not np.any(below):
        # T(1) > c_n: the whole grid is above the threshold. By Lemma 7
        # this means α̂_L = 1 (degenerate one-component fit at the
        # signal end). Return 1.0 with a small grid-resolution caveat.
        alpha_L = 1.0
    else:
        alpha_L = float(gamma_grid[np.argmax(below)])

    # §5 heuristic estimator α̃_0: γ at which the second numerical
    # derivative of γ·d_n(F̂_s^γ, F̌_s^γ) = T(γ)/√n is maximised. Reported
    # for completeness; α̂_L is the load-bearing quantity for our verdict
    # architecture, and the paper itself warns (page 9, end of §5) that
    # α̃_0 can fail to estimate the elbow consistently in some settings.
    d2T = np.diff(T, n=2)
    # The second-difference index i corresponds to gamma_grid[i + 1].
    alpha_hat_heuristic = float(gamma_grid[int(np.argmax(d2T)) + 1])

    return {
        "alpha_L": alpha_L,
        "alpha_hat_heuristic": alpha_hat_heuristic,
        "n": int(n),
        "confidence": float(confidence),
        "c_n": float(c_n),
    }


# Default slice count for SIR (Li 1991 §4 recommendation; chosen
# pre-data in phase_a and not tunable post-data). 10 is the
# convention for n in the low-1e4 range with d_X ~ 10.
_SIR_DEFAULT_N_SLICES = 10

# Ridge stabilizer on Σ_x for the generalized eigenproblem. Small
# multiple of the average diagonal — preserves the basis when Σ_x
# is well-conditioned (post-standardization Σ_x ≈ correlation matrix
# of X, ~unit eigenvalues), kicks in only when a column collapses.
_SIR_SIGMA_X_RIDGE = 1e-8


def sliced_inverse_regression_fit(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    n_slices: int = _SIR_DEFAULT_N_SLICES,
) -> dict:
    """Sliced Inverse Regression (Li 1991) for the central subspace.

    Estimates an ordered set of dimension-reducing directions
    ``β_1, …, β_{d_X}`` such that the conditional law of ``Y`` given
    ``X`` depends on ``X`` only through projections ``β_k^T X``.
    The top ``β_1`` is the direction maximizing the slice-mean variance
    of ``X`` against the slices of ``Y``.

    Production-grade fitter for the resolution-paths line of inquiry's
    Q1A' node ("q1a-prime-sdr-ica-sibling";
    notes/preregistrations/2026-05-27_q1a-prime-sdr-ica-sibling/).
    Without it, SIR would be reimplemented inline in the Q1A' script
    and trigger /audit code-path REIMPLEMENTED.

    Reference: Li, K.-C. (1991), "Sliced Inverse Regression for
    Dimension Reduction", *JASA* **86**(414), 316–327.

    Algorithm (eq. (3.3)–(3.6) of Li 1991)
    --------------------------------------
    1. Center: ``X̃ = X − X̄``.
    2. Sort rows by ``Y``; partition into ``H = n_slices`` slices of
       equal count ``p_h = n_h / n``.
    3. Slice means ``m_h = mean of X̃ over slice h``; weighted between-
       slice covariance ``Σ_b = Σ_h p_h m_h m_h^T``.
    4. Solve the generalized eigenproblem
       ``Σ_b β = λ Σ_x β``   (with ``Σ_x = cov(X)``)
       via ``scipy.linalg.eigh(Σ_b, Σ_x_ridge)``.

    The generalized form is necessary, NOT a plain eigendecomposition
    of ``Σ_b``: with z-scored inputs ``Σ_x`` is the correlation matrix
    of ``X`` (cyclic columns and lag-1/lag-24 demand correlate by
    construction); ``eigh(Σ_b)`` would land on a different basis whose
    projections are not the central-subspace directions.

    A small ridge ``_SIR_SIGMA_X_RIDGE * mean(diag(Σ_x)) * I`` is added
    to ``Σ_x`` for numerical stability — preserves the basis when
    ``Σ_x`` is well-conditioned, kicks in only against rank deficiency.

    Eigenvalues / directions are returned in DESCENDING order of
    eigenvalue. ``directions`` is in the SAME basis as the input ``X``
    (i.e. ``scores = X @ directions[:, 0]`` gives the top-component
    projection of each row); the directions are NOT orthonormal under
    the Euclidean inner product, but ARE ``Σ_x``-orthonormal
    (``β_j^T Σ_x β_k = δ_{jk}``), which is the natural inner product
    for SIR.

    Parameters
    ----------
    X
        ``(n, d_X)`` design matrix. The caller is responsible for
        column-wise standardization; we centre internally for
        ``Σ_b`` but do NOT re-standardize so that the returned
        directions live in the basis the caller passed in.
    Y
        ``(n,)`` response vector. Real-valued (continuous or
        ordinal). SIR is invariant to monotone transforms of ``Y``
        — only the slice-membership ordering matters.
    n_slices
        ``H`` in Li 1991. Default 10 (the SIR convention for
        moderate ``n`` and ``d_X ~ 10``).

    Returns
    -------
    dict with keys:
        directions : ndarray of shape ``(d_X, d_X)``
            Columns are the SIR directions in the basis of ``X``,
            sorted by descending eigenvalue (``directions[:, 0]``
            is the top component).
        eigenvalues : ndarray of shape ``(d_X,)``
            Generalized eigenvalues, sorted descending.
        slice_means : ndarray of shape ``(H, d_X)``
            ``m_h`` for each slice (centred). Diagnostic-only.
        slice_counts : ndarray of shape ``(H,)``
            ``n_h`` for each slice. Diagnostic-only.
        n : int
            Sample size.
        n_slices : int
            ``H``.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float).ravel()
    if X.ndim != 2:
        raise ValueError(f"sliced_inverse_regression_fit: X must be 2-D; got shape {X.shape}")
    n, d_X = X.shape
    if Y.shape != (n,):
        raise ValueError(
            f"sliced_inverse_regression_fit: Y shape {Y.shape} does not match X rows {n}"
        )
    if n_slices < 2:
        raise ValueError(f"sliced_inverse_regression_fit: n_slices >= 2 required; got {n_slices}")
    if n < n_slices * 2:
        raise ValueError(
            f"sliced_inverse_regression_fit: n={n} too small for {n_slices} slices "
            "(need n >= 2 H); halve n_slices or pool data."
        )
    if not (np.all(np.isfinite(X)) and np.all(np.isfinite(Y))):
        raise ValueError(
            "sliced_inverse_regression_fit: X or Y contains non-finite "
            "entries. Drop or impute before calling."
        )

    # Centre X for Σ_b; Σ_x computed on the un-centred input (np.cov
    # subtracts the mean internally).
    X_mean = X.mean(axis=0)
    X_c = X - X_mean

    # Σ_x over all of X. rowvar=False so np.cov treats columns as variables.
    Sigma_x = np.cov(X_c, rowvar=False)
    # Ridge for numerical stability; tiny when Σ_x is well-conditioned.
    ridge = _SIR_SIGMA_X_RIDGE * float(np.mean(np.diag(Sigma_x)))
    Sigma_x_ridge = Sigma_x + ridge * np.eye(d_X)

    # Slice by sorted Y; equal-count partition. Use np.argsort + np.array_split
    # so the slice count is honoured even when n is not divisible by n_slices.
    order = np.argsort(Y, kind="stable")
    slice_idx_lists = np.array_split(order, n_slices)

    slice_means = np.zeros((n_slices, d_X))
    slice_counts = np.zeros(n_slices, dtype=np.int64)
    Sigma_b = np.zeros((d_X, d_X))
    for h, idx_h in enumerate(slice_idx_lists):
        n_h = len(idx_h)
        if n_h == 0:
            continue
        m_h = X_c[idx_h].mean(axis=0)
        slice_means[h] = m_h
        slice_counts[h] = n_h
        p_h = n_h / n
        Sigma_b += p_h * np.outer(m_h, m_h)

    # Generalized eigenproblem: Σ_b β = λ Σ_x β. scipy.linalg.eigh
    # solves the symmetric generalized form and returns eigenvalues in
    # ASCENDING order; flip to descending so the top component is index 0.
    from scipy.linalg import eigh  # noqa: PLC0415

    eigvals, eigvecs = eigh(Sigma_b, Sigma_x_ridge)
    # Reverse to descending; .copy() so the returned arrays are contiguous.
    order_desc = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order_desc].copy()
    directions = eigvecs[:, order_desc].copy()

    return {
        "directions": directions,
        "eigenvalues": eigvals,
        "slice_means": slice_means,
        "slice_counts": slice_counts,
        "n": int(n),
        "n_slices": int(n_slices),
    }


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
