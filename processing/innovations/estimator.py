"""Local Gaussian predictive-semigroup estimator.

For each anchor time the estimator fits a local linear drift ``C_j`` (the
edynamics WLS map, which is correct) and a local diffusion ``Sigma_j``
recomputed from *raw* residuals (the library's own residual covariance is the
covariance of the kernel-weighted residual ``Wy - WX@C`` ~ ``w^2 * Q`` and
collapses to ~1e-9 under wide kernels -- verified by the VAR(1) gate, see
``validation/synthetic.py``).

Together ``(C_j, Sigma_j)`` parameterise the per-anchor Gaussian kernel
``x' | x ~ N(x @ C_j, Sigma_j)`` -- an empirical estimator of the programme's
``Pi_Delta`` predictive-semigroup kernel. No novelty/operator-naming claims
are made here; terminology defers to the Resolvent_Framework programme.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from edynamics.modelling_tools import Embedding
from edynamics.modelling_tools.estimators import LocalGLSelector
from edynamics.modelling_tools.kernels import Gaussian
from edynamics.modelling_tools.projectors import WeightedLeastSquares


@dataclass
class SemigroupEstimate:
    """Per-anchor local Gaussian semigroup parameters.

    Shapes (``N`` anchors, ``d`` embedding dimension):
        coefficients : (N, d, d)  -- drift ``C_j`` (``x_next = x @ C_j``)
        covariances  : (N, d, d)  -- diffusion ``Sigma_j`` (SPD)
        resid_means  : (N, d)     -- residual mean ``mu_j``
        eigvals      : (N, d)     -- ascending eigenvalues of ``Sigma_j``
        theta_star   : (N,)       -- selected drift bandwidth per anchor
        sigma_star   : (N,)       -- selected diffusion bandwidth per anchor
        anchor_times : (N,) int64 -- nanosecond anchor timestamps
    """

    coefficients: np.ndarray
    covariances: np.ndarray
    resid_means: np.ndarray
    eigvals: np.ndarray
    theta_star: np.ndarray
    sigma_star: np.ndarray
    anchor_times: np.ndarray


def raw_residual_diffusion(
    *,
    embedding: Embedding,
    anchor: pd.Timestamp,
    C: np.ndarray,
    residual_kernel,
) -> tuple[np.ndarray, np.ndarray]:
    """Diffusion ``Sigma`` and residual mean ``mu`` from *raw* residuals.

    Uses the (correct) library-recovered drift ``C`` but recomputes the
    residual covariance from unweighted residuals ``Y - X@C`` with only the
    residual kernel applied -- avoiding the ``w^2`` collapse in the library's
    ``_local_stats_from_weighted``. Leave-one-out (drop the anchor row)
    mirrors ``WeightedLeastSquares.project(leave_out=True)``.
    """
    block = embedding.block
    blk = block.loc[block.index != anchor]
    X_full = blk.iloc[:-1].values
    Y_full = blk.iloc[1:].values
    resid = Y_full - X_full @ C
    g = residual_kernel.weigh(np.linalg.norm(resid, axis=1))
    mu = np.average(resid, axis=0, weights=g)
    rc = resid - mu[None, :]
    Sigma = (rc * g[:, None]).T @ rc / (g.sum() + 1e-12)
    return Sigma, mu


def build_local_gaussian_semigroup(
    *,
    embedding: Embedding,
    anchors: pd.DatetimeIndex,
    theta_grid: np.ndarray,
    sigma_grid: np.ndarray,
    gl_penalty_C: float = 2.0,
) -> SemigroupEstimate:
    """Fit per-anchor local Gaussian semigroup parameters.

    Drift comes from the edynamics WLS projector (reliable); diffusion is
    recomputed from raw residuals via :func:`raw_residual_diffusion`. The
    per-anchor ``(theta*, sigma*)`` come from the pointwise
    Goldenshluger-Lepski selector.
    """
    d = embedding.block.shape[1]
    anchors = pd.DatetimeIndex(anchors)

    wls = WeightedLeastSquares(
        kernel=Gaussian(theta=1.0, dim=d),
        residual_kernel=Gaussian(theta=1.0, dim=d),
    )
    selector = LocalGLSelector(theta_grid, sigma_grid, lwls=wls, C=gl_penalty_C)
    selector.fit(embedding, anchors)

    res = wls.project(
        embedding=embedding,
        points=embedding.get_points(anchors),
        steps=1,
        step_size=1,
        leave_out=True,
        return_coefficients=True,
        return_residual_stats=True,
        use_innovations=False,
    )

    C_all = res.coefficients.cpu().numpy()[:, 0]      # (N, d, d) -- correct
    theta_star = selector.theta_star.cpu().numpy()    # (N,)
    sigma_star = selector.sigma_star.cpu().numpy()    # (N,)

    N = len(anchors)
    Sigma_all = np.empty((N, d, d), dtype=float)
    mu_all = np.empty((N, d), dtype=float)
    eig_all = np.empty((N, d), dtype=float)

    for i, (anchor_t, sig) in enumerate(zip(anchors, sigma_star)):
        wls.residual_kernel.theta = float(sig)
        Sigma, mu = raw_residual_diffusion(
            embedding=embedding,
            anchor=anchor_t,
            C=C_all[i],
            residual_kernel=wls.residual_kernel,
        )
        Sigma_all[i] = Sigma
        mu_all[i] = mu
        eig_all[i] = np.linalg.eigvalsh(Sigma)  # ascending, symmetric

    return SemigroupEstimate(
        coefficients=C_all,
        covariances=Sigma_all,
        resid_means=mu_all,
        eigvals=eig_all,
        theta_star=theta_star,
        sigma_star=sigma_star,
        anchor_times=anchors.asi8.astype(np.int64),
    )
