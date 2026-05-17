"""VAR(1) synthetic ground-truth recovery test.

Generates a stable VAR(1) process

    x_{t+1} = A x_t + eps_t,   eps_t ~ N(0, Q)

with a *known* drift ``A`` (spectral radius < 1) and a *known* SPD diffusion
``Q``, flows it through the exact call chain the rebuilt innovations stage
will use --

    Embedding -> LocalGLSelector.fit -> WeightedLeastSquares.project(
        return_coefficients=True, return_residual_stats=True,
        use_innovations=True)

-- and asserts that the recovered per-anchor drift and diffusion match the
ground truth.

Convention note
---------------
The projector computes ``C = lstsq(WX, Wy)`` and forecasts ``x_next = x @ C``
(row-vector / right-multiply convention). For the column-convention VAR(1)
``x_{t+1} = A x_t`` this means the recovered ``C`` estimates ``A.T``. The
checks below compare ``C_hat`` against ``A.T`` accordingly.

Diffusion note (verified 2026-05-17)
------------------------------------
The library's ``RoseResult.covariances`` is the covariance of the *weighted*
residual ``Wy - WX@C = w·(Y - XC)``, i.e. it scales as ``w²·Q`` and collapses
to ~1e-9 under wide kernels. It does **not** estimate ``Q``. Diffusion is
therefore recomputed here from **raw** residuals ``Y - X@C`` with only the
residual kernel applied -- the same recomputation the rebuilt
``estimator.py`` will use. The library drift (``RoseResult.coefficients``)
*is* correct and is reused.

Run standalone::

    python -m processing.innovations.validation.synthetic
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from edynamics.modelling_tools import Embedding, Lag
from edynamics.modelling_tools.estimators import LocalGLSelector
from edynamics.modelling_tools.kernels import Gaussian
from edynamics.modelling_tools.projectors import WeightedLeastSquares


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


def raw_residual_diffusion(
    *,
    embedding: Embedding,
    anchor: pd.Timestamp,
    C: np.ndarray,
    residual_kernel,
) -> tuple[np.ndarray, np.ndarray]:
    """Diffusion ``Σ`` and residual mean ``μ`` from *raw* residuals.

    Uses the (library-recovered, correct) drift ``C`` but recomputes the
    residual covariance from unweighted residuals ``Y - X@C`` with only the
    residual kernel applied -- avoiding the ``w²`` collapse in the library's
    own ``_local_stats_from_weighted``. This is the exact computation the
    rebuilt ``estimator.py`` will perform.
    """
    block = embedding.block
    blk = block.loc[block.index != anchor]  # leave-one-out, matching project()
    X_full = blk.iloc[:-1].values
    Y_full = blk.iloc[1:].values
    resid = Y_full - X_full @ C
    g = residual_kernel.weigh(np.linalg.norm(resid, axis=1))
    mu = np.average(resid, axis=0, weights=g)
    rc = resid - mu[None, :]
    Sigma = (rc * g[:, None]).T @ rc / (g.sum() + 1e-12)
    return Sigma, mu


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

    # Wide kernels: a broad drift bandwidth -> near-global linear fit (the
    # right limit for a *globally* linear VAR(1)); a broad residual bandwidth
    # -> the kernel-weighted residual covariance approaches the plain
    # covariance, i.e. Q.
    wls = WeightedLeastSquares(
        kernel=Gaussian(theta=1.0, dim=d),
        residual_kernel=Gaussian(theta=1.0, dim=d),
    )
    theta_grid = np.linspace(5.0, 50.0, 8)
    sigma_grid = np.linspace(5.0, 50.0, 8)
    selector = LocalGLSelector(theta_grid, sigma_grid, lwls=wls, C=2.0)
    selector.fit(embedding, anchors)

    res = wls.project(
        embedding=embedding,
        points=embedding.get_points(anchors),
        steps=1,
        step_size=1,
        leave_out=True,
        return_coefficients=True,
        return_residual_stats=True,
        use_innovations=True,
    )

    C_hat = res.coefficients.cpu().numpy()[:, 0]    # (N, d, d) -- correct
    sigma_star = selector.sigma_star.cpu().numpy()  # (N,)

    A_T = A.T
    eig_Q = np.sort(np.linalg.eigvalsh(Q))

    drift_errs, diff_errs, eig_errs = [], [], []
    for Ci, anchor_t, sig in zip(C_hat, anchors, sigma_star):
        # Diffusion recomputed from raw residuals (library Σ is w²-collapsed).
        wls.residual_kernel.theta = float(sig)
        Si, _ = raw_residual_diffusion(
            embedding=embedding,
            anchor=anchor_t,
            C=Ci,
            residual_kernel=wls.residual_kernel,
        )
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


if __name__ == "__main__":
    result, A, Q = recover()
    print("VAR(1) recovery:")
    print(f"  anchors evaluated  : {result.n_anchors}")
    print(f"  drift  rel err     : {result.drift_rel_err:.4f}  (tol {DRIFT_TOL})")
    print(f"  diff   rel err     : {result.diffusion_rel_err:.4f}  (tol {DIFFUSION_TOL})")
    print(f"  eig(Σ) rel err     : {result.eig_rel_err:.4f}  (tol {EIG_TOL})")
    ok = (
        result.n_anchors > 0
        and result.drift_rel_err < DRIFT_TOL
        and result.diffusion_rel_err < DIFFUSION_TOL
        and result.eig_rel_err < EIG_TOL
    )
    print("RESULT:", "PASS" if ok else "FAIL")
