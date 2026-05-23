"""EXPLORATORY: forward-SMC multi-step predictor for the local-Gaussian
estimator.

The production multi-step (experiment.predict_multistep) propagates the
conditional MEAN (point trajectory) -- it iterates `x @ C` and discards
the diffusion. Forward-SMC instead propagates the kernel: at each step
sample `x_next ~ N(x @ C, Sigma)` (or empirical-residual draw), iterate
M independent particles, and report the ensemble.

Two variants -- the Liu-Gao discriminator (memory mitacs-liu-gao-prediction):
  - 'gaussian'  : draw next-state innovation from N(0, Sigma)
  - 'empirical' : resample a real residual row from the local fit

Rank-1 structure (memory mitacs-rank1-structural): for the single-variable
delay embedding used on Ontario, coords 1..d-1 of the one-step image are
deterministic lag-shifts of the current state -- only coord 0 carries
stochastic content. Detected automatically from the rank of Sigma; the
SMC step uses the lag-shift specialisation when rank-1, and full
multivariate Gaussian / empirical-row sampling when not (the VAR(d)
validation gate is multivariate).

PROVENANCE-GRADE: INSPECTION-ONLY.
"""
from __future__ import annotations

import numpy as np


def _is_rank1(Sigma: np.ndarray, tol: float = 1e-6) -> bool:
    """True iff Sigma has exactly one numerically-non-zero eigenvalue.

    For the single-variable delay embedding the smaller eigenvalue is
    typically 1e-7..1e-15 relative to the larger (deterministic lag
    shifts in coords 1..d-1); for a true multivariate fit (e.g. VAR(d)
    with d>1 variables at tau=0) all eigenvalues are O(1).
    """
    eig = np.linalg.eigvalsh(Sigma)
    eig = np.sort(np.abs(eig))[::-1]              # descending
    if len(eig) < 2 or eig[0] <= 0:
        return False
    return (eig[1] / eig[0]) < tol


def _sample_next_state_rank1(
    x: np.ndarray, C: np.ndarray, Sigma: np.ndarray,
    resid0: np.ndarray, variant: str, rng: np.random.Generator,
) -> np.ndarray:
    """One-step SMC sample on a single-variable delay embedding.

    Coords 1..d-1 of the next state are exact lag shifts of x; only the
    scalar coord-0 innovation is sampled. resid0 is the per-anchor
    coord-0 residual array (shape (n,)) -- the empirical conditional
    innovation distribution.
    """
    d = len(x)
    z_pred = float(x @ C[:, 0])                    # drift prediction at coord 0
    if variant == "gaussian":
        sd = float(np.sqrt(max(Sigma[0, 0], 0.0)))
        r0 = float(rng.normal(0.0, sd))
    elif variant == "empirical":
        r0 = float(rng.choice(resid0))
    else:
        raise ValueError(f"unknown variant {variant!r}")
    # Lag-shift: x_next = [z_pred + r0, x_0, x_1, ..., x_{d-2}]
    x_next = np.empty(d)
    x_next[0] = z_pred + r0
    x_next[1:] = x[:-1]
    return x_next


def _sample_next_state_full(
    x: np.ndarray, C: np.ndarray, Sigma: np.ndarray,
    resid: np.ndarray, variant: str, rng: np.random.Generator,
) -> np.ndarray:
    """One-step SMC sample under full multivariate stochasticity.
    For the VAR(d) validation gate (no lag-shift structure)."""
    mean = x @ C
    if variant == "gaussian":
        try:
            L = np.linalg.cholesky(Sigma + 1e-12 * np.eye(Sigma.shape[0]))
        except np.linalg.LinAlgError:
            # symmetric eigendecomp fallback (Sigma SPSD by construction)
            w, V = np.linalg.eigh(Sigma)
            w = np.clip(w, 0.0, None)
            L = V @ np.diag(np.sqrt(w))
        r = L @ rng.standard_normal(Sigma.shape[0])
    elif variant == "empirical":
        # resample a residual row
        r = resid[rng.integers(len(resid))]
    else:
        raise ValueError(f"unknown variant {variant!r}")
    return mean + r


def step_sample(
    x: np.ndarray, C: np.ndarray, Sigma: np.ndarray,
    resid: np.ndarray, variant: str, rng: np.random.Generator,
) -> np.ndarray:
    """One SMC step -- dispatches on Sigma's rank.

    `resid` is the per-anchor residual array Y - X@C (shape (n, d)).
    The rank-1 path reads only resid[:, 0]; the full path reads rows.
    """
    if _is_rank1(Sigma):
        return _sample_next_state_rank1(
            x, C, Sigma, resid[:, 0], variant, rng)
    return _sample_next_state_full(x, C, Sigma, resid, variant, rng)


def smc_trajectory(
    x0: np.ndarray, H: int, fit_at, variant: str, M: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Run M independent SMC trajectories of length H from x0.

    `fit_at(x)` -> (C, Sigma, resid) at query state x. Called M*H times
    in general; the caller can short-circuit when (C, Sigma, resid) is
    state-INDEPENDENT (global-OLS) by passing a closure that ignores x.

    Returns: array of shape (M, H, d), each [m, h, :] is particle m at
    horizon h+1 (1-indexed: h=0 is the first step from x0).
    """
    d = len(x0)
    out = np.empty((M, H, d))
    for m in range(M):
        x = x0.copy()
        for h in range(H):
            C, Sigma, resid = fit_at(x)
            x = step_sample(x, C, Sigma, resid, variant, rng)
            out[m, h] = x
    return out


def ensemble_summary(traj: np.ndarray) -> dict:
    """Per-horizon ensemble mean, covariance, and quantiles.

    `traj`: (M, H, d) array. Returns dict with arrays indexed by horizon.
    """
    M, H, d = traj.shape
    mean = traj.mean(axis=0)                       # (H, d)
    cov = np.empty((H, d, d))
    for h in range(H):
        cov[h] = np.cov(traj[:, h, :], rowvar=False, ddof=1) if d > 1 \
            else np.array([[traj[:, h, 0].var(ddof=1)]])
    # 1-sigma-equivalent quantile band on coord 0 (the predicted variable)
    q_lo = np.quantile(traj[:, :, 0], 0.158655, axis=0)   # ~ -1 sigma
    q_hi = np.quantile(traj[:, :, 0], 0.841345, axis=0)   # ~ +1 sigma
    return {
        "mean": mean,
        "cov": cov,
        "q_lo": q_lo,
        "q_hi": q_hi,
        "M": M, "H": H, "d": d,
    }
