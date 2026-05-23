"""SMC VALIDATION GATE: does forward-SMC recover the closed-form
VAR(1) h-step predictive DISTRIBUTION (mean + covariance)?

The existing multistep-composition gate (processing/innovations/validation/
synthetic.py:multistep_recovery) validates iterated POINT estimates of
A^h and accumulated diffusion via the production estimator's composed
(C, Sigma). This gate is its distributional analogue: an SMC ensemble
of M particles must, at every horizon h, give an ensemble mean and
covariance that match the closed-form VAR(1) (A^h x_0, accumulated Q)
within tolerance.

The recovery gate spirit (memory mitacs-design-decisions): no Ontario
SMC number is trusted until this gate passes. Tests both SMC variants:
  - 'gaussian'  : should be tight (correctly-specified)
  - 'empirical' : should also recover (Gaussian innovations -> empirical
                  draws are i.i.d. from the same Gaussian; modulo finite
                  M Monte Carlo noise)

PROVENANCE-GRADE: INSPECTION-ONLY (scratch gate; the production gate
is processing/innovations/validation/synthetic.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from processing.innovations.validation.synthetic import (
    _true_multistep,
    build_embedding,
    make_var1_params,
    simulate_var1,
)
from processing.innovations.estimator import build_local_gaussian_semigroup

from scratch.smc import ensemble_summary, smc_trajectory


# Same horizon set as the production composition gate.
HORIZONS = [1, 2, 4, 8, 12, 24]

# Tolerances. The signal-to-noise here is hard: A's spectral radius is
# 0.5, so |mean_true| decays as 0.5^h (1e-7 at h=24) while |cov_true|
# is O(10) constant. Dividing by |mean_true| is meaningless once it has
# decayed below the noise floor; the right metric is "is the ensemble
# mean within Monte Carlo error of the truth, where MC error is
# sigma_MC = ||L||/sqrt(M)". `K_MEAN` is the number of MC-sigmas
# allowed; 5 is the conventional "consistent with noise" threshold,
# widened modestly with H to absorb (C_hat, Sigma_hat) estimator
# compounding. Covariance uses a relative tolerance (|cov_true| stays
# O(1) so dividing is well-conditioned), same shape as the production
# multistep gate.
K_MEAN = {1: 6, 2: 6, 4: 6, 8: 7, 12: 8, 24: 10}    # MC-sigma multiples
COV_TOL = {1: 0.30, 2: 0.35, 4: 0.45, 8: 0.60, 12: 0.75, 24: 1.15}


def _smc_recovery(d: int, variant: str, M: int, seed: int):
    """Run SMC on a fitted VAR(1); compare ensemble (mean, cov) at every
    horizon to the closed-form truth. Returns per-horizon rel-errors."""
    A, Q = make_var1_params(d, seed=seed)
    X = simulate_var1(A, Q, n=4000, burn=500, seed=seed + 1)
    embedding, idx = build_embedding(X)

    # one fit (production path) using the median over many anchors --
    # the SAME aggregation the production composition gate uses.
    rng = np.random.default_rng(seed + 2)
    interior = idx[d + 1: -2]
    sel = np.sort(rng.choice(len(interior), 60, replace=False))
    est = build_local_gaussian_semigroup(
        embedding=embedding, anchors=pd.DatetimeIndex(interior[sel])
    )
    C1 = np.median(est.coefficients, axis=0)       # ~ A.T  (row convention)
    S1 = np.median(est.covariances, axis=0)        # ~ Q

    # residual array for empirical sampling: derive once from the fit
    # at one anchor (the empirical conditional distribution; for VAR(1)
    # Gaussian noise this is itself ~Gaussian, so 'empirical' should
    # also recover -- a sanity gate on the empirical sampler itself).
    X_arr = embedding.block.iloc[:-1].values
    Y_arr = embedding.block.iloc[1:].values
    resid = Y_arr - X_arr @ C1                     # (n, d)

    # SMC trajectories from a fixed query state. Use a representative
    # interior state (median library row) as x0.
    x0 = np.median(X_arr, axis=0)

    # fit_at returns (C, Sigma, resid) -- here state-INDEPENDENT
    # (VAR(1) is globally linear) so we can return the same triple.
    fit_at = lambda _x: (C1, S1, resid)

    H_max = max(HORIZONS)
    rng_smc = np.random.default_rng(seed + 3)
    traj = smc_trajectory(x0, H_max, fit_at, variant, M, rng_smc)
    summ = ensemble_summary(traj)

    rows = []
    for H in HORIZONS:
        Ah_t, S_true = _true_multistep(A, Q, H)
        mean_true = x0 @ Ah_t.T                    # row convention: x_0 @ A^h.T
        cov_true = S_true

        h_idx = H - 1                              # 0-indexed
        mean_smc = summ["mean"][h_idx]
        cov_smc = summ["cov"][h_idx]

        # MC-sigma metric for the mean: how many ensemble-mean sigmas
        # away is the SMC mean from the truth? sigma_MC = sqrt(tr(Sigma_h)/M)
        # is the standard error of the ensemble mean's magnitude.
        sigma_mc = float(np.sqrt(np.trace(cov_true) / M))
        me_sigmas = float(np.linalg.norm(mean_smc - mean_true) / sigma_mc)
        ce = float(np.linalg.norm(cov_smc - cov_true, "fro")
                   / np.linalg.norm(cov_true, "fro"))
        rows.append((H, me_sigmas, ce))
    return rows


def main():
    print("=" * 70)
    print("SMC VALIDATION GATE (INSPECTION-ONLY) -- VAR(1) closed-form recovery")
    print("=" * 70)
    for variant in ("gaussian", "empirical"):
        print(f"\nVariant: {variant}  (M=400, d=3, seed=7)")
        rows = _smc_recovery(d=3, variant=variant, M=400, seed=7)
        print(f"  {'H':>3} {'mean_MCsig':>11} {'(tol_K)':>8} "
              f"{'cov_relerr':>12} {'(tol)':>7} {'verdict':>8}")
        all_pass = True
        for H, me, ce in rows:
            ok_mean = me < K_MEAN[H]
            ok_cov = ce < COV_TOL[H]
            ok = ok_mean and ok_cov
            all_pass = all_pass and ok
            print(f"  {H:>3} {me:>11.2f} {K_MEAN[H]:>8.0f} "
                  f"{ce:>12.4f} {COV_TOL[H]:>7.2f} "
                  f"{'PASS' if ok else 'FAIL':>8}")
        print(f"  variant {variant}: {'PASS' if all_pass else 'FAIL'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
