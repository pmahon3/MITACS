"""VAR(1) synthetic ground-truth recovery test.

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
from processing.innovations.estimator import build_local_gaussian_semigroup


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


# Seed manifest for the C4 provenance artifact: every RNG seed that
# determines this gate's numbers. Hardcoded here AND in the calls below
# (single source would obscure the gate logic); kept in lockstep.
GATE_SEEDS = {
    "non_gaussianity_var1_params": 11,
    "non_gaussianity_simulate": 12,
    "non_gaussianity_anchors": 1,
    "non_gaussianity_student_t_df": 5,
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
    return "\n".join(out) + "\n", bool(ok and ng_ok and ms_ok)


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
