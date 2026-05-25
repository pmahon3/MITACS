"""Chapman-Kolmogorov consistency gate for the local-Gaussian estimator.

THE QUESTION. The production estimator fits Pi_Delta (a single-step
local Gaussian kernel). Forward-SMC propagates the kernel by *iteration*
-- effectively asserting (Pi_Delta)^h is the right object at horizon h.
That assertion is the {Pi_t} semigroup property, and the estimator does
NOT verify it (memory ``mitacs-theory-correspondence`` qualifier 3:
"single-step only -- Chapman-Kolmogorov / the {Pi_t} semigroup is
neither constructed nor verified"). This gate tests it.

THE TEST.

  (A) Synthetic VAR(1) sanity gate (must PASS).
      With known ground-truth ``x_{t+1} = x_t A + eps, eps ~ N(0, Q)``,
      fit:
        - Pi_Delta directly                via ``_local_fit_at`` with shift=1
        - Pi_{h*Delta} directly            via ``_local_fit_at`` with shift=h
        - (Pi_Delta)^h closed-form         A_iter = A^h,  Q_iter = sum_{k=0}^{h-1} (A^T)^k Q A^k
      For a true VAR(1) generative process the three drifts agree
      (a unique linear best fit), and the diffusions satisfy
      ``Q_h_direct ~= Q_iter`` within sampling noise.

  (B) Ontario in-sample probe (DIAGNOSTIC).
      On the live embedding at the *training-cutoff* state (the same
      cutoff the registered forecast uses) compare:
        - C_h_direct = ``_local_fit_at`` on (x_t, x_{t+h}) pairs
        - C_iter     = (C_Delta)^h  using the production single-step fit
        - Sigma_iter = sum_{k=0}^{h-1} (C_Delta^T)^k Sigma_Delta C_Delta^k
      Output Frobenius-norm differences. There is NO ground truth here:
      gaps tell us how far the semigroup property fails on Ontario; they
      are *measurements*, not pass/fail criteria.

WHY THIS IS NOT A "REIMPLEMENTATION" diagnostic. The fits are done
through ``_local_fit_at`` (the production entry point) -- only the
``Y`` array changes (shifted by ``h`` rows instead of ``1``). The
iterated comparison is closed-form matrix algebra on the production
single-step output. No diagnostic reimplements production logic.

PROVENANCE-GRADE: INSPECTION-ONLY (no claim-grade results emitted).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from processing.innovations.estimator import _local_fit_at


# ---------------------------------------------------------------------------
# closed-form iteration of the linear Gaussian semigroup
# ---------------------------------------------------------------------------
def iterate_linear_gaussian(C: np.ndarray, Sigma: np.ndarray, h: int
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form ``(C^h, Sigma_iter_h)`` where state map is x_next = x @ C.

    Drift after h steps: C^h.
    Diffusion after h steps: sum_{k=0}^{h-1} (C^k)^T Sigma C^k
        (because at step k+1, x_{t+k+1} = x_{t+k} @ C + r_k, and the
         marginal covariance accumulates the propagated step-noise).
    """
    d = C.shape[0]
    C_pow = np.eye(d)
    Sigma_iter = np.zeros_like(Sigma)
    for _k in range(h):
        Sigma_iter = Sigma_iter + C_pow.T @ Sigma @ C_pow
        C_pow = C_pow @ C
    return C_pow, Sigma_iter


# ---------------------------------------------------------------------------
# Part A: synthetic VAR(1) gate
# ---------------------------------------------------------------------------
def _generate_var1(A: np.ndarray, Q: np.ndarray, n: int,
                   rng: np.random.Generator) -> np.ndarray:
    """Simulate x_{t+1} = x_t A + eps,  eps ~ N(0, Q),  x_0 from stationary."""
    d = A.shape[0]
    # discard burn-in to wash out the deterministic start
    burn = 500
    x = np.zeros((n + burn, d))
    L = np.linalg.cholesky(Q + 1e-12 * np.eye(d))
    for t in range(n + burn - 1):
        x[t + 1] = x[t] @ A + rng.standard_normal(d) @ L.T
    return x[burn:]


def _build_lagged_pairs(x: np.ndarray, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, Y) where Y[i] = x[i + h]."""
    return x[: -h, :], x[h:, :]


def synthetic_gate(n: int = 4000,
                   horizons: tuple[int, ...] = (2, 4, 8, 12, 24),
                   tol_drift: float = 0.05,
                   tol_diff: float  = 0.20,
                   seed: int = 12345) -> int:
    """Run the synthetic VAR(1) gate. Returns 0 on PASS, 1 on FAIL."""
    rng = np.random.default_rng(seed)
    # A 3D VAR(1) with eigenvalues inside the unit disk; not too sloppy.
    A = np.array([
        [ 0.70,  0.10,  0.00],
        [-0.05,  0.65,  0.20],
        [ 0.00, -0.10,  0.55],
    ])
    # Symmetric positive-definite innovation covariance.
    Q = np.array([
        [1.00, 0.20, 0.05],
        [0.20, 0.80, 0.10],
        [0.05, 0.10, 0.60],
    ])
    d = A.shape[0]
    x = _generate_var1(A, Q, n, rng)

    # one-step direct fit through the production entry point
    X1, Y1 = _build_lagged_pairs(x, 1)
    # use the population mean as the query state, so distance is uniform
    # and the LOO-CV converges to a small theta -- a near-global fit, which
    # is the right limit for a true VAR(1).  Caller can sweep query states
    # in extension work.
    x_query = X1.mean(axis=0)
    C1, S1, mu1, th1, _ = _local_fit_at(X1, Y1, x_query, d)

    print("\nSynthetic VAR(1) gate")
    print(f"  n = {n}  d = {d}  theta* (one-step) = {th1:.3f}")
    print(f"  ||C_hat - A||_F      = {np.linalg.norm(C1 - A, 'fro'):.4f}")
    print(f"  ||Sigma_hat - Q||_F  = {np.linalg.norm(S1 - Q, 'fro'):.4f}")

    all_pass = (np.linalg.norm(C1 - A, 'fro') < tol_drift) \
               and (np.linalg.norm(S1 - Q, 'fro') < tol_diff)

    print(f"\n  {'h':>3s} {'||C_h - A^h||F':>16s} {'||Sigma_h_dir - Sigma_iter||F':>32s} {'rel':>10s} pass")
    for h in horizons:
        Xh, Yh = _build_lagged_pairs(x, h)
        x_query_h = Xh.mean(axis=0)
        Ch, Sh, _mu, _th, _ = _local_fit_at(Xh, Yh, x_query_h, d)
        # iterated reference  (closed form on the *true* (A, Q) is the right
        # target, since the synthetic is genuinely VAR(1))
        Ah, Qiter_true = iterate_linear_gaussian(A, Q, h)
        drift_err = np.linalg.norm(Ch - Ah, 'fro')
        diff_err  = np.linalg.norm(Sh - Qiter_true, 'fro')
        diff_rel  = diff_err / max(np.linalg.norm(Qiter_true, 'fro'), 1e-12)
        ok = drift_err < tol_drift * (1 + 0.5 * np.log2(h)) \
             and diff_rel < tol_diff * (1 + 0.5 * np.log2(h))
        all_pass = all_pass and ok
        print(f"  {h:>3d} {drift_err:>16.4f} {diff_err:>32.4f} {diff_rel:>10.3%}"
              f"  {'PASS' if ok else 'FAIL'}")
    print(f"\n  Synthetic gate: {'PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 1


# ---------------------------------------------------------------------------
# Part B: Ontario in-sample probe
# ---------------------------------------------------------------------------
def ontario_probe(daytype: str = "weekday",
                  horizons: tuple[int, ...] = (2, 4, 8, 12, 24),
                  out_csv: Path | None = None) -> None:
    """Run the Ontario in-sample CK probe.  No pass/fail -- diagnostic only."""
    from edynamics.modelling_tools import Embedding, Lag
    from config import load_config
    from experiment import freeze
    from experiment._actuals import (
        load_actuals, zscore_params, zscore_transform,
    )
    from experiment.predict import _daytype

    cfg = load_config()
    spec = freeze.load_verified()
    cutoff = pd.Timestamp(spec["data_cutoff"])
    anchor_h = spec["predictor"]["day_anchor_hour"]
    clim_method = spec["predictor"].get("climatology_method", "month_hour")
    fourier_k_year = spec["predictor"].get("fourier_k_year")
    fourier_k_day = spec["predictor"].get("fourier_k_day")

    # Build the pre-cutoff z-score series the same way generate() does
    zp = zscore_params(
        cutoff, method=clim_method,
        k_year=fourier_k_year, k_day=fourier_k_day,
    )
    raw = load_actuals(cutoff=cutoff)
    zser = zscore_transform(raw, zp)
    # Filter to the requested daytype on the pre-cutoff window
    dtags = zser.index.map(lambda t: _daytype(t, anchor_h))
    zser = zser[dtags == daytype].asfreq("h")
    d = int(cfg.embedding_dim(daytype))
    lags = [Lag(variable_name="zscore", tau=-k) for k in range(d)]
    emb = Embedding(
        data=zser.to_frame("zscore"),
        observers=lags,
        library_times=zser.dropna().index[d:-1],
    )
    emb.compile()
    block = emb.block.dropna()
    arr = block.values

    print(f"\nOntario in-sample probe -- daytype={daytype}  d={d}  n={len(arr)}")
    # one-step fit at the population mean (in-sample best-linear baseline)
    X1, Y1 = arr[:-1, :], arr[1:, :]
    x_query = X1.mean(axis=0)
    C1, S1, mu1, th1, _ = _local_fit_at(X1, Y1, x_query, d)
    print(f"  theta* (one-step, population-mean query) = {th1:.3f}")
    print(f"  cond(Sigma_Delta)                       = {np.linalg.cond(S1):.2e}")

    rows = []
    print(f"\n  {'h':>3s} {'theta*_h':>10s} {'cond(S_h_dir)':>14s} "
          f"{'||C_h_dir-C_iter||F':>22s} "
          f"{'||S_h_dir-S_iter||F':>22s} {'rel_drift':>10s} {'rel_diff':>10s}")
    for h in horizons:
        Xh, Yh = arr[:-h, :], arr[h:, :]
        x_query_h = Xh.mean(axis=0)
        Ch, Sh, _mu, th_h, _ = _local_fit_at(Xh, Yh, x_query_h, d)
        C_iter, S_iter = iterate_linear_gaussian(C1, S1, h)
        drift_err = np.linalg.norm(Ch - C_iter, 'fro')
        diff_err  = np.linalg.norm(Sh - S_iter, 'fro')
        drift_rel = drift_err / max(np.linalg.norm(C_iter, 'fro'), 1e-12)
        diff_rel  = diff_err  / max(np.linalg.norm(S_iter, 'fro'), 1e-12)
        cond_h = np.linalg.cond(Sh)
        print(f"  {h:>3d} {th_h:>10.3f} {cond_h:>14.2e} "
              f"{drift_err:>22.4f} {diff_err:>22.4f} "
              f"{drift_rel:>10.3%} {diff_rel:>10.3%}")
        rows.append({
            "daytype":   daytype,
            "horizon":   h,
            "theta_h":   th_h,
            "cond_S_h":  cond_h,
            "drift_err": drift_err,
            "diff_err":  diff_err,
            "drift_rel": drift_rel,
            "diff_rel":  diff_rel,
        })
    if out_csv is not None:
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"\n  wrote {out_csv}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--synthetic", action="store_true",
                   help="Run synthetic VAR(1) gate (Part A)")
    p.add_argument("--ontario", action="store_true",
                   help="Run Ontario in-sample probe (Part B)")
    p.add_argument("--daytype", default="weekday",
                   choices=("weekday", "saturday", "sunday"))
    p.add_argument("--out-csv", type=Path, default=None,
                   help="Optional CSV path for Ontario probe output")
    p.add_argument("--seed", type=int, default=12345)
    args = p.parse_args()

    if not (args.synthetic or args.ontario):
        # default: run both
        args.synthetic = True
        args.ontario = True

    rc = 0
    if args.synthetic:
        rc |= synthetic_gate(seed=args.seed)
    if args.ontario:
        try:
            ontario_probe(daytype=args.daytype, out_csv=args.out_csv)
        except Exception as exc:
            print(f"\nOntario probe failed (non-fatal): {exc}")
            # do not flip rc -- diagnostic
    return rc


if __name__ == "__main__":
    sys.exit(main())
