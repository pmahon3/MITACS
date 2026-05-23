"""Pipeline-aware null for Patra-Sen on Ontario residuals.

The advisor flagged that pure-N(0,1) pooling does NOT capture the
estimation-pipeline noise: local_drift_and_diffusion's residuals at
each anchor are not i.i.d.; they're correlated through the WLS fit
and overlapping libraries.

This script runs the EXACT same protocol as
scratch/apply_patra_sen_ontario.py but on synthetic VAR(1) data
known to be σ-additive (Gaussian innovations). The Patra-Sen number
on this baseline is the pipeline-induced null floor against which
the Ontario numbers should be compared.

Also runs a t-innovation VAR(1) (known heavy-tailed) as a positive
control — a setting where contamination IS present at α=1.

Uses the existing gate-validated synthetic.py simulators to ensure
the data is bit-comparable to the validation-gate fixtures.

PROVENANCE-GRADE: INSPECTION-ONLY (pipeline-null calibration).
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd
from scipy.stats import norm

from edynamics.modelling_tools import Embedding, Lag
from processing.innovations.estimator import local_drift_and_diffusion
from processing.innovations.validation.synthetic import (
    make_var1_params, simulate_var1, simulate_var1_t,
)
from scratch.patra_sen import patra_sen

# Match Ontario protocol exactly
N_ANCHORS = 30
N_PER_ANCHOR_CAP = 2000
RNG_SEED = 7


def _build_embedding_from_X(X: np.ndarray) -> tuple[Embedding, pd.DatetimeIndex]:
    """Single-variable delay-embedding (NOT the synthetic.py version which
    uses tau=0 multi-variable). This matches the Ontario pipeline's
    actual structure: one variable, delay-lags 0..d-1."""
    d = X.shape[1] if X.ndim > 1 else 1
    # Use first coord as the single z series; standard delay-lag embedding
    z = X[:, 0] if X.ndim > 1 else X
    idx = pd.date_range("2000-01-01", periods=len(z), freq="h")
    df = pd.DataFrame({"zscore": z}, index=idx).asfreq("h")
    # Match Ontario's per-day-type d (use d=3 as a representative value)
    d_emb = 3
    observers = [Lag(variable_name="zscore", tau=-i) for i in range(d_emb)]
    library_times = idx[d_emb:-1]
    embedding = Embedding(data=df, observers=observers,
                          library_times=library_times)
    embedding.compile()
    return embedding, library_times


def _pool_standardised_residuals(embedding, library_times,
                                 n_anchors: int, n_per_anchor: int,
                                 rng) -> np.ndarray:
    """Mirror apply_patra_sen_ontario._pool_standardised_residuals."""
    anchors = library_times[
        np.sort(rng.choice(len(library_times),
                           min(n_anchors, len(library_times)), replace=False))
    ]
    pooled = []
    for a in anchors:
        try:
            _C, S, _mu, _theta, resid = local_drift_and_diffusion(
                embedding=embedding, anchor=a,
            )
        except Exception:
            continue
        r0 = resid[:, 0]
        s00 = float(S[0, 0])
        if s00 <= 0 or not np.isfinite(s00):
            continue
        z_std = r0 / np.sqrt(s00)
        z_std = z_std[np.isfinite(z_std)]
        if len(z_std) > n_per_anchor:
            idx = rng.choice(len(z_std), n_per_anchor, replace=False)
            z_std = z_std[idx]
        pooled.append(z_std)
    return np.concatenate(pooled) if pooled else np.array([])


def _run(label: str, X: np.ndarray, t0: float, rng):
    print(f"[{time.time()-t0:6.0f}s] {label}: building embedding...", flush=True)
    emb, libt = _build_embedding_from_X(X)
    print(f"[{time.time()-t0:6.0f}s] {label}: pooling residuals "
          f"({N_ANCHORS} anchors × cap {N_PER_ANCHOR_CAP})...", flush=True)
    pooled = _pool_standardised_residuals(
        emb, libt, N_ANCHORS, N_PER_ANCHOR_CAP, rng,
    )
    if len(pooled) < 100:
        print(f"  too few residuals: {len(pooled)}")
        return None
    kurt = float(pd.Series(pooled).kurtosis())
    print(f"[{time.time()-t0:6.0f}s] {label}: n_pooled={len(pooled)}, "
          f"mean={pooled.mean():+.4f}, std={pooled.std():.4f}, kurt={kurt:.2f}",
          flush=True)
    print(f"[{time.time()-t0:6.0f}s] {label}: running Patra-Sen...", flush=True)
    res = patra_sen(pooled, F_b=lambda x: norm.cdf(x), gamma_grid_size=600)
    print(f"[{time.time()-t0:6.0f}s] {label}: → α̂ = {res.alpha_hat:.4f}, "
          f"α̂_L^95% = {res.alpha_L_95:.4f}", flush=True)
    return {"label": label, "n": len(pooled), "kurt": kurt,
            "alpha_hat": res.alpha_hat, "alpha_L_95": res.alpha_L_95}


def main():
    t0 = time.time()
    print("=" * 84)
    print("Pipeline-aware null calibration for Patra-Sen on Ontario residuals")
    print("=" * 84)
    print(f"Protocol matches apply_patra_sen_ontario.py exactly: "
          f"{N_ANCHORS} anchors × cap {N_PER_ANCHOR_CAP} per-anchor, pooled, "
          f"F_b=N(0,1).")
    print()

    # Use same n as Ontario library (~24k pre-cutoff per day-type for weekday)
    n_sim = 24000
    rows = []

    # === Null A: VAR(1) with Gaussian innovations (truly σ-additive) ===
    print(f"=== Null A: VAR(1) with Gaussian innovations (d=3, n={n_sim}) ===")
    A, Q = make_var1_params(d=3, seed=42)
    Xg = simulate_var1(A, Q, n=n_sim, burn=500, seed=43)
    rng = np.random.default_rng(RNG_SEED)
    rows.append(_run("VAR1_gauss", Xg, t0, rng))

    # === Null B: VAR(1) with Student-t(5) innovations (known heavy-tailed) ===
    print(f"\n=== Positive control: VAR(1) with t(5) innovations (d=3, n={n_sim}) ===")
    A, Q = make_var1_params(d=3, seed=42)
    Xt = simulate_var1_t(A, Q, n=n_sim, burn=500, seed=44, df=5)
    rng = np.random.default_rng(RNG_SEED)
    rows.append(_run("VAR1_t(5)", Xt, t0, rng))

    # === Stronger heavy-tail positive control: t(4.5) ===
    print(f"\n=== Positive control: VAR(1) with t(4.5) innovations (d=3, n={n_sim}) ===")
    Xt45 = simulate_var1_t(A, Q, n=n_sim, burn=500, seed=45, df=5)
    # simulate_var1_t doesn't take non-int df; manual override:
    rng_t = np.random.default_rng(46)
    L = np.linalg.cholesky(Q)
    d = 3
    df_t = 4.5
    scale = np.sqrt((df_t - 2) / df_t)
    T = n_sim + 500
    Xt45 = np.empty((T + 1, d))
    Xt45[0] = rng_t.standard_normal(d)
    for t in range(T):
        eps = rng_t.standard_t(df_t, size=d) * scale
        Xt45[t + 1] = A @ Xt45[t] + L @ eps
    Xt45 = Xt45[500:]
    rng = np.random.default_rng(RNG_SEED)
    rows.append(_run("VAR1_t(4.5)", Xt45, t0, rng))

    # Summary
    print()
    print("=" * 84)
    print("Pipeline null calibration:")
    print("=" * 84)
    print(f"{'condition':>16s} {'n':>7s} {'kurt':>8s} "
          f"{'alpha_hat':>11s} {'alpha_L_95':>11s}  reading")
    for r in rows:
        if r is None:
            continue
        if r["alpha_L_95"] < 0.05:
            reading = "PIPELINE OK (null α̂_L ≈ 0)"
        elif r["alpha_L_95"] < 0.20:
            reading = "modest pipeline-induced floor"
        else:
            reading = "PIPELINE INDUCES α̂_L (concerning)"
        print(f"{r['label']:>16s} {r['n']:>7d} {r['kurt']:>8.2f} "
              f"{r['alpha_hat']:>11.4f} {r['alpha_L_95']:>11.4f}  {reading}")

    print()
    print("Compare against Ontario (apply_patra_sen_ontario.py):")
    print("  weekday(fr): kurt=22.0, α̂=0.58, L95=0.51")
    print("  saturday(fr): kurt=20.5, α̂=0.60, L95=0.52")
    print("  sunday(fr): kurt=23.2, α̂=0.81, L95=0.74")
    print()
    print("Reading rule:")
    print("  VAR1_gauss α̂_L^95% ≈ 0  → Ontario α̂_L is GENUINE SIGNAL")
    print("    Ontario contamination is real, not pipeline-induced.")
    print("  VAR1_gauss α̂_L^95% > 0.1 → pipeline induces baseline")
    print("    Ontario excess above baseline is signal; magnitude reframes.")
    print("  VAR1_gauss α̂_L^95% ≈ 0.5+ → pipeline IS the artefact;")
    print("    Ontario reading is largely methodological; retract.")


if __name__ == "__main__":
    main()
