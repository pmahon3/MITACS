"""Critical discriminating check for the 'mixture-structured' claim
about Ontario residuals.

The Patra-Sen result on Ontario (α̂_L^95% = 0.49-0.74) could in
principle be consistent with a single heavy-tailed law (just one
α₀ = 0 + heavy tails through the partial-identifiability mechanism
of Patra-Sen Lemma 4).

To discriminate: run Patra-Sen on AR(1) + t(ν) data where ν is
chosen to match Ontario's residual kurt (kurt ≈ 22 ⟹ ν ≈ 4.27 by
the t-distribution formula 6/(ν-4) = kurt). If this gives α̂_L
similar to Ontario's, Ontario is consistent with a single heavy-
tailed AR(1) process. If it gives substantially smaller α̂_L,
Ontario has structure beyond what any single heavy-tailed law
produces — supporting the mixture-structured claim.

Also include AR(1) + N(0,1) at the SAME AR strength (a=0.95) as a
matched-strength null, to confirm AR autocorrelation doesn't itself
inflate α̂_L.

PROVENANCE-GRADE: INSPECTION-ONLY (discriminating check for
patra_sen_synthetic_null.py).
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
from scipy.stats import norm

from edynamics.modelling_tools import Embedding, Lag
from processing.innovations.estimator import local_drift_and_diffusion
from scratch.patra_sen import patra_sen

N_ANCHORS = 30
N_PER_ANCHOR_CAP = 2000


def _build_emb(z):
    idx = pd.date_range("2000-01-01", periods=len(z), freq="h")
    df = pd.DataFrame({"zscore": z}, index=idx).asfreq("h")
    obs = [Lag(variable_name="zscore", tau=-i) for i in range(3)]
    libt = idx[3:-1]
    emb = Embedding(data=df, observers=obs, library_times=libt)
    emb.compile()
    return emb, libt


def _pool(emb, libt, rng):
    anchors = libt[np.sort(rng.choice(len(libt), N_ANCHORS, replace=False))]
    out = []
    for a in anchors:
        try:
            _C, S, _mu, _theta, resid = local_drift_and_diffusion(
                embedding=emb, anchor=a, day_anchor_hour=None,
            )
        except Exception:
            continue
        r0 = resid[:, 0]
        s00 = float(S[0, 0])
        if s00 <= 0:
            continue
        z_std = r0 / np.sqrt(s00)
        z_std = z_std[np.isfinite(z_std)]
        if len(z_std) > N_PER_ANCHOR_CAP:
            z_std = z_std[rng.choice(len(z_std), N_PER_ANCHOR_CAP, replace=False)]
        out.append(z_std)
    return np.concatenate(out) if out else np.array([])


def _run_one(label, z, t0):
    rng = np.random.default_rng(7)
    emb, libt = _build_emb(z)
    pooled = _pool(emb, libt, rng)
    kurt = float(pd.Series(pooled).kurtosis())
    res = patra_sen(pooled, F_b=lambda x: norm.cdf(x), gamma_grid_size=600)
    print(f"[{time.time()-t0:6.0f}s] {label}: n={len(pooled)}, "
          f"kurt={kurt:.2f}, α̂={res.alpha_hat:.4f}, "
          f"α̂_L^95%={res.alpha_L_95:.4f}", flush=True)
    return {"label": label, "kurt": kurt,
            "alpha_hat": res.alpha_hat, "alpha_L_95": res.alpha_L_95}


def _sim_ar1_t(n, df_t, a=0.95, burn=500, seed=99):
    rng = np.random.default_rng(seed)
    scale = np.sqrt((df_t - 2) / df_t)  # unit-variance t innovation
    T = n + burn
    z = np.zeros(T + 1)
    for t in range(T):
        z[t + 1] = a * z[t] + scale * rng.standard_t(df_t)
    return z[burn + 1:]


def _sim_ar1_g(n, a=0.95, burn=500, seed=88):
    rng = np.random.default_rng(seed)
    T = n + burn
    z = np.zeros(T + 1)
    for t in range(T):
        z[t + 1] = a * z[t] + rng.standard_normal()
    return z[burn + 1:]


def main():
    print("Kurt-matched discriminating check: does pure AR(1) + t(ν) "
          "reach Ontario's α̂_L?")
    print(f"Ontario kurt ~ 22-23; Patra-Sen α̂_L ~ 0.49-0.74")
    print(f"t(ν) population kurt = 6/(ν-4), so kurt=22 ⟹ ν≈4.27")
    print()

    t0 = time.time()
    n_sim = 24000

    rows = []
    # AR(1) + t(ν) at increasing tail heaviness
    for nu in [4.27, 4.18, 4.10]:
        pop_kurt = 6 / (nu - 4) if nu > 4 else float("inf")
        z = _sim_ar1_t(n_sim, df_t=nu, a=0.95, seed=int(100 * nu))
        label = f"AR(1) a=0.95 + t({nu})  [pop kurt={pop_kurt:.1f}]"
        rows.append(_run_one(label, z, t0))

    # Matched-strength null
    print()
    z_g = _sim_ar1_g(n_sim, a=0.95)
    rows.append(_run_one("AR(1) a=0.95 + N(0,1) [matched null]", z_g, t0))

    print()
    print("=" * 80)
    print("Conclusion:")
    print("=" * 80)
    print("  Ontario α̂_L^95% = 0.49-0.74; sample kurt 20-23")
    print(f"  Best single-law match here (AR(1)+t(4.27), pop kurt 22):")
    nu_match = [r for r in rows if "4.27" in r["label"]][0]
    print(f"    sample kurt = {nu_match['kurt']:.2f}, α̂_L^95% = "
          f"{nu_match['alpha_L_95']:.4f}")
    print()
    diff = 0.51 - nu_match["alpha_L_95"]  # min Ontario L95 minus single-law match
    if diff > 0.20:
        print(f"  GAP = Ontario L95 (~0.51 min) − single-law L95 "
              f"({nu_match['alpha_L_95']:.3f}) = {diff:.3f}")
        print(f"  Ontario is SUBSTANTIALLY MORE 'contaminated' than any single")
        print(f"  heavy-tailed AR(1) produces. The 'mixture-structured' "
              "claim SURVIVES.")
    else:
        print(f"  GAP small ({diff:.3f}); Ontario is consistent with single")
        print(f"  heavy-tailed AR(1). The 'mixture-structured' claim RETRACTS.")


if __name__ == "__main__":
    main()
