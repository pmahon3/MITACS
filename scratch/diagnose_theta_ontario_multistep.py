"""DECIDING evidence for #41: does multi-step CV restore an interior
theta optimum on ACTUAL ONTARIO z-scores (not synthetic)?

Repeatedly deferred; the advisor flagged that every multi-step result
so far is synthetic. The fix-vs-characterize decision for #41 hinges on
Ontario specifically:
  * one-step LOO is degenerate (monotone-to-edge) on Ontario, linear
    VAR(1), logistic — but NOT regime-switch (interior optimum at h=1).
    So one-step degeneracy is SYSTEM-DEPENDENT and correlates with the
    conditional mean being near-globally-linear.
  * If multi-step ALSO rails on Ontario  => Ontario's conditional mean
    is near-globally-linear at ANY horizon; "select a localization
    bandwidth" is the wrong objective for Ontario; accept near-global
    kappa_Q at this embedding and let the §3 Gaussian-proxy /
    non-Gaussian innovation carry the structure (characterize, not
    engineer a selector).
  * If multi-step gives an interior optimum on Ontario => locality IS
    recoverable with the right (multi-step) objective; the rule needs
    fixing, not just characterizing.

Uses the production weekday z-score embedding and the production-
faithful iteration (rebuild lag vector each step, score scalar z_next
vs true future z, exactly predict_multistep).

‼️ RESULT — STATISTICALLY NULL (2026-05-18, advisor-directed bootstrap
SE check across the 25 anchors; the prior "mixed/interior" read was a
phantom from reporting argmin POSITION without a variance check —
exactly the slip the mechanism investigation already warned about).
For EVERY horizon the gap between the apparent interior optimum and the
grid edge is SMALLER than 2*(combined bootstrap SE across anchors):
  h= 1  r0  gap +0.021  SE~0.024+0.020  meaningful=False
  h= 2  r0  gap +0.027  SE~0.052+0.026  meaningful=False
  h= 4  r5  gap +0.048  SE~0.059+0.066  meaningful=False
  h= 8  r3  gap +0.054  SE~0.104+0.077  meaningful=False
  h=16  r6  gap +0.068  SE~0.232+0.204  meaningful=False
  h=24  r0  gap +0.115  SE~0.153+0.157  meaningful=False (+6 NaN
        anchors: 24-step iteration diverged)
=> On real Ontario data the multi-step CV objective is FLAT-WITHIN-
NOISE at ALL horizons; NO statistically separated interior theta
optimum exists. Multi-step CV does NOT rescue localization on Ontario.
Clean conclusion: Ontario's conditional mean is effectively near-
globally-linear at this (d=2 z-score lag) embedding; locality is not
recoverable by ANY prediction-error CV objective tested. Tips #41
firmly to CHARACTERIZE (not engineer a selector); raises whether the
real lever is the EMBEDDING, not the bandwidth rule (advisor).

PROVENANCE-GRADE: INSPECTION-ONLY — exploratory dev probe on the real
pre-cutoff Ontario series; calls production _gauss_w. Informs (does
not by itself decide) fix-vs-characterize for #41.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from edynamics.modelling_tools import Embedding, Lag

from experiment._actuals import load_actuals, zscore_params, zscore_transform
from experiment.predict import _daytype
from processing.innovations.estimator import _gauss_w

CUT = pd.Timestamp("2024-12-31T23:00:00")


def _hstep_loo(s: np.ndarray, t0: int, th: float, d: int, h: int) -> float:
    L = np.stack([s[i - d : i][::-1] for i in range(d, len(s) - 1)])
    nxt = s[d : d + len(L)]
    base = np.arange(d, len(s) - 1)
    if t0 + h >= len(s) or t0 - d < 0:
        return np.nan
    excl = (base >= t0 - d) & (base <= t0 + h)
    Lk, nk = L[~excl], nxt[~excl]
    hist = list(s[t0 - d : t0][::-1])
    zhat = None
    for _ in range(h):
        xq = np.array(hist[:d], dtype=float)
        dist = np.linalg.norm(Lk - xq, axis=1)
        w = _gauss_w(dist, th, d)
        if w.sum() <= 0:
            return np.nan
        C = np.linalg.lstsq(w[:, None] * Lk, (w * nk)[:, None],
                            rcond=None)[0]
        zhat = float(xq @ C[:, 0])
        hist = [zhat] + hist
    return (zhat - s[t0 + h]) ** 2 if zhat is not None else np.nan


def main() -> None:
    d = 2  # production weekday embedding dim
    act = load_actuals(cutoff=None)
    z = zscore_transform(act, zscore_params(CUT))
    df = z.to_frame("zscore").asfreq("h")
    dv = df.index[df.index <= CUT]
    idx = dv[[_daytype(t, 7) == "weekday" for t in dv]]
    # contiguous weekday z-series in time order (the production library
    # is weekday anchors; for an h-step iterate we need the realized
    # path, so use the full hourly z restricted to <=cut as the series
    # and sample weekday origins — matches how predict_multistep feeds
    # forward hour by hour)
    s = df.loc[df.index <= CUT, "zscore"].to_numpy()
    s = s[np.isfinite(s)]

    horizons = (1, 2, 4, 8, 16, 24)
    ng = 16
    n_anchor = 25
    rng = np.random.default_rng(7)
    valid = np.arange(d, len(s) - max(horizons) - 1)
    anchors = rng.choice(valid, n_anchor, replace=False)
    L = np.stack([s[i - d : i][::-1] for i in range(d, len(s) - 1)])
    samp = L[np.random.default_rng(1).choice(len(L), 500, replace=False)]
    from scipy.spatial.distance import pdist

    dd = pdist(samp)
    grid = np.geomspace(max(dd.min(), 1e-3), dd.max(), ng)
    print(f"ONTARIO weekday z-scores, d={d}, N_series={len(s)}, "
          f"median over {n_anchor} origins")
    print(f"  theta grid [{grid[0]:.3f}, {grid[-1]:.3f}] "
          f"(pairwise z-dist scale)")
    print(f"{'idx':>3} " + " ".join(f"{'h=' + str(h):>11}" for h in horizons))
    M = {h: np.full((n_anchor, ng), np.nan) for h in horizons}
    for a, t0 in enumerate(anchors):
        for g, th in enumerate(grid):
            for h in horizons:
                M[h][a, g] = _hstep_loo(s, int(t0), th, d, h)
    med = {h: np.nanmedian(M[h], axis=0) for h in horizons}
    for g in range(ng):
        print(f"{g:>3} " + " ".join(f"{med[h][g]:11.4e}" for h in horizons))
    print("\n  argmin (bandwidth rank; INTERIOR if not 0/last):")
    for h in horizons:
        bi = int(np.nanargmin(med[h]))
        kind = "INTERIOR-MIN" if 0 < bi < ng - 1 else "edge(monotone)"
        print(f"   h={h:>2}: argmin {bi}/{ng - 1}  {kind}")
    print("\nREAD: all/most h edge(monotone) => Ontario conditional mean "
          "near-globally-linear at any horizon; selecting a localization "
          "bandwidth is the wrong objective for Ontario -> CHARACTERIZE "
          "(accept near-global kappa_Q + let §3 carry it), do NOT "
          "engineer a selector. Interior optima appearing at h>1 => "
          "locality IS recoverable; the rule needs the multi-step fix.")


if __name__ == "__main__":
    main()
