"""DECIDING TEST (advisor-required) for the theta-rail finding.

Diagnostics 1+2 (diagnose_theta_loo_railpin.py) showed the production
`_theta_loo_cv` rails to the grid ceiling on Ontario z-scores AND on a
linear VAR(1). That alone cannot distinguish:
  (A) the rule is broken (rails regardless of whether locality exists);
  (B) the data is genuinely near-linear so global IS LOO-optimal.

This probes a system where locality is UNAMBIGUOUSLY present and the
correct bandwidth is small: the logistic map x' = r x (1-x) embedded in
delay coordinates. Its one-step map has a state-dependent slope
(derivative r(1-2x)) ranging sign and magnitude across the attractor —
a global linear fit is provably wrong; a competent bandwidth selector
MUST pick theta small enough to localize.

Decision rule:
  * _theta_loo_cv picks theta well BELOW the per-anchor distance scale
    (localizing) on the logistic map  => rule CAN detect locality;
    the Ontario/VAR(1) railing is the DATA being near-linear (branch B).
    Consequence: §3 Gaussian-proxy story strengthens; §2 "locally"
    wording softens to "reduces to global on this data".
  * _theta_loo_cv ALSO rails to grid max on the logistic map
    => the SELECTOR is broken (branch A); fix is in the rule. The
    recovery gate never caught it because a linear VAR(1) is recovered
    by global OLS anyway.

Also runs a regime-switching VAR (drift flips by half-space) as a
second, higher-dimensional locality witness.

RESULT (2026-05-18, production `_theta_loo_cv`): **BRANCH A — SELECTOR
BROKEN.** theta/grid_max median = 1.000 for 100% of anchors on the
logistic map (d=2 AND d=3) and the regime-switching VAR; median
library fraction inside the half-weight kernel ≈ 1.0 (fully global).
These systems are UNAMBIGUOUSLY state-dependent (logistic slope
r(1-2x); regime drift flips by half-space) — a competent selector MUST
localize and CANNOT here under any near-linear reading. Confirmed
across FOUR independent systems (Ontario z-scores, linear VAR(1),
logistic map, regime-switching VAR): `_theta_loo_cv` rails to the grid
ceiling irrespective of whether locality exists. The recovery gate
never caught it because a linear VAR(1) is recovered by global OLS
regardless of θ. Conclusion: the bandwidth selector is degenerate; the
"locally-Gaussian" estimator is, as run, global OLS with a constant
kernel. Fix NOT pre-decided (surfaced to advisor per discipline rule).

PROVENANCE-GRADE: INSPECTION-ONLY — exploratory dev diagnostic that
calls the PRODUCTION `_theta_loo_cv`. It DECIDES the framing question
(branch A); the fix is a separate, not-yet-taken decision.
"""
from __future__ import annotations

import numpy as np

from processing.innovations.estimator import _gauss_w, _theta_loo_cv


def _delay_embed(s: np.ndarray, d: int) -> tuple[np.ndarray, np.ndarray]:
    cols = [s[i : len(s) - (d - i)] for i in range(d)]
    X = np.column_stack(cols)[:-1]
    Y = np.column_stack([c[1:] for c in cols])
    n = min(len(X), len(Y))
    return X[:n], Y[:n]


def logistic_series(n: int, r: float = 3.95, x0: float = 0.37) -> np.ndarray:
    x = np.empty(n)
    x[0] = x0
    for i in range(1, n):
        x[i] = r * x[i - 1] * (1.0 - x[i - 1])
    return x


def regime_switch_var(n: int, d: int = 3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A_hi = 0.6 * np.eye(d) + 0.1 * rng.standard_normal((d, d))
    A_lo = -0.5 * np.eye(d) + 0.1 * rng.standard_normal((d, d))
    X = np.zeros((n, d))
    X[0] = rng.standard_normal(d)
    for i in range(1, n):
        A = A_hi if X[i - 1, 0] > 0 else A_lo  # drift flips by half-space
        X[i] = X[i - 1] @ A.T + 0.05 * rng.standard_normal(d)
    return X


def probe(name: str, X: np.ndarray, Y: np.ndarray, d: int,
          n_anchor: int = 25) -> None:
    rng = np.random.default_rng(1)
    qi = rng.choice(len(X), n_anchor, replace=False)
    rows = []
    for q in X[qi]:
        dists = np.linalg.norm(X - q, axis=1)
        th = _theta_loo_cv(dists, X, Y, d)
        sub = 150
        rg = np.random.default_rng(0)
        sid = rg.choice(len(dists), sub, replace=False) if len(dists) > sub \
            else np.arange(len(dists))
        ds = dists[sid]
        gmax = ds.max()
        # fraction of library within the selected kernel's half-weight
        w = _gauss_w(dists, th, d)
        frac_eff = float((w > 0.5 * w.max()).mean())
        rows.append((th, gmax, th / gmax, frac_eff))
    a = np.array(rows)
    print(f"=== {name} (d={d}, N={len(X)}) ===")
    print(f"  theta/grid_max: median={np.median(a[:,2]):.3f} "
          f"min={a[:,2].min():.3f} max={a[:,2].max():.3f}")
    print(f"  frac at grid max (theta/gmax>0.98): {(a[:,2]>0.98).mean():.2f}")
    print(f"  median library fraction inside half-weight kernel: "
          f"{np.median(a[:,3]):.3f}  "
          f"(small => localizing; ~1.0 => global/degenerate)")
    verdict = (
        "RAILS (degenerate here too => SELECTOR broken, branch A)"
        if (a[:, 2] > 0.98).mean() > 0.5
        else "LOCALIZES (rule works => Ontario near-linear, branch B)"
    )
    print(f"  -> {verdict}\n")


def main() -> None:
    s = logistic_series(6000)
    for d in (2, 3):
        X, Y = _delay_embed(s, d)
        probe(f"logistic map r=3.95", X, Y, d)
    Xr = regime_switch_var(6000, d=3)
    probe("regime-switching VAR", Xr[:-1], Xr[1:], 3)
    print("READ: logistic/regime are UNAMBIGUOUSLY state-dependent; a "
          "competent selector MUST localize there. If it rails like it "
          "did on Ontario/VAR(1), the selector is broken (branch A). If "
          "it localizes, the Ontario railing is the data being "
          "near-linear (branch B).")


if __name__ == "__main__":
    main()
