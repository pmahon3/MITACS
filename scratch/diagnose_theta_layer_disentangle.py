"""DISCRIMINATING TEST (#41): is multi-step-CV theta* the kappa_Q
state-space locality scale, or a {Pi_t}-layer (horizon) artifact?

Theory (Resolvent_Framework program_overview.md): kappa_Q is the
disintegration layer with NO time index; {Pi_t} is DERIVED from kappa_Q
by Chapman-Kolmogorov and is where horizon h lives. A genuine kappa_Q
locality scale is therefore h-INVARIANT. The earlier multi-step probe
showed theta* DRIFTS with h on the logistic map — consistent with
multi-step CV measuring the composed {Pi_t} layer, not kappa_Q. But a
coherent theory argument is a hypothesis until discriminated (project
discipline). This builds a system with a KNOWN, TUNABLE state-space
locality scale ell* set INDEPENDENTLY of h, and asks which one theta*
follows.

Construction: a controlled map whose local Jacobian varies sinusoidally
in state with spatial wavelength ell*:

    x_{t+1} = a(x_t) * x_t + small noise,
    a(x) = rho * cos(2*pi*x / ell*)

The local-linear approximation of this map is accurate within ~ell*/(2*pi)
of any anchor and provably wrong on scales >> ell* (the slope itself
turns over on scale ell*). ell* is the true state-space locality scale
and is decoupled from any forecast horizon h.

Predictions:
  * kappa_Q-faithful  : theta* ~ c * ell*, ROUGHLY CONSTANT in h, and
    theta* SCALES with ell* when ell* is varied.
  * {Pi_t}-layer       : theta* tracks h (grows/shrinks with horizon)
    and is INSENSITIVE to ell* — i.e. changing the true locality scale
    does not move theta*, but changing h does.
  * MIXED              : theta* depends on BOTH -> the two scalings are
    entangled in any prediction-error criterion; report and reconsider
    the criterion family (plug-in / disintegration-geometry) rather
    than picking a horizon.

Output: theta*(h, ell*) table. Read the rows (vary h at fixed ell*) vs
the columns (vary ell* at fixed h).

PROVENANCE-GRADE: INSPECTION-ONLY — exploratory dev probe; calls the
production _gauss_w and iterates as production predict_multistep.
Discriminates the theory framing for #41; the fix decision follows the
result, not this script.
"""
from __future__ import annotations

import numpy as np

from processing.innovations.estimator import _gauss_w


def make_series(n: int, ell: float, rho: float = 0.92,
                noise: float = 0.02, seed: int = 0) -> np.ndarray:
    """Scalar series whose one-step local slope a(x)=rho*cos(2 pi x/ell)
    turns over on spatial scale ell (the true state-space locality
    scale). Bounded, mixing; ell decoupled from any horizon."""
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = 0.1
    for t in range(1, n):
        a = rho * np.cos(2.0 * np.pi * x[t - 1] / ell)
        x[t] = a * x[t - 1] + noise * rng.standard_normal()
    return x


def _hstep_loo(s: np.ndarray, t0: int, th: float, d: int, h: int) -> float:
    """Production-faithful held-out h-step iterated sq err at bandwidth
    th (rebuild lag vector each step, score scalar vs true future)."""
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


def theta_star(s: np.ndarray, h: int, d: int = 3,
               n_anchor: int = 25, ng: int = 16) -> float:
    """Median-over-anchors argmin theta for horizon h (grid in absolute
    distance units so theta* is comparable across ell*)."""
    rng = np.random.default_rng(7)
    valid = np.arange(d, len(s) - h - 1)
    anchors = rng.choice(valid, n_anchor, replace=False)
    L = np.stack([s[i - d : i][::-1] for i in range(d, len(s) - 1)])
    samp = L[np.random.default_rng(1).choice(len(L), 400, replace=False)]
    from scipy.spatial.distance import pdist

    dd = pdist(samp)
    grid = np.geomspace(max(dd.min(), 1e-3), dd.max(), ng)
    M = np.full((n_anchor, ng), np.nan)
    for a, t0 in enumerate(anchors):
        for g, th in enumerate(grid):
            M[a, g] = _hstep_loo(s, int(t0), th, d, h)
    med = np.nanmedian(M, axis=0)
    bi = int(np.nanargmin(med))
    return float(grid[bi])


def main() -> None:
    ells = (0.5, 1.0, 2.0)        # true state-space locality scales
    horizons = (1, 2, 4, 8, 16)   # forecast horizons (Pi_t layer)
    print("theta*(h, ell*)  — rows: vary h at fixed ell* ; "
          "cols: vary ell* at fixed h")
    hdr = "h\\ell*"
    print(f"{hdr:>8}" + "".join(f"{e:>12.2f}" for e in ells))
    series = {e: make_series(6000, e) for e in ells}
    table = {}
    for h in horizons:
        row = []
        for e in ells:
            ts = theta_star(series[e], h)
            row.append(ts)
            table[(h, e)] = ts
        print(f"{h:>8}" + "".join(f"{v:>12.4f}" for v in row))

    # quantify which factor moves theta*
    H = np.array(horizons, float)
    E = np.array(ells, float)
    T = np.array([[table[(h, e)] for e in ells] for h in horizons])
    # variation along h (fixed ell) vs along ell (fixed h), normalized
    dh = np.nanmean(np.nanstd(T, axis=0) / np.nanmean(T, axis=0))
    de = np.nanmean(np.nanstd(T, axis=1) / np.nanmean(T, axis=1))
    print(f"\n  mean rel. spread of theta* ACROSS h (fixed ell*): "
          f"{np.nanmean(np.nanstd(T, axis=0)/np.nanmean(T,axis=0)):.3f}")
    print(f"  mean rel. spread of theta* ACROSS ell* (fixed h): "
          f"{np.nanmean(np.nanstd(T, axis=1)/np.nanmean(T,axis=1)):.3f}")
    print("\nREAD: theta* nearly constant down each column (small "
          "spread across h) AND scaling across ell* => kappa_Q-faithful "
          "(multi-step CV OK). theta* moving down columns (with h) and "
          "flat across ell* => {Pi_t}-layer artifact (multi-step is the "
          "wrong register; go to plug-in / disintegration-geometry). "
          "Both move => entangled; reconsider the criterion family.")


if __name__ == "__main__":
    main()
