"""EXPLORATORY: does a MULTI-STEP CV objective restore an interior
theta optimum where one-step did not? (#41 fix-family probe;
user-directed investigate-first, 2026-05-18.)

Mechanism (diagnose_theta_mechanism.py, confirmed across anchors / d /
model class): one-step squared error is intrinsically monotone in
bandwidth -> theta rails to global. PREDICTION from that mechanism: the
locality bias COMPOUNDS over an iterated multi-step forecast, so a
too-global theta should be penalized at horizon h>1, creating an
interior optimum. This script tests that prediction BEFORE adopting a
multi-step criterion as the fix.

Decisive shape: for each horizon h in {1,2,4,8,16}, compute the
held-out h-step iterated squared error vs theta (same per-anchor grid,
median over anchors) on the logistic map (provably state-dependent;
a correct criterion MUST have an interior optimum). Expectation if the
mechanism is right and multi-step is the fix family:
  h=1  -> monotone-to-edge (reproduces the known failure)
  h increasing -> an interior minimum appears and DEEPENS (theta* moves
  off the grid ceiling toward a localizing value).
If multi-step is ALSO monotone-to-edge at all h -> multi-step is NOT
the fix; report that and reconsider (plug-in / locality-targeted).

RESULT (2026-05-18, production-faithful iteration — rebuilds the lag
vector each step exactly as predict_multistep, scores the scalar
z_next vs the true future scalar; h=1 sanity gate PASSES: 0.194,
monotone-to-edge, reproduces the known one-step failure — the earlier
1e-32 was a diagnostic bug, advisor-caught: old code scored the
trivially-shifted coord of an embedded target):
  h=1  argmin 15/15  edge(monotone)   — known failure reproduced
  h=2  argmin 14/15  INTERIOR-MIN
  h=4  argmin 12/15  INTERIOR-MIN
  h=8  argmin  8/15  INTERIOR-MIN
  h=16 argmin  7/15  INTERIOR-MIN
=> CONFIRMED: a MULTI-STEP CV objective restores an interior theta
optimum where one-step does not, and theta* moves progressively toward
localization as h grows — the mechanism-predicted signature (locality
bias compounds over the iterated forecast). Multi-step CV IS a viable
fix family, and the principled one (it matches production, which IS an
iterated 24-step forecaster).
OPEN (advisor-flagged, for the fix design — NOT decided here): theta*
DRIFTS with h (rank 14->12->8->7), i.e. multi-step CV picks a
horizon-dependent bandwidth. The fix must choose which horizon /
aggregate the criterion targets (production forecasts a full 24-step
day). That is the next decision, on this evidence.

PROVENANCE-GRADE: INSPECTION-ONLY — exploratory dev probe; calls the
production _gauss_w + iterates as production does. Establishes the fix
family + the open horizon question for #41; adopting/wiring it is a
separate step.
"""
from __future__ import annotations

import numpy as np

from processing.innovations.estimator import _gauss_w
from scratch.diagnose_theta_locality_probe import logistic_series


def _hstep_loo(s, sid_t, th, d, h):
    """Mean held-out h-step iterated squared error at bandwidth th,
    iterating EXACTLY as production predict_multistep does:

      z_next = x_query @ C[:, 0]; rebuild the lag vector from the
      running history each step (NOT matrix iteration of x@C).

    `s` is the scalar series; `sid_t` are candidate origin times t
    (indices into s) with d lags available and h future steps. For each
    t the library is all OTHER (lag-window, next) pairs, kernel-weighted
    by distance from the query lag-vector; the held-out origin's own
    pairs are excluded (true LOO). Locality bias compounds over the h
    iterated rebuilds — the production-faithful definition of h-ahead.
    """
    # build the full (lagvec -> next-scalar) library once
    L = np.stack([s[i - d : i][::-1] for i in range(d, len(s) - 1)])
    nxt = s[d : len(s) - 1 + 1][: len(L)]  # s at the step after each lagvec
    base = np.arange(d, len(s) - 1)  # origin time of row k = base[k]
    tot = 0.0
    cnt = 0
    for t in sid_t:
        if t + h >= len(s) or t - d < 0:
            continue
        # exclude library rows whose origin is within this forecast's
        # own [t-d, t+h] span (true LOO; no peeking at the held path)
        excl = (base >= t - d) & (base <= t + h)
        Lk, nk, bk = L[~excl], nxt[~excl], base[~excl]
        hist = list(s[t - d : t][::-1])  # most-recent-first lag vector
        zhat = None
        for _ in range(h):
            xq = np.array(hist[:d], dtype=float)
            dist = np.linalg.norm(Lk - xq, axis=1)
            w = _gauss_w(dist, th, d)
            if w.sum() <= 0:
                zhat = np.nan
                break
            C = np.linalg.lstsq(
                w[:, None] * Lk, (w * nk)[:, None], rcond=None
            )[0]
            zhat = float(xq @ C[:, 0])
            hist = [zhat] + hist  # feed forward, rebuild next lag vec
        if zhat is None or not np.isfinite(zhat):
            continue
        tot += (zhat - s[t + h]) ** 2
        cnt += 1
    return tot / cnt if cnt else np.nan


def main() -> None:
    d = 2
    s = logistic_series(8000)
    horizons = (1, 2, 4, 8, 16)
    n_anchor = 20
    ng = 16
    rng0 = np.random.default_rng(7)
    # origin times t with d lags + max-horizon future available
    valid_t = np.arange(d, len(s) - max(horizons) - 1)
    anchors = rng0.choice(valid_t, n_anchor, replace=False)
    # a representative distance scale for the theta grid: pairwise
    # lag-vector distances on a sample of origins
    L = np.stack([s[i - d : i][::-1] for i in range(d, len(s) - 1)])
    samp = L[np.random.default_rng(1).choice(len(L), 400, replace=False)]
    from scipy.spatial.distance import pdist

    dd = pdist(samp)
    grid = np.geomspace(max(dd.min(), 1e-3), dd.max(), ng)

    curves = {h: np.full((n_anchor, ng), np.nan) for h in horizons}
    for a, t0 in enumerate(anchors):
        for g, th in enumerate(grid):
            for h in horizons:
                curves[h][a, g] = _hstep_loo(s, [int(t0)], th, d, h)

    print(f"logistic d={d}, median over {n_anchor} anchors; "
          f"grid index = bandwidth rank (0=tightest, {ng - 1}=global)")
    print(f"{'idx':>3} " + " ".join(f"{'h=' + str(h):>11}" for h in horizons))
    med = {h: np.nanmedian(curves[h], axis=0) for h in horizons}
    for g in range(ng):
        print(f"{g:>3} " + " ".join(f"{med[h][g]:11.4e}" for h in horizons))

    print("\n  argmin (bandwidth-rank index; INTERIOR if not 0/last):")
    for h in horizons:
        c = med[h]
        bi = int(np.nanargmin(c))
        kind = "INTERIOR-MIN" if 0 < bi < ng - 1 else "edge(monotone)"
        print(f"   h={h:>2}: argmin {bi}/{ng - 1}  {kind}")
    print("\nREAD: hypothesis = h=1 edge/monotone (reproduces failure); "
          "as h grows an INTERIOR minimum appears & theta* moves off "
          "the global ceiling => multi-step CV IS the fix family. If "
          "all h stay edge/monotone => multi-step is NOT the fix; "
          "report and reconsider (plug-in / locality-targeted).")


if __name__ == "__main__":
    main()
