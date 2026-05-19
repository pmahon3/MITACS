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

‼️ RESULT SUPERSEDED — DOES NOT REPLICATE (2026-05-18, second system
added per advisor pushback #4: a single synthetic was a discipline
violation). Two systems, production-faithful iteration:

  logistic      : h=1 edge(15) -> h=2..16 interior 14,12,8,7
                  (one-step monotone; theta* drifts toward localization)
  regime-switch : h=1 INTERIOR(2) -> h=2..16 interior 2,1,1,6
                  (one-step NOT monotone; theta* ~h-INVARIANT)

The two earlier "findings" are BOTH logistic-specific:
  (1) "one-step CV is monotone/degenerate" — FALSE on regime-switch
      (it has an interior optimum already at h=1).
  (2) "theta* drifts monotonically with h" — FALSE on regime-switch
      (theta* ~h-invariant: ranks 2,2,1,1,6).
The smooth continuously-varying slope of the logistic map (r(1-2x))
interacts with one-step prediction error very differently than the
regime map's discontinuous slope. The kappa_Q-vs-{Pi_t} / theta-drift
theory narrative generalized from logistic SMOOTHNESS, not a general
estimator/localization property. Advisor pushback #4 confirmed
empirically; the theory framing is NOT supported across systems.
NEXT: do NOT build the layer-disentangle synthetic (its premise is
gone). Reframe #41 around what IS robust: one-step CV degeneracy is
NOT universal (regime-switch resolves it) — so the Ontario degeneracy
may be an Ontario-data property, not a pure rule defect. Re-open the
branch-A-vs-B question with this two-system evidence; reconcile w/
advisor.

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


def regime_switch_scalar(n: int, seed: int = 0) -> np.ndarray:
    """Scalar observable of a regime-switching AR process: the AR
    coefficient flips by the sign of the recent value (state-dependent
    dynamics, qualitatively different from the smooth logistic map).
    Second system for replication — a single synthetic was a discipline
    violation (advisor)."""
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = 0.1
    for t in range(1, n):
        a = 0.7 if x[t - 1] > 0 else -0.6  # regime flips by half-line
        x[t] = a * x[t - 1] + 0.15 * rng.standard_normal()
    return x


def _curves_for(s, d, horizons, n_anchor, ng):
    rng0 = np.random.default_rng(7)
    valid_t = np.arange(d, len(s) - max(horizons) - 1)
    anchors = rng0.choice(valid_t, n_anchor, replace=False)
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
    return curves


def main() -> None:
    d = 2
    horizons = (1, 2, 4, 8, 16)
    n_anchor = 20
    ng = 16
    systems = {
        "logistic": logistic_series(8000),
        "regime-switch": regime_switch_scalar(8000),
    }
    for sysname, s in systems.items():
        print(f"\n########## SYSTEM: {sysname} (d={d}) ##########")
        curves = _curves_for(s, d, horizons, n_anchor, ng)
        _report(curves, horizons, ng)


def _report(curves, horizons, ng):
    print(f"  median over anchors; grid index = bandwidth rank "
          f"(0=tightest, {ng - 1}=global)")
    print(f"{'idx':>3} " + " ".join(f"{'h=' + str(h):>11}" for h in horizons))
    med = {h: np.nanmedian(curves[h], axis=0) for h in horizons}
    for g in range(ng):
        print(f"{g:>3} " + " ".join(f"{med[h][g]:11.4e}" for h in horizons))

    print("  argmin (bandwidth-rank index; INTERIOR if not 0/last):")
    for h in horizons:
        bi = int(np.nanargmin(med[h]))
        kind = "INTERIOR-MIN" if 0 < bi < ng - 1 else "edge(monotone)"
        print(f"   h={h:>2}: argmin {bi}/{ng - 1}  {kind}")
    print("  READ: replication check — does the h=1-monotone -> "
          "h>1-interior-min, theta*-drifts-with-h pattern hold on BOTH "
          "systems? If only logistic shows it, the phenomenon is "
          "system-specific and the theory framing rests on one point.")


if __name__ == "__main__":
    main()
