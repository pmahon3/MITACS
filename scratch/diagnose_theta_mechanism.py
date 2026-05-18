"""EXPLORATORY: WHY is one-step LOO prediction error monotone in theta
on our embedded systems? (Mechanism step for #41 — investigate before
choosing a fix; user-directed 2026-05-18.)

Established (diagnose_theta_locality_probe.py): the production
_theta_loo_cv score curve is monotone-decreasing in theta to the grid
edge even on the logistic map / regime-switching VAR where locality is
provably required. Question: which mechanism produces the monotonicity?

Candidate mechanisms, each with a discriminating test on the LOGISTIC
map (provably state-dependent; a competent criterion MUST show an
interior optimum there):

  M1  delay-embedding temporal autocorrelation: consecutive embedded
      points overlap in time, so plain LOO never truly holds the point
      out — its near-duplicate temporal neighbours stay in the fit at
      ANY theta; larger theta just pools more correlated points and
      lowers variance with no real OOS penalty.
      TEST: PURGED LOO (exclude a temporal window +-g around the held
      -out index). If a minimum appears => M1 is the mechanism.

  M2  no model-complexity term across bandwidths: WLS local-linear has
      the same d params at every theta; bigger theta strictly cuts fit
      variance, one-step squared error never charges the locality bias.
      TEST: add a locality-bias-aware term (compare LOO of the local
      linear vs a global linear baseline; an EXCESS-over-global score).
      If excess score is U-shaped while raw is monotone => M2 visible.

  M3  pooled metric mixes regimes: averaging sq-err over all anchors
      can be monotone even if a per-locale criterion is not.
      TEST: restrict scoring to a single small neighbourhood.

The winning mechanism dictates the fix family (purged-CV vs
bias-aware/plug-in vs local criterion) — do NOT pre-pick the remedy.

RESULT (2026-05-18, median over 20 anchors, d=2 AND d=3, advisor-
directed multi-anchor + model-class discrimination):
  * M1 (delay-embed temporal leakage): RULED OUT — purged LOO
    (±24 window) is identical to raw; both monotone to edge.
  * Branch-1 (single-anchor artifact): RULED OUT — median over 20
    anchors has the same monotone-to-edge shape.
  * Branch-2 (local-LINEAR slope absorbs locality, NW would differ):
    RULED OUT — local-CONSTANT (Nadaraya-Watson, no slope param) is
    ALSO monotone to edge (0.339→0.199 d=2; 0.572→0.296 d=3).
  * Branch-3 CONFIRMED: one-step squared error is intrinsically
    monotone in bandwidth here for ANY local model class; excess-over
    -global never goes negative (local NEVER beats global at one step;
    min excess ~5e-3..7e-3).
=> The criterion is the wrong OBJECTIVE for a localization bandwidth,
independent of model class / argmin / temporal structure. Minimizing
one-step prediction MSE correctly drives theta→global because at one
step more data always helps and the locality bias never dominates
one-step MSE. Fix family = plug-in / bias-aware / multi-step or
locality-targeted criterion (NOT purged-CV, NOT a model-class swap —
both empirically excluded). Fix itself NOT pre-decided.

PROVENANCE-GRADE: INSPECTION-ONLY — exploratory dev mechanism probe;
calls the production _gauss_w. Decided the fix FAMILY for #41 across
proper variation; the specific fix is a separate step.
"""
from __future__ import annotations

import numpy as np

from processing.innovations.estimator import _gauss_w
from scratch.diagnose_theta_locality_probe import (
    _delay_embed,
    logistic_series,
)


def _loo_score(
    Xs: np.ndarray,
    Ys: np.ndarray,
    ds: np.ndarray,
    th: float,
    d: int,
    purge: int = 0,
    order_idx: np.ndarray | None = None,
) -> float:
    """Mean held-out one-step sq err at bandwidth th. ``purge`` excludes
    a +-purge window in ORIGINAL time order around each held-out row
    (requires order_idx = original indices of the subsample rows)."""
    w = _gauss_w(ds, th, d)
    m = len(Xs)
    tot = 0.0
    for i in range(m):
        keep = np.ones(m, dtype=bool)
        keep[i] = False
        if purge and order_idx is not None:
            near = np.abs(order_idx - order_idx[i]) <= purge
            keep &= ~near
        if keep.sum() < d + 1:
            return np.nan
        wi = w[keep]
        Ci = np.linalg.lstsq(
            wi[:, None] * Xs[keep], wi[:, None] * Ys[keep], rcond=None
        )[0]
        tot += np.sum((Ys[i] - Xs[i] @ Ci) ** 2)
    return tot / m


def _nw_loo(Xs, Ys, ds, th, d, purge=0, order_idx=None):
    """LOO mean sq err for LOCAL CONSTANT (Nadaraya-Watson): the held
    -out prediction is the kernel-weighted mean of neighbour targets
    (no slope param, so it cannot absorb the local linear trend the way
    local-linear can). Discriminates advisor branch 2."""
    w = _gauss_w(ds, th, d)
    m = len(Xs)
    tot = 0.0
    for i in range(m):
        keep = np.ones(m, dtype=bool)
        keep[i] = False
        if purge and order_idx is not None:
            keep &= ~(np.abs(order_idx - order_idx[i]) <= purge)
        ww = w[keep]
        sw = ww.sum()
        if sw <= 0:
            return np.nan
        pred = (ww[:, None] * Ys[keep]).sum(0) / sw
        tot += np.sum((Ys[i] - pred) ** 2)
    return tot / m


def _shape(c):
    c = np.asarray(c)[~np.isnan(c)]
    if len(c) < 3:
        return "degenerate(too few)"
    bi = int(np.argmin(c))
    return (f"argmin {bi}/{len(c) - 1} "
            f"{'INTERIOR-MIN' if 0 < bi < len(c) - 1 else 'edge(monotone)'}")


def main() -> None:
    # advisor: median over MANY anchors, not one; and local-constant vs
    # local-linear to test whether the linear slope absorbs locality.
    for d in (2, 3):
        s = logistic_series(6000)
        X, Y = _delay_embed(s, d)
        n_anchor = 20
        rng0 = np.random.default_rng(7)
        anchors = rng0.choice(len(X), n_anchor, replace=False)
        ng = 18
        raw_M = np.full((n_anchor, ng), np.nan)
        pur_M = np.full((n_anchor, ng), np.nan)
        exc_M = np.full((n_anchor, ng), np.nan)
        nw_M = np.full((n_anchor, ng), np.nan)
        for a, anchor in enumerate(anchors):
            q = X[anchor].copy()
            dists = np.linalg.norm(X - q, axis=1)
            sub = 150
            sid = np.random.default_rng(a).choice(
                len(dists), sub, replace=False
            )
            Xs, Ys, ds = X[sid], Y[sid], dists[sid]
            grid = np.geomspace(max(ds.min(), 1e-3), ds.max(), ng)
            Cg = np.linalg.lstsq(Xs, Ys, rcond=None)[0]
            glob = float(np.mean(
                [np.sum((Ys[i] - Xs[i] @ Cg) ** 2) for i in range(len(Xs))]
            ))
            for g, th in enumerate(grid):
                raw = _loo_score(Xs, Ys, ds, th, d)
                raw_M[a, g] = raw
                pur_M[a, g] = _loo_score(
                    Xs, Ys, ds, th, d, purge=24, order_idx=sid
                )
                exc_M[a, g] = raw - glob
                nw_M[a, g] = _nw_loo(Xs, Ys, ds, th, d)
        # median curve across anchors (grid differs per anchor, but
        # index position = relative bandwidth rank, which is the shape
        # we care about)
        raw_med = np.nanmedian(raw_M, axis=0)
        pur_med = np.nanmedian(pur_M, axis=0)
        exc_med = np.nanmedian(exc_M, axis=0)
        nw_med = np.nanmedian(nw_M, axis=0)
        print(f"=== logistic d={d}, median over {n_anchor} anchors "
              f"(grid index = bandwidth rank) ===")
        print(f"{'idx':>3} {'rawLL':>11} {'purgedLL':>11} "
              f"{'exc/glob':>11} {'NW(const)':>11}")
        for g in range(ng):
            print(f"{g:>3} {raw_med[g]:11.4e} {pur_med[g]:11.4e} "
                  f"{exc_med[g]:11.3e} {nw_med[g]:11.4e}")
        print(f"  local-LINEAR raw : {_shape(raw_med)}")
        print(f"  local-LINEAR purg: {_shape(pur_med)}")
        print(f"  excess-over-glob : {_shape(exc_med)} "
              f"(min {np.nanmin(exc_med):.2e}; <0 => local beats global)")
        print(f"  local-CONSTANT   : {_shape(nw_med)}  "
              f"(interior-min here but not for linear => model-class, "
              f"not criterion)\n")
    print("READ (advisor branches): both LL & NW monotone across "
          "anchors => criterion wrong for any localization (plug-in/"
          "bias-aware fix). NW interior-min but LL monotone => model-"
          "class: linear slope absorbs locality (fix = select via NW "
          "or penalize slope variance). exc<0 at mid-theta => single-"
          "anchor result was misleading; criterion may work on median.")


if __name__ == "__main__":
    main()
