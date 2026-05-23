"""EXPLORATORY: simplex-interpolated theta field as a predictor variant.

Strategy (user-directed). Instead of re-running LOO-CV at every query
point (production `_local_fit_at`), pre-compute the LOO-CV best theta*
at every pre-cutoff library point ONCE -- the frozen theta-field -- then
at predict time:

  1. find the query's (d+1)-simplex of nearest library points (the
     Sugihara simplex; production KD-tree, `Embedding.distance_tree`);
  2. barycentrically interpolate the d+1 stored theta* values into a
     single theta(x) -- exponential simplex weights, Sugihara
     convention, normalised;
  3. ordinary local WLS fit at x with theta(x) -> (C(x), Sigma(x)).

This is Reading A (interpolate the scalar bandwidth, refit the operator)
-- it stays on the kappa_Q layer: C(x) is an honest Gaussian-projected
local operator at x, not a blend of operators fitted at other anchors.

Method discipline. The frozen theta-field IS the production theta*: each
field point's theta* is read straight out of `local_drift_and_diffusion`
(the production estimator), so it carries the production anchor-exclusion
AND the day-anchor seam mask -- no parallel reimplementation, no drift.
Step 3 then calls the production `_smap_w` and replays `_local_fit_at`'s
drift/diffusion lines verbatim; the ONLY change from production is that
theta comes from the simplex field instead of a per-query `_theta_loo_cv`.
At a library point the simplex collapses onto it (zero distance) and
theta(x) returns that point's stored production theta*.

PROVENANCE-GRADE: INSPECTION-ONLY -- exploratory predictor variant; the
gate probe (`scratch/diagnose_theta_fibre.py`) found the theta field
statistically indistinguishable from a constant field, so this is a
direct empirical test of that gate's prediction, NOT a registered or
claim-grade result. MUST NOT be cited as a result.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from scipy.spatial import cKDTree

from edynamics.modelling_tools import Embedding

from processing.innovations.estimator import (
    _smap_w,
    _theta_loo_cv,
    local_drift_and_diffusion,
)


@dataclass
class ThetaField:
    """A frozen theta* field over a fixed library.

    points     : (M, d)  embedded library states (the field's anchors)
    theta_star : (M,)    PRODUCTION theta* at each point
    X, Y       : (L, d)  the one-step library pairs the refit draws on
    tree       : KD-tree over `points`, for simplex neighbour lookup
    d          : embedding dimension
    """
    points: np.ndarray
    theta_star: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    tree: object
    d: int


def _theta_at_anchor_pos(
    block: np.ndarray, anchor_pos: int, d: int,
) -> float:
    """Production theta* at one anchor, from arrays only.

    Replays `local_drift_and_diffusion`: drop the anchor row from the
    block, form one-step (X, Y) pairs, `_theta_loo_cv`. Array-only so
    it can run in a worker without shipping the heavy `Embedding`. The
    target-hour seam mask was dropped 2026-05-22 (see estimator.py and
    memory mitacs-realignment); only the anchor-row exclusion remains.
    """
    keep_row = np.ones(len(block), dtype=bool)
    keep_row[anchor_pos] = False              # drop the anchor row
    blk = block[keep_row]
    X = blk[:-1]
    Y = blk[1:]
    x0 = block[anchor_pos]
    dists = np.linalg.norm(X - x0, axis=1)
    return float(_theta_loo_cv(dists, X, Y, d))


def _theta_chunk(args):
    """Worker: production theta* for a contiguous chunk of anchor
    positions. Returns (positions, theta* values)."""
    block, positions, d = args
    out = np.empty(len(positions))
    for k, pos in enumerate(positions):
        out[k] = _theta_at_anchor_pos(block, pos, d)
    return positions, out


def build_theta_field(
    embedding: Embedding,
    anchors: pd.DatetimeIndex,
    pool=None,
    chunk_size: int = 256,
) -> ThetaField:
    """Freeze the production theta* at every field anchor.

    theta* at each anchor is the production selection rule -- the
    array-only `_theta_at_anchor_pos` replays `local_drift_and_diffusion`
    (anchor-row drop + `_theta_loo_cv`), so the field is the production
    theta* by construction.

    `pool` (a `ray.util.multiprocessing.Pool` or any `.map`-capable pool)
    parallelises the per-anchor LOO-CV -- embarrassingly parallel, ~10x on
    a 10-core box. Workers receive compact arrays only; the heavy
    `Embedding` never crosses the IPC boundary. `pool=None` runs serial.

    The library (X, Y) for the predict-time refit is the full embedding
    block one-step pairs -- the same arrays production WLS draws on.
    """
    block_df = embedding.block
    d = block_df.shape[1]
    block = block_df.values
    X = block[:-1]
    Y = block[1:]

    # anchor row positions in the block
    pos_of = {ts: i for i, ts in enumerate(block_df.index)}
    anchor_pos = np.array([pos_of[ts] for ts in anchors])
    pts = block[anchor_pos]

    if pool is None:
        theta_star = np.array([
            _theta_at_anchor_pos(block, p, d) for p in anchor_pos
        ])
    else:
        chunks = [anchor_pos[i:i + chunk_size]
                  for i in range(0, len(anchor_pos), chunk_size)]
        tasks = [(block, ch, d) for ch in chunks]
        theta_star = np.empty(len(anchor_pos))
        # map preserves order; positions returned for an explicit scatter
        base = 0
        for positions, vals in pool.map(_theta_chunk, tasks):
            theta_star[base:base + len(vals)] = vals
            base += len(vals)

    tree = cKDTree(pts)
    return ThetaField(points=pts, theta_star=theta_star,
                      X=X, Y=Y, tree=tree, d=d)


def interpolate_theta(field: ThetaField, x_query: np.ndarray) -> float:
    """theta(x) from the (d+1)-simplex of nearest field points.

    Sugihara simplex weights: w_k = exp(-dist_k / dist_min), normalised.
    At a field point dist_min -> 0 and the simplex collapses onto it, so
    theta(x) -> that point's stored theta* (the bit-for-bit check).
    """
    k = field.d + 1
    dists, idx = field.tree.query(x_query, k=k)
    dists = np.atleast_1d(dists).astype(float)
    idx = np.atleast_1d(idx)
    dmin = dists.min()
    if dmin <= 1e-12:                       # query coincides with a point
        return float(field.theta_star[idx[dists.argmin()]])
    w = np.exp(-dists / dmin)
    w /= w.sum()
    return float(np.dot(w, field.theta_star[idx]))


def _fit_given_theta(
    X: np.ndarray, Y: np.ndarray, x_query: np.ndarray, d: int, theta: float
) -> tuple[np.ndarray, np.ndarray]:
    """(C, Sigma) at x_query for an explicit theta -- the drift/diffusion
    lines verbatim from estimator._local_fit_at, with theta supplied
    instead of `_theta_loo_cv`-selected. The shared fit core for both the
    simplex-interpolated and the fixed-theta predictors."""
    del d  # S-map kernel is dimension-independent; argument retained
    dists = np.linalg.norm(X - x_query, axis=1)
    w = _smap_w(dists, theta)
    C = np.linalg.lstsq(w[:, None] * X, w[:, None] * Y, rcond=None)[0]
    resid = Y - X @ C
    mu = resid.mean(axis=0)
    rc = resid - mu[None, :]
    Sigma = rc.T @ rc / len(rc)
    return C, Sigma


def fit_at_simplex_theta(
    field: ThetaField, x_query: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """(C, Sigma, theta) at x_query using the simplex-interpolated theta."""
    theta = interpolate_theta(field, x_query)
    C, Sigma = _fit_given_theta(field.X, field.Y, x_query, field.d, theta)
    return C, Sigma, theta


def fit_at_fixed_theta(
    X: np.ndarray, Y: np.ndarray, x_query: np.ndarray, d: int, theta: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """(C, Sigma, theta) at x_query for a single FIXED theta -- no
    per-query selection, no field. The falsification control: if a fixed
    theta recovers the simplex predictor's multi-step gain, the gain is a
    bandwidth-LEVEL effect, not a localisation or field effect."""
    C, Sigma = _fit_given_theta(X, Y, x_query, d, theta)
    return C, Sigma, theta


def fit_global_ols(
    X: np.ndarray, Y: np.ndarray, x_query: np.ndarray, d: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """(C, Sigma, theta) at x_query with w_i = 1 for all i -- the literal
    global unweighted OLS fit. This is the limit the wide-theta backtests
    were asymptoting toward; running it directly gives the asymptote's
    value at a single, achievable point (no theta sweep needed). Returns
    theta=0.0 as a sentinel meaning 'uniform weights'.

    No state-dependence at all: the same (C, Sigma) is produced at every
    query state. So `x_query` is unused except as a signature match.
    """
    del x_query  # global OLS: query state irrelevant to the fit
    C = np.linalg.lstsq(X, Y, rcond=None)[0]
    resid = Y - X @ C
    mu = resid.mean(axis=0)
    rc = resid - mu[None, :]
    Sigma = rc.T @ rc / len(rc)
    return C, Sigma, 0.0
