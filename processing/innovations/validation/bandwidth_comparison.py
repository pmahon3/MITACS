"""Bandwidth-rule comparison (investigation, not pipeline).

The edynamics GL selector is degenerate for the normalized Gaussian kernel
(no interior optimum; verified). The Resolvent_Framework programme has no
bandwidth-selection convention to defer to (verified). So the rule is a pure
applied-statistics choice. This module compares four candidates so the
choice is evidence-based:

  - knn      : theta/sigma set so the kernel spans ~k effective neighbours
  - cv       : theta by LOO one-step forecast error; sigma by residual
               Gaussian negative-log-likelihood (both well-posed, U-shaped)
  - eps_dmap : Coifman-Lafon epsilon-scaling (log-log slope of sum exp(-d^2/eps))
  - fixed    : median-distance global scale (non-adaptive control; this is
               closest to what the programme's own withdrawn Paper III did)

theta (drift kernel, over neighbour distances) and sigma (residual kernel,
over residual norms given the drift fit) are selected SEPARATELY -- they
fail differently and sigma-selection is conditional on the drift fit.

Run::

    python -m processing.innovations.validation.bandwidth_comparison
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from edynamics.modelling_tools import Embedding, Lag

from config import load_config

from ..estimator import raw_residual_diffusion


# ── kernel (matches edynamics normalized Gaussian) ────────────────────────
def gauss_w(dist: np.ndarray, theta: float, dim: int) -> np.ndarray:
    norm = (2 * np.pi) ** (-dim / 2) * (1.0 / theta**dim)
    return norm * np.exp(-0.5 * (dist / theta) ** 2)


class _K:
    """Minimal residual-kernel shim for raw_residual_diffusion (needs .weigh)."""

    def __init__(self, theta: float, dim: int):
        self.theta, self.dim = theta, dim

    def weigh(self, d):
        return gauss_w(np.asarray(d, float), self.theta, self.dim)


# ── theta selectors (on neighbour distances to the anchor) ────────────────
def theta_knn(dists: np.ndarray, d: int, k: int = 100) -> float:
    """Bandwidth = distance to the k-th nearest neighbour."""
    k = min(k, len(dists) - 1)
    return float(np.partition(dists, k)[k]) or float(np.median(dists))


def theta_fixed(dists: np.ndarray, d: int) -> float:
    """Non-adaptive control: median neighbour distance."""
    return float(np.median(dists))


def theta_eps_dmap(dists: np.ndarray, d: int) -> float:
    """Coifman-Lafon: eps where d/d(log eps) log sum exp(-dist^2/eps) peaks.

    The maximal-slope eps brackets the scale where the kernel transitions
    from local to global; sqrt(eps) is the length-scale bandwidth.
    """
    eps = np.geomspace(
        max(dists.min() ** 2, 1e-6), dists.max() ** 2 + 1e-12, 40
    )
    L = np.array([np.log(np.sum(np.exp(-(dists**2) / e)) + 1e-300) for e in eps])
    slope = np.gradient(L, np.log(eps))
    return float(np.sqrt(eps[int(np.argmax(slope))]))


def theta_cv(
    dists: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    d: int,
    sub: int = 150,
    seed: int = 0,
) -> float:
    """theta minimising TRUE leave-one-out one-step forecast error.

    True LOO (refit C excluding the held-out row) is genuinely U-shaped in
    theta -- verified by a naive-vs-shortcut cross-check. An *in-sample*
    weighted residual is monotone in theta and rail-pins; that earlier
    proxy was wrong. The naive LOO is O(m^2) per theta, so it is evaluated
    on a bounded random subsample of ``sub`` library points (exact, no
    convention-sensitive hat-matrix shortcut).
    """
    n = len(dists)
    rng = np.random.default_rng(seed)
    idx = (
        rng.choice(n, sub, replace=False) if n > sub else np.arange(n)
    )
    Xs, Ys, ds = X[idx], Y[idx], dists[idx]
    m = len(idx)
    grid = np.geomspace(max(ds.min(), 1e-3), ds.max(), 18)
    best = (np.inf, grid[len(grid) // 2])
    for th in grid:
        w = gauss_w(ds, th, d)
        if w.sum() <= 0:
            continue
        tot = 0.0
        for i in range(m):
            keep = np.arange(m) != i
            wi = w[keep]
            Ci = np.linalg.lstsq(
                wi[:, None] * Xs[keep], wi[:, None] * Ys[keep], rcond=None
            )[0]
            tot += np.sum((Ys[i] - Xs[i] @ Ci) ** 2)
        err = tot / m
        if err < best[0]:
            best = (err, th)
    return float(best[1])


# ── sigma selectors (on residual norms given the drift fit) ───────────────
def sigma_knn(rn: np.ndarray, d: int, k: int = 100) -> float:
    k = min(k, len(rn) - 1)
    return float(np.partition(rn, k)[k]) or float(np.median(rn))


def sigma_fixed(rn: np.ndarray, d: int) -> float:
    return float(np.median(rn))


def sigma_eps_dmap(rn: np.ndarray, d: int) -> float:
    return theta_eps_dmap(rn, d)  # same scale rule on residual norms


def sigma_cv(rn: np.ndarray, d: int) -> float:
    """sigma maximising Gaussian predictive likelihood of residual norms."""
    grid = np.geomspace(max(rn.min(), 1e-4), rn.max() + 1e-9, 25)
    best = (np.inf, grid[len(grid) // 2])
    for s in grid:
        g = gauss_w(rn, s, d)
        nll = -np.sum(np.log(g + 1e-300))
        if nll < best[0]:
            best = (nll, s)
    return float(best[1])


RULES = ("knn", "cv", "eps_dmap", "fixed")
_TH = {"knn": theta_knn, "cv": None, "eps_dmap": theta_eps_dmap, "fixed": theta_fixed}
_SG = {"knn": sigma_knn, "cv": sigma_cv, "eps_dmap": sigma_eps_dmap, "fixed": sigma_fixed}


@dataclass
class RuleResult:
    daytype: str
    rule: str
    theta_med: float
    theta_std: float
    sigma_med: float
    sigma_std: float
    drift_norm_med: float
    diff_norm_med: float
    r_hat_med: float
    n_anchors: int


def _build_embedding(cfg, dt: str):
    d = cfg.embedding_dim(dt)
    df = pd.read_csv(
        cfg.paths.clustered_csv, index_col=0, parse_dates=True
    ).asfreq("h")
    lags = [Lag(variable_name=cfg.data.variable_name, tau=-i) for i in range(d)]
    fl = df.index[df["daytype"] == dt][d:-1]
    emb = Embedding(data=df, observers=lags, library_times=fl)
    emb.compile()
    return emb, fl, d


def compare(n_anchors: int = 8, seed: int = 1) -> list[RuleResult]:
    cfg = load_config()
    out: list[RuleResult] = []
    for dt in cfg.data.daytypes:
        emb, fl, d = _build_embedding(cfg, dt)
        block = emb.block
        rng = np.random.default_rng(seed)
        anchors = fl[np.sort(rng.choice(len(fl), min(n_anchors, len(fl)), replace=False))]

        acc = {r: {"th": [], "sg": [], "dn": [], "fn": [], "rh": []} for r in RULES}
        for t in anchors:
            x = block.loc[t].values
            m = block.index != t
            blo = block.loc[m]
            dists = np.linalg.norm(blo.values - x, axis=1)
            X, Y = blo.iloc[:-1].values, blo.iloc[1:].values
            dists = dists[:-1]
            for r in RULES:
                th = (
                    theta_cv(dists, X, Y, d)
                    if r == "cv"
                    else _TH[r](dists, d)
                )
                w = gauss_w(dists, th, d)
                if w.sum() <= 0:
                    continue
                WX, WY = w[:, None] * X, w[:, None] * Y
                C = np.linalg.lstsq(WX, WY, rcond=None)[0]
                rn = np.linalg.norm(Y - X @ C, axis=1)
                sg = _SG[r](rn, d)
                Sigma, _ = raw_residual_diffusion(
                    embedding=emb, anchor=t, C=C, residual_kernel=_K(sg, d)
                )
                acc[r]["th"].append(th)
                acc[r]["sg"].append(sg)
                acc[r]["dn"].append(np.linalg.norm(C))
                acc[r]["fn"].append(np.linalg.norm(Sigma))
                ev = np.linalg.eigvalsh(Sigma)[::-1]
                ev = np.clip(ev, 0, None)
                gaps = ev[:-1] / (ev[1:] + 1e-12)
                acc[r]["rh"].append(int(np.argmax(gaps) + 1) if gaps.size else d)

        for r in RULES:
            a = acc[r]
            if not a["th"]:
                continue
            out.append(
                RuleResult(
                    dt, r,
                    float(np.median(a["th"])), float(np.std(a["th"])),
                    float(np.median(a["sg"])), float(np.std(a["sg"])),
                    float(np.median(a["dn"])), float(np.median(a["fn"])),
                    float(np.median(a["rh"])), len(a["th"]),
                )
            )
    return out


if __name__ == "__main__":
    rows = compare()
    hdr = (
        f"{'daytype':9s} {'rule':9s} {'theta~':>8s} {'th_sd':>7s} "
        f"{'sigma~':>9s} {'sg_sd':>8s} {'||C||~':>8s} {'||S||~':>9s} "
        f"{'r_hat~':>6s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for x in rows:
        print(
            f"{x.daytype:9s} {x.rule:9s} {x.theta_med:8.3f} {x.theta_std:7.3f} "
            f"{x.sigma_med:9.4f} {x.sigma_std:8.4f} {x.drift_norm_med:8.4f} "
            f"{x.diff_norm_med:9.5f} {x.r_hat_med:6.1f}"
        )
