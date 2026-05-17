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

# NOTE: this is a SUPERSEDED investigation artifact. It settled the theta/
# sigma bandwidth questions; the production estimator (estimator.py) now
# implements the conclusion (true-LOO-CV theta, NO residual kernel) and the
# old shared `raw_residual_diffusion` helper was removed with it. The
# kernel-weighted covariance the comparison itself studied is inlined below
# (`_kernel_weighted_cov`) so this record stays runnable and self-contained.


# ── kernel (matches edynamics normalized Gaussian) ────────────────────────
def gauss_w(dist: np.ndarray, theta: float, dim: int) -> np.ndarray:
    norm = (2 * np.pi) ** (-dim / 2) * (1.0 / theta**dim)
    return norm * np.exp(-0.5 * (dist / theta) ** 2)


def _kernel_weighted_cov(
    resid: np.ndarray, sigma: float, dim: int
) -> np.ndarray:
    """Residual-kernel-weighted covariance (the object this study compares)."""
    g = gauss_w(np.linalg.norm(resid, axis=1), sigma, dim)
    mu = np.average(resid, axis=0, weights=g)
    rc = resid - mu[None, :]
    return (rc * g[:, None]).T @ rc / (g.sum() + 1e-12)


class _K:
    """Retained for back-compat of older call sites; unused after inlining."""

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
                resid = Y - X @ C
                rn = np.linalg.norm(resid, axis=1)
                sg = _SG[r](rn, d)
                Sigma = _kernel_weighted_cov(resid, sg, d)
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


# ──────────────────────────────────────────────────────────────────────────
# Sigma-rule comparison: theta FIXED at true-LOO-CV (settled winner), vary
# only how the diffusion covariance is estimated from the resulting
# residuals. Decides whether to keep a residual-kernel bandwidth at all.
# ──────────────────────────────────────────────────────────────────────────
def _sigma_cv_lik_holdout(r: np.ndarray, d: int, seed: int = 0) -> np.ndarray:
    """Pick sigma by HELD-OUT Gaussian log-lik of residual vectors.

    Split residuals; for each sigma estimate Sigma on the train half
    (residual-kernel-weighted), score full multivariate-normal log-lik on
    the test half. Held-out (unlike the older monotone all-data NLL).
    Returns the chosen Sigma on all residuals at the selected sigma.
    """
    rng = np.random.default_rng(seed)
    n = len(r)
    perm = rng.permutation(n)
    tr, te = perm[: n // 2], perm[n // 2 :]
    rn = np.linalg.norm(r, axis=1)
    grid = np.geomspace(max(rn.min(), 1e-4), rn.max() + 1e-9, 20)
    best = (-np.inf, None)
    for s in grid:
        g = gauss_w(np.linalg.norm(r[tr], axis=1), s, d)
        if g.sum() <= 0:
            continue
        mu = np.average(r[tr], axis=0, weights=g)
        rc = r[tr] - mu
        S = (rc * g[:, None]).T @ rc / (g.sum() + 1e-12)
        S += 1e-9 * np.eye(d)
        try:
            sign, logdet = np.linalg.slogdet(S)
            Si = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            continue
        if sign <= 0:
            continue
        rt = r[te] - mu
        ll = -0.5 * (
            np.einsum("ij,jk,ik->i", rt, Si, rt) + logdet + d * np.log(2 * np.pi)
        ).mean()
        if ll > best[0]:
            best = (ll, s)
    s_star = best[1] if best[1] is not None else float(np.median(rn))
    g = gauss_w(rn, s_star, d)
    mu = np.average(r, axis=0, weights=g)
    rc = r - mu
    return (rc * g[:, None]).T @ rc / (g.sum() + 1e-12)


def _sigma_none(r: np.ndarray, d: int) -> np.ndarray:
    """No residual kernel: plain mu-centered covariance (sigma -> infinity)."""
    rc = r - r.mean(axis=0)
    return rc.T @ rc / len(r)


def _sigma_knn_resid(r: np.ndarray, d: int, k: int = 100) -> np.ndarray:
    rn = np.linalg.norm(r, axis=1)
    k = min(k, len(rn) - 1)
    s = float(np.partition(rn, k)[k]) or float(np.median(rn))
    g = gauss_w(rn, s, d)
    mu = np.average(r, axis=0, weights=g)
    rc = r - mu
    return (rc * g[:, None]).T @ rc / (g.sum() + 1e-12)


def _sigma_fixed_med(r: np.ndarray, d: int) -> np.ndarray:
    rn = np.linalg.norm(r, axis=1)
    g = gauss_w(rn, float(np.median(rn)), d)
    mu = np.average(r, axis=0, weights=g)
    rc = r - mu
    return (rc * g[:, None]).T @ rc / (g.sum() + 1e-12)


SIGMA_RULES = {
    "no_kernel": _sigma_none,
    "cv_lik": _sigma_cv_lik_holdout,
    "knn_resid": _sigma_knn_resid,
    "fixed_med": _sigma_fixed_med,
}


def compare_sigma(n_anchors: int = 8, seed: int = 1) -> None:
    """theta fixed at true-LOO-CV; compare Sigma-estimation rules."""
    cfg = load_config()
    hdr = (
        f"{'daytype':9s} {'sigma_rule':11s} {'||S||~':>10s} "
        f"{'minEig~':>10s} {'r_hat~':>7s} {'collapsed?':>10s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for dt in cfg.data.daytypes:
        emb, fl, d = _build_embedding(cfg, dt)
        block = emb.block
        rng = np.random.default_rng(seed)
        anchors = fl[
            np.sort(rng.choice(len(fl), min(n_anchors, len(fl)), replace=False))
        ]
        acc = {r: {"sn": [], "me": [], "rh": []} for r in SIGMA_RULES}
        for t in anchors:
            x = block.loc[t].values
            m = block.index != t
            blo = block.loc[m]
            dists = np.linalg.norm(blo.values - x, axis=1)
            X, Y = blo.iloc[:-1].values, blo.iloc[1:].values
            dists = dists[:-1]
            th = theta_cv(dists, X, Y, d)  # settled theta rule
            w = gauss_w(dists, th, d)
            if w.sum() <= 0:
                continue
            C = np.linalg.lstsq(
                w[:, None] * X, w[:, None] * Y, rcond=None
            )[0]
            resid = Y - X @ C
            for name, fn in SIGMA_RULES.items():
                S = fn(resid, d)
                ev = np.clip(np.linalg.eigvalsh(S)[::-1], 0, None)
                acc[name]["sn"].append(np.linalg.norm(S))
                acc[name]["me"].append(ev[-1])
                gaps = ev[:-1] / (ev[1:] + 1e-12)
                acc[name]["rh"].append(
                    int(np.argmax(gaps) + 1) if gaps.size else d
                )
        for name in SIGMA_RULES:
            a = acc[name]
            if not a["sn"]:
                continue
            sn = float(np.median(a["sn"]))
            me = float(np.median(a["me"]))
            collapsed = "YES" if sn < 1e-4 else "no"
            print(
                f"{dt:9s} {name:11s} {sn:10.5f} {me:10.2e} "
                f"{float(np.median(a['rh'])):7.1f} {collapsed:>10s}"
            )


if __name__ == "__main__":
    import sys

    if "--sigma" in sys.argv:
        compare_sigma()
    else:
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
                f"{x.daytype:9s} {x.rule:9s} {x.theta_med:8.3f} "
                f"{x.theta_std:7.3f} {x.sigma_med:9.4f} {x.sigma_std:8.4f} "
                f"{x.drift_norm_med:8.4f} {x.diff_norm_med:9.5f} "
                f"{x.r_hat_med:6.1f}"
            )
