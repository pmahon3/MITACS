"""EXPLORATORY: is the per-anchor theta FIBRE a spatially-organised field?

Background. `mitacs-theta-rail-pinning` established that the per-anchor
LOO-CV *argmin* theta* does not statistically discriminate -- there is no
separated interior optimum. That is an ARGMIN statement. It is silent on
the *shape of the curve* theta |-> C_j(theta) (and theta |-> Sigma_j(theta)).
The "theta as a field over the manifold" proposal does not depend on the
argmin; it depends on whether the FIBRE itself -- the whole operator family
across the theta-grid -- carries spatial information.

This probe runs the two-part gate that decides whether that proposal has a
foundation, BEFORE anything is built on it:

  Gate 1 -- spatial VARIATION. Do per-anchor fibres differ across the
    embedding by more than finite-sample fitting noise? Measured as the
    between-anchor fibre-shape spread, divided by the SAME spread on a
    constant-field VAR(d) at the matching embedding dimension. The VAR(d)
    field is genuinely constant, so its spread is pure WLS variability;
    ratio ~1 => Ontario variation is just that noise, ratio >>1 => fibres
    genuinely vary. (An earlier within-anchor bootstrap null was wrong --
    it measured stability at a FIXED query point, far tighter than true
    cross-anchor variability, and read ratio ~1.55 on a constant field.)

  Gate 2 -- spatial COHERENCE. Variation alone is not enough: for the
    (d+1)-simplex interpolation to be well-posed the field must be SMOOTH
    over the manifold (nearby anchors -> similar fibres). Measured as the
    correlation between fibre-shape dissimilarity and embedding distance,
    calibrated against an anchor-label PERMUTATION null -- shuffling which
    fibre pairs with which coordinate destroys spatial tie while leaving
    every descriptor value intact, so any descriptor leak cancels. The
    z-score of the observed correlation vs the permuted distribution is
    the trustworthy reading.

Fibre-shape is summarised TWO ways (the conclusion is robust iff they
agree): functional-PCA scores across anchors, and a scale-normalised
gradient/curvature profile. Both C_j(theta) and Sigma_j(theta) fibres are
gated -- Sigma_j carries the non-Dirac content of the kappa_Q mapping and
its theta-dependence may be where any spatial signal actually lives.

Method discipline. The fibre is the PRODUCTION fit path with exactly one
substitution: `_theta_loo_cv` (selection) is replaced by a sweep over the
grid. The weights come from the production `_gauss_w`; the drift/diffusion
lines replay `_local_fit_at` verbatim. A bit-for-bit sanity check (fibre
evaluated at the production theta* must equal the production C, Sigma)
confirms the sweep is the production path, not a reimplementation.

PROVENANCE-GRADE: INSPECTION-ONLY -- exploratory dev-set probe; MUST NOT be
cited as a result. Prints the two gate numbers and what each outcome would
mean; bakes in no verdict.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import load_config
from edynamics.modelling_tools import Embedding, Lag
from processing.innovations.estimator import (
    _gauss_w,
    local_drift_and_diffusion,
)

# Dev-set window: the 2021-22 honing window every other theta probe uses.
DEV_START = pd.Timestamp("2021-01-01")
DEV_END = pd.Timestamp("2022-12-31T23:00:00")

N_ANCHOR = 50          # stratified anchor subsample per day-type
N_GRID = 18            # match _theta_loo_cv's grid resolution
SEED = 20260522


# --------------------------------------------------------------------------
# Fibre construction -- production path, theta swept instead of selected.
# --------------------------------------------------------------------------
def _fit_at_theta(X, Y, x_query, d, theta):
    """One (C, Sigma) at an explicit theta -- the body of `_local_fit_at`
    with the `_theta_loo_cv` call replaced by the supplied theta. The three
    lines below are verbatim from estimator._local_fit_at."""
    dists = np.linalg.norm(X - x_query, axis=1)
    w = _gauss_w(dists, theta, d)
    C = np.linalg.lstsq(w[:, None] * X, w[:, None] * Y, rcond=None)[0]
    resid = Y - X @ C
    mu = resid.mean(axis=0)
    rc = resid - mu[None, :]
    Sigma = rc.T @ rc / len(rc)
    return C, Sigma


def _anchor_arrays(embedding, anchor, day_anchor_hour):
    """Reproduce local_drift_and_diffusion's X/Y/x_query derivation
    (incl. the day-anchor seam mask) so the swept fibre uses exactly the
    library the production fit would use at this anchor."""
    d = embedding.block.shape[1]
    block = embedding.block
    blk = block.loc[block.index != anchor]
    X_df, Y_df = blk.iloc[:-1], blk.iloc[1:]
    if day_anchor_hour is not None:
        keep = Y_df.index.hour != day_anchor_hour
        X_df, Y_df = X_df[keep], Y_df[keep]
    x0 = block.loc[anchor].values
    return X_df.values, Y_df.values, x0, d


def anchor_fibre(embedding, anchor, day_anchor_hour, grid):
    """The fibre {C_j(theta), Sigma_j(theta)} over `grid` at one anchor."""
    X, Y, x0, d = _anchor_arrays(embedding, anchor, day_anchor_hour)
    Cs = np.empty((len(grid), d, d))
    Ss = np.empty((len(grid), d, d))
    for k, th in enumerate(grid):
        Cs[k], Ss[k] = _fit_at_theta(X, Y, x0, d, th)
    return Cs, Ss, x0


# --------------------------------------------------------------------------
# Fibre-shape descriptors (two, per the user's request -- agreement = robust)
# --------------------------------------------------------------------------
def descriptor_grad(fibre_flat):
    """Scale-normalised gradient/curvature profile of a fibre.

    `fibre_flat` : (n_grid, p) -- operator entries flattened per theta.
    Returns the concatenated 1st+2nd finite differences along theta, after
    dividing out the fibre's overall Frobenius scale so the descriptor
    captures *shape* (how the operator bends with theta), not magnitude.
    """
    scale = np.linalg.norm(fibre_flat) / np.sqrt(fibre_flat.size)
    f = fibre_flat / (scale + 1e-12)
    g1 = np.diff(f, axis=0)
    g2 = np.diff(f, axis=0, n=2)
    return np.concatenate([g1.ravel(), g2.ravel()])


def descriptor_fpca(fibres_flat, n_comp=4):
    """Functional-PCA scores across anchors.

    `fibres_flat` : (n_anchor, n_grid*p) -- each row one anchor's fibre,
    mean-curve removed. Returns (n_anchor, n_comp) scores on the top modes
    of between-anchor fibre variation.
    """
    centred = fibres_flat - fibres_flat.mean(axis=0, keepdims=True)
    # SVD of the centred fibre matrix == functional PCA on the discretised
    # curves; right singular vectors are the eigen-fibres.
    U, S, _ = np.linalg.svd(centred, full_matrices=False)
    k = min(n_comp, S.size)
    return U[:, :k] * S[:k], S


# --------------------------------------------------------------------------
# Gates
#
# Gate 1 -- VARIATION. The earlier within-anchor bootstrap was the WRONG
# null: it measures fibre stability at a FIXED query point, which is far
# tighter than the genuine cross-anchor variability of finite-sample WLS
# fits. On a constant-field synthetic the bootstrap-ratio read ~1.55, not
# ~1 -- the descriptor/bootstrap mismatch, not real variation. The correct
# null is the between-anchor spread on a CONSTANT-FIELD system at the SAME
# embedding dimension (a VAR(d)). gate1_between() returns the raw spread;
# the caller divides Ontario's by the d-matched VAR(d) null's.
#
# Gate 2 -- COHERENCE. The descriptor leak that broke Gate 1 also feeds
# Gate 2 (same `descriptor_grad`). A raw correlation is therefore not
# trustworthy on its own. gate2_coherence() adds an anchor-label
# PERMUTATION null: shuffling which fibre is paired with which coordinate
# destroys any spatial tie while leaving every descriptor value intact, so
# any leak inflates observed and permuted correlations identically and
# cancels. The z-score of the observed correlation against the permuted
# distribution is the calibrated reading.
# --------------------------------------------------------------------------
def gate1_between(fibres_flat, n_grid):
    """Raw between-anchor fibre-shape spread (mean pairwise descriptor
    distance). Meaningful only as a ratio against the d-matched VAR(d)
    constant-field null -- see the module note above."""
    grad_desc = np.array([descriptor_grad(f.reshape(n_grid, -1))
                          for f in fibres_flat])
    return _mean_pairwise(grad_desc)


def gate2_coherence(fibres_flat, coords, n_grid, rng, n_perm=2000):
    """Coherence of fibre shape with embedding position, permutation-nulled.

    Returns (r_pearson, r_spearman, z_spearman, p_spearman). The z/p come
    from `n_perm` anchor-label shuffles: under no spatial organisation the
    observed correlation sits mid-distribution (z ~ 0). A descriptor leak
    shifts observed and permuted correlations together, so it cancels --
    the calibrated reading, unlike the raw r.
    """
    from scipy.stats import spearmanr

    grad_desc = np.array([descriptor_grad(f.reshape(n_grid, -1))
                          for f in fibres_flat])
    n = len(coords)
    iu = np.triu_indices(n, k=1)
    emb_d = np.array([np.linalg.norm(coords[i] - coords[j])
                      for i, j in zip(*iu)])

    def _fib_dists(desc):
        return np.array([np.linalg.norm(desc[i] - desc[j])
                         for i, j in zip(*iu)])

    fib_d = _fib_dists(grad_desc)
    r_pearson = float(np.corrcoef(fib_d, emb_d)[0, 1])
    r_spearman = float(spearmanr(fib_d, emb_d).statistic)

    perm = np.empty(n_perm)
    for k in range(n_perm):
        sh = rng.permutation(n)
        perm[k] = spearmanr(_fib_dists(grad_desc[sh]), emb_d).statistic
    mu, sd = perm.mean(), perm.std()
    z = float((r_spearman - mu) / sd) if sd > 0 else 0.0
    p = float((np.sum(np.abs(perm - mu) >= abs(r_spearman - mu)) + 1)
              / (n_perm + 1))
    return r_pearson, r_spearman, z, p


def _scale_grid(X, x0, n_grid):
    """Per-anchor scale-aware theta grid -- matches `_theta_loo_cv`'s
    abscissa: geomspace over the actual query-to-library distances."""
    dists = np.linalg.norm(X - x0, axis=1)
    return np.geomspace(max(dists.min(), 1e-3), dists.max(), n_grid)


def _mean_pairwise(D):
    n = len(D)
    tot, cnt = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            tot += np.linalg.norm(D[i] - D[j])
            cnt += 1
    return tot / max(cnt, 1)


# --------------------------------------------------------------------------
# Fibre collection over a set of anchors (shared: Ontario + VAR(d) null)
# --------------------------------------------------------------------------
def collect_fibres(embedding, anchors, dah):
    """Sweep the theta-fibre at every anchor. Returns C-fibres, Sigma-fibres
    (each (n_anchor, N_GRID, p)), anchor coordinates, and the bit-for-bit
    production-path sanity (max |fibre@theta* - production|)."""
    d = embedding.block.shape[1]
    C_fibres, S_fibres, coords = [], [], []
    sanity_max = 0.0
    for anchor in anchors:
        X, Y, x0, _ = _anchor_arrays(embedding, anchor, dah)
        grid = _scale_grid(X, x0, N_GRID)        # scale-aware, per anchor
        Cs, Ss, _ = anchor_fibre(embedding, anchor, dah, grid)
        C_fibres.append(Cs.reshape(N_GRID, -1))
        S_fibres.append(Ss.reshape(N_GRID, -1))
        coords.append(x0)
        # production fit must lie on the swept fibre at theta*
        C_prod, S_prod, _mu, th_star, _r = local_drift_and_diffusion(
            embedding=embedding, anchor=anchor)
        C_at, S_at = _fit_at_theta(X, Y, x0, d, th_star)
        sanity_max = max(sanity_max,
                         np.abs(C_at - C_prod).max(),
                         np.abs(S_at - S_prod).max())
    return (np.array(C_fibres), np.array(S_fibres),
            np.array(coords), sanity_max)


def var_null_between(d, rng):
    """Gate-1 null: between-anchor fibre-shape spread on a CONSTANT-FIELD
    VAR(d) -- every anchor sees identical dynamics, so any spread here is
    pure finite-sample WLS variability. Uses the PRODUCTION validation
    generators (`make_var1_params`, `simulate_var1`, `build_embedding`)
    so the null is on the same code path the recovery gate validates.
    Returns the null between-spread for the C- and Sigma-fibres."""
    from processing.innovations.validation.synthetic import (
        build_embedding, make_var1_params, simulate_var1,
    )
    A, Q = make_var1_params(d, seed=SEED + 100 + d)
    X = simulate_var1(A, Q, n=6000, burn=500, seed=SEED + 200 + d)
    embedding, idx = build_embedding(X)
    lib = idx[1:-1]                       # drop the un-embeddable ends
    m = min(N_ANCHOR, len(lib))
    anchors = pd.DatetimeIndex(
        np.sort(rng.choice(lib, size=m, replace=False)))
    Cf, Sf, _coords, sanity = collect_fibres(embedding, anchors,
                                             dah=None)  # no day-type seam
    return (gate1_between(Cf.reshape(len(Cf), -1), N_GRID),
            gate1_between(Sf.reshape(len(Sf), -1), N_GRID),
            sanity)


def gate_block(name, fibres, coords, null_between, rng):
    """Run both repaired gates on one fibre set and print.

    Gate 1 ratio = between-anchor spread / d-matched VAR(d) null spread.
    Gate 2 = permutation-nulled coherence (z, p)."""
    flat = fibres.reshape(len(fibres), -1)
    between = gate1_between(flat, N_GRID)
    ratio = between / (null_between + 1e-12)
    r_p, r_s, z, p = gate2_coherence(flat, coords, N_GRID, rng)
    _scores, svals = descriptor_fpca(flat)
    var_share = (svals[:3] ** 2 / (svals ** 2).sum()).round(3)
    print(f"  [{name} fibre]")
    print(f"    Gate 1  between / VAR(d)-null spread = {ratio:.2f}"
          f"   (between={between:.3e}, null={null_between:.3e})")
    print(f"    Gate 2  coherence  Spearman={r_s:+.3f}  "
          f"(Pearson={r_p:+.3f})  perm-null z={z:+.2f}  p={p:.3f}")
    print(f"    fPCA    top-3 mode variance share    = {list(var_share)}")
    return dict(between=between, ratio=ratio, pearson=r_p, spearman=r_s,
                z=z, p=p, fpca_var_share=var_share)


def run_daytype(cfg, df, dt, nulls, rng):
    d = cfg.embedding_dim(dt)
    lags = [Lag(variable_name=cfg.data.variable_name, tau=-i)
            for i in range(d)]
    dev = df[(df.index >= DEV_START) & (df.index <= DEV_END)]
    full_lib = dev.index[dev["daytype"] == dt][d:-1]
    embedding = Embedding(data=df, observers=lags, library_times=full_lib)
    embedding.compile()

    n = min(N_ANCHOR, len(full_lib))
    anchors = pd.DatetimeIndex(
        np.sort(rng.choice(full_lib, size=n, replace=False)))
    dah = cfg.data.day_anchor_hours

    print(f"\n[{dt}] d={d}  dev-library={len(full_lib)}  anchors={n}")
    Cf, Sf, coords, sanity = collect_fibres(embedding, anchors, dah)
    print(f"  production-path sanity (max |fibre@theta* - production|): "
          f"{sanity:.2e}  -> {'OK' if sanity < 1e-9 else 'MISMATCH'}")

    cnull, snull = nulls[d]                # d-matched VAR(d) null
    out = {}
    out["C"] = gate_block("C", Cf, coords, cnull, rng)
    out["Sigma"] = gate_block("Sigma", Sf, coords, snull, rng)
    return out


def main():
    cfg = load_config()
    # .asfreq("h") matches processing/innovations/process.py -- the
    # Embedding's Lag observers need an explicit index frequency.
    df = pd.read_csv(cfg.paths.clustered_csv, index_col=0,
                     parse_dates=True).asfreq("h")
    rng = np.random.default_rng(SEED)

    print("=" * 70)
    print("theta-FIBRE spatial-organisation gate  (INSPECTION-ONLY)")
    print(f"dev window {DEV_START.date()}..{DEV_END.date()}  seed={SEED}")
    print("=" * 70)

    # Gate-1 null: a VAR(d) constant-field spread per embedding dimension
    # actually used by the day-types. Also serves as the calibration unit
    # test -- the VAR(d) field IS constant, so its own Gate-1 ratio (null
    # vs null) is 1.00 by construction and Gate-2 z must read ~0.
    dims = sorted({cfg.embedding_dim(dt) for dt in cfg.data.daytypes})
    nulls = {}
    print("\nVAR(d) constant-field null  (Gate-1 denominator + calibration)")
    for d in dims:
        cnull, snull, sanity = var_null_between(d, rng)
        nulls[d] = (cnull, snull)
        print(f"  d={d}: null between-spread  C={cnull:.3e}  "
              f"Sigma={snull:.3e}   production-path sanity {sanity:.2e}")

    for dt in cfg.data.daytypes:
        run_daytype(cfg, df, dt, nulls, rng)

    print("\n" + "=" * 70)
    print("READING THE GATES (no verdict baked in):")
    print("  Gate 1 ratio  : between-anchor spread / VAR(d) constant-field")
    print("                  null. ~1 => variation is just finite-sample")
    print("                  WLS noise; >>1 => fibres genuinely vary.")
    print("  Gate 2 perm-z  : coherence vs anchor-label-shuffle null.")
    print("                  ~0 => fibre shape not tied to embedding")
    print("                  position; large +z => spatially organised.")
    print("  Reading B (interpolate operator fibres over a (d+1)-simplex)")
    print("  needs BOTH a Gate-1 ratio >> 1 AND a Gate-2 z clearly > 0.")
    print("  Gate-1 >>1 with Gate-2 z~0 => fibres vary but unorganised;")
    print("  simplex would interpolate noise. Gate-1 ~1 => constant field.")
    print("=" * 70)


if __name__ == "__main__":
    main()
