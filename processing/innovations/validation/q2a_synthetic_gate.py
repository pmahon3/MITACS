"""Q2A synthetic gate — derive chi^2 cuts from sample-size-matched null.

Pre-experiment calibration for the distributional-class thread's Q2A
node ("richer-family-mixture-or-nonparametric";
``notes/preregistrations/2026-05-26_distributional-class-thread/thread.yaml``).
The output cuts (R-A2_cut, R-C2_cut, R-B2 band) become the
pre-registered chi^2 thresholds in Q2A's phase_a.yaml. The gate
STOPS at producing these cuts — it does NOT touch any Ontario
post-cutoff data, propose phase_a content, or render verdicts.

Pre-registered gate parameters (user-confirmed 2026-05-26):

  KURT_SWEEP     = (12, 20, 28, 40)  Pearson kurtosis (Gaussian = 3).
  K_REPS         = 500               synthetic replications per cell.
  M_PARTICLES    = 200               matches Q1 SMC.
  N_PIT_BINS     = 10                matches Q1.
  FAMILIES       = ("mixture_2", "mixture_3", "kde").

Two passes per (family, kurt):

  Same-family : library drawn AND fit/scored with the same family.
                95th percentile of the resulting chi^2 distribution is
                the upper edge of "calibrated under correct family".
  Cross-family: library drawn from family X, fit/scored with family Y
                (6 ordered pairs per kurt). 99th percentile of the
                resulting chi^2 distribution is the lower edge of
                "discriminably mis-specified".

Cuts:

  R-A2 cut = max over (family, kurt) of: 95th-pct same-family chi^2
  R-C2 cut = max(10 * R-A2_cut, 99th-pct cross-family chi^2)
  R-B2 band = (R-A2_cut, R-C2_cut)

Code-path discipline (CLAUDE.md §"Discipline rule"): production
fitters are the three new estimator entry points
(``mixture_2_gaussian_mle_fit``, ``mixture_3_gaussian_mle_fit``,
``kde_residual_fit``) — never reimplemented inline here. The library
generator uses a hardcoded ``C_TRUE`` literal so it avoids the
``np.linalg.lstsq``/inline-covariance audit traps; all OLS happens
inside the fitters.

Generator parameterisations (deviation from task spec, justified by
the kurtosis algebra ``Pearson kurt = 3 * sum(w_k s_k^4) / (sum(w_k
s_k^2))^2``):

* mix-2 generator: ``w = (0.95, 0.05)``, ``s1 = 1``, ``s2`` solved per
  kurt. The asymptotic ceiling ``3/w_min = 60`` covers the sweep
  (max kurt 40). The task spec's ``w2 = 0.3`` caps kurt at ``3/0.3 =
  10``, infeasible for the registered sweep; the smaller w2 carries
  the same "heavy-tail-from-a-rare-component" shape.
* mix-3 generator: ``w = (0.8, 0.15, 0.05)``, ``s1 = 1``, ``s2 = 2``,
  ``s3`` solved per kurt. Same ceiling ``3/0.05 = 60``. Distinguishable
  from mix-2 by the intermediate component.
* KDE generator: 10,000 samples from the kurt-matched mix-2 above,
  with KDE bandwidth = Silverman on those samples. Closest to the
  spec's "non-parametric residual law".

Closed-form derivations are in ``_solve_mix2_s2`` and
``_solve_mix3_s3`` docstrings.

Run standalone::

    python -m processing.innovations.validation.q2a_synthetic_gate
    python -m processing.innovations.validation.q2a_synthetic_gate --emit-result
"""
from __future__ import annotations

import argparse
import multiprocessing
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import ray

# Validate the *production* fitters, not copies. See CLAUDE.md
# "Discipline rule": a diagnostic that reimplements production logic
# is a hypothesis, not a finding. The three fitters below are the
# whitelisted (experiment/audit/code_path.py) Q2A entry points.
from processing.innovations.estimator import (
    kde_residual_fit,
    mixture_2_gaussian_mle_fit,
    mixture_3_gaussian_mle_fit,
)

# ─── Pre-registered gate parameters ─────────────────────────────────────────
KURT_SWEEP: tuple[float, ...] = (12.0, 20.0, 28.0, 40.0)
K_REPS: int = 500
M_PARTICLES: int = 200
N_PIT_BINS: int = 10
FAMILIES: tuple[str, ...] = ("mixture_2", "mixture_3", "kde")
N_LIBRARY: int = 5000

# Synthetic trajectory length derived from the registered Ontario
# post-cutoff window: cutoff = 2024-12-31T23:00:00 (per
# ``experiment.freeze.load_verified()``), "today" reference =
# 2026-05-26 (matches the project's currentDate at gate-design time),
# so days = 510 -> hours = 12240. The gate uses this as its per-cell
# sample size so the chi^2 null matches the Q2A test's sample size.
# Hardcoded to keep the gate hermetic (no actuals dependency).
SYNTHETIC_TRAJECTORY_LEN: int = 12240   # = 510 days * 24 hours

RNG_SEED_BASE: int = 20260526

# Drift literal (no lstsq used in the gate — keeps the code-path
# audit clean). Chosen small + diagonal-dominated so the synthetic
# regression is well-conditioned.
D_EMBED: int = 2
C_TRUE: np.ndarray = np.array([[0.7], [0.2]], dtype=float)

# mix-2 generator weights / mix-3 generator weights / mix-3 fixed s2
_MIX2_W2: float = 0.05      # rare heavy component
_MIX3_W: tuple[float, float, float] = (0.80, 0.15, 0.05)
_MIX3_S2: float = 2.0

# Stable enumeration for deterministic per-cell seeds. Python's built-in
# ``hash`` randomises per interpreter session (PYTHONHASHSEED), so
# ``hash((library_kind, fit_kind, int(kurt), rep))`` is NON-reproducible
# across runs and a provenanced artifact derived from it would not be
# reproducible from its recorded ``git_sha``. We encode the cell tuple
# into a deterministic 32-bit integer instead.
_FAMILY_ORD: dict[str, int] = {"mixture_2": 0, "mixture_3": 1, "kde": 2}


def _cell_seed(library_kind: str, fit_kind: str, kurt: float, rep: int) -> int:
    """Deterministic seed for the ``(library, fit, kurt, rep)`` cell.

    Encoded so the four axes occupy disjoint bit ranges -- no aliasing
    across cells, deterministic across interpreters.
    """
    lib_i = _FAMILY_ORD[library_kind]
    fit_i = _FAMILY_ORD[fit_kind]
    kurt_i = int(kurt)
    # field widths: lib=2 bits, fit=2 bits, kurt=8 bits (0..255), rep=20 bits (0..1M)
    code = (lib_i << 30) | (fit_i << 28) | ((kurt_i & 0xFF) << 20) | (rep & 0xFFFFF)
    return (RNG_SEED_BASE + code) & 0xFFFFFFFF


# ─── kurtosis-to-scale closed forms ─────────────────────────────────────────
def _solve_mix2_s2(K: float, w2: float = _MIX2_W2, s1: float = 1.0) -> float:
    """Closed-form ``s2`` for a centred mix-2 with target Pearson kurt ``K``.

    Pearson kurt of a centred mixture of zero-mean Gaussians

        K = E[r^4] / E[r^2]^2
          = 3 * sum_k w_k * s_k^4 / (sum_k w_k * s_k^2)^2.

    Setting ``r = s2^2`` and ``w1 = 1 - w2``:

        K * (w1 + w2 r)^2 = 3 (w1 + w2 r^2)

    is quadratic in ``r``; take the heavy-tailed root (``r > s1^2``).
    """
    w1 = 1.0 - w2
    a = K * w2 * w2 - 3 * w2
    b = 2 * K * w1 * w2
    c = K * w1 * w1 - 3 * w1
    disc = b * b - 4 * a * c
    if disc < 0:
        raise ValueError(f"infeasible kurt {K} for w2={w2}: discriminant<0")
    r_hi = (-b + np.sqrt(disc)) / (2 * a)
    r_lo = (-b - np.sqrt(disc)) / (2 * a)
    for r in (r_hi, r_lo):
        if r > s1 * s1:
            return float(np.sqrt(r))
    raise ValueError(f"no heavy-tail root for kurt={K}, w2={w2}")


def _solve_mix3_s3(
    K: float,
    w: tuple[float, float, float] = _MIX3_W,
    s12: tuple[float, float] = (1.0, _MIX3_S2),
) -> float:
    """Closed-form ``s3`` for a centred mix-3 with target Pearson kurt ``K``,
    pinning ``s1`` and ``s2``.

    Setting ``q = s3^2``, ``a0 = w1 s1^2 + w2 s2^2``, ``b0 = w1 s1^4 +
    w2 s2^4``:

        K * (a0 + w3 q)^2 = 3 (b0 + w3 q^2)

    is quadratic in ``q``; take the root with ``q > s2^2`` (the heavy
    tail).
    """
    s1, s2 = s12
    w1, w2, w3 = w
    a0 = w1 * s1 * s1 + w2 * s2 * s2
    b0 = w1 * s1 ** 4 + w2 * s2 ** 4
    A = K * w3 * w3 - 3 * w3
    B = 2 * K * a0 * w3
    C_ = K * a0 * a0 - 3 * b0
    disc = B * B - 4 * A * C_
    if disc < 0:
        raise ValueError(f"infeasible kurt {K} for w={w}, s12={s12}")
    q_hi = (-B + np.sqrt(disc)) / (2 * A)
    q_lo = (-B - np.sqrt(disc)) / (2 * A)
    for q in (q_hi, q_lo):
        if q > s2 * s2:
            return float(np.sqrt(q))
    raise ValueError(f"no heavy-tail root for kurt={K}")


# ─── true-family residual samplers ──────────────────────────────────────────
@dataclass(frozen=True)
class TrueFamily:
    """Generator parameters for one (family, kurt) combination.

    ``draw`` returns ``n`` i.i.d. centred residual samples from the
    population the gate calls the "true" law for that cell. The fitted
    family is the same name for the same-family pass; a *different*
    family name on the cross-family pass.
    """
    kind: str
    kurt_target: float
    # mix-2 params
    mix2_s2: float
    # mix-3 params
    mix3_s3: float
    # KDE bandwidth (set after sampling from mix-2 at the kurt target)
    kde_bandwidth: float
    kde_samples: np.ndarray   # frozen 10k samples used as the KDE reservoir


def _build_true_family(kind: str, kurt: float, rng: np.random.Generator) -> TrueFamily:
    s2 = _solve_mix2_s2(kurt)
    s3 = _solve_mix3_s3(kurt)
    # KDE reservoir: 10k samples from the kurt-matched mix-2; the
    # KDE's bandwidth is Silverman on those samples (the KDE produces
    # a smoothed version of the kurt-matched mix-2).
    kde_n = 10_000
    which = rng.choice(2, size=kde_n, p=[1.0 - _MIX2_W2, _MIX2_W2])
    z = rng.standard_normal(kde_n)
    kde_samples = np.where(which == 0, z * 1.0, z * s2)
    std = float(np.std(kde_samples, ddof=1))
    q25, q75 = np.quantile(kde_samples, [0.25, 0.75])
    iqr = float(q75 - q25)
    spread = min(std, iqr / 1.34) if iqr > 0 else std
    kde_bw = 0.9 * max(spread, 1e-9) * (kde_n ** (-1.0 / 5.0))
    return TrueFamily(
        kind=kind,
        kurt_target=kurt,
        mix2_s2=s2,
        mix3_s3=s3,
        kde_bandwidth=kde_bw,
        kde_samples=kde_samples,
    )


def _draw_residuals(tf: TrueFamily, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw n i.i.d. residuals from the named TRUE family."""
    if tf.kind == "mixture_2":
        which = rng.choice(2, size=n, p=[1.0 - _MIX2_W2, _MIX2_W2])
        z = rng.standard_normal(n)
        return np.where(which == 0, z * 1.0, z * tf.mix2_s2)
    if tf.kind == "mixture_3":
        w1, w2, w3 = _MIX3_W
        which = rng.choice(3, size=n, p=[w1, w2, w3])
        z = rng.standard_normal(n)
        scales = np.array([1.0, _MIX3_S2, tf.mix3_s3])
        return z * scales[which]
    if tf.kind == "kde":
        # Convolution rule: pick a reservoir sample + bandwidth-scaled
        # Gaussian noise. This is exactly the density the KDE encodes.
        idx = rng.integers(0, len(tf.kde_samples), size=n)
        return tf.kde_samples[idx] + tf.kde_bandwidth * rng.standard_normal(n)
    raise ValueError(f"unknown family kind {tf.kind!r}")


# ─── predictive sampler under the FITTED family ─────────────────────────────
def _sample_under_fitted(
    fit_kind: str,
    fit_params: dict,
    mean: float,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw n predictive samples for a state with conditional mean ``mean``,
    using the fitted family params.

    For the centred mixtures, the next-step value is ``mean + r`` with
    ``r`` from the fitted mixture. For KDE, ``mean + r`` with ``r``
    from the fitted convolution (samples + bandwidth * N(0,1)).
    """
    if fit_kind == "mixture_2":
        w = fit_params["weights"]
        s = fit_params["scales"]
        which = rng.choice(2, size=n, p=[w[0], w[1]])
        z = rng.standard_normal(n)
        return mean + np.where(which == 0, z * s[0], z * s[1])
    if fit_kind == "mixture_3":
        w = fit_params["weights"]
        s = fit_params["scales"]
        which = rng.choice(3, size=n, p=list(w))
        z = rng.standard_normal(n)
        scales_arr = np.array(s)
        return mean + z * scales_arr[which]
    if fit_kind == "kde":
        samples = fit_params["samples"]
        bw = fit_params["bandwidth"]
        idx = rng.integers(0, len(samples), size=n)
        return mean + samples[idx] + bw * rng.standard_normal(n)
    raise ValueError(f"unknown fit kind {fit_kind!r}")


# ─── one fitter call routed by name ─────────────────────────────────────────
def _fit_family(kind: str, X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, dict]:
    """Dispatch to the production fitter for the named family."""
    if kind == "mixture_2":
        C, params, _ = mixture_2_gaussian_mle_fit(X, Y)
        return C, {"weights": params["weights"], "scales": params["scales"]}
    if kind == "mixture_3":
        C, params, _ = mixture_3_gaussian_mle_fit(X, Y)
        return C, {"weights": params["weights"], "scales": params["scales"]}
    if kind == "kde":
        C, params, _ = kde_residual_fit(X, Y)
        return C, {
            "samples": params["samples"],
            "bandwidth": params["bandwidth"],
            "normalization": params["normalization"],
        }
    raise ValueError(f"unknown family kind {kind!r}")


# ─── one replication: build library, fit, score PIT chi^2 ───────────────────
def _one_rep(
    library_kind: str,
    fit_kind: str,
    kurt: float,
    rep: int,
) -> dict:
    """One synthetic replication.

    Build a library (X, Y) drawn from family ``library_kind`` at the
    target kurt; fit family ``fit_kind`` to it via the production
    fitter; for each of N_TRAJ test states, draw an actual next-step
    from the TRUE library family (i.e. NOT from the fitted family)
    and a sample of M predictive draws from the FITTED family; PIT =
    randomized mid-rank. Aggregate to a 10-bin chi^2 vs uniform.

    Returns a dict with the chi^2 and bin counts.
    """
    seed = _cell_seed(library_kind, fit_kind, kurt, rep)
    rng = np.random.default_rng(seed)

    # 1) Build library
    true_lib = _build_true_family(library_kind, kurt, rng)
    X_lib = rng.standard_normal((N_LIBRARY, D_EMBED))
    r_lib = _draw_residuals(true_lib, N_LIBRARY, rng)
    Y_lib = (X_lib @ C_TRUE).ravel() + r_lib

    # 2) Fit the (possibly different) family via production
    C_hat, fit_params = _fit_family(fit_kind, X_lib, Y_lib)

    # 3) Test states + PIT loop
    X_test = rng.standard_normal((SYNTHETIC_TRAJECTORY_LEN, D_EMBED))
    means_pred = (X_test @ C_hat).ravel()
    # Actuals drawn from the TRUE library family at the same states'
    # true conditional mean (i.e. X_test @ C_TRUE)
    true_means = (X_test @ C_TRUE).ravel()
    actuals = true_means + _draw_residuals(true_lib, SYNTHETIC_TRAJECTORY_LEN, rng)

    pit = np.empty(SYNTHETIC_TRAJECTORY_LEN)
    for i in range(SYNTHETIC_TRAJECTORY_LEN):
        particles = _sample_under_fitted(fit_kind, fit_params, means_pred[i], M_PARTICLES, rng)
        less = float(np.sum(particles < actuals[i]))
        eq = float(np.sum(particles == actuals[i]))
        U = rng.random()
        pit[i] = (less + U * (eq + 1.0)) / (M_PARTICLES + 1.0)

    # 4) 10-bin chi^2 vs uniform
    bins = np.linspace(0.0, 1.0, N_PIT_BINS + 1)
    hist, _ = np.histogram(pit, bins=bins)
    expected = SYNTHETIC_TRAJECTORY_LEN / N_PIT_BINS
    chi2 = float(((hist - expected) ** 2 / expected).sum())
    return {
        "library_kind": library_kind,
        "fit_kind": fit_kind,
        "kurt": float(kurt),
        "rep": int(rep),
        "chi2": chi2,
        "hist": hist.astype(int).tolist(),
    }


# ─── Ray remote wrapper ─────────────────────────────────────────────────────
@ray.remote
def _one_rep_remote(library_kind: str, fit_kind: str, kurt: float, rep: int) -> dict:
    return _one_rep(library_kind, fit_kind, kurt, rep)


# ─── cell enumeration ──────────────────────────────────────────────────────
def _same_family_cells() -> list[tuple[str, str, float]]:
    """All (library=fit, fit, kurt) cells for the same-family pass."""
    return [(f, f, k) for f in FAMILIES for k in KURT_SWEEP]


def _cross_family_cells() -> list[tuple[str, str, float]]:
    """All ordered (library != fit) cells for the cross-family pass."""
    return [(lib, fit, k)
            for lib in FAMILIES for fit in FAMILIES for k in KURT_SWEEP
            if lib != fit]


def _percentiles(arr: np.ndarray, qs: Iterable[float]) -> dict[float, float]:
    return {q: float(np.percentile(arr, q)) for q in qs}


# ─── main run + report ──────────────────────────────────────────────────────
def run_gate(
    *,
    k_reps: int = K_REPS,
    use_ray: bool = True,
    progress: bool = True,
) -> tuple[str, dict]:
    """Run the full gate, returning (artifact body text, structured payload)."""
    out: list[str] = []
    p = lambda *a: out.append(" ".join(str(x) for x in a))

    p("Q2A synthetic gate -- chi^2 cuts from sample-size-matched null")
    p("")
    p(f"  KURT_SWEEP             = {KURT_SWEEP}  (Pearson kurtosis; Gaussian=3)")
    p(f"  K_REPS                 = {k_reps}")
    p(f"  M_PARTICLES            = {M_PARTICLES}")
    p(f"  N_PIT_BINS             = {N_PIT_BINS}")
    p(f"  FAMILIES               = {FAMILIES}")
    p(f"  N_LIBRARY              = {N_LIBRARY}")
    p(f"  SYNTHETIC_TRAJECTORY_LEN = {SYNTHETIC_TRAJECTORY_LEN}")
    p(f"  RNG_SEED_BASE          = {RNG_SEED_BASE}")
    p(f"  D_EMBED                = {D_EMBED}")
    p(f"  C_TRUE                 = {C_TRUE.flatten().tolist()}")
    p(f"  mix-2 generator w2     = {_MIX2_W2}")
    p(f"  mix-3 generator w      = {_MIX3_W}, s2 = {_MIX3_S2}")
    p("")

    same = _same_family_cells()
    cross = _cross_family_cells()
    total = (len(same) + len(cross)) * k_reps
    p(f"  same-family cells      = {len(same)} ({len(FAMILIES)} families x {len(KURT_SWEEP)} kurts)")
    p(f"  cross-family cells     = {len(cross)} ({len(FAMILIES)*(len(FAMILIES)-1)} ordered pairs x {len(KURT_SWEEP)} kurts)")
    p(f"  total runs             = {total}")
    p("")

    t0 = time.time()
    futures = []
    for lib, fit, kurt in same:
        for rep in range(k_reps):
            if use_ray:
                futures.append(_one_rep_remote.remote(lib, fit, kurt, rep))
            else:
                futures.append(_one_rep(lib, fit, kurt, rep))
    for lib, fit, kurt in cross:
        for rep in range(k_reps):
            if use_ray:
                futures.append(_one_rep_remote.remote(lib, fit, kurt, rep))
            else:
                futures.append(_one_rep(lib, fit, kurt, rep))

    if use_ray:
        results = []
        # Progress with ray.wait
        remaining = list(futures)
        done_n = 0
        while remaining:
            ready, remaining = ray.wait(remaining, num_returns=min(64, len(remaining)))
            for r in ready:
                results.append(ray.get(r))
            done_n += len(ready)
            if progress and done_n % 512 == 0:
                elapsed = time.time() - t0
                rate = done_n / elapsed if elapsed > 0 else 0
                eta = (total - done_n) / rate if rate > 0 else float("inf")
                print(f"  [{done_n}/{total}] elapsed {elapsed:.0f}s, "
                      f"rate {rate:.1f}/s, ETA {eta:.0f}s",
                      flush=True)
    else:
        results = futures

    t_run = time.time() - t0
    p(f"  wall-clock             = {t_run:.1f}s ({t_run/60:.1f}min)")
    p("")

    # ─── Aggregate per cell ────────────────────────────────────────────────
    p("## Same-family cells (library = fit)")
    p("")
    p(f"  {'family':>12} {'kurt':>6} {'mean':>10} {'sd':>10} "
      f"{'50pct':>10} {'95pct':>10}")
    same_p95 = {}
    same_means = []
    for f in FAMILIES:
        for k in KURT_SWEEP:
            arr = np.array([r["chi2"] for r in results
                            if r["library_kind"] == f and r["fit_kind"] == f
                            and r["kurt"] == k])
            pct = _percentiles(arr, (50.0, 95.0))
            same_p95[(f, k)] = pct[95.0]
            same_means.append((f, k, float(arr.mean()), float(arr.std())))
            p(f"  {f:>12} {k:>6.1f} {arr.mean():>10.2f} {arr.std():>10.2f} "
              f"{pct[50.0]:>10.2f} {pct[95.0]:>10.2f}")
    p("")

    p("## Cross-family cells (library != fit)")
    p("")
    p(f"  {'library':>12} {'fit':>12} {'kurt':>6} "
      f"{'mean':>10} {'sd':>10} {'50pct':>10} {'99pct':>10}")
    cross_p99 = {}
    cross_means = []
    for lib in FAMILIES:
        for fit in FAMILIES:
            if lib == fit:
                continue
            for k in KURT_SWEEP:
                arr = np.array([r["chi2"] for r in results
                                if r["library_kind"] == lib and r["fit_kind"] == fit
                                and r["kurt"] == k])
                pct = _percentiles(arr, (50.0, 99.0))
                cross_p99[(lib, fit, k)] = pct[99.0]
                cross_means.append((lib, fit, k, float(arr.mean()), float(arr.std())))
                p(f"  {lib:>12} {fit:>12} {k:>6.1f} "
                  f"{arr.mean():>10.2f} {arr.std():>10.2f} "
                  f"{pct[50.0]:>10.2f} {pct[99.0]:>10.2f}")
    p("")

    # ─── Cuts ──────────────────────────────────────────────────────────────
    r_a2_cut = float(max(same_p95.values()))
    r_a2_arg = max(same_p95.items(), key=lambda kv: kv[1])[0]
    r_c2_cross_p99 = float(max(cross_p99.values()))
    r_c2_arg = max(cross_p99.items(), key=lambda kv: kv[1])[0]
    r_c2_cut = float(max(10.0 * r_a2_cut, r_c2_cross_p99))
    r_c2_basis = "10*R-A2_cut" if 10.0 * r_a2_cut >= r_c2_cross_p99 else "max cross-family 99pct"

    p("## Derived cuts (pre-registered Q2A inputs)")
    p("")
    p(f"  R-A2 cut = max over (family, kurt) of 95th-pct same-family chi^2")
    p(f"           = {r_a2_cut:.2f}   at cell {r_a2_arg}")
    p(f"  cross-family 99pct max = {r_c2_cross_p99:.2f}  at cell {r_c2_arg}")
    p(f"  R-C2 cut = max(10 * R-A2_cut, cross-family 99pct max)")
    p(f"           = max({10.0*r_a2_cut:.2f}, {r_c2_cross_p99:.2f})")
    p(f"           = {r_c2_cut:.2f}   (binding: {r_c2_basis})")
    p(f"  R-B2 band = ({r_a2_cut:.2f}, {r_c2_cut:.2f}]")
    p("")

    # ─── Discriminability sanity ────────────────────────────────────────────
    p("## Discriminability sanity (mean same-family vs mean cross-family per kurt)")
    p("")
    p(f"  {'kurt':>6} {'mean_same':>12} {'mean_cross':>12} {'ratio':>10}")
    discriminability_warnings = []
    for k in KURT_SWEEP:
        sa = [m for (f, kk, m, sd) in same_means if kk == k]
        cr = [m for (lib, fit, kk, m, sd) in cross_means if kk == k]
        msa = float(np.mean(sa))
        mcr = float(np.mean(cr))
        ratio = mcr / msa if msa > 0 else float("nan")
        p(f"  {k:>6.1f} {msa:>12.2f} {mcr:>12.2f} {ratio:>10.2f}")
        if ratio < 1.2:
            discriminability_warnings.append(
                f"kurt={k}: cross/same ratio {ratio:.2f} < 1.20 — families barely discriminable"
            )
    p("")

    if discriminability_warnings:
        p("## Discriminability WARNINGS")
        p("")
        for w in discriminability_warnings:
            p(f"  - {w}")
        p("")

    # Hyperparameter summary block for the artifact header
    payload = {
        "kurt_sweep": list(KURT_SWEEP),
        "k_reps": k_reps,
        "m_particles": M_PARTICLES,
        "n_pit_bins": N_PIT_BINS,
        "families": list(FAMILIES),
        "n_library": N_LIBRARY,
        "synthetic_trajectory_len": SYNTHETIC_TRAJECTORY_LEN,
        "rng_seed_base": RNG_SEED_BASE,
        "d_embed": D_EMBED,
        "c_true": C_TRUE.flatten().tolist(),
        "mix2_w2": _MIX2_W2,
        "mix3_w": list(_MIX3_W),
        "mix3_s2_fixed": _MIX3_S2,
        "wall_clock_s": float(t_run),
        "n_runs": int(total),
        "r_a2_cut": r_a2_cut,
        "r_a2_cut_arg": list(r_a2_arg),
        "r_c2_cut": r_c2_cut,
        "r_c2_cross_p99_max": r_c2_cross_p99,
        "r_c2_cross_p99_arg": list(r_c2_arg),
        "r_c2_binding": r_c2_basis,
        "r_b2_band": [r_a2_cut, r_c2_cut],
        "discriminability_warnings": discriminability_warnings,
    }
    return "\n".join(out) + "\n", payload


# ─── main / CLI ────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--emit-result", action="store_true",
        help="write the provenanced METHOD-grade artifact via "
             "experiment.provenance.make_result (refuses a dirty tree)",
    )
    ap.add_argument(
        "--k-reps", type=int, default=K_REPS,
        help="number of synthetic reps per (library, fit, kurt) cell "
             f"(default: {K_REPS})",
    )
    ap.add_argument(
        "--no-ray", action="store_true",
        help="run serially (sanity check; bypasses ray)",
    )
    args = ap.parse_args(argv)

    use_ray = not args.no_ray
    if use_ray:
        if not ray.is_initialized():
            ray.init(log_to_driver=False, num_cpus=multiprocessing.cpu_count())

    text, payload = run_gate(k_reps=args.k_reps, use_ray=use_ray, progress=True)
    print(text, end="")

    if args.emit_result:
        from config import PROJECT_ROOT
        from experiment.provenance import Grade, make_result

        out_path = (
            PROJECT_ROOT / "experiment" / "results" / "q2a_synthetic_gate.txt"
        )
        hdr = make_result(
            path=out_path,
            grade=Grade.METHOD,
            title="Q2A synthetic gate: chi^2 cuts from sample-size-matched "
                  "null (production-fitter path)",
            body=text,
            inputs=payload,
            seeds={"rng_seed_base": RNG_SEED_BASE,
                   "rng_per_cell_rule": "_cell_seed(library_kind, fit_kind, kurt, rep): "
                                          "(lib<<30)|(fit<<28)|((int(kurt)&0xFF)<<20)|(rep&0xFFFFF), "
                                          "offset by RNG_SEED_BASE, mod 2^32"},
            frozen_spec_required=False,   # synthetic; Ontario-data-free
        )
        print(f"\nwrote provenanced METHOD artifact -> {out_path}")
        print(f"  R-A2 cut           = {payload['r_a2_cut']:.2f}")
        print(f"  R-C2 cut           = {payload['r_c2_cut']:.2f}")
        print(f"  R-B2 band          = ({payload['r_a2_cut']:.2f}, {payload['r_c2_cut']:.2f}]")
        print(f"  inputs_fingerprint = {hdr['inputs_fingerprint'][:16]}...")
        print(f"  body_sha256        = {hdr['body_sha256'][:16]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
