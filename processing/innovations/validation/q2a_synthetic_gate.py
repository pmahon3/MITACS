"""Q2A synthetic gate — chi^2 cuts under a three-variant weight sweep.

Pre-experiment calibration for the distributional-class thread's Q2A
node ("richer-family-mixture-or-nonparametric";
``notes/preregistrations/2026-05-26_distributional-class-thread/thread.yaml``).
The output cuts (R-A2_cut, R-C2_cut, R-B2 band) become the
pre-registered chi^2 thresholds in Q2A's phase_a.yaml. The gate
STOPS at producing these cuts — it does NOT touch any Ontario
post-cutoff data, propose phase_a content, or render verdicts.

Three-variant sweep (added in response to the
weight-choice-sensitivity concern surfaced when the "w_min = 3/kurt"
rule was found at the Pearson kurtosis ceiling):

  V1 (fixed)        : prior gate's design (mix-2 w_r = 0.05; mix-3
                      w = (0.8, 0.15, 0.05) at every kurt; scales
                      solved per kurt).
  V2 (alpha=0.5)    : w_min(kurt) = 0.5 * 3/kurt = 1.5/kurt. mix-3
                      bulk weights split evenly: (w_b, w_b, w_r)
                      with 2 w_b + w_r = 1. mix-3 scales keep V1's
                      asymmetric-bulk pattern (s1=1, s2=2); s3
                      solved per (kurt, weights).
  V3 (rho=10)       : pin s_rare/s_bulk = 10. mix-3 is
                      bulk-symmetric (s_b1 = s_b2 = 1, s_r = 10).
                      Solve w_r(kurt) per family from the closed
                      form, take the falling-branch root (rare
                      heavy-tail interpretation matching the prior
                      gate's design).

Pre-registered gate parameters (user-confirmed 2026-05-26):

  KURT_SWEEP     = (12, 20, 28, 40)  Pearson kurtosis (Gaussian = 3).
  K_REPS         = 500               synthetic replications per cell.
  M_PARTICLES    = 200               matches Q1 SMC.
  N_PIT_BINS     = 10                matches Q1.
  FAMILIES       = ("mixture_2", "mixture_3", "kde").
  VARIANTS       = ("V1_fixed", "V2_alpha0.5", "V3_rho10").

Two passes per (variant, family, kurt):

  Same-family : library drawn AND fit/scored with the same family.
                95th percentile of the resulting chi^2 distribution is
                the upper edge of "calibrated under correct family".
  Cross-family: library drawn from family X, fit/scored with family Y
                (6 ordered pairs per kurt per variant). 99th percentile
                of the resulting chi^2 distribution is the lower edge
                of "discriminably mis-specified".

Final cuts (variant-max aggregation; conservative for phase_a):

  R-A2 cut = max over (variant, family, kurt)         of 95th-pct same-family chi^2
  R-C2 cut = max(10 * R-A2_cut,
                 max over (variant, lib, fit, kurt)   of 99th-pct cross-family chi^2)
  R-B2 band = (R-A2_cut, R-C2_cut]

Code-path discipline (CLAUDE.md §"Discipline rule"): production
fitters are the three estimator entry points
(``mixture_2_gaussian_mle_fit``, ``mixture_3_gaussian_mle_fit``,
``kde_residual_fit``) — never reimplemented inline here. The library
generator uses a hardcoded ``C_TRUE`` literal so it avoids the
``np.linalg.lstsq``/inline-covariance audit traps; all OLS happens
inside the fitters.

Generator parameterisations:

* mix-2 generator: scales (s1=1, s2) with weights (1-w_r, w_r).
  Closed form for s2 given (kurt, w_r) in ``_solve_mix2_s2``.
* mix-3 generator: scales (s1=1, s2, s3) with weights (w1, w2, w3).
  Closed form for s3 given (kurt, weights, s1, s2) in
  ``_solve_mix3_s3``.
* mix-3 V3 special case: bulk-symmetric (s2 = s1 = 1, s3 = 10);
  closed form for w_r given (kurt, rho) is the same quadratic as the
  mix-2 V3 case (symmetric bulks collapse marginally to a single
  bulk; the fitter still sees a 3-component model).
* KDE generator: 10,000 samples from the kurt-matched mix-2 above
  (the V1 mix-2, fixed across variants — the KDE family is variant-
  independent at the generator level), with bandwidth = Silverman.

Closed-form derivations are in the helper docstrings below.

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
from typing import Callable, Iterable

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
VARIANTS: tuple[str, ...] = ("V1_fixed", "V2_alpha0.5", "V3_rho10")
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

# V1 fixed weights / V1 fixed mix-3 inner scale (s2)
_V1_MIX2_W2: float = 0.05
_V1_MIX3_W: tuple[float, float, float] = (0.80, 0.15, 0.05)
_V1_MIX3_S2: float = 2.0

# V2 fraction-of-ceiling parameter (w_min = alpha * 3/kurt)
_V2_ALPHA: float = 0.5

# V3 scale ratio (s_rare / s_bulk)
_V3_RHO: float = 10.0

# Stable enumeration for deterministic per-cell seeds. Python's built-in
# ``hash`` randomises per interpreter session (PYTHONHASHSEED), so a
# tuple hash would be NON-reproducible across runs and a provenanced
# artifact derived from it would not be reproducible from its recorded
# ``git_sha``. We encode the cell tuple into a deterministic 32-bit
# integer instead.
_FAMILY_ORD: dict[str, int] = {"mixture_2": 0, "mixture_3": 1, "kde": 2}
_VARIANT_ORD: dict[str, int] = {"V1_fixed": 0, "V2_alpha0.5": 1, "V3_rho10": 2}


def _cell_seed(variant: str, library_kind: str, fit_kind: str, kurt: float, rep: int) -> int:
    """Deterministic seed for the ``(variant, library, fit, kurt, rep)`` cell.

    Encoded so the five axes occupy disjoint bit ranges -- no aliasing
    across cells, deterministic across interpreters.

    Field widths in the 32-bit envelope:
        variant : 2 bits  (positions 30..31)
        library : 2 bits  (positions 28..29)
        fit     : 2 bits  (positions 26..27)
        kurt    : 8 bits  (positions 18..25, kurt mod 256)
        rep     : 18 bits (positions  0..17, rep mod 2^18)
    Then offset by RNG_SEED_BASE mod 2^32.
    """
    var_i = _VARIANT_ORD[variant]
    lib_i = _FAMILY_ORD[library_kind]
    fit_i = _FAMILY_ORD[fit_kind]
    kurt_i = int(kurt)
    code = (
        ((var_i & 0x3) << 30)
        | ((lib_i & 0x3) << 28)
        | ((fit_i & 0x3) << 26)
        | ((kurt_i & 0xFF) << 18)
        | (rep & 0x3FFFF)
    )
    return (RNG_SEED_BASE + code) & 0xFFFFFFFF


# ─── kurtosis-to-scale closed forms (for V1, V2: solve scales given weights) ─
def _solve_mix2_s2(K: float, w2: float, s1: float = 1.0) -> float:
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
    w: tuple[float, float, float],
    s12: tuple[float, float],
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


# ─── V3 inverse closed form: solve w_r given scale ratio ────────────────────
def _solve_v3_wr_from_rho(K: float, rho: float = _V3_RHO) -> float:
    """For a mix-2 with scales ``(1, rho)`` and weights ``(1-w_r, w_r)``,
    solve ``w_r`` so the Pearson kurt equals ``K``.

    From ``K * (w_b + w_r rho^2)^2 = 3 (w_b + w_r rho^4)`` with
    ``w_b = 1 - w_r``, let ``a = rho^2 - 1`` and ``b = rho^4 - 1``:

        K * (1 + a w_r)^2 = 3 (1 + b w_r)

    is quadratic in ``w_r`` (``K a^2 w_r^2 + (2 K a - 3 b) w_r + (K - 3) = 0``).
    There are typically two valid roots in ``(0, 1)``:
        * rising-branch (small ``w_r``): a very-rare extreme component
          whose finite-sample sampling distribution is degenerate at
          ``N_LIBRARY = 5000``.
        * falling-branch (larger ``w_r``): the heavy-tail-rare-but-
          observable interpretation matching V1's design (V1's ``w_r =
          0.05`` lies on this branch).
    We take the falling-branch (max-valid) root to match V1 in
    interpretation; this is what makes V1 vs V3 a substantive
    comparison rather than a pathological corner-case.

    The same algebra applies to V3's bulk-symmetric mix-3 since the two
    bulks collapse marginally to a single bulk -- the fitter still sees
    a 3-component model.
    """
    a = rho * rho - 1.0
    b = rho ** 4 - 1.0
    A = K * a * a
    B = 2.0 * K * a - 3.0 * b
    C_ = K - 3.0
    disc = B * B - 4.0 * A * C_
    if disc < 0:
        raise ValueError(f"V3 w_r infeasible for kurt={K}, rho={rho}: disc<0")
    r_hi = (-B + np.sqrt(disc)) / (2.0 * A)
    r_lo = (-B - np.sqrt(disc)) / (2.0 * A)
    valid = [r for r in (r_hi, r_lo) if 0.0 < r < 1.0]
    if not valid:
        raise ValueError(
            f"V3 w_r infeasible for kurt={K}, rho={rho}: roots {r_hi}, {r_lo} not in (0,1)"
        )
    return float(max(valid))   # falling-branch / observable-rare-tail root


# ─── analytic kurt verifier (closed form, used by reachability gate) ────────
def _closed_form_kurt(weights: Iterable[float], scales: Iterable[float]) -> float:
    """Pearson kurt of a centred Gaussian mixture in closed form."""
    w = np.asarray(weights, dtype=float)
    s = np.asarray(scales, dtype=float)
    num = (w * s ** 4).sum()
    den = (w * s ** 2).sum() ** 2
    return float(3.0 * num / den)


# ─── per-variant generator specs ────────────────────────────────────────────
@dataclass(frozen=True)
class GeneratorSpec:
    """Resolved generator parameters for one (variant, family, kurt) cell."""
    variant: str
    family: str
    kurt_target: float
    weights: tuple[float, ...]
    scales: tuple[float, ...]

    def closed_form_kurt(self) -> float:
        return _closed_form_kurt(self.weights, self.scales)


def _v1_spec(family: str, kurt: float) -> GeneratorSpec:
    """V1 fixed-weight generator."""
    if family == "mixture_2":
        w_r = _V1_MIX2_W2
        s2 = _solve_mix2_s2(kurt, w_r)
        return GeneratorSpec("V1_fixed", family, kurt, (1.0 - w_r, w_r), (1.0, s2))
    if family == "mixture_3":
        s3 = _solve_mix3_s3(kurt, _V1_MIX3_W, (1.0, _V1_MIX3_S2))
        return GeneratorSpec(
            "V1_fixed", family, kurt, _V1_MIX3_W, (1.0, _V1_MIX3_S2, s3)
        )
    # KDE generator is variant-independent (samples from the V1 mix-2 reservoir).
    if family == "kde":
        w_r = _V1_MIX2_W2
        s2 = _solve_mix2_s2(kurt, w_r)
        return GeneratorSpec("V1_fixed", family, kurt, (1.0 - w_r, w_r), (1.0, s2))
    raise ValueError(f"unknown family {family!r}")


def _v2_spec(family: str, kurt: float) -> GeneratorSpec:
    """V2 fraction-of-ceiling generator (w_min = alpha * 3/kurt)."""
    w_r = _V2_ALPHA * 3.0 / kurt
    if family == "mixture_2":
        s2 = _solve_mix2_s2(kurt, w_r)
        return GeneratorSpec("V2_alpha0.5", family, kurt, (1.0 - w_r, w_r), (1.0, s2))
    if family == "mixture_3":
        # Bulk split evenly: (w_b, w_b, w_r) with 2 w_b + w_r = 1.
        # Keep V1's asymmetric-bulk SCALE structure (s1=1, s2=2) so the
        # mix-3 marginal genuinely differs from mix-2 (advisor note:
        # bulk-symmetric scales would collapse mix-3 to mix-2 marginally
        # and destroy cross-family discriminability).
        w_b = (1.0 - w_r) / 2.0
        w_tuple = (w_b, w_b, w_r)
        s3 = _solve_mix3_s3(kurt, w_tuple, (1.0, _V1_MIX3_S2))
        return GeneratorSpec(
            "V2_alpha0.5", family, kurt, w_tuple, (1.0, _V1_MIX3_S2, s3)
        )
    if family == "kde":
        # KDE generator under V2: same convention as V1 (samples from
        # the variant's mix-2 reservoir at the same w_r).
        s2 = _solve_mix2_s2(kurt, w_r)
        return GeneratorSpec("V2_alpha0.5", family, kurt, (1.0 - w_r, w_r), (1.0, s2))
    raise ValueError(f"unknown family {family!r}")


def _v3_spec(family: str, kurt: float) -> GeneratorSpec:
    """V3 pin-scale-ratio generator (s_rare/s_bulk = rho)."""
    w_r = _solve_v3_wr_from_rho(kurt, _V3_RHO)
    if family == "mixture_2":
        return GeneratorSpec("V3_rho10", family, kurt, (1.0 - w_r, w_r), (1.0, _V3_RHO))
    if family == "mixture_3":
        # Bulk-symmetric mix-3: s_b1 = s_b2 = 1, s_r = rho.
        # Weights: (w_b, w_b, w_r) with 2 w_b + w_r = 1.
        # Marginally collapses to V3 mix-2 by construction (two
        # co-located bulks behave as one); the fitter still sees a
        # 3-component model so cross-family mixture_2 vs mixture_3
        # remains a model-complexity test.
        w_b = (1.0 - w_r) / 2.0
        return GeneratorSpec(
            "V3_rho10", family, kurt, (w_b, w_b, w_r), (1.0, 1.0, _V3_RHO)
        )
    if family == "kde":
        # KDE under V3: reservoir from the V3 mix-2 at this w_r.
        return GeneratorSpec("V3_rho10", family, kurt, (1.0 - w_r, w_r), (1.0, _V3_RHO))
    raise ValueError(f"unknown family {family!r}")


_VARIANT_SPEC_BUILDERS: dict[str, Callable[[str, float], GeneratorSpec]] = {
    "V1_fixed":    _v1_spec,
    "V2_alpha0.5": _v2_spec,
    "V3_rho10":    _v3_spec,
}


# ─── reachability gate ──────────────────────────────────────────────────────
def _check_reachability(reltol: float = 0.01) -> dict[tuple[str, str, float], GeneratorSpec]:
    """Resolve all (variant, family, kurt) GeneratorSpecs and verify the
    closed-form kurt is within ``reltol`` of the target.

    BLOCKING: raises if any cell exceeds tolerance, naming the cell.
    Returns the dict of resolved specs keyed by ``(variant, family, kurt)``.
    """
    specs: dict[tuple[str, str, float], GeneratorSpec] = {}
    failures: list[str] = []
    for variant in VARIANTS:
        builder = _VARIANT_SPEC_BUILDERS[variant]
        for family in FAMILIES:
            for kurt in KURT_SWEEP:
                spec = builder(family, kurt)
                k_v = spec.closed_form_kurt()
                rel = abs(k_v - kurt) / kurt
                if rel > reltol:
                    failures.append(
                        f"({variant}, {family}, kurt={kurt}): "
                        f"target={kurt}, closed-form={k_v:.4f}, rel={rel:.4%}"
                    )
                specs[(variant, family, kurt)] = spec
    if failures:
        raise RuntimeError(
            "Reachability check FAILED for "
            f"{len(failures)} cell(s):\n  " + "\n  ".join(failures)
        )
    return specs


# ─── true-family residual sampler (per resolved spec) ───────────────────────
@dataclass(frozen=True)
class TrueFamily:
    """Generator parameters for one (variant, family, kurt) combination,
    plus the (kurt-matched) KDE reservoir."""
    spec: GeneratorSpec
    kde_bandwidth: float
    kde_samples: np.ndarray


def _build_true_family(spec: GeneratorSpec, rng: np.random.Generator) -> TrueFamily:
    # The KDE reservoir is sampled from the variant-resolved mix-2
    # (weights and scales for this variant at this kurt). This makes
    # the KDE family's "true law" track the variant — KDE is the
    # smoothed version of THIS variant's heavy-tailed mix-2.
    kde_n = 10_000
    if spec.family in ("mixture_2", "kde"):
        w = np.asarray(spec.weights); s = np.asarray(spec.scales)
        # spec for mixture_2 / kde is always 2-component, scale s = (s1, s2)
        which = rng.choice(len(w), size=kde_n, p=w)
        z = rng.standard_normal(kde_n)
        kde_samples = z * s[which]
    elif spec.family == "mixture_3":
        # 3-component reservoir for the KDE -- but the KDE family only
        # gets built for spec.family == "kde". This branch exists for
        # symmetry; we resample the mix-3 if asked, but in practice the
        # KDE bandwidth is only used when spec.family == "kde".
        w = np.asarray(spec.weights); s = np.asarray(spec.scales)
        which = rng.choice(len(w), size=kde_n, p=w)
        z = rng.standard_normal(kde_n)
        kde_samples = z * s[which]
    else:
        raise ValueError(f"unknown family {spec.family!r}")

    std = float(np.std(kde_samples, ddof=1))
    q25, q75 = np.quantile(kde_samples, [0.25, 0.75])
    iqr = float(q75 - q25)
    spread = min(std, iqr / 1.34) if iqr > 0 else std
    kde_bw = 0.9 * max(spread, 1e-9) * (kde_n ** (-1.0 / 5.0))
    return TrueFamily(spec=spec, kde_bandwidth=kde_bw, kde_samples=kde_samples)


def _draw_residuals(tf: TrueFamily, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw n i.i.d. residuals from the named TRUE family at this variant's spec."""
    spec = tf.spec
    if spec.family in ("mixture_2", "mixture_3"):
        w = np.asarray(spec.weights); s = np.asarray(spec.scales)
        which = rng.choice(len(w), size=n, p=w)
        z = rng.standard_normal(n)
        return z * s[which]
    if spec.family == "kde":
        # Convolution rule: pick a reservoir sample + bandwidth-scaled
        # Gaussian noise. This is exactly the density the KDE encodes.
        idx = rng.integers(0, len(tf.kde_samples), size=n)
        return tf.kde_samples[idx] + tf.kde_bandwidth * rng.standard_normal(n)
    raise ValueError(f"unknown family {spec.family!r}")


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
    variant: str,
    library_kind: str,
    fit_kind: str,
    kurt: float,
    rep: int,
) -> dict:
    """One synthetic replication.

    Build a library (X, Y) drawn from family ``library_kind`` at the
    target kurt under the named variant's spec; fit family
    ``fit_kind`` to it via the production fitter; for each of N_TRAJ
    test states, draw an actual next-step from the TRUE library family
    (i.e. NOT from the fitted family) and a sample of M predictive
    draws from the FITTED family; PIT = randomized mid-rank. Aggregate
    to a 10-bin chi^2 vs uniform.

    Returns a dict with the chi^2 and bin counts.
    """
    seed = _cell_seed(variant, library_kind, fit_kind, kurt, rep)
    rng = np.random.default_rng(seed)

    # 1) Build library under this variant's library-family spec
    lib_spec = _VARIANT_SPEC_BUILDERS[variant](library_kind, kurt)
    true_lib = _build_true_family(lib_spec, rng)
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
        "variant": variant,
        "library_kind": library_kind,
        "fit_kind": fit_kind,
        "kurt": float(kurt),
        "rep": int(rep),
        "chi2": chi2,
        "hist": hist.astype(int).tolist(),
    }


# ─── Ray remote wrapper ─────────────────────────────────────────────────────
@ray.remote
def _one_rep_remote(variant: str, library_kind: str, fit_kind: str, kurt: float, rep: int) -> dict:
    return _one_rep(variant, library_kind, fit_kind, kurt, rep)


# ─── cell enumeration ──────────────────────────────────────────────────────
def _same_family_cells() -> list[tuple[str, str, str, float]]:
    """All (variant, library=fit, fit, kurt) cells for the same-family pass."""
    return [(v, f, f, k) for v in VARIANTS for f in FAMILIES for k in KURT_SWEEP]


def _cross_family_cells() -> list[tuple[str, str, str, float]]:
    """All ordered (variant, library != fit) cells for the cross-family pass."""
    return [(v, lib, fit, k)
            for v in VARIANTS
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

    p("Q2A synthetic gate -- chi^2 cuts under three-variant weight sweep")
    p("")
    p(f"  KURT_SWEEP             = {KURT_SWEEP}  (Pearson kurtosis; Gaussian=3)")
    p(f"  K_REPS                 = {k_reps}")
    p(f"  M_PARTICLES            = {M_PARTICLES}")
    p(f"  N_PIT_BINS             = {N_PIT_BINS}")
    p(f"  FAMILIES               = {FAMILIES}")
    p(f"  VARIANTS               = {VARIANTS}")
    p(f"  N_LIBRARY              = {N_LIBRARY}")
    p(f"  SYNTHETIC_TRAJECTORY_LEN = {SYNTHETIC_TRAJECTORY_LEN}")
    p(f"  RNG_SEED_BASE          = {RNG_SEED_BASE}")
    p(f"  D_EMBED                = {D_EMBED}")
    p(f"  C_TRUE                 = {C_TRUE.flatten().tolist()}")
    p("")
    p("  Variant rules:")
    p(f"    V1_fixed     mix-2 w_r = {_V1_MIX2_W2}; mix-3 w = {_V1_MIX3_W}, fixed s2 = {_V1_MIX3_S2}; solve scales")
    p(f"    V2_alpha0.5  w_min(K) = {_V2_ALPHA} * 3/K = {_V2_ALPHA*3}/K; mix-3 bulks split evenly, V1 s2={_V1_MIX3_S2}; solve s3")
    p(f"    V3_rho10     s_rare/s_bulk = {_V3_RHO}; mix-3 bulk-symmetric (s_b={1.0}); solve w_r per K (falling-branch)")
    p("")

    # Reachability gate FIRST (blocking)
    p("## Reachability check (closed-form kurt vs target, reltol=1%)")
    p("")
    specs = _check_reachability(reltol=0.01)
    p(f"  {'variant':>14} {'family':>12} {'kurt':>6} "
      f"{'weights':>32} {'scales':>26} {'k_v':>8}")
    spec_payload = []
    for variant in VARIANTS:
        for family in FAMILIES:
            for kurt in KURT_SWEEP:
                spec = specs[(variant, family, kurt)]
                wstr = "[" + ",".join(f"{w:.4f}" for w in spec.weights) + "]"
                sstr = "[" + ",".join(f"{s:.4f}" for s in spec.scales) + "]"
                k_v = spec.closed_form_kurt()
                p(f"  {variant:>14} {family:>12} {kurt:>6.1f} "
                  f"{wstr:>32} {sstr:>26} {k_v:>8.3f}")
                spec_payload.append({
                    "variant": variant, "family": family, "kurt": kurt,
                    "weights": list(spec.weights), "scales": list(spec.scales),
                    "closed_form_kurt": k_v,
                })
    p("")
    p("  REACHABILITY OK (all cells within 1% of target).")
    p("")

    same = _same_family_cells()
    cross = _cross_family_cells()
    total = (len(same) + len(cross)) * k_reps
    p(f"  same-family cells      = {len(same)} ({len(VARIANTS)} variants x {len(FAMILIES)} families x {len(KURT_SWEEP)} kurts)")
    p(f"  cross-family cells     = {len(cross)} ({len(VARIANTS)} variants x {len(FAMILIES)*(len(FAMILIES)-1)} ordered pairs x {len(KURT_SWEEP)} kurts)")
    p(f"  total runs             = {total}")
    p("")

    t0 = time.time()
    futures = []
    for variant, lib, fit, kurt in same:
        for rep in range(k_reps):
            if use_ray:
                futures.append(_one_rep_remote.remote(variant, lib, fit, kurt, rep))
            else:
                futures.append(_one_rep(variant, lib, fit, kurt, rep))
    for variant, lib, fit, kurt in cross:
        for rep in range(k_reps):
            if use_ray:
                futures.append(_one_rep_remote.remote(variant, lib, fit, kurt, rep))
            else:
                futures.append(_one_rep(variant, lib, fit, kurt, rep))

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

    # ─── Aggregate per cell (per-variant) ─────────────────────────────────
    p("## Per-variant same-family cells (library = fit), 95th-pct chi^2")
    p("")
    p(f"  {'variant':>14} {'family':>12} {'kurt':>6} {'mean':>10} {'sd':>10} "
      f"{'50pct':>10} {'95pct':>10}")
    same_p95: dict[tuple[str, str, float], float] = {}
    same_means: list[tuple[str, str, float, float, float]] = []
    for variant in VARIANTS:
        for f in FAMILIES:
            for k in KURT_SWEEP:
                arr = np.array([r["chi2"] for r in results
                                if r["variant"] == variant
                                and r["library_kind"] == f and r["fit_kind"] == f
                                and r["kurt"] == k])
                pct = _percentiles(arr, (50.0, 95.0))
                same_p95[(variant, f, k)] = pct[95.0]
                same_means.append((variant, f, k, float(arr.mean()), float(arr.std())))
                p(f"  {variant:>14} {f:>12} {k:>6.1f} {arr.mean():>10.2f} {arr.std():>10.2f} "
                  f"{pct[50.0]:>10.2f} {pct[95.0]:>10.2f}")
    p("")

    p("## Per-variant cross-family cells (library != fit), 99th-pct chi^2")
    p("")
    p(f"  {'variant':>14} {'library':>12} {'fit':>12} {'kurt':>6} "
      f"{'mean':>10} {'sd':>10} {'50pct':>10} {'99pct':>10}")
    cross_p99: dict[tuple[str, str, str, float], float] = {}
    cross_means: list[tuple[str, str, str, float, float, float]] = []
    for variant in VARIANTS:
        for lib in FAMILIES:
            for fit in FAMILIES:
                if lib == fit:
                    continue
                for k in KURT_SWEEP:
                    arr = np.array([r["chi2"] for r in results
                                    if r["variant"] == variant
                                    and r["library_kind"] == lib and r["fit_kind"] == fit
                                    and r["kurt"] == k])
                    pct = _percentiles(arr, (50.0, 99.0))
                    cross_p99[(variant, lib, fit, k)] = pct[99.0]
                    cross_means.append((variant, lib, fit, k, float(arr.mean()), float(arr.std())))
                    p(f"  {variant:>14} {lib:>12} {fit:>12} {k:>6.1f} "
                      f"{arr.mean():>10.2f} {arr.std():>10.2f} "
                      f"{pct[50.0]:>10.2f} {pct[99.0]:>10.2f}")
    p("")

    # ─── Per-variant cuts (intra-variant aggregation) ─────────────────────
    p("## Per-variant cuts (intra-variant max of same/cross)")
    p("")
    p(f"  {'variant':>14} {'R-A2 (intra)':>14} {'cross-99 max (intra)':>22} "
      f"{'R-C2 (intra)':>14} {'R-A2 cell':>32} {'cross cell':>40}")
    per_variant_cuts: dict[str, dict] = {}
    for variant in VARIANTS:
        sf = {k: v for k, v in same_p95.items() if k[0] == variant}
        cf = {k: v for k, v in cross_p99.items() if k[0] == variant}
        a2_intra = max(sf.values())
        a2_arg = max(sf.items(), key=lambda kv: kv[1])[0]   # (variant, family, kurt)
        cf_max = max(cf.values())
        cf_arg = max(cf.items(), key=lambda kv: kv[1])[0]   # (variant, lib, fit, kurt)
        c2_intra = max(10.0 * a2_intra, cf_max)
        per_variant_cuts[variant] = {
            "r_a2_intra": float(a2_intra),
            "r_a2_intra_cell": [a2_arg[1], a2_arg[2]],     # (family, kurt)
            "cross_p99_max": float(cf_max),
            "cross_p99_arg": [cf_arg[1], cf_arg[2], cf_arg[3]],   # (lib, fit, kurt)
            "r_c2_intra": float(c2_intra),
        }
        p(f"  {variant:>14} {a2_intra:>14.2f} {cf_max:>22.2f} "
          f"{c2_intra:>14.2f} {str((a2_arg[1], a2_arg[2])):>32} "
          f"{str((cf_arg[1], cf_arg[2], cf_arg[3])):>40}")
    p("")

    # ─── Per-variant cross/same discriminability ──────────────────────────
    p("## Per-variant discriminability (mean cross-family / mean same-family per kurt)")
    p("")
    p(f"  {'variant':>14} {'kurt':>6} {'mean_same':>12} {'mean_cross':>12} {'ratio':>10}")
    discriminability_warnings: list[str] = []
    discriminability_table: list[dict] = []
    for variant in VARIANTS:
        for k in KURT_SWEEP:
            sa = [m for (v, f, kk, m, sd) in same_means if v == variant and kk == k]
            cr = [m for (v, lib, fit, kk, m, sd) in cross_means if v == variant and kk == k]
            msa = float(np.mean(sa))
            mcr = float(np.mean(cr))
            ratio = mcr / msa if msa > 0 else float("nan")
            p(f"  {variant:>14} {k:>6.1f} {msa:>12.2f} {mcr:>12.2f} {ratio:>10.2f}")
            discriminability_table.append({
                "variant": variant, "kurt": float(k),
                "mean_same": msa, "mean_cross": mcr, "ratio": ratio,
            })
            if ratio < 1.2:
                discriminability_warnings.append(
                    f"({variant}, kurt={k}): cross/same ratio {ratio:.2f} < 1.20 — families barely discriminable"
                )
    p("")

    # ─── Final variant-max cuts ───────────────────────────────────────────
    r_a2_cut = float(max(same_p95.values()))
    r_a2_arg = max(same_p95.items(), key=lambda kv: kv[1])[0]   # (variant, family, kurt)
    r_c2_cross_p99 = float(max(cross_p99.values()))
    r_c2_arg = max(cross_p99.items(), key=lambda kv: kv[1])[0]  # (variant, lib, fit, kurt)
    r_c2_cut = float(max(10.0 * r_a2_cut, r_c2_cross_p99))
    r_c2_basis = "10*R-A2_cut" if 10.0 * r_a2_cut >= r_c2_cross_p99 else "max cross-family 99pct"

    p("## Final variant-max cuts (pre-registered Q2A inputs)")
    p("")
    p(f"  R-A2 cut = max over (variant, family, kurt) of 95th-pct same-family chi^2")
    p(f"           = {r_a2_cut:.2f}   at cell {r_a2_arg}  (variant, family, kurt)")
    p(f"  cross-family 99pct max = {r_c2_cross_p99:.2f}  at cell {r_c2_arg}  (variant, library, fit, kurt)")
    p(f"  R-C2 cut = max(10 * R-A2_cut, cross-family 99pct max)")
    p(f"           = max({10.0*r_a2_cut:.2f}, {r_c2_cross_p99:.2f})")
    p(f"           = {r_c2_cut:.2f}   (binding: {r_c2_basis})")
    p(f"  R-B2 band = ({r_a2_cut:.2f}, {r_c2_cut:.2f}]")
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
        "variants": list(VARIANTS),
        "n_library": N_LIBRARY,
        "synthetic_trajectory_len": SYNTHETIC_TRAJECTORY_LEN,
        "rng_seed_base": RNG_SEED_BASE,
        "d_embed": D_EMBED,
        "c_true": C_TRUE.flatten().tolist(),
        "v1_mix2_w2": _V1_MIX2_W2,
        "v1_mix3_w": list(_V1_MIX3_W),
        "v1_mix3_s2_fixed": _V1_MIX3_S2,
        "v2_alpha": _V2_ALPHA,
        "v3_rho": _V3_RHO,
        "wall_clock_s": float(t_run),
        "n_runs": int(total),
        "r_a2_cut": r_a2_cut,
        "r_a2_cut_arg": [r_a2_arg[0], r_a2_arg[1], r_a2_arg[2]],
        "r_c2_cut": r_c2_cut,
        "r_c2_cross_p99_max": r_c2_cross_p99,
        "r_c2_cross_p99_arg": [r_c2_arg[0], r_c2_arg[1], r_c2_arg[2], r_c2_arg[3]],
        "r_c2_binding": r_c2_basis,
        "r_b2_band": [r_a2_cut, r_c2_cut],
        "per_variant_cuts": per_variant_cuts,
        "discriminability_warnings": discriminability_warnings,
        "discriminability_table": discriminability_table,
        "generator_specs": spec_payload,
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
        help="number of synthetic reps per (variant, library, fit, kurt) cell "
             f"(default: {K_REPS})",
    )
    ap.add_argument(
        "--no-ray", action="store_true",
        help="run serially (sanity check; bypasses ray)",
    )
    ap.add_argument(
        "--reachability-only", action="store_true",
        help="run only the closed-form reachability check (no SMC)",
    )
    args = ap.parse_args(argv)

    if args.reachability_only:
        specs = _check_reachability()
        print(f"REACHABILITY OK for {len(specs)} (variant, family, kurt) cells")
        for (variant, family, kurt), spec in specs.items():
            print(f"  {variant} {family} kurt={kurt}: "
                  f"weights={spec.weights}, scales={spec.scales}, "
                  f"k_v={spec.closed_form_kurt():.4f}")
        return 0

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
            title="Q2A synthetic gate: chi^2 cuts under three-variant weight sweep "
                  "(production-fitter path)",
            body=text,
            inputs=payload,
            seeds={"rng_seed_base": RNG_SEED_BASE,
                   "rng_per_cell_rule": "_cell_seed(variant, library_kind, fit_kind, kurt, rep): "
                                          "(var<<30)|(lib<<28)|(fit<<26)|((int(kurt)&0xFF)<<18)|(rep&0x3FFFF), "
                                          "offset by RNG_SEED_BASE, mod 2^32"},
            frozen_spec_required=False,   # synthetic; Ontario-data-free
        )
        print(f"\nwrote provenanced METHOD artifact -> {out_path}")
        print(f"  R-A2 cut           = {payload['r_a2_cut']:.2f}  at {payload['r_a2_cut_arg']}")
        print(f"  R-C2 cut           = {payload['r_c2_cut']:.2f}  (binding: {payload['r_c2_binding']})")
        print(f"  R-B2 band          = ({payload['r_a2_cut']:.2f}, {payload['r_c2_cut']:.2f}]")
        print(f"  inputs_fingerprint = {hdr['inputs_fingerprint'][:16]}...")
        print(f"  body_sha256        = {hdr['body_sha256'][:16]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
