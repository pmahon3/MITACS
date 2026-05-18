"""Promote the operator ↔ Resolvent_Framework correspondence from a
mutable memory note to a committed, hashed, citable artifact (C3).

The correspondence is the project's most distinctive contribution and,
before Task 35, had the WEAKEST auditability/integrity of any
claim-grade item (it lived only in a mutable memory note, not a repo
artifact a reviewer sees). This module is the single source of truth
for the correspondence TEXT; ``make_result`` stamps it with a
provenance header + a tamper-evident body hash.

It is **METHOD/DESIGN grade**, not CLAIM: it is a descriptive
structural mapping, not an empirical result. It deliberately states the
*structural* correspondence and the three *qualifiers* (durable, data-
state independent) and references — rather than restates — the
empirical magnitudes (which live in the C1/rebaseline artifacts), so
the document does not go stale when tracked actuals extend.

``frozen_spec_required=False``: the correspondence is about the
estimator's mathematical form, not a particular frozen predictor run.

Run::

    python -m experiment.emit_theory_correspondence   # writes the artifact
"""
from __future__ import annotations

from config import PROJECT_ROOT

from .provenance import Grade, make_result

# ---- single source of truth for the correspondence text ----------------
BODY = r"""# Operator ↔ Resolvent_Framework Correspondence

Structural mapping between the operator this codebase constructs and the
active theory programme at `github.com/pmahon3/Resolvent_Framework`.
This is a **descriptive structural mapping**, not a novelty or
publication-worthiness claim (explicitly out of scope). Grounded in the
programme text `notes/programme/program_overview.md` (Paper II), not an
informal summary.

## What the code constructs

Per anchor *j*, the estimator produces a pair `(C_j, Σ_j)` defining a
state-dependent Gaussian Markov kernel

    Π^(j)(x, ·) = N(x C_j, Σ_j)

on the delay-embedding state space.

- `C_j` — leave-one-out weighted-least-squares local linear **drift**
  (row convention: `x_next = x @ C_j`), bandwidth by true-LOO-CV.
- `Σ_j` — plain μ-centred covariance of the residuals `Y − X C_j`
  (no residual kernel; the kernel-weighted form collapses to ≈ w²·Q).
  **Rank-1 by construction** of the single-variable delay embedding:
  coords 2..d of the one-step image are deterministic shifts of the
  input, so the only stochastic content is the scalar coordinate-0
  innovation.

## The programme's three layers and the correspondence

The programme (Paper II) keeps three notationally-distinct layers:

| Programme layer | Role | Code correspondence |
|---|---|---|
| **κ_Q** disintegration kernel `P(F ∈ · | Q=q)`, "forced by the measure" | conditional-regularity kernel | **the layer the code TARGETS** — a parametric Gaussian estimate of the conditional law of the next embedded state |
| **Π_t** Markov semigroup, Chapman–Kolmogorov "derived not assumed" | the semigroup | code yields only the SINGLE-STEP `Π_Δ`; it does not compose anchors or verify CK → it is the generator slice, **not** the semigroup `{Π_t}` |
| **K_t / P_t\*** operator layer, Koopman–Perron duality | the operators | code never forms an operator, but `(C_j, Σ_j)` is exactly the kernel from which `P_Δ\* / K_Δ` would be built; `Σ_j ≠ 0` IS the programme's non-Dirac regime, `Σ_j → 0` reproduces the classical-Koopman collapse the programme names |

## Precise statement

> The coded operator is a finite-sample, parametric, single-step,
> *locally-Gaussian* estimator of the programme's conditional-regularity
> kernel **κ_Q** at the embedding layer, from which `K_Δ / P_Δ*` are
> recoverable in the programme's non-Dirac regime.

## Three honest qualifiers (MUST NOT be dropped in any writeup)

1. **Register mismatch.** The programme *discloses* κ_Q from the
   measure; the code *constructs* a Gaussian proxy and fits it. Same
   target object, opposite epistemic register — the correspondence is
   "estimator of," not "instance of."

2. **Gaussianity is an imposed assumption.** It coincides with the true
   κ_Q only where the local conditional law is actually Gaussian.
   Production-validated empirical status (see the rebaseline facts and
   the C1 backtest artifact): for the intra-day estimand the scalar
   one-step innovation is **strongly non-Gaussian** (gate-validated
   excess kurtosis ≈24–33, tail ratio ≈2.4–3.8; a clean VAR(1)
   reference reads ≈0/≈1). Because `Σ_j` is rank-1 by construction this
   is the correct *univariate* question, not a multivariate one.
   Qualifier 2 is therefore **decisively operative**: `Σ_j` is a
   Gaussian second-moment proxy of a strongly heavy-tailed scalar
   innovation law. This is a mandatory writeup caveat.

3. **Single-step only.** Chapman–Kolmogorov / `{Π_t}` is neither
   constructed nor checked. The code has `Π_Δ`, not the semigroup.

## Theoretically clean (no qualifier needed)

- Delay embedding = Paper II reconstruction object
  (`DelayEmbedding.lean` / `ReconstructionTheorem.lean`).
- Drift `C_j` = Paper II minimal-sufficient-factor / local
  linearization.
- Diffusion `Σ_j` = the correct weighted conditional covariance (the
  second moment of κ_Q's Gaussian projection).

## Provenance notes

- The non-Gaussianity figures and the rank-1 property are
  production-path validated; see the C1 backtest artifact and the
  synthetic validation gates artifact (both carry their own provenance
  headers). This document deliberately states the *structural*
  correspondence and the *qualifiers* (durable, data-state
  independent) and references — rather than restates — the empirical
  magnitudes, so it does not go stale when tracked actuals extend.
- Resolvent_Framework is a separate repository; cite its commit at the
  time of writeup. Programme source of record: Paper II,
  `notes/programme/program_overview.md`.
"""


def main() -> None:
    out = PROJECT_ROOT / "experiment" / "results" / "THEORY_CORRESPONDENCE.md"
    hdr = make_result(
        path=out,
        grade=Grade.METHOD,
        title="Operator ↔ Resolvent_Framework structural correspondence "
              "(κ_Q estimator + 3 qualifiers)",
        body=BODY,
        inputs={
            "programme_source": "Resolvent_Framework Paper II, "
            "notes/programme/program_overview.md",
            "supersedes_memory_note": "mitacs-theory-correspondence "
            "(promoted from mutable memory to a hashed artifact)",
        },
        seeds={},  # no computation; static structural mapping
        frozen_spec_required=False,
    )
    print(f"wrote provenanced C3 artifact -> {out}")
    print(f"  grade              = {hdr['grade']}")
    print(f"  inputs_fingerprint = {hdr['inputs_fingerprint'][:16]}…")
    print(f"  body_sha256        = {hdr['body_sha256'][:16]}…")


if __name__ == "__main__":
    main()
