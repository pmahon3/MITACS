# Multi-Scale Factor Coherence via the Residual Diffusion

**Status.** Seed note. Written 2026-05-25 in MITACS following the
framework-level investigation of post-cutoff bias in the Ontario
κ_Q predictor. The theory-side analogue belongs eventually in
`Research/Mathematics/Resolvent_Framework` for novelty / literature
audit, but it is seeded here because the lab's empirical evidence is
load-bearing for whether the question is worth asking and the formal
content is downstream of the empirical structure.

**Audience.** First reader: us, here in the lab. Second reader: the
applied-analogue audit workflow (TBD; see §8). Third reader, if and
when promoted: the theory-side `/audit pure` gate in
`Research/Mathematics/Resolvent_Framework`.

---

## 1. The empirical observation that forced this seed

The current predictor uses a one-step κ_Q estimator on an
hour-resolved delay embedding, iterates it to 24-hour horizons, and
destandardizes through hour-of-day `σ_{m,h}`. Post-cutoff (2025-2026)
it shows a structured per-HE bias: mid-day over-forecast (+200 to
+1000 MW), late-night under-forecast (−200 to −600 MW). We pulled
every lever the framework has — climatology recency window, μ-only
de-mean dropping σ, operator-library window, full transform removal —
and the per-HE bias *shape* is invariant under all of them; only its
magnitude shuffles. The shape is intrinsic to the procedure
"iterate-one-step-on-stationary-z, then re-amplify by hour-varying σ."

A related earlier finding (`mitacs-theta-fibre-simplex`): on a CK
consistency probe, synthetic VAR(1) passes (iterated kernel ≈ direct
kernel at h=24), but Ontario shows ≈100% relative drift in iterated
vs. directly-fit kernels by h=24. The 1-step κ_Q estimator is not
generating the h-step κ_Q's empirical structure; they disagree by an
amount that is not noise.

These two findings — the bias shape and the CK drift — are the same
underlying object measured two ways. **The iterated 1-step kernel
does not factor the actual multi-scale dynamics**, and the failure is
visible in (a) the mean bias and (b) the kernel discrepancy.

The empirical question that closed the session: **what would it look
like to estimate operators at multiple scales and require them to be
coherent with each other, rather than estimate one operator and
iterate?**

## 2. The conceptual move

In Resolvent_Framework, the dynamics of a process are described by an
operator `L` (or its associated semigroup `\{P_t\}_{t \geq 0}` and
resolvent `R(λ) = (λI − L)^{-1}`). The resolvent encodes the dynamics
*at every scale* simultaneously — it is a complete object.

When we infer dynamics from data, we don't directly estimate `L` (or
the resolvent). We estimate finitely many *factors*: the semigroup at
the scales h ∈ ℋ for which we have data and interest. In this
project, ℋ ⊆ {1, 2, …, 24} (hours). Each factor `κ_h` is a conditional
kernel at scale h — drift `C_h` and residual diffusion `Σ_h`.

The natural question is what relation the family `\{\hat{κ}_h\}_{h
\in ℋ}` must satisfy to be a coherent set of factors of *one*
underlying operator. Three pictures:

**(A) The strict semigroup picture.** There is one operator. All
factors arise from it: `κ_h = κ_1^{(h)}` (h-fold convolution).
Equivalently, the family is a semigroup homomorphism from `(ℕ, +)`
into operator composition. **This is what iterating the 1-step κ_Q
implicitly assumes.** The Ontario data refutes this picture: the
1-step iterated to 24 steps is empirically ≠ the directly-fit
24-step.

**(B) The no-structure picture.** Each `\hat{κ}_h` is an independent
estimator at its scale. No cross-scale relation is required or
expected. This is the naive "24 separate operators" reading. It
discards information: every (t, t+h) pair is data about the same
underlying dynamics regardless of h, but independent fits use each
scale only once.

**(C) The relaxed-factorization picture.** There is one underlying
object, but the family of empirically-inferred factors `\{\hat{κ}_h\}`
relates to it through a *relaxed* semigroup law:

  `\hat{κ}_{h_1 + h_2} = \hat{κ}_{h_1} \star \hat{κ}_{h_2} + Δ_{h_1, h_2}`

where `Δ` is the *coherence defect* — a measurable, structured object
that captures the obstruction to strict semigroup behaviour. The
defect is not noise: it has structure inherited from the dimensions
of the dynamics that the embedding cannot resolve. In the limit where
the embedding captures all relevant degrees of freedom and the noise
is well-mixed, `Δ → 0` and we recover (A); when the embedding is
arbitrarily impoverished, `Δ` dominates and we approach (B).

**Picture (C) is what we're after.** It says: the factors are not
independent (against (B)), but they are also not the iterated
image of a single estimator (against (A)). They are *coherent up to
a structured defect*, and the defect itself is informative about
what dynamics live outside the embedding.

This is structurally close to (though we do not yet claim
equivalent to) several existing formal objects: a semigroup
homomorphism with bounded multiplicative error; a renormalization-
group map relating effective dynamics at different scales; a
filtered family of operators where the filtration is by horizon.
Naming and literature-positioning is for the theory-side audit, not
for this seed.

## 3. Why the residual diffusion is the load-bearing object

The drift `C_h` and the diffusion `Σ_h` together define each
factor κ_h. Under (A), both are functions of one underlying
infinitesimal pair `(L, Q)`:

  `C_h = e^{hL}`,  `Σ_h = \int_0^h e^{sL} Q e^{sL^\top} ds`     (★)

The drift identity composes trivially (matrix exponentials always
satisfy `e^{(h_1 + h_2)L} = e^{h_1 L} e^{h_2 L}`), so cross-scale
drift coherence is automatic *for any L* — it imposes only that the
estimated drifts be consistent matrix powers of *something*. The
diffusion identity (★) is non-trivial: it constrains the rate of
variance accumulation, its eigenmode structure, and the off-diagonal
couplings, all simultaneously, in terms of `(L, Q)`. Two factors with
the same `C_{h_1}, C_{h_2}` can have radically different `Σ_{h_1},
Σ_{h_2}`, only some of which are consistent with (★).

Therefore: **the diffusion is what makes the cross-scale coherence
constraint bite.** It is the part of the factor that, when measured
at multiple scales, distinguishes "coherent factors of one
resolvent" from "factors that happen to agree at the mean."

There is a second reason the diffusion is the right object,
particular to this project: **the empirical bias mechanism we
identified is itself a diffusion mechanism.** Our session-end
analysis showed that the per-HE bias arises from the operator's
σ_{m,h}-amplification of z-domain residuals into MW. Restated: the
operator's prediction at scale h is `forecast = μ_{m,h}(t+h) + σ_{m,h}(t+h)
\cdot z_{next}`, where `z_{next}` is the iterated mean and `σ_{m,h}` is
the seasonal scale. The bias-shape we see is essentially the hour-of-day
profile of σ_{m,h}, scaled by the operator's z-prediction. *The
forecast's uncertainty structure (σ-determined) is mis-matched to
the data's empirical h-step variance structure.* Said in the
language of (★): the framework's implicit `Σ_h` (built from
iterated `Σ_1` and the destandardize step) does not equal the
data's empirical `\hat{Σ}_h`. The bias is the diffusion defect
made visible through point forecasts.

So the diffusion is right both abstractly (the joint identifier of L
and Q across scales) and concretely (the place the framework is
empirically observed to fail).

## 4. The three meanings of "scale" and how they relate

The conversation slid between three meanings of "scale" that deserve
to be pinned down because the formal claims depend on which is meant:

**(a) Horizon scale** — `h ∈ {1, …, 24}`. The lookahead distance of
each factor. This is the indexing of the family of factors.

**(b) Frequency / eigenmode scale** — eigenvalues `λ_k` of `L`. The
*dynamical timescales* present in the operator: fast modes (large
`|λ_k|`) relax quickly, slow modes (small `|λ_k|`) persist.

**(c) Resolution scale** — set by the data: sampling interval (1 h)
on the fast end, total observation window on the slow end. The
*identifiable band* in (b) is bounded by (c).

The three are linked: horizon h "sees" eigenmodes whose timescales
`1/|λ_k|` are comparable to or shorter than h. Far-horizon
factors (large h) carry information mainly about slow modes;
near-horizon factors (small h) carry information about fast modes.
The empirical resolvent estimation problem is identifying `(L, Q)`
in the (b)-band determined by (c), using factor data at the (a)
points we have. Anything outside the identifiable band must be
assumed, regularized, or left implicit; it cannot be measured.

In particular: 24-hour Ontario data, hourly sampling, cannot resolve
sub-hourly dynamics or multi-day modes from this dataset alone. The
"resolvent at all scales" lift in §2 is a *statement about the
underlying object*; our empirical access is to a *band* of scales,
and the seed's question about coherence is necessarily restricted to
that band.

## 5. What the relaxed-factorization picture predicts, concretely

If (C) is the right picture, the empirical signature is:

**5.1 Drift agrees, diffusion diverges.** The mean prediction of the
1-step-iterated factor at h=12 may track the directly-fit 12-step
drift fairly well (because composition of mean predictors is
relatively forgiving). But the iterated covariance
`\hat{P}_1^{11} \hat{Σ}_1 \hat{P}_1^{11\top}` should *systematically
diverge* from the empirical `\hat{Σ}_{12}`. The divergence has a
specific structural form: under-estimation in directions where
unmodeled exogenous variance enters; over-estimation in directions
where the iteration accumulates spurious correlated noise.

**5.2 Coherence defect concentrates at specific (h, eigenmode)
pairs.** If the embedding misses one degree of freedom whose
characteristic timescale is τ, then the coherence defect `Δ_{h, h'}`
is largest at scales `h, h'` such that `h + h' ≈ τ` — i.e., where
that mode's correlation is starting to matter. The defect is *not*
spread uniformly across (h, h') pairs.

**5.3 Direct-fit factors give different point forecasts AND
different uncertainty than iterated.** The per-HE bias shape we
observed is, under (C), the leading-order signature of the iterated
mean missing the direct-fit mean at horizons where the missing-mode
relaxation is active. We predict that direct-fit `\hat{κ}_h` for
h=12, applied to a fresh delivery day, will produce mid-day
forecasts closer to actuals AND wider uncertainty intervals than
iterated κ_1^{12}.

These are testable predictions, not just framings.

## 6. The empirical programme (in lab terms)

Done in order; each step is a falsifiable experiment, not just a
descriptive analysis.

**6.1 Direct multi-scale factor estimation.** For h ∈ {1, 2, 6, 12,
24}, fit `(\hat{C}_h, \hat{Σ}_h)` independently from (t, t+h) pairs
in the pre-cutoff library, holding the embedding fixed at the
current spec. Per-day-type as we do now. Outputs: 5 × 3 = 15
factor pairs. Cost: ~15 OLS fits, seconds.

**6.2 Iterated-vs-direct comparison.** For each h in the set,
compute `(\hat{P}_1^h, \hat{P}_1^{h-1} \hat{Σ}_1 \hat{P}_1^{(h-1)\top})`
from the existing 1-step fit. Compare element-by-element to (6.1).
Falsification target: if `\hat{P}_1^h \approx \hat{C}_h` AND
iterated-Σ ≈ \hat{Σ}_h within sampling noise, the strict-semigroup
picture (A) survives and (C) is unmotivated for this data.
Confirmation target: structured divergence — drift small, diffusion
large; or both with specific (h, eigenmode) concentration.

**6.3 Direct-factor backtest.** Run a forecast pipeline using
direct `\hat{κ}_h` for each h instead of iterating κ_1. Compare per-HE
bias and per-HE coverage to the current pipeline. Falsification
target: per-HE bias shape unchanged → (C) not the right picture
either, and the framework limitation is deeper than coherence
across factors. Confirmation target: per-HE bias shape *flattens*
(consistent with §5.3), even if MAE doesn't improve overall.

**6.4 Joint inference of `(L, Q)`.** Maximum-likelihood under the
strict (★) parameterization, summing log-likelihoods across h.
Outputs: one `(\hat{L}, \hat{Q})` and per-scale residual
goodness-of-fit. Failure modes are informative: which scales the
joint fit cannot match, which eigenmodes are unstable, whether
convergence is achieved at all.

**6.5 Coherence-defect characterization.** If (6.4) leaves residual
structure, examine `Δ_{h_1, h_2} := \hat{κ}_{h_1 + h_2}
- (\hat{κ}_{h_1} \star \hat{κ}_{h_2})` across scale pairs. Test for:
(i) concentration at specific (h, eigenmode) pairs, (ii) dependence
on origin hour, (iii) dependence on post-cutoff vs. pre-cutoff
state. This step IS the inference of the coherence-defect-as-object
that picture (C) claims is meaningful.

**6.6 If the empirical evidence supports (C):** lift to a
theory-side audit (see §8) and ask whether the framing has known
formal antecedents or is a new structural object.

## 7. What this is NOT claiming

To keep the seed honest about scope:

- **Not claiming the joint multi-horizon SDE-generator-inference
  estimator is novel.** It is essentially Aït-Sahalia / Bibby-
  Sørensen / Hansen-Scheinkman territory in continuous time, and
  Klus-Schütte / Williams-Rowley-Kevrekidis territory in
  Koopman / EDMD. The theory audit will check this.

- **Not claiming the coherence-defect-as-object framing is novel.**
  It may collapse into multi-scale spectral inference, into Hansen-
  Sargent misspecification-robust filtering, into renormalization-
  group reduced-order modelling, or into something else known under
  a name we haven't searched yet.

- **Not claiming the Ontario empirics are sufficient evidence for
  (C) over (A).** The CK drift and the bias-shape invariance are
  *consistent* with (C); they do not yet rule out (A) — the joint
  inference in (6.4) is the falsification test.

- **Not claiming this resolves the registered forward experiment.**
  The frozen spec (`experiment/freeze.py`) is unchanged. This is
  retrospective re-analysis informed by what the registered
  experiment exposed.

- **Not claiming the seed is publishable as-is.** Seeds are pre-
  audit by definition.

## 8. What we need to build before this can be vetted

The theory-side workflow (`Research/Mathematics/Resolvent_Framework/CLAUDE.md`)
has a clear pipeline: seed note → `/audit pure` → either active lead
or honest park. The gate-keeper is the hostile-referee audit. That
workflow assumes a *theory-paper* end product, where novelty against
the mathematical literature is the binding constraint.

Our context is different. We have:

- An empirical companion (this repo) where falsifiable backtests are
  cheap and fast.
- A registered forward experiment (`experiment/freeze.py`) with
  provenance constraints that the theory workflow does not need to
  respect.
- Lab-grade artifacts (pickles, figures, memos) rather than
  formalization-grade artifacts (Lean proofs).
- A different binding constraint: *does the framing produce a
  diagnostic / predictor / structural claim that the data can
  support or refute?* Novelty against the mathematical literature
  matters for the eventual theory paper but not for the question
  "should we run the experiment."

What this seed needs, that we don't yet have:

**(8.1) An applied-analogue of the audit workflow.** We need to know,
before committing implementation effort, whether the seed is:
- Empirically tractable in this dataset (computational cost, sample
  size, identifiability with the available signal-to-noise)
- Consistent with our existing pre-registered structure (does the
  forward experiment need updating? does this conflict with frozen
  spec?)
- Worth doing relative to other open questions (exogenous covariates,
  direct multi-step on raw, the existing dashboard work)

This is closer to "applied research design audit" than "hostile
mathematical referee." Adapting the existing agent set or building
new ones is a separate piece of work; see §9.

**(8.2) Eventually, the theory-side audit.** If §6 produces evidence
for (C), the framing needs the `/audit pure` gate in the theory repo
before being claimed as anything other than an empirical diagnostic.
That step is the hostile-referee novelty audit against the SDE-
generator-inference, Koopman-EDMD, multi-scale-projection, and RG
literature, as the prior draft of this seed already enumerated.

The lab perspective and the theory perspective will *not* agree on
when to stop. Lab considers it done when the experiment runs cleanly
and the diagnostic is established; theory considers it done when the
framing is either novel and proved, or known and cited. Both
verdicts should be obtained before claiming any structural finding;
they answer different questions.

## 9. Next steps in this repo

**(9.1) Stand up an applied-research-audit workflow.** We have a
theory-side agent ecosystem to crib from (literature-scout, devils-
advocate, thesis-advisor, audit pure/applied, editorial-pass) but
the right roles for *applied* research are not identical. Specific
adaptations needed:
- "Skeptical audit" → "experimental design audit" (different
  threat model: not "is this known to mathematicians" but "will
  the experiment cleanly distinguish hypotheses, given our data")
- "Literature scout" — still relevant, but biased toward applied /
  algorithmic literature first
- "Devils advocate" — relevant in both contexts
- New role: "lab-discipline audit" — does this respect the
  registered experiment? the provenance contract? the production-
  vs-diagnostic discipline established this session?
- New role: "applied-vs-theory border guard" — when does a finding
  cross from lab artifact to theory claim, and what's needed to
  cross it cleanly

This is enough work that it gets its own seed and its own audit.
For now: identify the gap, don't fill it inline.

**(9.2) Decide on (6.1) immediately.** It's cheap, falsifiable, and
informative regardless of the theory disposition. Direct h-step
factor estimation requires extending `_build_pre_cutoff` to take a
horizon argument and a thin wrapper around `_global_ols` to fit
`(\hat{C}_h, \hat{Σ}_h)`. The results plot the diffusion at each
scale alongside the iterated counterpart — that's the experiment.

**(9.3) Decide on (6.4) after (6.1).** Joint MLE of `(L, Q)` is a
more substantial piece of code (matrix logarithm initialization,
numerical optimization, parameterization choices to ensure `Q`
positive-semidefinite). It is worth building only if (6.1)–(6.3)
show structural divergence between iterated and direct, i.e., if
the strict semigroup is empirically refused. If not, joint MLE
is just a more complex way to recover the 1-step fit.

**(9.4) Document this seed's lineage.** This seed grew out of:
- `mitacs-clim-gap-nonstationary` (the framework-level
  exhaustion that prompted the lift)
- `mitacs-theta-fibre-simplex` (the CK drift observation)
- `mitacs-rebaseline-facts` (the diffusion-as-second-moment-proxy
  framing in the existing estimator)
- `mitacs-theory-correspondence` (the κ_Q ↔ Resolvent_Framework
  mapping that lets us cross between repos at all)

These memories should be updated to point back to this seed once
the empirical programme starts producing artifacts.

## 10. Honest assessment of the seed

The seed is asking a real question and the empirical companion can
test it cheaply. The conceptual content (relaxed factorization with
diffusion as cross-scale glue) is structured, falsifiable, and not
trivially equivalent to either strict semigroup or no-structure.

Where the seed is weakest:

- The relaxed factorization (C) is described in conceptual terms,
  not in formal terms. "Coherence defect with structure" is not yet
  a definition. The formal content lives downstream of the
  empirical evidence — we should not commit to a formal definition
  before knowing what the data demands.

- The empirical predictions in §5 are at the level of
  "structurally we expect X to deviate from Y" rather than "we
  predict the deviation is K MW with 95% CI [a, b]." Pre-
  registering the experiment requires moving to the latter, which
  is itself a piece of work.

- We have not actually tested whether (6.1) shows what we expect.
  Until it does, this is all framework-level speculation built on
  an empirical pattern (the bias shape) that has *multiple
  candidate explanations*.

Where the seed is strongest:

- It is grounded in empirics that surprised us. We did not start the
  session looking for a framework reformulation; we arrived at one
  by exhausting the levers available within the existing framework
  and finding the bias survived all of them.

- The proposed experiments are cheap and the falsification criteria
  are clear: if (6.1)–(6.3) don't show structural divergence
  between iterated and direct factors, the seed is wrong and the
  framework-level limitation is elsewhere (most likely missing
  exogenous covariates, as `mitacs-postcovid-probe` already
  suggested).

- It does not need to be novel mathematically to be useful in the
  lab. Even if joint multi-horizon SDE generator inference is
  classical, applying it as a diagnostic in this empirical context
  is a contribution to the goal-4 writeup's §5 limitations, and a
  potential improvement to the framework's documented limitations.
