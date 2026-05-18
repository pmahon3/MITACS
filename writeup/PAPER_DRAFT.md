<!--
DRAFT for Simon & Matt. Status: theory track written; application track
scaffolded with [PENDING] markers awaiting verified honed/full runs and
forward-experiment calendar time. Register discipline (agreed
2026-05-18): describe the operator as it is; cite Resolvent_Framework
where concepts originate; make NO novelty assertion in either direction
(neither "we contribute / first" nor "not novel / merely an instance").
Contribution-worthiness is left open for the reader to judge. The three
qualifiers are honest scope statements, not significance disclaimers.
Resolvent_Framework cited at pinned commit dbc7078,
notes/programme/program_overview.md.
-->

# A locally-Gaussian conditional-kernel estimator and its instantiation on Ontario electricity demand

**Authors:** [PLACEHOLDER — author list]
**Status:** working draft

---

## Abstract

[PLACEHOLDER — written last, once the application-track numbers are
verified. Will state: (a) the estimator and its correspondence to the
Resolvent_Framework conditional-regularity kernel κ_Q with three
explicit qualifiers; (b) the rank-1 / Gaussian-proxy structural finding;
(c) the Ontario instantiation, validated backtest, and registered
forward experiment.]

---

## 1. Introduction

This document frames short-horizon electricity demand as a question
about the *conditional law of a reconstructed dynamical state*: given a
delay embedding of recent demand, what is the distribution of the next
state, and what operator does estimating that distribution correspond
to?

This document has two tracks, given equal weight and joined by a
correspondence section.

- **Theory track (§2–§4).** We construct, per state-space anchor, a
  local linear drift `C_j` and a residual diffusion `Σ_j` that together
  define a state-dependent Gaussian Markov kernel on the delay-embedding
  space. We describe the relationship between this constructed object
  and the conditional-regularity kernel `κ_Q` of the Resolvent_Framework
  programme [RF], stating three qualifiers that bound the relationship.
  We then give the central structural finding: for a single-variable
  delay embedding `Σ_j` is rank-1 *by construction*, and the scalar
  innovation it summarises is strongly non-Gaussian — so `Σ_j` is a
  Gaussian second-moment proxy of a heavy-tailed law, which delimits
  what the Gaussian estimator of `κ_Q` can capture.

- **Application track (§5–§7).** We instantiate the estimator on Ontario
  IESO electricity demand: a configuration-driven pipeline
  (de-seasonalisation → day-type tagging → embedding-dimension
  selection → per-anchor fit), an out-of-sample backtest, a
  hash-stamped *registered* forward prediction experiment, and a
  directly-verified finding about the (in)feasibility of a retrospective
  head-to-head against IESO's own published forecasts.

§8 records reproducibility/provenance machinery; §9 collects
claim/method-grade limitations and open questions; §10 fences
development-set preliminary observations whose lower evidentiary grade
is stated explicitly and on which nothing in §1–§9 depends.

*Positioning relative to prior work is deliberately deferred* —
[PLACEHOLDER: prior-work context, for the author group to supply; this
draft makes no comparative-novelty claims].

---

## 2. The estimator

### Setup

Let a scalar observable (here, de-seasonalised hourly demand; §5) be
delay-embedded into `ℝ^d` by lags `x_t = (s_t, s_{t-1}, …, s_{t-(d-1)})`.
For a query state `x` we form a local neighbourhood in the embedding
space and fit, by weighted least squares, a local linear one-step map

    x_next ≈ x · C_j ,        (row convention)

where `j` indexes the anchor (query) and `C_j ∈ ℝ^{d×d}`. The residuals
`Y − X C_j` over the local neighbourhood define a diffusion

    Σ_j = plain μ-centred covariance of (Y − X C_j) ,

*without* a residual kernel. The pair `(C_j, Σ_j)` defines a
state-dependent Gaussian Markov kernel

    Π^(j)(x, ·) = 𝒩(x C_j, Σ_j)

on the embedding state space. `C_j` is the local conditional-mean
(drift) estimate; `Σ_j` is the local conditional second moment
(diffusion).

### Bandwidth selection

The neighbourhood bandwidth (`θ`) is selected by true leave-one-out
cross-validation per anchor. We note for the record that the
Goldenshluger–Lepski-style selector available in the supporting library
is degenerate for the normalised Gaussian kernel used here — its
criterion is monotone in bandwidth with no interior optimum — so it is
not used; the LOO-CV rule is validated against synthetic ground truth
(§4). The diffusion uses the plain (unweighted-by-residual-kernel)
covariance because the kernel-weighted form collapses toward `≈ w²·Q`
and is not the unbiased local second moment.

---

## 3. Correspondence to the Resolvent_Framework conditional-regularity kernel

The Resolvent_Framework programme [RF, §"Conditional regularity and
dynamics"] defines, by Rokhlin disintegration of two observations
`Q, F`, a Markov kernel

    κ_Q(q, ·) = P(F ∈ · | Q = q) ,

described there as "forced by the measure, not chosen", in a register
the programme labels "Disclosure, not construction. The structures are
*in* the measure." (direct quotations, [RF] lines 69–70 and 63).
Indexing over time yields a Markov semigroup
`{Π_t}` (Chapman–Kolmogorov *derived*, not assumed) and Koopman–Perron
duality `∫ K_t g dμ = ∫ g d(P_t^* μ)`; when the kernels are Dirac
measures `K_t` collapses to the classical Koopman operator. The
programme keeps three layers notationally distinct: `κ_Q`
(disintegration / conditional-regularity layer), `{Π_t}` (time-indexed
semigroup), `K_t` (operator layer).

Against that, the object §2 constructs is, stated plainly:

> a finite-sample, parametric, single-step, *locally-Gaussian*
> estimator of the conditional-regularity kernel `κ_Q` at the
> embedding layer, from which the operator-layer objects `K_Δ / P_Δ^*`
> are recoverable in the programme's non-Dirac regime (`Σ_j ≠ 0`).

The delay embedding corresponds to the programme's reconstruction
object; the drift `C_j` to a local linearisation of the
minimal-sufficient factor; the diffusion `Σ_j` to the second moment of
`κ_Q`'s Gaussian projection. `Σ_j ≠ 0` is exactly the programme's
non-Dirac regime; `Σ_j → 0` reproduces the classical-Koopman collapse
the programme names.

### Three qualifiers

These bound the correspondence and are stated as scope, not as
commentary on its significance.

1. **Register.** The programme *discloses* `κ_Q` from the measure; the
   code *constructs* a Gaussian proxy and fits it from finite data.
   Same target object, opposite epistemic register — the relationship
   is "estimator of," not "instance of."

2. **Imposed Gaussianity.** `Π^(j) = 𝒩(x C_j, Σ_j)` coincides with the
   true `κ_Q` only where the local conditional law is itself Gaussian.
   §4 shows this assumption is decisively operative here, not benign.

3. **Single step.** The construction yields only `Π_Δ` (one sampling
   interval). Chapman–Kolmogorov / the full semigroup `{Π_t}` is
   neither composed nor verified; `{Π_t}` in the programme is *derived*,
   whereas here only the single-step slice exists.

The delay embedding, the drift as local linearisation, and the
diffusion as the correct weighted conditional covariance require no
qualifier — these are clean correspondences.

---

## 4. Central structural finding: rank-1 diffusion and the Gaussian-proxy gap

Two facts about `Σ_j` together delimit what the Gaussian estimator of
`κ_Q` can represent for this embedding.

**`Σ_j` is rank-1 by construction.** For a single-variable delay
embedding, coordinates `2..d` of the one-step image are deterministic
shifts of the input coordinates (`Y[:, 1:] = X[:, :-1]` exactly). The
only stochastic content of the one-step map is the scalar
coordinate-0 innovation. Therefore `Σ_j` has (exactly) rank one,
independent of data: the multivariate diffusion is, structurally, a
single scalar innovation variance embedded in `ℝ^{d×d}`. Any
"multi-mode" reading of `Σ_j` is incoherent for this embedding and is
not reported.

**That scalar innovation is strongly non-Gaussian.** The relevant
non-Gaussianity question is therefore *univariate* (the rank-1 fact
makes the multivariate question vacuous). Measured through the
production estimator and validated against synthetic ground truth (§7
gates), the one-step innovation is strongly heavy-tailed: the
production-path re-baseline gives excess kurtosis ≈ 24–33 and tail
ratio ≈ 2.4–3.8, against a clean VAR(1) reference of ≈ 0 / ≈ 1. [These
are the current authoritative production-path values; the slot here
will be updated with the re-confirmed figure and artifact hash from the
verified honed/full run, but the finding and its order of magnitude are
established, not pending.]

**Consequence.** `Σ_j` is a Gaussian second-moment *proxy* of a
strongly heavy-tailed scalar innovation law. This is the operative form
of Qualifier 2: the estimator faithfully captures the conditional mean
(`C_j`) and the conditional *variance* (`Σ_j`), but the conditional
*law* is not Gaussian, so `Π^(j)` is not `κ_Q` itself even in the
large-sample limit — it is the best Gaussian summary of it. We treat
this as a result about the reach of a locally-Gaussian `κ_Q` estimator
on this class of system, not as a defect to be hidden: it is reported,
quantified, and carried as a standing caveat wherever `Π^(j)` is used
predictively.

---

## 5. Application: pipeline and de-seasonalisation

### Pipeline

The Ontario instantiation is a configuration-driven pipeline; all paths
and parameters resolve from a single `config/pipeline.yaml` relative to
the project root (no working-directory dependence). Stages:

1. **Acquire** — IESO public `PUB_Demand` series.
2. **De-seasonalise** (§5.1) — hourly demand → a stationary `zscore`
   series.
3. **Day-type tagging** — `{weekday, saturday, sunday}` using a
   configurable 07:00 day-anchor offset; transitions whose one-step
   target crosses the day-type rollover seam are excluded (without
   this, `Σ_j` is catastrophically ill-conditioned — the estimand is
   effectively forced to be *intra-day*).
4. **Embedding dimension** — selected per day-type by an elbow rule on
   the prediction-skill-vs-dimension curve.
5. **Per-anchor fit** — the §2 estimator.

### 5.1 The z-score transform is a framework coupling, not a preprocessing detail

De-seasonalisation is `z = (D − μ_{m,h}) / σ_{m,h}` where `μ, σ` are a
month×hour-of-day climatology estimated strictly from pre-cutoff data.
This single climatology object is woven through *all four* estimator
layers: it defines the embedded state, the drift fit, the diffusion
(`Σ_j` is the covariance of z-residuals), and the demand-space
round-trip. So the transform is not a benign preprocessing step; it is
an implicit seasonal model coupled into `κ_Q`.

We tested whether this coupling corrupts the dynamics (a leaky implicit
seasonal model — call it "Reading 2") or is a benign invertible
coordinate change ("Reading 1"), by re-running under climatology specs
of varying resolution including the no-transform extreme and measuring
the error structure in demand space. The phase/asymmetry structure was
**invariant** across specs including no-transform — Reading 1 for the
phase axis: the conditional-mean structure is genuine demand dynamics,
not a transform artifact. The tail structure *did* move with seasonal
resolution — a partial Reading 2 confined to the heavy-tail axis, which
is exactly the §4 caveat and is documented as a framework caveat, not a
separate defect. [This result is from a development-set investigation;
its role here is methodological — establishing that the coordinate is
sound before any state-representation honing.]

---

## 6. Application: validated backtest and registered forward experiment

### 6.1 Out-of-sample backtest

We backtest the multi-step (iterated one-step) predictor on real
Ontario ground truth for complete delivery days strictly after the
model-specification cutoff. The leakage guard is enforced directly:
fits and the z-score climatology use only ≤-cutoff data; post-cutoff
actuals enter solely as query history and as scoring ground truth.
Error is reported per horizon `h = 1..24`, with the explicit caveat
that in a day-ahead forecast horizon is collinear with hour-of-day, so
the error structure is *diurnal* (deep night easiest; dawn/dusk demand
ramps hardest), not pure horizon compounding.

The current committed provenanced C1 artifact gives MAE ≈ 786 MW,
MAPE ≈ 4.78% over 500 out-of-sample days, ≈1.5× better than seasonal
persistence at every hour. [These are the current CLAIM-grade values
from the committed provenanced artifact; the artifact hash will be
restated once the in-flight cosmetic header re-emit lands and after any
honing re-freeze. Per-horizon table → Appendix B.]

### 6.2 Registered forward prediction experiment

Because a fair comparison to IESO's published forecast is not
retrospectively recoverable (§6.3), we built a *registered* forward
experiment. A hash-stamped frozen specification captures everything
that determines a forecast — estimator identity, resolved
hyperparameters, data cutoff, code commit, and a content fingerprint of
the pre-cutoff actuals — and is verified on load (tamper-evident). The
predictor is local/lazy (it fits per anchor at prediction time from
≤-cutoff library data), so the freeze pins the function and its inputs
rather than a trained-weight blob, and a cutoff guard enforces that no
post-cutoff data enters a fit. Forecasts are issued forward, scraped
actuals are settled into an append-only write-once ledger (IESO
restates recent demand; revisions are appended, never overwritten), and
scoring is honest-by-construction against persistence baselines with
IESO published values recorded as differently-horizoned context.

[PENDING: forward-experiment results accrue with calendar time;
report once a sufficient settled window exists.]

### 6.3 The retrospective IESO head-to-head is infeasible from public archives

We sought to compare our day-ahead Ontario-demand forecast against
IESO's published day-ahead Ontario-demand forecast on historical
settled days, and establish — by full enumeration of all 133 public
report directories, not a sample — that no public IESO archive is
*both* Ontario-demand-basis *and* day-ahead-issued for historical
dates. Two independent, directly verified blockers: the
Ontario-demand-basis product (Adequacy3 `ForecastOntDemand`) retains
only the end-of-delivery-day version for days old enough to have
settled actuals (wrong horizon); the true day-ahead-issued products
(DATotals, PredispTotals) publish only market totals — not Ontario
demand — where "Total Load" carries a definitional ≈+2485 MW offset
over Ontario Demand (the MarketQuantity enumeration is Total
Energy/Loss/Load/Dispatchable/10S/10N/30R; none is Ontario demand). This is a finding, not a workaround
motivation: it is *why* the registered forward experiment (§6.2) is the
methodologically correct comparison. (Stated for public archives as of
the recorded access date; IESO restructures its public site.)

---

## 7. Validation against synthetic ground truth

The estimator is validated by three gates that call the *production*
code path (not reimplementations), which is what makes them
trustworthy:

1. **Recovery** — a VAR(1) system with known `(A, Q)` is recovered
   within tolerance (drift and diffusion).
2. **Non-Gaussianity** — a Gaussian-innovation VAR(1) reads
   approximately Gaussian; a Student-t-innovation VAR(1) is clearly
   flagged heavy-tailed. This validates the diagnostic that §4's
   finding rests on, against ground truth.
3. **Multi-step composition** — iterated one-step `Π_Δ` is checked
   against the VAR(1) closed form; drift error compounds with horizon
   *by construction* and is reported, not hidden, as an error-growth
   curve within tolerance.

All three gates pass [PENDING: cite the provenanced gate artifact +
its numbers from the verified run].

---

## 8. Reproducibility and provenance

Every claim- or method-grade result is written through a single
provenance envelope that stamps each artifact with two independently
meaningful hashes — an inputs fingerprint (git commit + frozen-spec
hash + library versions + RNG seeds + declared inputs) and a body hash
recomputed on read (tamper-evident) — refuses to write from a dirty
source tree, and carries a mandatory grade banner
(CLAIM / METHOD / INSPECTION-ONLY). Predictor-derived claim artifacts
additionally bind the frozen-spec hash. A requirements analysis
classified every scrutinizable conclusion by grade and provenance need
before any of this machinery was built. Inspection-only exploratory
material is banner-marked and is *not* cited as a result anywhere in
this document.

---

## 9. Limitations and open questions

- **Gaussian-proxy gap (§4).** The standing caveat: `Π^(j)` captures
  conditional mean and variance but not the heavy-tailed conditional
  law. Open: a non-Gaussian local conditional model for `κ_Q`.
- **State representation.** A possible reach limit of the univariate
  estimator at demand ramps is described, with its lower evidentiary
  grade explicitly fenced, in §10 (Preliminary observations) — it is
  *not* a claim-grade limitation and is deliberately not stated as one
  here.
- **Single-step only (Qualifier 3).** The semigroup `{Π_t}` is not
  constructed; multi-step is iteration of `Π_Δ`, validated for error
  growth but not a Chapman–Kolmogorov-verified semigroup.
- **Exogenous information.** The estimator uses no weather/exogenous
  covariates; the backtest error structure suggests this is a
  model-scope ceiling, not a hyperparameter issue.
- **[DEFER-RF] points.** [PLACEHOLDER: residual theory-correspondence
  questions for resolution with the Resolvent_Framework authors;
  several are mooted by the rank-1 finding.]

---

## 10. Preliminary observations (INSPECTION-ONLY — not claim-grade)

> **Evidentiary grade.** Everything in this section is from
> development-set exploratory investigation. It has *not* been
> established through the production-validated code path on
> claim-grade data and is *not* cited as a result anywhere else in
> this document. It is recorded here for collaborator visibility and
> to motivate an open decision — nothing in §1–§9 depends on it. Per
> the project's discipline rule, a diagnostic that has not been
> confirmed through the production path is a hypothesis, not a
> finding.

**Phase-centred conditional-mean bias at demand ramps.** A
development-set diagnostic on a separate (2021–22) inspection window
indicates a systematic, phase-dependent conditional-mean bias: the
local linear map appears to centre between rising and falling demand
phases, under-predicting overnight troughs and over-predicting evening
peaks. The working interpretation is under-embedding / missing phase
state at the ramps (a single per-day-type embedding dimension may be
simultaneously over-embedded at simple overnight hours and
under-embedded at the complex evening ramp). A complementary dev-set
check found a state-space-adaptive bandwidth to be a weak, high-risk
lever that does not touch this bias.

**Status / open decision.** Whether to address this by a
phase/derivative-augmented embedding, a per-hour embedding dimension,
or to accept and document it as a reach limit of the univariate
estimator is an open decision. Any change would be derived strictly on
the development set and locked into a *new* hash-stamped frozen
specification (superseding the current registration) *before* any
forward forecast is scored with the changed model — the freeze
mechanism is the wall between development and the registered test.
[Status: investigation recorded; decision pending — see project tasks
#30/#31.]

---

## Appendices

- **A. Estimator detail.** [PLACEHOLDER: WLS weighting, the LOO-CV θ
  rule, the GL-degeneracy note in full.]
- **B. Per-horizon backtest table.** [PENDING: from the C1 artifact.]
- **C. Provenance requirements matrix.** [Reference:
  `experiment/results/PROVENANCE_REQUIREMENTS.md`.]

---

## References

- **[RF]** Resolvent_Framework programme overview,
  `notes/programme/program_overview.md`, at pinned commit `dbc7078`
  (2026-05-17). Cite the specific commit when this draft is finalised;
  the §"Conditional regularity and dynamics" definitions of `κ_Q`,
  `{Π_t}`, `K_t` and the disclosure-not-construction register are at
  lines 60–77 of that file at that commit.
- [PLACEHOLDER: remaining references — for the author group.]
