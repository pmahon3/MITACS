<!--
DRAFT for Simon & Matt. APPLICATION paper (restructured 2026-05-18 from
an earlier two-track version). The standalone theory/method paper is
DEFERRED, not abandoned — its outline is parked at
writeup/THEORY_PAPER_OUTLINE.md; this application paper must stand alone
and only POSITIONS the theory (compressed §2), pointing to the separate
treatment for numerical demonstrations.

Register discipline (agreed 2026-05-18, unchanged): describe the
operator as it is; cite Resolvent_Framework where concepts originate;
make NO novelty assertion in either direction (neither "we contribute /
first" nor "not novel / merely an instance"). Contribution-worthiness
is left open for the reader. The three qualifiers are honest scope
statements, not significance disclaimers. Resolvent_Framework cited at
pinned commit dbc7078, notes/programme/program_overview.md.

Status: method positioning + application methodology written; numeric
results carry [PENDING] markers awaiting verified honed/full runs and
forward-experiment calendar time.

*** §2 UNDER REVIEW (2026-05-18) — DO NOT SHIP UNCHANGED. ***
§2.1 ("bandwidth selected by true leave-one-out cross-validation") and
§2.2 ("locally-Gaussian estimator") presuppose localization is
operative. A dashboard-surfaced investigation (scratch/
diagnose_theta_loo_railpin.py; memory mitacs-theta-rail-pinning PENDING
block) found the selected θ rails to the grid ceiling on Ontario AND on
the VAR(1) recovery gate, collapsing the kernel to ≈constant (weighted
LS → ~OLS); separately a config grid ceiling (theta_max=60,
config/pipeline.yaml) is ~4× the largest possible anchor distance.
Whether the rule fails or the data is genuinely near-linear is
UNRESOLVED pending a locality-probing synthetic. §2/§3 "locally"
wording may need softening; the §3 Gaussian-proxy story may STRENGTHEN.
Do not finalize §2 until resolved.
-->

# Empirical conditional-kernel forecasting of Ontario electricity demand: a registered prediction experiment

**Authors:** [PLACEHOLDER — author list]
**Status:** working draft (application paper; companion method paper in preparation)

---

## Abstract

[PLACEHOLDER — written last, once the application numbers are verified.
Will state: (a) a delay-embedding local-Gaussian conditional-kernel
predictor instantiated on Ontario IESO demand; (b) the validated
out-of-sample backtest; (c) the directly-verified infeasibility of a
retrospective IESO head-to-head and the registered forward experiment
built in response; (d) the rank-1 / Gaussian-proxy structural caveat
that bounds every predictive claim. The method's theoretical
positioning relative to the Resolvent_Framework programme is summarised
here and treated in depth in a companion paper.]

---

## 1. Introduction

This paper frames short-horizon Ontario electricity-demand forecasting
as a question about the *conditional law of a reconstructed dynamical
state*: given a delay embedding of recent demand, what is the
distribution of the next state? We instantiate a local-Gaussian
conditional-kernel estimator of that law on IESO data and evaluate it
as a *prediction experiment* — with an out-of-sample backtest, a
hash-stamped registered forward experiment, and an explicit accounting
of what can and cannot be compared against IESO's own published
forecasts.

The estimator has a precise relationship to the conditional-regularity
kernel `κ_Q` of the Resolvent_Framework programme [RF]. That
relationship — and its numerical demonstration on controlled systems —
is the subject of a **companion method paper in preparation**; this
paper states the correspondence compactly (§2) only insofar as it is
needed to interpret the applied object honestly, and is otherwise
self-contained.

The paper proceeds: §2 the estimator and its compact theoretical
positioning; §3 the structural caveat (rank-1 diffusion / Gaussian
proxy) that bounds every predictive number here; §4 the Ontario
pipeline and the z-score framework coupling; §5 the validated backtest
and the registered forward experiment; §6 the IESO head-to-head
infeasibility finding; §7 synthetic-ground-truth validation; §8
reproducibility/provenance; §9 claim/method-grade limitations; §10
fenced development-set preliminary observations.

*Positioning relative to prior work is deliberately deferred* —
[PLACEHOLDER: prior-work context, for the author group to supply; this
draft makes no comparative-novelty claims].

---

## 2. The estimator and its theoretical positioning

### 2.1 Construction

Let a scalar observable (here, de-seasonalised hourly demand; §4) be
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

on the embedding state space: `C_j` is the local conditional-mean
(drift) estimate, `Σ_j` the local conditional second moment
(diffusion). The neighbourhood bandwidth (`θ`) is selected by true
leave-one-out cross-validation per anchor. (For the record: the
Goldenshluger–Lepski-style selector in the supporting library is
degenerate for the normalised Gaussian kernel used here — its criterion
is monotone in bandwidth with no interior optimum — so it is not used;
the LOO-CV rule is validated against synthetic ground truth, §7. The
diffusion uses the plain, not residual-kernel-weighted, covariance
because the weighted form collapses toward `≈ w²·Q` and is not the
unbiased local second moment.) Estimator internals are detailed in
Appendix A; the companion method paper treats them in full.

### 2.2 Relationship to the Resolvent_Framework kernel `κ_Q`

The Resolvent_Framework programme [RF, §"Conditional regularity and
dynamics"] defines, by Rokhlin disintegration of two observations
`Q, F`, a Markov kernel `κ_Q(q, ·) = P(F ∈ · | Q = q)`, described there
as "forced by the measure, not chosen", in a register the programme
labels "Disclosure, not construction. The structures are *in* the
measure." (direct quotations, [RF] lines 69–70 and 63). Indexing over
time yields a Markov semigroup `{Π_t}` (Chapman–Kolmogorov *derived*,
not assumed) and Koopman–Perron duality
`∫ K_t g dμ = ∫ g d(P_t^* μ)`; when the kernels are Dirac measures
`K_t` collapses to the classical Koopman operator. The programme keeps
three layers notationally distinct: `κ_Q` (disintegration layer),
`{Π_t}` (semigroup), `K_t` (operator layer).

The object §2.1 constructs is, stated plainly:

> a finite-sample, parametric, single-step, *locally-Gaussian*
> estimator of the conditional-regularity kernel `κ_Q` at the
> embedding layer, from which the operator-layer objects `K_Δ / P_Δ^*`
> are recoverable in the programme's non-Dirac regime (`Σ_j ≠ 0`).

The delay embedding corresponds to the programme's reconstruction
object; `C_j` to a local linearisation of the minimal-sufficient
factor; `Σ_j` to the second moment of `κ_Q`'s Gaussian projection.
`Σ_j ≠ 0` is the non-Dirac regime; `Σ_j → 0` reproduces the
classical-Koopman collapse. **Three qualifiers** bound this
relationship and are stated as scope, not as commentary on its
significance:

1. **Register.** The programme *discloses* `κ_Q` from the measure; this
   work *constructs* a Gaussian proxy and fits it from finite data.
   Same target object, opposite epistemic register — "estimator of,"
   not "instance of."
2. **Imposed Gaussianity.** `Π^(j) = 𝒩(x C_j, Σ_j)` coincides with the
   true `κ_Q` only where the local conditional law is itself Gaussian.
   §3 shows this is decisively operative here, not benign.
3. **Single step.** The construction yields only `Π_Δ` (one sampling
   interval). The full semigroup `{Π_t}` is neither composed nor
   verified; in the programme `{Π_t}` is *derived*, whereas here only
   the single-step slice exists.

The delay embedding, the drift as local linearisation, and the
diffusion as the correct weighted conditional covariance require no
qualifier. *Numerical demonstration of this correspondence on
controlled systems is the subject of the companion method paper; this
paper carries only what is needed to interpret the Ontario results.*

---

## 3. Structural caveat: rank-1 diffusion and the Gaussian-proxy gap

This bounds every predictive number in this paper and is stated up
front, not buried in limitations.

**`Σ_j` is rank-1 by construction.** For a single-variable delay
embedding, coordinates `2..d` of the one-step image are deterministic
shifts of the input coordinates (`Y[:, 1:] = X[:, :-1]` exactly). The
only stochastic content of the one-step map is the scalar
coordinate-0 innovation; therefore `Σ_j` has exactly rank one,
independent of data — the multivariate diffusion is structurally a
single scalar innovation variance embedded in `ℝ^{d×d}`. Any
"multi-mode" reading of `Σ_j` is incoherent for this embedding and is
not reported.

**That scalar innovation is strongly non-Gaussian.** The relevant
non-Gaussianity question is therefore *univariate* (the rank-1 fact
makes the multivariate question vacuous). Measured through the
production estimator and validated against synthetic ground truth (§7),
the one-step innovation is strongly heavy-tailed: the production-path
re-baseline gives excess kurtosis ≈ 24–33 and tail ratio ≈ 2.4–3.8,
against a clean VAR(1) reference of ≈ 0 / ≈ 1. [Current authoritative
production-path values; the slot will be updated with the re-confirmed
figure and artifact hash from the verified honed/full run — the finding
and its order of magnitude are established, not pending.]

**Consequence.** `Σ_j` is a Gaussian second-moment *proxy* of a
strongly heavy-tailed scalar innovation law. The estimator faithfully
captures the conditional mean (`C_j`) and variance (`Σ_j`), but the
conditional *law* is not Gaussian, so `Π^(j)` is not `κ_Q` itself even
in the large-sample limit — it is the best Gaussian summary of it. We
report and quantify this rather than hide it, and carry it as a
standing caveat wherever `Π^(j)` is used predictively (notably the
interval coverage in §5).

---

## 4. Ontario pipeline and the z-score framework coupling

### 4.1 Pipeline

The instantiation is a configuration-driven pipeline; all paths and
parameters resolve from a single `config/pipeline.yaml` relative to the
project root (no working-directory dependence). Stages: **acquire**
IESO public `PUB_Demand`; **de-seasonalise** (§4.2) hourly demand → a
stationary `zscore`; **day-type tag** `{weekday, saturday, sunday}`
using a configurable 07:00 day-anchor offset (transitions whose
one-step target crosses the day-type rollover seam are excluded —
without this `Σ_j` is catastrophically ill-conditioned, so the estimand
is effectively forced to be *intra-day*); **embedding dimension**
selected per day-type by an elbow rule on the
prediction-skill-vs-dimension curve; **per-anchor fit** of the §2
estimator.

### 4.2 The z-score transform is a framework coupling, not a preprocessing detail

De-seasonalisation is `z = (D − μ_{m,h}) / σ_{m,h}` with `μ, σ` a
month×hour-of-day climatology estimated strictly from pre-cutoff data.
This single climatology object is woven through *all four* estimator
layers: the embedded state, the drift fit, the diffusion (`Σ_j` is the
covariance of z-residuals), and the demand-space round-trip. It is
therefore an implicit seasonal model coupled into `κ_Q`, not a benign
preprocessing step.

We tested whether this coupling corrupts the dynamics (a leaky implicit
seasonal model — "Reading 2") or is a benign invertible coordinate
change ("Reading 1"), by re-running under climatology specs of varying
resolution including the no-transform extreme and measuring error
structure in demand space. The phase/asymmetry structure was
**invariant** across specs including no-transform — Reading 1 for the
phase axis: the conditional-mean structure is genuine demand dynamics,
not a transform artifact. The tail structure *did* move with seasonal
resolution — a partial Reading 2 confined to the heavy-tail axis, which
is exactly the §3 caveat and is documented as a framework caveat, not a
separate defect. [Development-set investigation; its role here is
methodological — establishing the coordinate is sound before any
state-representation honing.]

---

## 5. Validated backtest and registered forward experiment

### 5.1 Out-of-sample backtest

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
persistence at every hour. [Current CLAIM-grade values from the
committed provenanced artifact; the artifact hash will be restated
after any honing re-freeze. Per-horizon table → Appendix B.] Interval
coverage is reported with the §3 Gaussian-proxy caveat made explicit
(a Gaussian interval around a heavy-tailed innovation is expected to
under-cover; this is disclosed, not hidden).

### 5.2 Registered forward prediction experiment

Because a fair comparison to IESO's published forecast is not
retrospectively recoverable (§6), we built a *registered* forward
experiment. A hash-stamped frozen specification captures everything
that determines a forecast — estimator identity, resolved
hyperparameters, data cutoff, code commit, and a content fingerprint of
the pre-cutoff actuals — and is verified on load (tamper-evident). The
predictor is local/lazy (it fits per anchor at prediction time from
≤-cutoff library data), so the freeze pins the function and its inputs
rather than a trained-weight blob, and a cutoff guard enforces that no
post-cutoff data enters a fit. Forecasts are issued forward; scraped
actuals are settled into an append-only write-once ledger (IESO
restates recent demand; revisions are appended, never overwritten);
scoring is honest-by-construction against persistence baselines with
IESO published values recorded as differently-horizoned context.

[PENDING: forward-experiment results accrue with calendar time; report
once a sufficient settled window exists.]

---

## 6. The retrospective IESO head-to-head is infeasible from public archives

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
Energy/Loss/Load/Dispatchable/10S/10N/30R; none is Ontario demand).
This is a finding, not a workaround motivation: it is *why* the
registered forward experiment (§5.2) is the methodologically correct
comparison. (Stated for public archives as of the recorded access
date; IESO restructures its public site.)

---

## 7. Validation against synthetic ground truth

The estimator is validated by three gates that call the *production*
code path (not reimplementations), which is what makes them
trustworthy:

1. **Recovery** — a VAR(1) system with known `(A, Q)` is recovered
   within tolerance (drift and diffusion).
2. **Non-Gaussianity** — a Gaussian-innovation VAR(1) reads
   approximately Gaussian; a Student-t-innovation VAR(1) is clearly
   flagged heavy-tailed. This validates the diagnostic that §3's
   caveat rests on, against ground truth.
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

- **Gaussian-proxy gap (§3).** The standing caveat: `Π^(j)` captures
  conditional mean and variance but not the heavy-tailed conditional
  law. Open: a non-Gaussian local conditional model for `κ_Q`.
- **State representation.** A possible reach limit of the univariate
  estimator at demand ramps is described, with its lower evidentiary
  grade explicitly fenced, in §10 — *not* a claim-grade limitation and
  deliberately not stated as one here.
- **Single-step only (Qualifier 3).** The semigroup `{Π_t}` is not
  constructed; multi-step is iteration of `Π_Δ`, validated for error
  growth but not a Chapman–Kolmogorov-verified semigroup. (The
  companion method paper is the natural home for the semigroup
  treatment.)
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
  rule, the GL-degeneracy note in full. The companion method paper is
  the primary home for this; reproduce here only what the application
  reader needs.]
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
- **[companion method paper]** [PLACEHOLDER — in preparation; the
  standalone treatment of the estimator and its `κ_Q` correspondence
  with numerical demonstrations on controlled systems. Outline:
  `writeup/THEORY_PAPER_OUTLINE.md`.]
- [PLACEHOLDER: remaining references — for the author group.]
