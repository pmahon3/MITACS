# Literature scout: stratified Patra–Sen as a conditional σ-algebra adequacy diagnostic

**Date:** 2026-05-26
**Scout:** literature-scout (applied side)
**Re-audits:** the disintegration note's Phase-2 audit row
*"Patra–Sen conditional diagnostic | Not published | Genuinely absent"*
(`~/Research/Mathematics/Resolvent_Framework/notes/unsorted/disintegration_diagnostic.md`,
PARKED 2026-05-26)
**Verdict:** **DONE BEFORE**

---

## TL;DR

The disintegration note's specific claim —

> "No published work stratifies by conditioning state to detect
> state-dependent mixture structure in conditional kernels."

— is **falsified** by **Deb, Saha, Guntuboyina, and Sen (2022 JASA;
arXiv:1810.07897, 2018)**, "Two-Component Mixture Model in the
Presence of Covariates." This paper, co-authored by Sen of
Patra–Sen (2016), generalizes the Patra–Sen two-component mixture
estimator to allow the signal proportion π* to depend on covariates
x, develops a tuning-parameter-free NPMLE for π*(x), and in
**Section 6 ("Are the covariates at all important?")** explicitly
addresses the σ-algebra adequacy question — "first check that
[the unconditional two-groups model] is inadequate. Only then does
it make sense to model the dependence between Y and X." They
reduce this to a distance-covariance independence test (Székely
et al. 2007).

The methods-novelty claim does not survive. The σ-algebra-adequacy
framing — interpreting variation in α̂(s) across strata as evidence
for a missing conditioning variable — is a relabeling of a question
the statistics community has been answering since at least
Scott et al. (2015 JASA, FDRreg) and rigorously solved by
Deb et al. (2022 JASA).

---

## Disintegration note's claim (verbatim)

From `disintegration_diagnostic.md`, §"Patra–Sen conditional
extension (genuinely absent from lit)":

> **This is the novel applied contribution that survives the audit.**
>
> Nobody has applied the Patra–Sen (2016) two-component mixture
> estimator to *conditional* distributions as a σ-algebra adequacy
> diagnostic. The idea: ... for each stratum s, collect the
> standardised residuals of κ̂_Q within s and apply Patra–Sen to
> obtain α̂_L(s) ... If α̂_L(s) varies systematically with s:
> evidence for *noisy structure*.
>
> **Why it's novel.** Patra–Sen (2016) operates on unconditional
> (marginal) distributions. Arias-Castro et al. (2021) extend to
> shape-constrained backgrounds but remain unconditional. No
> published work stratifies by conditioning state to detect
> state-dependent mixture structure in conditional kernels.

The novelty claim has two parts:
1. **No published Patra–Sen extension to π depending on
   conditioning state.** (Refuted by Deb et al. 2018/2022.)
2. **No published use of this as a σ-algebra adequacy diagnostic.**
   (Refuted by Section 6 of Deb et al. 2018/2022.)

---

## Names searched

Primary names:
- `Patra Sen mixture conditional`
- `Patra-Sen covariate stratified`
- `local mixture proportion estimation`
- `covariate-dependent mixing proportion estimation`
- `conditional contamination two-component mixture`
- `mixing proportion varying covariate`
- `Young Hunter Huang Yao semiparametric mixture`

Adjacent / alternative names (the lab's "renamed-elsewhere" failure
mode the advisor flagged):
- `conditional PIT local calibration stratified`
- `unobserved heterogeneity diagnostic residual stratified`
- `score test neglected heterogeneity mixture`
- `Heckman Singer unobserved heterogeneity test`
- `Chesher 1984 score test`
- `omitted variable test heteroscedasticity residual mixture`
- `conditioning set adequacy diagnostic`
- `conditional independence test non-Gaussian residual`
- `two-groups model covariates Sun Cai`
- `FDRreg NPMLEmix time-series application`
- `sigma-algebra enrichment diagnostic`
- `local non-Gaussianity residual diagnostic time series stratified`

## Communities checked

| Community | Coverage | Hit? |
|-----------|----------|------|
| Statistics (JASA, JRSSB, AoS) | Patra–Sen lineage + multiple-testing two-groups + Bayesian conditional density | **YES (dispositive)** |
| Probabilistic forecasting (Gneiting/Tsyplakov/Diebold lineage; arXiv) | Cal-PIT and local-PIT calibration diagnostics | Adjacent; different formalism |
| Econometrics (Econometrica, J Econ) | Chesher 1984, Heckman–Singer 1984, neglected-heterogeneity score tests | Adjacent; different machinery |
| Causal inference | Conditional-independence tests under misspecification (RBPT etc.) | Adjacent; not the same diagnostic |
| Applied ML (NeurIPS/ICML/JMLR) | covariate-dependent mixture-of-experts | Estimation, not adequacy diagnosis |
| Time-series / state-space | None apply Patra–Sen-class diagnostics | No prior |
| Energy / electricity load forecasting | Heavy-tailed forecasting literature | No application of this diagnostic |
| Pure math | Lower priority; the theory-side scout's Phase-2 audit already cataloged the obstruction's theorem sources | Confirmed for ID |

---

## Findings

### Dispositive prior (the lab's idea, done first)

**Deb, Saha, Guntuboyina, Sen (2022).** "Two-Component Mixture
Model in the Presence of Covariates." *Journal of the American
Statistical Association* **117**(540).
arXiv:1810.07897 (2018, rev. 2019).
DOI: 10.1080/01621459.2021.1888739.
R package: `NPMLEmix` (CRAN archived 2022; sourced via
GitHub `NabarunD/NPMLEmix`).

> **Relevance.** Generalizes the Patra–Sen (2016) two-component
> mixture estimator to a model where the signal proportion π* is
> a function of covariates x. Develops a tuning-parameter-free
> nonparametric MLE for the pair (π*, f_1*) using EM with a
> Kiefer–Wolfowitz inner step; derives near-parametric Hellinger
> rates of convergence. Discusses three classes for Π: constants
> (≡ Patra–Sen 2016), nondecreasing (Π_↑), and parametric link
> (logistic / probit, Π_g).

> **Section 6 — direct overlap with the lab's framing.** Verbatim:
> "Till now we have focused on the estimation of parameters
> assuming that model (6) holds. A basic and important question
> that we have not yet addressed is: 'Do the covariates provide
> any information at all on the distribution of Y'? Put another
> way, we must first check that model (1) is inadequate. Only
> then does it make sense to model the dependence between Y and
> X, as in (6). In statistical parlance, this reduces to testing
> for independence between X and Y."

> **Gap from the lab's proposal:** the lab proposed *stratifying*
> by s and running unconditional Patra–Sen per stratum; Deb et al.
> *jointly* estimate π*(x) continuously over x with regularity
> (monotone or parametric link). The lab's stratified estimator
> is the discretized, less efficient special case of what Deb
> et al. handle continuously. There is no methodological gap
> here — Deb et al. is strictly more general.

> **Author overlap that closes the question:** Bodhisattva Sen
> co-authored both Patra–Sen (2016) and Deb et al. (2022). The
> covariate-dependent generalization was published in JASA by the
> same statistical machinery's originator. This is not a niche
> result the disintegration note's audit could be excused for
> missing — it is the canonical follow-up.

### Earlier prior (the same idea with logistic link, 2015)

**Scott, Kelly, Smith, Zhou, Kass (2015).** "False Discovery Rate
Regression: an application to neural synchrony detection in primary
visual cortex." *Journal of the American Statistical Association*
**110**(510), 459–471.
arXiv:1307.3495 (2013).
DOI: 10.1080/01621459.2014.990973. R package: `FDRreg`.

> **Relevance.** Models π* as a logistic function of test-level
> covariates and estimates this *jointly* with f_1* in the
> two-groups model. Explicitly motivates the work as: "many
> large-scale screening problems have auxiliary information about
> each test available, [where] a combined analysis can lead to
> poorly calibrated error rates within different subsets of the
> experiment." This is the σ-algebra adequacy question stated in
> multiple-testing language: subgroups of tests have different
> signal rates, the constant-π model is wrong, condition on the
> auxiliary info.

> **Gap from the lab's proposal:** parametric logistic link only
> (not nonparametric / shape-constrained); does not derive
> Hellinger rates; does not provide the explicit adequacy test
> (that's added by Deb et al. 2018/2022 §6). But chronologically
> this is the earlier publication and is what Deb et al. cite as
> their direct predecessor.

### Foundational anchor (the disintegration note's correct citation)

**Patra, Sen (2016).** "Estimation of a Two-Component Mixture
Model with Applications to Multiple Testing." *Journal of the
Royal Statistical Society Series B* **78**(4).
arXiv:1204.5488 (2012).

> Unconditional two-component mixture with known F_0; tuning-
> parameter-free estimator of (π, F_1*) maximizing π. The
> disintegration note's anchor; cited correctly there.

### Confirmed *not* the lab's idea

**Arias-Castro, Jiang (2021).** "Extending the Patra-Sen Approach
to Estimating the Background Component in a Two-Component Mixture
Model." arXiv:2106.13925.

> **Confirms disintegration note:** this extension relaxes
> assumptions on F_0 (symmetric, monotone, log-concave) but
> remains *unconditional*. Does not address the covariate /
> stratification question.

### Adjacent (different formalism, same conceptual move)

**Dey, Zhao, Lee, Izbicki (2022).** "Towards Instance-Wise
Calibration: Local Amortized Diagnostics and Reshaping of
Conditional Densities" ("Cal-PIT"). arXiv:2205.14568.

> **Relevance.** Stratifies a calibration diagnostic (PIT) by
> conditioning state x — explicitly regresses PIT scores against
> x to detect *local* miscalibration. This is the same *kind* of
> move ("stratify a diagnostic by conditioning state") in a
> different formalism (PIT calibration rather than mixture-
> proportion estimation).
>
> **Gap from the lab's proposal:** doesn't use Patra–Sen
> machinery, doesn't estimate a non-Gaussian mixture fraction.
> Targets calibration, not σ-algebra-adequacy interpretation.
> But it is another instance of "this kind of conditional
> diagnostic is not novel."

### Adjacent (different machinery, same question)

**Chesher (1984).** "Testing for Neglected Heterogeneity."
*Econometrica* **52**(4), 865–872.

> **Relevance.** The canonical score test for whether an
> econometric model has neglected unobserved heterogeneity —
> i.e., whether the conditioning set σ(X) is adequate to
> explain the response. Recognized (in Chesher's paper) to
> coincide with White's Information Matrix test.
>
> **Gap from the lab's proposal:** uses second-moment / score
> machinery, not a non-Gaussian-mixture decomposition. Tests
> for *parameter* heterogeneity, not for an unobserved mixing
> variable's mixture-artifact in the residual distribution.
> Different statistical machinery, conceptually adjacent.

**Heckman, Singer (1984).** "A Method for Minimizing the Impact
of Distributional Assumptions in Econometric Models for Duration
Data." *Econometrica* **52**(2), 271–320.

> **Relevance.** Foundational duration-model treatment of the
> structured-noise vs noisy-structure obstruction (cited in
> disintegration note). Their resolution: time-varying covariates
> or repeated spells. Not a Patra–Sen-style mixture-fraction
> diagnostic.

### Estimation but not diagnostic (close-call near-misses)

**Young, Hunter (2010).** "Mixtures of regressions with predictor-
dependent mixing proportions." *Computational Statistics & Data
Analysis* **54**(10), 2253–2266.

**Huang, Yao (2012).** "Mixture of Regression Models With Varying
Mixing Proportions: A Semiparametric Approach." *JASA* **107**(498),
711–724.

> **Relevance.** Both *fit* a mixture-of-regressions model where
> the mixing proportion is a smooth function of a covariate
> (kernel regression / nonparametric). They are not Patra–Sen-
> based and do not use π̂(x)'s variation as an adequacy
> diagnostic — they are fitting tools.
>
> **Gap from the lab's proposal:** estimation rather than
> diagnostic; assume a parametric component family rather than
> Patra–Sen's nonparametric-with-known-F_0 setup. But these
> are the older covariate-varying mixture references that
> chronologically and conceptually precede Scott et al. (2015)
> and Deb et al. (2018).

---

## Closest prior work (single citation)

**Deb, N., Saha, S., Guntuboyina, A., Sen, B.** (2022).
"Two-Component Mixture Model in the Presence of Covariates."
*Journal of the American Statistical Association* **117**(540).
arXiv:1810.07897 (2018).

**Verbatim overlap.** Their Section 6 question — "first check
that [the unconditional two-groups model] is inadequate. Only then
does it make sense to model the dependence between Y and X" — is
the lab's σ-algebra-adequacy question stated in the same
language. They reduce it to a distance-covariance independence
test (Székely et al. 2007).

**Gap.** The lab proposes per-stratum unconditional Patra–Sen
+ interpretation of α̂_L(s) variation; Deb et al. provide a
strictly more general estimator (joint NPMLE of π*(x)) plus a
formal adequacy test. The σ-algebra-adequacy *vocabulary* (from
measure theory) is not in Deb et al.; the *operational content* is.
Vocabulary is not a methods contribution.

---

## Assessment

### Verdict: **DONE BEFORE**

The disintegration note's claim that no published work has applied
Patra–Sen-class estimators to conditional / covariate-stratified
settings is false. The published priors are in JASA (top
statistics journal), have a maintained R package, and have a
co-author overlap with Patra–Sen (2016) itself.

The σ-algebra-adequacy *framing* is unique to the disintegration
note. But framing is not protectable as a methods contribution.
The "novel applied contribution that survives the audit"
identified in the disintegration note (§"What survives", item (a))
does not survive this second-pass audit.

### What this means for the MITACS workflow

The disintegration note's Phase-2 audit row should be amended:

| Component | Status | Reference |
|-----------|--------|-----------|
| ~~Patra–Sen conditional diagnostic~~ | ~~Not published~~ | ~~Genuinely absent~~ |
| Patra–Sen with covariate-dependent π* | Published | **Deb, Saha, Guntuboyina, Sen (2022 JASA; arXiv:1810.07897)** |
| Earlier covariate-dependent two-groups | Published | Scott et al. (2015 JASA; arXiv:1307.3495) |
| Adequacy-of-conditioning test | Published | Deb et al. (2022) Section 6 + distance covariance |

The methods-novelty path identified as surviving the disintegration
note's audit is closed. What remains for the MITACS work:

- **Application use of Deb et al. (2022) / NPMLEmix on the
  Ontario κ_Q residuals** is a legitimate applied use of an
  existing method — not a methods contribution, but a substantive
  empirical analysis. The R package is archived (CRAN 2022) but
  reachable on GitHub; usability work would be required.
- **The σ-algebra-adequacy *interpretation*** could be cited as
  the measure-theoretic reading of an existing applied procedure,
  but it would not justify a methods paper.
- **The structured-noise / noisy-structure resolution program**
  is, per the disintegration note's own Phase-2 audit and this
  scout, completely covered by existing literature: the
  obstruction theorem is published (Bergna 2026, Allahverdyan
  2020, Heckman–Singer 1984, Hu-Xie-Zhang-Zhou 2026), and the
  resolution diagnostics are published (Deb et al. 2022, Scott
  et al. 2015 for adequacy; Bergna 2026 for synthetic-prior tie-
  breaking; Heckman–Singer 1984 for time-variation tie-breaking).

### Confidence: **HIGH**

Justification:
1. The dispositive paper is in JASA (the highest-visibility venue
   for this kind of statistics methodology) with a maintained R
   package and a GitHub repo.
2. The dispositive paper is co-authored by Sen (of Patra–Sen).
   This makes false-negative ("we missed the paper") very
   unlikely — the canonical extension was done by the same
   author.
3. Section 6 of the dispositive paper *verbatim* states the
   adequacy question the lab calls "σ-algebra adequacy."
4. The arXiv search across multiple keyword combinations
   converges on the same small set of references; no plausible
   competing candidates went unexamined.
5. The advisor flagged this as DONE BEFORE rather than NOVEL IN
   FORM after seeing the same evidence; both passes converged.

### What a deeper search would add

Marginal value:

- **Forward citations of NPMLEmix / Deb et al. (2022).** Could
  surface a time-series / electricity-forecasting application
  that already does what the MITACS work would do. None
  surfaced in this scout's NPMLEmix search; CRAN-archived
  package status suggests low downstream uptake. Gold-plating;
  unlikely to change verdict.
- **Time-series-econometrics search for "neglected heterogeneity"
  in forecasting residuals.** The IF lineage (Hansen, White) is
  vast; this scout sampled only the canonical Chesher 1984 entry.
  More entries would not change the verdict — they would
  reinforce the "different machinery, same question" finding.
- **Paywall checks.** The dispositive JASA paper is paywalled
  (Tandfonline returned 403); arXiv preprint is open access and
  was the source of all verbatim quotes above. The 2022
  published version may have minor revisions but the
  identifiability + Section 6 framing is in the 2019 arXiv.
  No paywall blocked the dispositive evidence.

### Adjacent line worth tracking

**Local/conditional PIT calibration** (Cal-PIT lineage —
Dey et al. 2022 arXiv:2205.14568; Tsyplakov auto-calibration
lineage; Allen et al. 2024 arXiv:2506.13687 on tail-calibration
training) is a vigorous current literature doing
"stratify-a-diagnostic-by-conditioning-state" with different
machinery (PIT regression rather than mixture decomposition).
The MITACS work's marginal-PIT chi² = 22,329 finding (S5) is
in this literature's natural territory and a forward MITACS
applied contribution might fruitfully cite this lineage. But
this is *adjacent*, not *prior* — it does not occupy the
Patra–Sen-class methods slot.

---

## Lab context the scout incorporated

- MITACS marginal κ_1 α̂_L ≥ 0.49–0.74, orthogonal to z-state
  (from `mitacs-qualifier-exhaustion.md`).
- Rebaseline κ_1 excess kurtosis 24–33
  (from `mitacs-rebaseline-facts.md`).
- Disintegration note (parked 2026-05-26) identifies three
  resolution paths and three theorem-source citations
  (Bergna et al. 2026 arXiv:2605.06413, verified;
   Allahverdyan 2020 arXiv:2002.07884, verified;
   Hu, Xie, Zhang, Zhou 2026 arXiv:2506.05116, verified;
   Heckman–Singer 1984; Manski 2003; Blackwell 1951).
- Q1 SETTLED finding (S8): Student-t kernel reduces marginal
  χ² 16× but does not calibrate; ν per day-type stable
  (4.19/4.62/4.61); D_RATIO 10× shrinkage relative to v2.

Anchor-ID verification log:
- arXiv:2605.06413 — Bergna, Depeweg, Hernández-Lobato,
  "Decoupled PFNs: Identifiable Epistemic-Aleatoric
  Decomposition via Structured Synthetic Priors" — confirms
  identifiability proposition + synthetic-prior tie-breaker.
- arXiv:2506.05116 — Hu, Xie, Zhang, Zhou, "The Spurious Factor
  Dilemma: Robust Inference in Heavy-Tailed Elliptical Factor
  Models" — confirms factor-model obstruction + magnification
  diagnostic.
- arXiv:2002.07884 — Allahverdyan, "Observational
  nonidentifiability, generalized likelihood and free energy" —
  confirms mixture-model obstruction.
- arXiv:2106.13925 — Arias-Castro, Jiang, "Extending the
  Patra-Sen Approach to Estimating the Background Component in a
  Two-Component Mixture Model" — confirms unconditional
  extension only; consistent with disintegration note.

All four anchors verified against the disintegration note's
attributions. No corrections required.

---

## Source list

1. Deb, N., Saha, S., Guntuboyina, A., Sen, B. (2022).
   "Two-Component Mixture Model in the Presence of Covariates."
   *JASA* **117**(540).
   <https://arxiv.org/abs/1810.07897> ·
   <https://doi.org/10.1080/01621459.2021.1888739> ·
   <https://github.com/NabarunD/NPMLEmix>

2. Scott, J. G., Kelly, R. C., Smith, M. A., Zhou, P., Kass, R. E.
   (2015). "False Discovery Rate Regression."
   *JASA* **110**(510), 459–471.
   <https://arxiv.org/abs/1307.3495> ·
   <https://doi.org/10.1080/01621459.2014.990973> ·
   <https://github.com/jgscott/FDRreg>

3. Patra, R. K., Sen, B. (2016). "Estimation of a Two-Component
   Mixture Model with Applications to Multiple Testing."
   *JRSSB* **78**(4).
   <https://arxiv.org/abs/1204.5488>

4. Arias-Castro, E., Jiang, H. (2021). "Extending the Patra-Sen
   Approach to Estimating the Background Component in a Two-
   Component Mixture Model." <https://arxiv.org/abs/2106.13925>

5. Dey, B., Zhao, D., Lee, J., Izbicki, R. (2022). "Towards
   Instance-Wise Calibration: Local Amortized Diagnostics and
   Reshaping of Conditional Densities" (Cal-PIT).
   <https://arxiv.org/abs/2205.14568>

6. Chesher, A. (1984). "Testing for Neglected Heterogeneity."
   *Econometrica* **52**(4), 865–872.

7. Heckman, J. J., Singer, B. (1984). "A Method for Minimizing
   the Impact of Distributional Assumptions in Econometric Models
   for Duration Data." *Econometrica* **52**(2), 271–320.

8. Young, D. S., Hunter, D. R. (2010). "Mixtures of regressions
   with predictor-dependent mixing proportions." *CSDA*
   **54**(10), 2253–2266.

9. Huang, M., Yao, W. (2012). "Mixture of Regression Models With
   Varying Mixing Proportions: A Semiparametric Approach."
   *JASA* **107**(498), 711–724.

10. Bergna, R., Depeweg, S., Hernández-Lobato, J. M. (2026).
    "Decoupled PFNs: Identifiable Epistemic-Aleatoric
    Decomposition via Structured Synthetic Priors."
    <https://arxiv.org/abs/2605.06413>

11. Allahverdyan, A. E. (2020). "Observational nonidentifiability,
    generalized likelihood and free energy."
    <https://arxiv.org/abs/2002.07884>

12. Hu, J., Xie, J., Zhang, Y., Zhou, W. (2026). "The Spurious
    Factor Dilemma: Robust Inference in Heavy-Tailed Elliptical
    Factor Models." <https://arxiv.org/abs/2506.05116>

13. Székely, G. J., Rizzo, M. L., Bakirov, N. K. (2007).
    "Measuring and testing dependence by correlation of
    distances." *Annals of Statistics* **35**(6), 2769–2794.
    [Cited by Deb et al. for the adequacy test.]
