# The κ_Q characterisation framing — applied evidence from MITACS

**Date:** 2026-05-30 (corrigendum 2026-05-30, post framework-side PARK verdict)  
**Status:** PROVENANCE-GRADE: INSPECTION-ONLY  
**Framework-side seed status: PARKED** (filed at
`~/Research/Mathematics/Resolvent_Framework/notes/covered_leads/kappa_q_characterisation_ladder.md`;
formal-audit verdict at
`~/Research/Mathematics/Resolvent_Framework/notes/covered_leads/kappa_q_characterisation_ladder_audit.md`).  
**Calibration anchor:**
`~/Research/Mathematics/Resolvent_Framework/notes/covered_leads/residual_structure_inference.md`
(parked 2026-05-27 on the same class of question; the κ_Q ladder seed's
Type 4 verdict inherits this calibration).  
**Operational status (this note):** The framework-side seed is parked;
the MITACS programme's operational use of the framing is **unaffected**
by the seed's PARK status. Diggle 1988 / DHLZ 2002 ch. 5 are canonical
published methodology; T1's planned secondary metric (within-cell
lag-1 ACF) cites them as the methodological source, not the seed.  
**Triggering question:** User reframe 2026-05-30 — "what is the limit
of definiteness to which we can characterise missing/exogenous
information within the bounds of the Resolvent Framework?"

---

## What the corrigendum changes

This note was originally drafted 2026-05-30 (earlier the same day) as
companion to a framework-side ladder note that claimed level-3
necessary-condition extraction as candidate-novel content. Two literature
scouts (applied-side + framework-side) converged on PARTIALLY PUBLISHED
with the surviving novelty narrowed to "inferential inversion + four-level
hierarchy + unification." The framework-side formal audit then evaluated
the surviving novelty against all seven contribution types and concluded:

- **Type 1 (theorem) FAIL** — content attributed to existing results
- **Type 2 (new proof) N/A** — no proof claimed
- **Type 3 (unification) FAIL** — no method transfer demonstrated
- **Type 4 (vocabulary) CONDITIONAL FAIL** — three-statements test
  fails; circumlocution ≠ non-stateability; `residual_structure_inference`
  calibration anchor applies
- **Type 5 (impossibility) FAIL** — impossibilities attributed to
  Bergna, Heckman-Singer, Allahverdyan, Andrews-Mallows
- **Type 6 (exposition) FAIL** — inferential inversion is standard
  stance (Cinelli-Hazlett; Chernozhukov et al. 2024); practitioner
  tutorials exist; hierarchy is Pearl/Manski-style tiering
- **Type 7 (methodology) CONDITIONAL** — T1 not executed; N=1;
  genuine structure but no demonstrated advantage yet

**Verdict:** PARK. Single surviving revival trigger is **T1 execution**:
either-sign outcome on the pre-registered within-cell lag-1 ACF drop
clears Type 7.

This note's substantive content (the level-by-level mapping of the
MITACS corpus) is unaffected; the *framing* changes from "level 3 is
the candidate framework contribution" to "level 3 = canonical Diggle
machinery applied in the inferential-inversion direction, a known
sensitivity-analysis stance." The MITACS programme uses the framing
operationally; it does not claim novelty for the framing.

## Why this matters for the MITACS application

The framework-side PARK does **not** affect any of the following:

1. **The level-by-level mapping of S9-S14 to the four levels of the
   ladder is correct as analysis** — the corpus has exhausted within-data
   levers (levels 2 and 3) and routes to auxiliary information (level 4)
   per Heckman-Singer / Manski / Pearl tiering applied to κ_Q's
   non-identifiability obstruction.
2. **T1's planned secondary metric (within-cell lag-1 ACF) is well-grounded
   methodologically** — it is a discrete-time three-summary-statistic
   instance of Diggle's residual variogram diagnostic (Diggle 1988
   *Biometrics* 44: 959-971; DHLZ 2002 ch. 5). The inferential-inversion
   reading of the variogram components ("the σ²ρ(u) component constrains
   what timescale a missing covariate must have") is a Cinelli-Hazlett /
   Chernozhukov 2024-style sensitivity-analysis stance applied to the
   variogram.
3. **The Goal-4 §7→§8 paragraph** (deferred earlier this session) is now
   writeable citing Diggle 1988 / DHLZ 2002 for machinery and
   Cinelli-Hazlett 2022 / Chernozhukov et al. 2024 for the
   inferential-inversion stance, with no novelty claim. PARK on the
   framework seed does not constrain the paper.

The framework-side seed and the MITACS operational use are decoupled by
the audit's finding: the operational tools are all published; the
operational stance is also published. There was never a novelty claim
on the MITACS side, and the framework session's PARK on the framework
side leaves the application side intact.

## The four-level mapping (unchanged from the original; framing only)

The framework decomposes "characterise Z from κ_Q-observable structure"
into four levels. Levels 1, 2, 3 are κ_Q-attainable; level 4 requires
auxiliary information. This corpus has executed levels 2 and 3 to
their κ_Q-side limits; level 1 is theorem-bound; level 4 routes to T1
(temperature arm) and beyond. **Each of the four levels is a known
position in the sensitivity-analysis / latent-variable literature;
the level-by-level read is for orientation, not novelty claim.**

### Level 1 — Existence regime (a vs b): NOT discriminable from κ_Q alone

**Framework basis.** Identifiability obstruction theorem. Bergna et
al. 2026 Prop 1, Heckman-Singer 1984, Allahverdyan 2020, Andrews-
Mallows 1974. No κ_Q-only diagnostic can distinguish "intrinsic
heavy-tailed innovation" (regime a) from "missing-Z mixture artifact"
(regime b).

**Applied evidence (S9 + S10).**

- **S9 (Q2A, richer-family).** Three richer-than-Student-t families
  (mixture-2-Gaussian, mixture-3-Gaussian, KDE) all chi²-
  indistinguishable at ~954 χ². Distributional flexibility is bounded;
  no kernel-family lift closes the gap at fixed Q. R-B2 mechanical =
  substantive.
- **S10 (Q2B, non-distributional).** Three same-Q mechanism
  ablations (mean bias, variance cap, seam exclusion). All shares
  below 15% null floor; two sign-determined negative. Naive same-Q
  manipulations inadequate. R-C3 mechanical = substantive.

Joint reading: applied evidence that the obstruction binds in this
case. Consistent with both regime (a) and regime (b); the marginal
evidence cannot break the tie (per the theorem).

### Level 2 — Localisation of Z's action: ATTAINABLE; executed at S11

**Framework basis.** Monotonicity (R*(Q') ≤ R*(Q) for Q' ⊃ Q) plus the
Deb-Saha-Guntuboyina-Sen (JASA 2022) NPMLEmix or its discrete
specialisation (stratified Patra-Sen / per-stratum α̂_L). Canonical
published methodology.

**Applied evidence (S11, P1).** Per-stratum α̂_L across five candidate
axes on Q1's M1 PIT residuals:

| Axis | Range | 95% CI | Binding? |
|------|-------|--------|----------|
| **hour_of_week** | **0.575** | **[0.448, 0.648]** | **YES** |
| time_of_day | 0.485 | [0.385, 0.578] | |
| day_type | 0.345 | [0.250, 0.430] | |
| season | 0.135 | [0.055, 0.240] | |
| demand_quantile | 0.075 | [0.023, 0.173] | |

Verdict R-A at landslide strength. Hot zone: bins 5/6/7 (Fri 10:00 –
Sun 23:00, α̂_L 0.55-0.61). Cold zone: bin 0 (Mon 01:00-21:00,
α̂_L 0.035).

**Subsequent corpus tested whether level-2 σ-algebra refinement closes
the gap at level 4:**

- **S12 (Q1A).** Conditioning on hour_of_week closes 0.18% of the
  residual gap. R-B1A.
- **S13 (Q1B).** Joint (Z_c, Z2_4level) closes 40% at point with CI
  spanning 60×. R-B1B substantive favourable.
- **S14 (Q1B').** SMC particle doubling widens CI 6.3%. Eval-window
  noise dominates; within-data tightening empirically impossible.
  INSPECTION-ONLY.

**Joint S12+S13+S14 reading.** The level-2 localisation (S11) is
correct — Z's action lives on hour_of_week — but conditioning on that
σ-algebra does not close the residual gap. The remaining ~660 χ²
lives in conditional structure that hour_of_week + time_of_day cannot
resolve. Z is a real-valued continuous covariate whose action
*projects onto* the hour_of_week stratification.

### Level 3 — Necessary conditions on Z's shape: ATTAINABLE; instance via Diggle variogram applied to PIT residuals

**Framework basis.** Diggle 1988 (*Biometrics* 44: 959-971) and Diggle-
Heagerty-Liang-Zeger 2002 ch. 5 establish the three-component
covariance decomposition (τ² nugget + σ²ρ(u) serial process + ν²
random intercept) via the empirical semi-variogram. The inferential-
inversion reading (treating fitted components as necessary conditions
on a missing covariate's properties) is standard sensitivity-analysis
stance (Cinelli & Hazlett 2022 JRSSB; Chernozhukov et al. 2024). All
machinery and stance are published; combining them is a methods choice,
not a novelty claim.

**Applied evidence (within-stratum residual diagnostic, 2026-05-29).**
INSPECTION-ONLY scratch (`scratch/within_stratum_residual_diagnostic_summary.md`).
Discrete-time three-summary-statistic instance of Diggle's variogram,
run on the same Q1 pit_M1 series P1 stratified, across 32
(day_type × hour_of_week_bin) cells.

| Statistic | Value | Diggle component | Inversion reading: necessary condition on Z |
|-----------|-------|------------------|---------------------------------------------|
| Mean lag-1 ACF | **+0.669** across all 32 cells | σ²ρ(1) dominant; ρ(u) decays slowly | Z is slow-varying (timescale 1/log(0.67) ≈ 2.5 hr); fast-varying Z's ruled out |
| Mean lag-5 ACF | Much weaker | σ²ρ(5) small | Z is not narrowband-periodic at 5-step scale |
| Variance ratio | 0.66 (under-dispersed within stratum) | within-stratum < full Σ_Q | Z's action increases variance; residual variance lives in between-stratum structure |
| SD per-day mean | ~0.21 | ν² random intercept present | Z has between-day systematic effect (location-shifting) |
| Bartlett 95% clearance | 32/32 cells | σ²ρ(1) structural | Decomposition is genuine, not chance |

**Necessary-but-not-sufficient characterisation.** Any Z that could
account for the remaining ~660 χ² must satisfy (1) slow-varying
continuous, (2) within-day continuous structure, (3) between-day
systematic component, (4) broadband or low-frequency, (5) weak
week-on-week tie. Temperature satisfies (1)-(5); solar irradiance
does; effective humidity does. Fast-varying Z's, discrete-categorical
Z's, and pure-seasonal-climatology Z's are ruled out.

**Status discipline.** The diagnostic is INSPECTION-ONLY. The necessary
conditions become registered-finding-eligible via T1's pre-registered
secondary metric (within-cell lag-1 ACF post-conditioning on
temperature; predicted to drop from +0.67 toward 0 if temperature is
sufficient; predicted to stay > +0.4 under DA's null reading). See
CLAUDE.md T1 forward-pointer.

### Level 4 — Identity of Z: NOT attainable from κ_Q alone

**Framework basis.** Requires auxiliary info type 1 (candidate Z +
test), type 2 (richer model class — exhausted at S9), or type 3
(structural noise assumption — out of scope per Goal-4 framing).

**Planned applied work (T1, T2, T3).** Per amendment A4 to
resolution-paths-thread (2026-05-29).

- **T1.** M3 with σ = (day_type, Z_c, temperature_decile or
  equivalent); cumulative baseline Q1A's settled 952.5167; primary
  metric χ²; **secondary metric within-cell lag-1 ACF** (the
  Diggle-σ²ρ(1) component, inversion-read as necessary condition on
  Z's timescale).
- **T2.** Q1B-analogue joint refinement; contingent T1 R-B1T.
- **T3.** Q1B'-analogue methodological tightening; optional non-gating
  sibling.

**Pre-requisite (blocking).** ECCC hourly weather data acquisition.

**Framework-side stake (new, post-PARK).** T1 execution is the **single
surviving revival trigger** for the framework-side seed at
`covered_leads/kappa_q_characterisation_ladder.md`. Either outcome
(R-A1T confirming the ACF drop or R-D1T falsifying it) demonstrates
the framing's operational utility per the audit verdict, since the
pre-registered prediction existed *because of* the framing. This adds
no MITACS-side obligation — T1 was already on the active programme
path — but adds visibility from the framework-side workflow.

## Summary table

| Level | Status | MITACS evidence |
|---|---|---|
| 1 | Theorem-bound; obstruction binds (joint S9+S10) | Q2A χ² ~954 family-invariant; Q2B mechanism shares below null floor |
| 2 | Achieved (S11) | hour_of_week binding at α̂_L 0.575 |
| 3 | Diggle variogram applied, inferentially inverted; INSPECTION-ONLY | Within-stratum lag-1 ACF +0.669 across 32 cells |
| 4 | Pending T1 | ECCC acquisition blocking; T1 secondary metric pre-registered as the level-3-becomes-citable mechanism |

**The publication-grade conclusion for the Goal-4 paper** remains:
within-data work on this corpus is bounded; auxiliary information
beyond the time index is required. Levels 2 and 3 have been executed
to their κ_Q-side limits. This conclusion does not depend on any
novelty claim — Heckman-Singer / Manski / Pearl tiering + Diggle
machinery + Cinelli-Hazlett / Chernozhukov inversion stance suffice
as published precedent.

## Revival trigger and downstream tracking

The framework-side seed PARK leaves **one surviving revival trigger**:

- **T1 executes.** Confirms or falsifies the pre-registered ACF-drop
  prediction. Either outcome clears Type 7 (methodology) at the
  framework side: the operational utility is demonstrated by the
  *prediction having existed because of the framing*, not by the
  prediction's sign. T1's primary metric (χ² reduction) is unrelated
  to the framework-side trigger.

Two other revival triggers were proposed in round 1 of the audit and
dropped in round 2 / final:

- **Diggle 1988 JSTOR read** — round 2 found Cinelli-Hazlett /
  Chernozhukov et al. 2024 already articulate the inversion stance;
  Diggle-specific institutional read no longer load-bearing.
- **Second application domain** — round 2 did not retain this as a
  Type-7-clearing path independent of T1. If T1 itself does not
  execute, a second-domain replication of the diagnostic (e.g., the
  IEEE DataPort Post-COVID dataset already used as inspection-only
  probe) might still clear Type 7 — but this is not registered as a
  current revival trigger and would require framework-side adjudication.

## Cross-references and lineage

- **Framework-side seed (PARKED 2026-05-30):**
  `~/Research/Mathematics/Resolvent_Framework/notes/covered_leads/kappa_q_characterisation_ladder.md`
- **Framework-side audit verdict:**
  `~/Research/Mathematics/Resolvent_Framework/notes/covered_leads/kappa_q_characterisation_ladder_audit.md`
- **Calibration anchor (parked 2026-05-27 on the same class of
  question):**
  `~/Research/Mathematics/Resolvent_Framework/notes/covered_leads/residual_structure_inference.md`
- **Identifiability obstruction (level 1):**
  `~/Research/Mathematics/Resolvent_Framework/notes/unsorted/disintegration_diagnostic.md`
- **Applied-side scout report:**
  `~/Research/Mathematics/Resolvent_Framework/notes/literature/2026-05-30_kappa-q-level-3-necessary-conditions-scout.md`
- **Within-stratum residual diagnostic (INSPECTION-ONLY):**
  `scratch/within_stratum_residual_diagnostic_summary.md`
- **MITACS-side input note for the type declaration:**
  `notes/lab/2026-05-30_mitacs-side-input-for-type-declaration.md`
- **Goal-4 paper §6-§8 (Act 1/2/3):** `writeup/tex/draft_body.tex`
- **Companion memo (predates this corrigendum):**
  `writeup/tex/missing_content_memo.tex` §9-§12

## What this note does NOT do

- Does not claim novelty for any level — all four are positions in
  published sensitivity-analysis / latent-variable / longitudinal-data
  literature.
- Does not register as a `freeze.py`-style locked specification — this
  is an INSPECTION-ONLY scratch synthesis.
- Does not modify the parked `disintegration_diagnostic.md` or any
  other framework-side artifact.
- Does not commit T1's phase_a — that requires the within-stratum
  diagnostic sanity-check (Gaussian-synthetic null) and ECCC data
  acquisition before the lag-1 ACF prediction can be pre-registered.
- Does not pre-judge T1's outcome — the inversion-read necessary
  conditions are consistent with temperature *and* with multiple other
  Z's; T1 pre-registers a specific test of temperature against the
  necessary conditions.
- Does not assert that the framework-side seed should revive — T1
  execution is the surviving trigger, but whether T1's outcome
  *qualifies* as revival evidence is the framework session's call, not
  this note's.
