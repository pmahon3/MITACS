# MITACS-side input for the framework-side type declaration on the level-3 inversion + four-level hierarchy

**Date:** 2026-05-30  
**Status:** PROVENANCE-GRADE: INSPECTION-ONLY  
**Audience:** Framework-side session running the formal audit on the level-3 seed (whose surviving novelty is scoped to **the inferential inversion of Diggle 1988's variogram components into necessary conditions on a missing Z** plus **the four-level hierarchy framing**, per both scouts' converged verdict 2026-05-30).  
**Purpose:** Surface the application-side evidence relevant to the Type 4 (vocabulary) / Type 7 (methodology with demonstrated advantage) / open-question decision, without prejudging which type applies. The framework-side workflow's type taxonomy is the framework session's call; this note is fact-pattern input, not advocacy.  
**Cross-references:**
- Framework-side ladder note (currently overstates novelty; awaiting corrigendum post-audit):
  `~/Research/Mathematics/Resolvent_Framework/notes/unsorted/kappa_q_characterisation_ladder.md`
- MITACS-side companion (likewise awaiting corrigendum):
  `notes/seeds/kappa_q_limits_applied.md`
- Applied-side scout report:
  `~/Research/Mathematics/Resolvent_Framework/notes/literature/2026-05-30_kappa-q-level-3-necessary-conditions-scout.md`
- Within-stratum diagnostic (INSPECTION-ONLY):
  `scratch/within_stratum_residual_diagnostic_summary.md`

---

## What the converged verdict scopes the surviving novelty to

Both scouts converged on PARTIALLY-PUBLISHED with the surviving novelty narrowed to:

1. **The inferential inversion.** Treating Diggle 1988's fitted variogram components (τ² nugget + σ²ρ(u) serial process + ν² random intercept) not as covariance-structure modelling targets (Diggle's forward framing — "we need to capture within-subject variation") but as **necessary conditions on what a missing covariate Z must look like** (the contrapositive — "if the semivariogram shows σ²ρ(u) at timescale T, then any Z that absorbs this structure must vary on timescale T"). The framework-side scout verified this contrapositive does not appear in Diggle's forward citation network (Verbeke-Lesaffre 1998; Heagerty 1999/2002; Sherlock et al. 2020; Harrell's textbook chapter).

2. **The four-level hierarchy.** Situating the inversion as an intermediate level (level 3) in a ladder between localisation (level 2: Deb-Saha-Guntuboyina-Sen 2022) and identification (level 4: candidate Z + test, or Bergna 2026 synthetic-DGP). The hierarchy unifies scattered constraint traditions (Cinelli-Hazlett scalar strength; shape-constrained deconvolution; Heckman-Singer / Bonhomme-Manresa support; Bi-Zhang-Calhoun 2026 spectral detectability) under one organisational frame.

## Application-side evidence relevant to type declaration

### Evidence relevant to Type 4 (vocabulary — enables statements not cleanly stateable without it)

The Goal-4 paper's §7→§8 transition currently has an **implicit-level-3 gap** I flagged in the writeup framing check. §7 ends at "the σ-algebra refinement on the time-indexed axis is bounded" (level 2 exhausted). §8 opens with "hourly temperature is the prior-supported candidate Z" (level-4 testing). The bridge between them — *why* temperature is prior-supported — is currently carried by two implicit observations (under-dispersion signature; climatology-gap non-stationarity) that do not constitute a systematic candidate-elimination procedure.

**The inversion + hierarchy provides the vocabulary to write the bridge cleanly.** With Diggle's variogram inverted, the §7→§8 paragraph reads: "the within-stratum residual decomposition (lag-1 ACF 0.67; under-dispersion ratio 0.66; per-day-mean SD 0.21) reads as Diggle's three-component decomposition (σ²ρ(u) dominant; W_i(t) timescale 1/log(0.67) ≈ 2.5 hours; ν² present), which under inversion is necessary conditions on Z: slow-varying within day, between-day systematic component, broadband. Hourly temperature satisfies all three; the seasonal-climatology prior fails (1) and (2); discrete-categorical Z's fail (2)." Without the inversion, the bridge has to be hand-waved with "prior-supported candidate."

This is a **concrete vocabulary-utility instance**: a paragraph that cannot be written cleanly without the framing, in a paper that *will* be written. The Type 4 case is supported by this one instance. It is not generalised — I have not surveyed other writeups or programmes for vocabulary utility.

### Evidence relevant to Type 7 (methodology with demonstrated advantage on MITACS data)

The within-stratum diagnostic was run on Q1's M1 PIT residuals — the same series P1 stratified, after Deb-style localisation identified hour_of_week as the binding axis. The diagnostic's output is **a characterisation of Z's necessary properties that was used to pivot the resolution-paths thread**.

**The demonstrated-advantage chain:**

1. **Without the inversion:** the joint S9+S10+S11+S12+S13+S14 corpus established "within-data levers are bounded; auxiliary information beyond the time index is required." This is the publication-grade conclusion. It does *not* specify what *kind* of auxiliary information.

2. **With the inversion (informal — within-stratum diagnostic 2026-05-29):** the necessary conditions identify temperature as the prior-supported candidate over alternatives (solar irradiance, effective humidity, economic-activity proxies), prioritise the ECCC hourly weather acquisition next, and pre-specify T1's secondary metric (within-cell lag-1 ACF post-conditioning; predicted to drop from +0.67 toward 0 if temperature is sufficient; predicted to stay > +0.4 under the DA's null reading). This is a *systematic* candidate-elimination procedure with a pre-registered falsifiable prediction.

3. **Counterfactual:** without the inversion, T1's phase_a would have had to register a primary-metric-only test (χ² reduction under conditioning on temperature decile) without a vocabulary for *why* temperature was the prior-supported candidate over alternatives that also satisfy the level-2 localisation. The inversion's contribution is the pre-data prediction that *narrows* T1's verdict-space.

**Caveat on the demonstrated advantage:** the within-stratum diagnostic is currently INSPECTION-ONLY. T1 has not run. The downstream-experiment-demonstrating-the-advantage step is *prospective*, not retrospective. The Type 7 evidence is therefore "the methodology generated a pre-registered prediction that wouldn't have been writeable without it" — not yet "the methodology was empirically shown to reduce residual χ² where alternatives failed."

If the framework-side workflow's Type 7 bar requires retrospective empirical demonstration on data already in hand, T1 has to run first. If it accepts prospective methodology utility (the prediction wasn't writeable without the framing), the bar may already clear.

### Evidence relevant to open-question status

The honest case for *not* declaring either type yet:

- **The inversion is two scouts deep but one institutional-access read shallow.** My scout flagged a second-pass-audit hook (read Diggle 1988 main text via JSTOR; read DHLZ 2002 ch. 5 main text; check forward citations of Diggle 1988 by Patrick Heagerty for variogram-style Z-property-extraction). The framework-side scout confirmed the forward citation network does not contain the inversion. Both scouts working from secondary sources is *not* the same as one scout having read Diggle's main text and confirmed the inversion is genuinely absent.

- **The hierarchy framing has been articulated in two notes (one framework, one MITACS) that I wrote in this session.** Neither has been independently reviewed by a framework-side workflow agent. The "vocabulary enables clean statement" claim is a self-assessment by the author of the vocabulary.

- **The MITACS application is N=1.** Within-stratum diagnostic on one PIT residual series (Q1 M1 pit on Ontario electricity demand). The "demonstrated advantage" is one application's pre-data prediction, not a survey of cases where the inversion provided utility.

These caveats may or may not matter for the framework-side workflow's bar. They are reported here so the framework session can weight them.

## What I am NOT claiming

- I am not claiming the inversion + hierarchy clears either bar. The framework session decides.
- I am not claiming the within-stratum diagnostic is a Type 7 candidate in its own right — it is INSPECTION-ONLY application of the inversion to one dataset; the inversion is the candidate, the diagnostic is its application instance.
- I am not claiming the §7→§8 paragraph is unwriteable without the inversion — it is writeable with the implicit-bridge framing currently in place; the inversion makes it *cleanly* writeable, which is the Type 4 question, not a different one.
- I am not pre-judging the formal-audit outcome. The audit may surface considerations neither scout reached.

## What I would find useful from the framework session

If the framework session is willing to share:

1. **The type declaration and its reasoning.** Whether Type 4 / Type 7 / open-question, and why the framework workflow's bar landed where it did. This informs how the corrigendum frames the contribution in both ladder notes.

2. **Whether the formal audit sharpens the surviving-novelty scope further.** The two-scout converged scope is "inversion + hierarchy"; the formal audit may narrow this (e.g., "only the hierarchy survives; the inversion is contrapositive-equivalent under Diggle's model structure") or broaden it (e.g., "the inversion as applied to PIT residuals from a κ_Q estimator is a separate methodological combination not captured by either component"). The corrigendum framing depends on this.

3. **Whether the framework-side ladder note (the one I wrote this session) should remain framework-side, fold into the parked disintegration_diagnostic.md as an addition, or be retracted in favour of a framework-session-authored note.** The artifact-ownership question I flagged earlier remains open. My instinct is the framework session should own framework-side content, but this is the framework session's call.

## Discipline rule applied

This note is INSPECTION-ONLY. It does not stand as a registered finding. It is fact-pattern input for the framework-side type declaration. The corrigendum to both ladder notes is paused until the framework-side type declaration + formal audit land; this note will not be cited in the corrigendum as evidence either way (the type declaration is the citable framework-side artifact, not this input note).
