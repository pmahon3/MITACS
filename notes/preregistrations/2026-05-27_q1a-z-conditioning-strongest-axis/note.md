# Q1A: Z-conditioning on the strongest P1 axis — SETTLED R-B1A

## The question

Q1A is the gating node of the [2026-05-27_resolution-paths-thread], the
successor to the distributional-class arc (Q1/Q2A/Q2B). P1 (S11) had just
identified `hour_of_week` as the binding localization axis at landslide
strength ($\hat\alpha_L$ range $0.575$ with 95% CI $[0.4475, 0.6475]$), with
the hot zone bins 5/6/7 spanning Friday 10:00 through Sunday 23:00. The
identifiability-obstruction reading of S9+S10 (Q2A bounded the
distributional-class lever; Q2B bounded the same-$\sigma$-algebra
non-distributional candidates) had pointed at $\sigma$-algebra enrichment
with auxiliary information as the next untested lever. Q1A's
`research_question` therefore asked: does enriching the M3 conditioning
$\sigma$-algebra from $\sigma(\text{day\_type})$ to
$\sigma(\text{day\_type}, Z)$, with $Z$ derived from `hour_of_week` per
P1's binding axis, reduce the post-cutoff marginal PIT $\chi^2$ from
Q2A's settled M3 baseline of $953.84$ to $\leq 228$ (the R-A1A
corroboration cut)?

The metric is a head-to-head between two pre-locked Z constructions
under one verdict cut: $Z_b$, the binary weekend indicator (1 if
`hour_of_week` bin $\geq 5$, else 0), and $Z_c$, the 8-level
categorical (bin index $\in \{0,\dots,7\}$ directly). Both candidates
refit M3 via the production fitter
`mixture_2_gaussian_mle_fit` per `(day_type, Z)` cell with
`MIN_CELL_SIZE = 200` pool-to-parent fallback. The verdict statistic is
$\chi^2_{\text{best}} = \min(\chi^2_{Z_b}, \chi^2_{Z_c})$ against the
inherited thread-skeleton cuts: R-A1A at $228$ (thread RESOLVED),
R-B1A in $(228, 2284]$ (route to Q1B), R-C1A above $2284$ (route to P2).

## The verdict, with math

**Mechanical: R-B1A fires unambiguously.** The realized values:

$$
\chi^2_{Z_b} = 986.02,\quad \chi^2_{Z_c} = 952.52,\quad
\chi^2_{\text{best}} = 952.52\ \text{with 95\% CI }[625.34,\ 1342.33].
$$

Both the point estimate and the entire bootstrap CI sit inside the
R-B1A band $(228, 2284]$: the lower CI bound is $\sim 2.7\times$ clear
of R-A1A, the upper CI bound is $\sim 1.7\times$ clear of R-C1A. The
verdict is comfortably inside the ambiguous band — neither the
corroboration cut (R-A1A) nor the falsification cut (R-C1A) is touched.

**Head-to-head is a NEAR-TIE.** The paired-day-bootstrap difference is:

$$
\chi^2_{Z_b} - \chi^2_{Z_c} = 33.50,\quad
\text{95\% CI }[-93.16,\ +142.64],\quad
\text{paired SE } \approx 56.84.
$$

Since $|\Delta\chi^2| = 33.5 < 1\sigma_{\text{paired}} = 56.84$, the
phase_a `NEAR-TIE` rule fires: the head-to-head sub-finding records
"no statistically-discernible winner." $Z_c$ wins at the point estimate;
the CI straddles zero. Neither side's signed-margin forecast (proponent:
$Z_c$ wins by $+220$; DA: $Z_b$ wins by $-150$) is empirically
discriminable.

**Z conditioning delivers $0.18\%$ of the gap.** The unconditioned
baseline reproduces Q2A's settled M3 chi^2 to four decimal places:

$$
\chi^2_{\text{no-Z}} = 953.8433\overline{3}\ \text{vs.}\ \chi^2_{\text{Q2A-settled}} = 953.84,\quad
|\Delta| = 3.3\times 10^{-3}.
$$

The Z-conditioned reduction is then:

$$
\Delta\chi^2_{\text{Z lever}} = 953.8433 - 952.5167 = 1.3267,
$$

and the share of the gap-to-close (R-A1A) absorbed by the lever is:

$$
\frac{\Delta\chi^2_{\text{Z lever}}}{\chi^2_{\text{no-Z}} - 228}
= \frac{1.3267}{725.8433} = 0.001827 \approx 0.18\%.
$$

The central premise of the resolution-paths thread — that
$\sigma$-algebra enrichment via `hour_of_week` would absorb the residual
non-Gaussian mass — moved the verdict statistic by under two parts in a
thousand of the gap. The remaining $99.82\%$ of the gap lies outside
the $(\text{day\_type}, Z_{\text{hour\_of\_week}})$ $\sigma$-algebra.

**Per-cell decomposition corroborates P1 but reveals a family-capacity
ceiling.** Under $Z_c$, the dominant single-cell contribution is
$\chi^2_{\text{sat, bin 6}} = 1634.49$ with $n = 1296$ rows — exactly P1's
hottest stratum, where $\hat\alpha_L = 0.61$. This single hot-zone cell
exceeds the *pooled* marginal $\chi^2$ ($952.5$) by $\sim 70\%$. The
fitted mixture-2-Gaussian in this cell has tail weight $w_2 = 0.20$,
scales $s_1 = 0.0963$, $s_2 = 0.215$: the family's flexibility is
deployed but does not absorb the within-cell heterogeneity on the
$\chi^2$ metric. Cold weekday strata (bins 0–4) all clear R-A1A
*within cell* ($\chi^2 = 37$–$78$); the chi^2 mass is structurally
concentrated where P1 said it was, but the same-family lever cannot
close it.

**Substantive verdict.** Mechanical R-B1A matches the
devil's-advocate's outcome category (proponent's R-A1A is falsified by
$\sim 700\ \chi^2$ units). DA's Z_c point forecast was $950$ against
realized $952.52$ — a $0.3\%$ residual, the single best
pre-registered point forecast in the workflow's history. Proponent's
$\hat\alpha_L$-localization-implies-$\chi^2$-closure premise was the
pre-registered bet; it is empirically falsified at the $95\%$ CI
level. The discriminating asymmetry the calibration record names is
that the **localization metric** ($\hat\alpha_L$, P1's setting) and the
**closure metric** ($\chi^2$, Q1A's setting) are different functionals
of the residual law: localization is necessary for $Z$-detection but is
not sufficient for $\chi^2$-closure under same-family mixture
refinement.

## Broader picture

Q1A is the **first within-data lever** tested in the
[2026-05-27_resolution-paths-thread]. Joint with
[2026-05-26_q2a-richer-family-mixture-or-nonparametric] (S9, distributional-class
bounded) and [2026-05-26_q2b-non-distributional-decomposition] (S10,
same-$Q$ non-distributional candidates bounded with two sign-determined
negative), Q1A's $0.18\%$-of-gap reading is **applied-side
corroboration of the identifiability obstruction** parked at
`~/Research/Mathematics/Resolvent_Framework/notes/unsorted/disintegration_diagnostic.md`.
Three within-data levers — distributional family flexibility (S9), same-Q
non-distributional mechanisms (S10), and $\sigma$-algebra enrichment
with the strongest time-index-derived $Z$ (S12 = Q1A) — are now
empirically bounded. The framework-prescribed paths require auxiliary
information beyond the time index (Bergna et al. 2026 Prop 1; Heckman &
Singer 1984; Allahverdyan 2020): exogenous $Z$ candidates (temperature
per `mitacs-clim-gap-nonstationary`), stratified Patra-Sen with
genuinely-orthogonal axes, or synthetic Gaussian-DGP calibration (P2).

Per `thread.yaml` `branching_rules.Q1A.R-B1A`, the thread now routes to
**Q1B** (second-strongest axis from P1: `time_of_day` at range $0.485$,
with overnight-vs-rest as the binding contrast). The carryforward
baseline for Q1B's cumulative $\chi^2$ is $Z_c$ (the better
point-estimate $Z$, even though the head-to-head is a near-tie), per
the registered `skeleton_thresholds_depend_on` rule. With $Z_1$
(`hour_of_week`) delivering $0.18\%$ of the gap and $\text{P1 range}_{Z_2}
= 0.485 < \text{P1 range}_{Z_1} = 0.575$, the prior on Q1B closing the
gap via the same lever is weaker still. The thread coordinator should
pre-flag the EXHAUSTED-UNRESOLVED trajectory: if Q1B fires R-B1B or
R-C1B, the natural follow-up is a thread amendment to reopen
[2026-05-27_resolution-paths-thread] Q1C (temperature fallback;
currently closed-sibling-not-fired) or P2 calibration.

This is the **fifth instance of the fourth-named-failure-mode pattern**
in the applied-audit workflow (after Q1 R-D-vs-R-C, eigenmodes-v1
O2-vs-O3, missing-content P1 tautology, Q2B mech_2/mech_3 negative
shares, resolution-paths P1 hour_of_week-not-season). The new shape
variant: **mechanical and within-bucket point forecasts both match
side A (DA); the head-to-head direction is point-correct for side B
(proponent) but the realized direction is statistically
indistinguishable from zero at the paired-SE level.** The
mechanical-verdict-plus-substantive-content shape continues to absorb
new variants without engineering fixes; per CLAUDE.md the pattern is
the workflow's correct response rather than a defect to be removed.

Q1A' ([2026-05-27_q1a-prime-sdr-ica-sibling], parallel non-state-machine
sibling per amendment A1) remains active in parallel; its
data-driven Z-recovery (SIR + FastICA) test on whether the methods
recover an `hour_of_week`-aligned direction without being pre-specified
is independent of Q1A's $\chi^2$-closure outcome and fires regardless.
