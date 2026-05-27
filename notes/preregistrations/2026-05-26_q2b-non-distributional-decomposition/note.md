## Q2B — non-distributional decomposition

### The question

Q2A ([2026-05-26_q2a-richer-family-mixture-or-nonparametric]) had just settled
R-B2: distributional flexibility above a two-component centred Gaussian mixture
is empirically bounded, and the binding family $M_3$ still leaves
$\chi^2_{M_3} = 953.84$ unresolved on the post-cutoff marginal PIT, sitting in
the *non-distributional* axis by elimination. Q2B asked: of the gap between
$\chi^2_{M_3}$ and the gate-validated R-A2 cut $\chi^2_{\text{R-A2}} = 228.44$,
how does it decompose across three pre-registered mechanism ablations —
iterated mean trajectory bias ($\text{mech}_1$), one-sided iterated variance
cap ($\text{mech}_2$), and day-anchor seam aggregation-domain restriction
($\text{mech}_3$, a REGISTERED NULL because production already excludes
seam-crossing iteration). The verdict architecture mirrored Q2A: R-A3 (one
mechanism dominates, $\max_m \text{share}[m] > 0.50$ and $> 2\times$ the
second), R-B3 (mixed; mandates thread amendment), R-C3 ($\max_m \text{share}[m]
< 0.15$, all below the noise floor set generously above $\text{mech}_3$'s
registered-null expectation).

### The verdict

For each mechanism $m$, the explained share is

$$\text{share}[m] \;=\; \frac{\chi^2_{\text{baseline}} - \chi^2_{\text{after\_fix}}[m]}{725.40},$$

denominator frozen at $725.40 = 953.84 - 228.44$ across the $1000$-resample
paired-day bootstrap. Realized point estimates with $95\%$ percentile CIs:
$\text{share}_{\text{mean\_bias}} = +0.109\;[-0.39,\,+0.74]$,
$\text{share}_{\text{variance\_div}} = -0.114\;[-0.22,\,-0.01]$,
$\text{share}_{\text{seam}} = -0.139\;[-0.17,\,-0.09]$. The R-C3 criterion
$\max_m \text{share}[m] < 0.15$ fires; $\text{mech}_1$ at $+0.109$ is the only
positive share and still below the floor. The substantive content is sharper
than the mechanical reading: two of three shares are sign-determined negative
at $95\%$ — the corresponding ablations actively *degraded* $\chi^2$. The
candidate set is not merely inadequate; $\text{mech}_2$ and $\text{mech}_3$
are mis-specified at the construction level. The mechanism lives in the
secondary arrays: $\sigma^2_{\text{iter}}(h, \text{day\_type}) <
\sigma^2_{\text{emp}}(h, \text{day\_type})$ at virtually every cell, so the
one-sided cap (which only scales down when $\sigma^2_{\text{iter}} >
\sigma^2_{\text{emp}}$) fires against the dominant direction and shrinks an
already under-dispersed distribution further. The self-consistency gate
$|\chi^2_{Q2B} - \chi^2_{Q2A}| = 8.12 \ll 50.0$ confirmed no pipeline drift.

### Broader picture

S9 ([2026-05-26_q2a-richer-family-mixture-or-nonparametric]) bounded the
kernel-refinement lever; Q2B (S10) bounds the σ-algebra-preserving naive
manipulations on the same $Q$. Jointly, S9 + S10 are the *applied* face of
the identifiability obstruction parked the same day in the sibling theory
programme (`disintegration_diagnostic.md`): structured noise versus noisy
structure with $Z$ orthogonal to the current $\sigma$-algebra are
observationally indistinguishable from $\kappa_Q$ alone (Bergna et al. 2026
Prop 1; Heckman–Singer 1984; Allahverdyan 2020). Resolution requires
auxiliary information, not richer post-hoc corrections. The
`distributional-class-thread` is therefore declared EXHAUSTED rather than
amended — its scope_limits ("tests the PREDICTIVE-DISTRIBUTION CLASS axis
only") are exactly what Q2B's R-C3 confirms the residual lies *outside*.
The descendant [2026-05-27_resolution-paths-thread] inherits this verdict
and structures the next round around the framework's three prescribed
paths: candidate $Z$ testing (temperature, secular drift, hour-of-week),
stratified NPMLEmix per-stratum localization of $\hat\alpha_L^{(0.95)}$
(Deb, Saha, Guntuboyina, Sen 2022), and synthetic Gaussian-DGP calibration
to fix the floor. Q2B is the second arbiter in succession (after Q2A)
without a mechanical/substantive gap, and the first in which the
substantive content runs *stronger* than the mechanical verdict.
