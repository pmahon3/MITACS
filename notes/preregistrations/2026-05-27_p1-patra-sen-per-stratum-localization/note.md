## P1 — Patra–Sen per-stratum localization

### The question

Q2A ([2026-05-26_q2a-richer-family-mixture-or-nonparametric], S9) bounded
the distributional-flexibility lever at residual $\chi^2_{M_3}\!\approx\!954$;
Q2B ([2026-05-26_q2b-non-distributional-decomposition], S10) showed the
naive same-$\sigma$-algebra non-distributional candidates all sat below the
$15\%$ null floor, two of three sign-determined negative. Joint S9+S10 is
the applied face of the identifiability obstruction parked at
`disintegration_diagnostic.md`: structured noise vs. noisy structure with
$Z$ orthogonal to the current $\sigma$-algebra are observationally
indistinguishable from $\kappa_Q$ alone (Bergna et al. 2026 Prop 1;
Heckman–Singer 1984). Resolution requires auxiliary information. P1 —
first node of [2026-05-27_resolution-paths-thread] — takes the
framework-prescribed *stratified candidate-$Z$ localization* path: for
each of five natural conditioning axes (demand-quantile, time-of-day,
season, day-type, hour-of-week), compute per-stratum
$\hat\alpha_L^{(0.95)}(s)$ (Patra–Sen 2016 lower confidence bound on the
non-Gaussian-mixture fraction) on $Q_1$'s $M_1$ PIT residuals
$u_{\text{PIT}}$. Verdict statistic
$\max_{\text{axis}}\,\mathrm{range}(\hat\alpha_L)$; cuts $0.20$ (R-A:
binding axis identifies where $Z$ is concentrated) / $0.10$ (R-C:
orthogonal to all five).

### The verdict, with math

SETTLED **R-A**, at landslide strength. For each stratum $s$,

$$\hat\alpha_L^{(0.95)}(s) \;=\; \inf\bigl\{\gamma \in [0,1] :\;
   \sqrt{n_s}\,\gamma\, d_n\!\bigl(\hat F_s^{\gamma},\,\check F_s^{\gamma}\bigr)
   \le c_n\bigr\},\qquad c_n = 0.6792,$$

against the calibrated PIT null $F_b = \mathrm{Uniform}[0,1]$ (Patra–Sen
Theorem 1's strictly-monotone reparameterization permits the Uniform null
on $u_{\text{PIT}}$). $1000$-resample paired-day bootstrap:
$\max_{\text{axis}}\,\mathrm{range}(\hat\alpha_L) = 0.575\,[0.4475,\,0.6475]$,
binding axis $\mathrm{hour\_of\_week}$ — R-A's cut at $0.20$ cleared by
more than $2\times$ at the lower CI bound. Per-axis ranges (point [CI],
ordered):
$\mathrm{hour\_of\_week}\;0.575\,[0.448,\,0.648]$,
$\mathrm{time\_of\_day}\;0.485\,[0.385,\,0.578]$,
$\mathrm{day\_type}\;0.345\,[0.250,\,0.430]$,
$\mathrm{season}\;0.135\,[0.055,\,0.240]$,
$\mathrm{demand\_quantile}\;0.075\,[0.023,\,0.173]$. Four of five carry
localizing signal. The substantive headline: the missing-$Z$ structure
is *cyclical/social-weekly*, not *seasonal/climatological*. With
hour-of-week binned by $\lfloor(\mathrm{weekday}\cdot 24 + (h-1))/21\rfloor$
(weekday $0=$ Mon, $h\in\{1,\dots,24\}$), the heavy-tail strata are bins
$5/6/7$ ($\hat\alpha_L=0.55/0.61/0.55$) spanning Fri 10:00 through Sun
23:00; the lightest is bin $0$ ($\hat\alpha_L=0.035$), Mon 01:00–21:00.
The $\mathrm{day\_type}$ ranking saturday $0.6475$ > sunday $0.485$ >
weekday $0.3025$ independently corroborates Q2A's saturday-$4\times$-
weekday finding (S9 #2) and eigenmodes-v1's structural weekday/weekend
asymmetry (S2/S3) from a distinct estimator ($\hat\alpha_L$ rather than
$\chi^2$). The proponent's R-A call was directionally right but
$95\%$-CI-falsified on three of four sub-quantities: binding axis
($\mathrm{season}$ forecast, $\mathrm{hour\_of\_week}$ realized);
$\max$-range ($0.30\,[0.18,0.45]$ forecast, $0.575$ realized, overshooting
the CI upper bound by $0.125$); $\hat\alpha_L^{\text{marginal}}$ ($0.60$
forecast, $0.4125$ realized — INSPECTION-ONLY per phase_a). The
devil's-advocate's R-C is decisively falsified (four of five per-axis
CIs entirely above $0.10$), and DA's own *what would change my mind*
condition #2 ("R-A fires for hour-of-week with range $>0.25$") is
exactly the realized signal. Per arbiter.md taxonomy this is the fourth
mechanical-vs-substantive-gap instance in the workflow's history, in a
new orthogonal shape: mechanical matches the proponent's outcome category,
substantive within-bucket forecasts match neither side.

### Broader picture

Routes per `branching_rules.P1.R-A` to Q1A with $Z$ pinned to
$\mathrm{hour\_of\_week}$ — not $\mathrm{season}$, the implicit prior
inherited from [mitacs-clim-gap-nonstationary]. Substantive downward
update on the *temperature-as-localizing-axis* hypothesis: temperature
remains candidate for the marginal pathology, but in the post-cutoff PIT
residuals the localizing signal sits on a weekly/social grid. The
proponent's *what would change my mind* condition #4 also fires
($\mathrm{range}(\mathrm{day\_type}) = 0.345 > 0.15$): $M_1$'s per-
day-type conditioning on the Student-$t$ shape $\hat\nu$ does not
absorb the mixture-fraction asymmetry across day-types. The flag does
not retract Q1/Q2A's settled findings; Q1A's $Z$ on hour-of-week
subsumes day-type at finer resolution. P1 is the first applied response
to the identifiability obstruction parked in
[2026-05-26_q2b-non-distributional-decomposition] and the
[2026-05-27_resolution-paths-thread]'s root; localization succeeds where
decomposition failed because P1 introduces auxiliary information —
stratifications outside the $\sigma$-algebra Q2B was confined to —
exactly as the framework's resolution paths call for.
