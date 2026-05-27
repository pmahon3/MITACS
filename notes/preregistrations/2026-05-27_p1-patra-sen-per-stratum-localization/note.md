## P1 — Patra–Sen per-stratum localization (test note)

### The question

If the κ_1 predictor's residuals show non-Gaussian mixture structure $\alpha > 0$,
does that mixture fraction vary systematically across natural conditioning strata?
A "yes" localizes where the missing $Z$ is concentrated; a "no" says $Z$ is
orthogonal to all natural axes we can stratify on.

### The math

For each stratum $s$, the Patra–Sen (2016) lower confidence bound on the mixture
fraction is

$$\hat\alpha_L^{(0.95)}(s) \;=\; \inf\bigl\{\gamma \in [0,1] :\; \sqrt{n_s}\,\gamma\,
  d_n\!\bigl(\hat F_s^\gamma,\, \check F_s^\gamma\bigr) \le c_n\bigr\},$$

with $c_n = 0.6792$ the 95th asymptotic quantile of the Cramér–von Mises statistic
(paper line preceding Theorem 6, distribution-free under $F$ continuous).
$\hat F_s^\gamma = (F_n - (1-\gamma)F_b)/\gamma$ is the naive signal CDF, and
$\check F_s^\gamma$ is its PAVA monotone projection clipped to $[0,1]$ (Lemma 1).

The verdict statistic is $\max_{\text{axis}} \mathrm{range}(\hat\alpha_L)$ across
the axis's strata. Verdict bands: $> 0.20 \Rightarrow$ R-A (signal localized);
$(0.10, 0.20] \Rightarrow$ R-B; $< 0.10 \Rightarrow$ R-C.

### Why this entry

P1 is the first node of the `resolution-paths-thread`, the response to S9+S10's
joint applied evidence for the identifiability obstruction in the conditional
PIT defect. Cannot resolve the obstruction; CAN localize it.
