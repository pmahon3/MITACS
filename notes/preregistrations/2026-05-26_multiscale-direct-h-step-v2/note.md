## v2 — multiscale direct h-step (relaxed factorization picture corroborated)

### The question

The multiscale seed (`notes/seeds/multiscale_factor_coherence.md`) asks
whether the directly-fit $h$-step factor $(C_h, \Sigma_h)$ coincides with the
iterated $1$-step factor $\bigl(C_1^h,\, \sum_{j=0}^{h-1} C_1^j \Sigma_1
(C_1^j)^{\!\top}\bigr)$ on Ontario pre-cutoff demand, or whether the two
diverge as the seed's relaxed-factorization picture (C) would predict. v1
([2026-05-26_multiscale-direct-h-step]) registered the relative-Frobenius
drift metric $M_1 = \|C_h - C_1^h\|_F / \|C_h\|_F$ and read
DEVILS-ADVOCATE-CONFIRMED — but the synthetic VAR(1) gate gave $M_1 = 1.83$
on a system where the strict semigroup holds *by construction* (as $t$ and
$t+h$ decorrelate, $C_h \to 0$ and the denominator collapses), so $M_1$ is
degenerate at large $h$. v2 replaces the metric with a distributional
divergence on the scalar innovation coordinate — well-posed where $\Sigma_j$
is rank-1 by construction ([[mitacs-rank1-structural]]) — and promotes the
synthetic gate to a precondition. Cuts on
$\mathrm{D\_RATIO} = \mathrm{kl}_{\text{Ontario}}(\text{weekday}, h{=}12) /
\mathrm{kl}_{\text{synthetic}}(h{=}12)$: falsification at
$\mathrm{D\_RATIO} \le 2.0$, corroboration at $\mathrm{D\_RATIO} > 5.0$,
ambiguous in $(2.0,\,5.0]$.

### The verdict

The headline statistic is the empirical-state-averaged $1$D Gaussian KL on
the scalar innovation coordinate,

$$\mathrm{kl}_{\text{Ontario}}(h, dt) \;=\; \mathbb{E}_{x \in \mathcal{L}_{h, dt}}\!\left[\,\mathrm{KL}\!\bigl(\mathcal{N}(c_{\text{direct}}(x), s_{\text{direct}}^2)\,\big\|\,\mathcal{N}(c_{\text{iter}}(x), s_{\text{iter}}^2)\bigr)\right],$$

with $c_{\text{direct}} = (X C_h)_0$, $c_{\text{iter}} = (X C_1^h)_0$,
$s_{\text{direct}}^2 = \Sigma_h[0,0]$, $s_{\text{iter}}^2 = \bigl(\sum_j C_1^j
\Sigma_1 (C_1^j)^{\!\top}\bigr)[0,0]$. The synthetic precondition cleared:
$\mathrm{kl}_{\text{synthetic}}(h{=}12) = 0.0012$ nats, two orders of
magnitude below the $0.20$-nat ceiling and stable across seven seeds
($0.0011$–$0.0016$). Ontario:
$\mathrm{D\_RATIO} = 28.96\,[26.82,\,31.43]$ over $1000$ paired-day
bootstraps — the $95\%$ CI lies entirely above the corroboration bar $5.0$,
falsification cleared by $>12$ SE. The secondary $\mathrm{kl}_{\text{Ontario}}$
table grows monotonically in $h$ within each day-type and is markedly larger
on weekend day-types (Sunday $h{=}24$ at $0.330$ nats is the largest cell),
so picture (A) is inadequate across the *whole* grid, not just the headline
cell. PROPONENT-CONFIRMED, magnitude underestimated: the proponent's $0.95$
interval $[6,\,25]$ failed to bracket the realization. A material qualifier
from downstream Q1 (S8c, [2026-05-26_q1-student-t-vs-gaussian]): under the
Student-t kernel the analogous ratio shrinks to
$\mathrm{D\_RATIO}_t = 2.93\,[2.62,\,3.24]$ — a $\sim 10\times$ collapse —
so $\sim 90\%$ of v2's headline magnitude is kernel-family artifact, not
structural multiscale signal. The SETTLED verdict is *retained*
($\mathrm{D\_RATIO}_t$ still clears falsification) but carries an asterisk
on magnitude.

### Broader picture

This is the first SETTLED finding in the applied-audit registry — the
canonical demonstration that the workflow's gates (`code-path-auditor`,
synthetic precondition, hash-chained pre-registration) compose without
softening the verdict. The retraction catalogue in
[[mitacs-session-lessons-corpus]] names "diagnostic reimplements production"
as the dominant failure mode; v2 caught a sibling — degenerate-metric
reimplementation of a structural question — on a single replay of the same
underlying fits, inverting v1's DEVILS-ADVOCATE-CONFIRMED to
PROPONENT-CONFIRMED not because the data changed but because the metric
finally posed the question picture (C) predicts about. Downstream the seed's
$\S 6.3$–$\S 6.5$ branches fired in turn: [2026-05-26_multiscale-defect-eigenmodes]
(S2/S3) projected the per-$(dt, h)$ KL signal onto eigenmodes of $C_1$ and
SETTLED MIXED — defect diffuse across two modes, not single, and the
weekday/weekend asymmetry structural. Then
[2026-05-26_q1-student-t-vs-gaussian] (S8) cracked the kernel-family
assumption: heavy-tailedness lives in the $\kappa_1$ innovation, and once
modelled the marginal PIT $\chi^2$ shrinks $16\times$ and v2's
$\mathrm{D\_RATIO}$ shrinks $10\times$ — fixing both the strength of v2's
substantive finding (survives the kernel correction) and the limit of its
magnitude claim. The seed's $\S 6.4$ branch — joint $(L, Q)$ inference under
strict-semigroup constraint with the per-$h$ failure pattern as informative
signal — remains open, now to be conducted under $\kappa_t$ rather than
$\kappa_Q$.
