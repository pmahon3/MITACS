## P1 — PIT Calendar Fingerprint

### The question

Root node $P1$ of the [2026-05-26_missing-content-thread], downstream of
v2's settled defect ([2026-05-26_multiscale-direct-h-step-v2]): does the
marginal *PIT* residual of the iterated $\kappa_1$ predictor carry a
calendar-fingerprint signal pointing to *where* its miscalibration lives?
For each post-cutoff $(d,h)$ we form
$$u(d,h) \;=\; \Phi\!\left(\frac{z_{\text{actual}}(d,h) - \mu_{\text{iter}}(d,h)}{\sqrt{\mathrm{Var}_{\text{iter}}(d,h)}}\right),$$
stratify into 36 cells $c = (\text{season}, \mathrm{HE\,bin}, \text{day\_type})$,
and per candidate $v$ compute
$\mathrm{score}(v) = \sum_c |T_c|\,|\rho_c(v)|$, with $T_c$ the cell-mean
$\hat F_n$ deviation and $\rho_c$ the within-cell Spearman against
$u - \tfrac{1}{2}$. Verdict architecture: $O\text{-}A$ (seasonal dominates,
$\geq 2\times$ margin), $O\text{-}B$ (weekly dominates), $O\text{-}C$
(both clear null, ratio in $(0.5, 2.0)$ — proponent's forecast), $O\text{-}D$
(no candidate clears null — DA's forecast). Anything else is AMBIGUOUS.

### The verdict, with math

**SETTLED-AMBIGUOUS.** Mechanical AMBIGUOUS fires because the top
seasonal/top weekly ratio is $0.27$, outside $O\text{-}C$'s required
$(0.5, 2.0)$ band, and no class clears the $2\times$ dominance bar.
But — and this is the load-bearing content — the AMBIGUOUS verdict
sits on top of three durable empirical findings the registered taxonomy
could not absorb. **(a) Marginal PIT pathology.** A simple 10-bin
$\chi^2$ against $\mathrm{Uniform}(0,1)$ gives
$\chi^2_{\text{uniform}} = 22{,}328.6$ — three orders of magnitude past
the $\chi^2_{9, 0.95} = 16.92$ critical value. $67\%$ of *PIT* mass
sits in the extreme bins $[0, 0.1] \cup [0.9, 1.0]$; the distribution
is sharply U-shaped with a heavy spike at $0$. The Gaussian forecast
assumption is empirically wrong, and this is the forward-observable
consequence of the strongly non-Gaussian innovation already documented
in [mitacs-rebaseline-facts] (excess kurtosis $\approx 24$–$33$).
**(b) Secular drift is the strongest legitimate signal.** Excluding
two tautological candidates (he\_dev\_from\_daytype\_mean and
dow\_he\_interaction — both near-monotone functions of $z_{\text{actual}}$
within cells, median within-cell $|\rho|=0.65$), the ranking becomes
$\text{year\_since\_cutoff} = 1.11 > \sin_{\text{semi}} = 0.93 > \sin_{\text{ann}} = 0.87 > \cos_{\text{semi}} = 0.76 > \cos_{\text{ann}} = 0.68 > \text{dow}_{\text{Mon}} = 0.46$
(borderline, $p=0.043$). **(c) Day-type-specific bias pattern.** Sunday
cells have $T_c \in [+0.05, +0.20]$ (forecast under-predicts); weekday
and Saturday cells have $T_c \in [-0.33, -0.13]$ (forecast over-predicts),
with magnitude growing in winter and shoulder; deepest cell is
winter/morning/saturday at $T_c = -0.33$.

### Broader picture

First AMBIGUOUS verdict in the applied-audit registry, and the
workflow's first encounter with the "criterion fires on legitimate-but-
unanticipated data structure" failure mode — the arbiter pattern's
mechanical-plus-substantive shape absorbed it cleanly. Under
$\texttt{branching\_rules}[P1][\text{AMBIGUOUS}] = \texttt{null}$, the
[2026-05-26_missing-content-thread] exhausted at $P1$, preemptively
closing $P2A/P2B/P2C/P2D$. The U-shaped marginal *PIT* motif became
the empirical proximate cause for opening the distributional-class
inquiry: [2026-05-26_q1-student-t-vs-gaussian] swapped the Gaussian
predictive kernel for Student-$t$ and reduced marginal $\chi^2$ from
$22{,}329$ to $1{,}371$ — a $16\times$ shrinkage that also collapsed
v2's headline $D_{\text{RATIO}}$ from $28.96$ to $2.93$, retroactively
qualifying $(S1)$. The secular-drift signal $\text{year\_since\_cutoff} = 1.11$
corroborates [mitacs-clim-gap-nonstationary] and stands as the
most-cited candidate $Z$ for the eventual stratified Patra-Sen test on
the disintegration obstruction. Phase-A design hole noted: candidates
constructed from $z_{\text{actual}}$ are tautologically circular
against $u - \tfrac{1}{2}$ in this metric and must be excluded a
priori in any successor preregistration.
