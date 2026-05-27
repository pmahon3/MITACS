## Multiscale defect — eigenmodes (v1)

### The question

Following [2026-05-26_multiscale-direct-h-step-v2], which settled
$D_{\text{RATIO}} \approx 29$ as the headline Gaussian-kernel coherence
defect ([[mitacs-multiscale-direct-h-step-v2]]), this experiment asked a
structural follow-up registered against the multiscale seed's §5.2: does
$D_h = \Sigma_h - \Sigma_h^{\text{iter}}$ concentrate in one eigendirection
of the drift operator $C_1$, or spread across modes? The metric was the
per-eigenmode diagonal residual
$R_k(h, \text{dt}) = (V^{-1} D_h V^{-T})_{kk}$ with $C_1 = V \Lambda V^{-1}$,
share $\text{share}_k = |R_k| / \sum_j |R_j|$. Three mutually exclusive
outcomes were pre-registered: (O1) single-mode dominant —
$\max_k \text{share}_k > 0.70$ with the same dominant index per day-type;
(O2) horizon-localized — dominant index changes with $h$; (O3) diffuse —
all cells in $[0.40, 0.65]$ under 2-mode geometry. An asymmetry sub-test
asked whether v2's weekend/weekday KL gap is library-noise driven
($R_{\text{sat}}/R_{\text{wkd}} \in [3, 8]$, centred on
$n_{\text{wkd}}/n_{\text{sat}} \approx 5$) or structural.

### The verdict

Mechanical reading is O2; substantive reading is O3 — the **first MIXED**
finding in the workflow's history. Every weekday and saturday cell's
dominant share lies in $[0.50, 0.53]$; sunday in $[0.52, 0.58]$ with the
slow real mode (eigenvalue $\approx 0.977$) leading. Weekday shares run
$0.534 \to 0.514 \to 0.506 \to 0.503$ across $h \in \{2, 6, 12, 24\}$; the
argmax flip at $h{=}24$ that fires O2 mechanically is noise on a tie, not
horizon-localization. The substantive picture is O3 (diffuse across two
modes). Proponent's $\text{share}_{h=24}^{\text{wkd}} = 0.85$ forecast is
wrong by $0.34$; devil's-advocate's $0.55$ is within $0.05$ of the
observed $0.503$. Two Phase A criterion defects made the gap possible —
O2 lacked a "clearly above tie" share-margin guard, and O3's sunday band
ignored its mixed 1D-real + 2D-pair + 1D-real block geometry. The DA's
*mechanism* (rank-1 $\Sigma$ makes the projection ill-posed,
[[mitacs-rank1-structural]]) was refuted by the synthetic VAR(1)
cross-check: strict-semigroup data project to $R_k \approx 0.01$, three
orders below Ontario's $0.3$–$20$. The asymmetry sub-test is the
headline independent finding: all eight $R_{\text{sat}}/R_{\text{wkd}}$
ratios fall in $[1.26, 2.44]$, every one *below* the library band
$[3, 8]$. Direction: weekday per-cell defect exceeds the library-noise
prediction; saturday falls below. The weekend/weekday asymmetry is
**structural**, not statistical. This is the "fourth named failure mode"
from CLAUDE.md — a pre-registered criterion firing on
legitimate-but-unanticipated data structure — and the arbiter's
mechanical-verdict-plus-substantive-content shape is the workflow's
correct response.

### Broader picture

These are (S2) and (S3) in the CLAUDE.md settled-findings ledger. The
diffuse-defect reading rules out the seed's "embedding misses one
long-timescale DOF" recipe and points to multi-timescale missing
content — what [[mitacs-postcovid-probe]] and
[[mitacs-clim-gap-nonstationary]] flagged as the most-cited candidate
$Z$ (weather + secular drift). Downstream, the diffuse-defect picture
joins the case that residual non-Gaussianity is not reducible to a
sufficient sub-$\sigma$-algebra of the current state — formalized later
by joint S9 + S10 from
[2026-05-26_q2a-richer-family-mixture-or-nonparametric] and
[2026-05-26_q2b-non-distributional-decomposition], the applied face of
the identifiability obstruction in `disintegration_diagnostic.md`. The
structural-asymmetry headline corroborates v2's marginal weekend/weekday
KL pattern at the per-mode level while partly retracting one reading:
the marginal gap mixes genuine dynamical asymmetry with library-noise
inflation, in a direction more interesting than the marginal alone
implied. Workflow-significance: this entry prototyped the
mechanical-vs-substantive arbiter pattern reused at S8 (Q1, R-D
mechanical / R-C substantive) before S9 and S10 closed the
distributional-class thread. The Phase A criterion defects flagged here
(share-margin guards; block-dimension-weighted bands) are the durable
methodology lesson for future eigenmode-shape experiments.
