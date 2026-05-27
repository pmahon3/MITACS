## Q1 — Student-t vs Gaussian kernel on the marginal PIT defect

### The question

P1 ([2026-05-26_pit-calendar-fingerprint]) settled the marginal PIT
pathologically non-uniform at $\chi^2 = 22{,}329$ on $10$ uniform
bins — $\sim\!67\%$ of post-cutoff PIT mass piled in the two extreme
bins. The natural mechanism is in [[mitacs-rebaseline-facts]]: the
production-validated one-step innovation has excess kurtosis
$\kappa \in [24,\,33]$, far outside any Gaussian neighbourhood. Q1
asks the smallest non-trivial question that permits: does lifting
the iterated $\kappa_1$ predictor's kernel family from Gaussian to
Student-t calibrate the marginal? The verdict architecture cuts the
$\chi^2$ axis at R-A ($\min \chi^2 \le 30$, kernel-family is
load-bearing), R-B ($\chi^2 \in (30,\,500]$), R-C ($\chi^2 > 500$,
non-distributional mechanism), and R-D (MLE pathological — fitted
$\nu < 2.5$ or $\nu > 200$). Three models: M0 the registered
Gaussian; M1 Student-t kernel with one global $\nu$ per day-type;
M2 full Student-t commitment with per-(month, hour-of-day) Student-t
climatology. Secondary: the Student-t analogue $D_{\text{RATIO},t}$
of v2's headline $(h{=}12,\,\text{weekday})$ cell.

### The verdict, with math

SETTLED with a mechanical/substantive split — third instance of the
arbiter pattern resolving "pre-registered criterion fires on
legitimate-but-unanticipated data structure". Mechanical R-D: $207$
of $288$ M2 climatology bins fitted $\nu$ at the upper boundary
$\nu = 300$, tripping "$\nu > 200$". Substantively that saturation
is the Student-t MLE's correct response when a bin's $\sim\!690$
pre-cutoff samples are approximately Gaussian by CLT-like
aggregation over weather variation — not estimator failure but a
phase_a design hole. Excising R-D as artefact, the substantive
verdict is R-C:
$\min(\chi^2_{M_1},\,\chi^2_{M_2}) = 1{,}371\,[1{,}069,\,1{,}817] > 500$,
a $16\times$ reduction from M0's $22{,}329\,[19{,}028,\,25{,}731]$
but nowhere near R-A's $30$. The $\nu$-side of both forecasters'
prediction was essentially exact: from
$\kappa_{\text{excess}} = 6/(\nu-4)$ both predicted $\nu \approx 4.5$;
observed $\nu_{\text{wkday}} = 4.19$, $\nu_{\text{sat}} = 4.62$,
$\nu_{\text{sun}} = 4.61$ — the in-sample kurtosis-to-$\nu$ mapping
holds out-of-sample. The secondary collapsed:
$D_{\text{RATIO},t} = 2.93\,[2.62,\,3.24]$ against v2's Gaussian
$D_{\text{RATIO}} = 28.96$, a $\sim\!10\times$ shrinkage —
$\sim\!90\%$ of v2's headline magnitude was distributional-class
artefact, the (S8c) asterisk on
[2026-05-26_multiscale-direct-h-step-v2]. $M_1 < M_2$ on marginal
$\chi^2$ ($1{,}371 < 1{,}754$) localizes the heavy-tailedness to the
$\kappa_1$ innovation, not the seasonal climatology.

### Broader picture

The residual $\chi^2 \approx 1{,}371$ drives the rest of the
distributional-class line of inquiry. The arbiter flagged the
mechanical-R-D branching as wrong (Q2D would redesign a working
estimator) and routed via thread amendment A1 to
[2026-05-26_q2a-richer-family-mixture-or-nonparametric], which
SETTLED R-B2 (S9) with three richer families
$\chi^2$-indistinguishable at $\chi^2 \approx 954$ — bounding the
kernel-refinement lever. That motivated
[2026-05-26_q2b-non-distributional-decomposition], which SETTLED
R-C3 (S10) with all three same-$Q$ ablation shares below the $15\%$
null floor and two sign-determined negative. The joint S9+S10
reading is the applied face of the identifiability obstruction
(`Resolvent_Framework/notes/unsorted/disintegration_diagnostic.md`):
structured noise versus noisy structure are observationally
indistinguishable from $\kappa_Q$ alone, so resolution requires
auxiliary information — picked up by
[2026-05-27_resolution-paths-thread]. Q1 is the first node of
[2026-05-26_distributional-class-thread]; every later node inherits
its headline finding that the heavy-tailed kernel is necessary but
not sufficient. The (S1)/(S6) qualifier on
[2026-05-26_multiscale-direct-h-step-v2] is permanent: v2's SETTLED
verdict is retained, but the magnitude reads $D_{\text{RATIO},t}
\approx 3$ once the kernel family is corrected, not
$D_{\text{RATIO}} \approx 29$.
