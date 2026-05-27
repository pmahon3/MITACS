## v1 — multiscale direct h-step (RETIRED; superseded by v2)

### The question we meant to answer

v1 set out to test the multiscale seed's relaxed-factorization picture (C):
whether the directly-fit $h$-step factor $(C_h, \Sigma_h)$ diverges
structurally from the iterated $1$-step factor
$\bigl(C_1^{h},\, \sum_{j=0}^{h-1} C_1^{j}\,\Sigma_1\,(C_1^{j})^{\!\top}\bigr)$
on pre-cutoff Ontario z-scores at $h \in \{2, 6, 12, 24\}$ per day-type. Two
scalar metrics were pre-registered. Drift agreement: the relative-Frobenius
norm $M_1 = \max_{h, dt}\,\|C_h - C_1^{h}\|_F / \|C_h\|_F$. Diffusion
underestimation: the diagonal ratio $M_2 = \Sigma_h[0,0] /
(\sum_j C_1^{j}\,\Sigma_1\,(C_1^{j})^{\!\top})[0,0]$ at the headline cell
(weekday, $h{=}12$). Corroboration required $M_1 \le 0.20$ AND $M_2 > 1.5$;
falsification, $M_1 > 0.50$ OR ($M_2 \in [0.7, 1.3]$ AND $M_1 \le 0.20$).
The proponent forecast $(M_1, M_2) = (0.12, 2.4)$; the devil's-advocate
countered $(0.55, 1.1)$, naming `degenerate-large-h-factor` as the rank-1
failure mode.

### The metric defect that retired it

The Ontario reading at the headline cell was $M_1 = 0.847\,[0.826, 0.896]$
and $M_2 = 0.730\,[0.719, 0.740]$ — the $M_1 > 0.50$ clause fired cleanly,
and $M_2$ ran in the *opposite* direction to the proponent's forecast. But
the synthetic VAR(1) gate is what retired the metric, not the result. On a
system where the strict semigroup $C_h = C_1^{h}$ holds *by construction* —
i.e. where picture (C) is false by fiat — the same pipeline returned
$M_1 = 1.83$ at $h{=}12$. The pathology is structural and was the
devil's-advocate's predicted failure mode:
$M_1 = \|C_h - C_1^{h}\|_F / \|C_h\|_F$ is degenerate as $\|C_h\|_F \to 0$.
For a stationary system, the OLS pair $(z(t), z(t+h))$ decorrelates with
$h$, so $C_h$ shrinks toward the noise floor; the denominator vanishes
faster than the numerator and $M_1$ blows up regardless of whether the
iterated and direct factors are close in absolute terms. The cuts on $M_1$
therefore cannot separate "successful composition of the strict semigroup"
from "vanishing operator." The arbiter recorded DEVILS-ADVOCATE-CONFIRMED
at the mechanical level but stamped the finding PROVISIONAL with explicit
re-audit instructions: redefine the metric to be well-posed in the
$C_h \to 0$ limit. v2 ([2026-05-26_multiscale-direct-h-step-v2]) discarded
$M_1$ entirely in favour of an empirical-state-averaged 1-D Gaussian KL on
the scalar innovation coordinate — $D_{\text{RATIO}} =
\mathrm{kl}_{\text{Ontario}} / \mathrm{kl}_{\text{synthetic}}$, well-posed
under the rank-1 $\Sigma_j$ structure ([[mitacs-rank1-structural]]) — and
the verdict on the same underlying fits *inverted* to PROPONENT-CONFIRMED at
$D_{\text{RATIO}} = 28.96\,[26.82, 31.43]$.

### Why this entry remains in the registry

Append-only discipline. Failed experiments are not deleted; they receive a
successor and a PROVISIONAL marker on the YAMLs, and the audit trail
preserves what was tried. v1 stays for two reasons. First, the
v1 $\to$ v2 transition is itself one of the workflow's named retraction
modes — degenerate-metric reimplementation of a structural question,
sibling to the "diagnostic reimplements production" class catalogued in
[[mitacs-session-lessons-corpus]] — and silently deleting v1 would erase
the precedent that justifies promoting the synthetic baseline from
diagnostic to precondition (as v2 did). Second, any future revisit of the
multiscale picture has to know what was tried: $M_1$ as defined cannot be
revived for stationary systems at large $h$, and a replacement metric
must demonstrate well-posedness under $C_h \to 0$ before it earns a
phase_a. The downstream eigenmode characterization
[2026-05-26_multiscale-defect-eigenmodes] (S2/S3) is a v2 child; v1's only
durable contribution is the negative result on the metric, plus the
synthetic gate it carried — promoted to v2's precondition.
