## Q2A — richer family: mixture or nonparametric

### The question

Q1 ([2026-05-26_q1-student-t-vs-gaussian]) had just settled (S8) that
lifting the kernel from Gaussian to Student-t closes most of the marginal
PIT pathology — a $16\times$ chi-square reduction,
$\chi^2_{M_0} = 22{,}329 \to \chi^2_{M_1} = 1{,}371$ — but $\chi^2 \approx
1{,}371$ remained. Q2A asked whether richer-than-Student-t families close
the rest. Three candidates against $M_1$: $M_3$ (two-component centred
Gaussian mixture, 3 shape params per day-type), $M_4$ (three-component
mixture, 5 params), $M_5$ (nonparametric KDE, Silverman bandwidth,
$\sim 5000$ effective samples). Verdict metric:
$\chi^2_{\text{effective}} = \min(\chi^2_{M_3},\,\chi^2_{M_4},\,\chi^2_{M_5})$
on the 10-bin post-cutoff marginal PIT. Cuts from a synthetic gate
(METHOD/DESIGN, body_sha256 `2bab514e...`) under a variant-max conservative
rule: R-A2 corroboration at $\chi^2 \leq 228.44$, R-C2 falsification at
$\chi^2 > 2284.44$, R-B2 ambiguous in between with a phase_a-mandated
thread amendment on fire.

### The verdict, with math

Realized: $\chi^2_{M_3} = 953.84$, $\chi^2_{M_4} = 965.81$, $\chi^2_{M_5} =
966.60$, so
$$\chi^2_{\text{effective}} = \min(\chi^2_{M_3},\,\chi^2_{M_4},\,\chi^2_{M_5}) = 953.84\;[685.23,\,1356.92]$$
from a $1000$-resample paired-day bootstrap. R-B2 fires mechanically —
$\sim 4.2\times$ above the corroboration bar, $\sim 2.4\times$ below the
falsification bar. Three substantive findings carry more weight than the
mechanical label.
(1) **Distributional flexibility is bounded.** $M_3$, $M_4$, $M_5$ are
chi-square-indistinguishable: spread $\chi^2_{M_5} - \chi^2_{M_3} = 12.76$,
$\approx 1.4\%$ of any of the three, with $95\%$ CIs heavily overlapping.
The *simplest* candidate $M_3$ binds; the *most flexible* $M_5$ (KDE) is
*worst* by $0.8$ — refuting both forecasters' KDE-preferred sub-prediction.
This matches the proponent's pre-registered `what_would_change_my_mind`
#4 verbatim ("all three families produce nearly identical $\chi^2$"),
which prescribed routing to Q2B-class non-distributional investigations.
(2) **Day-type asymmetry dominates.** Per-day-type (recomputed from
pickle; phase_a secondary): $\chi^2_{\text{weekday}} \approx 285$,
$\chi^2_{\text{sunday}} \approx 560$, $\chi^2_{\text{saturday}} \approx
1{,}180$ — saturday is $\sim 4\times$ weekday, family-invariant. Pooled
$\sim 954$ is their $n$-weighted average; saturday binds.
(3) **The reduction curve asymptotes.** $M_0 \to M_1$: $16\times$;
$M_1 \to Q2A_{\text{best}}$: $1.44\times$. The first family lift extracted
most of the distributional signal that exists; the second a fraction.
This is the **first arbiter in the workflow's history without a
mechanical/substantive gap** — the gate-derived ambiguous band absorbed
the realization with room on both sides, and the substantive picture
(R-B2 pointing to Q2B) coincides with the mechanical fire. Thread
amendment A2 followed per phase_a.

### Broader picture

Q2A bounds the kernel-refinement lever: if $M_5$ cannot close the gap,
no further-flexible *distributional* test will, and the remaining $\chi^2
\approx 954$ must lie in non-distributional content. Thread amendment A2
reopened Q2B ([2026-05-26_q2b-non-distributional-decomposition]) under
the same $Q$ to test three non-distributional ablations against the
$725.40$ gap; Q2B settled R-C3 — all shares below the $15\%$ null floor,
two sign-determined negative — bounding the naive
$\sigma$-algebra-preserving manipulations too. Jointly S9 and S10 are
the *applied* face of the identifiability obstruction parked the same
day in the sibling theory programme: structured noise versus noisy
structure with $Z$ orthogonal to the current $\sigma$-algebra are
observationally indistinguishable from $\kappa_Q$ alone (Bergna et al.
2026 Prop 1; Heckman–Singer 1984; Allahverdyan 2020). Resolution requires
auxiliary information, not richer post-hoc corrections — picked up by
[2026-05-27_resolution-paths-thread]. Finding (2) independently
corroborates eigenmodes-v1's S2/S3
([2026-05-26_multiscale-defect-eigenmodes]); with Q1's S8c asterisk on
v2's $D_{\text{ratio}} = 28.96$, Q2A confirms no family will recover the
original magnitude. The [2026-05-26_distributional-class-thread] is
correctly classed EXHAUSTED — its `scope_limits` ("PREDICTIVE-
DISTRIBUTION CLASS axis only") is exactly what Q2A's R-B2 plus Q2B's
R-C3 jointly confirm the residual lies *outside*.
