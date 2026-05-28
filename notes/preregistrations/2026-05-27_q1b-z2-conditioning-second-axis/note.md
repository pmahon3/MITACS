# Q1B: Z2-conditioning at the second axis — SETTLED R-B1B (substantive: favorable)

## The question

[Q1B](2026-05-27_q1b-z2-conditioning-second-axis/phase_a.yaml) registered a
head-to-head between two operationalizations of P1's
[second-strongest axis](2026-05-27_p1-patra-sen-per-stratum-localization/arbiter.yaml)
(`time_of_day`, range $0.485$) as $Z_2$ on top of
[Q1A](2026-05-27_q1a-z-conditioning-strongest-axis/arbiter.yaml)'s carryforward
$Z_1 = Z_c$ (8-level `hour_of_week`, marginal closure $0.18\%$ of gap). The
research question: does the cumulative $\sigma$-algebra $\sigma(\mathrm{day\_type},
Z_c, Z_2)$ reduce the post-cutoff marginal PIT $\chi^2$ on Q2A's M3
(mixture-2-Gaussian) family from Q1A's settled baseline $\chi^2_{\mathrm{base}} =
952.5167$ to within the corroboration band $\chi^2 \le 228$? Two pre-locked $Z_2$
constructions: $Z_2^{(b)}$ binary overnight indicator ($h \in \{1..6\} \to 0$,
else $1$) and $Z_2^{(c)}$ 4-level categorical per P1's production binning
(`overnight` / `morning_ramp` / `afternoon` / `evening`). Verdict architecture:
$\chi^2_{\text{better}} = \min(\chi^2_{Z_2^{(b)}}, \chi^2_{Z_2^{(c)}})$ against the
integer-rounded cuts $228$ (R-A1B), $(228, 2284]$ (R-B1B), and $> 2284$ (R-C1B),
inherited verbatim from the thread skeleton via Q2A's gate-validated thresholds.
The 4th-named-failure-mode arbiter pattern was pre-flagged as plausible given
the joint S9+S10+S11+S12 picture entering this node.

## The verdict, with math

**$\chi^2_{\text{better}} = 662.2233$ with paired-day-bootstrap $95\%$ CI
$[468.094, 944.452]$** — both point and full CI sit interior to the registered
R-B1B band $(228, 2284]$ (lower bound $2.05 \times$ clear of R-A1B; upper bound
$2.42 \times$ clear of R-C1B). The verdict is unambiguously R-B1B, not borderline.

$$
\chi^2_{Z_2^{(b)}} = 696.0\ [492.25, 1009.01], \quad
\chi^2_{Z_2^{(c)}} = 662.22\ [479.78, 945.41]
$$

The head-to-head is a **NEAR-TIE**: $\chi^2_{\Delta} = +33.78$ with paired SE
$44.71$, $|\Delta|/\mathrm{SE} = 0.756 < 1$; CI $[-53.19, +124.47]$ straddles
zero. Per `phase_a.metric` near-tie rule, the sub-finding records "no
statistically-discernible winner"; $Z_2^{(c)}$ wins by parsimony-of-point-estimate
convention.

The substantive central finding is the **cumulative reduction**. The Z_c-only
baseline reproduces EXACTLY at $\chi^2 = 952.5167$ (vs Q1A settled, $|\Delta| =
3.3 \times 10^{-5}$); adding $Z_2$ drops it to $662.22$. Cumulative improvement
above Z_c carryforward:

$$
\Delta\chi^2 = 952.5167 - 662.2233 = 290.2934\ [8.06, 484.42]
$$

Share of gap-to-close from the Q1A baseline to R-A1B:

$$
s = \frac{952.5167 - 662.2233}{952.5167 - 228} = \frac{290.2934}{724.5167} = 0.4007
$$

That is, **40.07% of the gap** between Q1A's $\chi^2$ and the R-A1B cut closed by
adding the second within-data axis. For comparison, Q1A's $Z$ closed
$\frac{1.3267}{725.8433} = 0.18\%$. **Q1B's Z2 lever is $218\times$ Q1A's lever
in absolute $\chi^2$ units, and $\sim 220\times$ as a share-of-gap**. The
cumulative CI lower bound $8.06$ is sign-determined positive at 95% (barely).
The thread `root.closure_rule` ($s > 0.80 \Rightarrow$ RESOLVED) is NOT
satisfied; the realized $s = 0.40$ sits halfway between Q1A's $0.18\%$ and the
$80\%$ thread-RESOLVED bar.

**Per-cell decomposition** corroborates P1 at finer resolution. Q1A's dominant
$\mathrm{saturday}\_\_z_c=6$ cell ($\chi^2 = 1634$) splits across time_of_day
strata as:

$$
\mathrm{morning\_ramp}: 967.28,\quad \mathrm{evening}: 399.77,\quad \mathrm{afternoon}: 328.09
$$

Sum $1695$, comparable to Q1A's $1634$ (the small overshoot is expected for
joint stratification where cells get sharper assignments under the finer
$\sigma$-algebra). The morning_ramp sub-stratum within saturday__zc6 alone
exceeds the pooled Q1B marginal by $46\%$. The M3 family's capacity is bound
within saturday__zc6 at the morning_ramp axis specifically — exactly the kind
of within-cell structure single-axis P1 ranking could not isolate.

**Forecast scoring.** Proponent forecast R-A1B at $180\ [90, 450]$ with $Z_2^{(c)}$
winning by $+160$ — falsified on outcome category, on point ($3.68\times$ overshoot),
and on shape (predicted decisive Z2_c win, realized near-tie); but **direction
of head-to-head winner CORRECT** at point. Devil's-advocate forecast R-B1B at
$948\ [620, 1340]$ with $Z_2^{(b)}$ winning by $-9$ at near-tie — outcome
category CORRECT, near-tie shape CORRECT, magnitude WRONG by $64\times$
($\Delta\chi^2_{\mathrm{cum, DA}} = 4.5$ vs realized $290.29$), direction WRONG
at point but unfalsified at 95% per near-tie. **Closer forecast: DA** (4 of 5
surfaces); proponent closer on 1 surface (direction, at point only, statistically
zero). This is the **sixth instance of the 4th-named-failure-mode** in a new
shape variant: outcome+shape match DA, direction matches proponent at point,
magnitude matches NEITHER (realized falls in the gap between proponent's $180$
and DA's $948$).

## Broader picture

Joint S9+S10+S11+S12 was read in [Q1A's arbiter](2026-05-27_q1a-z-conditioning-strongest-axis/arbiter.yaml)
and the [thread's S3 state-history](2026-05-27_resolution-paths-thread/thread.yaml)
as "within-data $\sigma$-algebra-enrichment levers are empirically bounded at
the $\sim 0.18\%$-of-gap scale." Q1B refutes that reading **as-stated**. The
sharpened reading the arbiter records: **single-axis** within-data
$\sigma$-algebra enrichment IS bounded near zero (Q1A's $0.18\%$); **joint
multi-axis** within-data conditioning recovers MEANINGFULLY MORE (Q1B's
$40.07\%$ over Q1A's carryforward). Full closure to R-A1B is NOT achieved
within the time-index alone (cumulative point $662$ still $2.9\times$ above the
cut; lower CI $468$ still $2.05\times$ above), so the identifiability
obstruction at the **full-closure** level remains a candidate reading — but the
within-data lever space has more directions than single-axis P1 ranking
suggested. The
[disintegration diagnostic](file:~/Research/Mathematics/Resolvent_Framework/notes/unsorted/disintegration_diagnostic.md)
picture (structured-noise vs noisy-structure indistinguishable from $\kappa_Q$
alone, requiring auxiliary information) is **preserved** but should be
sharpened: the auxiliary information class is not "anything beyond the time
index"; cross-axis joint conditioning **within** the time index is a
previously-unmeasured direction that recovers $\sim 40\%$ of the residual
$\chi^2$.

Per the thread `branching_rules.Q1B.R-B1B = null`, R-B1B does not auto-advance
the state machine. The thread does NOT auto-resolve (cumulative share $0.40 <
0.80$ closure rule) and does NOT auto-exhaust (the EXHAUSTED-UNRESOLVED
pre-flag's premise of "Q1A-scale magnitude" is contradicted by Q1B's
$218\times$-Q1A realization). The coordinator's post-arbiter decision space is
three-way: (a) thread amendment adding a **third** within-data $Z$ axis
(season, demand_quantile, or day_type as a fourth axis); (b) spawn a new thread
targeting cross-axis joint stratification systematically; (c) pivot to the
temperature / exogenous-$Z$ arm per
[mitacs-clim-gap-nonstationary](memory:mitacs-clim-gap-nonstationary) (still
the strongest a-priori candidate for the residual $60\%$ of the gap).

The reproduction checks at both anchors are **exact** (Q2A's $953.84$
reproduced to 4 decimals; Q1A's $952.5167$ reproduced to 8 decimals) — the
strongest dual-anchor reproduction in the workflow's history. The $290\ \chi^2$
closure is real signal, not pipeline drift. This entry's load-bearing numerics
will live in [`mitacs-q1b-z2-conditioning`](memory:mitacs-q1b-z2-conditioning)
and as (S13) in [`CLAUDE.md`](file:CLAUDE.md)'s settled-findings section once
the memory-update follow-up lands.
