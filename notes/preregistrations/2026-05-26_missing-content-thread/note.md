## missing-content-thread

### The question and scope

This thread asked: what content is missing from the iterated $\kappa_1$
predictor that produces the coherence defect documented in
[2026-05-26_multiscale-direct-h-step-v2] $(S1)$ and
[2026-05-26_multiscale-defect-eigenmodes] $(S2)$/$(S3)$ — and which
dimensions of variation (calendar, day-type fine-structure, secular
drift, exogenous weather) does the demand-history embedding fail to
span that the post-cutoff data actually carries? Scope was deliberately
narrow: a *characterisation* line of inquiry, not a candidate-predictor
construction line. Backtest-style "does adding $X$ make the predictor
better?" experiments (seed §6.3) and joint-MLE on richer
parameterizations within the existing embedding (seed §6.4) were
out-of-scope by explicit `scope_limits` — they belong to downstream
threads if and when this one identified a specific candidate. The
thread was meant as a cheap quick-win probe before the heavier
distributional-class line: ~3–4 hours of code, no data acquisition,
re-using v2's $(C_1, \Sigma_1)$ from `factors_v2_final.pkl` via the
existing $\texttt{experiment/predict}$ machinery.

### The tree and the rules

Single-root tree: $P1$ (`pit-calendar-fingerprint`) was the only
unconditional node; $P2A$/$P2B$/$P2C$/$P2D$ were sibling children
contingent on $P1$'s outcome class. Branching rules pre-registered
the full mapping from $P1$'s `outcome_categories` to children:
$O\text{-}A \!\to\! P2A$ (temperature against a seasonal fingerprint),
$O\text{-}B \!\to\! P2B$ (holiday calendar fine-structure),
$O\text{-}C \!\to\! P2C$ (weather $\times$ calendar joint), and
$O\text{-}D \!\to\! P2D$ (nonlinearity-vs-exogenous fallback). The
arbiter-level outcomes were mapped explicitly:
$\texttt{branching\_rules}[P1][\textsf{AMBIGUOUS}] = \texttt{null}$
and $\texttt{branching\_rules}[P1][\textsf{MIXED}] = \texttt{null}$.
That last mapping is the load-bearing pre-registration here: AMBIGUOUS
was registered as a *terminus*, not a continuation prompt. The
rationale: with no calendar candidate clearly dominating, naming a
post-hoc follow-up would amount to choosing the next experiment with
the data in hand — exactly the silent-pivot failure mode threads exist
to catch. Closure-by-design protects the line of inquiry against the
analyst's own temptation to amend mid-stream.

### Post-mortem

$P1$ settled SETTLED-AMBIGUOUS $(S4)/(S5)$ with three durable findings:
**(a)** the marginal $\textit{PIT}$ is pathologically U-shaped —
$\chi^2_{\text{uniform}} = 22{,}328.6$ against $\chi^2_{9,0.95} = 16.92$,
with $\sim\!67\%$ of $\textit{PIT}$ mass in the extreme bins
$[0,0.1] \cup [0.9,1.0]$, confirming the Gaussian forecast assumption
is empirically wrong; **(b)** $\texttt{year\_since\_cutoff}$ (secular
drift, score $1.11$) is the strongest *legitimate* signal once two
tautological candidates ($\texttt{he\_dev\_from\_daytype\_mean}$ and
$\texttt{dow\_he\_interaction}$, both near-monotone in $z_{\text{actual}}$
with median within-cell $|\rho| = 0.65$) are excluded; **(c)** a
sign-specific day-type bias — Sunday cells under-predict
($T_c \in [+0.05, +0.20]$), weekday and Saturday cells over-predict
($T_c \in [-0.33, -0.13]$), deepest at winter/morning/saturday. The
mechanical AMBIGUOUS fired the registered
$\texttt{branching\_rules}[P1][\textsf{AMBIGUOUS}] = \texttt{null}$
edge; sibling nodes $P2A$/$P2B$/$P2C$/$P2D$ were preemptively closed;
the thread transitioned `active_to_exhausted` ($\texttt{state\_history}$
entry $S1$, $\texttt{prior\_hash} = 805df9d6\ldots$) with no
amendment pursued. The U-shaped marginal $\textit{PIT}$ motif was
*the* finding that motivated the
[2026-05-26_distributional-class-thread] immediately afterward —
heavy-tailedness needed a kernel change, not more calendar features,
and [2026-05-26_q1-student-t-vs-gaussian] $(S8)$ duly reduced the
marginal $\chi^2$ by $16\times$ to $1{,}371$. The secular-drift signal
cross-links to [mitacs-clim-gap-nonstationary] as the most-cited
candidate $Z$ (temperature) for the eventual disintegration-resolution
work. Phase-A design hole logged: candidates constructed from
$z_{\text{actual}}$ are tautologically circular against $u - \tfrac{1}{2}$
and must be excluded a priori in any successor preregistration.
