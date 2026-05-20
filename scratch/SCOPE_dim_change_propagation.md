# Scoping note: dimension-change-aware variance propagation

**Status: EXECUTED 2026-05-19. Option C FALSIFIED by a known-answer
check; Option A built, gate-validated, and run on all 132 days.
OUTCOME: NEITHER dominates (validated full-population), corroborating
the single-dim sub-result — the load-bearing exclusion is RESOLVED.
See the resolution box below.**
**Provenance-grade: INSPECTION-ONLY — design analysis + outcome record,
not a make_result artifact.**

> ### ✅ RESOLUTION 2026-05-19 — Option A executed, exclusion resolved
> Option A (d_max-augmented state) was built into
> scratch/multistep_variance_propagation.py
> (`augmented_state_transition`), Stage-1 gate-validated (machine-
> precision vs VAR(1) closed form; bit-identical to the single-dim
> recursion on single-dim days, max|Δ|=2.8e-15), then run on all 132
> delivery days via scratch/run_error_decomposition.py (scope
> `augmented`). Result: A ≈4.8% [0.7,8.6], B ≈2.3% [-0.0,4.8], ν≈6.5,
> **NEITHER dominates** — agrees with the single-dim `clean` verdict
> (≈6.1%/≈9.1%). Discriminating per-row check confirms `augmented` is
> genuinely distinct from the unvalidated `project` probe (machine-
> zero on single-dim days, diff up to 7e-2 on dim-change days). The
> ~42% exclusion is resolved, not caveated. Substantive conclusion
> unchanged: document both A & B, build nothing, registered experiment
> unchanged. Full record: memory mitacs-error-decomposition-verdict.

> ### ⚠ CORRECTION 2026-05-19 — Option C does not exist as written
> The §3 "Option C" claim that a 1-D scalar recursion
> `s²_{k+1} = c0[0]²·s²_k + σ²` suffices *because Σ_j is rank-1* is
> **false**. The per-step innovation is rank-1, but the *propagated*
> covariance `P_k` is **not** maintained rank-1: the shift in `J`
> generates non-zero off-diagonal `P_k[a,b]` (a,b>0) that feed back
> into `P_k[0,0]`. A deterministic known-answer check (d=2, fixed
> shift-with-c0 `J`, rank-1 `Σ_step`) shows the true full recursion
> `P_k[0,0]` = 0.050, 0.082, **0.126180** for k=1..3 while the
> proposed scalar recursion gives 0.050, 0.082, **0.102480** — they
> diverge once the cross-terms feed back (k=3). `P_k[0,0]` genuinely
> requires the **full matrix** recursion.
> **Consequence:** Option C is withdrawn. Extending it to track the
> missing cross-terms *is* Option A — do not relabel. Option B remains
> disqualified (no known-answer gate). **Option A is the path, not the
> fallback.** §3–§6 below are kept verbatim for the design trail; read
> them through this correction.

## 1. The problem this would solve

The error-decomposition diagnostic's clean (validated) verdict —
*NEITHER deficiency A nor B dominates; document both, build nothing* —
is computed **strictly on single-embedding-dimension delivery days**
(~58% of the 2021–22 dev benchmark). Day-type-rollover days, where the
embedding dimension changes mid-forecast (weekday d=2, saturday d=4,
sunday d=3; the day-type rolls at the 07:00 anchor), are excluded
because the k-step predictive-variance composition

```
P_{i+1} = J_i^T P_i J_i + Σ_step,i      (J_i is d_i × d_i)
```

is **mathematically undefined when d_i ≠ d_{i+1}** (the matrix product
`J_i^T P_i J_i` requires conformable dimensions).

The sensitivity sweep (reset / project / h1) **failed its
pre-registered robustness rule**: the verdict is *not* shown robust to
how the excluded ~42% are handled, so the exclusion is **load-bearing**.
The clean verdict stands but is *conditional on the single-dim
subpopulation*. The honest concern: if Deficiency A (phase
mis-centring) concentrates at day-type seams — plausible, since seams
are where the asymmetric ramp is most violent — the clean verdict
**under-counts A**, and the "neither dominates" conclusion could change
on the full population.

A correct dimension-change-aware propagation is the **only honest route
to including the rollover days** and settling whether the verdict holds
population-wide.

## 2. Why the point forecast is fine but the covariance is not

The production *point* forecast already crosses dimension changes
cleanly: it rebuilds the lag-state vector from `zhist` each step
regardless of `d`, so a scalar prediction is always produced. Only the
*covariance* composition breaks, because `P` is a `d×d` object that
must be pushed through a Jacobian whose input and output dimensions
differ across the seam. The state vector itself changes meaning at the
seam (different number of lag coordinates), so there is no canonical
`d_i×d_{i+1}` linear map without an explicit modelling choice.

## 3. Math options for propagating P across a dimension change

Three are credible. Each is a *modelling choice* with different
honesty/cost.

### Option A — Common max-dimension state augmentation
Embed every step in a fixed `d_max`-dimensional state (here d_max = 4,
the saturday dimension). Lower-dimensional day-types use only the first
`d` coordinates; the unused trailing lag coordinates are carried as
**deterministic, known** components (they are actual past z-values, not
stochastic). The Jacobian is then always `d_max × d_max` with a
block structure: the active `d×d` block is the real local map, the
inactive block is an identity/shift on known coordinates with **zero
innovation variance**.

- **Pro:** the propagation recursion becomes dimension-stable by
  construction; `J^T P J` is always conformable. The inactive
  coordinates contribute zero innovation (they are known), so this is
  not an approximation *of the variance* — it is an exact bookkeeping
  of a constant-dimension state.
- **Con:** requires care that the inactive coordinates really are
  deterministic at each step (true here: they are lagged actuals /
  prior point predictions, same as the point-forecast path uses).
  The "augmented" covariance has structurally zero rows/cols for
  inactive coords — must verify this doesn't destabilise the
  rank-1 structure or the eigen-bookkeeping.
- **Known-answer gate exists?** YES. A constant-`d_max` VAR(1) with a
  block-structured transition (active block = A, inactive = shift,
  Q rank-1 in coord 0) has the same closed form
  `Σ_k = Σ_{i<k} (Aᵀ)^i Q A^i` already used in the Stage-1 gate. The
  augmentation is validated by showing it reproduces the
  single-dim closed form on single-dim days AND a hand-computed
  block-augmented K=2 reference. **This is the decisive advantage.**

### Option B — Explicit cross-dimension transition operator
Define an explicit `d_i × d_{i+1}` linear map at the seam (e.g. the
least-squares map between the two day-types' embedding blocks on a
shared time window) and propagate `P_{i+1} = M^T P_i M + Σ_step`.

- **Pro:** does not assume the inactive-coordinate structure of A;
  data-derived.
- **Con:** the cross-dim map `M` is itself an **estimated object with
  no closed-form ground truth** — it would need its own validation,
  and there is no synthetic system whose true cross-embedding-dimension
  covariance is known analytically. This re-creates exactly the
  "unvalidated approximation" status that disqualified reset/project.
- **Known-answer gate exists?** NO clean one. This is the weakest
  option on the project's validate-before-use bar.

### Option C — Marginalize to the scalar innovation channel
Abandon the full `d×d` P. Track only the scalar predictive variance
`s²_k` of coordinate 0 via a 1-D recursion that uses only the
coordinate-0 row of each `J_i` (the predictive-coefficient vector) and
the scalar `Σ_i[0,0]`. Because `Σ_j` is **rank-1 by construction**
(only coord 0 is stochastic — established in §3 of the paper), the
multivariate P may be unnecessary for what the diagnostic actually
consumes (it only ever reads `P_k[0,0]`).

- **Pro:** a scalar recursion has *no dimension-conformability problem
  at all* — it sidesteps the seam entirely. Cheapest. Directly aligned
  with the rank-1 structural fact the project already established and
  validated.
- **Con:** requires proving the scalar recursion is *exactly
  equivalent* to `P_k[0,0]` from the full recursion on single-dim days
  (it should be, given rank-1, but this is the thing to verify, not
  assume). The cross-seam scalar map still needs the coord-0
  coefficients defined across the dimension change — but a *vector*
  of predictive coefficients is well-defined for any `d` (it is just
  the local regression of next-z on the d lags), so the scalar channel
  does not have the matrix-conformability obstruction.
- **Known-answer gate exists?** YES — same VAR(1) closed form,
  read at `[0,0]`; plus an exact-equivalence check against the
  validated full recursion on single-dim days (a pure refactor test,
  no new ground truth needed).

## 4. Recommendation (for the user to decide, not pre-decided)

**Option A or Option C are the only two that meet the project's
validate-before-use bar** (both have a closed-form known-answer gate;
Option B does not and would repeat the reset/project failure). Between
them:

- **Option C is cheapest and most aligned** with the already-validated
  rank-1 structural fact, and its gate is partly a refactor-equivalence
  test (cheap, no new synthetic). Risk: the equivalence proof must be
  done carefully, not assumed.
- **Option A is more general** (keeps the full covariance, useful if
  any future diagnostic needs more than `[0,0]`) but heavier and the
  augmentation bookkeeping must be gate-checked end-to-end.

Recommended if pursued: **Option C**, with Option A as the fallback if
the scalar-equivalence proof does not hold cleanly.

## 5. Cost / benefit

**Cost (Option C):** a Stage-1-style task — ~1 focused build:
implement the scalar cross-seam recursion; prove + gate exact
equivalence to the validated full recursion on single-dim days; add the
VAR(1) closed-form `[0,0]` gate; then re-run the A-vs-B diagnostic on
**all 132 days** and re-apply the pre-registered rule. Comparable in
size to the Stage-1 + Stage-2 work already done this session.

**Benefit:** settles whether "NEITHER dominates" holds on the full
population or whether including the rollover days flips it (most likely
toward A, if phase mis-centring concentrates at seams). Either outcome
is publishable-honest: confirmation hardens §3; a flip would *change
the honing conclusion* (A could become the licensed lever).

**What it does NOT change / is NOT required for:**
- It does **not** alter the clean-scope verdict already documented in
  §3 (that remains the validated single-dim result regardless).
- It is **not required for the registered forward experiment** — that
  proceeds with the model unchanged either way.
- It does **not** by itself license any forecasting-model change; it
  only potentially re-opens *which* deficiency (if any) the evidence
  points to.

## 6. Decision the user faces

1. **Build Option C now** — settle the load-bearing exclusion;
   ~1 Stage-1+2-sized effort; could change the honing conclusion.
2. **Defer** — record this note, proceed to the registered experiment
   with the §3-documented single-dim verdict; revisit only if the
   forward results or reviewers demand the full-population answer.
3. **Drop** — accept the single-dim verdict as the final honest
   statement; the exclusion stands as a documented scope limitation
   (it already is, in §3).

No option is pre-selected. Option 2 is the lowest-risk default and is
fully consistent with the project's "characterize, don't over-engineer"
discipline; Option 1 is the only path that could *strengthen or change*
the conclusion rather than just document it.
