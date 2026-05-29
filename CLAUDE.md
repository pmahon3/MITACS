# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research codebase for nonlinear dynamics / empirical-dynamic-modelling analysis of Ontario
electricity demand (IESO). It detrends raw demand into a stationary z-score series, finds an
embedding dimension per day-type, fits local-linear (weighted-least-squares) one-step forecasts
across a θ-grid, and explores the residuals / local linear operators in Dash dashboards.

The current layout was established by commit `629948e "changed course"` on the **`operator`**
branch (the default branch is `main`; on `main` the layout differs — it predates this restructure).

## Critical environment fact

`edynamics.modelling_tools` (used by every `processing/` script: `Lag`, `Embedding`,
`WeightedLeastSquares`, `Exponential`, `Minkowski`, `dimensionality`) is **not** declared
anywhere in this repo — there is no `requirements.txt`, `setup.py`, or `pyproject.toml`.
It is an **editable install** pointing at a sibling project:

```
/Users/pmahon/Research/Dynamics/Takens_Whitney/EmpiricalDynamics/src
```

(see `.venv/lib/python3.10/site-packages/__editable__.edynamics-0.3.14.pth`). If imports of
`edynamics` fail, the fix is in that sibling repo / the venv, not here. Despite
`.venv/pyvenv.cfg` claiming Python 3.12, the live `site-packages` is `python3.10/` — treat the
venv as Python 3.10.

Other key deps already in `.venv`: `torch`, `ray`, `dash`, `plotly`, `pandas`, `bs4`, `pyyaml`.

## Configuration (read this before touching paths)

All pipeline paths and parameters live in **`config/pipeline.yaml`**, loaded once via
`from config import load_config`. `config/` resolves every path relative to `PROJECT_ROOT`
(the repo root, derived from `config/__init__.py`'s location) — **nothing depends on the
current working directory**. The loader rejects leading-slash / `..` paths, so the old
root-anchored-path bug class fails loudly instead of silently reading `/`.

This **replaces** the old `.env` + `data/__init__.py:convert_to_absolute_path` mechanism
(which had a `preprocessing` vs `pre_processing` typo and resolved relative to the wrong
directory). `data.convert_to_absolute_path` is dead — do not reintroduce it. `cfg.embedding_dim(daytype)`
is the single home for the `pd.read_csv(...).idxmax()` embedding-dimension idiom.

## Pipeline (run order and the artifact contract)

Stages 2–4 have been refactored to a `main(cfg)` function plus a thin `if __name__` CLI
(`python -m <module>`); they are config-driven and headless-safe (`matplotlib.use("Agg")`).
Stages 5–6 are **not yet refactored** (see "Status" below). Stages communicate through files;
the **filename convention is the contract** (`save_all`/`load_all` discover artifacts by glob).

1. **Acquire** — `data/scraping.py` scrapes IESO `PUB_Demand*.csv`; `data/ontario/pickling.py` → `.pkl`.
2. **Detrend** — `pre_processing/year_wise_standardization.py` → `stationary_candidate.csv`
   (hourly `zscore` = demand de-seasonalized by month+hour-of-day) plus diagnostic plots.
3. **Day-type tagging** — `processing/clustering/process.py` adds `daytype ∈ {weekday, saturday,
   sunday}` using the configurable **07:00 day-anchor offset** (`data.day_anchor_hours`).
4. **Embedding dimension** — `processing/dimensions/process.py` → `params/results_<daytype>.csv`
   (uses `ray`; `ray.init()` is guarded so re-runs don't double-init).
5. **Fit / residuals** — `processing/{innovations, locality/global, locality/pointwise}/process.py`
   write the canonical artifacts: `residuals_<tag>.pt`, `coeffs_<tag>.pt` `(N,d,d)`,
   `times_<tag>.npy` `(N,)` int64 ns, `thetas_<tag>.npy`. `<tag>` = day-type.
6. **Explore** — Dash apps in `post_processing/` read a run dir via `load_all(run_dir)`:
   `python post_processing/innovations/dashboard.py --run-dir <output_dir> [--port 8050]`.

## Estimator design (current, post-rebuild)

The `innovations` stage estimates, per anchor, a local linear drift `C_j` and a
diffusion `Σ_j` defining the Gaussian Markov kernel `x' | x ~ N(x C_j, Σ_j)` on the
delay-embedding state space — an estimator of the programme's conditional-regularity
kernel `κ_Q` (see memory `mitacs-theory-correspondence` for the precise mapping and
its qualifiers; operator naming defers to `Resolvent_Framework`, no novelty claims).

Key facts (each was hard-won this session; do not "simplify" without reading the memory):

- **Bandwidth selection is true-LOO cross-validation**, not the edynamics
  `LocalGLSelector`. The GL criterion is degenerate for the normalized Gaussian kernel
  (no interior optimum — score monotone in bandwidth); the `Resolvent_Framework`
  programme has no bandwidth convention to defer to. `_theta_loo_cv` in `estimator.py`
  is the validated rule. (memory `mitacs-theta-rail-pinning`)
- **Diffusion is the plain μ-centred residual covariance** (`Y−X@C_j`), no residual
  kernel. The edynamics kernel-weighted covariance is `≈ w²·Q` and collapses; the
  plain covariance is the unbiased local second moment.
- **`Σ_j` is rank-1 by construction** of a single-variable delay embedding (coords
  2..d of the one-step image are deterministic shifts of the input). The only
  stochastic content is the scalar coordinate-0 innovation. `r_hat`/"multi-mode" is
  therefore incoherent for this embedding and is not reported. (memory
  `mitacs-rank1-structural`)
- **`day_anchor_hour`**: transitions whose one-step target lands on the day-anchor
  hour cross the day-type rollover seam and must be excluded — without this, `Σ_j` is
  catastrophically ill-conditioned. `process.py` passes `cfg.data.day_anchor_hours`.

## Validation gates (both must PASS — `python -m processing.innovations.validation.synthetic`)

Both gates call the **production** estimator directly (not reimplementations) — that
is why they are trustworthy:

1. **Recovery gate**: VAR(1) with known `(A, Q)` → `build_local_gaussian_semigroup`
   must recover drift/diffusion within tolerance.
2. **Non-Gaussianity gate**: Gaussian-innovation VAR(1) must read ≈Gaussian;
   Student-t-innovation VAR(1) must be clearly flagged heavy-tailed — validating
   `innovation_diagnostics` against ground truth.

`processing/innovations/validation/rebaseline.py` is the authoritative
production-path factual record (`mitacs-rebaseline-facts`).

## Discipline rule (the durable lesson of this session)

**A diagnostic that reimplements production logic is a hypothesis, not a finding,
until confirmed through the production code path.** This session, ≥3 confident Ontario
claims (sunday multi-mode; "mild"/"10×" non-Gaussianity; a 4-decimal "match") were all
reimplemented-diagnostic artifacts that inverted when finally run through production
functions. The gates never misled because they call production directly. Any Ontario
diagnostic must call production functions or its output is labelled provisional and
not committed as fact.

## Applied-audit workflow (added 2026-05-26)

A pre-registration / arbiter / thread workflow now sits in front of any
experiment that will produce a citable finding. Lives at:

- `.claude/agents/` — 11 agent definitions (code-path-auditor,
  phase-fidelity, preregister, devils-advocate, arbiter,
  thread-coordinator, multiverse, framework-escalator,
  pre-experiment-checklist, editorial-pass, literature-scout).
- `.claude/skills/audit/SKILL.md` — `/audit <mode>` dispatcher with
  explicit blocking rules per check level.
- `notes/preregistrations/` — hash-stamped registry artifacts
  (phase_a, proponent, devils_advocate, result, arbiter, multiverse,
  thread). README + `_template/` directory inside.
- `experiment/audit/registry.py` — Python CLI for the registry
  (`registry list`, `registry verify`, `registry status`,
  `registry new`, `registry thread {new,status,list}`).
- `experiment/audit/code_path.py` — production-vs-reimplementation
  static auditor with a 27-symbol whitelist (`student_t_mle_fit` and
  `global_ols_fit` are now in the whitelist).
- `notes/seeds/applied_audit_workflow.md` — design source.
- `notes/seeds/multiscale_factor_coherence.md` — the seed under
  current investigation.

Lifecycle for any new experiment that will produce a citable finding:

1. `preregister` writes phase_a.yaml; numeric falsification +
   corroboration criteria + numeric forecast required.
2. `devils-advocate` writes a counter-prediction stamped
   simultaneously with phase_a (sole agent that may write SETTLED
   is arbiter; devil's-advocate's prediction is itself graded).
3. `code-path-auditor` confirms the experiment script calls
   production functions, not reimplementations. BLOCKING if not.
4. `phase-fidelity` Check P confirms script implements what phase_a
   pre-registered. L1 (structural) BLOCKING; L2 (methodological)
   advisory but unresolved L2 blocks arbiter at result time.
5. Run.
6. `phase-fidelity` Check R confirms result.yaml reports the
   dependent variables phase_a registered. BLOCKING.
7. `arbiter` renders verdict against pre-registered criteria. ONLY
   agent that may write SETTLED.
8. If under a thread: `thread-coordinator` advances state per
   pre-registered branching_rules.

Lines of inquiry across multiple experiments use **threads**
(`<date>_<slug>-thread/thread.yaml`). The thread artifact pins
the tree of planned experiments and the branching rules so silent
post-hoc pivoting is detectable. Amendments append-internal,
hash-chained. State advancement preserves prior hash in
`state_history` for descendant back-references.

Three named retraction modes the workflow catches:
1. Diagnostic reimplements production (caught by `code-path`).
2. Argmin without variance (caught by `preregister` refusing
   non-numeric criteria).
3. In-sample / on-development reading (caught by pre-registration).

Fourth named failure mode the **arbiter pattern handles** rather than
preventing: pre-registered criterion fires on legitimate-but-
unanticipated data structure. Three instances so far (eigenmodes-v1
O2 near-tie, P1 tautology, Q1 MLE-ν saturation). The arbiter's
mechanical-verdict-plus-substantive-content shape is the workflow's
correct response; no engineering fix attempted.

See `memory/mitacs-session-lessons-corpus` for the full mapping
of threat-model items to workflow mechanisms.

## Result provenance (Tier-1, implemented 2026-05-18)

Every claim-grade / method-grade result artifact MUST be written through
`experiment.provenance.make_result()` — never hand-written. It stamps a header with
two distinct, independently-verifiable hashes: `inputs_fingerprint`
(git SHA + frozen_spec hash + lib versions + RNG seeds + declared inputs =
"rerun these, get this") and `body_sha256` (tamper-evidence; `load_verified_result()`
recomputes it on read). Policy is **strict refuse-on-dirty** (same bar as `freeze.py`;
a recorded git SHA that can't reproduce the artifact is the hole this closes) and a
mandatory `Grade` banner (CLAIM-GRADE / METHOD/DESIGN / INSPECTION-ONLY). Predictor-
derived CLAIM results pass `frozen_spec_required=True` so a missing/tampered spec
raises rather than stamping `null`. Shared primitives live in `experiment/_prov_core.py`
(ONE definition; `freeze.py` consumes it — do not copy-paste git/hash helpers). The
emitters are `experiment/backtest.py --emit-result` (C1), `processing/innovations/
validation/synthetic.py --emit-result` (C4), `experiment/emit_theory_correspondence.py`
(C3), `experiment/emit_ieso_infeasibility.py` (C2). The requirements analysis driving
this is `experiment/results/PROVENANCE_REQUIREMENTS.md` (memory
`mitacs-provenance-requirements`); `freeze.py`/the registered experiment is the C5
gold-standard template the rest was raised to. Inspection-only scratch carries a
greppable `PROVENANCE-GRADE: INSPECTION-ONLY` docstring banner — never cite it as a
result.

## Status (as of 2026-05-27)

### Pipeline and infrastructure
- ✅ Config-driven pipeline (`config/`, `run_pipeline.py` with stage registry,
  skip-if-exists, `--from/--to/--only/--daytype/--profile/--dry-run/--force`; `fast`/
  `full` profiles in `pipeline.yaml`). Stages 2–6 config-driven.
- ✅ Estimator rebuilt and **both validation gates pass** (recovery + non-Gaussianity).
- ✅ **Authoritative empirical facts** established via production-path re-baseline
  (`mitacs-rebaseline-facts` — trust this file for Ontario facts; all earlier Ontario
  claims are superseded). Intra-day conditioning required (full-process Σ singular);
  scalar one-step innovation strongly non-Gaussian (excess kurtosis ≈24–33).
- ✅ Applied-audit workflow operational (see "Applied-audit workflow" section
  above). Thirteen registered entries (three threads, ten experiments;
  Q1A SETTLED 2026-05-27, Q1A' AWAITING-B). Run
  `python -m experiment.audit.registry list` for current state.

### Two writeups under active development (do not conflate)
- `writeup/tex/draft.tex` → `draft_body.tex` — the **Goal-4 application paper**.
  Self-contained ~10pp PDF. State as of 2026-05-21: 38% cut for professional
  polish; pre-registration language scoped to §4.5 forward experiment only.
  See `mitacs-writeup-state`.
- `writeup/tex/missing_content_memo.tex` → `missing_content_memo.pdf` — the
  **kernel-family-agnostic theory memo**, 18pp, builds clean. Master citation
  of (S1)–(S10) settled findings. Added 2026-05-26 from the seed
  `notes/seeds/multiscale_factor_coherence.md`. NOT a publication target;
  internal memo that documents the framework + Q1/Q2A/Q2B results.

### Settled findings since 2026-05-21 (chronological)
The applied-audit workflow has produced nine settled / arbitered findings.
Authoritative source is the memo's (S1)–(S10) numbering + the per-experiment
memory files; (S11) and (S12) are registered and memo'd but not yet folded into the
kernel-family-agnostic theory memo (the memo predates them).

- (S1, S6) **v2 multiscale-direct-h-step SETTLED**:
  D_RATIO = 28.96 [26.82, 31.43] at headline cell; strict-semigroup
  picture (A) empirically inadequate; picture (C) corroborated under
  the Gaussian kernel. Note (S8c) below qualifies the magnitude.
- (S2, S3) **eigenmodes-v1 SETTLED MIXED**: defect diffuse across
  C_1's eigenmodes (not single-mode); weekday/weekend asymmetry is
  structural, not library-noise.
- (S4, S5) **PIT-calendar-fingerprint SETTLED AMBIGUOUS**: marginal PIT
  pathologically non-uniform (χ² = 22,329); secular drift is strongest
  legitimate calendar signal in PIT residuals.
- (S8) **Q1 distributional-class SETTLED** (R-D mechanical / R-C
  substantive): Student-t kernel reduces marginal χ² by 16× (to 1,371)
  but does not fully calibrate; D_RATIO_t = 2.93 [2.62, 3.24] — a
  **10× shrinkage from v2's 28.96** — so ~90% of v2's headline magnitude
  was distributional-class artifact; v2 SETTLED finding retained with
  asterisk on magnitude. Heavy-tailedness lives in the κ_1 innovation,
  not the seasonal climatology (M1 outperforms M2). ν per day-type:
  4.19/4.62/4.61 — kurtosis → ν mapping holds out-of-sample.
- (S9) **Q2A richer-family SETTLED R-B2** (mechanical = substantive):
  Three richer-than-Student-t families (mixture-2/3-Gaussian, KDE)
  tested. effective_χ² = 953.84 [685, 1357] in R-B2 band (228, 2284].
  Three substantive findings: (a) M3 = M4 = M5 chi²-indistinguishable
  (spread 13; KDE *worst* by 0.8) → distributional flexibility is
  empirically bounded; (b) Saturday 4× weekday chi², family-invariant
  → corroborates eigenmodes-v1 (S2/S3); (c) reduction curve asymptotes
  (M0→M1 = 16×, M1→Q2A = 1.44×) → remaining ~954 χ² is
  **non-distributional**. First arbiter in the workflow's history
  without mechanical/substantive gap; gate-validated cuts absorbed
  the realization cleanly. Thread amendment required (R-B2 = mandatory
  amendment per phase_a, NOT auto-advance); arbiter recommends
  reopening Q2B (non-distributional investigation).
- (S10) **Q2B non-distributional decomposition SETTLED R-C3**
  (mechanical = substantive; substantive STRONGER than mechanical):
  Three-mechanism ablation decomposition of Q2A's χ² = 954 baseline.
  Shares: mean_bias +0.109 [-0.39, +0.74]; variance_div -0.114
  [-0.22, -0.01]; seam -0.139 [-0.17, -0.09]. All three below the
  15% null floor → Q2B candidates empirically inadequate. **Two of
  three sign-determined NEGATIVE at 95% CI** — ablations actively
  WORSENED χ², which means mis-specification at the construction
  level (mech_2's one-sided variance cap and mech_3's seam-domain
  exclusion fail because the ensemble UNDER-disperses everywhere
  and production iteration already handles the seam correctly).
  Second arbiter without mechanical/substantive gap. **Framework-level
  interpretation:** S9 + S10 are joint applied evidence for the
  identifiability obstruction parked the same day at
  `~/Research/Mathematics/Resolvent_Framework/notes/unsorted/disintegration_diagnostic.md`
  (structured noise vs noisy structure are observationally
  indistinguishable from κ_Q alone; Bergna et al. 2026 Prop 1,
  Heckman-Singer 1984, Allahverdyan 2020). Resolution requires
  auxiliary information: candidate Z testing (weather, hour-of-week,
  secular drift), stratified Patra-Sen (genuinely novel applied
  contribution per the framework's Phase-2 audit), or synthetic
  Gaussian-DGP calibration. Thread should be declared EXHAUSTED;
  new thread targets the three framework-prescribed resolution paths.
- (S11) **P1 Patra-Sen per-stratum localization SETTLED R-A**
  (PROPONENT-CONFIRMED on outcome category; within-bucket
  forecasts falsified — fourth named-failure-mode instance,
  orthogonal shape): First applied response to the joint S9+S10
  identifiability obstruction. Per-stratum α̂_L (Patra-Sen 2016
  Theorem 1, F_b = Uniform on u_PIT per pre-run methodology
  carve-out) over 5 axes × strata on Q1's M1 PIT residuals.
  **max_axis_range = 0.5750 [0.4475, 0.6475]** at landslide R-A
  (lower CI bound clears the R-A cut at 0.20 by more than 2×).
  Per-axis range (ordered): hour_of_week 0.575 [0.448, 0.648]
  (**binding**); time_of_day 0.485 [0.385, 0.578]; day_type 0.345
  [0.250, 0.430]; season 0.135 [0.055, 0.240]; demand_quantile
  0.075 [0.023, 0.173]. **The missing-Z structure is
  cyclical/social-weekly, NOT seasonal/climatological** — hot
  zone bins 5/6/7 span Fri 10:00 – Sun 23:00 (α̂_L 0.55–0.61);
  cold zone bin 0 = Mon 01:00–21:00 at α̂_L 0.035 (essentially
  Gaussian). day_type ordering saturday 0.6475 > sunday 0.485 >
  weekday 0.3025 independently corroborates Q2A Finding 2
  (saturday 4× weekday) and eigenmodes-v1 (S2/S3) asymmetry from
  a distinct estimator. Proponent's temperature-via-season prior
  translation is 95%-CI falsified (season range CI upper bound
  0.240 barely clears R-A, not the predicted 0.30); DA's R-C
  pure-orthogonality position decisively refuted on 4 of 5 axis
  CIs. day_type range 0.345 (> 0.15) triggers proponent's
  `what_would_change_my_mind` condition #4 — Q1's per-day-type ν
  + per-(month, hour) climatology does not absorb day-type
  α̂_L asymmetry (non-blocking flag; Q1A's hour_of_week Z
  subsumes day_type at finer resolution). **Routing:** R-A
  routes to Q1A with binding axis = hour_of_week as Q1A's
  conditioning variable; Q1C (temperature fallback) closed as
  sibling-not-fired (temperature prior survives as candidate for
  marginal pathology but is no longer the prior-supported
  localizing axis).
- (S12) **Q1A Z-conditioning head-to-head SETTLED R-B1A**
  (DA-CONFIRMED on outcome category with exact point forecast;
  fifth named-failure-mode instance, new shape variant): First
  applied chi^2-side test of P1's binding-axis result on the
  M3 residual law. Production-path `mixture_2_gaussian_mle_fit`
  per (day_type, Z) cell, head-to-head Z_b (binary weekend
  indicator, hour_of_week bin ≥ 5) vs Z_c (8-level categorical
  bin 0..7). Reproduction-check EXACT (Q1A no-Z chi^2 = 953.84
  reproduces Q2A settled 953.84; |diff| = 0.0). Inline synthetic
  gate PASS (known Z-conditional DGP recovery + per-cell test
  chi^2 well below ceiling; false-Z companion collapses to
  marginal). **chi2_Z_b = 986.0 [678, 1382]; chi2_Z_c = 952.5
  [627, 1374]; better_chi2 = 952.5** in (228, 2284] → R-B1A.
  **chi2_diff = +33.5 [-93, +143]** with paired SE = 56.84 →
  TRUE NEAR-TIE (|diff| < 1σ). Z_c wins at point estimate but
  no statistical ranking. **Z conditioning shaved only 1.33
  chi^2 off Q2A's 953.84 baseline — 0.18% of the 725.84 gap
  to R-A1A.** The substantive finding is stronger than the
  R-B1A category: hour_of_week σ-algebra enrichment delivers
  essentially NO marginal reduction beyond bootstrap noise.
  **Per-cell decomposition exactly corroborates P1**:
  saturday__bin6 contributes chi^2 = 1634 (single cell exceeds
  pooled marginal 952.5 by 70%); P1 had reported α̂_L = 0.61
  in this stratum. **The contamination IS where P1 said it
  would be — but the mixture-2-Gaussian family is at capacity
  within strata.** DA's Z_c point forecast (950 vs realized
  952.52, 0.3% residual) is the strongest pre-registered point
  forecast in the workflow's history; proponent's R-A1A (200
  [130, 320]) decisively falsified (4.8× overshoot of realized
  lower CI 625). **Joint S9+S10+S11+S12** is now applied
  chi^2-side corroboration of the identifiability obstruction —
  three within-data levers (richer distributional families,
  mechanism ablation, hour_of_week stratification) all
  empirically bounded. Resolution requires **auxiliary
  information beyond the time index** (temperature / exogenous
  Z). **Routing:** R-B1A routes to Q1B (time_of_day) with Z_c
  carryforward (near-tie caveat); S3 state-history carries an
  advisory pre-flag that EXHAUSTED-UNRESOLVED is a plausible Q1B
  trajectory given joint S9+S10+S11+S12, but Q1B's verdict is
  not prejudged — its phase_a must register cuts and forecasts
  on Q1B's own merits.
- (S13) **Q1B Z2 conditioning SETTLED R-B1B substantive
  FAVORABLE** (2026-05-27; 6th named-failure-mode instance,
  mixed-scoring shape): First applied chi^2-side test of JOINT
  multi-axis Z conditioning on M3. Production-path
  `mixture_2_gaussian_mle_fit` per (day_type, Z_c, Z2_4level) cell
  with Z_c = Q1A's settled hour_of_week carryforward (NEAR-TIE
  caveat preserved) and Z2 = head-to-head between Z2_b (binary
  overnight) and Z2_c (4-level per P1 production `_time_of_day_label`
  binning). Reproduction-checks BOTH EXACT (Q2A 953.84 → |diff|=0.003;
  Q1A 952.5167 → |diff|=0.0000 — strongest dual-anchor in workflow
  history). Synthetic gate + false-Z2 collapse PASS. **chi2_Z2_b
  = 696.0 [492, 1009]; chi2_Z2_c = 662.2 [480, 945]; better_chi2
  = 662.2** in (228, 2284] → R-B1B; winner Z2_c (NEAR-TIE: chi2_diff
  +33.8 [-53, +124] paired SE 44.71 → |diff|<1σ). **Cumulative
  improvement above Z_c carryforward = +290.3 [+8.06, +484.4]** =
  40% of remaining gap AT POINT, but **CI spans 60×**: lower bound
  +8.06 is "barely above Q1A's 1.33"; upper bound +484.4 is "67%
  closure within data." Framework reading SHARPENED, not refuted:
  single-axis bounded near zero (Q1A 0.18%); joint multi-axis
  bounded above zero with wide CI; full closure to R-A1B doesn't
  fire even at upper CI. Per-cell decomposition exactly corroborates
  P1: saturday__zc6 (P1 α̂_L=0.61 stratum, Q1A chi²=1634) splits
  across Z2 as morning_ramp 967 > evening 400 > afternoon 328.
  Methodologically: overnight z20 cells contribute substantially
  per-cell despite P1 axis-level α̂_L=0.11 → **axis-level α̂_L does
  NOT predict cell-level chi^2 contribution under joint
  stratification.** Proponent (R-A1B, 180 [90, 450]) decisively
  falsified; DA (R-B1B, 948 [620, 1340]) confirmed on outcome
  category but realized 662 falls in the gap between proponent's
  CI upper 450 and DA's CI lower 620 (mixed scoring: outcome+shape
  DA, direction proponent, magnitude neither). Thread paused at
  current_node=null (no auto-advance under branching_rules.Q1B.R-B1B
  = null); S3 EXHAUSTED-UNRESOLVED pre-flag DOWNGRADED per
  premise-refuted-by-+290-realization; three-way amendment
  decision space surfaced for user adjudication: (a) third within-data
  Z axis, (b) cross-axis joint thread, (c) temperature/exogenous-Z
  arm.
- (S14) **Q1B' SMC tightening SETTLED R-B1B' INSPECTION-ONLY**
  (2026-05-29; 7th named-failure-mode instance, new shape variant
  where BOTH forecasters directionally falsified on width central
  tendency): Non-gating sibling under thread amendment A2,
  re-runs Q1B's winning (day_type, Z_c, Z2_c) configuration at
  N_PARTICLES=400 (vs Q1B's 200) to bracket whether Q1B's wide CI
  is methodological (SMC sampling noise) or substantive (eval-window
  noise on 12,000-row post-cutoff window + paired-day bootstrap).
  Pre-registered verdict on `cumulative_chi2_improvement_above_Z_c`
  CI95 WIDTH with cuts at 357.27 (≤75% of Q1B's realized 476.358,
  implying V_smc/V_total ≥ 0.875) and 524.0 (>110% triggers R-C1B').
  Plus point-shift sub-criterion |better_chi2 - 662.223| ≤ 50.
  ALL BLOCKING GATES PASS (code-path PRODUCTION-PATH; Check P PASS;
  Check T CONSISTENT; Check R 5/5; reproduction checks within tol).
  **CI WIDENED, did NOT narrow**: width 506.198 vs Q1B's 476.358
  (+6.3%). Point-shift 16.905 (within tol). Mechanically: R-B1B'
  (width in (357.27, 524.0]; 89.3% through band, 17.80 chi²
  headroom below R-C1B'). **Substantive reading sharpens
  framework picture: eval-window noise dominates Q1B's wide CI;
  within-data tightening empirically impossible.** Proponent's OWN
  pre-registered falsification condition #1 (`width > 400 →
  strong-form SMC-dominance refuted`) TRIGGERED. **Reproduction-
  check side-channel**: no-Z drift |diff|=5.2 vs Z_c-stratified
  drift |diff|=44.6 (8.5× ratio) — SMC variance non-trivial at
  per-stratum layer but NOT at no-Z pooled layer; corroborates
  DA's modest-SMC-variance reading, refutes proponent's strong-form
  SMC-dominance. **Joint S9+S10+S11+S12+S13+S14 = THREE INDEPENDENT
  LAYERS of identifiability-obstruction evidence**: distributional-
  class layer (S9+S10 bounded), σ-algebra-enrichment layer
  (S11+S12+S13 bounded with wide CI), methodological-tightening
  layer (S14 SMC particle count cannot narrow). **Resolution
  requires auxiliary information beyond the time index — now a
  publication-grade conclusion.** GRADE: PROVENANCE-GRADE
  INSPECTION-ONLY per .txt-artifact gap (script's hardcoded `.txt`
  output path collided with Q1B's CLAIM-GRADE artifact; Check P
  enumeration missed this string; .txt restored from HEAD; user
  adjudicated demote rather than re-emit). Downstream citation
  must read "inspection-only diagnostic." Thread paused at
  current_node=null (no auto-advance; Q1B' non-gating per A2).
  (a)/(b)/(c) adjudication PRIORITY UPGRADE: (c) temperature/
  exogenous-Z arm is the only direction not empirically refuted
  within the registered thread.

### Lines of inquiry
- `2026-05-26_missing-content-thread`: **exhausted** under registered
  branching (P1 settled AMBIGUOUS → null per branching_rules; continuing
  requires amendment).
- `2026-05-26_distributional-class-thread`: **exhausted** under
  registered branching. Q1 SETTLED (S8) → A1 closed Q2D / rerouted
  to Q2A → Q2A SETTLED R-B2 (S9) → A2 reopened Q2B → Q2B SETTLED
  R-C3 (S10) → thread-coordinator closed the thread per Q2B
  arbiter recommendation; scope ("predictive-distribution class
  axis only") is empirically bounded in both directions.
- `2026-05-27_resolution-paths-thread`: **active, current_node =
  null (paused awaiting user adjudication)**. Successor to the
  distributional-class thread, targets the three framework-prescribed
  resolution paths for the joint S9+S10 identifiability obstruction
  (candidate Z testing, stratified Patra-Sen, synthetic Gaussian-DGP
  calibration). State (S1→S5; current thread body_sha256 164f8bb9...):
  P1 SETTLED R-A (S11; landslide on `hour_of_week`) → Q1A SETTLED
  R-B1A (S12; 0.18% of gap, NEAR-TIE) → Q1B SETTLED R-B1B substantive
  FAVORABLE (S13; cumulative +290 [+8, +484] = 40% AT POINT, CI 60×)
  → amendment A2 added Q1B' as non-gating tightening sibling
  (2026-05-29) → Q1B' SETTLED R-B1B' INSPECTION-ONLY (S14; CI WIDENED
  6.3% under SMC particle doubling; within-data tightening
  empirically impossible) → amendment A3 closed Q1A' as
  EXHAUSTED-PRE-DATA without running (2026-05-29; FastICA
  selection-rule design defect; see "Q1A' CLOSED" below and
  [[q1a-prime-fastica-selection-defect]]) → amendment A4 grafts
  the **temperature-arm sub-tree** (2026-05-29; new planned
  nodes T1, T2, T3 — extends candidate-Z testing to exogenous
  Z; T1 phase_a fill blocked on ECCC hourly weather data
  acquisition next session; current_node still null;
  state-machine advancement is data-acquisition-gated, not
  amendment-gated). Settled nodes: P1, Q1A, Q1B, Q1B'.
  Exhausted (pre-data, non-gating): Q1A'. Closed: Q1C
  (S2 sibling-not-fired). Planned: T1 (root of temperature arm
  per A4), T2 (Q1B-analogue joint refinement; contingent T1
  R-B1T), T3 (Q1B'-analogue methodological tightening;
  optional non-gating sibling), P2 (synthetic Gaussian-DGP
  calibration; reachable via T1.R-C1T handoff or future
  amendment).

  **Joint S9+S10+S11+S12+S13+S14 = three independent layers of
  identifiability-obstruction evidence**: distributional-class
  (S9+S10), σ-algebra-enrichment (S11+S12+S13), methodological-
  tightening (S14). **Resolution requires auxiliary information
  beyond the time index — publication-grade conclusion.** Thread
  paused; three-way amendment decision in front of user, with
  S14-weighted priors:

  - (a) Third within-data Z axis (season range 0.135 / demand_quantile
    range 0.075 — both weaker than Z2's 0.485). **DIMINISHING RETURNS
    prior STRENGTHENED by S14** (within-data lever space bounded
    methodologically as well as substantively).
  - (b) Cross-axis joint stratification thread. Still possible but
    Q1B's wide CI now demonstrated intrinsic; cross-axis thread
    would inherit the same eval-window noise floor unless it also
    addresses sample-size-per-cell.
  - (c) **Temperature / exogenous-Z arm. PRIORITY UPGRADED by S14**:
    auxiliary information beyond time index is the only direction
    that has NOT been empirically refuted within the registered
    thread. Pre-requisite is ECCC hourly weather acquisition (still
    on background list per `mitacs-clim-gap-nonstationary`).

### Frozen forward experiment (separate from the above)
- ✅ **IESO retrospective head-to-head shown infeasible** from public archives.
- ✅ Registered *forward* prediction experiment (`experiment/freeze.py` frozen spec)
  is the design built in response. Accrues with calendar time.

### Archived / superseded
- 🗄️ `processing/locality/` (old θ-sweep, broken on the WLS API) removed; archived at
  `archive/legacy_locality_sweep.zip` (gitignored; recoverable from git history).
- `scratch/` and `processing/innovations/scratch.py` are exploratory; not pipeline.
  `scratch/postcovid_*.py` are inspection-only probes on the IEEE DataPort Post-COVID
  competition dataset (its CSVs are gitignored — see `scratch/data/postcovid/README.md`).
- `processing/innovations/validation/bandwidth_comparison.py` is a superseded
  investigation record (its conclusion is implemented); kept runnable for trace.

### Open
- Web dashboard.
- Forward-experiment results accrue with calendar time.
- `resolution-paths-thread` active on temperature arm
  (amendment A4 grafted T1/T2/T3 sub-tree 2026-05-29 per user
  Position-C pivot decision). current_node=null until T1
  phase_a is filled — gated on **ECCC hourly weather data
  acquisition** (the next-session infrastructure work). T1
  root spec: M3 with σ = (day_type, Z_c, temperature_decile or
  equivalent), cumulative baseline = Q1A's settled 952.5167.
  Verdict cuts (228, 2284] inherited from thread skeleton or
  re-derived per T1 phase_a author. T1.R-A1T resolves the
  thread on path (i) extended; T1.R-B1T routes to T2 (Q1B-
  analogue joint refinement); T1.R-C1T handoff to NPMLEmix /
  P2 synthetic-DGP / alternative-Z. T3 is the Q1B'-analogue
  CI-tightening run, pre-registered only if T1/T2 CIs are wide.
  **T1 phase_a author note**: read
  `scratch/within_stratum_residual_diagnostic_summary.md`
  (INSPECTION-ONLY scratch, 2026-05-29) before filling. The
  diagnostic characterised the within-stratum residual
  structure as a slow-varying-continuous-factor signature
  (mean lag-1 ACF across all 32 cells = +0.67, weak lag-5)
  consistent with temperature's footprint. Per user direction
  2026-05-29, T1 should register a **secondary metric**
  alongside the χ² primary: post-conditioning within-cell
  lag-1 ACF (predicted to drop from +0.67 toward 0 if
  temperature is the missing Z; stays > +0.4 under DA's
  null reading). The diagnostic itself is INSPECTION-ONLY and
  not citable as a result; its content becomes
  registered-finding-eligible via T1's secondary metric.
  ECCC scope from the diagnostic: hourly resolution required
  (not daily-mean); single-station sufficient for first cut;
  forecast temperature relevant for the registered forward
  experiment.
- **Goal-4 writeup integration LANDED** (2026-05-29): per user
  "Position 2" decision (write up the within-data exhaustion
  story before opening the temperature arm), then reframed under
  user's "three-act directed-search narrative" framing. Final
  draft is 23 pages with three-act narrative (Act 1: what
  univariate data gives you under the kernel framing; Act 2:
  directed search through the residual lever space; Act 3:
  framework-prescribed exogenous-Z follow-on). Each act carries
  its own figure complement (six new figures added across two
  visual-storytelling-pass commits, including the §7 Act-2
  forest plot summary). Two `\todo` blocks remain for the
  author group: prior-work context (§1) and forward-experiment
  results (§7, accrues with calendar time). Companion memo
  (`writeup/tex/missing_content_memo.tex`) at 34pp still uses the
  bounded-finding framing in §9-§12 (the draft has since been
  reframed) — that's a defer-to-next-pass debt.
- **Q1A' CLOSED via thread amendment A3** (2026-05-29):
  EXHAUSTED-PRE-DATA. The FastICA selection-rule defect IS the
  substantive deliverable; see
  [[q1a-prime-fastica-selection-defect]] memory for the
  generalisable lesson. Empirical exploration showed FastICA's
  pre-registered max-|excess_kurtosis| selection picks the
  structurally-non-Gaussian binary day-type indicators
  (saturday, sunday) over the continuous cyclic signal the
  cyclic-ρ verdict statistic expects — bimodal rho_signal
  {~0.65, ~0.01}, rho_null at X-cyclic-loading floor 0.55-0.71;
  pre-registered 0.70/0.30 cuts unsatisfiable. SIR side science
  (consistency-check on P1's hand-specified hour_of_week axis
  via independent SDR method) is preserved as a deferred follow-
  up — would require a new pre-registration thread, not a Q1A'
  amendment restart. Per workflow rule, close-via-amendment was
  chosen over second-amendment-to-same-phase_a-in-one-session.
  Q1A' was non-gating from A1, so A3 doesn't affect
  current_node (still null) or thread state (still active).
  Workflow note: registry per-experiment list shows Q1A' as
  AWAITING-B because the per-artifact view doesn't read
  thread.yaml; thread-level state lives in
  `nodes."Q1A'".status=exhausted`. Workflow-design retro item:
  should `registry list` read thread.yaml for thread-amended
  closures? Open question.
- The Goal-4 paper's abstract (written last). With S11 + S12
  in hand, the resolution-paths arc has settled-to-negative
  on the most-promising within-data Z lever; the writeup now
  carries an honest "auxiliary information beyond time index is
  required" reading as a publication-grade conclusion.
- The kernel-family-agnostic theory memo's `/audit pure` audit in
  the theory-side `Resolvent_Framework` programme (not yet run;
  the memo is currently lab-only). The memo also predates (S11)
  and (S12) — if the audit triggers a refresh, fold S11 + S12
  in then rather than mid-stream now.

## Data files

CSV and `.pkl` demand files under `data/` are tracked (large; LFS was removed in `db8ff5b`).
`.gitignore` excludes Python bytecode/caches (`__pycache__/`, `*.pyc`, `.pytest_cache/`) and a
couple of legacy `src/main/resources/...` climate paths no longer in this layout.
