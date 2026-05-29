# Next-session resume prompt — 2026-05-30 (or whenever the next session starts)

Paste this into the next session's first message, or have the user
paste it. Structure mirrors the resume prompt this session received
at start, which worked well for cold orientation.

---

Resume the MITACS dynamics project. Working dir:
  /Users/pmahon/Research/Dynamics/MITACS. Branch: operator.
  Working tree clean; in sync with origin (pushed at end of
  session 2026-05-29-b).

Orient yourself in this order before doing anything else:

1. Read `notes/lab/2026-05-29-b.md` in full. This was the
   long Q1B' end-to-end + writeup three-act reframe + figures +
   Q1A' close + A4 temperature sub-tree + registry housekeeping +
   within-stratum diagnostic session (26 commits; 3.5× the
   workflow's session-momentum-throttle threshold; structured-
   multi-stage with user adjudication at every stage transition).
   The frontmatter's `deferred_to_next_session` field is the
   authoritative entry-point list. The body's "Next session entry
   point" section narrates the same list with priority. Pay
   special attention to: (i) the within-stratum diagnostic result
   (lag-1 ACF +0.67 across ALL 32 cells; consistent with
   slow-varying-continuous-factor / temperature signature; lives
   at `scratch/within_stratum_residual_diagnostic_summary.md`);
   (ii) the discipline rule that the diagnostic stays
   INSPECTION-ONLY and gets registered via T1's secondary metric
   rather than promoted to its own claim-grade experiment.

2. Launch the lab notebook dashboard if useful for
   cross-referencing:
     .venv/bin/python -m post_processing.lab_notebook.app --port 8050
   Home tab surfaces the active thread (resolution-paths-thread,
   current_node=null/paused; thread.yaml body_sha256 c3be1f75...;
   4 amendments A1-A4; 10 nodes including the new T1/T2/T3
   temperature sub-tree planned via A4).

3. `.venv/bin/python -m experiment.audit.registry list` —
   current state:
   - 10 SETTLED experiments (multiscale-h-step-v2 S1/S6,
     eigenmodes S2/S3, pit-calendar S4/S5 AMBIGUOUS, Q1 S8,
     Q2A S9, Q2B S10, P1 S11, Q1A S12, Q1B S13, Q1B' S14)
   - 1 PROVISIONAL: multiscale-h-step v1 (superseded)
   - 2 EXHAUSTED threads: distributional-class, missing-content
   - 1 ACTIVE thread: resolution-paths-thread (current_node = null;
     paused-but-active-on-temperature-arm via A4; T1 phase_a fill
     blocked on ECCC hourly weather data acquisition)
   - 1 AWAITING-B-by-per-artifact-view-but-EXHAUSTED-by-thread-
     view: Q1A' (close-via-amendment A3; the per-artifact view
     doesn't read thread.yaml — see
     `notes/preregistrations/README.md` § "Registry-list view vs
     thread view (known limitation)" for the explanation. Thread
     view is source-of-truth for thread membership; per-artifact
     view is source-of-truth for artifact existence.)

What landed last session (commits 42ed0eb → a113053, 26 commits)

This was the second 2026-05-29 session ("-b"); the first
("2026-05-29.md") was Q1B SETTLED + amendment locking decision
(d) for the tightening run. -b session executed (d), then went
through Position-A/B/C strategic adjudication, then executed
Position-C end-to-end. Five distinct workstreams in commit
order:

1. **Q1B' end-to-end → SETTLED R-B1B' INSPECTION-ONLY (S14)**
   (commits 42ed0eb → 42cc130, 11 commits). Pre-registered
   tightening run at N_PARTICLES=400; both forecasters
   directionally falsified on width central tendency (proponent's
   own pre-registered condition #1 fired); CI widened +6.3%
   under SMC doubling, confirming eval-window noise dominates
   the wide CI rather than estimator sampling. Reproduction-
   check side-channel asymmetry 8.5× corroborates DA's modest-
   SMC-variance reading; refutes proponent's strong-form SMC-
   dominance. 7th named-failure-mode instance, new shape variant.
   Joint S9+S10+S11+S12+S13+S14 = THREE independent layers of
   identifiability-obstruction evidence (distributional-class,
   σ-algebra-enrichment, methodological-tightening). Within-data
   tightening empirically impossible.

   Workflow incidents this stage surfaced: (a) one cosmetic
   `>=` → `>` pre-stamp-chain amendment, captured in memory
   `q1b-prime-amendment-cost-asymmetry.md` (cosmetic carve-out
   cost-asymmetry — proposed workflow rule refinement; deferred);
   (b) script parameterization required 10-string substitution
   inline edit; (c) CLAIM-grade .txt artifact path collision
   demoted Q1B' result.yaml to INSPECTION-ONLY per user
   adjudication. CLAUDE.md (S14) + S13 entry + thread state
   block all updated.

2. **Writeup integration (Position 2) → three-act directed-
   search reframe → six-figure visual storytelling pass**
   (commits 50f70d4 → bdd503a, 7 commits). First the memo
   §9-§12 (S11-S14) update + draft §5.5 + Conclusion sharpen
   (Position-2 bounded-finding framing). Then under user direction
   "alright I think in general we just have a collection of
   results but I think the story might be something closer to,
   here's what we can do with univariate data alone..." → full
   reframe under three-act directed-search narrative under the
   kernel framing. Title changed to "What univariate kernel
   forecasting can extract from Ontario electricity demand, and
   where the rest lives". §6 = Act 1 (what univariate data gives
   you = Q1's 16× Student-t lift); §7 = Act 2 (directed search:
   Q2A → Q2B → P1 → Q1A+Q1B → Q1B'); §8 = Act 3 (auxiliary
   information beyond time index = registered forward experiment).
   Then visual storytelling pass (research-agent grid; figure-
   taxonomy diagnosis: 6 existing apparatus figures, 0 in
   substantive acts) added 6 figures across two commits (forest
   plot summary + 4 orphan-PDF integrations + per-cell heatmap).
   Final draft 23pp; companion memo at 34pp but still uses
   bounded-finding framing in §9-§12 (deferred-debt: memo
   refresh post-/audit-pure).

3. **Q1A' close-out via thread amendment A3** (commits 4c0e3dc
   → ddb2f51, 2 commits). Position-C step 1. Q1A' had been
   parked AWAITING-B for three sessions; FastICA selection-rule
   design defect surfaced in exploratory synthetic-gate-build.
   Per advisor: close-via-amendment rather than redesign or
   run-as-is. FastICA defect captured as generalisable memory
   `q1a-prime-fastica-selection-defect.md` (third within-repo
   instance of "textbook method fails on legitimate data
   structure" pattern after M2 MLE saturation S8a and SIR
   permuted-Y X-cyclic-loading floor). Thread amendment A3
   changes Q1A'.status active → exhausted; branching_rules
   unchanged (all-null per A1's non-gating); current_node still
   null. SIR-side science deferred as potential future SDR-only
   thread.

4. **Act-3 pivot via thread amendment A4 + registry housekeeping**
   (commits bb47a25 → 1de933f, 4 commits). Position-C step 2a.
   A4 grafts temperature-arm sub-tree onto resolution-paths-
   thread (continuation of same research question rather than
   sibling thread). Three new planned nodes: T1 (root; M3 with
   σ=(day_type, Z_c, temperature_decile or equivalent);
   cumulative baseline Q1A's settled 952.5167), T2 (Q1B-analogue
   joint refinement; contingent T1 R-B1T), T3 (Q1B'-analogue
   methodological tightening; optional non-gating sibling).
   Registry housekeeping (Position-C step 2c): Q1A'.phase_a_path
   field fill (A3 dispatch had missed it; one-line re-stamp
   documented as "completes A3"); README documentation of
   registry-list-vs-thread-view mismatch as known limitation
   (deferred fix per N=1 doesn't-warrant-infrastructure-work).

5. **Within-stratum residual diagnostic** (commits c0c1730 →
   a113053, 2 commits). User asked: before going to ECCC, can
   we construct/infer properties of the hypothetical exogenous
   Z from existing residual structure? Per advisor reframing:
   yes — within-stratum residual structure analysis on existing
   Q1 pit_M1 pickle (the same series P1 ran on); INSPECTION-ONLY
   scratch. Result: **mean lag-1 ACF +0.669 across ALL 32 cells
   (range +0.33 to +0.82; 32/32 clearing Bartlett 95% null
   bound)**. Lag-5 much weaker. Variance ratio 0.66
   (under-dispersed). SD per-day means ~0.21 (between-day
   systematic shape). Interpretation: missing factor is slow-
   varying continuous covariate with strong within-day continuity
   and weak week-on-week tie — temperature's signature. Rules
   out fast-varying factors, discrete-day-of-week-categorical
   factors, pure seasonal drift. Does NOT certify temperature
   (identifiability obstruction stands); characterises what
   *any* successful Z must look like under the latent-Z reading.
   Per user direction: stays INSPECTION-ONLY; lag-1 ACF target
   gets registered into T1's phase_a as pre-data secondary
   metric (proponent: ACF drops from +0.67 toward 0 if Z is
   temperature; DA: ACF stays > +0.4). CLAUDE.md forward-pointer
   from T1 open-item to diagnostic added so future-session
   T1-phase_a author sees the dependency.

First task: ECCC hourly weather data acquisition

Per the user's locked direction (Position-C plan, step 2b):
ECCC is the infrastructure pre-requisite for T1 phase_a fill.
Scope guidance per the within-stratum diagnostic:

- **Temporal resolution**: HOURLY (not daily-mean). The lag-1
  ACF +0.67 signature lives in within-day continuity; daily-mean
  data won't capture it; T1 would plausibly fail on the χ² side
  even if temperature is the right Z at the daily level.

- **Station selection**: SINGLE-STATION sufficient for first cut.
  Within-day continuity is preserved at any station in the IESO
  Ontario zone. Multi-station averaging is a secondary concern
  (consider for v2; or as a robustness sub-finding within T1).

- **Time coverage**: pre-cutoff library window (2003-01-01 to
  2024-12-31) + post-cutoff eval window (2025-01-01 to
  2026-05-16). Same temporal scope as Q1, Q2A, Q1A, Q1B used.

- **Forecast vs observed**: BOTH eventually relevant. Observed
  temperature for T1's pre-registered test. **Forecast** (issued
  ahead of delivery hour) for the registered forward experiment
  (`experiment/freeze.py`-style live prediction). Note: T1 itself
  uses OBSERVED temperature at delivery hour — the test is
  whether observed temperature, treated as if available to the
  predictor, would close the χ² gap. The forecast-temperature
  question is downstream.

Suggested approach:

   a. **Scout ECCC data products first** (~30 min). Environment
      and Climate Change Canada has several historical-weather
      data products: (i) hourly weather station observations
      via their Historical Climate Data portal
      (https://climate.weather.gc.ca/historical_data/), (ii)
      bulk download via their FTP, (iii) more recent
      programmatic access via the Adjusted and Homogenized
      Canadian Climate Data (AHCCD) or the ECCC API. Pick the
      most-reliable historical hourly source covering 2003-2026
      for an Ontario IESO-zone station. Toronto Pearson
      International Airport (CYYZ) is the canonical choice but
      check Hamilton or London too if their coverage is better.

   b. **Build a loader** at `data/ontario/weather/` (mirrors
      the existing `data/ontario/forecast/` structure). Bulk-
      download or scrape the historical hourly temperature
      series; cache; parse into pandas DataFrame with
      `delivery_day` + `h` indexing aligned to IESO's HE1-HE24
      delivery-clock convention.

   c. **Sanity checks** before declaring the data ready:
      coverage completeness over 2003-2026; any missing-hour
      patterns; outlier values; cross-check against a single
      day's expected diurnal pattern.

   d. **Sanity-check the within-stratum diagnostic** before
      letting it inform T1's phase_a forecast: re-run
      `scratch/within_stratum_residual_diagnostic.py` on a
      known-Gaussian-process synthetic. If lag-1 ACF ≈ 0 on
      synthetic, Ontario signature is real. If +0.67 on
      synthetic too, the script has a bug. Cheap (~15 min); do
      this in parallel with ECCC scouting.

   e. Then dispatch the preregister agent to fill T1's phase_a
      per the spec in `nodes."T1".phase_a_skeleton` of
      `notes/preregistrations/2026-05-27_resolution-paths-thread/
      thread.yaml`, with the secondary metric on within-cell
      lag-1 ACF per CLAUDE.md's T1 forward-pointer.

If the user opens with "what's next"

Confirm the order above before starting: scout ECCC, then build
loader, then sanity-check diagnostic, then T1 phase_a. The data-
acquisition work is genuinely uncertain in scope (depends on what
ECCC's data-access mechanisms look like in practice) so the user
may want to scope it before committing.

If the user opens with a different direction

Follow it, but read in this order: `notes/lab/2026-05-29-b.md`
(the full session retro), `scratch/within_stratum_residual_
diagnostic_summary.md` (the substantive content the diagnostic
delivered), the new §6/§7/§8 of `writeup/tex/draft_body.tex` (the
three-act narrative; commit 4a94a67 + figure-pass commits e5bc369
and bdd503a), and the thread.yaml state (Q1A' exhausted; A4
added T1/T2/T3 planned).

Open follow-ups (not blocking next session)

- **Companion memo refresh**: `writeup/tex/missing_content_memo.tex`
  §9-§12 still uses bounded-finding framing; draft has since been
  reframed under directed-search. Co-author reviewer might flag.
  Cost: ~1 hour rewrite. Defer until `/audit pure` on the
  Resolvent_Framework side triggers it or co-author flags.

- **Goal-4 paper `\todo` blocks**: §1 prior-work context;
  §7 forward-experiment results. Both for author group; don't
  block next session's experimental work.

- **Workflow-infrastructure-scheduling**: three items deferred
  per `workflow-infrastructure-scheduling-pattern.md` memory
  (cosmetic-amendment carve-out refinement; --prereg-dir CLI
  refactor; registry status enum extension). Trigger for actual
  scheduling: deferred-count reaches 5+ OR specific instance
  trigger (e.g., second close-via-amendment case for the status
  enum). None of the three would block T1 work.

- **Within-stratum diagnostic as standalone methodological
  paper**: the analysis is generalisable beyond Ontario; could
  be a small methods paper independent of the Goal-4 application
  paper. Flagged for future consideration; out of scope for
  next session.

Workflow rules that have caught retractions (do not bypass)

- /audit code-path BEFORE running; /audit phase-fidelity Check P
  BEFORE running; Check R BEFORE arbiter renders.
- Proponent + DA forecasts stamped simultaneously, both numeric,
  both locked before data is touched. Simultaneity is NON-
  OBSERVATION, not stamp ordering.
- Pre-run methodology amendment carve-out: applied 5× now (P1
  F_b switch; Q1A' SIR cut; Q1B P1-binning correction; Q1B'
  cosmetic `>=` → `>` (cost-asymmetry case); Q1A'.phase_a_path
  field fill on thread.yaml ("completes A3"). All legitimate,
  all pre-data. A SECOND amendment to the same phase_a in one
  session is the threshold for "rewriting as I build it" —
  surface to user instead.
- Arbiter writes note.md as part of dispatch.
- Thread-coordinator rewrites note.md at every procedure.
- Cannot change branching_rules[node_id] for nodes that have
  settled (their branches already fired).
- The discipline rule is SYMMETRIC: pre-judging a negative
  result is the same workflow risk as pre-judging a positive
  one.
- CI-prominent framing for any result where CI spans > 10× the
  point estimate or CI lower bound is near a prior-relevant
  baseline. CLAUDE.md "Settled findings" entries that anchor
  framework reading should wait for CI tightening or explicit
  user direction.
- INSPECTION-ONLY scratch carries a greppable
  `PROVENANCE-GRADE: INSPECTION-ONLY` banner — never cite it as
  a result. Diagnostics like
  `scratch/within_stratum_residual_diagnostic_summary.md` inform
  registered-experiment design (proponent-side context, ECCC
  scope guidance); they do not stand as findings.
- Amend thread.yaml directly (thread-coordinator is sole writer).
- Skip the inline synthetic gate before any Ontario read on any
  new experiment script.
- Touch settled-node entries in thread.yaml (P1, Q1A, Q1B, Q1B',
  Q1C; now also Q1A' which is exhausted).
- Restructure the memory corpus.
- Ship CLAUDE.md (S15) entry without CI tightening or explicit
  user direction (if T1 produces a wide-CI substantive finding,
  same rule applies).

Lab note workflow

- Session start: `.venv/bin/python notes/lab/tools/lab_note.py
  --new YYYY-MM-DD` (or `YYYY-MM-DD-b` for multi-session days).
  Scaffolds frontmatter from git + registry state.
- During session: prose in body as work happens.
- Session end: `lab_note.py --refresh notes/lab/<file>.md` to
  refresh machine-derivable fields. Judgment fields stay manual.

Session-volume note from 2026-05-29-b

The prior session ran 26 commits — 3.5× the workflow's session-
momentum-throttle threshold from `applied_audit_workflow.md`
§6.1 (>10 substantive commits). Discipline was preserved by
multi-stage user adjudication at every transition, but the
session also surfaced 3 workflow-infrastructure-extension items
that all got deferred. If next session approaches 10-15 commits,
that's a healthy ceiling; if it heads past 20, pause to ask
whether the session is structured-multi-stage with explicit user
input (legitimate) or single-direction momentum-drift (the
throttle's actual target).
