# Next-session resume prompt — 2026-05-31 (or whenever the next session starts)

Paste this into the next session's first message, or have the user
paste it. Structure mirrors the resume prompt this session received
at start, with one important update: the ECCC scout that was
originally first task in the 2026-05-30 resume prompt was *not*
done in the 2026-05-30 session — the user reframed the question
at session open to a framework-side comprehension question that
took priority. ECCC is now the first task for the next session
(again), with one significant change: T1 now has a framework-side
stake on top of its application-side primary purpose.

---

Resume the MITACS dynamics project. Working dir:
  /Users/pmahon/Research/Dynamics/MITACS. Branch: operator.
  Working tree clean (pushed at end of session 2026-05-30-a).

Orient yourself in this order before doing anything else:

1. Read `notes/lab/2026-05-30.md` in full. This was the
   framework-side cross-session coordination session: user
   reframed the question from ECCC scouting (the original 2026-05-30
   first task) to "what is the limit of definiteness to which we
   can characterise missing/exogenous information within the bounds
   of the Resolvent Framework?" — which routed into a four-level
   κ_Q ladder, two literature scouts (applied side + framework side)
   converging on PARTIALLY PUBLISHED, then a formal audit on the
   framework side that PARKED the seed with one revival trigger
   (T1 execution). The session concluded with a corrigendum to the
   MITACS-side companion note and a framework-side stake added to
   the CLAUDE.md T1 entry. Three commits; clean stopping point.
   Pay special attention to: (i) the audit verdict shape (Type 4
   vocabulary CONDITIONAL FAIL on circumlocution; Type 7 methodology
   CONDITIONAL pending T1 + N>1; all other types FAIL or N/A);
   (ii) the framework-side stake on T1 (either-sign ACF-drop verdict
   clears Type 7 — operational utility demonstrated by the
   prediction having existed because of the framing, not by its
   sign); (iii) the citation chain T1's secondary metric now lives
   under (Diggle 1988 *Biometrics* 44: 959-971 + DHLZ 2002 ch. 5
   for the variogram machinery; Cinelli-Hazlett 2022 / Chernozhukov
   et al. 2024 for the inferential-inversion stance).

2. Read `notes/seeds/kappa_q_limits_applied.md` — the MITACS-side
   companion to the framework-side parked seed. Carries the level-
   by-level S9-S14 mapping under the post-corrigendum framing and
   the Diggle-component-to-within-stratum-statistic table. T1's
   secondary metric framing is in here; the within-stratum
   diagnostic statistics map directly to Diggle components with
   an "inversion reading" column making the necessary-conditions-
   on-Z translation explicit.

3. Launch the lab notebook dashboard if useful for cross-referencing:
     .venv/bin/python -m post_processing.lab_notebook.app --port 8050
   Home tab surfaces the active thread (resolution-paths-thread,
   current_node=null/paused; A1-A4 amendments; T1/T2/T3 temperature
   sub-tree planned).

4. `.venv/bin/python -m experiment.audit.registry list` — current
   state unchanged from 2026-05-29-b:
   - 10 SETTLED experiments (S1-S14 plus multiscale v1
     PROVISIONAL/superseded)
   - 2 EXHAUSTED threads: distributional-class, missing-content
   - 1 ACTIVE thread: resolution-paths-thread (current_node = null;
     paused-but-active-on-temperature-arm via A4; T1 phase_a fill
     blocked on ECCC hourly weather data acquisition)
   - 1 AWAITING-B-by-per-artifact-view-but-EXHAUSTED-by-thread-view:
     Q1A' (close-via-amendment A3)

What landed last session (commits 9c66ea4 → 1ba6382, 3 commits)

This was the 2026-05-30-a session, short by commit count (3) but
high cognitive load (cross-session framework-side coordination
with two scouts, formal audit with two-round type-declaration
review, MITACS-side corrigendum). Single arc through five stages:

1. **Reframe at session open.** User: "I still want to understand
   what is the limit of definiteness to which we can characterising
   missing/exogenous information within the bounds of the Resolvent
   Framework." Original ECCC scout task pre-empted; comprehension
   work took priority.

2. **Comprehension notes drafted.** Read framework-side parked
   note `disintegration_diagnostic.md` (level-1 identifiability
   obstruction theorem; Bergna 2026 / H-S 1984 / Allahverdyan 2020
   / A-M 1974) + audit correction. Advisor-call surfaced four-level
   ladder scaffold. Wrote both halves (framework-side abstract +
   MITACS-side S9-S14 mapping); cross-linked. Both initially
   overstated level-3 novelty.

3. **Two scouts.** Applied-side scout dispatched in background;
   verdict PARTIALLY PUBLISHED with Diggle 1988 / DHLZ 2002 ch. 5
   as canonical match (three-component covariance decomposition).
   User connected framework-side scout's verdict back; initial
   disagreement (framework side characterised "temporal/spectral
   dimension essentially absent"; my side named Diggle as canonical)
   resolved through one round-trip with user as channel —
   framework-side scout confirmed Diggle provides machinery but
   not inferential inversion; both sides converged on
   "PARTIALLY PUBLISHED, surviving novelty = inferential inversion
   + four-level hierarchy + unification."

4. **MITACS-side input note for type declaration.** Per user
   sequencing (surface input → wait for type declaration + formal
   audit → corrigendum after), wrote
   `notes/lab/2026-05-30_mitacs-side-input-for-type-declaration.md`.
   One concrete Type 4 instance (Goal-4 §7→§8); prospective Type 7
   evidence (T1's pre-registered ACF-drop prediction); honest
   caveats. Framework session preserved verbatim in its ladder
   note.

5. **Framework-side formal audit (two rounds) → PARK.** Round 1:
   PARK with three revival triggers (T1 execution; Diggle 1988
   JSTOR read; second application domain). Type 4 CONDITIONAL FAIL
   on circumlocution; Type 7 CONDITIONAL pending T1 + N>1; Type 6
   flagged but not declared. Round 2: Type 6 FAIL (Cinelli-Hazlett
   / Chernozhukov 2024 standard stance; Pearl/Manski-style tiering);
   final PARK with one surviving revival trigger (T1 execution).
   Framework session moved ladder note + audit to `covered_leads/`.

6. **MITACS-side corrigendum + CLAUDE.md update.** Three commits
   (input note 9c66ea4; corrigendum bac8865; CLAUDE.md update
   1ba6382). MITACS-side companion now reframed as "level 3 =
   canonical Diggle machinery + Cinelli-Hazlett / Chernozhukov
   inferential-inversion stance"; no novelty claim on either side.
   CLAUDE.md T1 entry has framework-side stake sub-paragraph.

First task: ECCC hourly weather data acquisition (resumed from
2026-05-30 resume prompt)

Per the user's locked direction (Position-C plan, step 2b, from
the 2026-05-29-b session), ECCC remains the infrastructure
pre-requisite for T1 phase_a fill. Scope guidance per the
within-stratum diagnostic + new framework-side stake:

- **Temporal resolution**: HOURLY (not daily-mean). The lag-1
  ACF +0.67 signature lives in within-day continuity; daily-mean
  data won't capture it; T1 would plausibly fail on both the χ²
  side AND the framework-side-stake secondary metric even if
  temperature is the right Z at the daily level.

- **Station selection**: SINGLE-STATION sufficient for first cut.
  Within-day continuity is preserved at any station in the IESO
  Ontario zone. Multi-station averaging is a secondary concern
  (consider for v2; or as a robustness sub-finding within T1).

- **Time coverage**: pre-cutoff library window (2003-01-01 to
  2024-12-31) + post-cutoff eval window (2025-01-01 to
  2026-05-16). Same temporal scope as Q1, Q2A, Q1A, Q1B used.

- **Forecast vs observed**: BOTH eventually relevant. Observed
  temperature for T1's pre-registered test (both primary χ² and
  secondary lag-1 ACF). **Forecast** (issued ahead of delivery
  hour) for the registered forward experiment
  (`experiment/freeze.py`-style live prediction).

Suggested approach (unchanged from 2026-05-30 resume prompt):

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

   d. **Sanity-check the within-stratum diagnostic** (~15 min,
      in parallel). Re-run
      `scratch/within_stratum_residual_diagnostic.py` on a
      known-Gaussian-process synthetic. If lag-1 ACF ≈ 0 on
      synthetic, Ontario +0.67 signature is real. If +0.67 on
      synthetic too, the script has a bug — fix before T1's
      secondary metric is pre-registered. Cheap; de-risks the
      framework-side stake.

   e. Then dispatch the preregister agent to fill T1's phase_a
      per the spec in `nodes."T1".phase_a_skeleton` of
      `notes/preregistrations/2026-05-27_resolution-paths-thread/thread.yaml`,
      with the secondary metric on within-cell lag-1 ACF per
      CLAUDE.md's T1 forward-pointer + the framework-side stake
      added 2026-05-30-a.

What's new since the 2026-05-30 resume prompt

The substantive new content for the T1 phase_a author is the
framework-side stake. T1 execution is now the single surviving
revival trigger for the framework-side `kappa_q_characterisation_
ladder` seed at `~/Research/Mathematics/Resolvent_Framework/notes/
covered_leads/`. Either sign of T1's pre-registered ACF-drop
prediction clears the framework-side Type 7 (methodology) bar —
the operational utility is demonstrated by the prediction having
existed because of the framing, not by its sign. This adds:

- **Visibility.** The T1 phase_a author should be aware that T1
  result is being watched from the framework side, but this is
  not gating discipline — T1's primary metric (χ² reduction
  under conditioning on temperature) is independent of the
  framework-side stake.

- **Citation chain for the secondary metric.** T1's pre-
  registration should cite Diggle 1988 / DHLZ 2002 ch. 5 as the
  methodological source for the lag-1 ACF statistic (σ²ρ(1)
  component of the variogram) + Cinelli-Hazlett 2022 /
  Chernozhukov et al. 2024 for the inferential-inversion stance
  (treating the fitted component as a necessary condition on Z).
  No novelty claim.

- **Goal-4 §7→§8 paragraph cite list.** The deferred §7→§8
  paragraph is now writeable citing the same chain (Diggle for
  machinery; Cinelli-Hazlett / Chernozhukov for stance). No
  novelty claim. Deferrable until next writeup pass; not
  blocking T1.

If the user opens with "what's next"

Confirm the order above before starting: scout ECCC, then build
loader, then sanity-check diagnostic (or run that in parallel),
then T1 phase_a. The data-acquisition work remains genuinely
uncertain in scope (depends on what ECCC's data-access mechanisms
look like in practice) so the user may want to scope it before
committing.

If the user opens with a different direction

Follow it, but read in this order: `notes/lab/2026-05-30.md` (the
session retro), `notes/seeds/kappa_q_limits_applied.md` (the
post-corrigendum applied-side companion), the framework-side
covered_leads/ files (`kappa_q_characterisation_ladder.md`,
`kappa_q_characterisation_ladder_audit.md`, `residual_structure_
inference.md` as calibration anchor), and CLAUDE.md's T1 entry
for the framework-side stake.

Open follow-ups (not blocking next session)

- **Companion memo refresh**: `writeup/tex/missing_content_memo.tex`
  §9-§12 still uses bounded-finding framing pre-corrigendum;
  PARK verdict + Diggle/Cinelli-Hazlett/Chernozhukov citations
  load-bearing if it gets refreshed. Defer until co-author flag
  or `/audit pure` on the Resolvent_Framework side triggers it.

- **Goal-4 paper `\todo` blocks**: §1 prior-work context (now
  could include Cinelli-Hazlett / Chernozhukov for the
  sensitivity-analysis precedent); §7 forward-experiment results
  (accrues with calendar time). Both for author group.

- **Workflow-infrastructure-scheduling**: three items deferred
  per `workflow-infrastructure-scheduling-pattern.md` memory
  (cosmetic-amendment carve-out refinement; --prereg-dir CLI
  refactor; registry status enum extension); now four after the
  2026-05-30-a observation about lab-note-tool commit-count
  off-by-one. None blocking T1 work.

- **N=2 second-domain application** of the within-stratum
  diagnostic (e.g., IEEE DataPort Post-COVID dataset): framework-
  side audit round 2 did not retain this as Type-7-clearing
  independent of T1, but might still clear Type 7 if ECCC
  acquisition turns out infeasible. Surfaced for awareness.

Workflow rules that have caught retractions (do not bypass)

(Carried forward from 2026-05-30 resume prompt with one addition
this session.)

- /audit code-path BEFORE running; /audit phase-fidelity Check P
  BEFORE running; Check R BEFORE arbiter renders.
- Proponent + DA forecasts stamped simultaneously, both numeric,
  both locked before data is touched. Simultaneity is NON-
  OBSERVATION, not stamp ordering.
- Pre-run methodology amendment carve-out (5 instances; carve-out
  rule unchanged).
- Arbiter writes note.md as part of dispatch.
- Thread-coordinator rewrites note.md at every procedure.
- Cannot change branching_rules[node_id] for nodes that have
  settled (their branches already fired).
- The discipline rule is SYMMETRIC: pre-judging a negative
  result is the same workflow risk as pre-judging a positive
  one.
- CI-prominent framing for any result where CI spans > 10× the
  point estimate or CI lower bound is near a prior-relevant
  baseline.
- INSPECTION-ONLY scratch carries a greppable `PROVENANCE-GRADE:
  INSPECTION-ONLY` banner — never cite it as a result.
- Amend thread.yaml directly (thread-coordinator is sole writer).
- Skip the inline synthetic gate before any Ontario read on any
  new experiment script.
- Touch settled-node entries in thread.yaml (P1, Q1A, Q1B, Q1B',
  Q1C; now also Q1A' which is exhausted).
- Restructure the memory corpus.
- Ship CLAUDE.md (S15) entry without CI tightening or explicit
  user direction (if T1 produces a wide-CI substantive finding,
  same rule applies).
- **NEW from 2026-05-30-a:** Cross-session ownership discipline.
  When a framework-side note has been substantively edited by
  the framework session, do not edit it from this side without
  explicit framework-session go-ahead. Flag stale paragraphs;
  let the owning session clean them up. (Internalised after the
  framework session's edit of `kappa_q_characterisation_ladder.md`
  this session — the framework-side note left some pre-converged-
  scout paragraphs stale, but corrigendum responsibility was
  theirs, not mine.)

Lab note workflow

- Session start: `.venv/bin/python notes/lab/tools/lab_note.py
  --new YYYY-MM-DD` (or `YYYY-MM-DD-b` for multi-session days).
- During session: prose in body as work happens.
- Session end: `lab_note.py --refresh notes/lab/<file>.md` to
  refresh machine-derivable fields. Judgment fields stay manual.
- **Known issue (2026-05-30-a):** lab-note-tool commit-count can
  be off-by-one if a commit landed mid-scaffold. Manually verify
  `commits:` in frontmatter at session end.

Session-volume note from 2026-05-30-a

3 commits — well under the workflow's throttle threshold (>10).
Cognitive load was higher than the count suggests (cross-session
coordination + formal audit + two-round type review). The
"is this the same workstream or am I drifting?" check held —
single arc from session-open reframe through PARK + corrigendum.
Stopping point came naturally with the audit verdict. Next session
opens fresh with ECCC as the obvious first task and the framework-
side T1 stake as new context to carry.
