# Applied-Research-Audit Workflow

**Status.** Seed note. Written 2026-05-25, immediately after
`multiscale_factor_coherence.md` identified the need for an applied
analogue of the theory-side `/audit` workflow. Two background sources:
a focused literature scan on applied-ML pre-registration / multiverse
/ adversarial-collaboration / trustworthy-experimentation, and a
corpus review of this project's 158-commit, 26-memory history
documented in [[mitacs-session-lessons-corpus]].

**Question this seed answers.** The theory `/audit pure` is built
around novelty — adversary is the field. Applied work in this lab
has a different binding constraint: 13 of 26 memory files carry
explicit RETRACTED / RESOLVED / SUPERSEDED markers, and the
retraction pattern is consistent. The applied audit's threat model
is "the analyst's own future self" (and previous selves), not "the
field."

This seed proposes what the workflow should look like, mapped to
both the existing applied-research literature and to this project's
empirically-observed failure modes.

---

## 1. The threat model — what we are actually defending against

From the corpus review, retraction sources in this project, in
descending order of frequency:

1. **Diagnostic-reimplements-production.** Single most common
   cause. A custom script computes something that *looks* like the
   production metric but isn't. The result is treated as a finding;
   later it falls when run through the actual production path.
   Examples: "sunday multi-mode r_hat≈3", "Mardia 10–31×",
   "4-decimal match".

2. **Argmin-without-variance.** A peak / optimum is read from a
   curve without a bootstrap or sampling-noise check. The peak
   turns out to be within noise of the curve being flat or
   monotone. Examples: "d wants 5–8", "true-LOO-CV validated",
   the entire #41 family.

3. **In-sample / on-development reading.** A pattern fit on the
   data being analyzed is presented as having predictive content.
   Pre-registration breaks the loop by forcing OOS evaluation;
   absent that, this fails routinely.

4. **Cosmetic vs mechanistic confusion.** A parameter sweep
   produces a metric improvement; the improvement looks
   mechanistically motivated but is actually accidental. Tonight's
   W=2y rescore was an example; the σ-shrinkage cancelled an
   unrelated climatology gap.

5. **Framework-level vs parameter-level confusion.** Tuning levers
   within a broken framework while believing the framework is
   sound. Hours of work go into something the framework cannot
   fix. Tonight's evening session was a partial instance —
   recognized after five framework-level variant tests.

6. **Single-session momentum.** Findings cited as established
   within the same session that generated them. Memory headers
   in this project explicitly read "RESOLVED — supersedes the
   earlier RESOLVED block in this same memory" because corrections
   stacked on corrections in single sessions.

The applied audit must address each of these directly. The theory
audit addresses none of them (its threat model is different).

---

## 2. Existing literature — adopt where it fits

A focused scan of the applied-ML methodology literature (results
in this session's general-purpose agent call, 2026-05-25) identified
the following established practices that map to threats 2–6:

| Threat | Established name | Source |
|---|---|---|
| 3. In-sample / pre-reg violation | Two-phase predictive-modeling pre-registration | Hofman et al., arXiv:2311.18807 |
| 4. Cosmetic vs mechanistic | Specification curve / multiverse analysis | Simonsohn et al. 2020; Steegen et al. 2016 |
| 5. Framework vs parameter | Garden of forking paths | Gelman & Loken 2013 |
| 2. Argmin-without-variance | Falsifiable-replicable-reproducible (FRR) ML | arXiv:2405.18077 |
| 6. Scope creep / momentum | Adversarial collaboration | Kahneman; Mellers et al. 2001 |
| (Pre-experiment guards) | Patterns of Trustworthy Experimentation | Microsoft Research ExP |

**Threats 1 (diagnostic-vs-production) and the registered-experiment-
mutation specifics are not adequately covered in the literature.**
The corpus review confirms that the diagnostic-vs-production issue is
the single highest-leverage rule in this project's history
([[mitacs-session-lessons-corpus]], §"What catches them" rule 1).
The literature focuses on test-set-leakage and analyst-degree-of-
freedom issues, but not on the more basic problem of the diagnostic
code path drifting from the production code path. **This is original
discipline content that the workflow needs to enforce explicitly.**

---

## 3. What we already have — assets in this repo

The lab is not starting from zero. Existing assets:

- **`experiment/freeze.py` + frozen spec hash.** Pre-registration
  mechanism for the forward experiment. Strict refuse-on-dirty.
- **`experiment/provenance/make_result()` + `inputs_fingerprint` +
  `body_sha256`.** Artifact-level provenance envelope. Already
  exceeds NeurIPS / MLRC reproducibility standards
  ([[mitacs-provenance-requirements]]).
- **Synthetic validation gates calling the production estimator
  directly** (`processing/innovations/validation/synthetic.py`,
  `rebaseline.py`). The discipline rule "a diagnostic that
  reimplements production logic is a hypothesis, not a finding,
  until confirmed through the production code path" is stated in
  `CLAUDE.md`.
- **Memory system with explicit retraction tags.** ★ markers,
  "RETRACTED" headers, "supersedes the earlier RESOLVED block in
  this same memory" lineage tracking.
- **The advisor() tool.** Documented in `CLAUDE.md` as required
  before substantive work.

The audit workflow doesn't replace these — it composes them into a
gated pipeline.

---

## 4. The proposed workflow

Six agent / skill roles, ordered by where they sit in the
research lifecycle. Naming follows the theory side's convention
(`/audit <mode>` for skills, individual agents for narrow roles).

### 4.1. `/audit code-path` — diagnostic-vs-production drift detector

**Threat addressed:** 1 (the single highest-leverage one).

**Adopts from literature:** nothing — this is original to this lab.

**What it does.** Given a claim and the script that produced it,
inspect whether the script:
(a) calls production functions (`build_local_gaussian_semigroup`,
    `experiment.backtest.main`, `experiment.predict`, etc.), or
(b) reimplements production logic in the script.

If (b): the claim is labelled **PROVISIONAL** by default and
cannot be cited as a finding without explicit production-path
re-verification. The agent identifies which production function
*should* have been called and what would need to change for the
script to do so.

**Trigger.** Before any memory file is updated with a new finding
or a writeup section is drafted citing an empirical claim.

### 4.2. `/audit preregister` — two-phase pre-registration

**Threats addressed:** 3 (in-sample reading), 4 (cosmetic vs
mechanistic), 6 (scope creep).

**Adopts from literature:** Hofman et al. (2023) two-phase
predictive-modeling pre-registration; Kahneman/Mellers adversarial-
collaboration "what would change my mind" rule.

**What it does.** Produces a hash-stamped artifact in
`notes/preregistrations/` with two sections:

- **Phase A (before any data is touched):** research question,
  variables, train/dev/test partition, metric and threshold,
  baselines, **pre-stated falsification criterion**, **pre-stated
  corroboration criterion**, **region of ambiguity** between them,
  and the analyst's *own forecast* of which side will win.

- **Phase B (before touching held-out test):** algorithm,
  hyperparameter selection rule, random seeds, planned secondary
  analyses, **declared deviations from Phase A with reasoning**.

The agent refuses to mark "complete" if either falsification
criterion or corroboration criterion is non-numeric. Diff-guards
subsequent commits against the Phase A artifact.

**Trigger.** Before any experiment that will produce a citable
finding. For multistep parameter selection: before reading any
result curve.

### 4.3. `/audit multiverse` — specification-curve

**Threat addressed:** 4 (cosmetic vs mechanistic).

**Adopts from literature:** Simonsohn/Simmons/Nelson specification
curve; Steegen et al. multiverse analysis.

**What it does.** Given a claimed effect and the analytic choices
that produced it, enumerate the defensible alternatives at each
choice point (kernel bandwidth, embedding dim, day-anchor, recency
window, day-type stratification, climatology method, ...). Run
the experiment across the grid. Report the effect-size distribution
across the specification grid.

- **Mechanistic verdict** if the effect survives most defensible
  specifications.
- **Cosmetic verdict** if the effect lives in one cell of the grid.

This directly addresses what would have caught the W=2y rescore
result before it was treated as a finding.

**Trigger.** After a parameter sweep produces a "best" choice, and
before that choice is committed to a writeup.

### 4.4. `/audit framework` — framework-vs-parameter escalation

**Threat addressed:** 5.

**Adopts from literature:** Garden of Forking Paths critique
(Gelman & Loken); no clean packaged methodology exists, so this is
partial fresh design.

**What it does.** Two-tier extension of `/audit preregister`:
when `/audit multiverse` returns a "cosmetic" verdict, escalate to
the framework-level question. Force the analyst to:
(a) name the framework assumption being implicitly tuned around,
(b) name at least one alternative framework,
(c) name an experiment that would distinguish them.

This was the discrete reframe that worked in tonight's session
("wait, the z-transform is interfering with short-scale
dynamics" → tested 3 transforms → recognized the bias shape was
framework-intrinsic). Encoding it as a workflow step makes it
reachable without needing a fresh insight every time.

**Trigger.** After `/audit multiverse` returns cosmetic.

### 4.5. `/audit pre-experiment` — engineering design review

**Threats addressed:** 6 (scope creep), registered-experiment
mutation, AA / variance assumptions.

**Adopts from literature:** Microsoft ExP "Patterns of Trustworthy
Experimentation" pre-experiment stage.

**What it does.** A one-page checklist the agent walks through
before any experiment runs:
- AA test / detrending baseline split clean across subgroups?
- Sample Ratio Mismatch between registered population and scoring
  population?
- Are baselines pre-specified and unchanged from registration?
- Is the registered spec hash unchanged?
- Bootstrap / CI method specified for every reported number?
- Is there a stopping criterion?
- Refuses `N/A` without justification.

**Trigger.** Before submitting a job (SLURM, local, anything that
will write a results pickle).

### 4.6. `/audit arbiter` — neutral judgment between proponent and devil's-advocate

**Threats addressed:** 6 (single-session momentum), Kahneman/Mellers
unfalsifiable-skepticism failure mode of the existing devil's-advocate.

**Adopts from literature:** Mellers et al. 2001 adversarial-
collaboration role split.

**What it does.** A third agent given:
- the proponent's pre-registered prediction (from `/audit
  preregister`),
- the devil's-advocate's pre-registered counter-prediction
  (logged at the same time, not retrospectively),
- the experimental result.

Returns: which prediction the data supports, with quantitative
margin. Only this agent can declare a hypothesis "resolved." The
existing `devils-advocate` agent (`.claude/agents/devils-advocate.md`
from the theory side, adaptable) is the counter-prediction source.

**Trigger.** After experiment results are in, before any memory
update cites the result as settled.

---

## 5. Shared infrastructure

All six skills share a single registry — the natural extension of
`experiment/freeze.py` and `experiment/provenance/_prov_core.py`:

```
notes/preregistrations/
  <date>_<topic>/
    phase_a.yaml          (hash-stamped, written before data)
    phase_b.yaml          (hash-stamped, written before test set)
    falsification.yaml    (numeric criteria)
    proponent.yaml        (proponent's forecast)
    devils_advocate.yaml  (counter-forecast, locked simultaneously)
    multiverse.yaml       (specification grid + verdict)
    arbiter.yaml          (final verdict, with quantitative margin)
```

Append-only, hash-chained. The audit skills read and write into
this registry; results published via `make_result()` must reference
the corresponding registry entry.

---

## 6. Throttle: the single-session-momentum case

The corpus review identified that >10 substantive commits in one
session and multi-step reasoning chains are the volume profile
where retractions historically happen. Two patterns to address this:

**6.1 Mandatory consolidation pause.** After N commits in a session
(N=10 as a starting heuristic), the workflow requires a "stop and
consolidate" pass: re-read the day's memory updates, identify which
findings are hypotheses vs verified, list which production-path
re-runs are still owed. No new findings can be filed until this is
done.

**6.2 Hypothesis tag persistence.** Every finding written into a
memory file in the current session is automatically tagged
"SESSION-PROVISIONAL" until either (a) it has passed `/audit
code-path` and `/audit arbiter`, or (b) a subsequent session
explicitly re-affirms it. The MEMORY.md index renders
SESSION-PROVISIONAL findings differently so they cannot be cited as
established.

---

## 7. What this workflow is NOT

- **Not a replacement for the advisor() tool.** The advisor catches
  cases the workflow doesn't (it can see the full conversation; the
  workflow only sees explicit artifacts). Keep both.

- **Not a replacement for the theory-side `/audit pure`.** When and
  if a finding crosses from lab artifact to theory claim, it needs
  the theory-side novelty audit. The two workflows answer different
  questions.

- **Not novel research.** Five of the six roles adopt existing
  literature directly. The contribution is the *assembly* (and the
  diagnostic-vs-production rule that is genuinely fresh). No
  publication claim attaches.

- **Not yet implemented.** This seed proposes the workflow; building
  the agents and the registry is a separate piece of work, scoped
  in §9.

---

## 8. What this workflow is NOT yet known to be sufficient for

The corpus review identified several patterns that this workflow
addresses *structurally* but where I cannot yet predict whether it
will catch them in practice:

- **Display-rounding overconfidence.** "4-decimal match" was a
  numerical precision issue; the audit doesn't directly inspect
  number-of-significant-figures reporting. Maybe gets caught by
  `/audit code-path` (the right diagnostic would have shown the
  1e-5 difference); maybe doesn't.

- **"Robust across X" claims where X is the wrong axis.** A
  multiverse pass would catch this if X is in the specification
  grid, but if X is *outside* the grid (different embedding dim,
  for instance), it won't.

- **Drift in the registry itself.** If preregistration artifacts
  themselves accumulate errors, the workflow doesn't self-audit.
  This is the same issue the theory side has with its memories;
  no clean answer.

These should be tracked and the workflow updated as new failure
modes are observed.

---

## 9. Implementation order

The workflow proposed in §4 is not all-or-nothing. Most-leverage
sequence:

**(9.1) Build `/audit code-path` first** [task #44 → 46]. This
threat has caused the most retractions in this project and the
fewest existing mitigations. It is mostly orthogonal to the others;
landing it does not block anything else.

**(9.2) Build `/audit preregister` second** [→ 47]. Pre-registration
discipline is the second-highest-leverage and the most thoroughly
covered by literature. Most of the implementation is the Phase A /
Phase B YAML schema and the diff-guard.

**(9.3) Build `/audit multiverse` third** [→ 48]. Heavier
implementation lift but high-payoff for the specific failure mode
that tonight's session caught manually. Requires extending
`experiment/backtest.py` to take a specification-grid argument.

**(9.4) Build the others as needed** [→ 49+]. `/audit arbiter`,
`/audit framework`, `/audit pre-experiment`. None are blocking.

**(9.5) Eventually: re-audit the existing memory corpus.** Pass
every existing memory file through `/audit code-path` and
`/audit multiverse`. Expect some additional retractions. Treat
this as a single batch, not a finding-by-finding scrutiny — the
goal is to bring the corpus to a known state.

---

## 10. Honest assessment

This is a seed about a workflow, not yet a workflow. It is informed
by:
- a real and frequent failure pattern (retractions in 13/26 memory
  files);
- a focused literature scan that identified established names for
  most of the failure modes;
- this session's specific experience of pulling framework-level
  levers and reading framework-level mismatches.

Where the seed is weakest: the implementation lift is real. Building
six agents, a registry, a memory-tag system, and integrating them
with `make_result()` is multiple days of careful work. The minimum
viable version is **only `/audit code-path` + a small memory tag
extension** — that alone would have caught the majority of historical
retractions and is the single highest-leverage build.

Where the seed is strongest: it is grounded in this project's actual
empirics, not in a generic methodology checklist. The retraction
ratio in the corpus is high enough that the *expected value* of
the workflow is positive even in conservative estimates.

---

## 11. Retroactive validation (2026-05-26)

The workflow's first self-test: run `/audit code-path` on the
script that produced the `mitacs-honing-methodology` TEST B
retraction ("d wants 5–8" / "1 of 24 hours" / argmin-without-
variance), and verify that the workflow would have flagged the
finding *before* it became a citable claim.

**Script audited.** `scratch/diagnose_embedding_adequacy.py` —
the original (consolidated in commit 0b881c1, May 18) that
produced the retracted finding.

**`/audit code-path` verdict.** MIXED. The script imports four
production functions (`load_actuals`, `zscore_params`,
`zscore_transform`, `_daytype`) but reimplements OLS inline at
lines 93 and 117 (`np.linalg.lstsq(X, Y, rcond=None)[0]`) rather
than calling `processing.innovations.estimator.local_drift_and_diffusion`.
The script is therefore PROVISIONAL under the workflow's rules
and cannot be cited as a finding.

**`/audit preregister` would have caught it too.** The original
TEST B had no `ci_method` field — argmin was read without a
variance check. The `preregister` agent refuses to mark Phase A
complete without `ci_method.type`, `ci_method.details`, and
`ci_method.width`. So even if the script had been production-
path, the absence of a bootstrap-SE rule would have blocked the
finding.

**Cross-validation.** The re-check script
(`scratch/recheck_test_b_bootstrap.py`) — which *did* find that
1 of 24 hours cleared the bar — is also MIXED (inline OLS at line
109). The bootstrap-SE check is present in code but the agent
does not statically detect that; the `preregister` agent would
have required the CI method to be declared in the YAML up front.

**Verdict on the workflow's retroactive validity.** Both
independent gates (`code-path` and `preregister`) would have
flagged the original TEST B before it produced a citable finding.
Either alone is sufficient; together is robust.

The validation does NOT prove the workflow catches every retraction
mode in the corpus — only that it catches this canonical instance
of the #2 retraction source (argmin-without-variance) plus the #1
source (diagnostic-reimplements-production). Other classes (e.g.
the dayanchor-seam diagnostic-harness artifacts in
`mitacs-dayanchor-seam`) need separate audit passes once the
multiverse and framework agents become operational; preliminary
inspection (see commit message of #48) suggests `code-path` would
also flag those.

---

## 12. Lineage

- [[mitacs-session-lessons-corpus]] — the retraction-pattern
  review that grounds the threat model.
- [[mitacs-provenance-requirements]] — the existing artifact-
  level layer this workflow extends to the claim level.
- `multiscale_factor_coherence.md` (this directory) — the seed
  that surfaced the need for this workflow.
- Theory-side `.claude/agents/` and `.claude/skills/audit/` in
  `Research/Mathematics/Resolvent_Framework` — the structural
  template for skill/agent file format.
- General-purpose agent literature scan, 2026-05-25 — the source
  of the §2 mapping table.
