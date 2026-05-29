# Preregistration Registry

Hash-stamped pre-registration artifacts for the lab. One subdirectory
per claim, structured to be the input the audit workflow
(`/audit *`, see `.claude/skills/audit/SKILL.md`) reads and writes.

Design: `notes/seeds/applied_audit_workflow.md` §5.
Threat model: `memory/mitacs-session-lessons-corpus.md`.
Hash primitives: `experiment/_prov_core.py`
(`git_sha`, `git_clean`, `canonical`, `sha256_hex`).

## Why this exists

The lab's largest retraction category is **argmin-without-variance**
and **in-sample reading** of results. The registry forces a numeric
falsification criterion, a numeric corroboration criterion, an
ambiguous region, a proponent forecast, and a devil's-advocate
counter-forecast — *all written down before data is touched*. After
the result is in, `/audit arbiter` grades the four predictions
against the numeric criteria.

This is direct adaptation of Hofman et al. (arXiv:2311.18807)
two-phase predictive-modeling pre-registration, plus
Kahneman/Mellers adversarial-collaboration role split. See
`.claude/agents/preregister.md` and `.claude/agents/arbiter.md`.

## Directory structure

Two kinds of entries live here: **individual experiments** and **threads**.

```
notes/preregistrations/
  README.md                       ← this file
  _template/                      ← stubs for each artifact type
    phase_a.yaml                  ← individual experiment (or thread node)
    phase_b.yaml
    proponent.yaml
    devils_advocate.yaml
    multiverse.yaml
    arbiter.yaml
    result.yaml
    thread.yaml                   ← thread (line of inquiry)

  <YYYY-MM-DD>_<topic-slug>/      ← one per experiment, append-only
    phase_a.yaml                  ← (written first; before any data)
    devils_advocate.yaml          ← (written simultaneously with phase_a)
    proponent.yaml                ← (written simultaneously with phase_a)
    phase_b.yaml                  ← (written before touching test set)
    multiverse.yaml               ← (written when /audit multiverse runs)
    result.yaml                   ← (written after experiment runs)
    arbiter.yaml                  ← (written last; the only file that may
                                     declare SETTLED)

  <YYYY-MM-DD>_<topic-slug>-thread/   ← one per thread, append-internal
    thread.yaml                   ← the line-of-inquiry artifact
                                     (amendments append inside the same
                                     file; see "Threads" below)
```

## Individual experiments vs threads

An **individual experiment** answers one scoped question with one
arbiter verdict. Phase A pre-registers the question, the metric, the
criteria, the forecasts; arbiter renders the verdict; the entry is
done. Most entries to date follow this pattern.

A **thread** is a *line of inquiry*: a sequence of experiments whose
structure is itself pre-registered. The thread artifact (`thread.yaml`)
holds the scientific question, the planned tree of experiment nodes,
the branching rules (which parent verdict fires which child), and the
current state pointer. Individual experiments under the thread
register normally as `phase_a.yaml` under their own dated entries, but
each MUST reference the thread and node it belongs to, and the
arbiter MUST fire one of the thread's pre-registered branches.

Threads exist because the lab's common failure mode at the *cross-
experiment* level is silent post-hoc pivoting: results come in, the
analyst chooses the most interesting follow-up post-hoc, the choice
looks decisive because it was informed by the result. Threads make
the next-experiment decision pre-registered against the prior
verdict, so a pivot is visible.

A thread may end in three ways:

  - **resolved** — the question is answered; final claim in the
    thread's resolution block.
  - **exhausted** — all branches consumed without resolving; to
    continue requires an amendment.
  - **abandoned** — analyst chose to drop; reason recorded.

Threads may be **amended** during their lifetime (new branches added,
old branches closed, skeletons rewritten). Amendments are
**append-internal**: the thread.yaml file is overwritten but each
amendment appends to the `amendments` list inside it, chained by
prior-tree-hash for tamper detection.

## File invariants

Every YAML file in the registry has frontmatter:

```yaml
schema: <phase_a | phase_b | proponent | devils_advocate | multiverse | result | arbiter>
written_at: <ISO 8601>
git_sha: <commit at write time>
git_clean: <bool>
body_sha256: <hash of canonical body, excluding the body_sha256 field itself>
references:
  - file: <relative path, e.g. phase_a.yaml>
    body_sha256: <hash of referenced file>
```

The `body_sha256` chain makes the registry tamper-evident:
`/audit arbiter` recomputes hashes on read and refuses to arbitrate
a claim whose chain is broken.

`git_clean: false` is allowed for individual files (the lab works
on dirty trees during exploration), but `arbiter.yaml` REFUSES to
arbitrate any claim where the `result.yaml` was written from a
dirty tree. This mirrors the `freeze.py` policy: a recorded SHA
must reproduce the artifact.

## Append-only rule

Files in `<topic>/` are written once and never modified. Revising
means a NEW dated entry:

```
2026-05-26_multiscale_h2_drift/                  ← original
2026-05-30_multiscale_h2_drift_v2/               ← revision
  phase_a.yaml
    references:
      - file: ../2026-05-26_multiscale_h2_drift/phase_a.yaml
        body_sha256: <hash>
        relation: supersedes
        reason: <one sentence>
```

The lineage is the audit trail. Git history records what changed;
the registry records why.

## How the agents use this

| Agent | Reads | Writes |
|-------|-------|--------|
| `thread-coordinator` | `_template/thread.yaml`, prior thread state | `<topic>-thread/thread.yaml` (creation + amendments + state advancement) |
| `preregister` | `_template/phase_a.yaml`, `_template/phase_b.yaml`, optionally `<thread>/thread.yaml` | `<topic>/phase_a.yaml`, `<topic>/phase_b.yaml` |
| `devils-advocate` | `<topic>/phase_a.yaml` | `<topic>/devils_advocate.yaml` |
| `multiverse` | `<topic>/phase_a.yaml`, `<topic>/phase_b.yaml` | `<topic>/multiverse.yaml` |
| `arbiter` | the entire `<topic>/` directory + the result artifact + (if thread) the thread.yaml | `<topic>/arbiter.yaml` + memory update + (if thread) thread state update via `thread-coordinator` |

The proponent forecast is captured by `preregister` (inside `phase_a.yaml`'s
`hypothesis.proponent` field) AND in a separate `<topic>/proponent.yaml`
for symmetric grading against `devils_advocate.yaml`.

### Thread-specific agent contracts

When an experiment belongs to a thread:

- `preregister` MUST verify the new `phase_a.yaml`'s content matches
  the thread's pre-registered skeleton for the declared `node_id`.
  Mismatch is a registration violation; the agent refuses.

- `arbiter` MUST set `branch_fired` from the thread's pre-registered
  `branching_rules[node_id]`. The arbiter cannot render SETTLED
  without selecting a branch. After SETTLED, `thread-coordinator`
  advances the thread state.

- `/audit thread <slug>` verifies the entire thread's chain:
  hash-chain across all child experiments, branch consistency,
  amendment-chain integrity.

## Status tags

After arbitration, every finding is tagged with one of:

- `SETTLED` — written by `arbiter` only. May be cited as fact.
- `PROVISIONAL` — written by any other agent. Must re-audit before
  citing.
- `SESSION-PROVISIONAL` — auto-set on findings created during a
  high-volume session (`>10 substantive commits`, per the
  single-session-momentum throttle in
  `applied_audit_workflow.md` §6.1).
- `RETRACTED` — explicit retraction; a successor entry must
  reference this and explain.

Memory citations (in `~/.claude/projects/.../memory/*.md`) MUST
include the tag. The convention is to put `[SETTLED]` or
`[PROVISIONAL]` next to any reference to a registry entry.

## Registry-list view vs thread view (known limitation)

`registry list` (defined in `experiment/audit/registry.py:list_entries`
calling `entry_status`) shows a per-experiment status derived strictly
from artifact existence in the experiment directory: `AWAITING-B` if
only `phase_a.yaml` exists, `AWAITING-RES` if `phase_b.yaml` exists
but no `result.yaml`, `AWAITING-ARB` if `result.yaml` but no
`arbiter.yaml`, then `SETTLED` / `PROVISIONAL` / `AMBIGUOUS` once the
arbiter has rendered. It does NOT read `thread.yaml`.

This creates a known mismatch when a thread node is **closed via
thread amendment without running** (status: `exhausted`,
`closed`). The per-experiment view continues to show whichever
phase the existing artifacts suggest (typically `AWAITING-B` if
phase_a was stamped before the amendment closed the node).

**Current instance:** Q1A' under `2026-05-27_resolution-paths-thread`.
Closed via thread amendment A3 (2026-05-29; FastICA selection-rule
design defect; see memory `q1a-prime-fastica-selection-defect`).
`registry list` shows it as `AWAITING-B`; `thread status
2026-05-27_resolution-paths-thread` shows it as `exhausted`. The
**thread view is the source-of-truth for thread-membership state**;
the per-experiment view is the source-of-truth for artifact existence
on disk. They are not in conflict; they answer different questions.

**Convention for readers**: when scanning for "what experiments are
still open / awaiting work", consult `thread status <slug>` for the
relevant thread before trusting `registry list`. A node that
`registry list` reports as `AWAITING-B` may already be `closed` /
`exhausted` at the thread level via amendment.

**Workflow-design retro question** (not addressed; N=1 current
instance does not warrant infrastructure change): should
`entry_status()` cross-reference `thread.yaml` for nodes that
belong to a thread? If a second close-via-amendment case arises,
revisit. The minimum-cost extension would be a `closed.yaml` marker
file written by `thread-coordinator` when it executes a
close-via-amendment, and a new `entry_status()` return value
`EXHAUSTED-PRE-DATA` / `CLOSED-PRE-DATA` keyed on the marker.

## Reading order for a new claim

1. Read the latest `arbiter.yaml` in each relevant subdirectory.
   If absent or `PROVISIONAL`, do not cite.
2. If `SETTLED`, follow references back to `phase_a.yaml` for
   the original framing.
3. For methodology: read the agent prompts in
   `.claude/agents/`.

## Listing of current entries

(Auto-populated by `experiment/audit/registry.py` once that lands;
manual until then.)

Run `python -m experiment.audit.registry list` for the authoritative
current state. Notable threaded entries:

- `2026-05-27_resolution-paths-thread/` — thread on framework-prescribed
  paths for S9+S10 obstruction. Current node: P1.
- `2026-05-27_p1-patra-sen-per-stratum-localization/` — phase_a +
  proponent + devils_advocate stamped. Node P1 of resolution-paths-thread.
  Plain Patra-Sen (2016) per stratum on Q1 M1 residuals; 5 stratification
  axes; verdict on max-axis range vs 0.20 / 0.10 thresholds.
- `2026-05-29_q1b-prime-smc-tightening/` — phase_a stamped. Non-gating
  sibling node Q1B' under resolution-paths-thread (parent_branch=R-A,
  parent_id=Q1B, per amendment A2). Single-arm empirical CI-tightening
  re-run of Q1B's winning (day_type, Z_c, Z2_c) configuration at
  N_PARTICLES=400 (vs Q1B's 200). Verdict on the realized CI95 width of
  `cumulative_chi2_improvement_above_Z_c`: R-A1B' <= 357.27 (<= 75% of
  Q1B's realized 476.358), R-B1B' in (357.27, 524.0], R-C1B' > 524.0 OR
  |point shift| > 50. Tests whether Q1B's wide CI is SMC-particle-noise
  dominated (R-A1B') or eval-window-noise dominated (R-B1B').
