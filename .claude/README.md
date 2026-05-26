# Lab Audit Workflow

Applied analogue of the theory-side workflow in
`Research/Mathematics/Resolvent_Framework/.claude/`. Design source:
`notes/seeds/applied_audit_workflow.md`. Threat model and lessons
from corpus review: `memory/mitacs-session-lessons-corpus.md`.

## Layout

```
.claude/
  agents/
    code-path-auditor.md      ← diagnostic-vs-production drift detector (FRESH design)
    preregister.md            ← two-phase pre-registration (Hofman et al.)
    multiverse.md             ← specification curve (Simonsohn/Steegen)
    framework-escalator.md    ← framework-vs-parameter escalation (partly fresh)
    pre-experiment-checklist.md ← engineering review (Microsoft ExP)
    arbiter.md                ← proponent-vs-devils neutral judge (Mellers)
    devils-advocate.md        ← counter-prediction generator (tuned from theory)
    editorial-pass.md         ← 16-rule writeup polish (ported from theory)
    literature-scout.md       ← applied-ML literature search (ported from theory)
    thread-coordinator.md     ← line-of-inquiry lifecycle owner (FRESH design)
  skills/
    audit/
      SKILL.md                ← /audit dispatcher
```

## Threat model

Lab threat model is **the analyst's own future self**, not the
mathematical field. Concretely, the eight failure modes the
workflow defends against (in descending frequency from corpus
review):

1. Diagnostic-reimplements-production
2. Argmin-without-variance
3. In-sample / on-development reading
4. Cosmetic-vs-mechanistic confusion
5. Framework-vs-parameter confusion
6. Single-session momentum
7. Registered-experiment mutation
8. Provenance drift

Each agent addresses one or more of these. See the seed doc
(`notes/seeds/applied_audit_workflow.md` §1, §4) for the mapping.

## Implementation status

- ✅ **Agents specified**: 10 `.md` files (9 original + thread-coordinator).
- ✅ **Registry built**: `notes/preregistrations/` with 8 templates
  including thread.yaml; hash-chained verification + CLI in
  `experiment/audit/registry.py`.
- ✅ **`experiment/audit/` package built**: `code_path.py` and
  `registry.py` with thread support (`thread new`, `thread status`,
  `thread list` subcommands).
- ✅ **Workflow validated retroactively** (TEST B retraction would
  have been caught) AND prospectively (v2 multiscale settled cleanly,
  v3 eigenmode characterization caught a mechanical-vs-substantive
  gap via MIXED verdict).
- ✅ **Threads added** as a new artifact type for lines of inquiry —
  pre-registered branching trees that prevent silent post-hoc
  pivoting at the cross-experiment level.
- ⏳ **First thread not yet written**: missing-content thread for the
  multiscale follow-up programme is the first live use of the
  thread workflow.

## Reading order for a new collaborator

1. `CLAUDE.md` (project root) — what the lab is, how the pipeline
   runs, the discipline rule.
2. `notes/seeds/applied_audit_workflow.md` — why this workflow
   exists.
3. `memory/mitacs-session-lessons-corpus.md` — the lab's actual
   failure history.
4. This directory's `agents/*.md` — the role definitions.
5. `skills/audit/SKILL.md` — how to invoke.
