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

- ✅ **Agents specified** (this directory): 9 `.md` files, all with
  frontmatter + procedure + output schema + rules.
- ⏳ **Registry not built**: `notes/preregistrations/` directory and
  YAML templates not yet implemented. Per the seed §9.1, this is
  the natural extension of `experiment/freeze.py` and
  `experiment/provenance/_prov_core.py`.
- ⏳ **`experiment/audit/` package not built**: `code_path.py` and
  `registry.py` referenced in the seed do not yet exist.
- ⏳ **Workflow itself unaudited**: the design has not been validated
  by running it on a known-retracted finding to confirm it would
  have caught the failure. Highest-leverage validation step
  (per seed §9.5): re-audit `mitacs-honing-methodology` TEST B
  retraction with `/audit code-path` and `/audit preregister`
  retrospectively.

## Reading order for a new collaborator

1. `CLAUDE.md` (project root) — what the lab is, how the pipeline
   runs, the discipline rule.
2. `notes/seeds/applied_audit_workflow.md` — why this workflow
   exists.
3. `memory/mitacs-session-lessons-corpus.md` — the lab's actual
   failure history.
4. This directory's `agents/*.md` — the role definitions.
5. `skills/audit/SKILL.md` — how to invoke.
