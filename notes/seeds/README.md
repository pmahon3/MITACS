# Seeds (Applied)

Lab analogue of `notes/unsorted/` in
`Research/Mathematics/Resolvent_Framework`. A seed is an idea that
emerged from working with this codebase and is not yet vetted by any
audit gate.

## What lives here

- **Framework-level questions** prompted by empirical findings — what
  would it mean if X, given that we observed Y?
- **Experimental designs** that we'd like to run but haven't decided
  to commit to yet.
- **Cross-repo bridges** that name an object in the lab and ask
  whether the theory side has a name for it.

## What does NOT live here

- Implementation work — code in `scratch/` or `experiment/`.
- Memos about completed work — those go in
  `~/.claude/projects/.../memory/` per `CLAUDE.md`.
- Provenanced results — those use `experiment.provenance.make_result()`.
- Anything cleared by an audit — promoted out, see below.

## Seed lifecycle

```
notes/seeds/                          ← here; pre-audit
   ↓ applied audit (TBD; see seed §9 of any seed that needs it)
notes/active_questions/               ← (not yet created) — running
                                        empirical programme
   ↓ empirical evidence accrues
   either: result emitted via provenance, or seed parked

[if framework-level: separately]
   ↓ theory-side /audit pure
   in Research/Mathematics/Resolvent_Framework/notes/unsorted/
   → notes/active_leads/ if novel, or honest park
```

Lab and theory verdicts are independent. Seeds may be empirically
useful regardless of theory novelty.

## Filename convention

`<descriptive_object_name>.md`. Names should describe the *object* or
*question*, not brand it. Renaming after audit is normal and expected.

## Format (loose)

There is no rigid template. Existing seeds typically include:

- A "where this came from" section (empirical or theoretical
  provenance)
- The question or proposal in lab terms
- What it would predict if true / how it could fail
- A proposed empirical programme (ordered, with falsification
  criteria)
- An honest scope-limitation section (what it is NOT claiming)
- A lineage section listing the memories / commits / artifacts it
  builds on

The Resolvent_Framework `notes/unsorted/` README has a useful
taxonomy of seed types; we don't yet need that level of structure.
Revisit when the count grows.

## Current seeds

- [multiscale_factor_coherence.md](multiscale_factor_coherence.md) —
  treating the iterated 1-step κ_Q's failure as the visible part of
  a coherence-defect object across factors at multiple horizon
  scales; proposes direct h-step factor estimation with the residual
  diffusion as the cross-scale coherence constraint.
- [applied_audit_workflow.md](applied_audit_workflow.md) — proposes
  the lab-side analogue of the theory `/audit` workflow: six skills /
  agents (code-path, preregister, multiverse, framework, pre-experiment,
  arbiter) plus a shared registry and a single-session-momentum throttle.
  Grounded in the corpus review of project retraction patterns and a
  focused scan of the applied-research methodology literature.
