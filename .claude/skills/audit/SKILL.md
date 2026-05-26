---
name: audit
description: Applied research audit dispatcher. Gates the lab pipeline at design, pre-experiment, post-experiment, and finding-promotion checkpoints.
arguments: [mode, target]
disable-model-invocation: true
allowed-tools: WebSearch WebFetch Read Glob Grep Bash
model: opus
---

# Applied Research Audit: $2 (mode: $1)

You are dispatching one or more applied-audit agents for the MITACS
lab. Mode determines which agent(s) to invoke. Each agent's contract
is in `.claude/agents/<name>.md`; consult it before invocation.

This skill differs from the theory-side `/audit pure | applied` in
`Research/Mathematics/Resolvent_Framework` — that one is novelty-
oriented (adversary is the field). This one is correctness-oriented
(adversary is the analyst's own future self). Threat model and
design are in `notes/seeds/applied_audit_workflow.md`.

---

## Modes

| Mode | Invokes | When |
|------|---------|------|
| `code-path` | `code-path-auditor` | Before any memory or writeup cites an empirical result |
| `preregister` | `preregister` | Before an experiment that will produce a citable finding |
| `multiverse` | `multiverse` | After a parameter sweep produces a "best" choice |
| `framework` | `framework-escalator` | When `multiverse` returns cosmetic |
| `pre-experiment` | `pre-experiment-checklist` | Before submitting a SLURM / local job that writes results |
| `arbiter` | `arbiter` | After experiment results in, before memory cites them as settled |
| `devils-advocate` | `devils-advocate` | At any decision checkpoint; produces pre-registered counter-prediction |
| `editorial` | `editorial-pass` | Before any writeup commit |
| `literature` | `literature-scout` | When a seed needs prior-art context, or a finding needs citation |
| `full` | runs `code-path` → `preregister` → `pre-experiment` → `arbiter` in sequence | Use for a claim-grade result going into the writeup or `make_result()` |

---

## Output format

```
## Audit: [target] (mode: [mode])

### Verdict: [PASS / FAIL / PROVISIONAL / ESCALATE]
[one-sentence summary]

### Per-agent results
[each invoked agent's output, in invocation order]

### Recommendation
[next workflow step — proceed, gate-fail, escalate to framework
audit, mark SESSION-PROVISIONAL, etc.]

### Registry entry
[path to the notes/preregistrations/ artifact created or updated]
```

---

## Rules

- A `code-path` failure is BLOCKING: no finding may be cited without
  resolution.
- `multiverse` "cosmetic" verdicts MUST escalate to `framework`.
- `arbiter` is the only agent that may write "SETTLED" status on a
  finding; all others write "PROVISIONAL" or leave status unset.
- Registry entries (`notes/preregistrations/<date>_<topic>/`) are
  append-only and hash-stamped. Do not edit existing entries; emit
  a new entry that references the prior one.
- When in doubt, gate-fail. The cost of a re-audit is far less than
  the cost of a retracted claim.
