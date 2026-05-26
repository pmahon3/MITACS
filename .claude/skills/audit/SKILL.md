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
| `thread-coordinator` | `thread-coordinator` | At thread creation, amendment, state advancement, or closure |
| `thread` | full thread chain verification (all child experiments + thread.yaml + amendments) | When auditing a complete line of inquiry |
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
- `thread-coordinator` is the only agent that writes `thread.yaml`.
  Other agents (preregister, arbiter) READ it.
- Registry entries (`notes/preregistrations/<date>_<topic>/`) are
  append-only and hash-stamped. Do not edit existing entries; emit
  a new entry that references the prior one. EXCEPTION: thread.yaml
  is append-internal — amendments are hash-chained blocks within
  the same file, written by `thread-coordinator` only.
- When in doubt, gate-fail. The cost of a re-audit is far less than
  the cost of a retracted claim.

## Thread mode (`/audit thread <slug>`)

Full chain verification across an entire line of inquiry. Reads
`<slug>-thread/thread.yaml` and walks every child experiment:

1. Verify the thread.yaml itself (schema + hash chain + amendment
   chain).
2. For each `node` with `phase_a_path` populated: verify the linked
   phase_a.yaml exists, its `thread.thread_body_sha256` matches the
   thread's body hash at the time of phase_a registration (or a
   prior amendment hash), its `thread.node_id` matches the node,
   its `thread.parent_branch` matches the thread's
   `branching_rules` for that node.
3. For each settled experiment: verify the arbiter.yaml's
   `thread_state_update.branch_fired` is consistent with
   `branching_rules[node_id]`, and the resulting `next_node_id`
   matches the thread's pointer history.
4. Verify amendment chain: each amendment's `prior_tree_hash`
   matches the previous amendment's resulting hash (or the
   thread's original-creation hash).

Output: PASS or one of FAIL_SCHEMA / FAIL_CONSISTENCY /
FAIL_AMENDMENT_CHAIN with specific details.
