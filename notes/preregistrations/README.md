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

```
notes/preregistrations/
  README.md                       ← this file
  _template/                      ← stubs for each phase artifact
    phase_a.yaml
    phase_b.yaml
    proponent.yaml
    devils_advocate.yaml
    multiverse.yaml
    arbiter.yaml
    result.yaml
  <YYYY-MM-DD>_<topic-slug>/      ← one per claim, append-only
    phase_a.yaml                  ← (written first; before any data)
    devils_advocate.yaml          ← (written simultaneously with phase_a)
    proponent.yaml                ← (written simultaneously with phase_a)
    phase_b.yaml                  ← (written before touching test set)
    multiverse.yaml               ← (written when /audit multiverse runs)
    result.yaml                   ← (written after experiment runs)
    arbiter.yaml                  ← (written last; the only file that may
                                     declare SETTLED)
```

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
| `preregister` | `_template/phase_a.yaml`, `_template/phase_b.yaml` | `<topic>/phase_a.yaml`, `<topic>/phase_b.yaml` |
| `devils-advocate` | `<topic>/phase_a.yaml` | `<topic>/devils_advocate.yaml` |
| `multiverse` | `<topic>/phase_a.yaml`, `<topic>/phase_b.yaml` | `<topic>/multiverse.yaml` |
| `arbiter` | the entire `<topic>/` directory + the result artifact | `<topic>/arbiter.yaml` + memory update |

The proponent forecast is captured by `preregister` (inside `phase_a.yaml`'s
`hypothesis.proponent` field) AND in a separate `<topic>/proponent.yaml`
for symmetric grading against `devils_advocate.yaml`.

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

- *(none yet — registry just opened)*
