# notes/lab/ — session lab notebook

A flat directory of dated markdown entries, one per session. The
discipline mirrors the other markdown surfaces in the repo
(`notes/seeds/`, `notes/literature/`, `notes/preregistrations/`) —
plain markdown + YAML frontmatter + git as the append-only record.

The design rationale is in `eln_options.md` (the research report that
preceded this scaffolding). The short version: this is **Option A**
from that report — convention extension over the existing markdown
discipline, zero new dependencies.

## What lives here

- **`YYYY-MM-DD.md`** — one entry per session. The frontmatter
  is a navigable index of what the session touched (commits,
  preregistrations, memory files, writeups, cross-programme
  references, settled findings, dispatches, deferred items). The
  body is free-form markdown prose.
- **`_template/session_entry.md`** — the template to copy at the
  start of a session.
- **`eln_options.md`** — the research report that produced the
  Option-A recommendation; durable design document.

This directory does NOT hold experiments (those live in
`notes/preregistrations/`), seeds (those live in `notes/seeds/`),
literature scouts (those live in `notes/literature/`), or writeups
(those live in `writeup/tex/`). It indexes work across all of them.

## File-naming convention

- One session per day: `YYYY-MM-DD.md`.
- Multi-session days: `YYYY-MM-DD-a.md`, `YYYY-MM-DD-b.md`, … in
  chronological order. The `session_id` frontmatter field mirrors
  the filename suffix.
- Slugs after the date are permitted for thematic continuity
  (e.g. `YYYY-MM-DD-b_q2-rerun.md`) but the date-then-letter
  prefix is REQUIRED for sortability.

## Frontmatter schema

The full schema is in `_template/session_entry.md` (verbatim from
`eln_options.md` §2). Required fields:

- `date`, `session_id`, `focus`, `status_at_end`

The remaining fields are convention-required (write them when
applicable, leave as empty list when not); they are how the next
session locates this one's context.

## Append discipline

- **One entry per session.** Sessions are the natural unit (not
  days, not commits).
- **Body**: write-once; subsequent additions go in a clearly-
  marked `## Continuation: <topic>` section appended below the
  original body. The original prose stays untouched. The git
  history is the audit trail.
- **Frontmatter**: MAY be refreshed via
  `notes/lab/tools/lab_note.py --refresh <path>`. The helper
  rewrites only the machine-derivable fields (`commits`,
  `preregistrations_touched`, `memory_files_touched`,
  `writeups_touched`, `settled_this_session`,
  `dispatched_this_session`) from git + registry state since
  the session's first commit. The helper does NOT touch
  `focus`, `cross_programme_refs`, `deferred_to_next_session`,
  or `status_at_end` — those are judgment fields the human owns.
  Subject-line annotations on commits (e.g., em-dashes,
  parenthetical hashes) are clobbered by `--refresh` — the
  frontmatter is the *index*; rich narrative belongs in the
  body.
- **Cross-reference** at the bottom of an entry when a future
  session resumes or supersedes it (e.g. "→ continued in
  2026-05-30.md after deferred items completed"). This is
  always permitted regardless of body-immutability.
- **Status discipline:**
  - `status_at_end: wrapped` — session ended at a natural
    stopping point; deferred items are real next-actions, not
    abandoned threads.
  - `status_at_end: in-progress` — session ended mid-stream;
    the next session is expected to continue without an
    intervening day.
  - `status_at_end: blocked` — session ended because something
    external (dependency, dependency on another agent, data
    unavailable) blocked progress; the deferred list should
    identify what would unblock.

## Tooling

`notes/lab/tools/lab_note.py` is a frontmatter helper. It owns
nothing the agents (preregister / arbiter / thread-coordinator)
already own and nothing the human writes (the body). It owns
the *index* — the machine-derivable fields that go stale as the
session continues past the entry's initial write.

```
# Scaffold a new entry with frontmatter pre-populated:
.venv/bin/python notes/lab/tools/lab_note.py --new 2026-05-28
.venv/bin/python notes/lab/tools/lab_note.py --new 2026-05-28-b   # multi-session day

# Refresh an existing entry's frontmatter (rewrites the index,
# preserves the body verbatim):
.venv/bin/python notes/lab/tools/lab_note.py --refresh notes/lab/2026-05-28.md

# Preview what --refresh would change without writing:
.venv/bin/python notes/lab/tools/lab_note.py --diff notes/lab/2026-05-28.md
```

The session-boundary detection uses the existing `commits`
field's first SHA as the start of the session window (parent of
that commit). When `commits` is empty (a freshly scaffolded
entry), it falls back to the commit that first added the file
itself. This lets `--refresh` work correctly even when several
sessions occur on the same calendar day or when an entry was
written mid-session.

Why a helper rather than an agent: the lab-note frontmatter is
mechanical bookkeeping over git + registry state. Agents earn
their dispatch overhead when they exercise judgment (preregister
chooses cuts; arbiter renders verdicts; thread-coordinator
amends trees). The frontmatter index does not. The agent-per-
artifact pattern is preserved at the level it makes sense:
agents own the hash-stamped registry artifacts they emit; the
human owns the body prose; this helper owns the derivable index.

## Workflow ownership map

| Artifact | Owner | When written |
|---|---|---|
| `phase_a.yaml`, `proponent.yaml`, `devils_advocate.yaml` | `preregister`, `devils-advocate` agents | At experiment pre-registration |
| `arbiter.yaml` + per-experiment `note.md` | `arbiter` agent | After Check R passes |
| `thread.yaml` + thread `note.md` | `thread-coordinator` agent (sole writer) | Thread creation / amendment / state-advancement / closure |
| `phase_fidelity_check_*.yaml` | `phase-fidelity` agent | Before run + before arbiter |
| Memory files (`~/.claude/.../memory/*.md`) + `MEMORY.md` | **Human** | After arbiter declares `memory_update_required: true` (the user batches; agents identify the target but don't write) |
| `CLAUDE.md` | **Human** | Discretionary; usually at settled-finding closure |
| Lab note **body** | **Human** | At session start (template) + during/after session (prose) |
| Lab note **frontmatter** | `lab_note.py --new` / `--refresh` helper + human (for judgment fields) | At session start (scaffold) + at session end (refresh) |

## Relationship to other content surfaces

| Surface | What it carries | Lab-notebook role |
|---|---|---|
| `notes/preregistrations/` | hash-stamped registry artifacts (phase_a, proponent, devils_advocate, result, arbiter, thread.yaml) | Lab entry references by directory name in `preregistrations_touched` / `settled_this_session` / `dispatched_this_session`. |
| `~/.claude/.../memory/*.md` | persistent context across Claude Code sessions, with `description` line indexed in `MEMORY.md` | Lab entry references by filename in `memory_files_touched`; new memories typically arise from a session. |
| `writeup/tex/*.tex` | papers + memos under active development | Lab entry references in `writeups_touched`. |
| `notes/seeds/*.md` | unvetted ideas that emerged from the work | Seeds may motivate a session; sessions may produce new seeds. |
| `notes/literature/*.md` | literature scouts for specific topics | Lab entries reference scouts that informed the session's direction. |
| Cross-programme paths | `~/Research/Mathematics/Resolvent_Framework/notes/...` etc. | Lab entry references in `cross_programme_refs` when a theory-side artifact was consulted or updated. |

## Why not extend `experiment/audit/registry.py`?

Considered (Option D in `eln_options.md`). Rejected because:

- The registry hash-chain discipline serves tamper-evidence for
  CLAIM-grade artifacts; session entries are commentary, not
  claims.
- Append-once-then-immutable matches git's natural shape; the
  hash stamping would be ceremony without benefit.
- The lab notebook indexes the registry, so a circular-import
  shape would be uncomfortable.

If the lab entries ever produce claim-grade content (they
shouldn't — claims belong in registry artifacts), the dispatch
mechanism is "open a preregistration."

## Browsing

The Dash app at `post_processing/lab_notebook/` renders this
directory + the registry + memory + writeups + seeds + literature
as a single browser. Launch:

```
.venv/bin/python -m post_processing.lab_notebook.app --port 8050
```

The lab notebook is read-only in the app; editing happens in
`$EDITOR` + git as usual.
