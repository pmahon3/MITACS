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
- **Never edit a prior entry's body or frontmatter once written**,
  except to append a cross-reference note at the bottom (e.g.
  "→ continued in 2026-05-30.md after deferred items completed").
  The git history is the audit trail; treating entries as
  immutable matches the registry's hash-stamped artifact
  discipline.
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
