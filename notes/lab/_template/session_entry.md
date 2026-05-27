<!--
Session entry template — copy this file to notes/lab/YYYY-MM-DD.md (or
YYYY-MM-DD-a.md, -b.md, ... for multi-session days) at the start of a
session. The frontmatter is the navigable index; the prose body is the
human-readable narrative. Append-only: never edit a prior entry except
to add a cross-reference at the bottom.

Frontmatter schema is the one specified in notes/lab/eln_options.md §2.
Optional fields may be omitted; required fields are marked below.
-->
---
date: YYYY-MM-DD                # REQUIRED
session_id: YYYY-MM-DD-a        # REQUIRED — append -a/-b/-c if multi-session
duration_hours: 0               # optional, self-reported
focus:                          # REQUIRED — 1-3 short phrases
  - phrase-one
commits: []                     # SHAs touched this session (short SHAs OK)
preregistrations_touched: []    # directory names under notes/preregistrations/
memory_files_touched: []        # filenames under ~/.claude/.../memory/
writeups_touched: []            # paths under writeup/tex/
cross_programme_refs: []        # outside-repo paths if relevant
settled_this_session: []        # registry IDs of arbiter-rendered SETTLED
dispatched_this_session: []     # registry IDs of preregistrations newly created
deferred_to_next_session: []    # one-liners the next session resumes from
status_at_end: in-progress      # in-progress | wrapped | blocked
---

# Session YYYY-MM-DD — short title

## What happened

(Free-form prose. Chronological narrative of the session. The
frontmatter above is the navigable index; this is the human
readable account.)

## Decisions

(Decisions locked in this session and their rationale.)

## Open questions

(Questions raised but not resolved.)

## Next session entry point

(One or more one-liners the next session should resume from, in
order of dependency. Mirror these into `deferred_to_next_session`
in the frontmatter.)
