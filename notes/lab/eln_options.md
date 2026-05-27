# ELN options for the MITACS lab

**Date:** 2026-05-27
**Status:** research report — no tools installed, nothing scaffolded
**Scope:** session-by-session "lab notebook" that coexists with the
existing markdown + LaTeX + YAML-registry + git + Claude Code workflow

---

## TL;DR

Recommend **Option A: plain markdown under `notes/lab/YYYY-MM-DD.md`**
with a YAML-frontmatter template and a thin Python scaffolder
(`scripts/new_lab_entry.py`). It matches the discipline already in
place (markdown + frontmatter + git + audit registry) and adds zero
dependencies. Concrete setup steps are in §4.

The popular-in-research GUI options (Obsidian / Logseq / Foam /
Dendron) and the web-app ELNs (eLabFTW / SciNote / Chemotion) are
both **anti-patterns** for this workflow — see §5.

---

## 1. Summary table

| # | Option | Format | Tool surface | Fit with current stack | Setup cost | Maintenance cost |
|---|---|---|---|---|---|---|
| **A** | **Plain markdown + template** under `notes/lab/` | `.md` + YAML frontmatter | none beyond `$EDITOR` + git | **Excellent** — identical to `notes/seeds/`, `notes/literature/`, `notes/preregistrations/` | none | none |
| **B** | A + Python scaffolder (`scripts/new_lab_entry.py`) | as A | one ~80-line script | **Excellent** — matches `experiment/audit/registry.py` tooling pattern | ~1 hour to write | trivial |
| **C** | `jrnl` CLI over `notes/lab/` | `.md`-exportable, native txt | Python CLI (pipx-installable) | Good — adds search/tag layer over (A); native tags + date queries | 5 min (`pipx install jrnl`) | low |
| **D** | Extend `experiment.audit.registry` with `session_log` schema | hash-stamped YAML | extension of existing CLI | Tightest possible integration; significant engineering | half-day | medium (own code) |
| **E** | Quarto `.qmd` per session | markdown + executable cells | `quarto` CLI (separate install) | Mismatch — built for rendered reports, not append-only logs; overlaps with `writeup/tex/` | 10 min install + learning | medium |
| **F** | marimo notebook per session | pure-Python `.py` | `pip install marimo` | Mismatch — reactive *analysis* notebook, not a journal | 5 min install | medium |
| **G** | Obsidian (vault = repo) with `obsidian-git` | `.md` (Obsidian flavour) | desktop GUI app + plugins | Mismatch — adds GUI dependency; bidirectional links duplicate registry's job | 30 min | low (GUI clicks) |
| **H** | eLabFTW / SciNote (web app, self-hosted) | DB-backed | Docker + MySQL + browser | **Anti-pattern** for a solo + git workflow | hours; ongoing admin | high |

Detailed write-ups of the top three follow.

---

## 2. Option A — Plain markdown under `notes/lab/` (recommended)

### What

A flat directory of dated markdown entries:

```
notes/lab/
  _template/
    session_entry.md
  README.md
  2026-05-27.md
  2026-05-28_q2-rerun.md
  ...
```

One file per session (or per day if multi-session). Filenames:
`YYYY-MM-DD.md` for the day's primary entry, optional
`YYYY-MM-DD_<slug>.md` for additional sessions on the same day.

Each entry carries a YAML frontmatter header that lists the
cross-refs (commits, preregistrations, memory files, settled
findings, deferred items). The body is free-form markdown.

### Why it fits

- The repo's other notebooks (`notes/seeds/`, `notes/literature/`,
  `notes/preregistrations/`) are already exactly this shape.
  Adding `notes/lab/` is a **convention extension**, not a new
  system.
- Git is already the append-only record. Commits per session are
  the natural unit. The entry is the human-readable index to that
  commit (or those commits).
- Frontmatter matches the memory-file convention in
  `~/.claude/projects/.../memory/` — Claude Code can read these
  next session and reconstruct context.
- Grep + ripgrep already work. No new query language.
- The applied-audit workflow's hash-stamping happens in the
  registry, not here — the lab entry just *references* registry
  IDs, it does not duplicate them.

### Proposed frontmatter schema

```yaml
---
date: 2026-05-27
session_id: 2026-05-27-a       # YYYY-MM-DD-{a,b,c} if multi-session
duration_hours: 4.5             # optional, self-reported
focus:                          # 1-3 phrases, what the session was about
  - q2-mechanical-advancement
  - thread-amendment-distributional-class
commits:                        # SHAs touched this session
  - c6487ec
  - 54d39e7
preregistrations_touched:       # paths under notes/preregistrations/
  - 2026-05-27_p1-patra-sen-per-stratum-localization
memory_files_touched:           # filenames under ~/.claude/.../memory/
  - mitacs-q1-student-t-vs-gaussian.md
  - mitacs-rebaseline-facts.md
writeups_touched:               # paths under writeup/tex/
  - writeup/tex/missing_content_memo.tex
cross_programme_refs:           # outside-repo paths if relevant
  - ~/Research/Mathematics/Resolvent_Framework/notes/unsorted/disintegration_diagnostic.md
settled_this_session: []        # registry IDs of arbiter-rendered SETTLED
dispatched_this_session: []     # registry IDs of preregistrations newly created
deferred_to_next_session:       # one-liners the next session should resume from
  - "Run Q2D after thread amendment of R-D criterion"
status_at_end: in-progress      # in-progress | wrapped | blocked
---

# Session 2026-05-27 — Q2 mechanical advancement

## What happened

(free-form prose; the index above is the navigable map)

## Decisions

## Open questions

## Next session entry point
```

### Setup steps

1. `mkdir -p notes/lab/_template`
2. Write `notes/lab/_template/session_entry.md` with the schema above
3. Write `notes/lab/README.md` describing the convention (mirrors
   the README pattern in `notes/preregistrations/`)
4. Add a one-line entry in `CLAUDE.md` so the next Claude Code
   session knows to write here

### What a session entry looks like

See the frontmatter block above plus three short prose sections.
A typical entry is 30–80 lines including frontmatter — small enough
to scan, structured enough to grep.

### Honest assessment

This is what the user is already doing implicitly in the commit
graph + memory files; making it a first-class artifact costs
nothing and pays off whenever a session ends mid-thread and the
next one needs an entry point. Failure mode: skipping entries.
Mitigation: make the scaffolder (Option B) one command.

---

## 3. Option B — A + a Python scaffolder (`scripts/new_lab_entry.py`)

### What

A small Python script that prefills today's entry:

```
$ .venv/bin/python scripts/new_lab_entry.py
Created notes/lab/2026-05-27.md
Prefilled:
  commits since last entry: 7
  active preregistrations: 3 (under notes/preregistrations/)
  memory files modified since last entry: 2
Open in $EDITOR? [Y/n]
```

Implementation sketch (no need to write this now):
- Find the last `notes/lab/*.md` by date; read its `date:` field
- `git log --since=<date> --pretty=%h` → seed `commits:`
- List `notes/preregistrations/*/state.yaml` where status == active
  (call `python -m experiment.audit.registry list` and parse)
- `ls -lt ~/.claude/projects/.../memory/*.md` filtered by mtime >
  last entry's date → seed `memory_files_touched:`
- Drop in the template; open in `$EDITOR` (respect `$VISUAL` too)

### Why it fits

- Matches the existing audit tooling pattern: `experiment.audit.
  registry` is exactly this kind of "thin CLI over markdown +
  YAML" tool. The user is comfortable with it.
- Eliminates the only friction in Option A (remembering to
  manually copy the template and find what changed).
- Stays inside the project venv. No new top-level dep.

### Setup steps

1. Adopt Option A first
2. Write ~80 lines of Python under `scripts/new_lab_entry.py`
3. Optional: alias `lab` to `.venv/bin/python scripts/new_lab_entry.py`

### Honest assessment

Worth building once Option A has been used for a week and you know
which frontmatter fields actually get populated. Premature to
build before that — you'll guess the schema wrong. Same failure
mode as A (skipping entries) but with one less excuse.

---

## 4. Option C — `jrnl` CLI over `notes/lab/` (lightest alternate)

### What

[`jrnl`](https://jrnl.sh/) is an actively-maintained Python CLI for
timestamped journaling (v4.3 released Feb 2026,
[releases page](https://github.com/jrnl-org/jrnl/releases)). It
stores entries as plain text, supports inline `@tags`, searches by
date / tag / content, and **exports natively to markdown grouped by
year/month**.

You configure `jrnl` to point at `notes/lab/` and use it for
quick-fire timestamped notes during a session; at session end you
`jrnl --export markdown` and commit. Or use it as the entry-creation
front-end on top of the Option A template.

### Why it might fit

- Pure CLI, Python, runs on macOS via `pipx install jrnl`
  ([PyPI](https://pypi.org/project/jrnl/))
- Adds the one thing plain markdown lacks: easy
  date-range + tag search (`jrnl -from "2026-05-20" @q1 --short`)
- Doesn't fight the rest of the stack — output is markdown-exportable

### Why it might not

- Native storage format is its own `.txt`/`.yaml` blob with one
  long file per journal; exporting to per-day `.md` is a
  post-processing step, not the native unit. Means an extra
  workflow step that Option A doesn't have.
- Tag search is nice but `rg -e '@q1' notes/lab/` already does it
- Adds a dep to manage outside the project venv

### Setup steps

1. `pipx install jrnl` (uses your system Python, not the project venv)
2. `jrnl --config-file ~/.config/jrnl/mitacs.yaml` and point
   `journals.default.journal` at a directory under `notes/lab/`
3. Use `jrnl @tag "entry text"` from any terminal; export at session end

### Honest assessment

Real risk this becomes a *second* place to look for session notes,
not a better one. If you find yourself wanting quick timestamped
fragments during long sessions (rather than one block at the end),
add `jrnl` later as a fragment-collection layer. Otherwise skip.

---

## 5. Anti-patterns (don't do these, and why)

### Obsidian / Logseq / Foam / Dendron — GUI PKM apps

All four are popular in research circles and all four are wrong
for *this* workflow:

- **They add a GUI dependency** to a stack that currently has none.
  Every other artifact (registry, memory, writeups) is editable
  from any text editor and inspectable by Claude Code without
  launching an app.
- **Bidirectional links duplicate what the registry already does.**
  The hash-chained YAML registry (`experiment.audit.registry`) is
  the lab's reference graph. A second graph layer in Obsidian
  splits the truth.
- **[Dendron is in maintenance-only mode](https://github.com/dendronhq/dendron)** as of late 2024 / early 2025 — active development has stopped. Avoid.
- **Foam** is fine as a VS Code extension but its main value-add
  is the wiki-link graph, which (see above) duplicates work.
- **Obsidian** is the best of the four if a GUI is wanted — it has
  an active git plugin
  ([Vinzent03/obsidian-git](https://github.com/Vinzent03/obsidian-git))
  and a domain-specific scientific-research vault template
  ([LalieA/obsidian-scientific-research-vault](https://github.com/LalieA/obsidian-scientific-research-vault)). But the user hasn't asked for a GUI and the
  workflow doesn't need one.
- **Logseq** has nice built-in daily-journals but uses a block-
  outline model that fights with the prose+YAML shape of every
  other artifact in this repo.

There is also [`fcskit/obsidian-eln`](https://github.com/fcskit/obsidian-eln) — an Obsidian-based ELN template with 12+ templates for chemistry-instrument-data import (BioLogic, ZEISS SmartSEM, Horiba). Domain mismatch; the MITACS work is statistical, not wet-lab.

### eLabFTW / SciNote / Chemotion / LabFolder — web-app ELNs

[**eLabFTW**](https://github.com/elabftw/elabftw) is the canonical
open-source ELN; [**SciNote**](https://github.com/scinote-eln/scinote-web) is the other popular one. Both require:

- Docker + MySQL (eLabFTW) or Docker + Postgres + Ruby on Rails
  (SciNote)
- A web server to host
- An admin who maintains them ("regularly applying updates,
  configuring the backups properly, and hardening the host
  operating system" — direct from eLabFTW docs)
- A browser-based GUI workflow that doesn't compose with git or
  with Claude Code's text-based introspection

These exist to give wet-lab teams of 5–50 a shared, audited,
multi-user record of physical experiments with sample tracking,
reagent inventories, and regulatory compliance. The MITACS work
is a solo statistician's research log with a git-tracked codebase
and a hash-chained YAML registry already doing the audit job.
**Pure mismatch.** SciNote and Chemotion are additionally
biology- and chemistry-specific.

### Quarto / marimo / Jupyter — wrong tool, right domain

[Quarto](https://quarto.org/) and
[marimo](https://github.com/marimo-team/marimo) are excellent and
both run cleanly on macOS. They're for **executable, reproducible
analyses with prose** — i.e. for an *experiment script that
renders to a report*, not for a *session log*. The repo already
has `scratch/` for exploratory analysis and `writeup/tex/` for
rendered output; a session log sits between those, not alongside.

If you ever want to write up a single experiment as a rendered
HTML/PDF with embedded plots (e.g. an extension of `processing/
innovations/validation/rebaseline.py`'s output), Quarto would be
the right tool. That is a different need from this one.

### Jekyll-based lab notebooks — about publishing, not authoring

[`tlnagy/jekyll-lab-notebook`](https://github.com/tlnagy/jekyll-lab-notebook),
[`fdschneider/jekyll-lablog`](https://github.com/fdschneider/jekyll-lablog),
and [`fredpdavis/mdlabbook`](https://github.com/fredpdavis/mdlabbook) are
static-site generators for **publishing** an existing markdown lab
notebook as a browsable HTML site. None of them helps with the
authoring side, which is what the user is asking about. `mdlabbook`
has 5 total commits and looks abandoned. Revisit only if the user
later wants a public-facing version of the notebook.

### `LARG/eln` and other "lab notebook" GitHub finds

[`LARG/eln`](https://github.com/LARG/eln) (UT Austin Learning Agents Research Group) and similar group-specific tools are typically built around one lab's specific needs and don't transfer cleanly. Skim if curious; don't adopt.

---

## 6. Recommendation

**Do Option A this week. Add Option B (the scaffolder) once
you've written 3-5 entries by hand** and know which frontmatter
fields you actually populate vs leave blank.

Concrete first-step instructions (the user can execute these by
hand; do not let Claude scaffold them yet — the user asked for the
report only):

1. `mkdir -p notes/lab/_template`
2. Create `notes/lab/_template/session_entry.md` with the
   frontmatter schema from §2
3. Create `notes/lab/README.md` mirroring the pattern of
   `notes/preregistrations/README.md` — one paragraph on
   convention, one paragraph on naming, one paragraph on the
   relationship to the registry and to memory files
4. Add to `CLAUDE.md` under a new `## Lab notebook` section: one
   paragraph saying "session entries live in `notes/lab/YYYY-MM-DD.md`;
   each entry references registry IDs and memory filenames; entries
   are committed alongside the rest of the session's work"
5. Write the first entry by hand for the next session to validate
   the schema

After a week or two:

6. Write `scripts/new_lab_entry.py` (Option B) — small CLI that
   computes the "what changed since last entry" frontmatter
   fields and opens the new file in `$EDITOR`. Pattern after
   `experiment/audit/registry.py`.

Do **not** install jrnl, Obsidian, or any other tool until A+B has
been used for a month and a concrete unmet need has surfaced.
There is no need that isn't met by plain markdown + git here.

---

## 7. What I checked and surprises

**Verified actively maintained (2025–2026):**
- [`jrnl` v4.3, Feb 2026](https://github.com/jrnl-org/jrnl/releases)
- [marimo (~15k stars, monthly releases)](https://github.com/marimo-team/marimo)
- [Quarto (Posit/RStudio backing)](https://quarto.org/)
- [Foam (VS Code extension updated within last month)](https://github.com/foambubble/foam)
- [Obsidian-git plugin (active)](https://github.com/Vinzent03/obsidian-git)
- [eLabFTW (active; v5.3.11 docs current)](https://github.com/elabftw/elabftw)
- [SciNote (active)](https://github.com/scinote-eln/scinote-web)

**Verified abandoned or in maintenance-only:**
- [Dendron — maintenance-only, active dev stopped](https://github.com/dendronhq/dendron)
- [`fredpdavis/mdlabbook` — 5 total commits, no recent activity](https://github.com/fredpdavis/mdlabbook)
- [`tlnagy/jekyll-lab-notebook` — README lists development as "TODO"; limited recent activity](https://github.com/tlnagy/jekyll-lab-notebook)

**Surprises:**
- The **Royal Society of Chemistry's "GitHub as an open ELN"** paper
  ([DOI 10.1039/D3DD00032J](https://pubs.rsc.org/en/content/articlehtml/2023/dd/d3dd00032j)) is a serious peer-reviewed endorsement of exactly the
  pattern in Option A: markdown entries, git as record, GitHub
  issues as experiment pages. This is corroborating evidence that
  the plain-markdown path is a legitimate research-community
  choice, not an under-engineered fallback.
- [**MadsenLab's open notebook**](https://notebook.madsenlab.org/labnotebook.html) is a fully worked example of the
  Jekyll-publishing flow on top of plain-markdown authoring —
  useful reference if §5's "publish the notebook later" path is
  ever taken. They split notes vs essays and use a Rakefile +
  GitHub Pages to publish.
- [`fcskit/obsidian-eln`](https://github.com/fcskit/obsidian-eln) is well thought through (12+ templates, Dataview integration, Python data importers) but is built for wet-lab instrument-data import. Mentioned for completeness; mismatched.
- I expected Quarto to be a closer fit than it is. It isn't,
  because the unit of work is a *session log* not a *rendered
  analysis*. Saving Quarto for the right job (per-experiment
  rendered writeups) is the better call.

---

## Sources

- [GitHub as an open electronic laboratory notebook (RSC, 2023)](https://pubs.rsc.org/en/content/articlehtml/2023/dd/d3dd00032j)
- [jrnl](https://jrnl.sh/) — [GitHub](https://github.com/jrnl-org/jrnl) — [PyPI](https://pypi.org/project/jrnl/) — [releases](https://github.com/jrnl-org/jrnl/releases)
- [marimo](https://marimo.io/) — [GitHub](https://github.com/marimo-team/marimo) — [for researchers](https://marimo.io/for-researchers)
- [Quarto](https://quarto.org/) — [GitHub](https://github.com/quarto-dev/quarto-cli)
- [Obsidian-git plugin](https://github.com/Vinzent03/obsidian-git)
- [LalieA/obsidian-scientific-research-vault](https://github.com/LalieA/obsidian-scientific-research-vault)
- [fcskit/obsidian-eln](https://github.com/fcskit/obsidian-eln)
- [Logseq](https://logseq.com/) — [community hub on journaling](https://hub.logseq.com/use-cases/1Sr4awszMQzD4GM5KvWim7/how-to-get-started-with-using-logseq-as-a-digital-journal/x3MzLx9XKMsfyDPm9yKNEn)
- [Foam](https://foambubble.github.io/foam/) — [GitHub](https://github.com/foambubble/foam)
- [Dendron](https://github.com/dendronhq/dendron) (maintenance-only)
- [eLabFTW](https://github.com/elabftw/elabftw) — [docs](https://doc.elabftw.net/) — [prerequisites](https://doc.elabftw.net/docs/install/prerequisites/)
- [SciNote](https://github.com/scinote-eln/scinote-web) — [open source](https://www.scinote.net/open-source-code/)
- [MadsenLab open notebook](https://notebook.madsenlab.org/labnotebook.html)
- [tlnagy/jekyll-lab-notebook](https://github.com/tlnagy/jekyll-lab-notebook)
- [fdschneider/jekyll-lablog](https://github.com/fdschneider/jekyll-lablog)
- [fredpdavis/mdlabbook](https://github.com/fredpdavis/mdlabbook)
- [GitHub topic: lab-notebook](https://github.com/topics/lab-notebook)
- [GitHub topic: electronic-lab-notebook](https://github.com/topics/electronic-lab-notebook)
