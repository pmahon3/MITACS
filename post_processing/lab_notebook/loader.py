"""Content loaders for the lab notebook Dash app.

Pure data access. No Dash imports, no UI logic. Functions here are
importable headless (useful both for the Dash callbacks and for
ad-hoc debugging from `.venv/bin/python -c`).

The corpus is small enough to load all four surfaces at app startup;
no file watchers, no incremental refresh. Re-launch the app to pick
up new files.

Content surfaces:

- **Session entries** under ``notes/lab/YYYY-MM-DD*.md`` — markdown
  with YAML frontmatter (the lab notebook itself).
- **Registry entries** under ``notes/preregistrations/<topic>/`` — a
  directory per topic containing phase_a/proponent/.../arbiter YAML
  files (and ``thread.yaml`` for threads). The existing
  ``experiment.audit.registry`` module supplies the YAML loading +
  schema-aware status string; this loader composes it into a per-
  entry record without reimplementing.
- **Seeds + literature** under ``notes/seeds/*.md`` and
  ``notes/literature/*.md`` — plain markdown.
- **Memory files** under ``~/.claude/.../memory/*.md`` — markdown
  with YAML frontmatter (``description``, ``metadata.type``). The
  directory may not exist on machines without Claude Code; this
  loader degrades to an empty list gracefully.
- **Writeup PDFs** under ``writeup/tex/*.tex`` paired with built
  ``*.pdf`` — page count via ``pdfinfo`` subprocess (degrades to
  ``None`` if pdfinfo is not installed).

All loader functions accept absolute ``Path`` arguments; the app
resolves paths from ``config.PROJECT_ROOT`` before calling.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from experiment.audit import registry as _registry


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionEntry:
    """One ``notes/lab/YYYY-MM-DD*.md`` file."""

    path: Path
    session_id: str
    date: str
    frontmatter: dict[str, Any]
    body: str

    @property
    def focus(self) -> list[str]:
        f = self.frontmatter.get("focus") or []
        return [str(x) for x in f]

    @property
    def status_at_end(self) -> str:
        return str(self.frontmatter.get("status_at_end", ""))

    @property
    def title(self) -> str:
        # First H1 in the body, or the session_id as fallback.
        for line in self.body.splitlines():
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()
        return self.session_id


@dataclass(frozen=True)
class RegistryEntry:
    """One ``notes/preregistrations/<topic>/`` directory."""

    path: Path
    name: str
    is_thread: bool
    status: str  # one-word from registry.entry_status (SETTLED / THREAD:active / ...)
    # Files present (the ones registry semantics knows about).
    files_present: list[str]
    # If thread: thread_state() output. Else: per-file YAML data (small;
    # OK to keep in memory).
    thread_state: dict[str, Any] | None
    artifact_data: dict[str, dict[str, Any]]
    verdict_summary: str | None  # short string for arbiter outcome if SETTLED
    # Optional ``note.md`` — exposition prose with MathJax math attached
    # to this entry. Written by the arbiter (for experiments) or the
    # thread-coordinator (for threads); unpinned (no body_sha256), so
    # may be re-edited without re-stamping the load-bearing YAMLs. The
    # YAML artifacts are the registered claims; the note is exposition.
    note_md: str | None = None


@dataclass(frozen=True)
class MemoryFile:
    """One memory markdown file (``~/.claude/.../memory/*.md``)."""

    path: Path
    name: str  # filename without extension
    description: str  # from frontmatter, or first body line
    metadata: dict[str, Any]
    body: str
    archived: bool = False  # True if file is under memory/_archive/


@dataclass(frozen=True)
class NoteFile:
    """A seed or literature markdown file."""

    path: Path
    category: str  # 'seed' | 'literature'
    title: str
    body: str


@dataclass(frozen=True)
class WriteupPDF:
    """A LaTeX source + built PDF pair under ``writeup/tex/``."""

    tex_path: Path
    pdf_path: Path | None  # None if not built
    name: str  # filename stem
    page_count: int | None  # from pdfinfo; None if unavailable
    size_bytes: int | None


@dataclass(frozen=True)
class SearchHit:
    """One substring-match in some content surface."""

    source_kind: str  # 'session' | 'registry' | 'memory' | 'note' | 'writeup'
    source_id: str  # path stem or registry topic name
    path: Path
    snippet: str  # ~120 char window around the match
    match_count: int


@dataclass
class LoadedCorpus:
    """Everything the app needs to render. Loaded once at startup."""

    sessions: list[SessionEntry] = field(default_factory=list)
    registry: list[RegistryEntry] = field(default_factory=list)
    notes: list[NoteFile] = field(default_factory=list)
    memory: list[MemoryFile] = field(default_factory=list)
    writeups: list[WriteupPDF] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Frontmatter parsing (markdown with YAML header between '---' lines)
# ---------------------------------------------------------------------------


_FRONTMATTER_RE = re.compile(
    r"\A---\s*\n(?P<fm>.*?)\n---\s*\n(?P<body>.*)\Z",
    re.DOTALL,
)


def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Extract a YAML frontmatter block from a markdown file.

    Returns ``({}, text)`` if no frontmatter is present. Parse errors
    fall back to an empty dict (we still want to display the body).
    """
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    fm_text = m.group("fm")
    body = m.group("body")
    try:
        fm = yaml.safe_load(fm_text) or {}
    except yaml.YAMLError:
        fm = {}
    if not isinstance(fm, dict):
        fm = {}
    return fm, body


# ---------------------------------------------------------------------------
# Session entries (notes/lab/)
# ---------------------------------------------------------------------------


_SESSION_FILENAME_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})(?P<suffix>[-_].*)?\.md$"
)


def load_session_entries(lab_dir: Path) -> list[SessionEntry]:
    """Read every ``YYYY-MM-DD*.md`` under ``lab_dir``.

    Sorted ascending by ``date`` then by filename suffix (so multi-
    session days come out in -a, -b, -c order). Non-matching files
    (e.g. README.md, eln_options.md) are skipped.
    """
    if not lab_dir.is_dir():
        return []

    entries: list[SessionEntry] = []
    for p in sorted(lab_dir.iterdir()):
        if p.is_dir():
            continue
        m = _SESSION_FILENAME_RE.match(p.name)
        if not m:
            continue
        text = p.read_text(encoding="utf-8")
        fm, body = _split_frontmatter(text)
        date = str(fm.get("date") or m.group("date"))
        session_id = str(fm.get("session_id") or p.stem)
        entries.append(
            SessionEntry(
                path=p,
                session_id=session_id,
                date=date,
                frontmatter=fm,
                body=body,
            )
        )
    return entries


# ---------------------------------------------------------------------------
# Registry entries (notes/preregistrations/)
# ---------------------------------------------------------------------------


# Standard artifact filenames a topic directory may contain. Order
# matters for the "files_present" display.
_REGISTRY_ARTIFACTS = (
    "thread.yaml",
    "phase_a.yaml",
    "phase_b.yaml",
    "proponent.yaml",
    "devils_advocate.yaml",
    "multiverse.yaml",
    "result.yaml",
    "arbiter.yaml",
    "phase_fidelity_check_r.yaml",
    "phase_fidelity_check_p.yaml",
    "phase_fidelity_check_t.yaml",
)


def _verdict_summary(arbiter_data: dict[str, Any]) -> str:
    """One-line summary of an arbiter verdict (mech + substantive, or
    just verdict if both fields aren't present)."""
    num = arbiter_data.get("numerical_record") or {}
    mech = num.get("outcome_mechanical")
    subst = num.get("outcome_substantive")
    verdict = arbiter_data.get("verdict")
    fs = arbiter_data.get("finding_status")
    parts = []
    if fs:
        parts.append(str(fs))
    if verdict and verdict != fs:
        parts.append(str(verdict))
    if mech and subst and mech == subst:
        parts.append(f"({mech})")
    elif mech or subst:
        parts.append(f"(mech={mech}, subst={subst})")
    return " ".join(parts) if parts else ""


def load_registry_entries(prereg_dir: Path) -> list[RegistryEntry]:
    """Read every topic directory under ``prereg_dir``.

    Defers to ``experiment.audit.registry`` for YAML parsing, status
    determination, and thread-state introspection. Reads each present
    artifact's YAML once (small corpus) so the components layer can
    render details without re-touching disk.
    """
    if not prereg_dir.is_dir():
        return []

    out: list[RegistryEntry] = []
    for d in sorted(prereg_dir.iterdir()):
        if not d.is_dir() or d.name.startswith("_") or d.name == "README.md":
            continue

        is_thread = _registry.is_thread_dir(d)
        try:
            status = _registry.entry_status(d)
        except Exception as exc:  # noqa: BLE001 — surface unexpected breakage as text
            status = f"ERROR:{type(exc).__name__}"

        files_present: list[str] = []
        artifact_data: dict[str, dict[str, Any]] = {}
        for fname in _REGISTRY_ARTIFACTS:
            fp = d / fname
            if not fp.exists():
                continue
            files_present.append(fname)
            try:
                artifact_data[fname] = _registry.read_yaml(fp)
            except (ValueError, yaml.YAMLError):
                artifact_data[fname] = {"__parse_error__": True}

        thread_state: dict[str, Any] | None = None
        if is_thread:
            try:
                thread_state = _registry.thread_state(d)
            except Exception as exc:  # noqa: BLE001
                thread_state = {"__error__": str(exc)}

        verdict = None
        arb = artifact_data.get("arbiter.yaml")
        if arb and "__parse_error__" not in arb:
            verdict = _verdict_summary(arb)

        # Optional exposition note (``note.md``). Read raw; no parsing.
        # Unpinned by design — see RegistryEntry.note_md docstring.
        note_md: str | None = None
        note_path = d / "note.md"
        if note_path.exists():
            try:
                note_md = note_path.read_text(encoding="utf-8")
            except OSError:
                note_md = None

        out.append(
            RegistryEntry(
                path=d,
                name=d.name,
                is_thread=is_thread,
                status=status,
                files_present=files_present,
                thread_state=thread_state,
                artifact_data=artifact_data,
                verdict_summary=verdict,
                note_md=note_md,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Seeds + literature
# ---------------------------------------------------------------------------


def _first_h1(text: str) -> str:
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("# "):
            return s[2:].strip()
    return ""


def load_notes(notes_dir: Path) -> list[NoteFile]:
    """Read every ``notes/seeds/*.md`` + ``notes/literature/*.md`` under
    ``notes_dir`` (which is the parent ``notes/`` directory).

    Excludes ``README.md``. Title is the first H1, or the filename stem
    if no H1 is present.
    """
    out: list[NoteFile] = []
    for sub, category in (("seeds", "seed"), ("literature", "literature")):
        sub_dir = notes_dir / sub
        if not sub_dir.is_dir():
            continue
        for p in sorted(sub_dir.iterdir()):
            if p.is_dir() or not p.name.endswith(".md") or p.name == "README.md":
                continue
            try:
                body = p.read_text(encoding="utf-8")
            except OSError:
                continue
            title = _first_h1(body) or p.stem
            out.append(
                NoteFile(path=p, category=category, title=title, body=body)
            )
    return out


# ---------------------------------------------------------------------------
# Memory files (~/.claude/.../memory/)
# ---------------------------------------------------------------------------


def load_memory_files(memory_dir: Path) -> list[MemoryFile]:
    """Read every ``*.md`` under ``memory_dir`` AND ``memory_dir/_archive/``.

    Files under ``_archive/`` are loaded with ``archived=True`` so the
    UI can group them separately. Returns an empty list if the
    directory does not exist (the app runs fine on a machine without
    Claude Code installed).
    """
    if not memory_dir.is_dir():
        return []

    out: list[MemoryFile] = []
    # Scan active dir + _archive subdir if present.
    scan_targets: list[tuple[Path, bool]] = [(memory_dir, False)]
    archive_dir = memory_dir / "_archive"
    if archive_dir.is_dir():
        scan_targets.append((archive_dir, True))

    for scan_dir, is_archived in scan_targets:
        for p in sorted(scan_dir.iterdir()):
            if p.is_dir() or not p.name.endswith(".md"):
                continue
            # README files are directory metadata, not memories.
            if p.stem.upper() == "README":
                continue
            try:
                text = p.read_text(encoding="utf-8")
            except OSError:
                continue
            fm, body = _split_frontmatter(text)
            description = str(fm.get("description") or _first_h1(body) or p.stem)
            metadata = fm.get("metadata") if isinstance(fm.get("metadata"), dict) else {}
            out.append(
                MemoryFile(
                    path=p,
                    name=p.stem,
                    description=description,
                    metadata=metadata,
                    body=body,
                    archived=is_archived,
                )
        )
    return out


# ---------------------------------------------------------------------------
# Writeups (writeup/tex/*.tex + .pdf)
# ---------------------------------------------------------------------------


def _pdf_page_count(pdf_path: Path) -> int | None:
    """Page count via ``pdfinfo``. Returns None if pdfinfo isn't
    installed or fails. Subprocess is short and non-interactive."""
    try:
        result = subprocess.run(
            ["pdfinfo", str(pdf_path)],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        if line.startswith("Pages:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except ValueError:
                return None
    return None


def load_writeup_pdfs(writeup_dir: Path) -> list[WriteupPDF]:
    """Pair every ``*.tex`` under ``writeup_dir`` with its sibling
    ``*.pdf`` if built. PDF metadata is read once at load time."""
    if not writeup_dir.is_dir():
        return []
    out: list[WriteupPDF] = []
    for tex in sorted(writeup_dir.glob("*.tex")):
        # Skip preamble-style includes (no standalone PDF).
        if tex.stem.startswith("_"):
            continue
        pdf = tex.with_suffix(".pdf")
        if pdf.exists():
            pages = _pdf_page_count(pdf)
            try:
                size = pdf.stat().st_size
            except OSError:
                size = None
            out.append(
                WriteupPDF(
                    tex_path=tex,
                    pdf_path=pdf,
                    name=tex.stem,
                    page_count=pages,
                    size_bytes=size,
                )
            )
        else:
            out.append(
                WriteupPDF(
                    tex_path=tex,
                    pdf_path=None,
                    name=tex.stem,
                    page_count=None,
                    size_bytes=None,
                )
            )
    return out


# ---------------------------------------------------------------------------
# Top-level load + search
# ---------------------------------------------------------------------------


def load_corpus(
    *,
    lab_dir: Path,
    prereg_dir: Path,
    notes_dir: Path,
    memory_dir: Path,
    writeup_dir: Path,
) -> LoadedCorpus:
    """One-shot loader the app calls at startup."""
    return LoadedCorpus(
        sessions=load_session_entries(lab_dir),
        registry=load_registry_entries(prereg_dir),
        notes=load_notes(notes_dir),
        memory=load_memory_files(memory_dir),
        writeups=load_writeup_pdfs(writeup_dir),
    )


def _make_snippet(haystack: str, needle_lower: str, window: int = 120) -> str:
    """Return a ~``window``-char window centred on the first match,
    with ellipses on the cut sides. Matches case-insensitively."""
    lower = haystack.lower()
    idx = lower.find(needle_lower)
    if idx < 0:
        return ""
    half = window // 2
    start = max(0, idx - half)
    end = min(len(haystack), idx + len(needle_lower) + half)
    snip = haystack[start:end].replace("\n", " ").strip()
    if start > 0:
        snip = "... " + snip
    if end < len(haystack):
        snip = snip + " ..."
    return snip


def search_content(query: str, corpus: LoadedCorpus) -> list[SearchHit]:
    """Case-insensitive substring search across all loaded surfaces.

    Returns hits sorted by ``match_count`` descending (most-relevant
    first by simple count). Empty query returns an empty list.
    """
    query = query.strip()
    if not query:
        return []
    q = query.lower()

    hits: list[SearchHit] = []

    def _scan(kind: str, src_id: str, path: Path, text: str) -> None:
        if not text:
            return
        count = text.lower().count(q)
        if count == 0:
            return
        snip = _make_snippet(text, q)
        hits.append(
            SearchHit(
                source_kind=kind,
                source_id=src_id,
                path=path,
                snippet=snip,
                match_count=count,
            )
        )

    for s in corpus.sessions:
        _scan("session", s.session_id, s.path, s.body)
    for r in corpus.registry:
        # Concatenate the textual content of all artifacts in the entry
        # (a topic-level search). Pretty-print each YAML deterministically
        # so substring search finds field-name and value occurrences.
        merged_parts = [r.name, r.status]
        for fname, data in r.artifact_data.items():
            try:
                merged_parts.append(yaml.safe_dump(data, sort_keys=False))
            except Exception:  # noqa: BLE001
                pass
        if r.note_md:
            merged_parts.append(r.note_md)
        _scan("registry", r.name, r.path, "\n".join(merged_parts))
    for n in corpus.notes:
        _scan(f"note:{n.category}", n.path.stem, n.path, n.body)
    for m in corpus.memory:
        _scan(
            "memory",
            m.name,
            m.path,
            f"{m.description}\n{m.body}",
        )
    for w in corpus.writeups:
        # Index .tex source as searchable text (the PDF itself is
        # opaque to us; the .tex is the searchable form).
        try:
            tex_text = w.tex_path.read_text(encoding="utf-8")
        except OSError:
            tex_text = ""
        _scan("writeup", w.name, w.tex_path, tex_text)

    hits.sort(key=lambda h: h.match_count, reverse=True)
    return hits
