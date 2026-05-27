"""MITACS Lab Notebook — unified Dash browser over the lab's existing
content surfaces.

Read-only navigation + simple full-text search across:

- ``notes/lab/YYYY-MM-DD*.md`` — the session lab notebook
- ``notes/preregistrations/<topic>/`` — hash-stamped registry
  artifacts (experiments + threads)
- ``notes/seeds/*.md`` + ``notes/literature/*.md`` — exploratory and
  scout notes
- ``~/.claude/projects/.../memory/*.md`` — Claude Code memory
- ``writeup/tex/*.tex`` + their built ``*.pdf`` — papers / memos

Launch::

    .venv/bin/python -m post_processing.lab_notebook.app [--port 8050]

Mirrors the ``post_processing.operator_dashboard`` shape (app + loader
+ components split; loaders pure, components stateless, the app is the
only Dash-aware module). MVP scope: 4 tabs (Lab / Registry / Notes /
Writeups), single 'Working' view, server-side substring search, PDF
embed via Flask route. Deferred to v2: Working/Reading mode toggle,
thread-as-graph visualization, in-place editing.

The corpus is small enough to load all four surfaces at startup;
re-launch to pick up new files. No file watchers.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import flask
from dash import Dash, Input, Output, dcc, html
from dash.exceptions import PreventUpdate

from config import PROJECT_ROOT

from . import components as ui
from .loader import LoadedCorpus, load_corpus, search_content


# ---------------------------------------------------------------------------
# CLI + defaults
# ---------------------------------------------------------------------------


DEFAULT_LAB = PROJECT_ROOT / "notes" / "lab"
DEFAULT_PREREG = PROJECT_ROOT / "notes" / "preregistrations"
DEFAULT_NOTES = PROJECT_ROOT / "notes"
DEFAULT_WRITEUP = PROJECT_ROOT / "writeup" / "tex"
# Path scheme for the Claude Code memory directory for this project.
DEFAULT_MEMORY = Path(
    "~/.claude/projects/-Users-pmahon-Research-Dynamics-MITACS/memory"
).expanduser()


TAB_LAB = "lab"
TAB_REGISTRY = "registry"
TAB_NOTES = "notes"
TAB_WRITEUPS = "writeups"


# ---------------------------------------------------------------------------
# App construction
# ---------------------------------------------------------------------------


def build_app(
    *,
    lab_dir: Path,
    prereg_dir: Path,
    notes_dir: Path,
    memory_dir: Path,
    writeup_dir: Path,
) -> Dash:
    """Construct and wire the Dash app. Loads the corpus once at
    startup; callbacks read from the loaded snapshot."""
    corpus: LoadedCorpus = load_corpus(
        lab_dir=lab_dir,
        prereg_dir=prereg_dir,
        notes_dir=notes_dir,
        memory_dir=memory_dir,
        writeup_dir=writeup_dir,
    )

    # Index by id for fast callback lookup.
    sessions_by_id = {s.session_id: s for s in corpus.sessions}
    registry_by_name = {r.name: r for r in corpus.registry}
    notes_by_path = {str(n.path): n for n in corpus.notes}
    memory_by_name = {m.name: m for m in corpus.memory}
    writeups_by_name = {w.name: w for w in corpus.writeups}

    app = Dash(__name__, title="MITACS Lab Notebook", suppress_callback_exceptions=True)

    # ---- PDF route on the underlying Flask server.
    # Built PDFs live under writeup_dir; expose them at /pdf/<name>.pdf
    # via send_from_directory (path-safety enforced by Flask).
    @app.server.route("/pdf/<path:filename>")
    def _serve_pdf(filename: str):  # noqa: ANN202 — Flask handler
        # Only serve files that match a loaded writeup, by name.
        # Defensive: also restrict to the .pdf suffix.
        if not filename.endswith(".pdf"):
            flask.abort(404)
        stem = filename[: -len(".pdf")]
        if stem not in writeups_by_name or writeups_by_name[stem].pdf_path is None:
            flask.abort(404)
        return flask.send_from_directory(
            writeup_dir.resolve(), filename, mimetype="application/pdf"
        )

    # ---- Layout
    app.layout = html.Div(
        style=ui.PAGE_STYLE,
        children=[
            ui.header_bar(active_tab=TAB_LAB),
            html.Div(
                style=ui.CONTAINER_STYLE,
                children=[
                    dcc.Tabs(
                        id="tabs",
                        value=TAB_LAB,
                        children=[
                            dcc.Tab(label="Lab", value=TAB_LAB),
                            dcc.Tab(label="Registry", value=TAB_REGISTRY),
                            dcc.Tab(label="Notes", value=TAB_NOTES),
                            dcc.Tab(label="Writeups", value=TAB_WRITEUPS),
                        ],
                        style={"marginBottom": "16px"},
                    ),
                    # Selection state per tab (so flipping back returns to
                    # the previously-selected item).
                    dcc.Store(id="sel-lab", data=_default_session_id(corpus)),
                    dcc.Store(id="sel-registry", data=_default_registry_id(corpus)),
                    dcc.Store(id="sel-notes", data=_default_notes_id(corpus)),
                    dcc.Store(id="sel-writeups", data=_default_writeup_id(corpus)),
                    # Search-results panel (shown above tab content when
                    # search has text).
                    html.Div(id="search-panel"),
                    html.Div(id="tab-content"),
                ],
            ),
        ],
    )

    # -----------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------

    @app.callback(
        Output("search-panel", "children"),
        Input("search-input", "value"),
    )
    def _on_search(query: str | None):  # noqa: ANN202
        if not query or not query.strip():
            return ""
        hits = search_content(query, corpus)
        return html.Div(
            style={
                "background": "#f7fafc",
                "border": "1px solid #cbd5e0",
                "borderRadius": "4px",
                "padding": "12px 14px",
                "marginBottom": "16px",
            },
            children=[
                html.Div(
                    f"search: \"{query}\"",
                    style={
                        "fontSize": "12px",
                        "fontWeight": "600",
                        "color": "#2d3748",
                        "marginBottom": "8px",
                        "fontFamily": "ui-monospace, monospace",
                    },
                ),
                ui.search_results(hits),
            ],
        )

    @app.callback(
        Output("tab-content", "children"),
        Input("tabs", "value"),
        Input("sel-lab", "data"),
        Input("sel-registry", "data"),
        Input("sel-notes", "data"),
        Input("sel-writeups", "data"),
    )
    def _on_tab(
        tab: str,
        sel_lab: str | None,
        sel_registry: str | None,
        sel_notes: str | None,
        sel_writeups: str | None,
    ):  # noqa: ANN202
        if tab == TAB_LAB:
            sidebar = ui.sessions_sidebar(corpus, sel_lab)
            entry = sessions_by_id.get(sel_lab) if sel_lab else None
            main = (
                ui.session_detail(entry)
                if entry is not None
                else _placeholder("Select a session entry from the left.")
            )
            return ui.two_column_layout(sidebar, main)

        if tab == TAB_REGISTRY:
            sidebar = ui.registry_sidebar(corpus, sel_registry)
            entry = registry_by_name.get(sel_registry) if sel_registry else None
            main = (
                ui.registry_detail(entry)
                if entry is not None
                else _placeholder("Select a registry entry from the left.")
            )
            return ui.two_column_layout(sidebar, main)

        if tab == TAB_NOTES:
            sidebar = ui.notes_sidebar(corpus, sel_notes)
            # sel_notes may be a notes-path or a memory-name; check both.
            if sel_notes and sel_notes in notes_by_path:
                main = ui.note_detail(notes_by_path[sel_notes])
            elif sel_notes and sel_notes in memory_by_name:
                main = ui.memory_detail(memory_by_name[sel_notes])
            else:
                main = _placeholder(
                    "Select a seed, literature scout, or memory file from the left."
                )
            return ui.two_column_layout(sidebar, main)

        if tab == TAB_WRITEUPS:
            sidebar = ui.writeups_sidebar(corpus, sel_writeups)
            w = writeups_by_name.get(sel_writeups) if sel_writeups else None
            if w is not None:
                pdf_url = (
                    f"/pdf/{w.pdf_path.name}" if w.pdf_path is not None else None
                )
                main = ui.writeup_detail(w, pdf_url)
            else:
                main = _placeholder(
                    "Select a writeup from the left to preview its PDF."
                )
            return ui.two_column_layout(sidebar, main)

        return _placeholder(f"Unknown tab: {tab}")

    # Pattern-matching callbacks for sidebar clicks live in
    # _wire_selection_callbacks (uses Dash ALL pattern + allow_duplicate).
    return _wire_selection_callbacks(app, corpus)


def _wire_selection_callbacks(app: Dash, corpus: LoadedCorpus) -> Dash:
    """Pattern-matching callbacks for sidebar clicks. One callback per
    tab's selection store; no duplicate outputs needed."""
    from dash import ALL, ctx

    @app.callback(
        Output("sel-lab", "data"),
        Input({"type": "session-card", "id": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def _on_session_click(_clicks):  # noqa: ANN202
        if not ctx.triggered_id or not any(_clicks or []):
            raise PreventUpdate
        return ctx.triggered_id["id"]

    @app.callback(
        Output("sel-registry", "data"),
        Input({"type": "registry-card", "id": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def _on_registry_click(_clicks):  # noqa: ANN202
        if not ctx.triggered_id or not any(_clicks or []):
            raise PreventUpdate
        return ctx.triggered_id["id"]

    @app.callback(
        Output("sel-notes", "data"),
        Input({"type": "note-card", "id": ALL}, "n_clicks"),
        Input({"type": "memory-card", "id": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def _on_note_click(_n1, _n2):  # noqa: ANN202
        if not ctx.triggered_id:
            raise PreventUpdate
        # The id dict carries the right value either way (note path
        # for notes, memory name for memory).
        return ctx.triggered_id["id"]

    @app.callback(
        Output("sel-writeups", "data"),
        Input({"type": "writeup-card", "id": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def _on_writeup_click(_clicks):  # noqa: ANN202
        if not ctx.triggered_id or not any(_clicks or []):
            raise PreventUpdate
        return ctx.triggered_id["id"]

    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _placeholder(msg: str) -> html.Div:
    return html.Div(
        msg,
        style={
            "padding": "40px 20px",
            "color": "#718096",
            "fontStyle": "italic",
            "background": "#ffffff",
            "border": "1px dashed #cbd5e0",
            "borderRadius": "4px",
            "textAlign": "center",
        },
    )


def _default_session_id(c: LoadedCorpus) -> str | None:
    return c.sessions[-1].session_id if c.sessions else None


def _default_registry_id(c: LoadedCorpus) -> str | None:
    # First SETTLED entry, falling back to first entry.
    for r in c.registry:
        if r.status == "SETTLED":
            return r.name
    return c.registry[0].name if c.registry else None


def _default_notes_id(c: LoadedCorpus) -> str | None:
    if c.notes:
        return str(c.notes[0].path)
    if c.memory:
        return c.memory[0].name
    return None


def _default_writeup_id(c: LoadedCorpus) -> str | None:
    # Prefer a built PDF.
    for w in c.writeups:
        if w.pdf_path is not None:
            return w.name
    return c.writeups[0].name if c.writeups else None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument(
        "--lab-dir", type=Path, default=DEFAULT_LAB,
        help="directory of session entries (default: notes/lab/)",
    )
    ap.add_argument(
        "--prereg-dir", type=Path, default=DEFAULT_PREREG,
        help="registry root (default: notes/preregistrations/)",
    )
    ap.add_argument(
        "--notes-dir", type=Path, default=DEFAULT_NOTES,
        help="parent notes/ dir (for seeds + literature; default: notes/)",
    )
    ap.add_argument(
        "--memory-dir", type=Path, default=DEFAULT_MEMORY,
        help="Claude Code memory dir (default: project-specific)",
    )
    ap.add_argument(
        "--writeup-dir", type=Path, default=DEFAULT_WRITEUP,
        help="LaTeX/PDF writeups dir (default: writeup/tex/)",
    )
    args = ap.parse_args()

    app = build_app(
        lab_dir=args.lab_dir,
        prereg_dir=args.prereg_dir,
        notes_dir=args.notes_dir,
        memory_dir=args.memory_dir,
        writeup_dir=args.writeup_dir,
    )
    app.run(debug=False, port=args.port, host="127.0.0.1")


if __name__ == "__main__":
    main()
