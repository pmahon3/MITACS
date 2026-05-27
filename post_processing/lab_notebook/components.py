"""Dash component builders for the lab notebook app.

Stateless. Every function returns a Dash component tree given its
inputs; no callbacks, no global state. The app module wires these
together with callbacks and selection state.

Styling is inline + minimal — same posture as
``operator_dashboard.app`` (a system-font sans-serif, a content
column with reasonable max-width). The objective is signal density,
not visual polish.

LaTeX in markdown is rendered via ``dcc.Markdown(mathjax=True)``
(Dash 2.0+). This is intentionally simple: the writeups carry their
own LaTeX/PDF pipeline; markdown rendering here is for the lab
entries + memory + seeds + literature.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from dash import dcc, html

from .loader import (
    LoadedCorpus,
    MemoryFile,
    NoteFile,
    RegistryEntry,
    SearchHit,
    SessionEntry,
    WriteupPDF,
)


# ---------------------------------------------------------------------------
# Styling primitives
# ---------------------------------------------------------------------------

PAGE_STYLE: dict[str, Any] = {
    "fontFamily": "system-ui, -apple-system, sans-serif",
    "color": "#1a1a1a",
    "background": "#fafafa",
    "minHeight": "100vh",
    "margin": 0,
}

CONTAINER_STYLE: dict[str, Any] = {
    "maxWidth": "1280px",
    "margin": "0 auto",
    "padding": "0 16px 32px",
}

CARD_STYLE: dict[str, Any] = {
    "border": "1px solid #e0e0e0",
    "borderRadius": "4px",
    "padding": "12px 14px",
    "marginBottom": "10px",
    "background": "#ffffff",
    "cursor": "pointer",
}

CARD_STYLE_ACTIVE: dict[str, Any] = {
    **CARD_STYLE,
    "borderColor": "#2c5282",
    "background": "#ebf5ff",
}

CHIP_STYLE: dict[str, Any] = {
    "display": "inline-block",
    "padding": "1px 7px",
    "marginRight": "5px",
    "borderRadius": "10px",
    "fontSize": "11px",
    "background": "#edf2f7",
    "color": "#2d3748",
}

CODEBLOCK_STYLE: dict[str, Any] = {
    "background": "#1e1e1e",
    "color": "#dcdcdc",
    "fontFamily": "ui-monospace, SFMono-Regular, Menlo, monospace",
    "fontSize": "12px",
    "padding": "10px 12px",
    "borderRadius": "4px",
    "overflowX": "auto",
    "whiteSpace": "pre",
}


# Status -> color mapping for registry entry badges.
_STATUS_COLORS: dict[str, tuple[str, str]] = {
    # (background, foreground)
    "SETTLED": ("#d4edda", "#13502b"),
    "PROVISIONAL": ("#fff3cd", "#5c3c00"),
    "AMBIGUOUS": ("#fde2cc", "#7a3400"),
    "AWAITING-ARB": ("#cfe2ff", "#054295"),
    "AWAITING-RES": ("#cfe2ff", "#054295"),
    "AWAITING-B": ("#e2e3e5", "#383d41"),
    "EMPTY": ("#f0f0f0", "#6c757d"),
    "BROKEN": ("#f8d7da", "#721c24"),
}


def _status_chip(status: str) -> html.Span:
    """Coloured pill for an entry's status. Threads encode their state
    in ``THREAD:<state>``; we colour by the inner state for those."""
    base = status.split(":", 1)[-1].upper() if status.startswith("THREAD:") else status
    bg, fg = _STATUS_COLORS.get(base, ("#e2e3e5", "#383d41"))
    if status.startswith("THREAD:"):
        bg = "#e0d4f7"
        fg = "#3d2380"
    return html.Span(
        status,
        style={
            "display": "inline-block",
            "padding": "2px 8px",
            "borderRadius": "10px",
            "fontSize": "11px",
            "fontWeight": "600",
            "background": bg,
            "color": fg,
            "fontFamily": "ui-monospace, monospace",
        },
    )


# ---------------------------------------------------------------------------
# Header / navigation
# ---------------------------------------------------------------------------


def header_bar(active_tab: str, search_value: str = "") -> html.Div:
    """Top bar: title, search box, tab nav. The tabs are rendered by
    the app's ``dcc.Tabs``; this bar holds the title + search box only."""
    return html.Div(
        style={
            "background": "#1a202c",
            "color": "#ffffff",
            "padding": "12px 16px",
            "borderBottom": "1px solid #2d3748",
            "marginBottom": "16px",
        },
        children=[
            html.Div(
                style={
                    "maxWidth": "1280px",
                    "margin": "0 auto",
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "16px",
                    "flexWrap": "wrap",
                },
                children=[
                    html.H2(
                        "MITACS Lab Notebook",
                        style={
                            "margin": 0,
                            "fontSize": "18px",
                            "fontWeight": "600",
                            "letterSpacing": "0.3px",
                        },
                    ),
                    html.Span(
                        f"tab: {active_tab}",
                        style={
                            "fontSize": "12px",
                            "color": "#a0aec0",
                            "fontFamily": "ui-monospace, monospace",
                        },
                    ),
                    html.Div(style={"flex": 1}),
                    dcc.Input(
                        id="search-input",
                        type="text",
                        placeholder="search all content (case-insensitive substring)",
                        value=search_value,
                        debounce=True,
                        style={
                            "padding": "6px 10px",
                            "fontSize": "13px",
                            "border": "1px solid #4a5568",
                            "borderRadius": "4px",
                            "background": "#2d3748",
                            "color": "#ffffff",
                            "minWidth": "280px",
                        },
                    ),
                ],
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Cards (sidebar list items)
# ---------------------------------------------------------------------------


def session_entry_card(entry: SessionEntry, is_active: bool = False) -> html.Div:
    """A sidebar card for one session entry."""
    style = CARD_STYLE_ACTIVE if is_active else CARD_STYLE
    chips = [
        html.Span(f, style=CHIP_STYLE) for f in entry.focus[:3]
    ]
    status = entry.status_at_end
    status_color = {
        "wrapped": "#13502b",
        "in-progress": "#5c3c00",
        "blocked": "#721c24",
    }.get(status, "#6c757d")
    return html.Div(
        id={"type": "session-card", "id": entry.session_id},
        n_clicks=0,
        style=style,
        children=[
            html.Div(
                style={
                    "display": "flex",
                    "justifyContent": "space-between",
                    "alignItems": "baseline",
                },
                children=[
                    html.Div(
                        entry.session_id,
                        style={
                            "fontWeight": "600",
                            "fontFamily": "ui-monospace, monospace",
                            "fontSize": "13px",
                        },
                    ),
                    html.Div(
                        status,
                        style={
                            "fontSize": "11px",
                            "color": status_color,
                            "fontWeight": "600",
                        },
                    ),
                ],
            ),
            html.Div(
                entry.title,
                style={
                    "fontSize": "13px",
                    "color": "#4a5568",
                    "margin": "4px 0 6px",
                },
            ),
            html.Div(chips) if chips else None,
        ],
    )


def registry_entry_card(
    entry: RegistryEntry, is_active: bool = False
) -> html.Div:
    """A sidebar card for one registry topic."""
    style = CARD_STYLE_ACTIVE if is_active else CARD_STYLE
    sub_lines: list[Any] = [
        html.Div(
            style={
                "display": "flex",
                "justifyContent": "space-between",
                "alignItems": "baseline",
                "gap": "8px",
            },
            children=[
                html.Div(
                    entry.name,
                    style={
                        "fontWeight": "600",
                        "fontFamily": "ui-monospace, monospace",
                        "fontSize": "12px",
                        "wordBreak": "break-word",
                    },
                ),
                _status_chip(entry.status),
            ],
        )
    ]

    if entry.is_thread and entry.thread_state:
        ts = entry.thread_state
        sub_lines.append(
            html.Div(
                f"thread · {ts.get('node_count', 0)} nodes · "
                f"current={ts.get('current_node_id') or '—'}",
                style={
                    "fontSize": "11px",
                    "color": "#4a5568",
                    "marginTop": "4px",
                },
            )
        )
    elif entry.verdict_summary:
        sub_lines.append(
            html.Div(
                entry.verdict_summary,
                style={
                    "fontSize": "11px",
                    "color": "#4a5568",
                    "marginTop": "4px",
                    "fontFamily": "ui-monospace, monospace",
                },
            )
        )

    return html.Div(
        id={"type": "registry-card", "id": entry.name},
        n_clicks=0,
        style=style,
        children=sub_lines,
    )


def note_card(note: NoteFile, is_active: bool = False) -> html.Div:
    """A sidebar card for one seed/literature note."""
    style = CARD_STYLE_ACTIVE if is_active else CARD_STYLE
    return html.Div(
        id={"type": "note-card", "id": str(note.path)},
        n_clicks=0,
        style=style,
        children=[
            html.Div(
                note.category,
                style={
                    "fontSize": "10px",
                    "fontWeight": "600",
                    "color": "#718096",
                    "textTransform": "uppercase",
                    "letterSpacing": "0.5px",
                },
            ),
            html.Div(
                note.title,
                style={
                    "fontSize": "13px",
                    "fontWeight": "500",
                    "margin": "3px 0 4px",
                },
            ),
            html.Div(
                note.path.name,
                style={
                    "fontSize": "11px",
                    "color": "#a0aec0",
                    "fontFamily": "ui-monospace, monospace",
                },
            ),
        ],
    )


def writeup_card(w: WriteupPDF, is_active: bool = False) -> html.Div:
    """A sidebar card for one writeup tex/pdf pair."""
    style = CARD_STYLE_ACTIVE if is_active else CARD_STYLE
    if w.pdf_path is None:
        sub = "(no PDF built)"
        color = "#a0aec0"
    elif w.page_count is not None:
        sub = f"{w.page_count} pp · {w.size_bytes // 1024 if w.size_bytes else 0} KB"
        color = "#4a5568"
    else:
        sub = "PDF built (pdfinfo unavailable)"
        color = "#4a5568"
    return html.Div(
        id={"type": "writeup-card", "id": w.name},
        n_clicks=0,
        style=style,
        children=[
            html.Div(
                w.name,
                style={
                    "fontWeight": "600",
                    "fontFamily": "ui-monospace, monospace",
                    "fontSize": "13px",
                },
            ),
            html.Div(
                sub,
                style={"fontSize": "11px", "color": color, "marginTop": "4px"},
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Detail panels
# ---------------------------------------------------------------------------


def markdown_panel(text: str) -> html.Div:
    """Render markdown text with MathJax support for LaTeX."""
    return html.Div(
        dcc.Markdown(
            text,
            mathjax=True,
            dangerously_allow_html=False,
            style={"fontSize": "14px", "lineHeight": "1.55"},
        ),
        style={
            "background": "#ffffff",
            "padding": "20px 28px",
            "borderRadius": "4px",
            "border": "1px solid #e0e0e0",
        },
    )


def yaml_panel(data: dict[str, Any], *, title: str | None = None) -> html.Div:
    """Pretty-print a YAML dict in a code block."""
    try:
        rendered = yaml.safe_dump(
            data, sort_keys=False, default_flow_style=False, width=110
        )
    except Exception as exc:  # noqa: BLE001
        rendered = f"(could not render YAML: {exc})"
    children: list[Any] = []
    if title:
        children.append(
            html.Div(
                title,
                style={
                    "fontFamily": "ui-monospace, monospace",
                    "fontSize": "12px",
                    "fontWeight": "600",
                    "color": "#4a5568",
                    "marginBottom": "6px",
                    "marginTop": "12px",
                },
            )
        )
    children.append(html.Pre(rendered, style=CODEBLOCK_STYLE))
    return html.Div(children=children)


def session_detail(entry: SessionEntry) -> html.Div:
    """Full view of one session entry: frontmatter + body."""
    frontmatter_chips: list[Any] = []
    for f in entry.focus:
        frontmatter_chips.append(html.Span(f, style=CHIP_STYLE))
    settled = entry.frontmatter.get("settled_this_session") or []
    dispatched = entry.frontmatter.get("dispatched_this_session") or []
    deferred = entry.frontmatter.get("deferred_to_next_session") or []

    summary_rows: list[Any] = []

    def _stat_row(label: str, values: list[Any]) -> Any:
        if not values:
            return None
        return html.Div(
            style={"marginBottom": "6px"},
            children=[
                html.Span(
                    f"{label}: ",
                    style={
                        "fontFamily": "ui-monospace, monospace",
                        "fontSize": "12px",
                        "color": "#4a5568",
                        "fontWeight": "600",
                    },
                ),
                html.Span(
                    ", ".join(str(v) for v in values),
                    style={"fontSize": "12px"},
                ),
            ],
        )

    for label, vals in (
        ("settled this session", settled),
        ("dispatched this session", dispatched),
        ("deferred to next session", deferred),
        (
            "preregistrations touched",
            entry.frontmatter.get("preregistrations_touched") or [],
        ),
        ("memory files touched", entry.frontmatter.get("memory_files_touched") or []),
        ("writeups touched", entry.frontmatter.get("writeups_touched") or []),
        ("commits", entry.frontmatter.get("commits") or []),
    ):
        r = _stat_row(label, vals)
        if r is not None:
            summary_rows.append(r)

    return html.Div(
        children=[
            html.Div(
                style={
                    "background": "#ffffff",
                    "padding": "18px 22px",
                    "borderRadius": "4px",
                    "border": "1px solid #e0e0e0",
                    "marginBottom": "12px",
                },
                children=[
                    html.Div(
                        style={
                            "display": "flex",
                            "justifyContent": "space-between",
                            "alignItems": "baseline",
                        },
                        children=[
                            html.H3(
                                entry.title,
                                style={"margin": 0, "fontSize": "18px"},
                            ),
                            html.Div(
                                entry.session_id,
                                style={
                                    "fontFamily": "ui-monospace, monospace",
                                    "fontSize": "13px",
                                    "color": "#718096",
                                },
                            ),
                        ],
                    ),
                    html.Div(
                        frontmatter_chips,
                        style={"margin": "8px 0 12px"},
                    ),
                    *summary_rows,
                ],
            ),
            markdown_panel(entry.body),
        ]
    )


def _node_table(thread_state: dict[str, Any]) -> Any:
    """Tabular per-node summary for a thread."""
    node_states = thread_state.get("node_states") or {}
    if not node_states:
        return html.Div("(no nodes)", style={"fontSize": "12px"})
    rows = [
        html.Tr(
            children=[
                html.Th(
                    h,
                    style={
                        "textAlign": "left",
                        "fontSize": "11px",
                        "padding": "4px 8px",
                        "borderBottom": "1px solid #cbd5e0",
                    },
                )
                for h in ("node", "status", "phase_a")
            ]
        )
    ]
    current = thread_state.get("current_node_id")
    for nid, ns in node_states.items():
        bg = "#fffbe6" if nid == current else "transparent"
        rows.append(
            html.Tr(
                style={"background": bg},
                children=[
                    html.Td(
                        nid + (" ◀" if nid == current else ""),
                        style={
                            "fontFamily": "ui-monospace, monospace",
                            "fontSize": "12px",
                            "padding": "3px 8px",
                        },
                    ),
                    html.Td(
                        ns.get("status") or "",
                        style={"fontSize": "12px", "padding": "3px 8px"},
                    ),
                    html.Td(
                        "OK" if ns.get("phase_a_exists")
                        else ("missing" if ns.get("phase_a_path") else "—"),
                        style={"fontSize": "12px", "padding": "3px 8px"},
                    ),
                ],
            )
        )
    return html.Table(
        rows,
        style={
            "borderCollapse": "collapse",
            "marginTop": "6px",
            "marginBottom": "12px",
        },
    )


def registry_detail(entry: RegistryEntry) -> html.Div:
    """Full view of one registry topic — header + thread summary (if
    thread) + all artifact YAMLs."""
    header_children: list[Any] = [
        html.Div(
            style={
                "display": "flex",
                "justifyContent": "space-between",
                "alignItems": "baseline",
                "gap": "12px",
                "flexWrap": "wrap",
            },
            children=[
                html.H3(
                    entry.name,
                    style={
                        "margin": 0,
                        "fontSize": "16px",
                        "fontFamily": "ui-monospace, monospace",
                        "wordBreak": "break-word",
                    },
                ),
                _status_chip(entry.status),
            ],
        )
    ]

    if entry.is_thread and entry.thread_state:
        ts = entry.thread_state
        header_children.append(
            html.Div(
                style={"marginTop": "10px", "fontSize": "13px"},
                children=[
                    html.Div(
                        [html.B("question: "), ts.get("question") or ""],
                        style={"marginBottom": "4px"},
                    ),
                    html.Div(
                        [
                            html.B("state: "),
                            ts.get("state") or "",
                            html.Span(
                                f"  ·  amendments: {ts.get('amendment_count', 0)}",
                                style={"color": "#718096"},
                            ),
                        ],
                        style={"marginBottom": "4px"},
                    ),
                    _node_table(ts),
                ],
            )
        )
    elif entry.verdict_summary:
        header_children.append(
            html.Div(
                entry.verdict_summary,
                style={
                    "marginTop": "8px",
                    "fontSize": "13px",
                    "fontFamily": "ui-monospace, monospace",
                    "color": "#4a5568",
                },
            )
        )

    children: list[Any] = [
        html.Div(
            style={
                "background": "#ffffff",
                "padding": "16px 20px",
                "borderRadius": "4px",
                "border": "1px solid #e0e0e0",
                "marginBottom": "12px",
            },
            children=header_children,
        )
    ]

    # Per-artifact panels
    for fname in entry.files_present:
        data = entry.artifact_data.get(fname, {})
        if "__parse_error__" in data:
            children.append(
                html.Div(
                    f"{fname}: parse error",
                    style={
                        "padding": "8px",
                        "background": "#f8d7da",
                        "color": "#721c24",
                        "borderRadius": "4px",
                        "marginBottom": "8px",
                        "fontFamily": "ui-monospace, monospace",
                        "fontSize": "12px",
                    },
                )
            )
            continue
        children.append(yaml_panel(data, title=fname))

    return html.Div(children=children)


def note_detail(note: NoteFile) -> html.Div:
    """Render a seed or literature note as markdown."""
    return html.Div(
        children=[
            html.Div(
                style={
                    "background": "#ffffff",
                    "padding": "10px 16px",
                    "borderRadius": "4px",
                    "border": "1px solid #e0e0e0",
                    "marginBottom": "10px",
                    "display": "flex",
                    "justifyContent": "space-between",
                    "alignItems": "center",
                },
                children=[
                    html.Span(
                        note.category.upper(),
                        style={
                            "fontSize": "11px",
                            "fontWeight": "700",
                            "color": "#718096",
                            "letterSpacing": "0.5px",
                        },
                    ),
                    html.Span(
                        note.path.name,
                        style={
                            "fontFamily": "ui-monospace, monospace",
                            "fontSize": "12px",
                            "color": "#a0aec0",
                        },
                    ),
                ],
            ),
            markdown_panel(note.body),
        ]
    )


def memory_detail(mem: MemoryFile) -> html.Div:
    """Render a memory file (frontmatter description on top, body
    rendered as markdown)."""
    return html.Div(
        children=[
            html.Div(
                style={
                    "background": "#ffffff",
                    "padding": "12px 18px",
                    "borderRadius": "4px",
                    "border": "1px solid #e0e0e0",
                    "marginBottom": "10px",
                },
                children=[
                    html.Div(
                        mem.name,
                        style={
                            "fontFamily": "ui-monospace, monospace",
                            "fontWeight": "600",
                            "fontSize": "13px",
                            "marginBottom": "6px",
                        },
                    ),
                    html.Div(
                        mem.description,
                        style={
                            "fontSize": "13px",
                            "color": "#2d3748",
                            "lineHeight": "1.5",
                        },
                    ),
                ],
            ),
            markdown_panel(mem.body),
        ]
    )


def pdf_iframe(pdf_url: str) -> html.Iframe:
    """Embed a PDF served by the Flask route."""
    return html.Iframe(
        src=pdf_url,
        style={
            "width": "100%",
            "height": "78vh",
            "border": "1px solid #e0e0e0",
            "borderRadius": "4px",
            "background": "#ffffff",
        },
    )


def writeup_detail(w: WriteupPDF, pdf_url: str | None) -> html.Div:
    """Full view of a writeup: PDF iframe + metadata, or a 'not built'
    notice if no PDF exists."""
    children: list[Any] = [
        html.Div(
            style={
                "background": "#ffffff",
                "padding": "12px 18px",
                "borderRadius": "4px",
                "border": "1px solid #e0e0e0",
                "marginBottom": "12px",
            },
            children=[
                html.Div(
                    w.name,
                    style={
                        "fontFamily": "ui-monospace, monospace",
                        "fontWeight": "600",
                        "fontSize": "14px",
                        "marginBottom": "4px",
                    },
                ),
                html.Div(
                    f"tex: {w.tex_path.name}"
                    + (
                        f"  ·  pdf: {w.pdf_path.name}  ·  {w.page_count or '?'} pp"
                        if w.pdf_path is not None
                        else "  ·  no PDF built"
                    ),
                    style={"fontSize": "12px", "color": "#4a5568"},
                ),
            ],
        )
    ]
    if pdf_url is not None:
        children.append(pdf_iframe(pdf_url))
    else:
        children.append(
            html.Div(
                "No built PDF for this writeup. Build the LaTeX source "
                "to see it rendered here.",
                style={
                    "padding": "20px",
                    "background": "#fff3cd",
                    "color": "#5c3c00",
                    "borderRadius": "4px",
                    "border": "1px solid #e0c068",
                    "fontSize": "13px",
                },
            )
        )
    return html.Div(children=children)


# ---------------------------------------------------------------------------
# Search results
# ---------------------------------------------------------------------------


def search_results(hits: list[SearchHit]) -> html.Div:
    """Render a list of search hits with their snippets."""
    if not hits:
        return html.Div(
            "no matches",
            style={
                "padding": "12px",
                "fontSize": "13px",
                "color": "#718096",
                "fontStyle": "italic",
            },
        )
    rows: list[Any] = [
        html.Div(
            f"{len(hits)} sources matched",
            style={
                "fontSize": "12px",
                "color": "#4a5568",
                "marginBottom": "8px",
                "fontFamily": "ui-monospace, monospace",
            },
        )
    ]
    for h in hits:
        rows.append(
            html.Div(
                style={
                    "padding": "10px 12px",
                    "marginBottom": "6px",
                    "background": "#ffffff",
                    "border": "1px solid #e0e0e0",
                    "borderRadius": "4px",
                },
                children=[
                    html.Div(
                        style={
                            "display": "flex",
                            "justifyContent": "space-between",
                            "alignItems": "baseline",
                            "marginBottom": "4px",
                        },
                        children=[
                            html.Span(
                                f"[{h.source_kind}] {h.source_id}",
                                style={
                                    "fontFamily": "ui-monospace, monospace",
                                    "fontSize": "12px",
                                    "fontWeight": "600",
                                    "color": "#2d3748",
                                },
                            ),
                            html.Span(
                                f"{h.match_count} match{'es' if h.match_count != 1 else ''}",
                                style={
                                    "fontSize": "11px",
                                    "color": "#718096",
                                },
                            ),
                        ],
                    ),
                    html.Div(
                        h.snippet,
                        style={
                            "fontSize": "12px",
                            "color": "#4a5568",
                            "fontFamily": "ui-monospace, monospace",
                            "lineHeight": "1.45",
                        },
                    ),
                ],
            )
        )
    return html.Div(children=rows)


# ---------------------------------------------------------------------------
# Sidebar lists
# ---------------------------------------------------------------------------


def sessions_sidebar(
    corpus: LoadedCorpus, active_id: str | None
) -> html.Div:
    """The left sidebar of the Lab tab."""
    if not corpus.sessions:
        return html.Div(
            "no session entries — add notes/lab/YYYY-MM-DD.md",
            style={"padding": "12px", "fontSize": "13px", "color": "#718096"},
        )
    # Most recent first.
    return html.Div(
        [
            session_entry_card(s, is_active=(s.session_id == active_id))
            for s in reversed(corpus.sessions)
        ]
    )


def registry_sidebar(
    corpus: LoadedCorpus, active_id: str | None
) -> html.Div:
    """The left sidebar of the Registry tab."""
    if not corpus.registry:
        return html.Div(
            "no registry entries",
            style={"padding": "12px", "fontSize": "13px", "color": "#718096"},
        )
    return html.Div(
        [
            registry_entry_card(r, is_active=(r.name == active_id))
            for r in corpus.registry
        ]
    )


def notes_sidebar(
    corpus: LoadedCorpus, active_id: str | None
) -> html.Div:
    """The left sidebar of the Notes tab (seeds + literature + memory)."""
    if not corpus.notes and not corpus.memory:
        return html.Div(
            "no notes",
            style={"padding": "12px", "fontSize": "13px", "color": "#718096"},
        )
    children: list[Any] = []

    if corpus.notes:
        children.append(
            html.Div(
                "seeds + literature",
                style={
                    "fontSize": "11px",
                    "fontWeight": "700",
                    "color": "#718096",
                    "letterSpacing": "0.5px",
                    "margin": "4px 0 6px",
                    "textTransform": "uppercase",
                },
            )
        )
        for n in corpus.notes:
            children.append(note_card(n, is_active=(str(n.path) == active_id)))

    if corpus.memory:
        children.append(
            html.Div(
                "memory files",
                style={
                    "fontSize": "11px",
                    "fontWeight": "700",
                    "color": "#718096",
                    "letterSpacing": "0.5px",
                    "margin": "14px 0 6px",
                    "textTransform": "uppercase",
                },
            )
        )
        for m in corpus.memory:
            children.append(
                html.Div(
                    id={"type": "memory-card", "id": m.name},
                    n_clicks=0,
                    style=(
                        CARD_STYLE_ACTIVE if m.name == active_id else CARD_STYLE
                    ),
                    children=[
                        html.Div(
                            m.name,
                            style={
                                "fontWeight": "600",
                                "fontFamily": "ui-monospace, monospace",
                                "fontSize": "12px",
                            },
                        ),
                        html.Div(
                            (m.description[:140] + "…")
                            if len(m.description) > 140
                            else m.description,
                            style={
                                "fontSize": "11px",
                                "color": "#4a5568",
                                "marginTop": "4px",
                                "lineHeight": "1.45",
                            },
                        ),
                    ],
                )
            )

    return html.Div(children)


def writeups_sidebar(
    corpus: LoadedCorpus, active_id: str | None
) -> html.Div:
    """The left sidebar of the Writeups tab."""
    if not corpus.writeups:
        return html.Div(
            "no writeups",
            style={"padding": "12px", "fontSize": "13px", "color": "#718096"},
        )
    return html.Div(
        [
            writeup_card(w, is_active=(w.name == active_id))
            for w in corpus.writeups
        ]
    )


# ---------------------------------------------------------------------------
# Two-column layout helper
# ---------------------------------------------------------------------------


def two_column_layout(sidebar: Any, main: Any) -> html.Div:
    """Standard sidebar + main panel split used by every tab."""
    return html.Div(
        style={
            "display": "grid",
            "gridTemplateColumns": "320px 1fr",
            "gap": "16px",
            "alignItems": "start",
        },
        children=[
            html.Div(
                sidebar,
                style={
                    "maxHeight": "82vh",
                    "overflowY": "auto",
                    "paddingRight": "6px",
                },
            ),
            html.Div(
                main,
                style={
                    "maxHeight": "82vh",
                    "overflowY": "auto",
                },
            ),
        ],
    )
