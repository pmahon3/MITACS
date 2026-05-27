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

import plotly.graph_objects as go
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


# Tab-id constants (single source of truth; mirrored in app.py).
TAB_HOME = "home"
TAB_LAB = "lab"
TAB_REGISTRY = "registry"
TAB_NOTES = "notes"
TAB_WRITEUPS = "writeups"


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
    "PLANNING": ("#e0d4f7", "#3d2380"),
    "ACTIVE": ("#cfe2ff", "#054295"),
    "EXHAUSTED": ("#e2e3e5", "#383d41"),
    "RESOLVED": ("#d4edda", "#13502b"),
}

# Sankey node colors per thread-node `status` string.
_THREAD_NODE_FILL: dict[str, str] = {
    "settled": "#48bb78",   # green
    "active": "#3182ce",    # blue (highlight)
    "planned": "#a0aec0",   # grey
    "closed": "#4a5568",    # dark grey / muted
}
_THREAD_NODE_FILL_DEFAULT = "#cbd5e0"


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
# Cross-tab navigation primitives
# ---------------------------------------------------------------------------


# Inline link style used for in-content cross-tab navigation.
_NAV_LINK_STYLE: dict[str, Any] = {
    "color": "#2c5282",
    "textDecoration": "underline",
    "cursor": "pointer",
    "background": "transparent",
    "border": "none",
    "padding": 0,
    "fontFamily": "inherit",
    "fontSize": "inherit",
    "lineHeight": "inherit",
}

_NAV_LINK_DEAD_STYLE: dict[str, Any] = {
    "color": "#a0aec0",
    "textDecoration": "line-through",
    "fontStyle": "italic",
}


def nav_link(
    target_tab: str,
    target_id: str | None,
    label: str,
    *,
    resolved: bool = True,
) -> html.Span | html.Button:
    """Cross-tab navigation link.

    Returns a clickable ``html.Button`` (styled as a link) when ``resolved``
    is True; the button carries a structured pattern-matching id
    ``{"type": "nav-link", "target_tab": ..., "target_id": ...}`` that a
    single callback in ``app.py`` handles.

    When ``resolved`` is False (the slug doesn't correspond to a real
    loaded entry), renders a muted strikethrough span — visible breakage
    is better than a silent dead link.
    """
    if not resolved or not target_id:
        return html.Span(label, title="(no matching entry)",
                         style=_NAV_LINK_DEAD_STYLE)
    return html.Button(
        label,
        id={"type": "nav-link", "target_tab": target_tab, "target_id": target_id},
        n_clicks=0,
        style=_NAV_LINK_STYLE,
    )


# Slug-resolution helpers. Each takes a raw frontmatter value and a
# corpus index, returns the canonical key to store in selection state
# (or None if unresolvable).


def resolve_registry_slug(slug: str, registry_keys: set[str]) -> str | None:
    """Registry slugs in frontmatter match `registry_by_name` directly."""
    return slug if slug in registry_keys else None


def resolve_memory_slug(slug: str, memory_keys: set[str]) -> str | None:
    """Memory entries are referenced with `.md` extension in
    frontmatter; the loader keys them by stem."""
    stem = slug[:-3] if slug.endswith(".md") else slug
    return stem if stem in memory_keys else None


def resolve_writeup_slug(slug: str, writeup_keys: set[str]) -> str | None:
    """Writeup references are paths like ``writeup/tex/foo.tex``;
    the loader keys them by stem (``foo``)."""
    # Tolerate full paths, bare filenames, with or without .tex
    name = Path(slug).name if "/" in slug else slug
    stem = name[:-4] if name.endswith(".tex") else (
        name[:-4] if name.endswith(".pdf") else name
    )
    return stem if stem in writeup_keys else None


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


def session_detail(
    entry: SessionEntry,
    *,
    registry_keys: set[str] | None = None,
    memory_keys: set[str] | None = None,
    writeup_keys: set[str] | None = None,
) -> html.Div:
    """Full view of one session entry: frontmatter + body.

    When the ``*_keys`` sets are passed, slug-bearing frontmatter rows
    (``preregistrations_touched`` / ``memory_files_touched`` /
    ``writeups_touched`` / ``settled_this_session`` /
    ``dispatched_this_session``) render their items as ``nav_link``
    buttons that switch tabs + select the target item via the
    ``nav-link`` pattern-matching callback. With no keys passed, the
    rows fall back to plain comma-joined text (backwards-compatible).
    """
    registry_keys = registry_keys or set()
    memory_keys = memory_keys or set()
    writeup_keys = writeup_keys or set()

    frontmatter_chips: list[Any] = []
    for f in entry.focus:
        frontmatter_chips.append(html.Span(f, style=CHIP_STYLE))

    summary_rows: list[Any] = []

    def _label_span(label: str) -> html.Span:
        return html.Span(
            f"{label}: ",
            style={
                "fontFamily": "ui-monospace, monospace",
                "fontSize": "12px",
                "color": "#4a5568",
                "fontWeight": "600",
            },
        )

    def _plain_row(label: str, values: list[Any]) -> Any:
        if not values:
            return None
        return html.Div(
            style={"marginBottom": "6px"},
            children=[
                _label_span(label),
                html.Span(
                    ", ".join(str(v) for v in values),
                    style={"fontSize": "12px"},
                ),
            ],
        )

    def _link_row(
        label: str,
        values: list[Any],
        kind: str,  # 'registry' | 'memory' | 'writeup'
    ) -> Any:
        if not values:
            return None
        children: list[Any] = [_label_span(label)]
        for i, raw in enumerate(values):
            slug = str(raw)
            if kind == "registry":
                resolved = resolve_registry_slug(slug, registry_keys)
                link = nav_link(
                    TAB_REGISTRY, resolved, slug,
                    resolved=resolved is not None,
                )
            elif kind == "memory":
                resolved = resolve_memory_slug(slug, memory_keys)
                link = nav_link(
                    TAB_NOTES, resolved, slug,
                    resolved=resolved is not None,
                )
            elif kind == "writeup":
                resolved = resolve_writeup_slug(slug, writeup_keys)
                link = nav_link(
                    TAB_WRITEUPS, resolved, slug,
                    resolved=resolved is not None,
                )
            else:
                link = html.Span(slug)
            children.append(link)
            if i < len(values) - 1:
                children.append(html.Span(", ", style={"color": "#a0aec0"}))
        return html.Div(
            style={"marginBottom": "6px", "fontSize": "12px"}, children=children
        )

    settled = entry.frontmatter.get("settled_this_session") or []
    dispatched = entry.frontmatter.get("dispatched_this_session") or []
    deferred = entry.frontmatter.get("deferred_to_next_session") or []
    pre_touched = entry.frontmatter.get("preregistrations_touched") or []
    mem_touched = entry.frontmatter.get("memory_files_touched") or []
    wri_touched = entry.frontmatter.get("writeups_touched") or []
    commits = entry.frontmatter.get("commits") or []

    for row in (
        _link_row("settled this session", settled, "registry"),
        _link_row("dispatched this session", dispatched, "registry"),
        # deferred items are free-text task descriptions, not slugs.
        _plain_row("deferred to next session", deferred),
        _link_row("preregistrations touched", pre_touched, "registry"),
        _link_row("memory files touched", mem_touched, "memory"),
        _link_row("writeups touched", wri_touched, "writeup"),
        _plain_row("commits", commits),
    ):
        if row is not None:
            summary_rows.append(row)

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


def thread_graph(
    thread_yaml: dict[str, Any],
    thread_state: dict[str, Any] | None = None,
) -> dcc.Graph:
    """Sankey rendering of a thread's node graph.

    Edges come from ``branching_rules[parent][branch_label] = child_id``
    in the raw thread YAML. Null children are terminal branches and are
    omitted. Nodes are colored by ``nodes.<id>.status`` (settled/active/
    planned/closed). The thread's ``current_node_id`` is marked with a
    "◀" suffix on its label (color stays driven by status — the current
    node can legitimately be settled post-arbitration).

    Returned as a ``dcc.Graph`` sized for embedding in the registry
    detail view (height=360px, mode-bar disabled).
    """
    nodes_block: dict[str, dict[str, Any]] = thread_yaml.get("nodes") or {}
    branching_rules: dict[str, dict[str, Any]] = (
        thread_yaml.get("branching_rules") or {}
    )
    current_node_id = (thread_state or {}).get(
        "current_node_id"
    ) or thread_yaml.get("current_node_id")

    # Stable node ordering: declaration order from the YAML.
    node_ids = list(nodes_block.keys())
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    labels: list[str] = []
    colors: list[str] = []
    statuses: list[str] = []
    for nid in node_ids:
        ndata = nodes_block.get(nid) or {}
        name = str(ndata.get("name") or "").strip()
        if len(name) > 38:
            name = name[:35] + "…"
        marker = " ◀" if nid == current_node_id else ""
        label = f"{nid}: {name}{marker}" if name else f"{nid}{marker}"
        status = str(ndata.get("status") or "").lower()
        labels.append(label)
        statuses.append(status)
        colors.append(_THREAD_NODE_FILL.get(status, _THREAD_NODE_FILL_DEFAULT))

    src: list[int] = []
    dst: list[int] = []
    link_labels: list[str] = []
    link_colors: list[str] = []
    for parent_id, branches in branching_rules.items():
        if parent_id not in id_to_idx or not isinstance(branches, dict):
            continue
        for branch_label, child_id in branches.items():
            if not child_id or child_id not in id_to_idx:
                continue
            src.append(id_to_idx[parent_id])
            dst.append(id_to_idx[child_id])
            link_labels.append(str(branch_label))
            # Tint the link by destination node's status so the eye can
            # trace the active branch.
            dest_color = _THREAD_NODE_FILL.get(
                statuses[id_to_idx[child_id]], _THREAD_NODE_FILL_DEFAULT
            )
            # Translucent link in the destination color.
            link_colors.append(_with_alpha(dest_color, 0.30))

    if not src:
        # No edges (e.g., a single-node thread or branching_rules empty).
        # Synthesize a placeholder so Sankey doesn't error and the user
        # still sees the node block. Self-loop with zero hover is ugly;
        # easier to return a small text-only Graph in that case.
        fig = go.Figure()
        fig.add_annotation(
            text="(no branching edges to graph)",
            showarrow=False,
            xref="paper", yref="paper", x=0.5, y=0.5,
            font=dict(size=12, color="#718096"),
        )
        fig.update_layout(
            height=120,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            paper_bgcolor="#ffffff",
        )
        return dcc.Graph(
            figure=fig,
            config={"displayModeBar": False, "scrollZoom": False},
            style={"marginBottom": "12px"},
        )

    sankey = go.Sankey(
        arrangement="snap",
        node=dict(
            label=labels,
            color=colors,
            pad=24,
            thickness=18,
            line=dict(color="#2d3748", width=0.5),
            customdata=statuses,
            hovertemplate="<b>%{label}</b><br>status: %{customdata}<extra></extra>",
        ),
        link=dict(
            source=src,
            target=dst,
            value=[1] * len(src),
            label=link_labels,
            color=link_colors,
            hovertemplate=(
                "%{source.label} → %{target.label}"
                "<br>branch: %{label}<extra></extra>"
            ),
        ),
    )
    fig = go.Figure(data=[sankey])
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=20, b=10),
        font=dict(family="ui-monospace, monospace", size=11),
        paper_bgcolor="#ffffff",
    )
    return dcc.Graph(
        figure=fig,
        config={
            "displayModeBar": False,
            "scrollZoom": False,
            "doubleClick": False,
            "staticPlot": False,
        },
        style={"marginBottom": "12px"},
    )


def _with_alpha(hex_color: str, alpha: float) -> str:
    """Convert ``#RRGGBB`` → ``rgba(r,g,b,a)``."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return hex_color
    try:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    except ValueError:
        return hex_color
    return f"rgba({r},{g},{b},{alpha:.2f})"


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


def _thread_reference_links(
    entry: RegistryEntry, registry_keys: set[str]
) -> list[Any]:
    """Cross-tab links for thread references in this entry's artifacts.

    Scans ``references`` / ``root.artifacts`` blocks of every artifact
    YAML for file paths matching ``*/thread.yaml`` and returns one
    ``nav_link`` per unique thread topic that resolves to a loaded
    registry entry.
    """
    seen: set[str] = set()
    out: list[Any] = []
    for fname, data in entry.artifact_data.items():
        if not isinstance(data, dict):
            continue
        refs: list[dict[str, Any]] = []
        for key in ("references", ):
            v = data.get(key)
            if isinstance(v, list):
                refs.extend(x for x in v if isinstance(x, dict))
        root_block = data.get("root")
        if isinstance(root_block, dict):
            arts = root_block.get("artifacts")
            if isinstance(arts, list):
                refs.extend(x for x in arts if isinstance(x, dict))
        for ref in refs:
            file_field = ref.get("file") or ""
            if not isinstance(file_field, str) or "thread.yaml" not in file_field:
                continue
            # `file` is registry-root-relative, like
            # ../2026-05-26_distributional-class-thread/thread.yaml
            parts = file_field.replace("\\", "/").split("/")
            topic_slug = None
            for i, p in enumerate(parts):
                if p == "thread.yaml" and i > 0:
                    topic_slug = parts[i - 1]
                    break
            if not topic_slug or topic_slug in seen:
                continue
            seen.add(topic_slug)
            resolved = resolve_registry_slug(topic_slug, registry_keys)
            relation = ref.get("relation")
            label = f"{topic_slug}" + (f" ({relation})" if relation else "")
            out.append(
                html.Div(
                    style={"marginBottom": "4px", "fontSize": "12px"},
                    children=[
                        html.Span(
                            "thread ref: ",
                            style={
                                "fontFamily": "ui-monospace, monospace",
                                "color": "#4a5568",
                                "fontWeight": "600",
                                "marginRight": "4px",
                            },
                        ),
                        nav_link(
                            TAB_REGISTRY, resolved, label,
                            resolved=resolved is not None,
                        ),
                    ],
                )
            )
    return out


def registry_detail(
    entry: RegistryEntry,
    *,
    registry_keys: set[str] | None = None,
) -> html.Div:
    """Full view of one registry topic — header + thread Sankey + per-node
    table (if thread) + all artifact YAMLs.

    When ``registry_keys`` is supplied, thread references in the entry's
    artifacts render as cross-tab ``nav_link`` buttons.
    """
    registry_keys = registry_keys or set()
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

    # Thread references (cross-tab links for thread-relation refs in
    # non-thread entries, and thread→thread refs in thread entries).
    ref_links = _thread_reference_links(entry, registry_keys)
    if ref_links:
        header_children.append(
            html.Div(
                ref_links,
                style={"marginTop": "8px"},
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

    # Exposition note (``note.md``). Rendered first, above the thread
    # graph / artifact YAMLs, because this is what the user is asking
    # the dashboard to "see" — the idea and the math attached to this
    # entry. Unpinned (no body_sha256); the YAML artifacts remain the
    # registered claims. Written by the arbiter (experiments) or
    # thread-coordinator (threads).
    if entry.note_md:
        children.append(
            html.Div(
                style={
                    "background": "#ffffff",
                    "padding": "10px 14px 16px 14px",
                    "borderRadius": "4px",
                    "border": "1px solid #e0e0e0",
                    "marginBottom": "12px",
                },
                children=[
                    html.Div(
                        "note",
                        style={
                            "fontFamily": "ui-monospace, monospace",
                            "fontSize": "12px",
                            "fontWeight": "600",
                            "color": "#4a5568",
                            "marginBottom": "6px",
                        },
                    ),
                    dcc.Markdown(
                        entry.note_md,
                        mathjax=True,
                        dangerously_allow_html=False,
                        style={"fontSize": "14px", "lineHeight": "1.6"},
                    ),
                ],
            )
        )

    # Thread-as-graph: Sankey above the node table for thread entries.
    if entry.is_thread and "thread.yaml" in entry.artifact_data:
        thread_yaml = entry.artifact_data["thread.yaml"]
        if isinstance(thread_yaml, dict) and "__parse_error__" not in thread_yaml:
            children.append(
                html.Div(
                    style={
                        "background": "#ffffff",
                        "padding": "10px 14px",
                        "borderRadius": "4px",
                        "border": "1px solid #e0e0e0",
                        "marginBottom": "12px",
                    },
                    children=[
                        html.Div(
                            "thread graph",
                            style={
                                "fontFamily": "ui-monospace, monospace",
                                "fontSize": "12px",
                                "fontWeight": "600",
                                "color": "#4a5568",
                                "marginBottom": "4px",
                            },
                        ),
                        thread_graph(thread_yaml, entry.thread_state),
                        _thread_legend(),
                    ],
                )
            )
        if entry.thread_state:
            children.append(
                html.Div(
                    style={
                        "background": "#ffffff",
                        "padding": "10px 14px",
                        "borderRadius": "4px",
                        "border": "1px solid #e0e0e0",
                        "marginBottom": "12px",
                    },
                    children=[
                        html.Div(
                            "node table",
                            style={
                                "fontFamily": "ui-monospace, monospace",
                                "fontSize": "12px",
                                "fontWeight": "600",
                                "color": "#4a5568",
                                "marginBottom": "4px",
                            },
                        ),
                        _node_table(entry.thread_state),
                    ],
                )
            )

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


def _thread_legend() -> html.Div:
    """Small inline legend for the Sankey node colors."""
    def _swatch(color: str, label: str) -> html.Span:
        return html.Span(
            children=[
                html.Span(
                    style={
                        "display": "inline-block",
                        "width": "10px",
                        "height": "10px",
                        "background": color,
                        "marginRight": "4px",
                        "verticalAlign": "middle",
                        "borderRadius": "2px",
                    }
                ),
                label,
            ],
            style={
                "marginRight": "12px",
                "fontSize": "11px",
                "color": "#4a5568",
                "fontFamily": "ui-monospace, monospace",
            },
        )

    return html.Div(
        children=[
            _swatch(_THREAD_NODE_FILL["settled"], "settled"),
            _swatch(_THREAD_NODE_FILL["active"], "active"),
            _swatch(_THREAD_NODE_FILL["planned"], "planned"),
            _swatch(_THREAD_NODE_FILL["closed"], "closed"),
            html.Span(
                "◀ = current_node_id",
                style={
                    "fontSize": "11px",
                    "color": "#718096",
                    "fontFamily": "ui-monospace, monospace",
                },
            ),
        ],
        style={"marginTop": "4px"},
    )


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
        active_memory = [m for m in corpus.memory if not m.archived]
        archived_memory = [m for m in corpus.memory if m.archived]

        def _memory_card(m: MemoryFile, *, muted: bool = False) -> html.Div:
            base = CARD_STYLE_ACTIVE if m.name == active_id else CARD_STYLE
            if muted and m.name != active_id:
                # Muted style for archived entries: lighter background, dimmer text.
                base = {**base, "backgroundColor": "#fafafa", "opacity": "0.78"}
            return html.Div(
                id={"type": "memory-card", "id": m.name},
                n_clicks=0,
                style=base,
                children=[
                    html.Div(
                        m.name,
                        style={
                            "fontWeight": "600",
                            "fontFamily": "ui-monospace, monospace",
                            "fontSize": "12px",
                            "color": "#6b7280" if muted else "#1a202c",
                        },
                    ),
                    html.Div(
                        (m.description[:140] + "…")
                        if len(m.description) > 140
                        else m.description,
                        style={
                            "fontSize": "11px",
                            "color": "#6b7280" if muted else "#4a5568",
                            "marginTop": "4px",
                            "lineHeight": "1.45",
                        },
                    ),
                ],
            )

        if active_memory:
            children.append(
                html.Div(
                    f"memory files · active ({len(active_memory)})",
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
            for m in active_memory:
                children.append(_memory_card(m))

        if archived_memory:
            children.append(
                html.Div(
                    f"memory files · archived ({len(archived_memory)})",
                    style={
                        "fontSize": "11px",
                        "fontWeight": "700",
                        "color": "#a0aec0",
                        "letterSpacing": "0.5px",
                        "margin": "18px 0 6px",
                        "textTransform": "uppercase",
                        "borderTop": "1px dashed #e2e8f0",
                        "paddingTop": "12px",
                    },
                )
            )
            for m in archived_memory:
                children.append(_memory_card(m, muted=True))

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
# Home landing page
# ---------------------------------------------------------------------------


def _stat_tile(label: str, value: Any, sub: str | None = None) -> html.Div:
    """A single small statistics tile."""
    return html.Div(
        style={
            "background": "#ffffff",
            "border": "1px solid #e0e0e0",
            "borderRadius": "4px",
            "padding": "12px 14px",
            "minWidth": "120px",
        },
        children=[
            html.Div(
                str(value),
                style={
                    "fontSize": "22px",
                    "fontWeight": "700",
                    "fontFamily": "ui-monospace, monospace",
                    "color": "#1a202c",
                    "lineHeight": "1.1",
                },
            ),
            html.Div(
                label,
                style={
                    "fontSize": "11px",
                    "color": "#4a5568",
                    "textTransform": "uppercase",
                    "letterSpacing": "0.5px",
                    "marginTop": "4px",
                },
            ),
            html.Div(
                sub,
                style={
                    "fontSize": "11px",
                    "color": "#718096",
                    "marginTop": "2px",
                    "fontFamily": "ui-monospace, monospace",
                },
            ) if sub else None,
        ],
    )


def _section_header(title: str) -> html.Div:
    return html.Div(
        title,
        style={
            "fontSize": "13px",
            "fontWeight": "700",
            "color": "#1a202c",
            "letterSpacing": "0.3px",
            "marginBottom": "8px",
            "marginTop": "4px",
            "textTransform": "uppercase",
        },
    )


def _card_panel(children: list[Any]) -> html.Div:
    return html.Div(
        style={
            "background": "#ffffff",
            "border": "1px solid #e0e0e0",
            "borderRadius": "4px",
            "padding": "14px 18px",
            "marginBottom": "12px",
        },
        children=children,
    )


def home_panel(
    corpus: LoadedCorpus,
    *,
    registry_keys: set[str],
    memory_keys: set[str],
    writeup_keys: set[str],
) -> html.Div:
    """Landing-page synthesis: deferred items, active threads, recent
    SETTLED findings, at-a-glance counts.

    Layout is stacked-card. The deferred-items section is the load-
    bearing one (it's what tells the user what to pick up); the others
    are supporting context.
    """
    # ---- Stats
    n_sessions = len(corpus.sessions)
    n_registry = len(corpus.registry)
    n_memory = len(corpus.memory)
    n_writeups = len(corpus.writeups)
    threads = [r for r in corpus.registry if r.is_thread]
    n_thread_active = sum(
        1 for r in threads
        if (r.thread_state or {}).get("state") in {"active", "planning"}
    )
    n_thread_exhausted = sum(
        1 for r in threads
        if (r.thread_state or {}).get("state") == "exhausted"
    )
    n_thread_resolved = sum(
        1 for r in threads
        if (r.thread_state or {}).get("state") == "resolved"
    )

    stats_row = html.Div(
        style={
            "display": "flex",
            "gap": "10px",
            "flexWrap": "wrap",
            "marginBottom": "16px",
        },
        children=[
            _stat_tile("sessions", n_sessions),
            _stat_tile("registry entries", n_registry),
            _stat_tile(
                "threads",
                len(threads),
                sub=(
                    f"{n_thread_active} act · "
                    f"{n_thread_exhausted} exh · "
                    f"{n_thread_resolved} res"
                ),
            ),
            _stat_tile("memory files", n_memory),
            _stat_tile("writeups", n_writeups),
        ],
    )

    # ---- Latest session's deferred items
    deferred_children: list[Any] = [_section_header("pick up where you left off")]
    if not corpus.sessions:
        deferred_children.append(
            html.Div(
                "(no session entries logged yet)",
                style={"fontSize": "13px", "color": "#718096"},
            )
        )
    else:
        latest = corpus.sessions[-1]
        deferred = latest.frontmatter.get("deferred_to_next_session") or []
        deferred_children.append(
            html.Div(
                [
                    html.Span(
                        "from session ",
                        style={"fontSize": "12px", "color": "#4a5568"},
                    ),
                    nav_link(TAB_LAB, latest.session_id, latest.session_id),
                    html.Span(
                        f"  ({latest.date})",
                        style={"fontSize": "12px", "color": "#718096"},
                    ),
                ],
                style={"marginBottom": "10px"},
            )
        )
        if not deferred:
            deferred_children.append(
                html.Div(
                    "(latest session has no deferred items)",
                    style={"fontSize": "13px", "color": "#718096"},
                )
            )
        else:
            item_children: list[Any] = []
            for i, raw in enumerate(deferred, 1):
                text = str(raw)
                # Heuristic: try to nav-resolve substrings of the text
                # against registry / writeup / memory keys. Add a trailing
                # set of link chips for the resolved hits without
                # mangling the task description itself.
                inline = _scan_inline_refs(
                    text, registry_keys, writeup_keys, memory_keys
                )
                row_children: list[Any] = [
                    html.Span(
                        f"{i}.  ",
                        style={
                            "color": "#a0aec0",
                            "fontFamily": "ui-monospace, monospace",
                            "fontSize": "12px",
                        },
                    ),
                    html.Span(text, style={"fontSize": "13px"}),
                ]
                if inline:
                    row_children.append(
                        html.Span(
                            "  → ", style={"color": "#a0aec0", "fontSize": "12px"}
                        )
                    )
                    for j, chip in enumerate(inline):
                        row_children.append(chip)
                        if j < len(inline) - 1:
                            row_children.append(
                                html.Span(
                                    ", ", style={"color": "#a0aec0"}
                                )
                            )
                item_children.append(
                    html.Li(
                        children=row_children,
                        style={"marginBottom": "8px", "lineHeight": "1.5"},
                    )
                )
            deferred_children.append(
                html.Ul(
                    item_children,
                    style={
                        "paddingLeft": "0",
                        "listStyleType": "none",
                        "margin": 0,
                    },
                )
            )

    # ---- Active threads
    active_threads = [
        r for r in threads
        if (r.thread_state or {}).get("state") in {"active", "planning"}
    ]
    threads_children: list[Any] = [_section_header("active threads")]
    if not active_threads:
        threads_children.append(
            html.Div(
                "(no active threads)",
                style={"fontSize": "13px", "color": "#718096"},
            )
        )
    else:
        for r in active_threads:
            ts = r.thread_state or {}
            threads_children.append(
                html.Div(
                    style={
                        "padding": "8px 0",
                        "borderBottom": "1px solid #edf2f7",
                    },
                    children=[
                        html.Div(
                            [
                                nav_link(TAB_REGISTRY, r.name, r.name),
                                html.Span(
                                    f"  ·  {ts.get('state')}",
                                    style={
                                        "fontSize": "11px",
                                        "color": "#4a5568",
                                        "fontFamily": "ui-monospace, monospace",
                                        "marginLeft": "6px",
                                    },
                                ),
                            ],
                            style={"marginBottom": "4px"},
                        ),
                        html.Div(
                            [
                                html.Span(
                                    "current: ",
                                    style={
                                        "fontSize": "12px",
                                        "color": "#4a5568",
                                        "fontFamily": "ui-monospace, monospace",
                                    },
                                ),
                                html.Span(
                                    str(ts.get("current_node_id") or "—"),
                                    style={
                                        "fontSize": "12px",
                                        "fontWeight": "600",
                                        "fontFamily": "ui-monospace, monospace",
                                    },
                                ),
                                html.Span(
                                    f"  ·  {ts.get('node_count', 0)} nodes"
                                    f"  ·  {ts.get('amendment_count', 0)} amendments",
                                    style={
                                        "fontSize": "11px",
                                        "color": "#718096",
                                        "marginLeft": "6px",
                                    },
                                ),
                            ]
                        ),
                        html.Div(
                            (str(ts.get("question") or "")[:160]
                             + ("…" if len(str(ts.get("question") or "")) > 160
                                else "")),
                            style={
                                "fontSize": "12px",
                                "color": "#4a5568",
                                "marginTop": "4px",
                                "lineHeight": "1.45",
                            },
                        ),
                    ],
                )
            )

    # ---- Recent SETTLED findings (top 5 by lex name = date-prefixed)
    settled = [r for r in corpus.registry if r.status == "SETTLED"]
    settled_recent = sorted(settled, key=lambda r: r.name, reverse=True)[:5]
    settled_children: list[Any] = [_section_header("recent SETTLED findings")]
    if not settled_recent:
        settled_children.append(
            html.Div(
                "(no settled findings yet)",
                style={"fontSize": "13px", "color": "#718096"},
            )
        )
    else:
        for r in settled_recent:
            settled_children.append(
                html.Div(
                    style={
                        "padding": "8px 0",
                        "borderBottom": "1px solid #edf2f7",
                    },
                    children=[
                        html.Div(
                            [
                                nav_link(TAB_REGISTRY, r.name, r.name),
                                _status_chip(r.status),
                            ],
                            style={
                                "display": "flex",
                                "justifyContent": "space-between",
                                "alignItems": "center",
                                "gap": "8px",
                                "marginBottom": "4px",
                            },
                        ),
                        html.Div(
                            r.verdict_summary or "",
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

    return html.Div(
        children=[
            html.Div(
                style={
                    "marginBottom": "8px",
                    "fontSize": "13px",
                    "color": "#4a5568",
                },
                children=[
                    html.Span(
                        "lab home  ",
                        style={
                            "fontWeight": "700",
                            "fontFamily": "ui-monospace, monospace",
                        },
                    ),
                    html.Span(
                        "·  synthesis of latest-session entry points, "
                        "active threads, recent verdicts.",
                        style={"color": "#718096"},
                    ),
                ],
            ),
            stats_row,
            _card_panel(deferred_children),
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "1fr 1fr",
                    "gap": "12px",
                    "alignItems": "start",
                },
                children=[
                    _card_panel(threads_children),
                    _card_panel(settled_children),
                ],
            ),
        ]
    )


def _scan_inline_refs(
    text: str,
    registry_keys: set[str],
    writeup_keys: set[str],
    memory_keys: set[str],
) -> list[Any]:
    """Find any registry/writeup/memory slug substrings in a deferred-
    item description and return them as ``nav_link`` chips.

    The deferred-item strings are free-text task descriptions; the
    nav-resolution is best-effort. Returns chips only for slugs that
    do resolve to a loaded entry — no dead links.
    """
    chips: list[Any] = []
    seen: set[tuple[str, str]] = set()

    # Registry topics: substring match against all known registry names.
    for k in registry_keys:
        if k in text and ("registry", k) not in seen:
            seen.add(("registry", k))
            chips.append(nav_link(TAB_REGISTRY, k, k))

    # Memory file names: match ``mitacs-foo.md`` / ``mitacs-foo`` /
    # ``MEMORY.md``.
    for k in memory_keys:
        if (k + ".md") in text or (k in text and (
            text.find(k) == 0 or not text[text.find(k) - 1].isalnum()
        )):
            key = ("memory", k)
            if key in seen:
                continue
            seen.add(key)
            chips.append(nav_link(TAB_NOTES, k, k))

    # Writeup stems: match ``writeup/tex/<stem>.tex`` or the bare stem.
    for k in writeup_keys:
        if k in text and ("writeup", k) not in seen:
            seen.add(("writeup", k))
            chips.append(nav_link(TAB_WRITEUPS, k, k))

    return chips


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
