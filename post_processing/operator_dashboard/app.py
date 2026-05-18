"""Operator analysis dashboard — the central objects connecting the
Ontario data to the kappa_Q / Resolvent_Framework theory.

NOT a forecast-vs-IESO scoreboard (that comparison is a separate, later
section; the retrospective head-to-head is infeasible anyway — see the
paper draft §6). This surfaces the *analysis objects*: the local drift
spectrum and the bandwidth field as the system traverses its
reconstructed orbit.

Launch::

    python -m post_processing.operator_dashboard.app \
        --output-dir processing/innovations/outputs [--port 8050]

Reads the current per-day-type estimator artifacts as-is and reports
their provenance state plainly in a banner (a viewer displays objects;
it does not assert claims, so it does not refuse unprovenanced input —
decision 2026-05-18).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from dash import Dash, dcc, html
from dash.dependencies import Input, Output

from .figures import (
    diffusion_scale_figure,
    drift_eigen_plane_figure,
    drift_spectrum_figure,
    theta_over_trajectory_figure,
)
from .loader import available_daytypes, load_daytype, provenance_state


def _banner(state: str) -> html.Div:
    unprovenanced = state.startswith("UNPROVENANCED")
    return html.Div(
        f"ARTIFACT PROVENANCE: {state}",
        style={
            "padding": "8px 14px",
            "marginBottom": "10px",
            "borderRadius": "4px",
            "fontFamily": "monospace",
            "fontSize": "13px",
            "color": "#5c3c00" if unprovenanced else "#13502b",
            "background": "#fff3cd" if unprovenanced else "#d4edda",
            "border": "1px solid "
            + ("#e0c068" if unprovenanced else "#9ad0ad"),
        },
    )


def build_app(output_dir: Path) -> Dash:
    dts = available_daytypes(output_dir)
    if not dts:
        raise FileNotFoundError(
            f"no estimator artifacts under {output_dir} — run the "
            f"innovations stage first"
        )
    state = provenance_state(output_dir)

    app = Dash(__name__, title="Operator analysis")
    app.layout = html.Div(
        style={"maxWidth": "1100px", "margin": "0 auto",
               "fontFamily": "system-ui, sans-serif"},
        children=[
            html.H2("Operator analysis — data ↔ theory objects"),
            html.P(
                "Local drift spectrum and bandwidth as the system "
                "traverses its reconstructed delay-embedding orbit "
                "(anchors ordered by time). Not a forecast scoreboard."
            ),
            _banner(state),
            html.Div(
                [
                    html.Label("Day-type:", style={"marginRight": "8px"}),
                    dcc.RadioItems(
                        id="daytype",
                        options=[{"label": d, "value": d} for d in dts],
                        value=dts[0],
                        inline=True,
                    ),
                ],
                style={"margin": "12px 0"},
            ),
            dcc.Graph(id="drift-spectrum"),
            dcc.Graph(id="drift-plane"),
            dcc.Graph(id="theta-traj"),
            dcc.Graph(id="diffusion-scale"),
            html.Hr(),
            html.P(
                "Follow-up (tracked): a true reconstructed-attractor "
                "scatter (re-deriving anchor coordinates from the "
                "z-score series under the frozen climatology) and the "
                "rank-1 / heavy-tail Gaussian-proxy view.",
                style={"fontSize": "12px", "color": "#666"},
            ),
        ],
    )

    @app.callback(
        Output("drift-spectrum", "figure"),
        Output("drift-plane", "figure"),
        Output("theta-traj", "figure"),
        Output("diffusion-scale", "figure"),
        Input("daytype", "value"),
    )
    def _update(daytype: str):
        a = load_daytype(output_dir, daytype)
        return (
            drift_spectrum_figure(a),
            drift_eigen_plane_figure(a),
            theta_over_trajectory_figure(a),
            diffusion_scale_figure(a),
        )

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("processing/innovations/outputs"),
        help="dir containing coeffs_<dt>.pt etc.",
    )
    ap.add_argument("--port", type=int, default=8050)
    args = ap.parse_args()
    app = build_app(args.output_dir)
    app.run(debug=False, port=args.port)


if __name__ == "__main__":
    main()
