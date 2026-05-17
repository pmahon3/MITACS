#!/usr/bin/env python3
"""
Vector‐Residual Explorer Dashboard (all‐times on slider)
=======================================================

Launch as:
    python dashboard.py --run-dir /path/to/outputs/<tag> --port 8050

This dashboard expects to find, in the given run‐directory:
  • residuals_<tag>.pt   (torch tensor of shape (N,K,d))
  • coeffs_<tag>.pt      (torch tensor of shape (N,K,d,d))
  • times_<tag>.npy      (NumPy int64 array of length N, representing nanoseconds)
  • thetas_<tag>.npy     (NumPy float64 array of length K)

It then lets you explore:
  • how the vector‐residual norm ∥r(tᵢ,θ)∥ varies with θ (Panel A)
  • the scatter of ∥r∥ vs ∥C∥ at a fixed tᵢ (Panel B)
  • per‐coordinate histograms of rₖ over all (tᵢ,θ) (Panel C)
  • per‐coordinate time‐series of rₖ(tᵢ,θ) at fixed θ (Panel D)
  • a full heatmap of ∥r(tᵢ,θ)∥ over (time‐index,θ) (Panel E)
  • summary footer showing best‐θ at the chosen tᵢ, global best‐θ, and
    summary stats of ∥r‖ at the currently selected θ.
"""

import argparse
import socket
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dash import Dash, dcc, html, Input, Output, callback
from logging import getLogger

from interface import load_vector_residuals, load_coefficients
from post_processing.dashboard.components import (
    plot_res_vs_theta,
    plot_res_vs_opnorm,
    plot_coord_histogram,
    plot_coord_scatter,
    plot_heatmap_residuals,
    build_footer
)

logger = getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# 1) CLI & Data Loading
# ──────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Launch Vector‐Residual Explorer")
parser.add_argument(
    "--run-dir",
    type=Path,
    required=True,
    help="Directory containing residuals_<tag>.pt, coeffs_<tag>.pt, times_<tag>.npy, thetas_<tag>.npy",
)
parser.add_argument("--port", type=int, default=8050, help="Port on which to run Dash")
args = parser.parse_args()

# Load residuals, times, thetas, and coefficients
vec_data = load_vector_residuals(args.run_dir)
residuals_np = vec_data["residuals"]  # shape = (N, K, d)
times_ns = vec_data["times"]  # shape = (N,) int64 nanoseconds
theta_grid = vec_data["thetas"]  # shape = (K,) float64
coeffs_np = load_coefficients(args.run_dir)  # shape = (N, K, d, d)

# Basic dimensions
N, K, d = residuals_np.shape
times_dt = pd.to_datetime(times_ns)  # pandas.DatetimeIndex of length N

# Precompute:
#   • vector‐norm of residuals (N,K)
res_norm_all = np.linalg.norm(residuals_np, axis=2)  # shape = (N, K)
#   • operator‐norm of coefficients (N,K)
coeffs_flat = coeffs_np.reshape((N, K, d * d))  # shape = (N, K, d*d)
op_norm_all = np.linalg.norm(coeffs_flat, axis=2)  # shape = (N, K)

# ──────────────────────────────────────────────────────────────────────────────
# Build “marks” for the sliders, using Quarters with Year on second line
# ──────────────────────────────────────────────────────────────────────────────

# 1) Walk through all timestamps and pick the first index of each (year, quarter)
quarter_indices = []
seen_quarters = set()
for i, ts in enumerate(times_dt):
    yq = (ts.year, ts.quarter)
    if yq not in seen_quarters:
        seen_quarters.add(yq)
        quarter_indices.append(i)

# 2) Build marks where each tick is labeled “Q<quarter>” on top line, “<year>” below
time_marks = {}
for i in quarter_indices:
    ts = times_dt[i]
    q = ts.quarter
    y = ts.year
    time_marks[int(i)] = f"Q{q}\n{y}"

# 3) Theta‐slider marks as before
if K <= 10:
    theta_marks = {int(j): f"{theta_grid[int(j)]:.2f}" for j in range(K)}
else:
    idxs = np.linspace(0, K - 1, 10, dtype=int)
    theta_marks = {int(j): f"{theta_grid[int(j)]:.2f}" for j in idxs}

# ──────────────────────────────────────────────────────────────────────────────
# 2) Build Dash Layout
# ──────────────────────────────────────────────────────────────────────────────
app = Dash(__name__)

children = [
    html.H2("Vector‐Residual Explorer"),
    html.H4(f"Embedding‐Dimension d = {d} | θ‐Grid size K = {K}"),

    # Time‐slider (index‐based)
    html.Div([
        html.Div("Select Time Index (i):", style={"display": "inline-block", "marginRight": "10px"}),
        dcc.Slider(
            id="time-slider",
            min=0,
            max=N - 1,
            value=0,
            step=1,
            marks=time_marks,
            updatemode="drag",
            tooltip={"placement": "bottom"}
        ),
    ], style={"marginBottom": "30px"}),

    # Panel Row 3: E (Heatmap)
    html.Div(dcc.Graph(id="plot‐E‐heatmap"), style={"marginBottom": "40px"}),

    # Theta‐slider (index‐based)
    html.Div([
        html.Div("Select θ Index (j):", style={"display": "inline-block", "marginRight": "10px"}),
        dcc.Slider(
            id="theta-slider",
            min=0,
            max=K - 1,
            value=0,
            step=1,
            marks=theta_marks,
            tooltip={"placement": "bottom"}
        ),
    ], style={"marginBottom": "30px"}),

    dcc.Store(id="y_range_store", data={"ymin": None, "ymax": None}),

    # Coordinate dropdown
    html.Div([
        html.Div("Select Coordinate k:", style={"display": "inline-block", "marginRight": "10px"}),
        dcc.Dropdown(
            id="coord-dd",
            options=[{"label": f"coord {int(k)}", "value": int(k)} for k in range(d)],
            value=0,
            clearable=False,
            style={"width": "200px"}
        ),
    ], style={"marginBottom": "30px"}),

    # Panel Row 1: A & B
    html.Div([
        html.Div(dcc.Graph(id="plot‐A‐res‐vs‐theta"), style={"width": "48%", "display": "inline-block"}),
        html.Div(dcc.Graph(id="plot‐B‐res‐vs‐opnorm"), style={"width": "48%", "display": "inline-block"}),
    ], style={"marginBottom": "40px"}),

    # Panel Row 2: C & D
    html.Div([
        html.Div(dcc.Graph(id="plot‐C‐coord‐hist"), style={"width": "48%", "display": "inline-block"}),
        html.Div(dcc.Graph(id="plot‐D‐coord‐ts"), style={"width": "48%", "display": "inline-block"}),
    ], style={"marginBottom": "40px"}),

    # Footer / summary
    html.Div(id="footer", style={"padding": "20px", "borderTop": "1px solid #ccc"}),
]

app.layout = html.Div(children, style={"width": "95%", "margin": "auto"})

# ──────────────────────────────────────────────────────────────────────────────
# 3) Callbacks (updated to add interactive highlights)
# ──────────────────────────────────────────────────────────────────────────────

from plotly import graph_objects as go


@callback(
    Output("plot‐A‐res‐vs‐theta", "figure"),
    Input("time-slider", "value"),  # time_idx (0 … N−1)
    Input("theta-slider", "value"),  # theta_idx (0 … K−1)
)
def update_plot_A(time_idx: int, theta_idx: int):
    """
    Panel A: ‖r(tᵢ, θ)‖ vs θ for a fixed tᵢ, but also highlight
    the chosen θₖ with a red dot.
    """
    # Find the norms for that time i
    norms_j = res_norm_all[time_idx, :]  # shape = (K,)

    # Build the base curve
    fig = go.Figure(go.Scatter(
        x=theta_grid,
        y=norms_j,
        mode="lines+markers",
        marker=dict(color="lightblue", size=6),
        name="all θ"
    ))

    # Add a red dot at the selected θₖ
    fig.add_trace(go.Scatter(
        x=[theta_grid[theta_idx]],
        y=[norms_j[theta_idx]],
        mode="markers",
        marker=dict(color="red", size=12),
        name=f"selected θ = {theta_grid[theta_idx]:.2f}"
    ))

    fig.update_layout(
        title=f"‖r(t={times_dt[time_idx].strftime('%Y-%m-%d %H:%M')}, θ)‖ vs θ",
        xaxis_title="θ",
        yaxis_title="‖r‖₂",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig


@callback(
    Output("plot‐B‐res‐vs‐opnorm", "figure"),
    Input("theta-slider", "value"),  # ONLY θ-index is needed now
)
def update_plot_B_all(theta_idx: int):
    """
    Panel B: scatter of ‖r‖ vs ‖C‖ for all (time, θ) pairs, plus a red overlay
    for the N points at the selected θ-index across all times.
    """
    # 1) Flatten all (‖C‖, ‖r‖) pairs across times and thetas
    x_all = op_norm_all.flatten()  # shape = (N*K,)
    y_all = res_norm_all.flatten()  # shape = (N*K,)

    fig = go.Figure()

    # 2) Plot every point as a small light gray dot
    fig.add_trace(go.Scatter(
        x=x_all,
        y=y_all,
        mode="markers",
        marker=dict(color="lightgray", size=4),
        name="all (time, θ)"
    ))

    # 3) Extract (‖C‖, ‖r‖) for the chosen θ-index across all times
    x_sel = op_norm_all[:, theta_idx]  # shape = (N,)
    y_sel = res_norm_all[:, theta_idx]  # shape = (N,)

    # 4) Overlay those N points in red, slightly larger
    fig.add_trace(go.Scatter(
        x=x_sel,
        y=y_sel,
        mode="markers",
        marker=dict(color="red", size=4),
        name=f"θ = {theta_grid[theta_idx]:.2f}"
    ))

    fig.update_layout(
        title="‖r‖ vs ‖C‖ over All (t, θ) Pairs",
        xaxis_title="‖C‖_F",
        yaxis_title="‖r‖₂",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig


@callback(
    Output("plot‐C‐coord‐hist", "figure"),
    Input("coord-dd",          "value"),
    Input("y_range_store",     "data")   # receive {"ymin":…, "ymax":…}
)
def update_plot_C(coord_idx: int, y_range_data: dict):
    # Extract ymin,ymax from the store (or default to None)
    yr = None
    if y_range_data and y_range_data.get("ymin") is not None:
        yr = (float(y_range_data["ymin"]), float(y_range_data["ymax"]))

    # Now draw histogram on the WEST (left) side, sharing y‐range = yr
    return plot_coord_histogram(
        residuals_np,
        coord_idx,
        direction="east",
        shared_axis_range=yr
    )



@callback(
    Output("plot‐D‐coord‐ts", "figure"),
    Output("y_range_store", "data"),     # <-- “Store” component to hold the y‐range
    Input("coord-dd",        "value"),
    Input("theta-slider",    "value"),
    Input("time-slider",     "value"),
)
def update_plot_D(coord_idx: int, theta_idx: int, time_idx: int):
    """
    Returns:
      1) Figure for the time‐series of r_k,
      2) A dict { "ymin": float, "ymax": float } so that histogram can match.
    """
    # 1) Extract full series
    y_full = residuals_np[:, theta_idx, coord_idx]
    mask = np.isfinite(y_full)
    x_all = times_dt[mask]
    y_all = y_full[mask]

    # Compute y‐range (pad by 5% for breathing room)
    pad = 0.05 * (np.nanmax(y_all) - np.nanmin(y_all) if np.nanmax(y_all) != np.nanmin(y_all) else 1.0)
    y_min, y_max = float(np.nanmin(y_all) - pad), float(np.nanmax(y_all) + pad)

    # 2) Build the time‐series figure (with vertical red line and red dot, as before)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_all,
        y=y_all,
        mode="markers",
        marker=dict(color="blue", size=4),
        opacity=0.8,
        name=f"r_{coord_idx}(t, θ={theta_grid[theta_idx]:.2f})"
    ))
    # Vertical red line at selected time
    fig.add_vline(
        x=times_dt[time_idx],
        line_width=1,
        line_color="red",
        layer="above"
    )
    # Red dot at selected point
    selected_y = residuals_np[time_idx, theta_idx, coord_idx]
    if np.isfinite(selected_y):
        fig.add_trace(go.Scatter(
            x=[times_dt[time_idx]],
            y=[selected_y],
            mode="markers",
            marker=dict(color="red", size=4),
            name="selected time"
        ))
    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="black")

    # Fix the vertical axis so it matches our computed y_min/y_max
    fig.update_yaxes(range=[y_min, y_max])
    fig.update_layout(
        title=f"r_{coord_idx}(t, θ={theta_grid[theta_idx]:.2f})",
        xaxis_title="Time",
        yaxis_title=f"r_{coord_idx}",
        margin=dict(l=40, r=40, t=40, b=40)
    )

    # 3) Return both the figure and the y‐range data
    return fig, {"ymin": y_min, "ymax": y_max}


@callback(
    Output("plot‐E‐heatmap", "figure"),
    Input("time-slider", "value"),
    Input("theta-slider", "value"),
)
def update_plot_E(time_idx: int, theta_idx: int):
    """
    Panel E: heatmap of ‖r(tᵢ,θⱼ)‖ over all i,j, plus:
      • a vertical red line at the selected time index,
      • a red dot at (selected time, selected θ).
    """
    # 1) Base heatmap
    fig = plot_heatmap_residuals(res_norm_all, times_dt, theta_grid)

    # 2) Vertical red line at chosen time
    fig.add_vline(
        x=times_dt[time_idx],
        line_width=2,
        line_color="red",
        opacity=0.8,
        name="selected time"
    )

    # 3) Red dot at (times_dt[time_idx], theta_grid[theta_idx])
    fig.add_trace(go.Scatter(
        x=[times_dt[time_idx]],
        y=[theta_grid[theta_idx]],
        mode="markers",
        marker=dict(color="red", size=10),
        name="selected θ"
    ))

    fig.update_layout(margin=dict(l=40, r=40, t=40, b=40))
    return fig


@callback(
    Output("footer", "children"),
    Input("time-slider", "value"),
    Input("theta-slider", "value"),
)
def update_footer_callback(time_idx: int, theta_idx: int):
    t_ns = int(times_ns[time_idx])
    return build_footer(
        res_norm_all,
        theta_grid,
        times_ns,
        times_dt,
        t_ns,
        theta_idx
    )


# ──────────────────────────────────────────────────────────────────────────────
# 4) Run the App
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    host = "0.0.0.0" if socket.gethostname() != "localhost" else "127.0.0.1"
    app.run(debug=True, host=host, port=args.port)
