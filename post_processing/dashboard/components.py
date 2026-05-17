import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash import html


def plot_res_vs_theta(res_norm_all, times_ns, times_dt, theta_grid, ts_ns_val):
    i = int(np.where(times_ns == ts_ns_val)[0][0])
    norms_j = res_norm_all[i, :]
    fig = go.Figure(go.Scatter(x=theta_grid, y=norms_j, mode="lines+markers"))
    fig.update_layout(
        title=f"‖r(t={times_dt[i]}, θ)‖ vs θ",
        xaxis_title="θ", yaxis_title="‖r‖₂"
    )
    return fig


def plot_res_vs_opnorm(res_norm_all, op_norm_all, theta_grid, times_ns, times_dt, ts_ns_val, theta_idx):
    i = int(np.where(times_ns == ts_ns_val)[0][0])
    j = theta_idx
    op_all, r_all = op_norm_all[i, :], res_norm_all[i, :]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=op_all, y=r_all, mode="markers", marker=dict(color="lightblue", size=6)))
    fig.add_trace(go.Scatter(x=[op_all[j]], y=[r_all[j]], mode="markers", marker=dict(color="crimson", size=12)))
    fig.update_layout(title=f"‖r‖ vs ‖C‖ at t={times_dt[i]}", xaxis_title="‖C‖", yaxis_title="‖r‖₂")
    return fig


import numpy as np
import plotly.express as px


def plot_coord_histogram(
        residuals_np: np.ndarray,
        coord_idx: int,
        direction: str = "west",
        shared_axis_range: tuple[float, float] | None = None
):
    """
    Flexible histogram of r_k over all (t,θ), placed on one of four sides.

    Parameters
    ----------
    residuals_np : np.ndarray
        Array of shape (N, K, d) containing residuals.
    coord_idx : int
        Which coordinate k to histogram.
    direction : {"west", "east", "north", "south"}
        Where to orient the histogram relative to a central time‐series plot:
          - "west": histogram on the left, horizontal bars (runs left→right).
          - "east": histogram on the right, horizontal bars (runs right→left).
          - "north": histogram on top, vertical bars (runs bottom→top).
          - "south": histogram on bottom, vertical bars (runs top→bottom).
    shared_axis_range : (min, max) or None
        If not None, use this to set the histogram’s “value‐axis” range so
        that zero lines line up with another plot that uses the same range.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    # Flatten residuals for coordinate k over all t,j
    arr_all = residuals_np[:, :, coord_idx].reshape(-1)
    arr_all = arr_all[np.isfinite(arr_all)]

    if direction not in ("west", "east", "north", "south"):
        raise ValueError(f"direction must be one of "
                         f"{{'west','east','north','south'}}, got '{direction}'")

    # Helper to apply shared range if provided
    def apply_shared_range(fig, axis: str):
        if shared_axis_range is not None:
            lo, hi = shared_axis_range
            if axis == "x":
                fig.update_xaxes(range=[lo, hi])
            elif axis == "y":
                fig.update_yaxes(range=[lo, hi])

    # WEST: histogram on left, horizontal bars increasing to the right
    if direction == "west":
        fig = px.histogram(
            y=arr_all,
            nbins=50,
            orientation="h",
            labels={"y": f"r_{coord_idx}", "x": "Count"},
            title=f"Histogram of r_{coord_idx}",
        )
        # Flip so smaller residuals appear at bottom (optional)
        fig.update_layout(yaxis=dict(autorange="reversed"))
        # If shared_axis_range was provided, align the histogram’s Y‐axis (values)
        apply_shared_range(fig, axis="y")

    # EAST: histogram on right, horizontal bars but reversed (bars run right→left)
    elif direction == "east":
        fig = px.histogram(
            y=arr_all,
            nbins=50,
            orientation="h",
            labels={"y": f"r_{coord_idx}", "x": "Count"},
            title=f"Histogram of r_{coord_idx}",
        )
        # Reverse the x-axis so bars extend leftward
        fig.update_layout(xaxis=dict(autorange="reversed"))
        # Set shared Y‐axis range if given
        apply_shared_range(fig, axis="y")

    # NORTH: histogram on top, vertical bars rising upward
    elif direction == "north":
        fig = px.histogram(
            x=arr_all,
            nbins=50,
            orientation="v",
            labels={"x": f"r_{coord_idx}", "y": "Count"},
            title=f"Histogram of r_{coord_idx}",
        )
        # Set shared X‐axis range if given
        apply_shared_range(fig, axis="x")

    # SOUTH: histogram on bottom, vertical bars but reversed (bars fall downward)
    else:  # direction == "south"
        fig = px.histogram(
            x=arr_all,
            nbins=50,
            orientation="v",
            labels={"x": f"r_{coord_idx}", "y": "Count"},
            title=f"Histogram of r_{coord_idx}",
        )
        # Reverse the y-axis so bars go down
        fig.update_layout(yaxis=dict(autorange="reversed"))
        # Set shared X‐axis range if given
        apply_shared_range(fig, axis="x")

    fig.update_layout(margin=dict(l=40, r=40, t=40, b=40))
    return fig


def plot_coord_timeseries(residuals_np, coord_idx, theta_idx, times_dt, theta_val):
    y = residuals_np[:, theta_idx, coord_idx]
    mask = np.isfinite(y)
    fig = go.Figure(go.Scatter(x=times_dt[mask], y=y[mask], mode="lines+markers"))
    fig.add_hline(y=0, line_dash="dash", line_color="black")
    fig.update_layout(title=f"r_{coord_idx}(t, θ={theta_val:.2f})", xaxis_title="Time", yaxis_title=f"r_{coord_idx}")
    return fig


def plot_coord_scatter(residuals_np, coord_idx, theta_idx, times_dt, theta_val):
    y = residuals_np[:, theta_idx, coord_idx]
    mask = np.isfinite(y)
    fig = go.Figure(go.Scatter(x=times_dt[mask], y=y[mask], mode="markers"))
    fig.add_hline(y=0, line_dash="dash", line_color="black")
    fig.update_layout(title=f"Scatter of r_{coord_idx}(t, θ={theta_val:.2f})", xaxis_title="Time",
                      yaxis_title=f"r_{coord_idx}")
    return fig


def plot_heatmap_residuals(res_norm_all, times_dt, theta_grid):
    fig = px.imshow(
        res_norm_all.T, x=times_dt, y=np.round(theta_grid, 3), origin="lower",
        color_continuous_scale="Viridis", aspect="auto", labels={"x": "Time", "y": "θ", "color": "‖r‖₂"},
        title="Heatmap of Residual Norm ‖r(t,θ)‖"
    )
    return fig


def build_footer(res_norm_all, theta_grid, times_ns, times_dt, ts_ns_val, theta_idx):
    i = int(np.where(times_ns == ts_ns_val)[0][0])
    j = theta_idx

    best_j = int(np.nanargmin(res_norm_all[i, :])) if np.any(np.isfinite(res_norm_all[i, :])) else None
    best_local = theta_grid[best_j] if best_j is not None else None
    best_global_idx = int(np.nanargmin(np.nanmean(res_norm_all, axis=0)))
    best_global = theta_grid[best_global_idx]

    valid_j = res_norm_all[:, j][np.isfinite(res_norm_all[:, j])]
    if valid_j.size == 0:
        m_mean = m_med = m_90 = float("nan")
    else:
        m_mean, m_med, m_90 = np.mean(valid_j), np.median(valid_j), np.percentile(valid_j, 90)

    return [
        html.Div(
            f"Best θ at t={times_dt[i].strftime('%Y-%m-%d %H:%M')}  → {best_local:.3f}" if best_local else "Best θ at this time → N/A"),
        html.Div(f"Global best θ (min mean ‖r‖) → {best_global:.3f}"),
        html.Div(f"For θ={theta_grid[j]:.3f} :  mean={m_mean:.3f},  median={m_med:.3f},  90th-pct={m_90:.3f}")
    ]
