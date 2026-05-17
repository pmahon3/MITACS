#!/usr/bin/env python3
"""
Forecast Dashboard
================================
Launch:

    python coeff_dashboard.py --run-dir ./outputs/<run_tag> [--port 8050]
"""
from __future__ import annotations
import argparse, socket
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, callback
from logging import getLogger

from interface import load_all  # post‑processing interface

logger = getLogger(__name__)

# ── CLI ────────────────────────────────────────────────────────────────────
cli = argparse.ArgumentParser()
cli.add_argument("--run-dir", type=Path, required=True)
cli.add_argument("--port", type=int, default=8050)
args = cli.parse_args()

# ── Load all artefacts ------------------------------------------------------
data = load_all(args.run_dir)
residuals_np = data["residuals"]  # (N,H,d)
ensembles = data["ensembles"]  # (N,M,H,d)
coeff_t = data["coeffs"]  # (N,H,d,d)
times = data["times"]  # (N,) datetime64[ns] or numeric
covs = data.get("covs")

# unify shapes ---------------------------------------------------------------
if residuals_np.ndim == 4:  # saved every member: take mean
    residuals_np = residuals_np.mean(axis=1)  # → (N,H,d)

N, M, H, d = ensembles.shape
mean_pred = ensembles.mean(axis=1)  # (N,H,d)
if M > 1:
    q05 = np.quantile(ensembles, 0.05, axis=1)
    q95 = np.quantile(ensembles, 0.95, axis=1)
else:
    q05 = q95 = mean_pred

# sort by time ---------------------------------------------------------------
order = np.argsort(times)
times_sort = times[order]
ts_ms = (times_sort.astype("int64") // 10 ** 6).astype(int)  # ms int key

residuals_np = residuals_np[order]
coeff_t = coeff_t[order]
ensembles = ensembles[order]
mean_pred = mean_pred[order]
q05, q95 = q05[order], q95[order]
if covs is not None:
    covs = covs[order]

# ── Metrics for scatter & histograms ────────────────────────────────────
#
# We define the *target variable* as coordinate 0 at horizon step 0.
# -----------------------------------------------------------------------

# 1) operator Frobenius norm per step
op_norm = np.linalg.norm(coeff_t, axis=(2, 3))          # shape (N,H)

# 2) signed residual of the target coord (vector element 0)
res_target = residuals_np[:, 0, 0]                      # shape (N,)

# 3) magnitude of full‑vector residual (for scatter)
res_norm = np.linalg.norm(residuals_np[:, 0, :], axis=1)  # shape (N,)

# 4) signed relative error  (residual / truth) with numerical safety
truth_target = mean_pred[:, 0, 0] - res_target          # shape (N,)

eps = 1e-8  # tolerance to avoid division blow‑ups near zero
rel_err = np.where(
    np.abs(truth_target) > eps,
    res_target / np.abs(truth_target),
    np.nan,                     # mark undefined when truth ~ 0
)

rel_err = rel_err[2 > rel_err]
rel_err = rel_err[-2 < rel_err]


# slider marks --------------------------------------------------------
mark_idx = np.linspace(0, N - 1, min(N, 10), dtype=int)
time_marks = {
    int(ts_ms[i]): str(pd.to_datetime(times_sort[i]))
    for i in mark_idx
}

# ── Dash app & layout -------------------------------------------------
app = Dash(__name__)
children = [
    html.H2("Local Linear Map Explorer"),
    html.H4(f"Ensemble size: {M},  Horizon: {H}"),
    dcc.Slider(
        id="time-slider",
        min=int(ts_ms.min()),
        max=int(ts_ms.max()),
        value=int(ts_ms[0]),
        step=None,
        marks=time_marks,
        updatemode="drag",
    ),
]
if H > 1:
    children.append(
        dcc.Slider(
            id="step-slider", min=0, max=H - 1, step=1, value=0,
            marks={h: str(h + 1) for h in range(H)},
            tooltip={"placement": "bottom"},
        )
    )

children += [
    # forecast fan chart
    html.Div(id="forecast-panel", children=[
        html.H4("H‑step Forecast"),
        dcc.Dropdown(
            id="coord-dd",
            options=[{"label": f"coord {k}", "value": k} for k in range(d)],
            value=0, clearable=False, style={"width": "200px"},
        ),
        dcc.Graph(id="forecast-fig"),
    ]),
    # matrix & eigenvalues
    html.Div([
        dcc.Graph(id="heatmap", style={"width": "48%", "display": "inline-block"}),
        dcc.Graph(id="eigscatter", style={"width": "48%", "display": "inline-block"}),
    ]),
    # scatter & histograms
    dcc.Graph(id="norm_scatter"),
    dcc.Graph(id="residual_hist"),
    dcc.Graph(id="relerr_hist"),  # ← NEW
]
app.layout = html.Div(children)


# ── Callbacks ---------------------------------------------------------------
def _idx_from_ts(ts_val: int) -> int:
    # convert ms‑integer slider value back to row index
    return int(np.where(ts_ms == ts_val)[0][0])


@callback(
    Output("heatmap", "figure"), Output("eigscatter", "figure"),
    Input("time-slider", "value"), Input("step-slider", "value"),
)
def update_matrix(ts_val: int, h: int | None):
    idx = _idx_from_ts(ts_val)
    h = 0 if h is None else h
    C = coeff_t[idx, h]
    ev = np.linalg.eigvals(C)

    hm = px.imshow(
        C, color_continuous_scale="RdBu", origin="lower",
        title=f"C @ {pd.to_datetime(times_sort[idx])} (step {h + 1})",
    )
    hm.update_layout(margin=dict(l=40, r=40, t=40, b=40))
    ev_fig = go.Figure(go.Scatter(x=ev.real, y=ev.imag, mode="markers"))
    ev_fig.update_layout(
        title="Eigenvalues", xaxis_title="Re", yaxis_title="Im",
        xaxis=dict(scaleanchor="y"),
    )
    return hm, ev_fig


@callback(
    Output("norm_scatter", "figure"),
    Input("time-slider", "value"), Input("step-slider", "value"),
)
def update_norm_scatter(ts_val: int, h: int | None):
    idx = _idx_from_ts(ts_val)
    h = 0 if h is None else h
    fig = px.scatter(
        x=op_norm[:, h], y=res_norm,
        labels={"x": f"‖C‖ᶠ(step {h + 1})", "y": "‖residual‖"},
        title="Residual vs Operator Norm",
    )
    fig.add_trace(
        go.Scatter(
            x=[op_norm[idx, h]], y=[res_norm[idx]],
            mode="markers", marker=dict(size=12, color="red"),
        )
    )
    return fig


# --- histogram of vector residual norm (coord aggregate) -------------------
@callback(
    Output("residual_hist", "figure"),
    Input("time-slider", "value"),
)
def update_residual_hist(ts_val: int):
    idx = _idx_from_ts(ts_val)
    vals = residuals_np[:, 0, 0]  # signed residual
    sel = vals[idx]

    counts, bins = np.histogram(vals, bins=100)
    centers = 0.5 * (bins[:-1] + bins[1:])
    colors = ["crimson" if bins[i] <= sel < bins[i + 1] else "lightgrey"
              for i in range(len(counts))]
    bar = go.Bar(x=centers, y=counts, marker_color=colors,
                 hovertemplate="Val:%{x:.3f}<br>Count:%{y}<extra></extra>")
    fig = go.Figure(bar)

    mean, med, std = vals.mean(), np.median(vals), vals.std()
    fig.add_annotation(
        x=0.02, y=0.95, xref="paper", yref="paper",
        text=(f"<b>Vector residual</b><br>"
              f"Mean={mean:.3f}<br>Med={med:.3f}<br>"
              f"Std={std:.3f}<br>Sel={sel:.3f}"),
        showarrow=False, align="left",
        bgcolor="rgba(255,255,255,0.7)",
    )
    fig.update_layout(
        title="Vector Residuals",
        xaxis_title="Residual value",
        yaxis_title="Count",
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


# --- NEW: histogram of signed relative error -------------------------------
@callback(
    Output("relerr_hist", "figure"),
    Input("time-slider", "value"),
)
def update_relerr_hist(ts_val: int):
    idx = _idx_from_ts(ts_val)
    vals = rel_err
    sel = vals[idx]

    counts, bins = np.histogram(vals, bins=100)
    centers = 0.5 * (bins[:-1] + bins[1:])
    colors = ["crimson" if bins[i] <= sel < bins[i + 1] else "lightgrey"
              for i in range(len(counts))]
    fig = go.Figure(
        go.Bar(
            x=centers, y=counts, marker_color=colors,
            hovertemplate="Val:%{x:.3f}<br>Count:%{y}<extra></extra>"
        )
    )
    mean, med, std = np.nanmean(vals), np.nanmedian(vals), np.nanstd(vals)
    fig.add_annotation(
        x=0.02, y=0.95, xref="paper", yref="paper",
        text=(f"<b>Relative‑error stats</b><br>"
              f"Mean={mean:.3f}<br>Med={med:.3f}<br>"
              f"Std={std:.3f}<br>Sel={sel:.3f}"),
        showarrow=False, align="left",
        bgcolor="rgba(255,255,255,0.7)",
    )
    fig.update_layout(
        title="Signed Relative Error (step 0, coord 0)",
        xaxis_title="Residual / Truth",
        yaxis_title="Count",
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


# --- forecast fan chart -----------------------------------------------------
@callback(
    Output("forecast-fig", "figure"),
    Input("time-slider", "value"),
    Input("coord-dd", "value"),
)
def update_forecast(ts_val: int, coord: int):
    idx = _idx_from_ts(ts_val)
    steps = np.arange(1, H + 1)
    mean_y = mean_pred[idx, :, coord]
    fig = go.Figure()
    if M > 1:
        fig.add_trace(go.Scatter(x=steps, y=q05[idx, :, coord],
                                 line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=steps, y=q95[idx, :, coord],
                                 line=dict(width=0), fill="tonexty",
                                 fillcolor="rgba(0,100,200,0.2)",
                                 name="90% band"))
    fig.add_trace(go.Scatter(x=steps, y=mean_y, name="mean",
                             line=dict(width=2)))
    fig.update_layout(
        title=f"Forecast (coord {coord})",
        xaxis_title="Step", yaxis_title="Value",
    )
    return fig


# ── Run app -----------------------------------------------------------------
if __name__ == "__main__":
    host = "0.0.0.0" if socket.gethostname() != "localhost" else "127.0.0.1"
    app.run(debug=True, host=host, port=args.port)
