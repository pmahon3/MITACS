"""The analysis-object figures: the things that connect data to theory.

Two core objects (user-selected as the must-haves):

1. **Drift spectrum over state space** — eig(C_j) as the system
   traverses its reconstructed orbit (anchors ordered by time). The
   eigenvalues of the local linear map ARE the local linearised
   dynamics: |lambda| vs 1 is the local expansion/contraction, the
   Koopman / kappa_Q conditional-mean content made visible across the
   trajectory. This is the deepest data<->theory link.

2. **Bandwidth theta over state space** — theta_j along the same
   trajectory axis. Where the estimator localises tight vs loose; ties
   to the LOO-CV rule and the honing question. If theta is constant in
   a run this view shows a FLAT field — that is the data speaking (the
   honing diagnostics found per-anchor theta adaptivity weak), not a
   bug, and the figure says so rather than hiding it.

"Over state space" axis: the anchor embedding coordinates are not
persisted, so anchors are ordered ALONG THE RECONSTRUCTED TRAJECTORY by
timestamp — the natural traversal of a delay-embedded orbit. A true
attractor scatter (re-deriving coordinates from the z-score series) is
a tracked follow-up.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .loader import DaytypeArtifacts


def drift_spectrum_figure(a: DaytypeArtifacts) -> go.Figure:
    """eig(C_j) along the trajectory. One trace per eigenvalue index
    (sorted by modulus per anchor so traces are continuous), |lambda|
    on y, time on x, with the unit-modulus stability line at 1."""
    ev = a.drift_eigvals()  # (N, d) complex
    mod = np.abs(ev)
    # sort eigenvalues by modulus within each anchor so trace k is
    # consistently "the k-th largest mode" along the trajectory
    mod_sorted = np.sort(mod, axis=1)[:, ::-1]  # (N, d) descending

    fig = go.Figure()
    for k in range(a.d):
        fig.add_trace(
            go.Scattergl(
                x=a.times,
                y=mod_sorted[:, k],
                mode="lines",
                name=f"|λ|<sub>{k + 1}</sub>",
                line=dict(width=1),
            )
        )
    fig.add_hline(
        y=1.0,
        line=dict(color="black", width=1, dash="dash"),
        annotation_text="|λ| = 1 (local stability boundary)",
        annotation_position="top left",
    )
    fig.update_layout(
        title=(
            f"Drift spectrum over the reconstructed trajectory "
            f"— {a.daytype} (d={a.d}, N={a.n})"
        ),
        xaxis_title="anchor time (trajectory order)",
        yaxis_title="|eigenvalue| of local drift C_j",
        legend=dict(orientation="h", y=1.02, yanchor="bottom"),
        margin=dict(l=60, r=30, t=60, b=50),
        height=420,
    )
    return fig


def drift_eigen_plane_figure(a: DaytypeArtifacts) -> go.Figure:
    """All eig(C_j) on the complex plane, coloured by trajectory time
    — the spectral cloud of the local dynamics with the unit circle.
    Inside the circle = locally contracting modes."""
    ev = a.drift_eigvals()  # (N, d)
    t = a.times.astype("int64").astype("float64")
    tnorm = (t - t.min()) / (np.ptp(t) or 1.0)
    re = ev.real.reshape(-1)
    im = ev.imag.reshape(-1)
    c = np.repeat(tnorm, a.d)

    th = np.linspace(0, 2 * np.pi, 200)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.cos(th), y=np.sin(th), mode="lines",
            line=dict(color="black", width=1, dash="dash"),
            name="unit circle", hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=re, y=im, mode="markers",
            marker=dict(
                size=4, color=c, colorscale="Viridis",
                colorbar=dict(title="time →"), opacity=0.6,
            ),
            name="eig(C_j)",
        )
    )
    fig.update_layout(
        title=f"Local-drift eigenvalues on ℂ — {a.daytype}",
        xaxis_title="Re(λ)", yaxis_title="Im(λ)",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        margin=dict(l=60, r=30, t=60, b=50), height=420,
    )
    return fig


def theta_over_trajectory_figure(a: DaytypeArtifacts) -> go.Figure:
    """theta_j along the trajectory. Honestly flags a constant field."""
    th = a.thetas
    spread = float(np.ptp(th))
    is_flat = spread < 1e-9 * (abs(float(th.mean())) + 1e-12)

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=a.times, y=th, mode="lines",
            line=dict(width=1), name="θ_j",
        )
    )
    if is_flat:
        note = (
            f"θ is CONSTANT (= {th[0]:.4g}) across all {a.n} anchors "
            f"in this run — per-anchor θ-adaptivity is inactive here. "
            f"This is the data, not a plotting artifact (cf. the "
            f"honing diagnostic: state-space-adaptive θ diagnosed weak)."
        )
    else:
        note = (
            f"θ varies over the trajectory (range {th.min():.4g}–"
            f"{th.max():.4g}, spread {spread:.4g})."
        )
    fig.update_layout(
        title=f"Bandwidth θ over the trajectory — {a.daytype}",
        xaxis_title="anchor time (trajectory order)",
        yaxis_title="θ_j (LOO-CV-selected bandwidth)",
        annotations=[
            dict(
                text=note, xref="paper", yref="paper",
                x=0, y=1.0, xanchor="left", yanchor="bottom",
                showarrow=False, font=dict(size=11),
            )
        ],
        margin=dict(l=60, r=30, t=70, b=50), height=380,
    )
    return fig


def diffusion_scale_figure(a: DaytypeArtifacts) -> go.Figure:
    """Companion context: the largest eig(Sigma_j) along the
    trajectory — the scalar innovation scale (Sigma_j is rank-1 by
    construction, so the top eigenvalue IS the innovation variance).
    Connects θ and the drift spectrum to the §3 Gaussian-proxy object
    without yet drawing the heavy-tail view (deferred)."""
    top = a.eigvals[:, -1]  # ascending -> last is largest
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=a.times, y=top, mode="lines",
            line=dict(width=1), name="max eig(Σ_j)",
        )
    )
    fig.update_layout(
        title=(
            f"Diffusion scale (top eig Σ_j ≈ scalar innovation "
            f"variance; Σ_j rank-1 by construction) — {a.daytype}"
        ),
        xaxis_title="anchor time (trajectory order)",
        yaxis_title="largest eigenvalue of Σ_j",
        margin=dict(l=60, r=30, t=60, b=50), height=340,
    )
    return fig
