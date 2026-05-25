"""Render the four figures used in writeup/tex/draft_body.tex.

All inputs are committed-to-repo artifacts:

  scratch/data/smc_smap_samples/cell_*.pkl       -- per-cell summaries
  scratch/data/smc_smap_samples/samples_*.npz    -- (D, M, H) particles
  scratch/data/smc_smap_samples/distrib/*.csv    -- distributional CSVs
  scratch/data/smc_smap_samples/distrib/ck_*.csv -- Ontario CK probe

Outputs are PDF, written to writeup/tex/figs/, sized for a 6.5-in
text column (the article geometry of writeup/tex/draft.tex):

  fig_per_horizon_mae.pdf      -- Figure 1, \\S 4.4
  fig_multisigma_coverage.pdf  -- Figure 2, \\S 4.5
  fig_pit_panel.pdf            -- Figure 3, \\S 4.5
  fig_ck_divergence.pdf        -- Figure 4, \\S 5

The CK divergence figure additionally re-runs the synthetic VAR(1)
gate to produce the baseline row, since that gate's per-h numbers
are not committed (the gate prints; the script reads stdout).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# paper-quality rcParams (serif font, sized for the 6.5-in column)
plt.rcParams.update({
    "font.family":         "serif",
    "font.size":           9,
    "axes.labelsize":      9,
    "axes.titlesize":      9,
    "legend.fontsize":     7.5,
    "xtick.labelsize":     8,
    "ytick.labelsize":     8,
    "axes.linewidth":      0.7,
    "lines.linewidth":     1.1,
    "lines.markersize":    3.0,
    "grid.linewidth":      0.4,
    "grid.alpha":          0.35,
    "figure.dpi":          150,
    "savefig.dpi":         300,
    "savefig.bbox":        "tight",
    "savefig.pad_inches":  0.02,
    "pdf.fonttype":        42,  # embed type-3-free fonts
})

ROOT = Path(__file__).resolve().parent.parent
SAMP = ROOT / "scratch" / "data" / "smc_smap_samples"
DIST = SAMP / "distrib"
FIGS = ROOT / "writeup" / "tex" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)


# canonical cell ordering + colour palette (kept consistent across figures)
CELLS = [
    ("A", "meaniter_prod",     "A. mean-iter, prod-$\\theta$"),
    ("B", "meaniter_global",   "B. mean-iter, $\\theta{=}0$"),
    ("C", "smc_global_gauss",  "C. SMC, $\\theta{=}0$, Gaussian"),
    ("D", "smc_global_emp",    "D. SMC, $\\theta{=}0$, empirical"),
    ("E", "smc_prod_gauss",    "E. SMC, prod-$\\theta$, Gaussian"),
    ("F", "smc_prod_emp",      "F. SMC, prod-$\\theta$, empirical"),
]
# Distinct hue per propagation/innovation combo
PALETTE = {
    "A": "#404040",   # mean-iter, neutral
    "B": "#808080",
    "C": "#2c7fb8",   # SMC Gaussian, cool blue
    "D": "#d95f0e",   # SMC empirical, warm orange
    "E": "#41b6c4",
    "F": "#fec44f",
}
LINESTYLE = {
    "A": "-",  "B": "--",
    "C": "-",  "D": "-",
    "E": "--", "F": "--",
}


# ---------------------------------------------------------------------------
# Figure 1: per-horizon MAE, all 6 cells
# ---------------------------------------------------------------------------
def fig_per_horizon_mae() -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    for code, tag, label in CELLS:
        df = pd.read_pickle(SAMP / f"cell_{code}_{tag}.pkl")
        per_h = (df.assign(abs_err=(df["our_forecast_mw"] - df["actual_mw"]).abs())
                   .groupby("horizon_h")["abs_err"].mean())
        ax.plot(per_h.index, per_h.values,
                color=PALETTE[code], linestyle=LINESTYLE[code],
                label=label, marker="o")
    ax.set_xlabel("horizon $h$ (hours)")
    ax.set_ylabel("MAE (MW)")
    ax.set_xlim(0.5, 24.5)
    ax.set_xticks([1, 4, 8, 12, 16, 20, 24])
    ax.grid(True)
    ax.legend(loc="lower right", ncol=2, frameon=False)
    out = FIGS / "fig_per_horizon_mae.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 2: multi-sigma coverage curves per cell (4 panels, SMC cells only)
# ---------------------------------------------------------------------------
def fig_multisigma_coverage() -> Path:
    cov = pd.read_csv(DIST / "coverage_multisigma.csv")
    # pool over horizons and daytypes (the all/all row): one curve per cell
    smc = [(c, t, l) for c, t, l in CELLS if c in ("C", "D", "E", "F")]
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.2), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, (code, tag, label) in zip(axes, smc):
        sub = cov[(cov["cell"] == code) & (cov["daytype"] == "all")]
        emp = sub.groupby("level")["cov_emp"].mean()
        ax.plot([0, 1], [0, 1], color="k", linewidth=0.5, linestyle=":",
                label="nominal")
        ax.plot(emp.index, emp.values, color=PALETTE[code],
                marker="o", label="empirical")
        ax.set_title(label, fontsize=8.5)
        ax.set_xlim(0.45, 1.01)
        ax.set_ylim(0.45, 1.01)
        ax.set_xticks([0.5, 0.683, 0.8, 0.9, 0.95, 0.99])
        ax.set_xticklabels(["50", "68", "80", "90", "95", "99"], fontsize=7)
        ax.set_yticks([0.5, 0.683, 0.8, 0.9, 0.95, 0.99])
        ax.set_yticklabels(["50", "68", "80", "90", "95", "99"], fontsize=7)
        ax.grid(True)
        if ax in (axes[2], axes[3]):
            ax.set_xlabel("nominal coverage (\\%)")
        if ax in (axes[0], axes[2]):
            ax.set_ylabel("empirical coverage (\\%)")
    axes[0].legend(loc="upper left", frameon=False)
    out = FIGS / "fig_multisigma_coverage.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 3: PIT 2x2 panel (pooled-all per cell), built from samples directly
# ---------------------------------------------------------------------------
def _pit_for_cell(code: str, tag: str, rng: np.random.Generator) -> np.ndarray:
    npz = np.load(SAMP / f"samples_{code}_{tag}.npz", allow_pickle=False)
    sm = npz["samples"]                                           # (D, M, H)
    dates = pd.to_datetime(npz["delivery_dates"]).normalize()
    pkl = pd.read_pickle(SAMP / f"cell_{code}_{tag}.pkl")
    pkl["delivery_date"] = pkl["delivery_date"].dt.normalize()
    a = (pkl.pivot_table(index="delivery_date", columns="horizon_h",
                         values="actual_mw", aggfunc="first")
            .sort_index()
            .reindex(dates).to_numpy().astype(np.float64))
    D, M, H = sm.shape
    less = (sm < a[:, None, :]).sum(axis=1)
    eq   = (sm == a[:, None, :]).sum(axis=1)
    u = rng.uniform(size=(D, H))
    return ((less + u * (eq + 1.0)) / (M + 1.0)).ravel()


def fig_pit_panel() -> Path:
    rng = np.random.default_rng(0xC0FFEE)
    smc = [(c, t, l) for c, t, l in CELLS if c in ("C", "D", "E", "F")]
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.2), sharex=True, sharey=True)
    axes = axes.flatten()
    n_bins = 20
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for ax, (code, tag, label) in zip(axes, smc):
        pit = _pit_for_cell(code, tag, rng)
        counts, _ = np.histogram(pit, bins=edges)
        density = counts / pit.size
        ax.bar((edges[:-1] + edges[1:]) / 2, density, width=1.0 / n_bins,
               edgecolor="black", linewidth=0.3, color=PALETTE[code],
               alpha=0.85)
        ax.axhline(1.0 / n_bins, color="black", linestyle=":", linewidth=0.7,
                   label="uniform")
        ax.set_title(label, fontsize=8.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, max(0.10, 1.4 / n_bins))
        if ax in (axes[2], axes[3]):
            ax.set_xlabel("PIT")
        if ax in (axes[0], axes[2]):
            ax.set_ylabel("density")
        ax.grid(True, axis="y")
    out = FIGS / "fig_pit_panel.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 4: CK consistency divergence, synthetic baseline + 3 Ontario day-types
# ---------------------------------------------------------------------------
def _synthetic_ck_curve() -> pd.DataFrame:
    """Re-run the synthetic VAR(1) gate to get per-h errors.

    Same generative process as processing.innovations.validation.ck_consistency
    (n = 4000, seed 12345). Reports absolute Frobenius drift error
    (the gate's own pass criterion is an absolute tolerance) and
    relative diffusion error (the diffusion scale ||Q_iter|| does not
    shrink with h on a stable VAR(1), so relative is meaningful).
    """
    from processing.innovations.estimator import _local_fit_at
    from processing.innovations.validation.ck_consistency import (
        _generate_var1, _build_lagged_pairs, iterate_linear_gaussian,
    )
    rng = np.random.default_rng(12345)
    A = np.array([[0.70, 0.10, 0.00],
                  [-0.05, 0.65, 0.20],
                  [0.00, -0.10, 0.55]])
    Q = np.array([[1.00, 0.20, 0.05],
                  [0.20, 0.80, 0.10],
                  [0.05, 0.10, 0.60]])
    x = _generate_var1(A, Q, 4000, rng)
    rows = []
    for h in (1, 2, 4, 8, 12, 24):
        Xh, Yh = _build_lagged_pairs(x, h)
        Ch, Sh, *_ = _local_fit_at(Xh, Yh, Xh.mean(axis=0), A.shape[0])
        Ah, Qiter = iterate_linear_gaussian(A, Q, h)
        rows.append({
            "horizon":   h,
            "drift_abs": np.linalg.norm(Ch - Ah, "fro"),
            "diff_rel":  np.linalg.norm(Sh - Qiter, "fro") / max(np.linalg.norm(Qiter, "fro"), 1e-12),
        })
    return pd.DataFrame(rows)


def fig_ck_divergence() -> Path:
    syn = _synthetic_ck_curve()
    ont = {dt: pd.read_csv(DIST / f"ck_ontario_{dt}.csv")
           for dt in ("weekday", "saturday", "sunday")}
    fig, (ax_d, ax_s) = plt.subplots(1, 2, figsize=(6.5, 3.0), sharex=True)
    palette = {
        "synthetic": "#404040",
        "weekday":   "#2c7fb8",
        "saturday":  "#d95f0e",
        "sunday":    "#7b3294",
    }
    markers = {"synthetic": "x", "weekday": "o", "saturday": "s", "sunday": "D"}
    # left: absolute Frobenius drift error -- shared units across systems
    ax_d.plot(syn["horizon"], syn["drift_abs"], color=palette["synthetic"],
              marker=markers["synthetic"], linestyle="--",
              label="synthetic VAR(1)")
    for dt, df in ont.items():
        ax_d.plot(df["horizon"], df["drift_err"], color=palette[dt],
                  marker=markers[dt], label=f"Ontario {dt}")
    ax_d.set_yscale("log")
    ax_d.set_xlabel("horizon $h$")
    ax_d.set_ylabel("drift error $\\|\\hat C_{h\\Delta}-(\\hat C_\\Delta)^h\\|_F$")
    ax_d.set_xticks([1, 2, 4, 8, 12, 24])
    ax_d.grid(True, which="both")
    ax_d.legend(loc="lower right", frameon=False)

    # right: relative diffusion error -- well-defined across systems
    # (||Sigma_iter^h|| does not shrink to 0; stationary process)
    ax_s.plot(syn["horizon"], syn["diff_rel"], color=palette["synthetic"],
              marker=markers["synthetic"], linestyle="--",
              label="synthetic VAR(1)")
    for dt, df in ont.items():
        ax_s.plot(df["horizon"], df["diff_rel"], color=palette[dt],
                  marker=markers[dt], label=f"Ontario {dt}")
    ax_s.set_yscale("log")
    ax_s.set_xlabel("horizon $h$")
    ax_s.set_ylabel("diffusion rel err $\\|\\hat\\Sigma_{h\\Delta}-\\Sigma^{\\mathrm{iter}}_h\\|_F\\,/\\,\\|\\Sigma^{\\mathrm{iter}}_h\\|_F$")
    ax_s.set_xticks([1, 2, 4, 8, 12, 24])
    ax_s.grid(True, which="both")

    out = FIGS / "fig_ck_divergence.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    for fn in (fig_per_horizon_mae,
               fig_multisigma_coverage,
               fig_pit_panel,
               fig_ck_divergence):
        path = fn()
        print(f"  wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
