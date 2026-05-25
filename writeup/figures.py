"""Render the figures used in writeup/tex/draft_body.tex.

All inputs are committed-to-repo artifacts:

  scratch/data/smc_smap_samples/cell_*.pkl       -- per-cell summaries
  scratch/data/smc_smap_samples/samples_*.npz    -- (D, M, H) particles
  scratch/data/smc_smap_samples/distrib/*.csv    -- distributional CSVs
  scratch/data/smc_smap_samples/distrib/ck_*.csv -- Ontario CK probe

Outputs are PDF, sized for a 6.5-in text column (the article geometry
of writeup/tex/draft.tex):

  fig_per_horizon_mae.pdf      -- Figure 1, \\S 4.4 -- delta MAE vs cell B
  fig_diurnal_envelope.pdf     -- Figure 2, \\S 4.4 -- per-hour demand by day-type
  fig_multisigma_coverage.pdf  -- Figure 3, \\S 4.5
  fig_pit_panel.pdf            -- Figure 4, \\S 4.5
  fig_ck_divergence.pdf        -- Figure 5, \\S 5

Styling is seaborn (`whitegrid`, `paper` context, serif font) for
uniformity. The CK divergence figure additionally re-runs the
synthetic VAR(1) gate to produce the baseline row, since that gate's
per-h numbers are not committed (the gate prints; the script reads
stdout).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---- uniform style ---------------------------------------------------------
sns.set_theme(context="paper", style="whitegrid", font="serif", font_scale=0.95)
plt.rcParams.update({
    "figure.dpi":          150,
    "savefig.dpi":         300,
    "savefig.bbox":        "tight",
    "savefig.pad_inches":  0.02,
    "pdf.fonttype":        42,                              # editable text
    "axes.linewidth":      0.7,
    "grid.linewidth":      0.4,
    "lines.linewidth":     1.2,
    "lines.markersize":    3.5,
})

ROOT = Path(__file__).resolve().parent.parent
SAMP = ROOT / "scratch" / "data" / "smc_smap_samples"
DIST = SAMP / "distrib"
FIGS = ROOT / "writeup" / "tex" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)


# canonical cell ordering and palette
CELLS = [
    ("A", "meaniter_prod",     "A. mean-iter, prod-$\\theta$"),
    ("B", "meaniter_global",   "B. mean-iter, $\\theta{=}0$"),
    ("C", "smc_global_gauss",  "C. SMC, $\\theta{=}0$, Gaussian"),
    ("D", "smc_global_emp",    "D. SMC, $\\theta{=}0$, empirical"),
    ("E", "smc_prod_gauss",    "E. SMC, prod-$\\theta$, Gaussian"),
    ("F", "smc_prod_emp",      "F. SMC, prod-$\\theta$, empirical"),
]
PALETTE_CELL = {
    "A": "#404040",   "B": "#808080",
    "C": "#2c7fb8",   "D": "#d95f0e",
    "E": "#41b6c4",   "F": "#fec44f",
}
LINESTYLE_CELL = {"A": "-", "B": "--", "C": "-", "D": "-", "E": "--", "F": "--"}
PALETTE_DT = {"weekday": "#2c7fb8", "saturday": "#d95f0e", "sunday": "#7b3294"}


# ---------------------------------------------------------------------------
# Figure 1: per-horizon delta-MAE vs cell B
# ---------------------------------------------------------------------------
def fig_per_horizon_mae() -> Path:
    """Delta-MAE vs cell B per horizon -- exposes the ~few-MW cell
    separation that gets washed out by the ~1100-MW peak when plotted
    in absolute MAE. The diurnal envelope is shown separately in
    fig_diurnal_envelope.pdf."""
    rows = []
    for code, tag, label in CELLS:
        df = pd.read_pickle(SAMP / f"cell_{code}_{tag}.pkl")
        per_h = (df.assign(abs_err=(df["our_forecast_mw"] - df["actual_mw"]).abs())
                   .groupby("horizon_h")["abs_err"].mean())
        for h, v in per_h.items():
            rows.append({"cell": code, "label": label, "horizon": h, "mae": v})
    long = pd.DataFrame(rows)
    base = long[long["cell"] == "B"].set_index("horizon")["mae"]
    long = long[long["cell"] != "B"].copy()
    long["delta_mae"] = long["mae"].values - base.reindex(long["horizon"]).values

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    sns.lineplot(
        data=long, x="horizon", y="delta_mae", hue="label", style="label",
        palette={lbl: PALETTE_CELL[c] for c, _t, lbl in CELLS if c != "B"},
        dashes={lbl: (1, 0) if LINESTYLE_CELL[c] == "-" else (4, 2)
                for c, _t, lbl in CELLS if c != "B"},
        markers=True, ax=ax,
    )
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xlabel("horizon $h$ (hours)")
    ax.set_ylabel("$\\Delta$MAE vs cell B  (MW)")
    ax.set_xlim(0.5, 24.5)
    ax.set_xticks([1, 4, 8, 12, 16, 20, 24])
    ax.legend(title=None, loc="upper left", ncol=2, frameon=False, fontsize=7.5)
    out = FIGS / "fig_per_horizon_mae.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 2: diurnal demand distribution by day-type (3-panel boxplot)
# ---------------------------------------------------------------------------
def fig_diurnal_envelope() -> Path:
    # Use any SMC cell -- actuals and daytype are identical across cells;
    # the mean-iter pickles (A, B) lack the daytype column. C is canonical.
    df = pd.read_pickle(SAMP / "cell_C_smc_global_gauss.pkl")
    # HE1..HE24 align to hour-of-day 0..23 under cfg.day_anchor_hours=0
    df = df.assign(hour=df["horizon_h"] - 1)
    daytypes = ("weekday", "saturday", "sunday")
    counts = {dt: df[df["daytype"] == dt]["delivery_date"].nunique()
              for dt in daytypes}

    fig, axes = plt.subplots(3, 1, figsize=(6.5, 5.4), sharex=True, sharey=True)
    for ax, dt in zip(axes, daytypes):
        sub = df[df["daytype"] == dt]
        sns.boxplot(
            data=sub, x="hour", y="actual_mw",
            color=PALETTE_DT[dt], width=0.66,
            fliersize=0, linewidth=0.5, ax=ax,
        )
        for patch in ax.patches:
            patch.set_alpha(0.65)
            patch.set_edgecolor("black")
        ax.set_xlabel("")
    axes[-1].set_xlabel("hour of delivery day (HE)")
    axes[-1].set_xticks([0, 4, 8, 12, 16, 20, 23])
    axes[-1].set_xticklabels([1, 5, 9, 13, 17, 21, 24])
    # Remove per-axes y-labels and use a single figure-level y-label.
    for ax in axes:
        ax.set_ylabel("")
    # Move the per-row daytype/count label to the right side so the left
    # side is free for the shared "Ontario demand (MW)" label.
    for ax, dt in zip(axes, daytypes):
        ax.text(1.02, 0.5, f"{dt}\n($n = {counts[dt]}$)",
                transform=ax.transAxes, rotation=0,
                va="center", ha="left", fontsize=8.5)
    fig.text(0.02, 0.5, "Ontario demand (MW)",
             rotation="vertical", va="center", fontsize=9)
    fig.subplots_adjust(left=0.10, right=0.86, hspace=0.12)
    out = FIGS / "fig_diurnal_envelope.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 3: multi-sigma coverage curves per cell (4 panels)
# ---------------------------------------------------------------------------
def fig_multisigma_coverage() -> Path:
    cov = pd.read_csv(DIST / "coverage_multisigma.csv")
    smc_codes = ["C", "D", "E", "F"]
    cov = cov[(cov["cell"].isin(smc_codes)) & (cov["daytype"] == "all")].copy()
    # pool over horizons
    pooled = (cov.groupby(["cell", "level"], as_index=False)["cov_emp"].mean())
    label_for = {c: lbl for c, _t, lbl in CELLS}
    pooled["label"] = pooled["cell"].map(label_for)

    palette_by_label = {label_for[c]: PALETTE_CELL[c] for c in smc_codes}
    g = sns.FacetGrid(pooled, col="label", col_wrap=2,
                      height=2.1, aspect=1.55,
                      sharex=True, sharey=True)
    def _line(data, **kwargs):
        lbl = data["label"].iloc[0]
        plt.plot(data["level"], data["cov_emp"],
                 color=palette_by_label[lbl], marker="o")
    g.map_dataframe(_line)
    for ax in g.axes.flatten():
        ax.plot([0, 1], [0, 1], color="black", linewidth=0.5, linestyle=":")
        ax.set_xlim(0.45, 1.01)
        ax.set_ylim(0.45, 1.01)
        ax.set_xticks([0.5, 0.683, 0.8, 0.9, 0.95, 0.99])
        ax.set_xticklabels(["50", "68", "80", "90", "95", "99"], fontsize=7.5)
        ax.set_yticks([0.5, 0.683, 0.8, 0.9, 0.95, 0.99])
        ax.set_yticklabels(["50", "68", "80", "90", "95", "99"], fontsize=7.5)
    g.set_axis_labels("nominal coverage (%)", "empirical coverage (%)")
    g.set_titles("{col_name}", size=8.5)
    g.figure.set_size_inches(6.5, 4.2)
    out = FIGS / "fig_multisigma_coverage.pdf"
    g.figure.savefig(out)
    plt.close(g.figure)
    return out


# ---------------------------------------------------------------------------
# Figure 4: PIT 2x2 panel (pooled-all per cell)
# ---------------------------------------------------------------------------
def _pit_for_cell(code: str, tag: str, rng: np.random.Generator) -> np.ndarray:
    npz = np.load(SAMP / f"samples_{code}_{tag}.npz", allow_pickle=False)
    sm = npz["samples"]
    dates = pd.to_datetime(npz["delivery_dates"]).normalize()
    pkl = pd.read_pickle(SAMP / f"cell_{code}_{tag}.pkl")
    pkl["delivery_date"] = pkl["delivery_date"].dt.normalize()
    a = (pkl.pivot_table(index="delivery_date", columns="horizon_h",
                         values="actual_mw", aggfunc="first")
            .sort_index()
            .reindex(dates).to_numpy().astype(np.float64))
    D, M, _H = sm.shape
    less = (sm < a[:, None, :]).sum(axis=1)
    eq   = (sm == a[:, None, :]).sum(axis=1)
    u = rng.uniform(size=less.shape)
    return ((less + u * (eq + 1.0)) / (M + 1.0)).ravel()


def fig_pit_panel() -> Path:
    rng = np.random.default_rng(0xC0FFEE)
    smc = [(c, t, l) for c, t, l in CELLS if c in ("C", "D", "E", "F")]
    rows = []
    for code, tag, label in smc:
        pit = _pit_for_cell(code, tag, rng)
        rows.append(pd.DataFrame({"cell": code, "label": label, "pit": pit}))
    df = pd.concat(rows, ignore_index=True)

    # Do NOT pass hue=label when col=label — it splits each panel's
    # data by the column variable, dividing each density by the panel
    # count and attenuating the visible bars 4x. Use FacetGrid + map_dataframe
    # with a custom colour-by-panel pattern instead.
    g = sns.FacetGrid(df, col="label", col_wrap=2,
                      height=2.1, aspect=1.55,
                      sharex=True, sharey=True)
    palette_by_label = {lbl: PALETTE_CELL[c] for c, _t, lbl in smc}
    def _hist(data, **kwargs):
        lbl = data["label"].iloc[0]
        sns.histplot(data=data, x="pit", bins=20, stat="density",
                     color=palette_by_label[lbl],
                     edgecolor="black", linewidth=0.3)
    g.map_dataframe(_hist)
    for ax in g.axes.flatten():
        ax.axhline(1.0, color="black", linewidth=0.7, linestyle=":")
        ax.set_xlim(0, 1)
    g.set_axis_labels("PIT", "density")
    g.set_titles("{col_name}", size=8.5)
    g.figure.set_size_inches(6.5, 4.2)
    out = FIGS / "fig_pit_panel.pdf"
    g.figure.savefig(out)
    plt.close(g.figure)
    return out


# ---------------------------------------------------------------------------
# Figure 5: CK consistency divergence
# ---------------------------------------------------------------------------
def _synthetic_ck_curve() -> pd.DataFrame:
    """Re-run the synthetic VAR(1) gate to get per-h errors.

    Same generative process as processing.innovations.validation.ck_consistency
    (n = 4000, seed 12345). Reports absolute Frobenius drift error
    (the gate's pass criterion is an absolute tolerance) and relative
    diffusion error (||Q_iter^h|| does not shrink on a stable VAR(1)).
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
    # long-form for seaborn
    long = pd.concat([
        syn[["horizon", "drift_abs", "diff_rel"]]
           .rename(columns={"drift_abs": "drift", "diff_rel": "diff"})
           .assign(system="synthetic VAR(1)"),
        *[df[["horizon", "drift_err", "diff_rel"]]
              .rename(columns={"drift_err": "drift", "diff_rel": "diff"})
              .assign(system=f"Ontario {dt}")
          for dt, df in ont.items()],
    ], ignore_index=True)

    palette = {
        "synthetic VAR(1)": "#404040",
        "Ontario weekday":  "#2c7fb8",
        "Ontario saturday": "#d95f0e",
        "Ontario sunday":   "#7b3294",
    }
    dashes = {
        "synthetic VAR(1)": (4, 2),
        "Ontario weekday":  (1, 0),
        "Ontario saturday": (1, 0),
        "Ontario sunday":   (1, 0),
    }

    fig, (ax_d, ax_s) = plt.subplots(1, 2, figsize=(6.5, 3.0), sharex=True)
    sns.lineplot(data=long, x="horizon", y="drift",
                 hue="system", style="system",
                 palette=palette, dashes=dashes,
                 markers=True, ax=ax_d, legend=True)
    sns.lineplot(data=long, x="horizon", y="diff",
                 hue="system", style="system",
                 palette=palette, dashes=dashes,
                 markers=True, ax=ax_s, legend=False)
    ax_d.set_yscale("log")
    ax_d.set_xlabel("horizon $h$")
    ax_d.set_ylabel("drift error $\\|\\hat C_{h\\Delta}-(\\hat C_\\Delta)^h\\|_F$")
    ax_d.set_xticks([1, 2, 4, 8, 12, 24])
    ax_d.legend(title=None, loc="lower right", frameon=False, fontsize=7.5)

    ax_s.set_yscale("log")
    ax_s.set_xlabel("horizon $h$")
    ax_s.set_ylabel("diffusion rel err $\\|\\hat\\Sigma_{h\\Delta}-\\Sigma^{\\mathrm{iter}}_h\\|_F\\,/\\,\\|\\Sigma^{\\mathrm{iter}}_h\\|_F$")
    ax_s.set_xticks([1, 2, 4, 8, 12, 24])

    out = FIGS / "fig_ck_divergence.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    for fn in (fig_per_horizon_mae,
               fig_diurnal_envelope,
               fig_multisigma_coverage,
               fig_pit_panel,
               fig_ck_divergence):
        path = fn()
        print(f"  wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
