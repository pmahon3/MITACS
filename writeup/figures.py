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


# Six predictors of the §4.4 grid.  Internal letter codes (A..F) are kept
# only as dict keys and file-name disambiguators; the labels shown to
# the reader in figure legends are the numeric forms "1"..."6", and the
# figure caption supplies the mapping in prose.  Predictor 1 (mean-iter,
# θ(x)) ↔ A, ..., Predictor 6 (SMC, θ(x), empirical) ↔ F.
PREDICTORS = [
    ("A", "meaniter_prod",     "1"),
    ("B", "meaniter_global",   "2"),
    ("C", "smc_global_gauss",  "3"),
    ("D", "smc_global_emp",    "4"),
    ("E", "smc_prod_gauss",    "5"),
    ("F", "smc_prod_emp",      "6"),
]
# Back-compat alias for any external callers that still import CELLS.
CELLS = PREDICTORS
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
    ax.set_ylabel("$\\Delta$MAE vs predictor 2  (MW)")
    ax.set_xlim(0.5, 24.5)
    ax.set_xticks([1, 4, 8, 12, 16, 20, 24])
    # Legend outside the plot on the right so it never overlaps the curves.
    ax.legend(title="predictor", loc="center left",
              bbox_to_anchor=(1.01, 0.5),
              frameon=False, fontsize=8, title_fontsize=8)
    fig.subplots_adjust(right=0.84)
    out = FIGS / "fig_per_horizon_mae.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 2: diurnal demand distribution by day-type (3-panel boxplot)
# ---------------------------------------------------------------------------
def fig_diurnal_envelope() -> Path:
    # Use any SMC predictor pickle -- actuals + daytype are identical across
    # predictors; the mean-iter pickles (1, 2) lack the daytype column.
    df = pd.read_pickle(SAMP / "cell_C_smc_global_gauss.pkl")
    # HE1..HE24 align to hour-of-day 0..23 under cfg.day_anchor_hours=0
    df = df.assign(hour=df["horizon_h"] - 1)
    daytypes = ("weekday", "saturday", "sunday")
    counts = {dt: df[df["daytype"] == dt]["delivery_date"].nunique()
              for dt in daytypes}

    fig, axes = plt.subplots(3, 1, figsize=(6.5, 5.4), sharex=True, sharey=True)
    # Single neutral colour across all three rows -- the panels are
    # separated by day-type already; colour adds nothing.
    neutral = "#6c8eb8"
    for ax, dt in zip(axes, daytypes):
        sub = df[df["daytype"] == dt]
        sns.boxplot(
            data=sub, x="hour", y="actual_mw",
            color=neutral, width=0.66,
            fliersize=0, linewidth=0.5, ax=ax,
        )
        for patch in ax.patches:
            patch.set_alpha(0.7)
            patch.set_edgecolor("black")
        ax.set_xlabel("")
    axes[-1].set_xlabel("hour of delivery day (HE)")
    axes[-1].set_xticks([0, 4, 8, 12, 16, 20, 23])
    axes[-1].set_xticklabels([1, 5, 9, 13, 17, 21, 24])
    # Strip per-axes y-labels; use a single figure-level shared label.
    for ax in axes:
        ax.set_ylabel("")
    # Day-type/count chip on the right of each panel.
    for ax, dt in zip(axes, daytypes):
        ax.text(1.015, 0.5, f"{dt}\n($n = {counts[dt]}$)",
                transform=ax.transAxes, rotation=0,
                va="center", ha="left", fontsize=8.5)
    fig.text(0.015, 0.5, "Ontario demand (MW)",
             rotation="vertical", va="center", fontsize=9)
    # Generous left margin so the shared y-label doesn't collide with
    # tick labels, and a right margin reserved for the day-type chip.
    fig.subplots_adjust(left=0.11, right=0.84, hspace=0.12)
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
    g.set_titles("predictor {col_name}", size=8.5)
    g.figure.set_size_inches(6.5, 4.2)
    g.figure.tight_layout()
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
    g.set_titles("predictor {col_name}", size=8.5)
    g.figure.set_size_inches(6.5, 4.2)
    g.figure.tight_layout()
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
    fig.tight_layout()

    out = FIGS / "fig_ck_divergence.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 6: Act-2 directed-search forest plot
# ---------------------------------------------------------------------------
#
# Reads the headline chi^2 + 95% paired-day-bootstrap CIs per pre-registered
# experiment from each experiment's result.yaml, draws a horizontal forest
# plot on a common chi^2 axis, with the pre-registered R-A (<=228), R-B
# ((228, 2284]), R-C (>2284) cut bands as background shading. This is the
# Section 7 summary figure: a reviewer skimming the abstract + this figure
# can recover the directed-search arc.
#
# Q1's M0 Gaussian baseline (22,328 chi^2) is the starting point Act 1
# closes by 16x; M0 is intentionally OFF the forest plot (would dominate
# the x-axis and make the in-band differences invisible). The caption
# names M0 so the reader sees where the search starts.
#
# P1 is a per-stratum alpha_L diagnostic, not a chi^2 statistic; it does
# not sit on the chi^2 forest. Its role in the narrative is "the localising
# diagnostic that named hour_of_week as the binding axis"; the caption
# names this role.


# Hard-coded headline numerics, verified against result.yaml on 2026-05-29
# (this script is meant to be cheap; the bootstrap numbers are stable and
# documented in the per-experiment registry artifacts).
FOREST_EXPERIMENTS = [
    # (label,    date,         metric_label,            point,    ci_low,    ci_high)
    ("Q1",     "2026-05-26", "M1 Student-$t$ kernel", 1371.6,   1069.1,    1817.2),
    ("Q2A",    "2026-05-26", "effective $\\chi^2$",    953.8,    685.2,    1356.9),
    ("Q2B",    "2026-05-26", "best mech. fix",         875.0,    627.6,    1213.7),
    ("Q1A",    "2026-05-27", "better $\\chi^2$",       952.5,    625.3,    1342.3),
    ("Q1B",    "2026-05-27", "better $\\chi^2$",       662.2,    468.1,     944.5),
    ("Q1B$'$", "2026-05-29", "better $\\chi^2$",       679.1,    480.9,     987.1),
]

# Pre-registered cuts, inherited from the resolution-paths-thread skeleton.
R_A_CUT = 228     # corroboration target (Q2A's R-A2 gate-derived cut)
R_C_CUT = 2284    # falsification cut (Q2A's R-C2 gate-derived cut)

# M0 Gaussian baseline (Q1's starting point) -- documented in caption, not
# plotted on the forest (would dominate the x-axis).
M0_POINT, M0_CI = 22328.6, (19028.1, 25730.8)


def fig_act2_forest() -> Path:
    """Act-2 directed-search summary forest plot. Reader skimming abstract
    + this figure recovers the arc.

    Convention: position-on-common-scale is the most-accurately-decoded
    visual encoding (Cleveland-McGill 1984); the pre-registered cut bands
    are the directed-search structure made visual."""
    fig, ax = plt.subplots(figsize=(6.5, 3.6))

    # ---- Background: pre-registered R-A / R-B / R-C bands -----------------
    # R-A (<= 228): corroboration target, the gap-closure outcome. Green.
    # R-B ((228, 2284]): partial; the band realised by every experiment.
    # R-C (> 2284): falsification, calibration regressed. Red.
    XMIN, XMAX = 100, 2500
    ax.axvspan(XMIN,    R_A_CUT, color="#2ca25f", alpha=0.10, zorder=0)
    ax.axvspan(R_A_CUT, R_C_CUT, color="#999999", alpha=0.08, zorder=0)
    ax.axvspan(R_C_CUT, XMAX,    color="#de2d26", alpha=0.10, zorder=0)
    ax.axvline(R_A_CUT, color="#2ca25f", lw=0.9, ls="--", zorder=1)
    ax.axvline(R_C_CUT, color="#de2d26", lw=0.9, ls="--", zorder=1)

    # ---- Per-experiment markers + CI bars --------------------------------
    # Y-axis: chronological order, top-down (most recent at bottom).
    # Each experiment uses a uniform marker style; only position+CI carry
    # info (per criterion 11: CIs visible and equally weighted).
    y_positions = list(range(len(FOREST_EXPERIMENTS), 0, -1))
    for ypos, (label, date, metric, pt, lo, hi) in zip(y_positions, FOREST_EXPERIMENTS):
        ax.errorbar(pt, ypos, xerr=[[pt - lo], [hi - pt]],
                    fmt="o", color="#252525", ecolor="#525252",
                    elinewidth=1.2, capsize=3, markersize=5.0, zorder=3)
        # Right-side annotation: point estimate + tight CI
        ax.text(2580, ypos, f"{pt:>6.1f}  [{lo:.0f}, {hi:.0f}]",
                va="center", ha="left", fontsize=8, family="monospace",
                color="#252525")

    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"{lab} ({d[5:]})" for lab, d, *_ in FOREST_EXPERIMENTS],
                       fontsize=9)
    ax.set_ylim(0.4, len(FOREST_EXPERIMENTS) + 0.6)

    # ---- X-axis: chi^2, with R-A / R-B / R-C band labels at top -----------
    ax.set_xscale("log")
    ax.set_xlim(XMIN, XMAX)
    ax.set_xticks([100, 228, 500, 1000, 2284])
    ax.set_xticklabels(["100", "228\n(R-A cut)", "500", "1000", "2284\n(R-C cut)"],
                       fontsize=8)
    ax.set_xlabel(r"marginal-PIT $\chi^2$  (post-cutoff; 95% paired-day bootstrap CI)",
                  fontsize=9)

    # Band labels (centred horizontally in each band, above the data).
    # Vertical position 0.8 above topmost row gives ~half-row clearance
    # for the right-side annotation header at 0.4.
    ax.text(150,  len(FOREST_EXPERIMENTS) + 0.8, "R-A\n(corrob.)",
            ha="center", va="center", fontsize=7.5, color="#1a7748",
            style="italic")
    ax.text(720,  len(FOREST_EXPERIMENTS) + 0.8, "R-B  (partial / inconclusive)",
            ha="center", va="center", fontsize=7.5, color="#525252",
            style="italic")
    ax.text(2400, len(FOREST_EXPERIMENTS) + 0.8, "R-C\n(falsif.)",
            ha="center", va="center", fontsize=7.5, color="#a30015",
            style="italic")

    ax.grid(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # Extend ylim slightly to fit the band labels on the top row
    ax.set_ylim(0.4, len(FOREST_EXPERIMENTS) + 1.1)
    # Numeric column header
    ax.text(2580, len(FOREST_EXPERIMENTS) + 0.8, "point  [95% CI]",
            va="center", ha="left", fontsize=8, family="monospace",
            color="#252525", weight="bold")

    fig.subplots_adjust(left=0.13, right=0.78, top=0.92, bottom=0.18)
    out = FIGS / "fig_act2_forest.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 7: Q1B per-cell chi^2 heatmap (Z2_c stratification)
# ---------------------------------------------------------------------------
#
# Reads Q1B's per_cell_chi2 dict from scratch/data/q1b_z2_conditioning/q1b.pkl.
# Plots three small-multiples (one per day-type) of chi^2 contribution per
# (Z_c, Z_2) cell under the M3 winning configuration. Highlights the
# saturday__zc6__morning_ramp cell (chi^2 = 967.3 = the single cell that
# contributes more than the pooled marginal 952.5) as the structural
# corroboration of P1's binding-axis localisation: P1 named hour_of_week
# bin 6 (Sat 07 - Sun 03) at alpha_L = 0.61; Q1A's per-cell decomposition
# showed that bin contributed chi^2 = 1634 under (day_type, Z_c)
# stratification alone; Q1B's joint (day_type, Z_c, Z_2) stratification
# splits that 1634 across Z_2 as morning_ramp 967 > evening 400 > afternoon
# 328, revealing structural within-stratum heterogeneity.
#
# This is the per-cell-decomposition story from S13 made visual.

Q1B_PKL = ROOT / "scratch" / "data" / "q1b_z2_conditioning" / "q1b.pkl"

# Display labels for the 4-level Z2 (P1's _time_of_day_label binning).
Z2_LABELS = ["overnight", "morning ramp", "afternoon", "evening"]
# Hour-of-week binning: 8 bins of 21 hours, anchored Mon 00. Labels
# trimmed to fit the heatmap row labels.
ZC_LABELS = [
    "0 (Mon 00-20)",
    "1 (Mon 21-Tu 17)",
    "2 (Tu 18-We 14)",
    "3 (We 15-Th 11)",
    "4 (Th 12-Fr 09)",
    "5 (Fr 10-Sa 06)",
    "6 (Sa 07-Su 03)",
    "7 (Su 04-Su 23)",
]


def fig_q1b_per_cell_heatmap() -> Path:
    """Per-cell chi^2 contribution under joint (day_type, Z_c, Z_2)
    stratification. The saturday__zc6__morning_ramp cell at chi^2=967.3
    is the structural corroboration of P1's binding-axis finding."""
    import pickle
    with Q1B_PKL.open("rb") as f:
        d = pickle.load(f)
    pc = d["per_cell_chi2"]["Z2_c"]  # dict[(day_type, z_c, z_2)] -> {chi2, n_rows}

    # Build a 3-panel (day_type) x 8 (z_c) x 4 (z_2) layout.
    # Empty (structurally-impossible) cells are masked.
    daytypes = ("weekday", "saturday", "sunday")
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 4.0),
                             gridspec_kw={"width_ratios": [1, 1, 1]})

    # Common color scale across panels for fair cross-day-type comparison.
    # vmax = 1000 puts the saturday hot cell (967) near the top of the
    # scale; cells above 1000 (if any) would saturate, which is fine ---
    # the visual point is "this cell is much bigger than the others",
    # not "exact value of the hot cell."
    vmax = 1000.0
    cmap = plt.get_cmap("YlOrRd")

    for ax, dt in zip(axes, daytypes):
        # Build the 8x4 matrix; np.nan for structurally-empty cells.
        M = np.full((8, 4), np.nan)
        for (d_t, zc, z2), v in pc.items():
            if d_t == dt:
                M[zc, z2] = v["chi2"]

        # Imshow with masking. cmap.bad sets the colour for nan.
        cmap_w_bad = cmap.copy()
        cmap_w_bad.set_bad("#ededed")  # light grey for structural NaN
        im = ax.imshow(M, cmap=cmap_w_bad, vmin=0, vmax=vmax,
                       aspect="auto", origin="upper")

        # Annotate each non-empty cell with its chi^2.
        for i in range(8):
            for j in range(4):
                if not np.isnan(M[i, j]):
                    val = M[i, j]
                    # White text on dark cells, black on light.
                    txt_color = "white" if val > 0.55 * vmax else "#252525"
                    ax.text(j, i, f"{val:.0f}",
                            ha="center", va="center",
                            fontsize=7.5, color=txt_color)

        ax.set_title(dt, fontsize=10, pad=4)
        ax.set_xticks(range(4))
        ax.set_xticklabels(Z2_LABELS, rotation=35, ha="right", fontsize=7.5)
        if ax is axes[0]:
            ax.set_yticks(range(8))
            ax.set_yticklabels(ZC_LABELS, fontsize=7.0)
            ax.set_ylabel("$Z_c$  (hour-of-week bin)", fontsize=9)
        else:
            ax.set_yticks(range(8))
            ax.set_yticklabels([])
        ax.set_xlabel("$Z_2$  (time-of-day)", fontsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    # Single shared colorbar at the right edge.
    cbar_ax = fig.add_axes([0.93, 0.18, 0.02, 0.66])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label(r"per-cell $\chi^2$", fontsize=9)
    cb.ax.tick_params(labelsize=7.5)

    # Highlight saturday__zc6__morning_ramp (the headline cell).
    # The saturday panel is axes[1]; coords (z_c=6, z_2=1).
    rect = plt.Rectangle((1 - 0.5, 6 - 0.5), 1, 1,
                         fill=False, edgecolor="#1a1a1a", linewidth=1.4,
                         zorder=5)
    axes[1].add_patch(rect)

    fig.subplots_adjust(left=0.18, right=0.91, top=0.92, bottom=0.22,
                        wspace=0.08)
    out = FIGS / "fig_q1b_per_cell_heatmap.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    for fn in (fig_per_horizon_mae,
               fig_diurnal_envelope,
               fig_multisigma_coverage,
               fig_pit_panel,
               fig_ck_divergence,
               fig_act2_forest,
               fig_q1b_per_cell_heatmap):
        path = fn()
        print(f"  wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
