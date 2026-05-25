"""EXPLORATORY: disentangle the mid-day forecast-high bias.

The PIT figure (writeup/tex/figs/fig_pit_panel.pdf) showed all four
SMC predictors with a right-decreasing PIT density (actuals
systematically in the lower half of the predictive). The §A.B
per-horizon coverage table (writeup/tex/draft_body.tex) showed
one-sigma coverage collapses to ~12-13% at h in [9, 13] -- the
demand-peak hours. This script disentangles two candidate mechanisms:

  (a) Phase mis-centring of C(x).  Embedding sees state level, not
      whether the state is on the rising or falling limb.  Library
      neighbours mix the two limbs, so the local linear fit centres
      between them and lags the morning ramp / overshoots the
      evening descent.  Signature: signed error sign-flips across
      the diurnal cycle.

  (b) Climatology under-representation.  The z-score climatology
      mu_{m,h} is fit on pre-cutoff library; if post-cutoff actuals
      drift higher at peak hours, a near-zero z forecast undershoots
      actual MW.  Signature: signed error uniformly forecast-high
      at peak hours (no sign flip).

Two diagnostics:

  (1) PIT by horizon: h in {1, 6, 12, 18, 24} for the four SMC
      predictors.  If PIT is uniform at h=1 but skewed at h=24, the
      bias is an iteration / propagation artifact, not C(x) at one
      step.  If PIT is already skewed at h=1, the bias lives in
      C(x).

  (2) Mean signed error by hour-of-day, stratified by day-type, for
      predictor 2 (mean-iteration, theta=0).  Tells us where in the
      cycle the bias lives and whether the sign flips.

Outputs (PDFs in writeup/tex/figs/ via the seaborn theme used by
writeup/figures.py):

  fig_pit_by_horizon.pdf
  fig_signed_error_by_hour.pdf

Also prints headline numbers to stdout for the discussion.

PROVENANCE-GRADE: INSPECTION-ONLY.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# uniform style (mirrors writeup/figures.py)
sns.set_theme(context="paper", style="whitegrid", font="serif", font_scale=0.95)
plt.rcParams.update({
    "figure.dpi":          150,
    "savefig.dpi":         300,
    "savefig.bbox":        "tight",
    "savefig.pad_inches":  0.02,
    "pdf.fonttype":        42,
    "axes.linewidth":      0.7,
    "grid.linewidth":      0.4,
    "lines.linewidth":     1.2,
    "lines.markersize":    3.5,
})

ROOT = Path(__file__).resolve().parent.parent
SAMP = ROOT / "scratch" / "data" / "smc_smap_samples"
FIGS = ROOT / "writeup" / "tex" / "figs"

SMC_CELLS = [
    ("C", "smc_global_gauss",  "3"),
    ("D", "smc_global_emp",    "4"),
    ("E", "smc_prod_gauss",    "5"),
    ("F", "smc_prod_emp",      "6"),
]
PALETTE = {"3": "#2c7fb8", "4": "#d95f0e", "5": "#41b6c4", "6": "#fec44f"}
PALETTE_DT = {"weekday": "#2c7fb8", "saturday": "#d95f0e", "sunday": "#7b3294"}


# ---------------------------------------------------------------------------
# PIT by horizon
# ---------------------------------------------------------------------------
def _pit_by_horizon_for_cell(code: str, tag: str,
                             rng: np.random.Generator) -> pd.DataFrame:
    """Return long-form PIT values per (day, horizon)."""
    npz = np.load(SAMP / f"samples_{code}_{tag}.npz", allow_pickle=False)
    sm = npz["samples"]                                              # (D, M, H)
    dates = pd.to_datetime(npz["delivery_dates"]).normalize()
    pkl = pd.read_pickle(SAMP / f"cell_{code}_{tag}.pkl")
    pkl["delivery_date"] = pkl["delivery_date"].dt.normalize()
    a = (pkl.pivot_table(index="delivery_date", columns="horizon_h",
                         values="actual_mw", aggfunc="first")
            .sort_index()
            .reindex(dates).to_numpy().astype(np.float64))
    D, M, H = sm.shape
    less = (sm < a[:, None, :]).sum(axis=1)                          # (D, H)
    eq   = (sm == a[:, None, :]).sum(axis=1)
    u = rng.uniform(size=less.shape)
    pit = (less + u * (eq + 1.0)) / (M + 1.0)                         # (D, H)
    return pd.DataFrame({
        "horizon": np.tile(np.arange(1, H + 1), D),
        "pit":     pit.ravel(),
    })


def fig_pit_by_horizon() -> Path:
    rng = np.random.default_rng(0xC0FFEE)
    target_h = [1, 6, 12, 18, 24]
    rows = []
    for code, tag, label in SMC_CELLS:
        long = _pit_by_horizon_for_cell(code, tag, rng)
        sub = long[long["horizon"].isin(target_h)].copy()
        sub["predictor"] = label
        rows.append(sub)
    df = pd.concat(rows, ignore_index=True)

    g = sns.FacetGrid(df, row="predictor", col="horizon",
                      row_order=["3", "4", "5", "6"], col_order=target_h,
                      height=1.4, aspect=1.4,
                      sharex=True, sharey=True, margin_titles=True)

    def _hist(data, **kwargs):
        pred = data["predictor"].iloc[0]
        sns.histplot(data=data, x="pit", bins=20, stat="density",
                     color=PALETTE[pred],
                     edgecolor="black", linewidth=0.25)

    g.map_dataframe(_hist)
    for ax in g.axes.flatten():
        ax.axhline(1.0, color="black", linewidth=0.6, linestyle=":")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 2.2)
    g.set_axis_labels("PIT", "density")
    g.set_titles(col_template="$h = {col_name}$",
                 row_template="predictor {row_name}", size=8.5)
    g.figure.set_size_inches(8.0, 5.6)
    g.figure.tight_layout()
    out = FIGS / "fig_pit_by_horizon.pdf"
    g.figure.savefig(out)
    plt.close(g.figure)

    # headlines for stdout
    print("\n=== PIT mass in [0, 0.5] (forecast-high bias signature) by horizon ===")
    print(f"{'pred':>6s} " + " ".join(f"{h:>7s}" for h in [f"h={x}" for x in target_h]))
    for code, tag, label in SMC_CELLS:
        long = _pit_by_horizon_for_cell(code, tag, rng)
        mass = []
        for h in target_h:
            sub = long[long["horizon"] == h]["pit"]
            mass.append((sub < 0.5).mean())
        print(f"{label:>6s} " + " ".join(f"  {m:5.3f}" for m in mass))
    return out


# ---------------------------------------------------------------------------
# Mean signed error by hour-of-day, stratified by day-type
# ---------------------------------------------------------------------------
def fig_signed_error_by_hour() -> Path:
    # We need daytype, but cells A/B (mean-iter) lack the column.
    # Pull daytype from cell C (any SMC cell has it) and merge onto B.
    b = pd.read_pickle(SAMP / "cell_B_meaniter_global.pkl").copy()
    c = pd.read_pickle(SAMP / "cell_C_smc_global_gauss.pkl")[
        ["target_dt", "horizon_h", "daytype"]]
    b = b.merge(c, on=["target_dt", "horizon_h"], how="left")
    b["hour"] = b["horizon_h"] - 1
    b["signed_err"] = b["our_forecast_mw"] - b["actual_mw"]  # +ve = forecast high

    # mean signed error per (hour, daytype) with bootstrap CI
    rng = np.random.default_rng(0)
    rows = []
    for dt in ("weekday", "saturday", "sunday"):
        sub_dt = b[b["daytype"] == dt]
        for h in range(24):
            sub = sub_dt[sub_dt["hour"] == h]["signed_err"].to_numpy()
            if len(sub) < 5:
                continue
            mu = float(sub.mean())
            # day-bootstrap CI (B=500)
            boot = rng.choice(sub, size=(500, len(sub)), replace=True).mean(axis=1)
            lo, hi = float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
            rows.append({"daytype": dt, "hour": h,
                         "signed_err": mu, "lo": lo, "hi": hi})
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(3, 1, figsize=(6.5, 5.4), sharex=True, sharey=True)
    for ax, dt in zip(axes, ("weekday", "saturday", "sunday")):
        sub = df[df["daytype"] == dt]
        ax.fill_between(sub["hour"], sub["lo"], sub["hi"],
                        color=PALETTE_DT[dt], alpha=0.25, linewidth=0)
        ax.plot(sub["hour"], sub["signed_err"],
                color=PALETTE_DT[dt], marker="o", linewidth=1.2)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.text(1.015, 0.5, dt, transform=ax.transAxes,
                va="center", ha="left", fontsize=9)
    axes[-1].set_xlabel("hour of delivery day (HE)")
    axes[-1].set_xticks([0, 4, 8, 12, 16, 20, 23])
    axes[-1].set_xticklabels([1, 5, 9, 13, 17, 21, 24])
    fig.text(0.015, 0.5,
             "mean signed error  $\\hat y - y$  (MW, +ve = forecast high)",
             rotation="vertical", va="center", fontsize=9)
    fig.subplots_adjust(left=0.13, right=0.86, hspace=0.12)
    out = FIGS / "fig_signed_error_by_hour.pdf"
    fig.savefig(out)
    plt.close(fig)

    # headlines for stdout
    print("\n=== Mean signed error (predictor 2) by hour-of-day (MW) ===")
    print(f"  hour  weekday   saturday   sunday")
    for h in range(24):
        cells = []
        for dt in ("weekday", "saturday", "sunday"):
            row = df[(df["daytype"] == dt) & (df["hour"] == h)]
            cells.append(f"{row['signed_err'].iloc[0]:+7.0f}" if len(row)
                         else "    n/a")
        print(f"  HE{h+1:>2d}  {cells[0]}    {cells[1]}    {cells[2]}")
    return out


# ---------------------------------------------------------------------------
def main() -> None:
    p1 = fig_pit_by_horizon()
    print(f"  wrote {p1.relative_to(ROOT)}")
    p2 = fig_signed_error_by_hour()
    print(f"  wrote {p2.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
