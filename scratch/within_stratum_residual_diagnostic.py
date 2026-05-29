"""
PROVENANCE-GRADE: INSPECTION-ONLY

Within-stratum residual structure on Q1's M1 PIT residuals, stratified
by (day_type, Z_c, Z_2) — the same stratification Q1B used. Goal: characterise
the structural properties any successful exogenous Z must have to close the
residual χ² that the within-data lever space leaves on the table.

P1 (S11) localised the missing-Z signal across strata: hour_of_week binding
axis; hot zone Fri-Sun; cold zone Mon AM. P1's per-stratum α̂_L tells us
WHERE the structure lives across cells. This diagnostic answers the natural
next question: WITHIN each cell, what does the residual structure LOOK LIKE?

Interpretation:
  - Strong within-cell lag-1 autocorrelation → slow-varying factor (temperature
    has this; loads shocks do not). Suggests temperature is a plausible Z.
  - White within-cell residuals → fast-varying factor that hour_of_week already
    captures the trend of; temperature would not help.
  - Cross-day systematic shape (per-delivery-day mean far from 0.5) → between-
    day non-stationarity; signature of a slowly-evolving exogenous covariate
    (e.g., seasonal temperature ramp, secular load growth).
  - Power spectrum peaks → periodic structure not captured by P1's binning;
    e.g., a 12-hour HVAC cycle would peak at f = 1/12 hr^-1 inside a cell.

This diagnostic does NOT:
  - Identify the latent Z (identifiability obstruction: Bergna et al. 2026
    Prop 1; same structure consistent with intrinsic heavy tails OR latent-Z).
  - Provide claim-grade evidence; INSPECTION-ONLY scratch run on existing
    pickled artifacts.
  - Replace the registered T1 experiment (ECCC temperature conditioning).

It DOES give T1's phase_a a falsifiable target: if the within-cell residuals
show no lag-class autocorrelation, temperature won't help and we should
reconsider the framework-prescribed candidate-Z testing path before
acquiring ECCC data.

Output: prose summary + per-cell numerical table written to
scratch/within_stratum_residual_diagnostic.txt.

Source: scratch/data/distributional_class_q1/q1.pkl (pit_M1 DataFrame).
        Same 12,000-row post-cutoff PIT residual series P1 ran on.
Binning: P1's production helpers (hour_of_week_bin, time_of_day_label).
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import NamedTuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Production binning helpers (re-imported here rather than copied — keeps
# us aligned with P1 and Q1A's source-of-truth definitions).
from experiment.p1_patra_sen.__main__ import (
    _hour_of_week_bin,
    _time_of_day_label,
)

Q1_PKL = ROOT / "scratch" / "data" / "distributional_class_q1" / "q1.pkl"
OUT_TXT = ROOT / "scratch" / "within_stratum_residual_diagnostic.txt"
OUT_FIG = ROOT / "scratch" / "within_stratum_residual_diagnostic.pdf"


_TIME_OF_DAY_LABEL_TO_Z2 = {
    "overnight": 0,
    "morning_ramp": 1,
    "afternoon": 2,
    "evening": 3,
}


def _annotate_with_zc_z2(pit: pd.DataFrame) -> pd.DataFrame:
    """Add z_c and z_2 columns per Q1B's stratification convention."""
    df = pit.copy()
    df["weekday"] = df["delivery_day"].dt.weekday  # 0=Mon..6=Sun
    df["z_c"] = df.apply(
        lambda r: _hour_of_week_bin(int(r["weekday"]), int(r["h"])), axis=1
    )
    df["z_2_label"] = df["h"].apply(_time_of_day_label)
    df["z_2"] = df["z_2_label"].map(_TIME_OF_DAY_LABEL_TO_Z2).astype(int)
    return df


class CellStats(NamedTuple):
    """Within-cell residual diagnostics, per (day_type, z_c, z_2)."""
    day_type: str
    z_c: int
    z_2: int
    z_2_label: str
    n: int
    # Lag-k autocorrelations of u_PIT - 0.5 within-cell ordering by
    # (delivery_day, h). Lag-1 is "next observation in this cell."
    # For e.g. saturday__z_c6__morning_ramp, consecutive cell rows are
    # within the same morning (HE7→HE8), so lag-1 carries continuous-time
    # within-day continuity.
    acf_lag1: float
    acf_lag5: float | None   # for cells with within-day length ~5 (z_2 windows of 5-6 hours)
    acf_lag24: float | None  # whole-day; only defined for cells spanning multiple days at consistent same-hour positions
    # Variance shape: empirical variance of u_PIT vs uniform[0,1]
    # expected variance 1/12 ≈ 0.0833. Ratio > 1 = over-dispersed; < 1 =
    # under-dispersed; combined with Q2B's finding of under-dispersion at
    # the cell level, expected ratio < 1 broadly.
    var_u_pit: float
    var_ratio_vs_uniform: float
    # Cross-day systematic shape: per-delivery-day mean of u_PIT - 0.5
    # within-cell. The standard deviation of those per-day means is the
    # signature of between-day non-stationarity. If it's small, residuals
    # average out across days; if large, specific days are
    # systematically over/under the predictive.
    n_days_present: int
    sd_per_day_mean: float
    # Power-spectrum peak (in cycles per cell-row): the frequency at which
    # the in-cell residual series has its largest periodogram amplitude.
    # Defined only for n >= 200 (advisor caveat: spectrum on small n is
    # noisy). None for under-threshold cells.
    spectrum_peak_freq: float | None
    spectrum_peak_power: float | None


def _safe_acf(x: np.ndarray, lag: int) -> float:
    """Sample autocorrelation at given lag. Returns nan if too few points."""
    if len(x) <= lag + 1:
        return float("nan")
    x = x - x.mean()
    num = (x[:-lag] * x[lag:]).sum()
    den = (x * x).sum()
    if den == 0:
        return float("nan")
    return float(num / den)


def _per_cell_stats(group: pd.DataFrame) -> CellStats:
    """Compute within-cell diagnostics. Group is one (day_type, z_c, z_2)."""
    g = group.sort_values(["delivery_day", "h"]).copy()
    u = g["u_PIT"].to_numpy()
    u_centred = u - 0.5
    n = len(u)
    dt = g["day_type"].iloc[0]
    zc = int(g["z_c"].iloc[0])
    z2 = int(g["z_2"].iloc[0])
    z2_label = str(g["z_2_label"].iloc[0])
    # Per-day mean (cross-day systematic shape).
    per_day_means = g.groupby("delivery_day")["u_PIT"].mean() - 0.5
    n_days = len(per_day_means)
    sd_per_day = float(per_day_means.std()) if n_days > 1 else float("nan")
    # Lag autocorrelations.
    acf_1 = _safe_acf(u_centred, 1)
    acf_5 = _safe_acf(u_centred, 5) if n > 6 else None
    # Lag-24 within-cell: only defined if cell has stable
    # day-to-day same-hour structure. For most cells it's between-day,
    # not within-day.
    acf_24 = _safe_acf(u_centred, 24) if n > 25 else None
    # Spectrum: periodogram of u_centred. Peak frequency in
    # cycles-per-cell-row.
    if n >= 200:
        freqs, power = scipy_signal.periodogram(u_centred, fs=1.0)
        # Skip the DC component at freq=0.
        peak_idx = np.argmax(power[1:]) + 1
        peak_freq = float(freqs[peak_idx])
        peak_power = float(power[peak_idx])
    else:
        peak_freq = None
        peak_power = None
    var_u = float(u.var(ddof=1))
    var_ratio = var_u / (1.0 / 12.0)  # uniform expected variance
    return CellStats(
        day_type=dt, z_c=zc, z_2=z2, z_2_label=z2_label, n=n,
        acf_lag1=acf_1, acf_lag5=acf_5, acf_lag24=acf_24,
        var_u_pit=var_u, var_ratio_vs_uniform=var_ratio,
        n_days_present=n_days, sd_per_day_mean=sd_per_day,
        spectrum_peak_freq=peak_freq, spectrum_peak_power=peak_power,
    )


def _render_table(stats: list[CellStats]) -> str:
    """Tidy table of per-cell diagnostics, sorted by chi^2-relevant ordering
    (large cells with substantial residual mass first; matches Q1B per-cell
    decomposition for cross-reference)."""
    rows = []
    for s in stats:
        rows.append({
            "day_type": s.day_type,
            "z_c": s.z_c,
            "z_2": s.z_2,
            "z_2_label": s.z_2_label,
            "n": s.n,
            "n_days": s.n_days_present,
            "acf_lag1": s.acf_lag1,
            "acf_lag5": s.acf_lag5 if s.acf_lag5 is not None else float("nan"),
            "var_ratio": s.var_ratio_vs_uniform,
            "sd_per_day_mean": s.sd_per_day_mean,
            "spec_peak_freq": s.spectrum_peak_freq if s.spectrum_peak_freq else float("nan"),
        })
    df = pd.DataFrame(rows)
    # Match Q1B per-cell-chi^2 ordering: day_type then z_c then z_2.
    df = df.sort_values(["day_type", "z_c", "z_2"]).reset_index(drop=True)
    df = df.drop(columns=["z_2"])  # keep label for output, drop numeric sort key
    return df.to_string(
        index=False,
        formatters={
            "acf_lag1":        lambda v: f"{v:+.3f}",
            "acf_lag5":        lambda v: f"{v:+.3f}" if not np.isnan(v) else "  n/a",
            "var_ratio":       lambda v: f"{v:.2f}",
            "sd_per_day_mean": lambda v: f"{v:.3f}",
            "spec_peak_freq":  lambda v: f"{v:.4f}" if not np.isnan(v) else "  n/a ",
        }
    )


def _plot_diagnostic(stats: list[CellStats]) -> None:
    """Quick 2-panel diagnostic plot. (Inspection-only; not for writeup.)"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    # Panel 1: lag-1 ACF vs cell size, coloured by day-type.
    dt_colour = {"weekday": "#2c7fb8", "saturday": "#d95f0e", "sunday": "#7b3294"}
    for s in stats:
        axes[0].scatter(s.n, s.acf_lag1, s=40, color=dt_colour[s.day_type],
                        edgecolor="black", linewidth=0.5, alpha=0.85,
                        label=s.day_type if (s.z_c, s.z_2) == (0, 0) else None)
        # Annotate the headline cell.
        if s.day_type == "saturday" and s.z_c == 6 and s.z_2_label == "morning_ramp":
            axes[0].annotate("sat,zc6,morn", (s.n, s.acf_lag1),
                            xytext=(8, 6), textcoords="offset points",
                            fontsize=8, color="#1a1a1a",
                            arrowprops=dict(arrowstyle="-", lw=0.7, color="#1a1a1a"))
    axes[0].axhline(0, color="black", lw=0.5)
    # Approximate Bartlett 95% null bound for white noise: 1.96/sqrt(n)
    n_range = np.linspace(50, 2500, 200)
    axes[0].plot(n_range,  1.96/np.sqrt(n_range), color="#999999", lw=0.7, ls="--")
    axes[0].plot(n_range, -1.96/np.sqrt(n_range), color="#999999", lw=0.7, ls="--",
                 label="±1.96/√n  (white-noise null 95%)")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("n (cell row count)", fontsize=9)
    axes[0].set_ylabel("lag-1 ACF of $u_{PIT} - 0.5$ within cell", fontsize=9)
    axes[0].set_title("Within-cell lag-1 autocorrelation: slow-factor signature",
                      fontsize=10)
    axes[0].legend(fontsize=8, loc="lower right")
    # Panel 2: per-day-mean SD vs cell size, coloured by day-type.
    for s in stats:
        axes[1].scatter(s.n_days_present, s.sd_per_day_mean, s=40,
                        color=dt_colour[s.day_type],
                        edgecolor="black", linewidth=0.5, alpha=0.85)
        if s.day_type == "saturday" and s.z_c == 6 and s.z_2_label == "morning_ramp":
            axes[1].annotate("sat,zc6,morn",
                            (s.n_days_present, s.sd_per_day_mean),
                            xytext=(8, 6), textcoords="offset points",
                            fontsize=8, color="#1a1a1a",
                            arrowprops=dict(arrowstyle="-", lw=0.7, color="#1a1a1a"))
    axes[1].set_xlabel("n delivery days with cell present", fontsize=9)
    axes[1].set_ylabel("SD of per-day mean $u_{PIT} - 0.5$ within cell", fontsize=9)
    axes[1].set_title("Cross-day systematic shape: between-day non-stationarity",
                      fontsize=10)
    for ax in axes:
        ax.grid(True, alpha=0.3)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("Within-stratum residual structure (INSPECTION-ONLY)",
                 fontsize=11, y=0.99)
    fig.subplots_adjust(top=0.88, bottom=0.13, left=0.08, right=0.97, wspace=0.28)
    fig.savefig(OUT_FIG, dpi=150)
    plt.close(fig)


def main() -> None:
    print("# PROVENANCE-GRADE: INSPECTION-ONLY")
    print("# Within-stratum residual structure on Q1's M1 PIT residuals.")
    print()
    print(f"Loading {Q1_PKL.relative_to(ROOT)} ...")
    with Q1_PKL.open("rb") as f:
        q1 = pickle.load(f)
    pit = q1["pit_M1"]
    print(f"  pit_M1 rows: {len(pit)} ({pit['delivery_day'].min().date()} → {pit['delivery_day'].max().date()})")
    pit = _annotate_with_zc_z2(pit)
    print(f"  annotated with (z_c, z_2) per P1's production binning.")
    print()
    # Per-cell.
    cells = []
    for (dt, zc, z2), g in pit.groupby(["day_type", "z_c", "z_2"]):
        cells.append(_per_cell_stats(g))
    print(f"# cells: {len(cells)}")
    print()
    # Table.
    print("## Per-cell diagnostics")
    print(_render_table(cells))
    print()
    # Headline summary.
    print("## Aggregate signatures")
    n_total = sum(c.n for c in cells)
    n_cells_with_significant_acf = sum(
        1 for c in cells
        if c.n >= 50 and abs(c.acf_lag1) > 1.96 / np.sqrt(c.n)
    )
    print(f"  total rows: {n_total}")
    print(f"  cells with lag-1 ACF clearing Bartlett 95% null bound: "
          f"{n_cells_with_significant_acf} / {len(cells)}")
    mean_acf = np.mean([c.acf_lag1 for c in cells])
    median_acf = np.median([c.acf_lag1 for c in cells])
    print(f"  mean lag-1 ACF across cells: {mean_acf:+.3f}")
    print(f"  median lag-1 ACF across cells: {median_acf:+.3f}")
    mean_var_ratio = np.mean([c.var_ratio_vs_uniform for c in cells])
    print(f"  mean variance ratio vs uniform[0,1]: {mean_var_ratio:.2f}  "
          f"(<1 = under-dispersed)")
    mean_sd_per_day = np.mean([c.sd_per_day_mean for c in cells
                                if not np.isnan(c.sd_per_day_mean)])
    print(f"  mean SD of per-day-means within-cell: {mean_sd_per_day:.3f}")
    print()
    # Plot.
    print(f"writing diagnostic plot: {OUT_FIG.relative_to(ROOT)}")
    _plot_diagnostic(cells)
    print("done.")


if __name__ == "__main__":
    main()
