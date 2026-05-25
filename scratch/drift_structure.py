"""EXPLORATORY: drift structure of the pre-cutoff climatology vs
post-cutoff actuals.

Cause 1 (mid-day forecast-high bias diagnosis): the pre-cutoff
month-by-hour-of-day climatology mu_{m,h} sits below the post-cutoff
empirical mean by a positive amount on average.  This script asks
WHICH STRUCTURE the drift has, before we commit a refit family:

  - Level: drift is approximately constant across all (m, h) bins.
    Fix: an additive level correction.
  - Additive (month + hour): drift = a(m) + b(h).  Some bins drift
    more by month, some by hour, but they decompose cleanly.
    Fix: month-specific or hour-specific level correction.
  - Full m x h interaction: drift varies by month AND hour in a way
    that doesn't decompose.  Fix: refit the whole climatology on a
    more recent window.

Outputs (local; not for the draft):

  scratch/data/drift_structure/heatmap.pdf
  scratch/data/drift_structure/marginals.pdf
  printed: R^2 of nested models {constant, additive, full}

Inputs: load_actuals + zscore_params from the production frozen spec
(strict pre-cutoff).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from experiment import freeze
from experiment._actuals import load_actuals, zscore_params, mu_at, sigma_at

sns.set_theme(context="paper", style="whitegrid", font="serif", font_scale=0.95)
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "scratch" / "data" / "drift_structure"
OUT.mkdir(parents=True, exist_ok=True)


def _setup():
    spec = freeze.load_verified()
    cutoff = pd.Timestamp(spec["data_cutoff"])
    clim_method = spec["predictor"].get("climatology_method", "month_hour")
    k_year = spec["predictor"].get("fourier_k_year")
    k_day  = spec["predictor"].get("fourier_k_day")
    zp = zscore_params(cutoff, method=clim_method, k_year=k_year, k_day=k_day)
    raw_full = load_actuals(cutoff=None)
    return spec, cutoff, raw_full, zp


def drift_table() -> pd.DataFrame:
    """Build a long-form DataFrame: rows = (month, hour) bins observed
    in the post-cutoff window. Columns: mu_pre, sigma_pre, actual_mean,
    drift_mw, drift_sigma, n."""
    spec, cutoff, raw_full, zp = _setup()
    post = raw_full[raw_full.index > cutoff].dropna()
    months = post.index.month
    hours  = post.index.hour
    df = pd.DataFrame({
        "month":   months,
        "hour":    hours,
        "actual":  post.values,
        "mu_pre":  mu_at(zp, post.index),
        "sigma_pre": sigma_at(zp, post.index),
    })
    by = df.groupby(["month", "hour"])
    out = pd.DataFrame({
        "n":           by.size(),
        "actual_mean": by["actual"].mean(),
        "mu_pre":      by["mu_pre"].mean(),
        "sigma_pre":   by["sigma_pre"].mean(),
    }).reset_index()
    out["drift_mw"]    = out["mu_pre"] - out["actual_mean"]   # +ve = mu high
    out["drift_sigma"] = out["drift_mw"] / out["sigma_pre"]   # in z units
    return out


# ---------------------------------------------------------------------------
def fig_heatmap(df: pd.DataFrame) -> Path:
    pivot = df.pivot(index="month", columns="hour", values="drift_mw")
    # Use a diverging palette centred on 0
    vmax = float(np.nanmax(np.abs(pivot.values)))
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    sns.heatmap(pivot, ax=ax, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
                cbar_kws={"label": "drift  $\\mu_{m,h}^{\\mathrm{pre}} - \\bar y_{m,h}^{\\mathrm{post}}$ (MW)"},
                xticklabels=[f"{h+1}" for h in range(pivot.shape[1])],
                yticklabels=["Jan","Feb","Mar","Apr","May","Jun",
                             "Jul","Aug","Sep","Oct","Nov","Dec"][:pivot.shape[0]],
                linewidths=0.2, linecolor="white")
    ax.set_xlabel("hour of delivery day (HE)")
    ax.set_ylabel("month")
    ax.set_title("Drift of pre-cutoff climatology vs post-cutoff actuals", fontsize=9)
    out = OUT / "heatmap.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_marginals(df: pd.DataFrame) -> Path:
    by_month = df.groupby("month").apply(
        lambda g: pd.Series({
            "mean": np.average(g["drift_mw"], weights=g["n"]),
            "n":    g["n"].sum(),
        }))
    by_hour = df.groupby("hour").apply(
        lambda g: pd.Series({
            "mean": np.average(g["drift_mw"], weights=g["n"]),
            "n":    g["n"].sum(),
        }))
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0))
    axes[0].bar(by_month.index, by_month["mean"], color="#2c7fb8", alpha=0.85)
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].set_xlabel("month")
    axes[0].set_ylabel("drift (MW)")
    axes[0].set_title("by month (averaged over hours)", fontsize=9)
    axes[0].set_xticks([1,3,5,7,9,11])

    axes[1].bar(by_hour.index + 1, by_hour["mean"], color="#d95f0e", alpha=0.85)
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_xlabel("hour of delivery day (HE)")
    axes[1].set_ylabel("")
    axes[1].set_title("by hour (averaged over months)", fontsize=9)
    axes[1].set_xticks([1, 5, 9, 13, 17, 21, 24])
    fig.tight_layout()
    out = OUT / "marginals.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
def nested_decomposition(df: pd.DataFrame) -> None:
    """Compare three models for drift_mw, weighted by n:
       (i) constant c
       (ii) additive a(m) + b(h)
       (iii) full m x h (no residual; saturates)
    Report sum-of-squared-residuals weighted by n and the fraction of
    variance explained."""
    y = df["drift_mw"].to_numpy()
    w = df["n"].to_numpy().astype(float)
    months = df["month"].to_numpy()
    hours  = df["hour"].to_numpy()
    n_months = int(months.max())
    n_hours  = int(hours.max() + 1)

    def ss(resid: np.ndarray) -> float:
        return float((w * resid ** 2).sum())

    def wmean(x: np.ndarray) -> float:
        return float((w * x).sum() / w.sum())

    # Total variance (around weighted mean)
    grand = wmean(y)
    ss_total = ss(y - grand)

    # (i) constant
    fit_const = grand * np.ones_like(y)
    ss_const = ss(y - fit_const)

    # (ii) additive a(m) + b(h)  via alternating projection (5 iter is enough)
    a = np.zeros(n_months + 1)
    b = np.zeros(n_hours)
    fit_add = grand * np.ones_like(y)
    for _ in range(20):
        resid = y - b[hours]
        for m in range(1, n_months + 1):
            mask = months == m
            if mask.sum() == 0:
                continue
            a[m] = (w[mask] * resid[mask]).sum() / w[mask].sum()
        a -= np.average(a[1:], weights=[w[months == m].sum()
                                         for m in range(1, n_months + 1)])
        resid = y - a[months]
        for h in range(n_hours):
            mask = hours == h
            if mask.sum() == 0:
                continue
            b[h] = (w[mask] * resid[mask]).sum() / w[mask].sum()
        fit_add = grand + a[months] + b[hours] - grand
    ss_add = ss(y - fit_add)

    # (iii) full m x h  (mean per bin = drift_mw itself since y IS the bin mean)
    ss_full = 0.0   # saturates

    print("Nested-model decomposition of drift (weighted by n):")
    print(f"  total SS                 = {ss_total:12.2f}")
    print(f"  constant model SS_res    = {ss_const:12.2f}   "
          f"R^2 = {1 - ss_const / ss_total:.3f}")
    print(f"  additive (m + h) SS_res  = {ss_add:12.2f}   "
          f"R^2 = {1 - ss_add / ss_total:.3f}")
    print(f"  full m x h        SS_res = {ss_full:12.2f}   R^2 = 1.000  (saturated)")
    print()
    print(f"  drift mean across bins   = {grand:+8.1f} MW")
    print(f"  drift std across bins    = {np.sqrt(ss_total/w.sum()):8.1f} MW (weighted)")


# ---------------------------------------------------------------------------
def main() -> None:
    df = drift_table()
    print(f"  built drift table: {len(df)} (month, hour) bins")
    print(f"  total n          : {df['n'].sum()}")
    print()
    nested_decomposition(df)
    p1 = fig_heatmap(df)
    print(f"\n  wrote {p1.relative_to(ROOT)}")
    p2 = fig_marginals(df)
    print(f"  wrote {p2.relative_to(ROOT)}")
    # Save the table for downstream use
    out_csv = OUT / "drift_by_bin.csv"
    df.to_csv(out_csv, index=False)
    print(f"  wrote {out_csv.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
