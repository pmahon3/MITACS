"""Stage 2 — detrend raw IESO demand into a stationary z-score candidate.

Reads the per-year ``PUB_Demand_<yr>.csv`` files, removes month+hour-of-day
seasonality, and writes ``stationary_candidate.csv`` (the ``zscore`` series)
plus annual parameters and diagnostic plots.

Run standalone::

    python -m pre_processing.year_wise_standardization
"""
from __future__ import annotations

import os

import matplotlib

# Headless-safe: the orchestration runner has no display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import linregress

from config import PipelineConfig, load_config


def main(cfg: PipelineConfig) -> None:
    # ── Config ───────────────────────────────────────────────────────────
    csvs = cfg.paths.historical_csvs
    plots_dir = cfg.paths.preprocess_plots
    plots_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.stationary_candidate.parent.mkdir(parents=True, exist_ok=True)
    years = range(*cfg.data.year_range, 1)

    # ── 1. Load all raw hours ────────────────────────────────────────────
    frames = []
    for yr in years:
        f = os.path.join(csvs, f"PUB_Demand_{yr}.csv")
        df = pd.read_csv(f, skiprows=3).dropna(
            subset=["Date", "Hour", "Ontario Demand"]
        )
        # build a proper datetime index
        df["Time"] = pd.to_datetime(
            df["Date"].astype(str)
            + " "
            + (df["Hour"] - 1).astype(int).astype(str)
            + ":00:00"
        )
        df.set_index("Time", inplace=True)
        frames.append(df[["Ontario Demand"]])

    hourly = pd.concat(frames, copy=False).sort_index()

    # ── 2. Annual mean & σ ───────────────────────────────────────────────
    mu_y = hourly["Ontario Demand"].groupby(hourly.index.year).mean()
    sigma_y = hourly["Ontario Demand"].groupby(hourly.index.year).std()
    cv_y = sigma_y / mu_y

    # ── 3. Monthly mean + ±1 σ (intramonth) ──────────────────────────────
    monthly_stats = hourly["Ontario Demand"].resample("M").agg(["mean", "std"])
    mu = monthly_stats["mean"]
    sigma = monthly_stats["std"]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(mu.index, mu, lw=1.6, label="Monthly mean (MW)")
    ax.fill_between(
        mu.index, mu - sigma, mu + sigma, alpha=0.25, label="±1 σ (intramonth)"
    )
    ax.set_ylabel("Demand (MW)")
    ax.set_xlabel("Year")
    ax.set_title("Ontario Electricity Demand – Monthly Mean ± 1 σ")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(plots_dir / "monthly_mean_sigma.png", dpi=300)
    plt.cla()

    # ── 4. Annual trends & linear fit ────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(mu_y.index, mu_y, marker="o", label="Annual mean (MW)")
    ax1.set_ylabel("Mean demand")
    ax1.set_xlabel("Calendar year")

    ax2 = ax1.twinx()
    ax2.plot(sigma_y.index, sigma_y, ls="--", color="tab:red", label="Annual σ (MW)")
    ax2.set_ylabel("Volatility (σ)")

    slope, intercept, r, p, _ = linregress(mu_y.index.to_numpy(), mu_y)
    ax1.plot(
        mu_y.index,
        intercept + slope * mu_y.index,
        color="k",
        label=f"Linear trend (slope={slope:0.1f} MW/yr, p={p:0.3g})",
    )

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")

    plt.tight_layout()
    plt.savefig(plots_dir / "annual_trends.png", dpi=300)
    plt.cla()

    # ── 5. Year-wise z-score (detrend by calendar year) ──────────────────
    hourly["mu_year"] = hourly.index.year.map(mu_y)
    hourly["sigma_year"] = hourly.index.year.map(sigma_y)
    hourly["z_year"] = (
        hourly["Ontario Demand"] - hourly["mu_year"]
    ) / hourly["sigma_year"]

    # ── 6a. Seasonality removal: month-of-year z-score ───────────────────
    hourly["mu_month"] = hourly.groupby(hourly.index.month)[
        "Ontario Demand"
    ].transform("mean")
    hourly["sigma_month"] = hourly.groupby(hourly.index.month)[
        "Ontario Demand"
    ].transform("std")
    hourly["z_month"] = (
        hourly["Ontario Demand"] - hourly["mu_month"]
    ) / hourly["sigma_month"]

    # ── 6b. Seasonality removal: month+hour-of-day z-score ───────────────
    hourly["mu_mh"] = hourly.groupby(
        [hourly.index.month, hourly.index.hour]
    )["Ontario Demand"].transform("mean")
    hourly["sigma_mh"] = hourly.groupby(
        [hourly.index.month, hourly.index.hour]
    )["Ontario Demand"].transform("std")
    hourly["zscore"] = (
        hourly["Ontario Demand"] - hourly["mu_mh"]
    ) / hourly["sigma_mh"]

    # ── 7. Pick the stationary candidate (fullest de-seasonalization) ────
    stationary_candidate = hourly["zscore"].dropna()

    # ── 8. Plot stationary candidate ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(
        stationary_candidate.index,
        stationary_candidate,
        lw=0.8,
        label="Month + hour z-score",
    )
    ax.axhline(0, color="k", lw=0.6, ls="--", alpha=0.7)
    ax.set_title("Ontario Demand – Stationary Candidate")
    ax.set_ylabel("z-score")
    ax.set_xlabel("Time")
    ax.legend()
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(plots_dir / "stationary_candidate.png", dpi=300)
    plt.cla()

    # ── 9. Distribution of stationary candidate ──────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(stationary_candidate, kde=True, bins=60, ax=ax)
    ax.set_title("Distribution of Stationary Candidate")
    ax.set_xlabel("z-score")
    plt.tight_layout()
    plt.savefig(plots_dir / "stationary_candidate_dist.png", dpi=300)

    # ── 10. Save outputs ─────────────────────────────────────────────────
    stationary_candidate.to_csv(cfg.paths.stationary_candidate)

    params = pd.DataFrame({"mean": mu_y, "std": sigma_y, "cv": cv_y})
    params.to_csv(cfg.paths.annual_params)


if __name__ == "__main__":
    main(load_config())
