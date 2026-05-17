"""Stage 3 — tag each hour by day type.

Reads the stationary candidate, assigns ``daytype`` ∈ {weekday, saturday,
sunday} using a configurable day-start offset (the "day" begins at 07:00, not
midnight), writes the tagged CSV, and saves a mean 24-h profile plot.

Run standalone::

    python -m processing.clustering.process
"""
from __future__ import annotations

import matplotlib

# Headless-safe: the orchestration runner has no display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import PipelineConfig, load_config


def main(cfg: PipelineConfig) -> None:
    plots_dir = cfg.paths.clustering_plots
    plots_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.clustered_csv.parent.mkdir(parents=True, exist_ok=True)

    # ── Load the stationary candidate (output of stage 2) ────────────────
    stationary_data = pd.read_csv(cfg.paths.stationary_candidate, index_col=0)
    stationary_data.index = pd.to_datetime(stationary_data.index)

    # ── 0. Setup ─────────────────────────────────────────────────────────
    offset = pd.Timedelta(hours=cfg.data.day_anchor_hours)  # 07:00 -> "midnight"
    shifted_idx = stationary_data.index - offset

    # ── 1. Tag each row by day type ──────────────────────────────────────
    daytype = np.select(
        [
            shifted_idx.dayofweek < 5,
            shifted_idx.dayofweek == 5,
            shifted_idx.dayofweek == 6,
        ],
        ["weekday", "saturday", "sunday"],
        default="other",
    )
    stationary_data["daytype"] = pd.Categorical(
        daytype, categories=["weekday", "saturday", "sunday"]
    )

    # Save the data with daytype (input to stages 4+)
    stationary_data.to_csv(cfg.paths.clustered_csv)

    # ── 2. Extract "time-of-day" bins starting at the day anchor ─────────
    stationary_data["tod"] = shifted_idx.floor("1h").time

    # ── 3. Aggregate: mean & σ for every hour of every day-type ─────────
    profile = (
        stationary_data.groupby(["daytype", "tod"])[cfg.data.variable_name]
        .agg(mean="mean", std="std")
        .reset_index()
    )

    # ── 4. Plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))

    for dtype, grp in profile.groupby("daytype"):
        x = np.array([t.hour + t.minute / 60 for t in grp["tod"]])
        (line,) = ax.plot(x, grp["mean"], label=dtype.capitalize())
        ax.fill_between(
            x,
            grp["mean"] - grp["std"],
            grp["mean"] + grp["std"],
            alpha=0.25,
            color=line.get_color(),
        )

    anchor = cfg.data.day_anchor_hours
    xticks = np.arange(0, 25, 2)
    xtick_labels = [f"{(anchor + h) % 24:02d}:00" for h in xticks]

    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, rotation=0)
    ax.set_xlim(0, 24)
    ax.set_ylabel("Mean value")
    ax.set_title("Mean 24-h profile (±1 σ band) by day type")
    ax.grid(alpha=0.3)
    ax.legend(title="Day type")
    plt.tight_layout()
    plt.savefig(plots_dir / "mean_24h_profile.png", dpi=300)


if __name__ == "__main__":
    main(load_config())
