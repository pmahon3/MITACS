import os
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("TkAgg")

for daytype in ["weekday", "saturday", "sunday"]:
    data = pd.read_csv(f'./csv/results_{daytype}.csv', index_col=0)
    data = data[data < 5].dropna()
    data.index = pd.to_datetime(data.index)
    data.sort_index(inplace=True)

    df = (
        data
        .reset_index()
        .melt(id_vars="index", var_name="parameter", value_name="error")
        .rename(columns={"index": "timestamp"})
    )
    df["error"] = df["error"] * 100
    df["hour"] = df["timestamp"].dt.hour.astype(int)
    df["parameter"] = df["parameter"].astype(int)

    # group by hour & parameter
    stats = (
        df.groupby(["hour", "parameter"])["error"]
          .agg(["mean", "std"])
          .reset_index()
    )

    mean_mat = stats.pivot(index="hour", columns="parameter", values="mean")
    std_mat = stats.pivot(index="hour", columns="parameter", values="std")

    hours = mean_mat.index  # 0–23
    params = mean_mat.columns.astype(int)  # 0–10

    # ── Compute a common yticks array ──────────────────────────────
    # y-ticks
    yticks = np.arange(0, 101, 20)
    ytick_labels = [f"{y:.0f}%" for y in yticks]

    fig, axes = plt.subplots(
        nrows=6, ncols=4,
        figsize=(12, 12),
        sharex=True, sharey=True
    )

    for i, hr in enumerate(hours):
        ax = axes.flat[i]
        y  = mean_mat.loc[hr].values
        sd = std_mat.loc[hr].values

        ax.plot(params, y, lw=1)
        ax.fill_between(params, y - sd, y + sd, alpha=0.2)

        # set ticks
        ax.set_xticks(params)
        ax.set_yticks(yticks)

        # only label bottom row / first column
        row, col = divmod(i, 4)
        ax.tick_params(
            axis='x',
            labelbottom=(row == 5),
            labelrotation=45,
            labelsize=8
        )
        ax.tick_params(
            axis='y',
            labelleft=(col == 0),
            labelsize=8
        )

        ax.set_title(f"{hr:02d}:00", fontsize=10)

    fig.supxlabel("Parameter Value", fontsize=12)
    fig.supylabel("MAPE (%)", fontsize=12)
    fig.suptitle(f"MAPE ±1σ by Hour and Parameter for {daytype.capitalize()}s", fontsize=14)

    fig.tight_layout(rect=[0.05, 0.05, 1.00, 0.96])
    plt.savefig(f"./plots/mape_by_hour_and_bandwidth_{daytype}.png")
    plt.cla()
