import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")

# ── Load your data ────────────────────────────────────────────────
stationary_data = pd.read_csv("/Users/pmahon/Research/Dynamics/MITACS/preprocessing/csv/stationary_candidate.csv", index_col=0)
stationary_data.index = pd.to_datetime(stationary_data.index)

# ── 0.  Setup  ──────────────────────────────────────────────────────────
offset = pd.Timedelta(hours=7)          # 07:00 ‑> “midnight”
shifted_idx = stationary_data.index - offset

# ── 1.  Tag each row by day type  ───────────────────────────────────────
daytype = np.select(
    [shifted_idx.dayofweek < 5, shifted_idx.dayofweek == 5, shifted_idx.dayofweek == 6],
    ["weekday", "saturday", "sunday"],
    default="other"
)
stationary_data["daytype"] = pd.Categorical(daytype, categories=["weekday", "saturday", "sunday"])

# Save the data with daytype
stationary_data.to_csv("./csv/stationary_candidate.csv")

# ── 2.  Extract “time‑of‑day” bins starting at 07:00h  ────────────────────
# here we bin to the nearest hour; for 5‑min data use .floor("5min")
stationary_data["tod"] = shifted_idx.floor("1h").time          # time objects 00:00 … 23:00

# ── 3.  Aggregate: mean & σ for every hour of every day‑type  ───────────
profile = (
    stationary_data.groupby(["daytype", "tod"])["zscore"]       # <-- replace with your column
      .agg(mean="mean", std="std")
      .reset_index()
)

# ── 4.  Plot  ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

for dtype, grp in profile.groupby("daytype"):
    # hours since the 07:00 start‑of‑day anchor
    x = np.array([t.hour + t.minute / 60 for t in grp["tod"]])

    # main curve
    (line,) = ax.plot(x, grp["mean"], label=dtype.capitalize())

    # coloured ±1 σ band (same colour as the curve, just translucent)
    ax.fill_between(
        x,
        grp["mean"] - grp["std"],
        grp["mean"] + grp["std"],
        alpha=0.25,
        color=line.get_color(),   # re‑use the auto‑selected line colour
    )

xticks = np.arange(0, 25, 2)                      # still every 2 h
xtick_labels = [f'{(7 + h) % 24:02d}:00'          # 07:00, 09:00, …, 05:00, 07:00
                for h in xticks]

ax.set_xticks(xticks)
ax.set_xticklabels(xtick_labels, rotation=0)
ax.set_xlim(0, 24)
ax.set_ylabel("Mean value")
ax.set_title("Mean 24‑h profile (±1 σ band) by day type")
ax.grid(alpha=0.3)
ax.legend(title="Day type")
plt.tight_layout()

plt.savefig("./plots/mean_24h_profile.png", dpi=300)
