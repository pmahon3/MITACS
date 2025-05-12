import os
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
from data import convert_to_absolute_path

# If you’re in Jupyter/VS Code you can drop this
matplotlib.use("TkAgg")

# ── Config ───────────────────────────────────────────────────────────────
csvs  = convert_to_absolute_path("HISTORICAL_CSVS")
years = range(*eval(os.environ["YEAR_RANGE"]), 1)    # e.g. (2015, 2025)

# ── 1. Load all raw hours ────────────────────────────────────────────────
frames = []
for yr in years:
    f = os.path.join(csvs, f"PUB_Demand_{yr}.csv")
    df = (
        pd.read_csv(f, skiprows=3)
          .dropna(subset=["Date","Hour","Ontario Demand"])
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

# ── 2. Annual mean & σ ───────────────────────────────────────────────────
mu_y    = hourly["Ontario Demand"].groupby(hourly.index.year).mean()
sigma_y = hourly["Ontario Demand"].groupby(hourly.index.year).std()
cv_y    = sigma_y / mu_y

# ── 3. Monthly mean + ±1 σ (intramonth) ─────────────────────────────────
monthly_stats = hourly["Ontario Demand"].resample("M").agg(["mean", "std"])
μ = monthly_stats["mean"]
σ = monthly_stats["std"]

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(μ.index, μ, lw=1.6, label="Monthly mean (MW)")
ax.fill_between(μ.index, μ - σ, μ + σ, alpha=0.25,
                label="±1 σ (intramonth)")
ax.set_ylabel("Demand (MW)")
ax.set_xlabel("Year")
ax.set_title("Ontario Electricity Demand – Monthly Mean ± 1 σ")
ax.legend(); ax.grid(alpha=0.25)
fig.autofmt_xdate(); plt.tight_layout()
plt.savefig("./plots/monthly_mean_sigma.png", dpi=300)
plt.cla()

# ── 4. Annual trends & linear fit ────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(mu_y.index, mu_y, marker="o", label="Annual mean (MW)")
ax1.set_ylabel("Mean demand"); ax1.set_xlabel("Calendar year")

ax2 = ax1.twinx()
ax2.plot(sigma_y.index, sigma_y, ls="--", color="tab:red",
         label="Annual σ (MW)")
ax2.set_ylabel("Volatility (σ)")

slope, intercept, r, p, _ = linregress(mu_y.index.to_numpy(), mu_y)
ax1.plot(mu_y.index,
         intercept + slope * mu_y.index,
         color="k",
         label=f"Linear trend (slope={slope:0.1f} MW/yr, p={p:0.3g})")

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper left")

plt.tight_layout()
plt.savefig("./plots/annual_trends.png", dpi=300)
plt.cla()

# ── 5. Year-wise z-score (detrend by calendar year) ──────────────────────
hourly["μ_year"]  = hourly.index.year.map(mu_y)
hourly["σ_year"]  = hourly.index.year.map(sigma_y)
hourly["z_year"]  = (
    hourly["Ontario Demand"] - hourly["μ_year"]
) / hourly["σ_year"]

# ── 6a. Seasonality removal: month-of-year z-score ───────────────────────
hourly["μ_month"] = hourly.groupby(hourly.index.month)["Ontario Demand"]\
                          .transform("mean")
hourly["σ_month"] = hourly.groupby(hourly.index.month)["Ontario Demand"]\
                          .transform("std")
hourly["z_month"] = (
    hourly["Ontario Demand"] - hourly["μ_month"]
) / hourly["σ_month"]

# ── 6b. Seasonality removal: month+hour-of-day z-score ──────────────────
hourly["μ_mh"] = hourly.groupby(
    [hourly.index.month, hourly.index.hour]
)["Ontario Demand"].transform("mean")
hourly["σ_mh"] = hourly.groupby(
    [hourly.index.month, hourly.index.hour]
)["Ontario Demand"].transform("std")
hourly["zscore"] = (
    hourly["Ontario Demand"] - hourly["μ_mh"]
) / hourly["σ_mh"]

# ── 7. Pick your stationary candidate ───────────────────────────────────
# e.g. for fullest de-seasonalization:
stationary_candidate = hourly["zscore"].dropna()

# ── 8. Plot stationary candidate ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(stationary_candidate.index, stationary_candidate,
        lw=0.8, label="Month + hour z-score")
ax.axhline(0, color="k", lw=0.6, ls="--", alpha=0.7)
ax.set_title("Ontario Demand – Stationary Candidate")
ax.set_ylabel("z-score"); ax.set_xlabel("Time"); ax.legend()
fig.autofmt_xdate(); plt.tight_layout()
plt.savefig("./plots/stationary_candidate.png", dpi=300)
plt.cla()

# ── 9. Distribution of stationary candidate ──────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
sns.histplot(stationary_candidate, kde=True, bins=60, ax=ax)
ax.set_title("Distribution of Stationary Candidate")
ax.set_xlabel("z-score")
plt.tight_layout()
plt.savefig("./plots/stationary_candidate_dist.png", dpi=300)

# ── 10. Save outputs ─────────────────────────────────────────────────────
stationary_candidate.to_csv("./csv/stationary_candidate.csv")

params = pd.DataFrame({
    "mean": mu_y,
    "std": sigma_y,
    "cv": cv_y
})
params.to_csv("./csv/annual_params.csv")
