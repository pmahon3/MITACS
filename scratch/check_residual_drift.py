"""Phase-1 check: does the h=1 forecast residual drift over time on
the extended dev set? If yes, scoring needs a linear detrend; if no,
the existing K/S protocols work as-is on the extended captures."""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd

CACHE = Path("/tmp/k_step_sequences_extended.pkl")
DRIFT_BAR = 0.05    # z-units per year — if residual drift exceeds this,
                    # add a linear residual detrend in scoring


def main() -> None:
    if not CACHE.exists():
        print(f"ERROR: {CACHE} not found", file=sys.stderr)
        sys.exit(1)

    with CACHE.open("rb") as f:
        blob = pickle.load(f)
    captures = blob["captures"]
    rows = []
    for day, day_rows in captures.items():
        r1 = day_rows[0]   # h=1
        if not np.isfinite(r1["z_actual"]) or not np.isfinite(r1["z_pred"]):
            continue
        rows.append({
            "day": pd.Timestamp(day),
            "daytype": r1["daytype"],
            "z_pred": r1["z_pred"],
            "z_actual": r1["z_actual"],
            "sigma00": r1["Sigma_00"],
        })
    df = pd.DataFrame(rows)
    df["resid"] = df["z_actual"] - df["z_pred"]
    df["z_std"] = df["resid"] / np.sqrt(np.maximum(df["sigma00"], 1e-9))
    df["days_since_start"] = (df["day"] - df["day"].min()).dt.days

    print(f"Extended dev set: {len(df)} days, {df['day'].min().date()} "
          f"to {df['day'].max().date()}")
    print(f"Window length: {df['days_since_start'].max()} days = "
          f"{df['days_since_start'].max() / 365:.2f} years")
    print()

    # Linear residual drift
    s, ic = np.polyfit(df["days_since_start"].values, df["resid"].values, 1)
    yearly_drift = s * 365
    print(f"h=1 residual linear drift:")
    print(f"  slope = {s:+.6f} z-units/day")
    print(f"  yearly drift = {yearly_drift:+.4f} z-units/year")
    print(f"  drift bar    = {DRIFT_BAR:+.4f} z-units/year")
    if abs(yearly_drift) < DRIFT_BAR:
        print("  -> RESIDUAL STATIONARY (within bar). "
              "Existing scoring protocols work as-is; no detrend needed.")
    else:
        print(f"  -> RESIDUAL DRIFTING. Add linear detrend to scoring: "
              f"residual_corrected = residual - ({s:+.6f} * days + {ic:+.4f}).")

    # Per-year residual stats
    df["year"] = df["day"].dt.year
    print()
    print("Per-year h=1 residual stats:")
    print(f"  {'year':>6} {'n':>5} {'mean':>9} {'std':>8} {'kurt':>9}")
    for year, g in df.groupby("year"):
        if len(g) < 4:
            continue
        m, s_ = g["resid"].mean(), g["resid"].std()
        k = float(pd.Series(g["resid"].values).kurtosis())
        print(f"  {year:>6} {len(g):>5} {m:>+9.4f} {s_:>8.4f} {k:>9.3f}")

    # Per-year standardised residual stats — the actual gate
    print()
    print("Per-year h=1 STANDARDISED residual stats (the kurt gate):")
    print(f"  {'year':>6} {'n':>5} {'mean':>9} {'std':>8} {'kurt':>9}")
    for year, g in df.groupby("year"):
        if len(g) < 4:
            continue
        z = g["z_std"].values
        z = z[np.isfinite(z)]
        if len(z) < 4:
            continue
        k = float(pd.Series(z).kurtosis())
        print(f"  {year:>6} {len(z):>5} {z.mean():>+9.4f} {z.std():>8.4f} "
              f"{k:>9.3f}")


if __name__ == "__main__":
    main()
