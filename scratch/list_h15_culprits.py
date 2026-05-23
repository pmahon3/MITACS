"""List the specific days that drive the 14:00->15:00 kurtosis collapse.

The full-library kurt-63 effect lives in the TAILS of the 14->15
demand-difference distribution. List the outliers (above the 99th and
below the 1st percentile, plus the absolute-extreme top-20) with date,
day-of-week, and demand-context. Lets us see if there is a calendar
pattern (e.g. always Mondays, always summer afternoons, holidays).

PROVENANCE-GRADE: INSPECTION-ONLY.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiment._actuals import load_actuals


def _daytype_simple(D: pd.Timestamp) -> str:
    """Day-of-week label for a calendar date."""
    return D.day_name()


def main():
    cutoff = pd.Timestamp("2024-12-31T23:00:00")
    raw = load_actuals(cutoff=None)
    raw = raw[raw.index <= cutoff]

    # Build a DataFrame of one row per delivery day with D(13), D(14),
    # D(15), D(16) and the three adjacent ΔD's. Drop days missing any.
    days = pd.DatetimeIndex(sorted({ts.normalize() for ts in raw.index}))
    rows = []
    for D in days:
        hh = [D + pd.Timedelta(hours=h) for h in (13, 14, 15, 16)]
        if not all(h in raw.index for h in hh):
            continue
        d13, d14, d15, d16 = (float(raw.loc[h]) for h in hh)
        rows.append({
            "date": D,
            "dow": _daytype_simple(D),
            "month": int(D.month),
            "year": int(D.year),
            "d13": d13, "d14": d14, "d15": d15, "d16": d16,
            "dD_13_14": d14 - d13,
            "dD_14_15": d15 - d14,
            "dD_15_16": d16 - d15,
        })
    df = pd.DataFrame(rows)
    print(f"complete delivery days: {len(df)}  "
          f"({df['date'].min().date()}..{df['date'].max().date()})")

    v = df["dD_14_15"].values
    print(f"\n14->15 ΔD pooled: mean={v.mean():+.0f} MW  "
          f"std={v.std():.0f}  min={v.min():+.0f}  max={v.max():+.0f}")
    p01, p99 = np.percentile(v, [1, 99])
    print(f"  1st pct = {p01:+.0f} MW   99th pct = {p99:+.0f} MW")

    # ----- Absolute extreme: top-20 by |ΔD - mean(ΔD)| -----------------
    mean_dD = v.mean()
    df["abs_dev_14_15"] = (df["dD_14_15"] - mean_dD).abs()
    top20 = df.sort_values("abs_dev_14_15", ascending=False).head(20)

    print("\n--- TOP 20 EXTREME 14->15 transitions (by |ΔD - mean|) ---")
    print(f"  {'date':>10}  {'dow':>10}  {'mo':>2}  {'D(14)':>6}  "
          f"{'D(15)':>6}  {'dD_14_15':>10}  {'context (Δ neighbours)':>26}")
    for _, r in top20.iterrows():
        print(f"  {str(r['date'].date()):>10}  {r['dow']:>10}  "
              f"{r['month']:>02d}  {r['d14']:>6.0f}  {r['d15']:>6.0f}  "
              f"{r['dD_14_15']:>+10.0f}  "
              f"({r['dD_13_14']:>+6.0f}, {r['dD_15_16']:>+6.0f})")

    # ----- Tail buckets: <1st and >99th percentile ---------------------
    lo = df[df["dD_14_15"] < p01].sort_values("dD_14_15")
    hi = df[df["dD_14_15"] > p99].sort_values("dD_14_15", ascending=False)

    print(f"\n--- LOWER tail (ΔD < {p01:+.0f} MW): n={len(lo)} ---")
    for _, r in lo.head(15).iterrows():
        print(f"  {str(r['date'].date()):>10}  {r['dow']:>10}  "
              f"mo={r['month']:>02d}  "
              f"D(14)={r['d14']:>6.0f}  D(15)={r['d15']:>6.0f}  "
              f"ΔD={r['dD_14_15']:>+8.0f}")

    print(f"\n--- UPPER tail (ΔD > {p99:+.0f} MW): n={len(hi)} ---")
    for _, r in hi.head(15).iterrows():
        print(f"  {str(r['date'].date()):>10}  {r['dow']:>10}  "
              f"mo={r['month']:>02d}  "
              f"D(14)={r['d14']:>6.0f}  D(15)={r['d15']:>6.0f}  "
              f"ΔD={r['dD_14_15']:>+8.0f}")

    # ----- Aggregate patterns: by day-of-week and by month ----------
    print("\n--- DAY-OF-WEEK distribution among top-100 extreme days ---")
    top100 = df.sort_values("abs_dev_14_15", ascending=False).head(100)
    dow_counts = top100["dow"].value_counts()
    overall_dow_share = df["dow"].value_counts(normalize=True)
    print(f"  {'dow':>10}  {'n_top100':>9}  {'expected':>9}  "
          f"{'over/under':>10}")
    for d in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
              "Saturday", "Sunday"]:
        n = int(dow_counts.get(d, 0))
        exp = float(overall_dow_share.get(d, 0)) * 100
        print(f"  {d:>10}  {n:>9d}  {exp:>9.1f}  "
              f"{(n - exp):>+10.1f}")

    print("\n--- MONTH distribution among top-100 extreme days ---")
    month_counts = top100["month"].value_counts()
    overall_month_share = df["month"].value_counts(normalize=True)
    print(f"  {'month':>5}  {'n_top100':>9}  {'expected':>9}  "
          f"{'over/under':>10}")
    for m in range(1, 13):
        n = int(month_counts.get(m, 0))
        exp = float(overall_month_share.get(m, 0)) * 100
        print(f"  {m:>5d}  {n:>9d}  {exp:>9.1f}  "
              f"{(n - exp):>+10.1f}")

    # ----- Year distribution: regime check -----------------------------
    print("\n--- YEAR distribution among top-100 extreme days ---")
    year_counts = top100["year"].value_counts().sort_index()
    for y, n in year_counts.items():
        share = float(year_counts[y]) / 100
        exp_share = float((df["year"] == y).mean())
        print(f"  {y}  n={n:>3d}  ({100*share:>4.1f}%)  "
              f"expected {100*exp_share:>4.1f}%")


if __name__ == "__main__":
    main()
