"""Step 5 -- day-ahead head-to-head vs IESO's DATotals forecast.

Goal 3's real deliverable. Our multi-step day-ahead forecast (validated
in Step 4) vs IESO's published **Day-Ahead Totals (DATotals)** forecast
-- the correct product (the earlier Adequacy3 attempt was a horizon
mismatch; see memory ``mitacs-ieso-product-correction``). Both scored
against the same settled actual, on post-cutoff delivery days where all
three exist.

ISSUE-TIME RELATIONSHIP (stated honestly, NOT "caveat lifted"):
  * IESO DATotals for delivery day D is issued ~D-1 midday
    (verified ``CreatedAt`` ~12:30 D-1), forecasting the full calendar
    day D, 24h.
  * Our predictor issues at D-1 23:00 -- ~10h LATER, i.e. with slightly
    MORE recent information than IESO. This is a small edge to US and is
    reported, not hidden. It is the opposite direction and far smaller
    than the bogus Adequacy3 mismatch (which gave IESO ~all of D). The
    comparison is *approximately* horizon-matched (same product class,
    same delivery window) with a known ~10h issue-time gap in our favour
    -- characterise it, do not claim parity.

Honest framing carried from Step 4 (do not drop):
  * error reported by HOUR-OF-DAY (collinear with horizon in a day-ahead
    forecast) and as MAPE (level-controlled -- night demand is lower);
  * ours is a univariate kappa_Q estimator with NO weather / NO calendar
    beyond day-type / NO operational inputs; IESO uses all of those.
    Method-novelty comparison, NOT a claim of beating operational
    forecasting on its own terms;
  * sample is calendar-bound, grows over time -- explicit n + span;
  * frozen model (<=2024-12-31); the overlap window is fully
    out-of-sample for our predictor.

Run::  python -m experiment.headtohead
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import load_config

from . import freeze
from ._actuals import load_actuals
from .predict_multistep import day_ahead
from .score import _daytype


def _both_available_days() -> tuple[pd.DatetimeIndex, pd.Series, pd.Series]:
    """Delivery days with a full 24h IESO DATotals day-ahead forecast AND
    full 24h settled actuals."""
    cfg = load_config()
    da_csv = cfg.paths.forecast_csv.parent / "ieso_da_forecast.csv"
    if not da_csv.exists():
        raise FileNotFoundError(
            f"{da_csv} missing -- run `python -m data.da_forecast_scraping` "
            f"(DATotals is the correct day-ahead product; Adequacy3 is not)"
        )
    fc = pd.read_csv(da_csv, parse_dates=["datetime"])
    ieso = fc.set_index("datetime")["da_forecast_mw"].sort_index()
    act = load_actuals(cutoff=None)
    fc_hours = set(ieso.index)
    days = sorted({d.normalize() for d in ieso.index})
    keep = [
        D for D in days
        if all((D + pd.Timedelta(hours=h)) in act.index for h in range(24))
        and all((D + pd.Timedelta(hours=h)) in fc_hours for h in range(24))
    ]
    return pd.DatetimeIndex(keep), act, ieso


def _mae(p, a):
    e = (p - a).dropna()
    return float(e.abs().mean()) if len(e) else np.nan


def _mape(p, a):
    m = a != 0
    e = ((p[m] - a[m]).abs() / a[m]).dropna()
    return float(e.mean() * 100) if len(e) else np.nan


def run_headtohead(anchor_h: int = 7) -> dict:
    spec = freeze.load_verified()
    cutoff = pd.Timestamp(spec["data_cutoff"])
    days, act, ieso = _both_available_days()
    days = days[days > cutoff]  # out-of-sample for the frozen model
    if len(days) == 0:
        return {"sufficient": False,
                "note": "no post-cutoff days with both IESO fc and actuals"}

    ours = day_ahead(days)
    if ours.empty:
        return {"sufficient": False, "note": "predictor produced no rows"}

    df = ours.copy()
    df["actual_mw"] = df["target_dt"].map(act)
    df["ieso_mw"] = df["target_dt"].map(ieso)
    df = df.dropna(subset=["actual_mw", "ieso_mw", "our_forecast_mw"])
    if df.empty:
        return {"sufficient": False, "note": "no aligned rows after join"}

    per_h = []
    for h, g in df.groupby("horizon_h"):
        per_h.append({
            "hour_of_day": int(h) - 1,
            "n": int(len(g)),
            "ours_mape": _mape(g["our_forecast_mw"], g["actual_mw"]),
            "ieso_mape": _mape(g["ieso_mw"], g["actual_mw"]),
            "ours_mae": _mae(g["our_forecast_mw"], g["actual_mw"]),
            "ieso_mae": _mae(g["ieso_mw"], g["actual_mw"]),
        })

    win = float((
        (df["our_forecast_mw"] - df["actual_mw"]).abs()
        < (df["ieso_mw"] - df["actual_mw"]).abs()
    ).mean() * 100)

    return {
        "sufficient": True,
        "framing": (
            "Day-ahead vs IESO DATotals (true day-ahead product). "
            "Approx horizon-matched: same product class & delivery "
            "window; our issue D-1 23:00 is ~10h LATER than IESO's "
            "~D-1 12:30 -> small info edge to US, stated not hidden. "
            "NOT 'caveat lifted / parity'."
        ),
        "n_delivery_days": int(df["delivery_date"].nunique()),
        "n_hourly": int(len(df)),
        "span": (str(days.min().date()), str(days.max().date())),
        "ours_overall_mae": _mae(df["our_forecast_mw"], df["actual_mw"]),
        "ours_overall_mape": _mape(df["our_forecast_mw"], df["actual_mw"]),
        "ieso_overall_mae": _mae(df["ieso_mw"], df["actual_mw"]),
        "ieso_overall_mape": _mape(df["ieso_mw"], df["actual_mw"]),
        "ours_hour_win_rate_pct": round(win, 1),
        "per_hour": per_h,
        "caveats": (
            "Univariate kappa_Q estimator, NO weather/calendar/operational "
            "inputs; IESO uses all of those. NOT a claim of beating "
            "operational forecasting on its terms -- a method-novelty "
            "comparison. Sample is calendar-bound (n grows over time). "
            "Frozen model <=2024-12-31; window fully out-of-sample. "
            "Intervals omitted here (Step-4 showed Gaussian band ~3x too "
            "narrow under the verified heavy-tailed innovation)."
        ),
    }


if __name__ == "__main__":
    r = run_headtohead()
    if not r.get("sufficient"):
        print("INSUFFICIENT:", r.get("note"))
    else:
        print(f"HEAD-TO-HEAD (day-ahead vs day-ahead) -- {r['framing']}")
        print(f"  span {r['span']}  days={r['n_delivery_days']}  "
              f"hourly={r['n_hourly']}")
        print(f"  OURS : MAE {r['ours_overall_mae']:.0f} MW  "
              f"MAPE {r['ours_overall_mape']:.2f}%")
        print(f"  IESO : MAE {r['ieso_overall_mae']:.0f} MW  "
              f"MAPE {r['ieso_overall_mape']:.2f}%")
        print(f"  hours where ours beats IESO: "
              f"{r['ours_hour_win_rate_pct']}%")
        print("  by hour-of-day (MAPE; level-controlled):")
        print(f"    {'hod':>3} {'ours':>7} {'ieso':>7} {'n':>4}")
        for ph in r["per_hour"]:
            print(f"    {ph['hour_of_day']:>3} {ph['ours_mape']:>6.2f}% "
                  f"{ph['ieso_mape']:>6.2f}% {ph['n']:>4}")
        print(" ", r["caveats"])
