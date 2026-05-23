"""Historical day-ahead backtest -- the basic-pipeline test, no IESO.

Validates the multi-step predictor end-to-end on real Ontario data
against ground truth we already have (the refreshed 2025-06..2026-05
actuals). No IESO, no calendar wait: for every complete delivery day in
the span, produce the iterated day-ahead forecast and compare to the
actual that already happened.

Honest by construction (memory mitacs-multistep-design / scoring-design):
  * error reported PER HORIZON h=1..24 -- the error-growth curve is the
    deliverable, never averaged into one number;
  * naive baselines: t-168h (day-type-CLEAN, primary) and t-24h (kept,
    its day-type-mismatch % reported);
  * leakage guard: predict_multistep already restricts fits to the
    frozen <=2024 cutoff; the refreshed actuals are used ONLY as query
    history up to each issue time and as scoring ground truth;
  * per-day checkpointing -> resumable; a partial run is still usable.

Run::

    python -m experiment.backtest                 # full span
    python -m experiment.backtest --max-days 30   # quick mechanics check
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from config import load_config

from ._actuals import load_actuals
from .predict_multistep import day_ahead
from .score import _daytype


def _complete_delivery_days(
    actual: pd.Series, anchor_h: int, dmax: int = 4
) -> pd.DatetimeIndex:
    """Dates D with all 24 actual hours of the *delivery-day window*
    (``D + anchor_h .. D + anchor_h + 23h``) AND >= ``dmax`` actual
    hours of pre-issue history immediately before ``D + anchor_h``.

    The window is anchored at ``cfg.data.day_anchor_hours`` so that the
    delivery-day clock and the day-type clock share one anchor (see
    memory ``mitacs-realignment``). ``anchor_h`` is mandatory -- the
    earlier ``anchor_h=7`` default coexisted silently with a 00:00
    delivery window and created the seam misalignment that produced the
    h=24 flip artefact.
    """
    days = pd.DatetimeIndex(sorted({ts.normalize() for ts in actual.index}))
    out = []
    idx = actual.index
    for D in days:
        win_start = D + pd.Timedelta(hours=anchor_h)
        day_hours = pd.date_range(win_start, periods=24, freq="h")
        if not all(h in idx for h in day_hours):
            continue
        hist = [win_start - pd.Timedelta(hours=i + 1) for i in range(dmax)]
        if all(h in idx for h in hist):
            out.append(D)
    return pd.DatetimeIndex(out)


def run_backtest(max_days: int | None = None) -> dict:
    cfg = load_config()
    anchor_h = cfg.data.day_anchor_hours
    actual = load_actuals(cutoff=None)
    cutoff = pd.Timestamp("2024-12-31T23:00:00")
    # backtest only on POST-cutoff days (true out-of-sample for the frozen
    # model; pre-cutoff days are training territory).
    days = _complete_delivery_days(actual, anchor_h)
    days = days[days > cutoff]
    if max_days:
        days = days[:max_days]
    if len(days) == 0:
        return {"sufficient": False, "note": "no complete post-cutoff days"}

    fc = day_ahead(days)
    if fc.empty:
        return {"sufficient": False, "note": "predictor produced no rows"}

    fc = fc.copy()
    fc["actual_mw"] = fc["target_dt"].map(actual)
    fc = fc.dropna(subset=["actual_mw"])

    # naive baselines aligned to each target
    fc["persist_168h"] = fc["target_dt"].map(
        lambda t: actual.get(t - pd.Timedelta(hours=168), np.nan)
    )
    fc["persist_24h"] = fc["target_dt"].map(
        lambda t: actual.get(t - pd.Timedelta(hours=24), np.nan)
    )

    def _mae(pred, act):
        e = (pred - act).dropna()
        return float(e.abs().mean()) if len(e) else np.nan

    def _mape(pred, act):
        m = act != 0
        e = ((pred[m] - act[m]).abs() / act[m]).dropna()
        return float(e.mean() * 100) if len(e) else np.nan

    # CRITICAL FRAMING: in a day-ahead forecast horizon h maps 1:1 to
    # hour-of-day (h=1 -> 00:00, h=24 -> 23:00). So "error by horizon" IS
    # "error by hour-of-day" -- they are perfectly collinear and must not
    # be read as pure horizon-compounding. The error structure is diurnal:
    # deep night (low usage) is easiest; the dawn/dusk demand RAMPS are
    # hardest (a well-known load-forecasting property), not the longest
    # horizons. Absolute MW error also tracks demand level, so MAPE
    # (level-controlled) is reported alongside MAE.
    per_h = []
    for h, g in fc.groupby("horizon_h"):
        per_h.append({
            "horizon_h": int(h),
            "hour_of_day": int(h) - 1,  # h=1 -> 00:00
            "n": int(len(g)),
            "ours_mae": _mae(g["our_forecast_mw"], g["actual_mw"]),
            "ours_mape": _mape(g["our_forecast_mw"], g["actual_mw"]),
            "persist168_mae": _mae(g["persist_168h"], g["actual_mw"]),
            "persist24_mae": _mae(g["persist_24h"], g["actual_mw"]),
            "mean_demand_mw": float(g["actual_mw"].mean()),
        })

    # t-24h day-type mismatch rate (the day-anchor crosses regime at
    # weekends -- the anchor is configured via cfg.data.day_anchor_hours).
    tt = fc["target_dt"]
    mism = np.mean([
        _daytype(t, anchor_h) != _daytype(t - pd.Timedelta(hours=24), anchor_h)
        for t in tt
    ]) * 100.0

    inside = (
        (fc["actual_mw"] >= fc["our_pi_lo_mw"])
        & (fc["actual_mw"] <= fc["our_pi_hi_mw"])
    ).mean() * 100.0

    return {
        "sufficient": True,
        "delivery_days": int(fc["delivery_date"].nunique()),
        "rows": int(len(fc)),
        "span": (str(days.min().date()), str(days.max().date())),
        "per_horizon": per_h,
        "overall_ours_mae": _mae(fc["our_forecast_mw"], fc["actual_mw"]),
        "overall_ours_mape": _mape(fc["our_forecast_mw"], fc["actual_mw"]),
        "overall_persist168_mae": _mae(fc["persist_168h"], fc["actual_mw"]),
        "persist24_daytype_mismatch_pct": round(float(mism), 1),
        "interval_coverage_pct": round(float(inside), 1),
        "interval_caveat": (
            "per-step Gaussian proxy, heavy-tailed innovation, no cross-step "
            "accumulation -> expected far from nominal; reported, not hidden"
        ),
        "framing": (
            "Backtest validates the multi-step predictor on real Ontario "
            "ground truth. NOT the IESO head-to-head (no IESO here); that "
            "is Step 5, horizon-matched. NOTE: horizon h is collinear with "
            "hour-of-day -- the error structure is DIURNAL (night easiest, "
            "dawn/dusk demand ramps hardest), NOT pure horizon compounding."
        ),
    }


def render_body(r: dict) -> str:
    """The human-readable backtest report. ONE renderer so the stdout
    view and the provenanced artifact are byte-identical by
    construction (no second formatting path to drift)."""
    L = [
        f"backtest: {r['delivery_days']} days, {r['rows']} hourly "
        f"forecasts, span {r['span']}",
        f"  overall ours MAE  = {r['overall_ours_mae']:.0f} MW  "
        f"MAPE = {r['overall_ours_mape']:.2f}%",
        f"  overall p168 MAE  = {r['overall_persist168_mae']:.0f} MW",
        f"  t-24h daytype mismatch = "
        f"{r['persist24_daytype_mismatch_pct']}%",
        f"  interval coverage = {r['interval_coverage_pct']}% "
        f"({r['interval_caveat']})",
        "  diurnal error structure (h == hour-of-day; MAPE controls "
        "for demand level):",
        f"    {'hod':>3} {'ours_MAE':>8} {'ours_MAPE':>9} "
        f"{'p168_MAE':>8} {'demand':>7} {'n':>4}",
    ]
    for ph in r["per_horizon"]:
        L.append(
            f"    {ph['hour_of_day']:>3} {ph['ours_mae']:>8.0f} "
            f"{ph['ours_mape']:>8.2f}% {ph['persist168_mae']:>8.0f} "
            f"{ph['mean_demand_mw']:>7.0f} {ph['n']:>4}"
        )
    L.append("  " + r["framing"])
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-days", type=int, default=None)
    ap.add_argument(
        "--emit-result", action="store_true",
        help="write the provenanced CLAIM-grade artifact (C1) via "
             "experiment.provenance.make_result (refuses a dirty tree)",
    )
    args = ap.parse_args()
    r = run_backtest(max_days=args.max_days)
    if not r.get("sufficient"):
        print("INSUFFICIENT:", r.get("note"))
        raise SystemExit(1)

    body = render_body(r)
    print(body, end="")

    if args.emit_result:
        from pathlib import Path

        from config import PROJECT_ROOT

        from ._actuals import actuals_fingerprint
        from .freeze import load_verified
        from .provenance import Grade, make_result

        cutoff = pd.Timestamp("2024-12-31T23:00:00")
        s0, s1 = r["span"]
        # Tag the artifact filename with the climatology method so
        # multiple frozen specs can coexist as parallel CLAIM-GRADE
        # artifacts on disk. month_hour is the historical default;
        # any other method produces a suffixed filename.
        spec = load_verified()
        clim_method = spec["predictor"].get("climatology_method", "month_hour")
        suffix = "" if clim_method == "month_hour" else f"_{clim_method}"
        out = (
            PROJECT_ROOT / "experiment" / "results"
            / f"backtest_postcutoff_{s0}_{s1}{suffix}.txt"
        )
        hdr = make_result(
            path=out,
            grade=Grade.CLAIM,
            title=f"Post-cutoff multi-step day-ahead backtest vs Ontario "
                  f"actuals (no IESO; climatology={clim_method})",
            body=body,
            inputs={
                "data_cutoff": cutoff.isoformat(),
                "pre_cutoff_actuals_sha256": actuals_fingerprint(cutoff),
                "backtest_span": list(r["span"]),
                "delivery_days": r["delivery_days"],
            },
            seeds={},  # the estimator/backtest path is RNG-free
            frozen_spec_required=True,  # predictor-derived: MUST bind
            extra={
                "climatology_method": clim_method,
                "note": (
                    f"Backtest under climatology_method={clim_method}. "
                    f"Parallel artifacts under alternative climatologies "
                    f"(if present) carry the climatology suffix in the "
                    f"filename (e.g. backtest_postcutoff_<span>_fourier.txt)."
                ),
            },
        )
        print(f"\nwrote provenanced C1 artifact -> {out}")
        print(f"  inputs_fingerprint = {hdr['inputs_fingerprint'][:16]}…")
        print(f"  body_sha256        = {hdr['body_sha256'][:16]}…")
        print(f"  frozen_spec_hash   = "
              f"{hdr['reproducibility']['frozen_spec_hash'][:16]}…")
