"""Scheduled scrape + actuals settlement -- the experiment's heartbeat.

Each run, idempotently:

  1. SCRAPE  -- refresh archived IESO day-ahead forecasts
               (data.forecast_scraping) and fetch current settled IESO
               actual demand (the Demand/ report) with provenance.
  2. SETTLE  -- write actuals into ONE append-only provenance ledger.
               actual_mw is WRITE-ONCE per target_hour; a differing later
               scrape APPENDS a revision row (never overwrites) -- IESO
               restates recent demand, and a registered experiment must
               not let its grading truth change silently. A same-value
               re-scrape stamps confirmed_unchanged_on and may promote
               prelim -> final.
  3. PREDICT -- generate our registered forecast for every forecast
               target whose <=t-1h query history now exists; stamp
               issue_time and was_real_time (issue_time < target_dt).
  4. JOIN    -- emit the (target, our_forecast, ieso_forecast, settled
               actual) frame for the experiment ledger (Task 27).

The settled-actuals ledger is the single source of provenance: the
"per-scrape manifest" and "prelim/final" views are queries over it, not
separate artifacts.

Usage::

    python -m experiment.collect              # full cycle
    python -m experiment.collect --no-scrape  # settle/predict from local
"""
from __future__ import annotations

import argparse
import hashlib
import io
import re
from datetime import datetime, timezone

import pandas as pd
import requests

from config import load_config

from . import freeze, predict
from ._actuals import load_actuals

DEMAND_BASE = "https://reports-public.ieso.ca/public/Demand/"
_CREATED_RE = re.compile(r"Created at ([0-9:\- ]+)")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── 1. actuals fetch with provenance ──────────────────────────────────────
def fetch_actuals_with_provenance(years: list[int]) -> dict:
    """Download the per-year PUB_Demand files, returning parsed hourly
    demand plus the provenance of each source file (IESO 'Created at'
    header + content sha256)."""
    cfg = load_config()
    dest = cfg.paths.historical_csvs
    dest.mkdir(parents=True, exist_ok=True)
    prov: dict[str, dict] = {}
    for yr in years:
        name = f"PUB_Demand_{yr}.csv"
        try:
            r = requests.get(DEMAND_BASE + name, timeout=90)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"  ! actuals fetch failed {name}: {e}")
            continue
        content = r.content
        text = content.decode("utf-8", "replace")
        m = _CREATED_RE.search(text[:200])
        prov[name] = {
            "ieso_created_at": m.group(1).strip() if m else "",
            "source_sha": hashlib.sha256(content).hexdigest(),
            "scrape_time": _now(),
            "source_url": DEMAND_BASE + name,
        }
        (dest / name).write_bytes(content)  # refresh local copy
    return prov


# ── 2. append-only settled-actuals provenance ledger ──────────────────────
SETTLED_COLS = [
    "target_hour", "actual_mw", "scrape_time", "ieso_created_at",
    "source_sha", "source_file", "status", "first_seen",
    "confirmed_unchanged_on",
]


def _empty_settled() -> pd.DataFrame:
    """Empty ledger with explicit dtypes. Without this, pandas infers
    object/float64 for empty columns and later string assignments (e.g.
    confirmed_unchanged_on timestamps) trip an incompatible-dtype error
    in modern pandas."""
    return pd.DataFrame(
        {
            "target_hour": pd.Series(dtype="datetime64[ns]"),
            "actual_mw": pd.Series(dtype="float64"),
            "scrape_time": pd.Series(dtype="object"),
            "ieso_created_at": pd.Series(dtype="object"),
            "source_sha": pd.Series(dtype="object"),
            "source_file": pd.Series(dtype="object"),
            "status": pd.Series(dtype="object"),
            "first_seen": pd.Series(dtype="object"),
            "confirmed_unchanged_on": pd.Series(dtype="object"),
        }
    )[SETTLED_COLS]


def _read_settled(path) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path, parse_dates=["target_hour"])
        for c in (
            "scrape_time", "ieso_created_at", "source_sha", "source_file",
            "status", "first_seen", "confirmed_unchanged_on",
        ):
            df[c] = df[c].astype("object")
        return df
    return _empty_settled()


def settle_actuals(prov: dict, target_hours: pd.DatetimeIndex) -> pd.DataFrame:
    """Apply the write-once / revision-as-append rule, scoped to
    EXPERIMENT TARGET hours only.

    The settled-actuals ledger is the experiment's grading truth -- it
    must contain only hours we are scoring a forecast against (the IESO
    forecast targets), NOT the entire 2002-present demand record. Settling
    all of history would conflate the model's training data with the
    experiment's evaluation truth.
    """
    cfg = load_config()
    path = cfg.paths.forecast_csv.parent / "settled_actuals.csv"
    ledger = _read_settled(path)
    known = {}  # target_hour -> latest actual_mw currently in ledger
    if not ledger.empty:
        latest = ledger.sort_values("scrape_time").groupby("target_hour").last()
        known = latest["actual_mw"].to_dict()

    all_actuals = load_actuals(cutoff=None)
    tset = pd.DatetimeIndex(target_hours)
    fresh = all_actuals[all_actuals.index.isin(tset)]  # targets only
    new_rows = []
    now = _now()
    # map each target hour to the provenance of the file it came from
    for ts, mw in fresh.items():
        yr = ts.year
        src = f"PUB_Demand_{yr}.csv"
        p = prov.get(src)
        if p is None:
            continue  # only settle hours from files we fetched this run
        prev = known.get(ts)
        if prev is None:
            new_rows.append({
                "target_hour": ts, "actual_mw": float(mw),
                "scrape_time": p["scrape_time"],
                "ieso_created_at": p["ieso_created_at"],
                "source_sha": p["source_sha"], "source_file": src,
                "status": "prelim", "first_seen": now,
                "confirmed_unchanged_on": "",
            })
        elif float(mw) != float(prev):
            # IESO restatement -> APPEND a revision row, never overwrite
            new_rows.append({
                "target_hour": ts, "actual_mw": float(mw),
                "scrape_time": p["scrape_time"],
                "ieso_created_at": p["ieso_created_at"],
                "source_sha": p["source_sha"], "source_file": src,
                "status": "revised", "first_seen": now,
                "confirmed_unchanged_on": "",
            })
        # else: unchanged -> stamp confirmation on the existing latest row
        elif not ledger.empty:
            mask = (ledger["target_hour"] == ts)
            if mask.any():
                idx = ledger.loc[mask, "scrape_time"].idxmax()
                ledger.loc[idx, "confirmed_unchanged_on"] = now
                if ledger.loc[idx, "status"] == "prelim":
                    ledger.loc[idx, "status"] = "final"

    if new_rows:
        nr = pd.DataFrame(new_rows, columns=SETTLED_COLS)
        # avoid the empty/all-NA concat dtype FutureWarning
        out = nr if ledger.empty else pd.concat(
            [ledger, nr], ignore_index=True
        )
    else:
        out = ledger  # only confirmation stamps changed (or nothing)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def gradable_actuals(settled: pd.DataFrame) -> pd.Series:
    """Latest (most recently scraped) actual per target hour -- the value
    a forecast is graded against."""
    if settled.empty:
        return pd.Series(dtype=float)
    latest = (
        settled.sort_values("scrape_time")
        .groupby("target_hour")
        .last()
    )
    return latest["actual_mw"]


# ── full cycle ────────────────────────────────────────────────────────────
def run_cycle(do_scrape: bool = True) -> pd.DataFrame:
    spec = freeze.load_verified()  # experiment integrity gate
    cfg = load_config()

    if do_scrape:
        from data.forecast_scraping import main as scrape_forecasts

        scrape_forecasts(cfg)  # refresh archived IESO forecasts
        fc = pd.read_csv(cfg.paths.forecast_csv, parse_dates=["datetime"])
        tgt_years = sorted({d.year for d in fc["datetime"]})
        prov = fetch_actuals_with_provenance(tgt_years)
    else:
        # offline: settle from local files only. Synthesize minimal prov
        # for the forecast-target years (clearly marked source=local so
        # provenance never falsely claims a network fetch).
        fc = pd.read_csv(cfg.paths.forecast_csv, parse_dates=["datetime"])
        prov = {
            f"PUB_Demand_{y}.csv": {
                "ieso_created_at": "", "source_sha": "local",
                "scrape_time": _now(), "source_url": "local",
            }
            for y in sorted({d.year for d in fc["datetime"]})
        }

    # forecast targets define the experiment's scoring scope
    fc = pd.read_csv(cfg.paths.forecast_csv, parse_dates=["datetime"])
    targets = pd.DatetimeIndex(fc["datetime"])

    settled = settle_actuals(prov, targets)  # targets-scoped grading truth
    grad = gradable_actuals(settled)

    # our registered forecasts for archived IESO targets whose query
    # history exists; predict.generate already enforces the hash + cutoff.
    issue_time = pd.Timestamp(_now())
    ours = predict.generate(pd.DatetimeIndex(fc["datetime"]))
    if not ours.empty:
        ours["issue_time"] = issue_time.isoformat()
        ours["was_real_time"] = ours["target_dt"] > issue_time

    # join: our forecast + IESO forecast + settled actual (where present)
    ieso = fc.rename(
        columns={"datetime": "target_dt", "forecast_mw": "ieso_forecast_mw"}
    )
    joined = ieso.merge(ours, on="target_dt", how="left") if not ours.empty \
        else ieso.assign(our_forecast_mw=pd.NA)
    joined["actual_mw"] = joined["target_dt"].map(grad)
    joined["spec_hash"] = spec["spec_hash"]

    n_set = joined["actual_mw"].notna().sum()
    n_our = (
        joined["our_forecast_mw"].notna().sum()
        if "our_forecast_mw" in joined else 0
    )
    print(
        f"cycle: {len(joined)} targets | {n_our} with our forecast | "
        f"{n_set} settled | settled-ledger rows {len(settled)}"
    )
    return joined


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--no-scrape", action="store_true",
        help="settle/predict from local files only (no network)",
    )
    args = ap.parse_args()
    run_cycle(do_scrape=not args.no_scrape)
