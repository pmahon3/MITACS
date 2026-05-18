"""Append-only immutable experiment ledger.

Every forecast we ever issue is recorded here, once, and never silently
revised -- the same registered-experiment discipline as the actuals
provenance ledger. One row per (target_dt, spec_hash). A later cycle may
*fill in* the settled actual + its provenance for a row, but the
forecast fields (our_forecast_mw, intervals, issue_time, was_real_time,
ieso_forecast_mw) are write-once.

SCOPE / FRAMING (do not let this drift -- see memory
``mitacs-scoring-design``): this ledger and the one-step skill it enables
are INFRASTRUCTURE plus an explicitly *interim* result. Our estimator is
one-step; IESO publishes day-ahead. The IESO column here is recorded only
as a different-horizon operational reference -- a 1h-ahead vs ~24h-ahead
error comparison is trivially won by the shorter horizon and is NOT a
finding. The real Goal 3 head-to-head needs the multi-step predictor at
IESO's horizon (Task 23).
"""
from __future__ import annotations

import pandas as pd

from config import load_config

# Forecast fields are write-once; only the settled-actual block is
# back-filled later.
LEDGER_COLS = [
    "target_dt", "spec_hash",
    "our_forecast_mw", "our_pi_lo_mw", "our_pi_hi_mw",
    "ieso_forecast_mw",            # different-horizon reference ONLY
    "issue_time", "was_real_time", "daytype", "embedding_dim",
    "zscore_params_fp",
    # back-filled at settlement (still write-once once non-null):
    "actual_mw", "actual_scrape_time", "actual_status", "actual_source_sha",
]

_KEY = ["target_dt", "spec_hash"]


def _ledger_path():
    return load_config().paths.forecast_csv.parent / "experiment_ledger.csv"


def _empty() -> pd.DataFrame:
    dt = {"target_dt": "datetime64[ns]", "was_real_time": "object",
          "our_forecast_mw": "float64", "our_pi_lo_mw": "float64",
          "our_pi_hi_mw": "float64", "ieso_forecast_mw": "float64",
          "embedding_dim": "float64", "actual_mw": "float64"}
    return pd.DataFrame(
        {c: pd.Series(dtype=dt.get(c, "object")) for c in LEDGER_COLS}
    )[LEDGER_COLS]


def read_ledger() -> pd.DataFrame:
    p = _ledger_path()
    if p.exists():
        df = pd.read_csv(p, parse_dates=["target_dt"])
        for c in LEDGER_COLS:
            if c not in df.columns:
                df[c] = pd.NA
        return df[LEDGER_COLS]
    return _empty()


def update_ledger(joined: pd.DataFrame) -> pd.DataFrame:
    """Merge a ``collect.run_cycle`` joined frame into the append-only
    ledger.

    Rules:
      * a (target_dt, spec_hash) not yet present -> appended (forecast
        fields write-once);
      * already present, actual still null, now settled -> back-fill the
        settled-actual block only (forecast fields untouched);
      * already present with a settled actual -> left ALONE (immutable;
        actual revisions live in the settled_actuals provenance ledger,
        not here -- this ledger records what was graded, once).
    """
    led = read_ledger()
    existing = {
        (pd.Timestamp(r.target_dt), r.spec_hash): i
        for i, r in led.iterrows()
    }

    incoming = joined.copy()
    if "target_dt" in incoming:
        incoming["target_dt"] = pd.to_datetime(incoming["target_dt"])

    new_rows = []
    for _, r in incoming.iterrows():
        key = (pd.Timestamp(r["target_dt"]), r.get("spec_hash"))
        rowd = {c: r.get(c, pd.NA) for c in LEDGER_COLS}
        if key not in existing:
            new_rows.append(rowd)
            continue
        i = existing[key]
        # back-fill settled actual ONLY if not already graded
        if pd.isna(led.at[i, "actual_mw"]) and not pd.isna(
            r.get("actual_mw", pd.NA)
        ):
            for c in ("actual_mw", "actual_scrape_time",
                      "actual_status", "actual_source_sha"):
                if c in r and not pd.isna(r.get(c)):
                    led.at[i, c] = r[c]

    if new_rows:
        nr = pd.DataFrame(new_rows, columns=LEDGER_COLS)
        led = nr if led.empty else pd.concat([led, nr], ignore_index=True)

    led = led.sort_values(_KEY).reset_index(drop=True)
    p = _ledger_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    led.to_csv(p, index=False)
    return led


if __name__ == "__main__":
    led = read_ledger()
    n_settled = led["actual_mw"].notna().sum() if len(led) else 0
    print(f"experiment ledger: {len(led)} rows, {n_settled} graded")
    if len(led):
        print(led.tail(5).to_string(index=False))
