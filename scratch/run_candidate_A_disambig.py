"""Disambiguation experiment for §4.2's branch (iii):
run the scale-only pipeline with embedding dim FORCED to the seasonal
assignment {wkdy: 2, sat: 4, sun: 3}, so the additive sub-step is the
only remaining difference vs the seasonal pipeline. The candidate-A
result used elbow-selected dim {4, 4, 4}; the advisor closeout
(commit 5e74dea) flagged that the dimension shift confounds the
mechanistic attribution.

Pre-registered three-branch outcome (§4.2, commit e1eb06d):
  D-i  (MAPE_{2,4,3} < 5%):  dimension shift was the dominant driver;
                              branch (iii)'s additive-step attribution
                              is overclaimed.
  D-ii (5% <= MAPE_{2,4,3} < 9%): both dimension and additive
                                   contribute substantively.
  D-iii (MAPE_{2,4,3} >= 9%):  branch (iii) confirmed; additive step
                                does real work independent of
                                dimension.

HOLD verdict on removal is robust to all three branches; only the
mechanistic attribution updates.

PROVENANCE: INSPECTION-ONLY exploratory dev-set; not a recorded
result.
"""
from __future__ import annotations

import sys
import time
import numpy as np

from experiment._actuals import load_actuals
from scratch.diagnose_deseasonalisation_removal import (
    _backtest,
    _clim_monthhour,
    _z_scale_only,
    channel_1,
)

# Forced dims (seasonal pipeline's elbow assignment from commit 63cf264).
FORCED_DIMS = {"weekday": 2, "saturday": 4, "sunday": 3}

# Baselines for context.
SEASONAL_BASELINE = {"MAE": 627.0, "MAPE": 3.86}
SCALE_ONLY_ELBOW_RESULT = {"MAE": 1950.5, "MAPE": 11.98}
NO_TRANSFORM_RESULT = {"MAE": 2317.3, "MAPE": 13.57}


def _say(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


def main() -> None:
    t0 = time.time()
    _say("loading actuals", t0)
    act = load_actuals(cutoff=None).asfreq("h")

    _say("computing climatology + scale-only transform", t0)
    clim = _clim_monthhour(act)
    z_A = _z_scale_only(act, clim)

    _say(f"forcing per-day-type dims to seasonal assignment "
         f"{FORCED_DIMS} (NOT elbow-selected)", t0)

    _say("Channel 1: scale-only backtest at forced dims (slow; "
         "~10-15 min at smaller d)", t0)
    fc = _backtest(act, FORCED_DIMS, "scale_only")
    ch1 = channel_1(fc)
    mae = ch1["MAE_MW"]
    mape = ch1["MAPE_pct"]

    _say(f"Ch1 scale-only at {{2,4,3}}: MAE = {mae:.1f} MW, "
         f"MAPE = {mape:.3f}%, n = {ch1['n']}", t0)

    # Side-by-side
    print("\n" + "=" * 70, flush=True)
    print("=== Disambiguation: scale-only at FORCED dims {2,4,3} ===",
          flush=True)
    print("=" * 70, flush=True)
    print(f"  Seasonal              (dims {{2,4,3}}): "
          f"MAE = {SEASONAL_BASELINE['MAE']:.1f} MW, "
          f"MAPE = {SEASONAL_BASELINE['MAPE']:.3f}%", flush=True)
    print(f"  Scale-only @ forced   (dims {{2,4,3}}): "
          f"MAE = {mae:.1f} MW, MAPE = {mape:.3f}%", flush=True)
    print(f"  Scale-only @ elbow    (dims {{4,4,4}}): "
          f"MAE = {SCALE_ONLY_ELBOW_RESULT['MAE']:.1f} MW, "
          f"MAPE = {SCALE_ONLY_ELBOW_RESULT['MAPE']:.3f}%", flush=True)
    print(f"  No-transform          (dims {{3,2,4}}): "
          f"MAE = {NO_TRANSFORM_RESULT['MAE']:.1f} MW, "
          f"MAPE = {NO_TRANSFORM_RESULT['MAPE']:.3f}%", flush=True)
    print(flush=True)

    # Three-branch verdict (pre-registered)
    if mape < 5.0:
        branch = ("D-i: dimension shift was the dominant driver; "
                  "branch (iii) overclaimed; additive step's marginal "
                  "contribution is small")
    elif mape < 9.0:
        branch = ("D-ii: both dimension shift and additive sub-step "
                  "contribute substantively; branch (iii) partially "
                  "supported but not the sole story")
    else:
        branch = ("D-iii: branch (iii) confirmed; additive step does "
                  "real work independent of dimension")
    print(f"PRE-REGISTERED BRANCH: {branch}", flush=True)

    # HOLD-verdict-robustness reminder
    print(flush=True)
    print("HOLD verdict on de-seasonalisation removal is unchanged "
          "across all three branches; only the mechanistic attribution "
          "updates.", flush=True)


if __name__ == "__main__":
    main()
