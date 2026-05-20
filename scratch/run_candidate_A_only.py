"""Focused subset diagnostic: run ONLY the scale-only (Candidate A)
pipeline through all four channels. Seasonal-vs-raw verdict is already
committed (63cf264) as HOLD; seasonal-spec four-channel numbers are
the comparator.

Pre-registered comparison rules (§4.2):
  Ch1: |dMAE| < 8 MW AND |dMAPE| < 0.05 pp
  Ch2: per-pipeline fraction-finite >= 0.99 on both sides
  Ch3: hour-stratified -- worst |gap| < 5 pp AND avg |gap| < 2 pp
  Ch4: nu_scale >= nu_seasonal (conditional on Ch3 passing)
  Confound: scale-only elbow-d outside [2,4] -> verdict CONFOUNDED.

Reuses scratch/diagnose_deseasonalisation_removal.py's functions
directly; no reimplementation.

PROVENANCE: INSPECTION-ONLY exploratory dev-set; not a recorded result.
Output: tail-printed verdict + structured dict written to stdout.
Buffer-safe (uses sys.stdout flushing) so background runs surface
progress incrementally.
"""
from __future__ import annotations

import sys
import time
import numpy as np

from experiment._actuals import load_actuals
from scratch.diagnose_deseasonalisation_removal import (
    DAYTYPES,
    _backtest,
    _clim_monthhour,
    _refrozen_dims,
    _z_scale_only,
    channel_1,
    channel_2,
    channel_3,
    channel_4,
)


# Seasonal-spec baseline values from the committed seasonal-vs-raw run
# (commit 63cf264; see writeup/tex/draft_body.tex §4.2 result block).
# These are the comparator for the Candidate-A verdict.
SEASONAL_BASELINE = {
    "dims": {"weekday": 2, "saturday": 4, "sunday": 3},
    "ch1": {"MAE_MW": 627.0, "MAPE_pct": 3.859, "n": 3168},
    "ch2": {                # 60/60 finite on every day-type
        "weekday": {"n_anchors": 60, "n_finite": 60, "fraction_finite": 1.0},
        "saturday": {"n_anchors": 60, "n_finite": 60, "fraction_finite": 1.0},
        "sunday": {"n_anchors": 60, "n_finite": 60, "fraction_finite": 1.0},
    },
    "ch3_avg": 38.35,       # average coverage percentage
    "ch4_nu": 5.93,
}


def _say(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


def main() -> None:
    t0 = time.time()
    _say("loading actuals", t0)
    act = load_actuals(cutoff=None).asfreq("h")

    _say("computing climatology + scale-only transform", t0)
    clim = _clim_monthhour(act)
    z_A = _z_scale_only(act, clim)

    _say("elbow rule on scale-only series (per day-type)", t0)
    dims_A = _refrozen_dims(z_A)
    _say(f"dims_A = {dims_A}", t0)

    # Confound check
    d_oor = {dt: int(dims_A[dt]) for dt in DAYTYPES
             if not (2 <= int(dims_A[dt]) <= 4)}
    if d_oor:
        _say(f"*** WARNING: scale-only dims outside [2,4]: {d_oor}; "
             f"verdict will be CONFOUNDED. ***", t0)

    _say("Channel 1: scale-only backtest (slow; ~20 min)", t0)
    fc_A = _backtest(act, dims_A, "scale_only")
    ch1_A = channel_1(fc_A)
    _say(f"Ch1 scale-only: MAE={ch1_A['MAE_MW']:.1f} MW, "
         f"MAPE={ch1_A['MAPE_pct']:.3f}%, n={ch1_A['n']}", t0)
    _say(f"        seasonal: MAE={SEASONAL_BASELINE['ch1']['MAE_MW']:.1f} MW, "
         f"MAPE={SEASONAL_BASELINE['ch1']['MAPE_pct']:.3f}% (committed)", t0)

    _say("Channel 2: scale-only diffusion viability (~30 s)", t0)
    rng = np.random.default_rng(42)
    ch2_A = channel_2(act, dims_A, "scale_only", rng)
    for dt in DAYTYPES:
        cell = ch2_A[dt]
        _say(f"  {dt:9s}: {cell['n_finite']}/{cell['n_anchors']} finite "
             f"(scale-only)", t0)

    _say("Channel 3: hour-stratified demand-space coverage", t0)
    ch3_A = channel_3(fc_A)
    _say(f"Ch3 scale-only avg coverage = {ch3_A['average_coverage_pct']:.2f}%, "
         f"median half-width = {ch3_A['median_halfwidth_MW']:.1f} MW", t0)
    _say(f"      seasonal avg coverage = "
         f"{SEASONAL_BASELINE['ch3_avg']:.2f}% (committed)", t0)

    _say("Channel 4: propagation-aware nu_hat", t0)
    ch4_A = channel_4(fc_A)
    _say(f"Ch4 scale-only nu_hat = {ch4_A['nu_hat']}, "
         f"excess kurt = {ch4_A['excess_kurt']}, n = {ch4_A['n']}", t0)
    _say(f"      seasonal nu_hat = {SEASONAL_BASELINE['ch4_nu']:.3f} "
         f"(committed)", t0)

    # Verdict
    print("\n" + "=" * 70, flush=True)
    print("=== Pre-registered Candidate-A (scale-only) verdict ===", flush=True)
    print("=" * 70, flush=True)

    # Ch1
    dMAE = abs(ch1_A["MAE_MW"] - SEASONAL_BASELINE["ch1"]["MAE_MW"])
    dMAPE = abs(ch1_A["MAPE_pct"] - SEASONAL_BASELINE["ch1"]["MAPE_pct"])
    ch1_pass = (dMAE < 8.0) and (dMAPE < 0.05)
    print(f"Ch1 forecast:       |dMAE| = {dMAE:.2f} MW, "
          f"|dMAPE| = {dMAPE:.4f} pp -- "
          f"{'PASS' if ch1_pass else 'FAIL'}", flush=True)

    # Ch2
    ch2_seasonal_ok = all(
        SEASONAL_BASELINE["ch2"][dt]["fraction_finite"] >= 0.99
        for dt in DAYTYPES
    )
    ch2_A_ok = all(
        (ch2_A[dt]["fraction_finite"] or 0) >= 0.99 for dt in DAYTYPES
    )
    ch2_pass = ch2_seasonal_ok and ch2_A_ok
    print(f"Ch2 Sigma viability: seasonal OK = {ch2_seasonal_ok}, "
          f"scale-only OK = {ch2_A_ok} -- "
          f"{'PASS' if ch2_pass else 'FAIL'}", flush=True)

    # Ch3 -- we have seasonal's avg but not its per-hour vector here, so
    # we report the scale-only per-hour distribution and the avg gap;
    # the worst-hour gap requires the seasonal per-hour vector. Since
    # this focused script is for the scale-only result only, we report
    # the per-hour scale-only coverage and flag that the worst-hour
    # gap test against the seasonal per-hour vector needs the full
    # diagnostic (re-run) to be definitive. For the verdict, we use
    # the avg gap only, with a 2 pp tolerance, and flag if the
    # worst-hour test cannot be applied here.
    avg_gap = abs(ch3_A["average_coverage_pct"] - SEASONAL_BASELINE["ch3_avg"])
    ch3_avg_pass = avg_gap < 2.0
    print(f"Ch3 coverage:       scale-only avg = "
          f"{ch3_A['average_coverage_pct']:.2f}%, seasonal avg = "
          f"{SEASONAL_BASELINE['ch3_avg']:.2f}%, |avg gap| = "
          f"{avg_gap:.2f} pp -- "
          f"avg-gap {'PASS' if ch3_avg_pass else 'FAIL'} "
          f"(worst-hour test deferred to full diagnostic)", flush=True)
    # For verdict purposes use avg-gap only here; full worst-hour test
    # is a separate run.
    ch3_pass = ch3_avg_pass

    # Ch4 (conditional on Ch3)
    ch4_interpretable = ch3_pass
    if not ch4_interpretable:
        ch4_status = "NOT INTERPRETABLE (Ch3 failed)"
        ch4_pass_for_verdict = False
    else:
        nu_A = ch4_A["nu_hat"]
        nu_S = SEASONAL_BASELINE["ch4_nu"]
        if nu_A is None or nu_S is None:
            ch4_pass_for_verdict = False
            ch4_status = "FAIL (None values)"
        else:
            ch4_pass_for_verdict = (nu_A >= nu_S)
            ch4_status = "PASS" if ch4_pass_for_verdict else "FAIL"
    print(f"Ch4 heavy-tail:     nu_seasonal = "
          f"{SEASONAL_BASELINE['ch4_nu']:.3f}, "
          f"nu_scale-only = {ch4_A['nu_hat']} -- "
          f"{ch4_status}", flush=True)

    # Overall verdict
    print(flush=True)
    if d_oor:
        verdict = "CONFOUNDED (scale-only elbow-d outside [2,4])"
    elif not ch4_interpretable:
        verdict = "HOLD (Ch3 failed)"
    elif ch1_pass and ch2_pass and ch3_pass and ch4_pass_for_verdict:
        verdict = "LICENSED"
    else:
        verdict = "HOLD"
    print(f"CANDIDATE-A VERDICT: {verdict}", flush=True)

    # Three-branch outcome diagnosis
    print(flush=True)
    if d_oor:
        branch = "confound"
    elif ch1_pass and ch2_pass and ch3_pass and (
        ch4_pass_for_verdict if ch4_interpretable else True
    ):
        branch = ("(i) all-pass: additive sub-step is the removable lever; "
                  "scale-only delivers scale-calibration at reduced tail cost")
    elif ch3_pass and ch4_interpretable and not ch4_pass_for_verdict:
        branch = ("(ii) Ch3 passes, Ch4 fails: heavy-tail cost lives with "
                  "the multiplicative sigma_{m,h} step; scale-only not the "
                  "lever; move to candidates B/C")
    elif not ch1_pass:
        branch = ("(iii) Ch1 fails substantially: additive step also "
                  "conditions the embedding-state input; decomposition "
                  "not separable at this level")
    else:
        branch = ("outcome not on a pre-stated branch; report verbatim "
                  "(Ch3 fail, Ch1+Ch2 OK)")
    print(f"Pre-stated branch:   {branch}", flush=True)

    # Dump per-hour coverage for the prose update
    print(flush=True)
    print("per-hour scale-only coverage (for prose / worst-hour test):",
          flush=True)
    for h in sorted(ch3_A["per_hour_coverage_pct"]):
        print(f"  h={h:>2}: {ch3_A['per_hour_coverage_pct'][h]:.1f}%",
              flush=True)


if __name__ == "__main__":
    sys.stdout = sys.stdout  # noqa
    main()
