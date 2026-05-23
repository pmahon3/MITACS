"""EXPLORATORY: data inspection on the h=24 flip in the multi-step backtest.

Wide-bandwidth predictors (simplex-theta, fixed theta>=4, global-OLS) all
show +12 to +26 MW dMAE at h=24 vs production -- a sign flip and growing
magnitude that contradicts the smooth-fade-with-bandwidth behaviour of
all other horizons. Theory-spinning so far has been glib. This probe is
the data inspection that should precede any explanation:

  Q1  Is h=24 on the same delivery-day population as h=1..23, or has the
      multi-step loop's KeyError-break dropped days specifically there?
  Q2  Is production's MAE genuinely lower at h=24 than h=23 (i.e. is the
      anomaly in production helping, not in wide-theta hurting)?
  Q3  Is there something specific about 23:00 actuals -- demand level,
      variance, etc. -- that produces a signal-to-noise floor that
      narrower kernels happen to handle better?
  Q4  What do per-day h=23 and h=24 forecast triples (production,
      global-OLS, actual) actually look like?

PROVENANCE-GRADE: INSPECTION-ONLY.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiment._actuals import load_actuals
from config import load_config
from experiment.backtest import _complete_delivery_days
from scratch.backtest_simplex_theta import _run


def main():
    actual = load_actuals(cutoff=None)
    cutoff = pd.Timestamp("2024-12-31T23:00:00")
    anchor_h = load_config().data.day_anchor_hours
    days = _complete_delivery_days(actual, anchor_h)
    days = days[days > cutoff]

    print("=" * 70)
    print("h=24 FLIP DATA INSPECTION  (INSPECTION-ONLY)")
    print(f"post-cutoff delivery days: {len(days)}")
    print("=" * 70)

    print("\nrunning production + global-OLS backtests on full span ...",
          flush=True)
    fc_prod = _run(days, mode="production")
    fc_glob = _run(days, mode="global")

    # join in actuals for both
    for fc in (fc_prod, fc_glob):
        fc["actual_mw"] = fc["target_dt"].map(actual)
        fc["err"] = fc["our_forecast_mw"] - fc["actual_mw"]

    # ----------- Q1: sample composition per horizon ---------------------
    print("\n--- Q1: rows per horizon (drop-outs would show up here) ---")
    print(f"  {'h':>3} {'prod_n':>7} {'glob_n':>7}")
    for h in range(1, 25):
        np_ = int((fc_prod['horizon_h'] == h).sum())
        ng_ = int((fc_glob['horizon_h'] == h).sum())
        if np_ != len(days) or ng_ != len(days):
            tag = "  <-- DROPS"
        else:
            tag = ""
        print(f"  {h:>3} {np_:>7} {ng_:>7}{tag}")

    # ----------- Q2: per-horizon production / global MAE ----------------
    print("\n--- Q2: per-horizon MAE -- is production MAE lower at h=24? ---")
    print(f"  {'h':>3} {'prod_MAE':>9} {'glob_MAE':>9} "
          f"{'dMAE':>8}  {'prod_signedERR_mean':>20}")
    for h in range(20, 25):
        g_prod = fc_prod[fc_prod["horizon_h"] == h].dropna(subset=["actual_mw"])
        g_glob = fc_glob[fc_glob["horizon_h"] == h].dropna(subset=["actual_mw"])
        mae_p = g_prod["err"].abs().mean()
        mae_g = g_glob["err"].abs().mean()
        signed_p = g_prod["err"].mean()
        print(f"  {h:>3} {mae_p:>9.1f} {mae_g:>9.1f} "
              f"{mae_g - mae_p:>+8.1f}  {signed_p:>+20.1f}")

    # ----------- Q3: per-hour-of-day actual statistics ------------------
    print("\n--- Q3: actual demand statistics, h=20..h=24 "
          "(does h=24 sit in a unique demand regime?) ---")
    print(f"  {'h':>3} {'mean':>8} {'median':>8} {'std':>8} "
          f"{'min':>8} {'max':>8}")
    for h in range(20, 25):
        g = fc_prod[fc_prod["horizon_h"] == h].dropna(subset=["actual_mw"])
        a = g["actual_mw"]
        print(f"  {h:>3} {a.mean():>8.0f} {a.median():>8.0f} {a.std():>8.0f} "
              f"{a.min():>8.0f} {a.max():>8.0f}")

    # ----------- Q4: per-day h=23 and h=24 forecast triples -------------
    print("\n--- Q4: first 8 delivery days, h=23 and h=24 forecast triples ---")
    print(f"  {'delivery_date':>14}  "
          f"{'h':>2}  {'actual':>7}  {'prod':>7}  {'glob':>7}  "
          f"{'prod_err':>8}  {'glob_err':>8}")
    for D in days[:8]:
        for h in (23, 24):
            t = D + pd.Timedelta(hours=h - 1)
            rp = fc_prod[(fc_prod["delivery_date"] == D) &
                         (fc_prod["horizon_h"] == h)]
            rg = fc_glob[(fc_glob["delivery_date"] == D) &
                         (fc_glob["horizon_h"] == h)]
            if len(rp) == 0 or len(rg) == 0:
                print(f"  {str(D.date()):>14}  {h:>2}  (missing)")
                continue
            a = rp["actual_mw"].iloc[0]
            pf = rp["our_forecast_mw"].iloc[0]
            gf = rg["our_forecast_mw"].iloc[0]
            print(f"  {str(D.date()):>14}  {h:>2}  "
                  f"{a:>7.0f}  {pf:>7.0f}  {gf:>7.0f}  "
                  f"{pf - a:>+8.0f}  {gf - a:>+8.0f}")

    # ----------- Q4b: aggregate signed errors, h=23 vs h=24 -------------
    print("\n--- Q4b: aggregate signed errors h=23 vs h=24 (bias check) ---")
    for label, fc in (("production", fc_prod), ("global-OLS", fc_glob)):
        for h in (23, 24):
            g = fc[fc["horizon_h"] == h].dropna(subset=["actual_mw"])
            print(f"  {label:>10} h={h}  signed-err "
                  f"mean={g['err'].mean():>+7.1f}  "
                  f"median={g['err'].median():>+7.1f}  "
                  f"abs-mean={g['err'].abs().mean():>7.1f}  n={len(g)}")


if __name__ == "__main__":
    main()
