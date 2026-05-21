"""Head-to-head comparison: K analyses under (month, hour) vs Fourier
preprocessing. Loads both capture pickles and reports:

  (1) Forecast accuracy in DEMAND-MW space (the goal-relevant metric).
      For each preprocessing, destandardise z_pred and z_actual back
      to demand MW using the respective climatology; compute MAE
      per day, focusing on first-of-month days (the days the seam
      affects under month-hour).
  (2) Residual kurt under each preprocessing (the kappa_Q gate from
      the qualifier-exhaustion programme).
  (3) Pooled K dB (Student-t vs Gaussian) under each preprocessing,
      with day-bootstrap CIs.

If Fourier delivers (a) lower MAE on first-of-month days specifically,
and (b) comparable-or-tighter mid-horizon K dB, then the A path
empirically confirms the diagnostic prediction.

PROVENANCE-GRADE: INSPECTION-ONLY.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t

from experiment._actuals import load_actuals, zscore_params, zscore_transform
from scratch.benchmark_2021_22_ieso import CUTOFF
from scratch.fourier_climatology import fit_fourier_params, fourier_destandardise

MH_CACHE = Path("/tmp/k_step_sequences.pkl")
FR_CACHE = Path("/tmp/k_step_sequences_fourier.pkl")

NU_MIN = 4.5
B = 1000
RNG = np.random.default_rng(20260520)


def _moms_nu(z_std):
    z = z_std[np.isfinite(z_std)]
    if len(z) < 4:
        return 1000.0
    k = pd.Series(z).kurtosis()
    if not np.isfinite(k) or k <= 0:
        return 1000.0
    return max(NU_MIN, 6.0 / k + 4.0)


def _kurt(x):
    x = x[np.isfinite(x)]
    if len(x) < 4:
        return float("nan")
    return float(pd.Series(x).kurtosis())


def _ci_kurt(x, days, B=B):
    uniq = np.unique(days)
    by = {d: np.where(days == d)[0] for d in uniq}
    boots = np.empty(B)
    for b in range(B):
        samp = RNG.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by[d] for d in samp])
        boots[b] = _kurt(x[idx])
    boots = boots[np.isfinite(boots)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return _kurt(x), float(lo), float(hi)


def load_h1_residuals_demand_mw(cache_path, preprocessing_label,
                                act, mh_params, fr_params):
    """Load h=1 rows from a capture, destandardise back to MW, return
    a DataFrame with raw_pred, raw_actual, MAE_mw columns."""
    with cache_path.open("rb") as f:
        blob = pickle.load(f)
    captures = blob["captures"]
    rows = []
    for day, day_rows in captures.items():
        r1 = day_rows[0]    # h=1
        z_pred = r1["z_pred"]
        z_actual = r1["z_actual"]
        ts = r1["dt"]
        if not (np.isfinite(z_pred) and np.isfinite(z_actual)):
            continue
        if preprocessing_label == "month-hour":
            key = (ts.month, ts.hour)
            mu = float(mh_params["mu_mh"].loc[key])
            sig = float(mh_params["sigma_mh"].loc[key])
            d_pred = z_pred * sig + mu
            d_actual = z_actual * sig + mu
        else:
            d_pred = fourier_destandardise(z_pred, ts, fr_params)
            d_actual = fourier_destandardise(z_actual, ts, fr_params)
        rows.append({
            "day": pd.Timestamp(day),
            "daytype": r1["daytype"],
            "z_pred": z_pred,
            "z_actual": z_actual,
            "sigma00": r1["Sigma_00"],
            "demand_pred_mw": d_pred,
            "demand_actual_mw": d_actual,
            "demand_err_mw": d_actual - d_pred,
        })
    return pd.DataFrame(rows)


def loo_dB(rows):
    """Pooled h=1 dB(Student-t vs Gauss) with LOO MoM nu and
    day-bootstrap CI."""
    rows = rows.dropna(subset=["z_actual", "z_pred", "sigma00"]).copy().reset_index(drop=True)
    rows["z_std"] = (rows["z_actual"] - rows["z_pred"]) / np.sqrt(np.maximum(rows["sigma00"], 1e-9))
    days = sorted(rows["day"].unique())
    n = len(rows)
    ll_g, ll_t = np.empty(n), np.empty(n)
    by_day = {d: rows.index[rows["day"] == d].to_numpy() for d in days}
    for held in days:
        train = rows[rows["day"] != held]
        nu_by_dt = {dt: _moms_nu(g["z_std"].to_numpy())
                    for dt, g in train.groupby("daytype")}
        for i in by_day[held]:
            r = rows.loc[i]
            y, mu = float(r["z_actual"]), float(r["z_pred"])
            var = max(float(r["sigma00"]), 1e-9)
            sd = np.sqrt(var)
            ll_g[i] = -norm.logpdf(y, loc=mu, scale=sd)
            nu = nu_by_dt.get(r["daytype"], 1000.0)
            ts = np.sqrt(var * (nu - 2.0) / nu)
            ll_t[i] = -(student_t.logpdf((y - mu) / ts, df=nu) - np.log(ts))
    days_arr = rows["day"].to_numpy()
    by_d = {d: np.where(days_arr == d)[0] for d in days}
    boots = np.empty(B)
    for b in range(B):
        samp = RNG.choice(days, size=len(days), replace=True)
        idx = np.concatenate([by_d[d] for d in samp])
        boots[b] = (ll_g[idx].mean() - ll_t[idx].mean()) / max(abs(ll_g[idx].mean()), 1e-9)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    pt = (ll_g.mean() - ll_t.mean()) / max(abs(ll_g.mean()), 1e-9)
    return pt, lo, hi


def main():
    print("=" * 88)
    print("A-vs-B head-to-head: K analyses under Fourier vs (month, hour) preprocessing")
    print("=" * 88)

    if not MH_CACHE.exists() or not FR_CACHE.exists():
        print(f"\nMissing cache(s):")
        print(f"  MH:  {MH_CACHE.exists()}  ({MH_CACHE})")
        print(f"  FR:  {FR_CACHE.exists()}  ({FR_CACHE})")
        return

    act = load_actuals(cutoff=None).asfreq("h")
    mh_params = zscore_params(CUTOFF)
    fr_params = fit_fourier_params(act, CUTOFF)

    mh = load_h1_residuals_demand_mw(MH_CACHE, "month-hour", act, mh_params, fr_params)
    fr = load_h1_residuals_demand_mw(FR_CACHE, "fourier",    act, mh_params, fr_params)
    print(f"\nMH n_days: {len(mh)}  |  FR n_days: {len(fr)}")

    # === (1) Demand-MW forecast accuracy ===
    print()
    print("=" * 88)
    print("(1) h=1 forecast accuracy in DEMAND-MW (the goal metric)")
    print("=" * 88)

    # Align on common days
    common = sorted(set(mh["day"]) & set(fr["day"]))
    mh_c = mh[mh["day"].isin(common)].set_index("day").sort_index()
    fr_c = fr[fr["day"].isin(common)].set_index("day").sort_index()

    print(f"  {'subset':>20s} {'n':>4s} {'MAE_mh':>10s} {'MAE_fr':>10s} {'Δ MAE':>10s}  "
          f"{'mean err_mh':>13s} {'mean err_fr':>13s}")
    for label, mask in [
        ("all days",          mh_c.index == mh_c.index),
        ("first-of-month",    mh_c.index.day == 1),
        ("non-first-of-month",mh_c.index.day != 1),
    ]:
        mh_sub = mh_c[mask]
        fr_sub = fr_c[mask]
        mae_mh = float(np.abs(mh_sub["demand_err_mw"]).mean())
        mae_fr = float(np.abs(fr_sub["demand_err_mw"]).mean())
        me_mh = float(mh_sub["demand_err_mw"].mean())
        me_fr = float(fr_sub["demand_err_mw"].mean())
        print(f"  {label:>20s} {len(mh_sub):>4d} {mae_mh:>10.1f} {mae_fr:>10.1f} "
              f"{mae_fr - mae_mh:>+10.1f}  {me_mh:>+13.1f} {me_fr:>+13.1f}")

    # First-of-month detail
    fom = mh_c.index.day == 1
    if fom.sum() > 0:
        print()
        print("  First-of-month detail (the days the seam affects under month-hour):")
        print(f"    {'day':>12s} {'daytype':>10s} "
              f"{'MH err MW':>11s} {'FR err MW':>11s} {'Δ':>9s}")
        for d in mh_c.index[fom]:
            r_mh = mh_c.loc[d]
            r_fr = fr_c.loc[d]
            err_mh = float(r_mh["demand_err_mw"])
            err_fr = float(r_fr["demand_err_mw"])
            print(f"    {str(d.date()):>12s} {r_mh['daytype']:>10s} "
                  f"{err_mh:>+11.1f} {err_fr:>+11.1f} {err_fr - err_mh:>+9.1f}")

    # === (2) Residual kurt ===
    print()
    print("=" * 88)
    print("(2) h=1 standardised-residual kurt (the kappa_Q gate)")
    print("=" * 88)
    for label, df in [("month-hour", mh), ("fourier", fr)]:
        df = df.dropna(subset=["z_pred", "z_actual", "sigma00"]).copy()
        df["z_std"] = (df["z_actual"] - df["z_pred"]) / np.sqrt(np.maximum(df["sigma00"], 1e-9))
        days_arr = df["day"].to_numpy()
        k_all, lo_all, hi_all = _ci_kurt(df["z_std"].to_numpy(), days_arr)
        # excl-first-of-month
        df_excl = df[df["day"].dt.day != 1]
        k_excl, lo_excl, hi_excl = _ci_kurt(
            df_excl["z_std"].to_numpy(), df_excl["day"].to_numpy()
        )
        print(f"  {label:>10s}: all-days kurt={k_all:>7.3f} [{lo_all:>+7.3f}, {hi_all:>+7.3f}]"
              f"  excl-1st kurt={k_excl:>7.3f} [{lo_excl:>+7.3f}, {hi_excl:>+7.3f}]")

    # === (3) Pooled K dB ===
    print()
    print("=" * 88)
    print("(3) h=1 pooled K dB(Student-t vs Gauss), LOO + per-daytype MoM nu")
    print("=" * 88)
    for label, df in [("month-hour", mh), ("fourier", fr)]:
        for arm_label, sub in [("all-days", df), ("excl-1st", df[df["day"].dt.day != 1])]:
            pt, lo, hi = loo_dB(sub)
            print(f"  {label:>10s} {arm_label:>10s}: dB={pt:>+7.4f} [{lo:>+7.4f}, {hi:>+7.4f}]")


if __name__ == "__main__":
    main()
