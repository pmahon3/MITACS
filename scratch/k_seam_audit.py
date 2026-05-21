"""K/S seam audit: re-score K and S arms with first-of-month delivery
days excluded, to test whether their dB signals survive seam exclusion.

Findings (output of this script, recorded in commit message of the
qualifier_exhaustion_memo §7 synthesis):

  baseline arm h=1 excl-1st: dB = -0.008 [CI -0.066, +0.045]
    -> the K writeup's "h=1 catastrophe channel" was the seam

  phi-state h=1 excl-1st: dB = +0.028 [CI -0.003, +0.052]
    -> the S writeup's "S obsoletes K (dB drops 16x)" was also
       seam-driven; the apparent dB drop dissolves on routine days

  baseline POOLED (all h) excl-1st: dB = +0.018 [CI +0.005, +0.031]
    -> pooled K verdict ROBUST to seam exclusion (lenient PASS)

  baseline POOLED excl-1st AND excl-h=1: dB = +0.018 [CI +0.004, +0.031]
    -> pooled signal does NOT depend on h=1; lives at h=3-9

  Per-h excl-1st (in commit message): h=4 PASS strict; h=3 dB
    ratio degenerate (baseline LL near zero); h=8, 9 PASS lenient;
    all other h tied.

  Sigma_00 medians per arm:
    baseline = 0.0154
    phi      = 0.0287 (~2x baseline, UNIFORM across anchors,
                       not seam-responsive)
    lag      = 0.0284 (same as phi)

  The sigma-doubling under state augmentation is uniform across all
  anchors (not concentrated at seam-near ones). The original S
  writeup's "sigma-inflation tames catastrophe events" mechanism
  story is RETRACTED: the doubling is a global property of the
  higher-dim WLS, not a seam-responsive property. Cause unidentified
  at this scope (candidates: curse-of-dim local-residual variance;
  worse-on-average augmented fit; LOO-CV bandwidth response).

PROVENANCE-GRADE: INSPECTION-ONLY.
"""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t

NU_MIN = 4.5
B = 1000
RNG = np.random.default_rng(20260520)

K_CACHE = Path("/tmp/k_loo_dB.pkl")
K_FORECASTS = Path("/tmp/bench2122_fc_propagated_augmented.pkl")
PHI_RESID = Path("/tmp/s_resid_phi.pkl")
LAG_RESID = Path("/tmp/s_resid_lag.pkl")


def _dt_of(d):
    wd = pd.Timestamp(d).weekday()
    return {5: "saturday", 6: "sunday"}.get(wd, "weekday")


def _moms_nu(z_std):
    z = z_std[np.isfinite(z_std)]
    if len(z) < 4:
        return 1000.0
    k = pd.Series(z).kurtosis()
    if not np.isfinite(k) or k <= 0:
        return 1000.0
    return max(NU_MIN, 6.0 / k + 4.0)


def loo_dB(rows):
    rows = rows.dropna(subset=["z_actual", "z_pred", "sigma00"]).copy().reset_index(drop=True)
    rows["z_std"] = (rows["z_actual"] - rows["z_pred"]) / np.sqrt(np.maximum(rows["sigma00"], 1e-9))
    days = sorted(rows["day"].unique())
    n = len(rows)
    ll_g = np.empty(n)
    ll_t = np.empty(n)
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
    # === Per-arm h=1 dB, all-days vs excl-1st-of-month ===
    print("=" * 80)
    print("Per-arm h=1 dB: full vs excl-first-of-month")
    print("=" * 80)
    with K_FORECASTS.open("rb") as f:
        fc = pickle.load(f)
    h1 = fc[fc["horizon_h"] == 1][["delivery_date", "z_pred", "z_actual", "s2_k"]].copy()
    h1.columns = ["day", "z_pred", "z_actual", "sigma00"]
    h1["daytype"] = h1["day"].apply(_dt_of)
    arms = {"baseline": h1.reset_index(drop=True)}
    for arm in ["phi", "lag"]:
        with open(f"/tmp/s_resid_{arm}.pkl", "rb") as f:
            arms[arm] = pickle.load(f)["rows"]

    for name, df in arms.items():
        df = df.dropna(subset=["z_actual", "z_pred", "sigma00"]).copy()
        df["is_first"] = pd.to_datetime(df["day"]).dt.day == 1
        dB_a, lo_a, hi_a = loo_dB(df)
        dB_e, lo_e, hi_e = loo_dB(df[~df["is_first"]])
        sig_med = df["sigma00"].median()
        print(f"  {name:>10s}: all dB={dB_a:>+7.4f} [{lo_a:>+7.4f}, {hi_a:>+7.4f}]"
              f"  excl-1st dB={dB_e:>+7.4f} [{lo_e:>+7.4f}, {hi_e:>+7.4f}]"
              f"  sigma_00 med={sig_med:.5f}")

    # === Per-horizon excl-1st dB on the K cache ===
    print()
    print("=" * 80)
    print("Per-horizon excl-1st-of-month dB on K cache (n=128 days)")
    print("=" * 80)
    with K_CACHE.open("rb") as f:
        blob = pickle.load(f)
    raw = blob["raw"].copy()
    raw["is_first"] = pd.to_datetime(raw["day"]).dt.day == 1
    excl = raw[~raw["is_first"]].dropna(
        subset=["ll_gauss", "ll_t"]
    ).reset_index(drop=True)
    print(f"{'h':>3} {'n':>5} {'baseline LL':>13} {'dB(h)':>10} "
          f"{'CI lo':>10} {'CI hi':>10}  verdict")
    for h, g in excl.groupby("h"):
        g = g.reset_index(drop=True)
        if len(g) < 30:
            continue
        ll_g = g["ll_gauss"].to_numpy()
        ll_t = g["ll_t"].to_numpy()
        days = g["day"].to_numpy()
        uniq = np.unique(days)
        by_d = {d: np.where(days == d)[0] for d in uniq}
        boots = np.empty(B)
        for b in range(B):
            samp = RNG.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([by_d[d] for d in samp])
            base = ll_g[idx].mean()
            tval = ll_t[idx].mean()
            boots[b] = (base - tval) / max(abs(base), 1e-9)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        pt = (ll_g.mean() - ll_t.mean()) / max(abs(ll_g.mean()), 1e-9)
        if lo > 0.01:
            v = "PASS strict"
        elif lo > 0:
            v = "PASS lenient"
        elif hi < 0:
            v = "K LOSES"
        else:
            v = "tied"
        print(f"{int(h):>3} {len(g):>5} {ll_g.mean():>+13.4f} "
              f"{pt:>+10.4f} {lo:>+10.4f} {hi:>+10.4f}  {v}")

    # === Pooled diagnostics: all-h, all-h excl-h1, ===
    print()
    days_all = sorted(excl["day"].unique())
    by_d = {d: excl.index[excl["day"] == d].to_numpy() for d in days_all}
    boots = np.empty(B)
    for b in range(B):
        samp = RNG.choice(days_all, size=len(days_all), replace=True)
        idx = np.concatenate([by_d[d] for d in samp])
        boots[b] = (excl.loc[idx, "ll_gauss"].mean()
                    - excl.loc[idx, "ll_t"].mean()) / max(
                        abs(excl.loc[idx, "ll_gauss"].mean()), 1e-9)
    lo_p, hi_p = np.percentile(boots, [2.5, 97.5])
    pt_p = ((excl["ll_gauss"].mean() - excl["ll_t"].mean())
            / max(abs(excl["ll_gauss"].mean()), 1e-9))
    print(f"POOLED excl-1st:                dB = {pt_p:+.4f}  CI=[{lo_p:+.4f}, {hi_p:+.4f}]")

    excl_no_h1 = excl[excl["h"] >= 2].reset_index(drop=True)
    days_b = sorted(excl_no_h1["day"].unique())
    by_b = {d: excl_no_h1.index[excl_no_h1["day"] == d].to_numpy() for d in days_b}
    boots = np.empty(B)
    for b in range(B):
        samp = RNG.choice(days_b, size=len(days_b), replace=True)
        idx = np.concatenate([by_b[d] for d in samp])
        boots[b] = (excl_no_h1.loc[idx, "ll_gauss"].mean()
                    - excl_no_h1.loc[idx, "ll_t"].mean()) / max(
                        abs(excl_no_h1.loc[idx, "ll_gauss"].mean()), 1e-9)
    lo_q, hi_q = np.percentile(boots, [2.5, 97.5])
    pt_q = ((excl_no_h1["ll_gauss"].mean() - excl_no_h1["ll_t"].mean())
            / max(abs(excl_no_h1["ll_gauss"].mean()), 1e-9))
    print(f"POOLED excl-1st AND excl-h=1:   dB = {pt_q:+.4f}  CI=[{lo_q:+.4f}, {hi_q:+.4f}]")
    print()
    print("Reading: pooled signal lives at h=3-9 (mid-horizon); h=1 contribution")
    print("under seam exclusion is null. K's PASS verdict is mid-horizon, not h=1.")


if __name__ == "__main__":
    main()
