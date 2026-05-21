"""Option S — score the phi-augmented and lag-only-enriched residuals
captured by scratch/s_capture.py against the baseline residual from
the augmented-scope cache /tmp/bench2122_fc_propagated_augmented.pkl
(used elsewhere in the qualifier_exhaustion memo as the reference
under the elbow-selected pure-lag baseline state).

For each arm (baseline / phi-aug / lag-enriched) and each day-type:
  * h=1 standardised residual z_std = (z_actual - z_pred) / sqrt(sigma00)
  * Fisher excess kurtosis (Gaussian = 0)
  * Day-bootstrap 95% CI on the kurtosis
  * Pooled dB(K) under the SAME LOO + per-day-type MoM + closed-form
    h=1 protocol as scratch/k_loo_score.py — i.e. compute the
    log-loss reduction if you also ran a Student-t kernel on TOP of
    this state's residual.

Three pre-stated outcomes (qualifier_exhaustion_memo §6, kurt
thresholds justified by kurt → ν → dB analytic relation):
  S-i  : pooled kurt_phi < 3 (and likely cleanly below baseline ~8.6)
         → augmented state alone makes Gaussian local kernel adequate;
         K is unnecessary on this state. Implied dB(K | S) << 1%.
  S-ii : 3 ≤ pooled kurt_phi < 8 → moderate reduction; S+K is the
         natural combined design.
  S-iii: pooled kurt_phi ≥ 8 → no material reduction; heavy tail is
         intrinsic to the per-step conditional law; K is the only
         available lever (Stage-1 Register lever is non-load-bearing
         for the kernel-form problem).

Confound control: kurt_lag (enrich-d at d+1 with no phi coord) vs
kurt_phi. If kurt_lag ≈ kurt_phi, "more state alone" is the
mechanism, not the phi coord specifically. If kurt_phi cleanly
beats kurt_lag, the qualitatively new coord is doing the work.

PROVENANCE-GRADE: INSPECTION-ONLY.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t

BASELINE_CACHE = Path("/tmp/bench2122_fc_propagated_augmented.pkl")
PHI_PATH = Path("/tmp/s_resid_phi.pkl")
LAG_PATH = Path("/tmp/s_resid_lag.pkl")
B_BOOT = 1000
TAU_MIN = 0.01
NU_MIN = 4.5
RNG = np.random.default_rng(20260520)


def _kurt(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 4:
        return float("nan")
    return float(pd.Series(x).kurtosis())


def _ci_kurt(x: np.ndarray, days: np.ndarray, B: int = B_BOOT
             ) -> tuple[float, float, float]:
    """Day-bootstrap 95% CI on Fisher excess kurtosis."""
    uniq = np.unique(days)
    by_day = {d: np.where(days == d)[0] for d in uniq}
    boots = np.empty(B)
    for b in range(B):
        samp = RNG.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by_day[d] for d in samp])
        boots[b] = _kurt(x[idx])
    boots = boots[np.isfinite(boots)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(_kurt(x)), float(lo), float(hi)


def _moms_nu(z_std: np.ndarray) -> float:
    """Per-day-type MoM nu_hat = max(NU_MIN, 6/kurt + 4)."""
    k = _kurt(z_std)
    if not np.isfinite(k) or k <= 0:
        return 1000.0     # effectively Gaussian
    return max(NU_MIN, 6.0 / k + 4.0)


def _loo_dB_h1(rows: pd.DataFrame) -> tuple[float, float, float, dict]:
    """LOO + per-day-type MoM nu_hat + closed-form h=1 density.
    Returns (pooled_dB_point, ci_lo, ci_hi, per_dt_nu)."""
    rows = rows.dropna(subset=["z_actual", "z_pred", "sigma00"]).copy()
    rows["z_std"] = ((rows["z_actual"] - rows["z_pred"])
                     / np.sqrt(np.maximum(rows["sigma00"], 1e-9)))
    days = sorted(rows["day"].unique())
    ll_gauss = np.empty(len(rows))
    ll_t = np.empty(len(rows))
    nu_hat_used = {}      # last-seen LOO nu per day-type, for reporting
    row_idx = {d: rows.index[rows["day"] == d].to_numpy() for d in days}

    for held in days:
        train = rows[rows["day"] != held]
        nu_by_dt = {}
        for dt, g in train.groupby("daytype"):
            nu_by_dt[dt] = _moms_nu(g["z_std"].to_numpy())
        nu_hat_used = nu_by_dt
        # score the held-out day's rows
        for i in row_idx[held]:
            r = rows.loc[i]
            y, mu = float(r["z_actual"]), float(r["z_pred"])
            var = max(float(r["sigma00"]), 1e-9)
            sd = float(np.sqrt(var))
            ll_gauss[rows.index.get_loc(i)] = float(
                -norm.logpdf(y, loc=mu, scale=sd)
            )
            nu = nu_by_dt.get(r["daytype"], 1000.0)
            t_scale = float(np.sqrt(var * (nu - 2.0) / nu))
            ll_t[rows.index.get_loc(i)] = float(
                -(student_t.logpdf((y - mu) / t_scale, df=nu)
                  - np.log(t_scale))
            )

    days_arr = rows["day"].to_numpy()
    by_day = {d: np.where(days_arr == d)[0] for d in days}
    boots = np.empty(B_BOOT)
    for b in range(B_BOOT):
        samp = RNG.choice(days, size=len(days), replace=True)
        idx = np.concatenate([by_day[d] for d in samp])
        g_mean = ll_gauss[idx].mean()
        t_mean = ll_t[idx].mean()
        boots[b] = (g_mean - t_mean) / max(abs(g_mean), 1e-9)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    point = (ll_gauss.mean() - ll_t.mean()) / max(abs(ll_gauss.mean()), 1e-9)
    return float(point), float(lo), float(hi), nu_hat_used


def _summarise_arm(name: str, rows: pd.DataFrame) -> dict:
    rows = rows.dropna(subset=["z_actual", "z_pred", "sigma00"]).copy()
    rows["z_std"] = ((rows["z_actual"] - rows["z_pred"])
                     / np.sqrt(np.maximum(rows["sigma00"], 1e-9)))

    print(f"\n=== arm: {name} (n={len(rows)}) ===")
    # Pooled
    z = rows["z_std"].to_numpy()
    d = rows["day"].to_numpy()
    pt, lo, hi = _ci_kurt(z, d)
    print(f"  POOLED  kurt = {pt:7.3f}  CI=[{lo:7.3f}, {hi:7.3f}]")

    # Per day-type
    for dt in ("weekday", "saturday", "sunday"):
        sub = rows[rows["daytype"] == dt]
        if len(sub) < 30:
            print(f"  {dt:>8}: n={len(sub)} (skipped, n<30 — qualitative)")
            qual_k = _kurt(sub["z_std"].to_numpy())
            print(f"              kurt (point only) = {qual_k:7.3f}")
            continue
        zd = sub["z_std"].to_numpy()
        dd = sub["day"].to_numpy()
        pt_dt, lo_dt, hi_dt = _ci_kurt(zd, dd)
        print(f"  {dt:>8}: kurt = {pt_dt:7.3f}  "
              f"CI=[{lo_dt:7.3f}, {hi_dt:7.3f}]  n={len(sub)}")

    # Pooled dB(K) under this state
    print(f"  LOO dB(K | this state) ...")
    dB_pt, dB_lo, dB_hi, nu_used = _loo_dB_h1(rows)
    print(f"  POOLED  dB(K) = {dB_pt:7.4f}  "
          f"CI=[{dB_lo:7.4f}, {dB_hi:7.4f}]  "
          f"vs tau_min={TAU_MIN}")
    print(f"  (LOO nu_hat per daytype, last fold: {nu_used})")
    return {
        "name": name, "n": len(rows),
        "pooled_kurt": pt, "pooled_kurt_ci": (lo, hi),
        "pooled_dB_K": dB_pt, "pooled_dB_K_ci": (dB_lo, dB_hi),
        "nu_hat_lastfold": nu_used,
    }


def _baseline_from_cache() -> pd.DataFrame:
    """Read the augmented-scope cache's h=1 rows in the same schema as
    the S captures: day, daytype, z_pred, z_actual, sigma00."""
    with BASELINE_CACHE.open("rb") as f:
        fc = pickle.load(f)
    fc = fc.copy()
    fc["day"] = fc["delivery_date"]
    h1 = fc[fc["horizon_h"] == 1][["day", "z_pred", "z_actual"]].copy()
    # need sigma00 == Sigma_aug[0, 0] for h=1 — same as s2_k at h=1
    # under the augmented propagation since P starts at zero:
    # P_1 = J^T P_0 J + Sigma = Sigma, so s2_k[h=1] = Sigma[0,0]
    h1["sigma00"] = fc[fc["horizon_h"] == 1]["s2_k"].values
    h1["daytype"] = h1["day"].apply(
        lambda d: ["weekday", "weekday", "weekday", "weekday", "weekday",
                   "saturday", "sunday"][pd.Timestamp(d).weekday()]
    )
    return h1


def main() -> None:
    print("=" * 76)
    print("Option S verdict: phi-augmented vs lag-enriched vs baseline")
    print("=" * 76)

    bsl = _baseline_from_cache()
    base_sum = _summarise_arm("baseline (cache, h=1)", bsl)

    with PHI_PATH.open("rb") as f:
        phi_blob = pickle.load(f)
    with LAG_PATH.open("rb") as f:
        lag_blob = pickle.load(f)

    phi_sum = _summarise_arm("phi-aug (z, lags, tanh(dz/s))",
                             phi_blob["rows"])
    lag_sum = _summarise_arm("lag-enriched (d+1 lags, no phi)",
                             lag_blob["rows"])

    # Verdict mapping
    print()
    print("=" * 76)
    print("S verdict per pre-stated outcomes:")
    print("=" * 76)
    kp = phi_sum["pooled_kurt"]
    if kp < 3:
        verdict = ("S-i: pooled kurt_phi < 3 → augmented state alone "
                   "makes Gaussian adequate; K is unnecessary on this "
                   "state. Stage-2 K lever is OBSOLETED by the Register "
                   "lever for THIS specific augmentation.")
    elif kp < 8:
        verdict = ("S-ii: 3 ≤ pooled kurt_phi < 8 → moderate reduction; "
                   "S+K is the natural combined design. Run #52 next.")
    else:
        verdict = ("S-iii: pooled kurt_phi ≥ 8 → no material reduction; "
                   "heavy tail is intrinsic to the per-step conditional "
                   "law. Register lever (this phi-augmentation) is not "
                   "load-bearing for the kernel-form problem. K (Stage 2) "
                   "remains the only available lever.")
    print(f"  Pooled kurt: baseline={base_sum['pooled_kurt']:.2f}, "
          f"phi-aug={phi_sum['pooled_kurt']:.2f}, "
          f"lag-enriched={lag_sum['pooled_kurt']:.2f}")
    print(f"  Verdict: {verdict}")
    print()
    # Confound check
    if abs(phi_sum["pooled_kurt"] - lag_sum["pooled_kurt"]) < 1.0:
        print("  Confound CONTROL: kurt_phi ≈ kurt_lag — the phi "
              "coord is NOT contributing; whatever reduction occurred "
              "(if any) is from bigger state alone.")
    else:
        delta = lag_sum["pooled_kurt"] - phi_sum["pooled_kurt"]
        print(f"  Confound CONTROL: kurt_phi cleanly differs from "
              f"kurt_lag by {delta:+.2f} — the phi coord IS doing "
              f"distinguishable work.")

    # K-still-needed reading
    dB = phi_sum["pooled_dB_K"]
    dB_lo = phi_sum["pooled_dB_K_ci"][0]
    print()
    if dB_lo > TAU_MIN:
        print(f"  K-on-top reading: dB(K | phi-state) = {dB:.4f} "
              f"[CI lower {dB_lo:.4f} > tau_min] — K still adds value "
              f"on top of the augmented state; S+K is the natural design.")
    elif dB > 0 and dB_lo > 0:
        print(f"  K-on-top reading: dB(K | phi-state) = {dB:.4f} "
              f"[CI lower {dB_lo:.4f} > 0] — K adds modest value on "
              f"top of phi (lenient PASS, mirroring K-alone).")
    else:
        print(f"  K-on-top reading: dB(K | phi-state) = {dB:.4f} "
              f"[CI lower {dB_lo:.4f}] — K adds no statistical "
              f"value on top of phi; the phi-augmented state's "
              f"Gaussian kernel is adequate on its own.")


if __name__ == "__main__":
    main()
