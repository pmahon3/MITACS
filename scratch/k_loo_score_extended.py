"""Extended-window LOO + MC scoring with report-both detrend handling.

Per advisor (commit message of this script's parent thread):
  - Per-day-type drift on n=132 was unresolvable empirically
    (|slope|/SE = 0.5-1.2 across slabs; no day-type clears 2-SE).
  - At n=365 the slope SE only halves; drift still likely unresolvable
    at any conventional bar.
  - Pivot from 'measure drift, decide detrend' to 'run both arms,
    check robustness'. The DIFFERENCE between detrend-on vs
    detrend-off IS the measurement of drift's impact on the K/S
    verdict.

This script imports and reuses scratch.k_loo_score's machinery
(LOO + per-day-type MoM nu_hat + closed-form h=1 + MC h>=2) but
runs it TWICE against the extended cache:

  arm 'raw'      : no detrend (k_loo_score's existing logic verbatim)
  arm 'detrend'  : per-day-type linear residual detrend, fit on
                   THIS cache's h=1 residuals, applied to z_pred at
                   all horizons before density evaluation. The
                   detrend is a property of the SCORING, not the
                   capture — captures contain raw z_pred and the
                   detrend is layered on at scoring time.

Outputs:
  /tmp/k_loo_dB_extended_raw.pkl
  /tmp/k_loo_dB_extended_detrend.pkl

PROVENANCE-GRADE: INSPECTION-ONLY.
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, norm, t as student_t

IN_PATH = Path("/tmp/k_step_sequences_extended.pkl")
OUT_RAW = Path("/tmp/k_loo_dB_extended_raw.pkl")
OUT_DET = Path("/tmp/k_loo_dB_extended_detrend.pkl")
N_MC = 500
B_BOOTSTRAP = 1000
TAU_MIN = 0.01
NU_MIN = 4.5
KDE_BW = "scott"
RNG_BOOT = np.random.default_rng(20260521)


def _say(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


def _moms_nu(z_std: np.ndarray) -> float:
    z = z_std[np.isfinite(z_std)]
    if len(z) < 4:
        return NU_MIN
    k = float(pd.Series(z).kurtosis())
    return max(NU_MIN, 6.0 / max(k, 1e-6) + 4.0)


def _fit_nu_perdaytype(captures: dict, held_out: pd.Timestamp,
                       detrend_by_dt: dict | None) -> dict:
    """Per-day-type MoM nu_hat from the h=1 standardised residuals
    on the dev set EXCLUDING the held_out day. If detrend_by_dt is
    not None, the predictor z_pred is shifted by the per-day-type
    linear-in-days fit before standardised residual is computed.

    detrend_by_dt: {daytype: (slope_per_day, intercept,
                              days_origin_timestamp)}
    """
    by_dt: dict[str, list[float]] = {"weekday": [], "saturday": [],
                                     "sunday": []}
    for day, rows in captures.items():
        if day == held_out:
            continue
        r1 = rows[0]
        y = r1["z_actual"]
        mu = r1["z_pred"]
        if detrend_by_dt is not None:
            dt = r1["daytype"]
            if dt in detrend_by_dt:
                slope, intc, origin = detrend_by_dt[dt]
                d_since = (pd.Timestamp(day) - origin).days
                mu = mu + slope * d_since + intc
        sd = np.sqrt(max(r1["Sigma_00"], 1e-9))
        z_std = (y - mu) / sd
        if np.isfinite(z_std):
            by_dt[r1["daytype"]].append(z_std)
    return {dt: _moms_nu(np.array(vals)) for dt, vals in by_dt.items()
            if len(vals) >= 4}


def _scaled_t_draws(n: int, df: float, var: float, rng) -> np.ndarray:
    if df <= 2:
        df = 2.1
    scale = np.sqrt(var * (df - 2.0) / df)
    return scale * rng.standard_t(df, size=n)


def _gauss_draws(n: int, var: float, rng) -> np.ndarray:
    return np.sqrt(max(var, 1e-9)) * rng.standard_normal(n)


def _mc_propagate_day(day_rows, nu_dt, arm, rng):
    D_MAX = day_rows[0]["J_aug"].shape[0]
    x0 = day_rows[0]["x_state"]
    state = np.zeros((N_MC, D_MAX))
    state[:, :len(x0)] = x0[np.newaxis, :]
    out = np.empty((N_MC, 24))
    for h_idx, row in enumerate(day_rows):
        J = row["J_aug"]
        var = float(row["Sigma_aug"][0, 0])
        next_state = state @ J
        if arm == "gauss":
            inno = _gauss_draws(N_MC, var, rng)
        else:
            inno = _scaled_t_draws(N_MC, nu_dt, var, rng)
        next_state[:, 0] += inno
        state = next_state
        out[:, h_idx] = state[:, 0]
    return out


def _logdensity_at(samples, y):
    s = samples[np.isfinite(samples)]
    if len(s) < 10:
        return float("nan")
    if np.std(s) < 1e-9:
        return -np.log(1e-9)
    try:
        kde = gaussian_kde(s, bw_method=KDE_BW)
        return float(kde.logpdf([y])[0])
    except np.linalg.LinAlgError:
        return float("nan")


def _fit_detrend_perdaytype(captures: dict) -> dict:
    """Fit per-day-type linear trend of (z_actual - z_pred) at h=1
    over the captures. Returns {daytype: (slope_per_day, intercept,
    origin_timestamp)}."""
    rows = []
    days_seen = []
    for day, day_rows in captures.items():
        r1 = day_rows[0]
        if not (np.isfinite(r1["z_actual"]) and np.isfinite(r1["z_pred"])):
            continue
        rows.append({"day": pd.Timestamp(day),
                     "daytype": r1["daytype"],
                     "resid": r1["z_actual"] - r1["z_pred"]})
        days_seen.append(pd.Timestamp(day))
    df = pd.DataFrame(rows)
    origin = min(days_seen)
    out = {}
    for dt, g in df.groupby("daytype"):
        if len(g) < 10:
            continue
        ds = (g["day"] - origin).dt.days.values
        slope, intc = np.polyfit(ds, g["resid"].values, 1)
        out[dt] = (float(slope), float(intc), origin)
    return out


def _run_arm(captures: dict, detrend_by_dt: dict | None, label: str,
             out_path: Path) -> None:
    t0 = time.time()
    days = sorted(captures.keys())
    rows = []
    _say(f"[{label}] LOO scoring on {len(days)} days "
         f"(detrend={'YES' if detrend_by_dt else 'NO'})", t0)
    for i, held in enumerate(days):
        nu_dt = _fit_nu_perdaytype(captures, held, detrend_by_dt)
        day_rows = captures[held]
        rng_g = np.random.default_rng(0x6A55_0001 + i)
        rng_t = np.random.default_rng(0x6A55_0002 + i)
        g_paths = _mc_propagate_day(day_rows, np.nan, "gauss", rng_g)
        t_paths = _mc_propagate_day(
            day_rows, nu_dt[day_rows[0]["daytype"]], "t", rng_t
        )
        # Per-day-type detrend shift to apply to mean predictions
        dt = day_rows[0]["daytype"]
        if detrend_by_dt is not None and dt in detrend_by_dt:
            slope, intc, origin = detrend_by_dt[dt]
            d_since = (pd.Timestamp(held) - origin).days
            shift = slope * d_since + intc
        else:
            shift = 0.0
        for h_idx, row in enumerate(day_rows):
            y = row["z_actual"]
            if not np.isfinite(y):
                continue
            if h_idx == 0:
                mu = float(row["z_pred"]) + shift
                var = max(float(row["Sigma_00"]), 1e-9)
                sd = float(np.sqrt(var))
                lg = float(norm.logpdf(y, loc=mu, scale=sd))
                nu = nu_dt[row["daytype"]]
                t_scale = float(np.sqrt(var * (nu - 2.0) / nu))
                lt = float(student_t.logpdf((y - mu) / t_scale, df=nu)
                           - np.log(t_scale))
            else:
                # For h>=2, shift the MC paths' means by the same
                # constant `shift` (the detrend correction is per-day,
                # not per-horizon; it applies to all forecasts on
                # this delivery day).
                lg = _logdensity_at(g_paths[:, h_idx] + shift, y)
                lt = _logdensity_at(t_paths[:, h_idx] + shift, y)
            rows.append({
                "day": held, "h": h_idx + 1,
                "daytype": row["daytype"],
                "nu_dt": nu_dt[row["daytype"]],
                "ll_gauss": -lg, "ll_t": -lt,
                "shift": shift,
            })
        if (i + 1) % 50 == 0:
            _say(f"[{label}]   {i+1}/{len(days)} scored", t0)

    df = pd.DataFrame(rows)
    _say(f"[{label}] scored {len(df)} (day, h) cells", t0)

    # Per-h summary
    print()
    print("=" * 84)
    print(f"[{label}] Per-h dB(h) at extended window (n={len(days)} days)")
    print("=" * 84)
    print(f"{'h':>3} {'n':>5} {'baseline LL':>12} {'dB(h)':>10} "
          f"{'CI lo':>8} {'CI hi':>8}  {'PASS?':<8}")

    unique_days = df["day"].unique()
    per_h = []
    for h, g in df.groupby("h"):
        g = g.dropna(subset=["ll_gauss", "ll_t"])
        if len(g) < 30:
            continue
        ll_g_mean = g["ll_gauss"].mean()
        days_h = g["day"].to_numpy()
        ll_g = g["ll_gauss"].to_numpy()
        ll_t = g["ll_t"].to_numpy()
        day_idx = {d: np.where(days_h == d)[0] for d in unique_days
                   if (days_h == d).any()}
        boots = np.empty(B_BOOTSTRAP)
        for b in range(B_BOOTSTRAP):
            samp = RNG_BOOT.choice(list(day_idx.keys()),
                                   size=len(day_idx), replace=True)
            idx = np.concatenate([day_idx[d] for d in samp])
            base = ll_g[idx].mean()
            tval = ll_t[idx].mean()
            boots[b] = (base - tval) / max(abs(base), 1e-9)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        dB_point = (ll_g_mean - g["ll_t"].mean()) / max(abs(ll_g_mean), 1e-9)
        verdict = "PASS" if lo > TAU_MIN else "fail"
        per_h.append({"h": int(h), "n": int(len(g)),
                      "baseline_ll": float(ll_g_mean),
                      "dB": float(dB_point),
                      "ci_lo": float(lo), "ci_hi": float(hi),
                      "verdict": verdict})
        print(f"{int(h):>3} {len(g):>5} {ll_g_mean:>12.3f} "
              f"{dB_point:>10.4f} {lo:>8.4f} {hi:>8.4f}  {verdict:<8}")

    # Pooled summary
    raw_g = df.dropna(subset=["ll_gauss", "ll_t"])
    days_arr = raw_g["day"].to_numpy()
    by_day = {d: raw_g.index[raw_g["day"] == d].to_numpy()
              for d in unique_days}
    boots = np.empty(B_BOOTSTRAP)
    for b in range(B_BOOTSTRAP):
        samp = RNG_BOOT.choice(list(by_day.keys()),
                               size=len(by_day), replace=True)
        idx = np.concatenate([by_day[d] for d in samp])
        g_mean = raw_g.loc[idx, "ll_gauss"].mean()
        t_mean = raw_g.loc[idx, "ll_t"].mean()
        boots[b] = (g_mean - t_mean) / max(abs(g_mean), 1e-9)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    pooled_dB = ((raw_g["ll_gauss"].mean() - raw_g["ll_t"].mean())
                 / max(abs(raw_g["ll_gauss"].mean()), 1e-9))
    print()
    print(f"[{label}] POOLED dB = {pooled_dB:.4f}  "
          f"CI=[{lo:.4f}, {hi:.4f}]  "
          f"vs tau_min={TAU_MIN}")

    with out_path.open("wb") as f:
        pickle.dump({"per_h": pd.DataFrame(per_h), "raw": df,
                     "pooled_dB": pooled_dB,
                     "pooled_dB_ci": (lo, hi),
                     "detrend_by_dt": detrend_by_dt}, f)
    _say(f"[{label}] saved {out_path}", t0)


def main() -> None:
    t0 = time.time()
    _say(f"loading captures from {IN_PATH}", t0)
    with IN_PATH.open("rb") as f:
        blob = pickle.load(f)
    captures = blob["captures"]
    _say(f"{len(captures)} days", t0)

    # Fit per-day-type detrend on THIS cache's h=1 residuals
    _say("fitting per-day-type linear residual detrend", t0)
    detrend_by_dt = _fit_detrend_perdaytype(captures)
    print()
    print("Per-day-type detrend fit:")
    for dt, (s, i, o) in detrend_by_dt.items():
        print(f"  {dt:>10}: slope = {s:+.6f} z-units/day = "
              f"{s*365:+.4f} z-units/year; intercept = {i:+.4f}")
    print()

    _run_arm(captures, None, "raw", OUT_RAW)
    _run_arm(captures, detrend_by_dt, "detrend", OUT_DET)


if __name__ == "__main__":
    main()
