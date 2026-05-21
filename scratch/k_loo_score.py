"""Option K — Step 2 of 2: leave-one-day-out scoring of the local
Student-t kernel vs the Gaussian baseline, via Monte-Carlo composition
through the augmented (J, Sigma) propagation. Consumes the per-step
sequences captured by scratch/k_capture.py.

Design (locked by user choices at design time):
  * nu_hat selection: per-day-type, method-of-moments
    nu_hat_dt = max(4.5, 6 / k_dt + 4)  where k_dt is the dev-set
    Fisher excess kurtosis of standardised h=1 residuals on that
    day-type slab (matching the rebaseline's per-day-type frame
    and run_error_decomposition.py:353's MoM formula).
  * Gate: leave-one-day-out — fit nu_hat_dt on the other 131 days'
    h=1 standardised residuals (per day-type), score on the held-out
    day. Rotate over all 132 days.
  * Horizon scope: all h in 1..24, via Monte-Carlo composition. At
    each anchor, simulate N_MC trajectories under the rank-1
    augmented (J_aug, Sigma_aug) dynamics with step innovations drawn
    from either N(0, Sigma_00_step) [Gaussian arm] or scaled-t
    with df=nu_hat_dt [Student-t arm]. KDE-density at z_actual.
  * Verdict: per-h dB(h) = mean log-loss reduction relative to
    Gaussian baseline, with day-bootstrap CI (B=1000). Per-h verdict
    PASS iff CI lower bound > tau_min=0.01.

PROVENANCE-GRADE: INSPECTION-ONLY.

Notes on what this is NOT:
  * Not the existing run_error_decomposition.py:run_m2 dB ~2.3%.
    That used Gaussian log-loss with a t-corrected variance scale on
    the same residuals — a misspecification probe. This is true
    Student-t density vs true Gaussian density on the same h-step
    error path (LOO-honest nu). Expect a different signal level;
    direct comparability to the published 2.3% is limited.
  * Not a multi-step composition of CLOSED-FORM Student-t kernels
    (composition of t through linear maps is not t). MC is the only
    correct route for the density.
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, norm, t as student_t

IN_PATH = Path("/tmp/k_step_sequences.pkl")
OUT_PATH = Path("/tmp/k_loo_dB.pkl")
N_MC = 500              # Monte-Carlo trajectories per anchor per arm
B_BOOTSTRAP = 1000      # day-bootstrap replicates for the dB CI
TAU_MIN = 0.01          # pre-registered materiality threshold
NU_MIN = 4.5            # MoM clamp (matches run_error_decomposition.py:353)
KDE_BW = "scott"        # Scott's rule plug-in (NOT CV; #41-safe)
RNG_MC = np.random.default_rng(20260520)
RNG_BOOT = np.random.default_rng(20260521)


def _say(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


def _moms_nu(z_std: np.ndarray) -> float:
    """MoM nu from standardised residuals (matching the formula in
    run_error_decomposition.py:353)."""
    z = z_std[np.isfinite(z_std)]
    if len(z) < 4:
        return NU_MIN
    k = float(pd.Series(z).kurtosis())     # Fisher excess
    return max(NU_MIN, 6.0 / max(k, 1e-6) + 4.0)


def _fit_nu_perdaytype(captures: dict, held_out: pd.Timestamp) -> dict:
    """Per-day-type MoM nu_hat from the h=1 standardised residuals on
    the dev set EXCLUDING the held_out day."""
    by_dt: dict[str, list[float]] = {"weekday": [], "saturday": [],
                                     "sunday": []}
    for day, rows in captures.items():
        if day == held_out:
            continue
        r1 = rows[0]   # h=1 row
        if not np.isfinite(r1["z_actual"]):
            continue
        # standardised h=1 residual under Gaussian one-step law
        sd = np.sqrt(max(r1["Sigma_00"], 1e-9))
        z_std = (r1["z_actual"] - r1["z_pred"]) / sd
        by_dt[r1["daytype"]].append(z_std)
    return {dt: _moms_nu(np.array(vals)) for dt, vals in by_dt.items()
            if len(vals) >= 4}


def _scaled_t_draws(n: int, df: float, var: float, rng) -> np.ndarray:
    """Draw n samples from a t(df) distribution scaled to have
    variance `var`. (t variance = scale^2 * df/(df-2).)"""
    if df <= 2:
        df = 2.1
    scale = np.sqrt(var * (df - 2.0) / df)
    return scale * rng.standard_t(df, size=n)


def _gauss_draws(n: int, var: float, rng) -> np.ndarray:
    return np.sqrt(max(var, 1e-9)) * rng.standard_normal(n)


def _mc_propagate_day(day_rows: list, nu_dt: float,
                      arm: str, rng) -> np.ndarray:
    """Monte-Carlo propagate a single day's 24-step forecast under
    either 'gauss' or 't' step innovations. Returns array shape
    (N_MC, 24) of z-values at the h=1..24 coordinates."""
    D_MAX = day_rows[0]["J_aug"].shape[0]
    # state at h=0 = the lag vector before step 1; for the augmented
    # state we pad with zeros (the augmentation adds dummy coords
    # that the J_aug already accounts for as identity shifts).
    x0 = day_rows[0]["x_state"]
    state = np.zeros((N_MC, D_MAX))
    # The augmented state's first d coordinates are the original lag
    # vector; the rest are zero (per augmented_state_transition).
    state[:, :len(x0)] = x0[np.newaxis, :]

    out = np.empty((N_MC, 24))
    for h_idx, row in enumerate(day_rows):
        J = row["J_aug"]
        var = float(row["Sigma_aug"][0, 0])    # rank-1: only [0,0]
        # Deterministic part: state @ J  (J is the augmented transition)
        # Note J_aug is set so that x_next = x @ J + innovation_vec, with
        # innovation only on coord 0; this matches the propagate_predictive_cov
        # convention used elsewhere in the codebase.
        next_state = state @ J
        if arm == "gauss":
            inno = _gauss_draws(N_MC, var, rng)
        elif arm == "t":
            inno = _scaled_t_draws(N_MC, nu_dt, var, rng)
        else:
            raise ValueError(arm)
        next_state[:, 0] += inno
        state = next_state
        out[:, h_idx] = state[:, 0]
    return out


def _logdensity_at(samples: np.ndarray, y: float) -> float:
    """KDE log-density at y from N_MC samples. Scott's-rule bandwidth
    (plug-in; #41-safe). Returns log p_hat(y)."""
    s = samples[np.isfinite(samples)]
    if len(s) < 10:
        return float("nan")
    if np.std(s) < 1e-9:
        # degenerate; treat as point mass — log density is +inf where
        # equal, -inf otherwise. Use a tiny floor.
        return -np.log(1e-9)
    try:
        kde = gaussian_kde(s, bw_method=KDE_BW)
        return float(kde.logpdf([y])[0])
    except np.linalg.LinAlgError:
        return float("nan")


def _mc_sanity_vs_analytic(captures: dict) -> None:
    """Sanity gate: MC sample variance under Gaussian innovations should
    match the analytic propagate_predictive_cov[s_k] from the existing
    augmented-scope cache, within ~1/sqrt(N_MC) noise. Run on one day
    only as a quick pre-flight; abort if mismatch >5x noise bound."""
    from scratch.multistep_variance_propagation import propagate_predictive_cov
    sample_day = next(iter(captures))
    rows = captures[sample_day]
    J_seq = np.stack([r["J_aug"] for r in rows])
    S_seq = np.stack([r["Sigma_aug"] for r in rows])
    P = propagate_predictive_cov(J_seq, S_seq)
    analytic_var = np.array([P[h, 0, 0] for h in range(24)])
    rng = np.random.default_rng(99)
    mc = _mc_propagate_day(rows, np.nan, "gauss", rng)
    mc_var = mc.var(axis=0, ddof=1)
    # noise bound for variance estimate: ~ var * sqrt(2/(N_MC-1))
    noise_bound = analytic_var * np.sqrt(2.0 / (N_MC - 1))
    delta = np.abs(mc_var - analytic_var)
    print("MC sanity vs analytic variance (day=first, N_MC=500):")
    print(f"  {'h':>3} {'analytic':>12} {'mc_sample':>12} "
          f"{'|delta|':>10} {'5x_noise':>10}")
    for h in [0, 5, 11, 23]:
        ok = delta[h] < 5 * noise_bound[h]
        print(f"  {h+1:>3} {analytic_var[h]:>12.4f} {mc_var[h]:>12.4f} "
              f"{delta[h]:>10.4f} {noise_bound[h]*5:>10.4f} "
              f"  {'OK' if ok else 'MISMATCH'}")
    if np.any(delta > 5 * noise_bound):
        raise RuntimeError(
            "MC sanity FAILED — propagation convention may be wrong. "
            "Inspect _mc_propagate_day vs propagate_predictive_cov."
        )
    print("  -> MC sanity PASS\n")


def main() -> None:
    t0 = time.time()
    _say(f"loading captures from {IN_PATH}", t0)
    with IN_PATH.open("rb") as f:
        blob = pickle.load(f)
    captures = blob["captures"]
    days = sorted(captures.keys())
    _say(f"{len(days)} days; D_MAX={blob['D_MAX']}; "
         f"dims={blob['dims']}", t0)

    _mc_sanity_vs_analytic(captures)

    # Per-day per-horizon scores: rows = (day, h, ll_gauss, ll_t)
    rows = []
    _say(f"LOO scoring: {len(days)} days x 24 h x 2 arms x "
         f"{N_MC} MC each (~few minutes)", t0)
    for i, held in enumerate(days):
        nu_dt = _fit_nu_perdaytype(captures, held)
        day_rows = captures[held]
        # Use a deterministic seed per held-out day so the MC noise
        # doesn't dominate the day-bootstrap CI.
        rng_g = np.random.default_rng(0x6A55_0001 + i)
        rng_t = np.random.default_rng(0x6A55_0002 + i)
        g_paths = _mc_propagate_day(day_rows, np.nan, "gauss", rng_g)
        t_paths = _mc_propagate_day(day_rows, nu_dt[day_rows[0]["daytype"]],
                                    "t", rng_t)
        for h_idx, row in enumerate(day_rows):
            y = row["z_actual"]
            if not np.isfinite(y):
                continue
            if h_idx == 0:
                # h=1: closed-form densities (MC unnecessary; KDE noise at
                # the tails is exactly where t should win, so use exact
                # densities here for the load-bearing horizon).
                mu = float(row["z_pred"])
                var = max(float(row["Sigma_00"]), 1e-9)
                sd = float(np.sqrt(var))
                lg = float(norm.logpdf(y, loc=mu, scale=sd))
                nu = nu_dt[row["daytype"]]
                # scaled t with Var = var
                t_scale = float(np.sqrt(var * (nu - 2.0) / nu))
                lt = float(student_t.logpdf((y - mu) / t_scale, df=nu)
                           - np.log(t_scale))
            else:
                lg = _logdensity_at(g_paths[:, h_idx], y)
                lt = _logdensity_at(t_paths[:, h_idx], y)
            rows.append({
                "day": held,
                "h": h_idx + 1,
                "daytype": row["daytype"],
                "nu_dt": nu_dt[row["daytype"]],
                "ll_gauss": -lg,    # log-loss = -log density
                "ll_t": -lt,
            })
        if (i + 1) % 20 == 0:
            _say(f"  {i+1}/{len(days)} days scored", t0)

    df = pd.DataFrame(rows)
    _say(f"scored {len(df)} (day, h) cells", t0)

    # === Per-h baseline mean log-loss + dB(h) point estimate + day-bootstrap CI ===
    print()
    print("=" * 84)
    print("Option K verdict: per-horizon dB(h) = "
          "(LL_gauss - LL_t) / |LL_gauss|, with day-bootstrap 95% CI")
    print(f"Pre-registered tau_min = {TAU_MIN:.3f}; PASS iff CI lower > tau_min")
    print("=" * 84)
    print(f"{'h':>3} {'n':>5} {'baseline LL':>12} {'dB(h)':>10} "
          f"{'CI lo':>8} {'CI hi':>8}  {'verdict':<8}")

    unique_days = df["day"].unique()
    n_d = len(unique_days)

    per_h: list[dict] = []
    for h, g in df.groupby("h"):
        # Drop NaN log-losses (KDE degeneracies — should be rare)
        g = g.dropna(subset=["ll_gauss", "ll_t"])
        if len(g) < 30:
            continue
        ll_g_mean = g["ll_gauss"].mean()
        # Day-bootstrap on the dB ratio
        days_h = g["day"].to_numpy()
        ll_g = g["ll_gauss"].to_numpy()
        ll_t = g["ll_t"].to_numpy()
        # group by day for resampling
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

    # === Headline ===
    print()
    print("=" * 84)
    print("Headline:")
    print("=" * 84)
    n_pass = sum(1 for r in per_h if r["verdict"] == "PASS")
    if n_pass == 0:
        print("Option K VERDICT: HOLD — no horizon clears the "
              f"pre-registered tau_min = {TAU_MIN:.2f} on the LOO MC dB.")
        print("Student-t local kernel does NOT licence a switch from the "
              "Gaussian baseline at any horizon.")
    elif n_pass < 6:
        print(f"Option K VERDICT: WEAK PASS — {n_pass}/24 horizons clear "
              f"tau_min. Localised improvement; full-population materiality "
              "depends on horizon weighting.")
    else:
        print(f"Option K VERDICT: PASS — {n_pass}/24 horizons clear "
              f"tau_min. Student-t local kernel is LICENCED.")

    # Persist
    out = {
        "per_h": pd.DataFrame(per_h),
        "raw": df,
        "config": {
            "N_MC": N_MC, "B_BOOTSTRAP": B_BOOTSTRAP,
            "TAU_MIN": TAU_MIN, "NU_MIN": NU_MIN, "KDE_BW": KDE_BW,
        },
    }
    with OUT_PATH.open("wb") as f:
        pickle.dump(out, f)
    _say(f"saved to {OUT_PATH}", t0)


if __name__ == "__main__":
    main()
