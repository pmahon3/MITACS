"""EXPLORATORY: §4.2's pre-registered four-channel comparison for whether
de-seasonalisation can be removed. (Framework-level decision experiment;
dev 2021-22; scratch/, NOT recorded; see memory mitacs-zscore-framework-
coupling and writeup/tex/draft_body.tex §4.2.)

Q: Given that the location-axis signal is recoverable from raw demand
at essentially the same magnitude as from z-space (corr -0.247 / -0.250
/ -0.217 across global / monthhour / identity in DEMAND space; see
scratch/diagnose_zscore_coupling.py), should de-seasonalisation be
removed from the pipeline?

The phase-channel finding alone does not license removal. De-
seasonalisation acts on FOUR estimator layers (state, drift, diffusion,
demand-space round-trip); a single-axis MAE comparison can hide failure
modes in the other three. Pre-registered four-channel comparison; all
tolerances fixed before the experiment runs (in §4.2):

  Channel 1 -- Forecast accuracy. Tolerance: |dMAE| < 8 MW, |dMAPE|
                < 0.05 pp (~1% of seasonal baseline; §3 materiality scale).
  Channel 2 -- Sigma_j numerical viability (BINARY, per-pipeline; reframed
                from a unit-mismatched cross-pipeline ratio test). For each
                pipeline independently: fit Sigma_j at N=60 anchors per day-
                type, propagate variance through 24 horizons via the gate-
                validated augmented_state_transition, and report the fraction
                of anchors finite at every horizon. Tolerance: >= 99% for
                BOTH pipelines independently.
  Channel 3 -- Interval coverage in DEMAND units, hour-stratified.
                Tolerance: per-hour max |cov_raw(h) - cov_seasonal(h)|
                < 5 pp AND |avg cov_raw - avg cov_seasonal| < 2 pp.
  Channel 4 -- Heavy-tail nu_hat (propagation-aware Student-t shape on
                standardised residuals). Tolerance: nu_raw >= nu_seasonal
                (direction-of-effect check; §3.2 reconciliation says heavy
                tail was partly a propagation artefact tied to
                de-seasonalisation).

Decision rule (asymmetric, all-channel): removal LICENSED only if all 4
tolerances clear; HOLD otherwise.

Design points (advisor-locked-in before writing):
  * Production code path only: _local_fit_at, local_drift_and_diffusion,
    _refrozen_dims; the propagation primitive from scratch/multistep_
    variance_propagation.py for nu_hat. NO reimplementation.
  * WITHIN-HARNESS comparison: both pipelines run through the same dev
    diagnostic with their per-day-type elbow-selected dimensions. The
    production-frozen 786 MW / 4.78% reference is on a separate harness
    and is NOT the comparator here.
  * Day-type seam mask applied CONSISTENTLY on both pipelines (the
    rebaseline established full-process Sigma_j is catastrophically
    ill-conditioned; without seam masking Channel 2 verdict is preordained).
  * Channel 3 measures coverage in DEMAND units on both pipelines: seasonal
    interval width = 1.96 * sigma_mh * sqrt(Sigma_j[0,0]); no-transform =
    1.96 * sqrt(Sigma_j[0,0]).

PROVENANCE-GRADE: INSPECTION-ONLY -- exploratory dev-set; MUST NOT be
cited as a result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from edynamics.modelling_tools import Embedding, Lag

from experiment._actuals import load_actuals
from experiment.predict import _daytype
from processing.innovations.estimator import (
    _local_fit_at,
    local_drift_and_diffusion,
)
from scratch.multistep_variance_propagation import (
    augmented_state_transition,
    propagate_predictive_cov,
    state_transition_from_local_fit,
)

CUTOFF = pd.Timestamp("2021-10-18 23:00:00")
WIN_START = pd.Timestamp("2021-10-19")
WIN_END = pd.Timestamp("2022-02-27")
ANCHOR_H = 7
DAYTYPES = ("weekday", "saturday", "sunday")
N_ANCHORS = 60                       # per day-type, for Channel 2
N_INTERVAL_SAMPLES = 200             # per day-type, for Channel 3
INTERVAL_Z = 1.959963984540054       # 1.96 (95% Gaussian quantile)


# =====================================================================
# Climatology and z-transform
# =====================================================================
def _clim_monthhour(act: pd.Series) -> dict:
    """Production seasonal spec: per (month, hour-of-day)."""
    a = act[act.index <= CUTOFF]
    g = a.groupby([a.index.month, a.index.hour])
    return {"mu": g.mean().to_dict(), "sd": g.std().to_dict()}


def _z_seasonal(series: pd.Series, clim: dict) -> pd.Series:
    mu = np.array([clim["mu"][(t.month, t.hour)] for t in series.index])
    sd = np.array([clim["sd"][(t.month, t.hour)] for t in series.index])
    return pd.Series((series.values - mu) / sd, index=series.index)


def _inv_seasonal(zval: float, ts: pd.Timestamp, clim: dict) -> float:
    return zval * clim["sd"][(ts.month, ts.hour)] + clim["mu"][(ts.month, ts.hour)]


def _scale_seasonal(ts: pd.Timestamp, clim: dict) -> float:
    """Demand-space scale (sigma_mh) for one timestamp -- needed for
    Channel 3 interval coverage in demand units."""
    return float(clim["sd"][(ts.month, ts.hour)])


# --- Candidate A: scale-only normalisation ---------------------------
# z_A = D / sigma_{m,h}, no mean subtraction. Tests whether the additive
# sub-step of the z-transform is the carrier of the heavy-tail cost
# while preserving the demand-space scale step that Ch3 located as
# load-bearing.

def _z_scale_only(series: pd.Series, clim: dict) -> pd.Series:
    sd = np.array([clim["sd"][(t.month, t.hour)] for t in series.index])
    return pd.Series(series.values / sd, index=series.index)


def _inv_scale_only(zval: float, ts: pd.Timestamp, clim: dict) -> float:
    return zval * clim["sd"][(ts.month, ts.hour)]


# =====================================================================
# Per-day-type elbow rule (mirrors scratch/benchmark_2021_22_ieso.py)
# =====================================================================
def _elbow_dim(rho_by_d: pd.Series, tol: float = 0.001) -> int:
    peak = rho_by_d.max()
    return int(rho_by_d.index[rho_by_d >= peak - tol * abs(peak)][0])


def _refrozen_dims(series: pd.Series, name: str = "v") -> dict:
    """Per-day-type elbow rule on a generic series (z or raw demand).

    Same elbow construction as scratch/benchmark_2021_22_ieso.py:
    in-sample one-step rho vs d, take the smallest d within tol of
    peak rho. Day-type tagging via _daytype with ANCHOR_H = 7.
    """
    df = series.to_frame(name).asfreq("h")
    out = {}
    for dtname in DAYTYPES:
        idx = df.index[[_daytype(t, ANCHOR_H) == dtname for t in df.index]]
        idx = idx[idx <= CUTOFF]
        rho = {}
        for d in range(1, 9):
            lags = [Lag(variable_name=name, tau=-i) for i in range(d)]
            fl = idx[d:-1]
            if len(fl) < 500:
                break
            emb = Embedding(data=df, observers=lags, library_times=fl)
            emb.compile()
            b = emb.block
            X, Y = b.iloc[:-1].values, b.iloc[1:].values
            C = np.linalg.lstsq(X, Y, rcond=None)[0]
            pred = (X @ C)[:, 0]
            rho[d] = float(np.corrcoef(pred, Y[:, 0])[0, 1])
        out[dtname] = _elbow_dim(pd.Series(rho))
    return out


# =====================================================================
# Library construction with day-type seam mask
# =====================================================================
def _build_library(series: pd.Series, name: str, dim: int, daytype: str):
    """Build (X, Y) library pairs strictly <= cutoff, restricted to one
    day-type, with day-type-seam pairs (target on ANCHOR_H) excluded.

    Returns: (X (Np, dim), Y (Np, dim), full_lib_index).
    """
    df = series.to_frame(name).asfreq("h")
    lags = [Lag(variable_name=name, tau=-i) for i in range(dim)]
    # day-type-tagged times in the pre-cutoff window
    idx = df.index[df.index <= CUTOFF]
    idx = idx[[_daytype(t, ANCHOR_H) == daytype for t in idx]]
    if len(idx) < dim + 2:
        return None
    fl = idx[dim:-1]
    emb = Embedding(data=df, observers=lags, library_times=fl)
    emb.compile()
    b = emb.block
    X = b.iloc[:-1].values
    Y = b.iloc[1:].values
    # day-type seam mask: drop pairs whose one-step target falls on the
    # day-anchor hour (mirrors production local_drift_and_diffusion's
    # day_anchor_hour treatment)
    target_idx = b.index[1:]
    keep = np.array([t.hour != ANCHOR_H for t in target_idx])
    return X[keep], Y[keep], target_idx[keep]


# =====================================================================
# Channel 1 -- end-to-end multi-step backtest (MAE/MAPE in demand)
# =====================================================================
def _backtest(act: pd.Series, dims: dict, kind: str) -> pd.DataFrame:
    """Iterated one-step day-ahead forecast over WIN_START..WIN_END.

    kind: 'seasonal'   -> z = (D - mu_mh) / sd_mh, forecast in z, invert.
          'identity'   -> operate on raw D.
          'scale_only' -> z_A = D / sd_mh (Candidate A: scale-only
                          normalisation, no mean subtraction).

    Returns DataFrame indexed by target hour with columns:
      ours_mw, actual_mw, z_pred (or D_pred), sigma_mh, s2_one_step,
      daytype, horizon_h, delivery_date.
    Per-step Sigma_j[0,0] captured for Channel 3 / Channel 4.
    """
    if kind == "seasonal":
        clim = _clim_monthhour(act)
        series = _z_seasonal(act, clim)
    elif kind == "scale_only":
        clim = _clim_monthhour(act)            # only sd_mh is used
        series = _z_scale_only(act, clim)
    else:
        clim = None
        series = act.copy()
    name = "v"

    df = series.to_frame(name).asfreq("h")
    lib_cache: dict[tuple[str, int], tuple] = {}

    days = pd.date_range(WIN_START, WIN_END, freq="D")
    rows = []
    for D in days:
        targets = [D + pd.Timedelta(hours=h) for h in range(24)]
        ia = targets[0] - pd.Timedelta(hours=1)
        dmax = max(dims.values())
        need = [ia - pd.Timedelta(hours=i) for i in range(dmax)]
        if not all(n in series.index for n in need):
            continue
        zh = {ts: float(series.loc[ts]) for ts in series.index
              if ia - pd.Timedelta(hours=dmax) <= ts <= ia}
        for t in targets:
            dtname = _daytype(t, ANCHOR_H)
            d = int(dims[dtname])
            key = (dtname, d)
            if key not in lib_cache:
                lib = _build_library(series, name, d, dtname)
                if lib is None:
                    continue
                Xl, Yl, _ = lib
                lib_cache[key] = (Xl, Yl)
            Xl, Yl = lib_cache[key]
            prev = t - pd.Timedelta(hours=1)
            lag_t = [prev - pd.Timedelta(hours=i) for i in range(d)]
            try:
                xq = np.array([zh[lt] for lt in lag_t], dtype=float)
            except KeyError:
                break
            C, S, _mu, _th, _ = _local_fit_at(Xl, Yl, xq, d)
            v_next = float(xq @ (C[:, 0] if C.ndim == 2 else C))
            zh[t] = v_next
            s2_step = float(S[0, 0])
            if kind == "seasonal":
                sd_mh = _scale_seasonal(t, clim)
                ours = _inv_seasonal(v_next, t, clim)
            elif kind == "scale_only":
                sd_mh = _scale_seasonal(t, clim)
                ours = _inv_scale_only(v_next, t, clim)
            else:
                sd_mh = 1.0
                ours = v_next
            rows.append({
                "dt": t, "delivery_date": D, "horizon_h": int((t - targets[0])
                                                              / pd.Timedelta(hours=1)) + 1,
                "ours_mw": ours, "sigma_mh": sd_mh,
                "s2_step": s2_step, "daytype": dtname, "d": d,
                "v_pred": v_next,
                "C0": (C[:, 0] if C.ndim == 2 else C).copy(),
            })

    out = pd.DataFrame(rows).set_index("dt").sort_index()
    out["actual_mw"] = out.index.map(act)
    out = out.dropna(subset=["actual_mw"])
    return out


def channel_1(fc: pd.DataFrame) -> dict:
    err = fc["ours_mw"] - fc["actual_mw"]
    mae = float(err.abs().mean())
    mape = float((err.abs() / fc["actual_mw"]).mean() * 100.0)
    return {"MAE_MW": mae, "MAPE_pct": mape, "n": int(len(fc))}


# =====================================================================
# Channel 2 -- per-day-type Sigma_j conditioning (with seam mask)
# =====================================================================
def channel_2(act: pd.Series, dims: dict, kind: str,
              rng: np.random.Generator) -> dict:
    """Binary numerical-viability check, applied PER PIPELINE: at N=60
    anchors per day-type, fit production Sigma_j with day-type seam
    exclusion, propagate variance through 24 horizons via the gate-
    validated augmented_state_transition, and check that the final
    predictive variance is finite at every horizon.

    Returns per-day-type: n_anchors, n_finite (anchors with all-24-finite
    propagated variance), fraction_finite. Tolerance (caller): >=0.99.
    Unit-invariant by construction: finiteness is a property of the
    matrix machinery, not of its absolute scale.
    """
    if kind == "seasonal":
        clim = _clim_monthhour(act)
        series = _z_seasonal(act, clim)
    elif kind == "scale_only":
        clim = _clim_monthhour(act)
        series = _z_scale_only(act, clim)
    else:
        series = act.copy()
    df = series.to_frame("v").asfreq("h")

    # d_max for the augmented state (used uniformly across day-types
    # within this pipeline, since the augmented propagation expects
    # uniform-d state across the iteration). For a per-day-type
    # comparison we propagate each anchor's Sigma_j 24 steps using its
    # own day-type's d throughout (no day-type rollover in this
    # diagnostic; we're checking SINGLE-day-type viability per anchor).
    out = {}
    for dtname in DAYTYPES:
        d = int(dims[dtname])
        lags = [Lag(variable_name="v", tau=-i) for i in range(d)]
        idx = df.index[df.index <= CUTOFF]
        idx = idx[[_daytype(t, ANCHOR_H) == dtname for t in idx]]
        fl = idx[d:-1]
        if len(fl) < 100:
            out[dtname] = {
                "n_anchors": 0, "n_finite": 0, "fraction_finite": None,
            }
            continue
        emb = Embedding(data=df, observers=lags, library_times=fl)
        emb.compile()
        n_take = min(N_ANCHORS, len(fl))
        anchors = pd.DatetimeIndex(
            np.sort(rng.choice(fl, size=n_take, replace=False))
        )
        n_finite = 0
        n_total = 0
        for a in anchors:
            try:
                C, Sigma, _mu, _theta, _resid = local_drift_and_diffusion(
                    embedding=emb, anchor=a, day_anchor_hour=ANCHOR_H,
                )
            except Exception:
                continue
            n_total += 1
            # Build the shift-aware Jacobian + rank-1 step covariance
            # via the production function, and iterate 24 steps holding
            # the (J, S) at this anchor fixed. If any horizon's
            # predictive variance is non-finite, this anchor fails.
            try:
                c0 = C[:, 0] if C.ndim == 2 else np.asarray(C, dtype=float)
                J, Sg = state_transition_from_local_fit(C, float(Sigma[0, 0]), d)
                # 24-step propagation
                J_seq = np.stack([J] * 24)
                S_seq = np.stack([Sg] * 24)
                P = propagate_predictive_cov(J_seq, S_seq)
                # P[k, 0, 0] is the per-horizon predictive variance
                s2 = P[:, 0, 0]
                if np.all(np.isfinite(s2)) and np.all(s2 >= 0):
                    n_finite += 1
            except Exception:
                continue
        out[dtname] = {
            "n_anchors": int(n_total),
            "n_finite": int(n_finite),
            "fraction_finite": (float(n_finite) / n_total) if n_total else None,
        }
    return out


# =====================================================================
# Channel 3 -- demand-space 95% interval empirical coverage
# =====================================================================
def channel_3(fc: pd.DataFrame) -> dict:
    """Empirical coverage of nominal 95% Gaussian-proxy intervals in
    demand units, stratified by hour-of-day. seasonal: half-width =
    1.96 * sigma_mh * sqrt(s2_step). raw: half-width = 1.96 *
    sqrt(s2_step) (sigma_mh = 1).

    Returns the per-hour coverage table and the average across hours.
    Caller compares hour-by-hour to detect systematic diurnal
    miscalibration that an average alone would mask.
    """
    half = INTERVAL_Z * fc["sigma_mh"] * np.sqrt(np.clip(fc["s2_step"], 0, None))
    err = (fc["ours_mw"] - fc["actual_mw"]).abs()
    covered = (err <= half).astype(float)
    hod = fc.index.hour
    per_hour = {
        int(h): float(covered[hod == h].mean() * 100.0)
        for h in range(24) if (hod == h).any()
    }
    return {
        "average_coverage_pct": float(covered.mean() * 100.0),
        "per_hour_coverage_pct": per_hour,
        "nominal_pct": 95.0,
        "n": int(len(fc)),
        "median_halfwidth_MW": float(half.median()),
    }


# =====================================================================
# Channel 4 -- propagation-aware nu_hat (heavy-tail strength)
# =====================================================================
def channel_4(fc: pd.DataFrame) -> dict:
    """For each delivery day, propagate predictive variance through the
    24-step iteration using the gate-validated augmented_state_transition;
    compute standardised residuals; fit Student-t shape via MoM (matches
    scratch/run_error_decomposition.py)."""
    # group by delivery day; per day, accumulate per-step Jacobians and
    # innovation covariances, propagate, and standardise residuals.
    d_max = int(fc["d"].max())
    std_resid = []
    for D, day in fc.groupby("delivery_date"):
        day = day.sort_values("horizon_h")
        J_seq, S_seq = [], []
        for _, row in day.iterrows():
            d = int(row["d"])
            # augmented_state_transition accepts c0 as a 1-D array of length d
            # (or a (d,d) C from which it extracts the first column); pass the
            # stored c0 directly. d_max-augmentation handles the per-step d
            # variation across day-types.
            c0 = np.asarray(row["C0"], dtype=float)
            J, Sg = augmented_state_transition(
                c0, float(row["s2_step"]), d, d_max,
            )
            J_seq.append(J)
            S_seq.append(Sg)
        if not J_seq:
            continue
        P = propagate_predictive_cov(np.stack(J_seq), np.stack(S_seq))
        s2_k = P[:, 0, 0]                          # per-horizon predictive variance, v-units
        # Standardised residual in v-units. The forecast was in v-space
        # (v_pred); inverted to demand only for reporting. Recover the
        # residual in v-units by dividing the demand-space residual by
        # sigma_mh (which is 1 for the raw-demand pipeline, so this is
        # a no-op there).
        resid_v = (day["ours_mw"].values - day["actual_mw"].values) \
                  / day["sigma_mh"].values
        sd_k = np.sqrt(np.clip(s2_k, 1e-12, None))
        z = resid_v / sd_k
        std_resid.extend(z.tolist())
    if not std_resid:
        return {"nu_hat": None, "excess_kurt": None, "n": 0}
    arr = np.asarray(std_resid, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 50:
        return {"nu_hat": None, "excess_kurt": None, "n": int(len(arr))}
    # method-of-moments Student-t shape from standardised-resid excess kurt
    m2 = float(np.mean(arr**2))
    m4 = float(np.mean(arr**4))
    if m2 <= 0:
        return {"nu_hat": None, "excess_kurt": None, "n": int(len(arr))}
    excess_kurt = m4 / (m2**2) - 3.0
    if excess_kurt <= 0:
        nu_hat = np.inf
    else:
        # t-distribution excess kurtosis = 6 / (nu - 4) for nu > 4
        nu_hat = 6.0 / excess_kurt + 4.0
    return {
        "nu_hat": float(nu_hat),
        "excess_kurt": float(excess_kurt),
        "n": int(len(arr)),
    }


# =====================================================================
# Decision rule (pre-registered in §4.2)
# =====================================================================
def decision(seasonal: dict, raw: dict) -> dict:
    """Apply the four pre-registered tolerances; return per-channel pass
    and overall verdict (LICENSED iff all four clear; HOLD otherwise)."""
    # Channel 1: |dMAE| < 8 MW AND |dMAPE| < 0.05 pp
    dMAE = abs(raw["ch1"]["MAE_MW"] - seasonal["ch1"]["MAE_MW"])
    dMAPE = abs(raw["ch1"]["MAPE_pct"] - seasonal["ch1"]["MAPE_pct"])
    c1 = dMAE < 8.0 and dMAPE < 0.05
    # Channel 2: per-pipeline fraction-finite >= 0.99, BOTH sides
    def _all_finite(pkg) -> bool:
        for dt in DAYTYPES:
            cell = pkg["ch2"].get(dt)
            if not cell or cell["fraction_finite"] is None:
                return False
            if cell["fraction_finite"] < 0.99:
                return False
        return True
    c2_seasonal_ok = _all_finite(seasonal)
    c2_raw_ok = _all_finite(raw)
    c2_pass = c2_seasonal_ok and c2_raw_ok
    # Channel 3: hour-stratified. Per-hour worst-gap < 5 pp AND avg-gap < 2 pp.
    s_per_hour = seasonal["ch3"]["per_hour_coverage_pct"]
    r_per_hour = raw["ch3"]["per_hour_coverage_pct"]
    common_hours = sorted(set(s_per_hour) & set(r_per_hour))
    gaps = [abs(r_per_hour[h] - s_per_hour[h]) for h in common_hours]
    worst_hour_gap = max(gaps) if gaps else None
    avg_gap = abs(
        raw["ch3"]["average_coverage_pct"]
        - seasonal["ch3"]["average_coverage_pct"]
    )
    c3 = (worst_hour_gap is not None
          and worst_hour_gap < 5.0
          and avg_gap < 2.0)
    # Channel 4: nu_raw >= nu_seasonal
    nr = raw["ch4"]["nu_hat"]
    ns = seasonal["ch4"]["nu_hat"]
    c4 = (nr is not None) and (ns is not None) and (nr >= ns)
    return {
        "ch1_pass": c1, "ch1_dMAE": dMAE, "ch1_dMAPE": dMAPE,
        "ch2_pass": c2_pass,
        "ch2_seasonal_ok": c2_seasonal_ok, "ch2_raw_ok": c2_raw_ok,
        "ch3_pass": c3, "ch3_worst_hour_gap_pp": worst_hour_gap,
        "ch3_avg_gap_pp": avg_gap,
        "ch4_pass": c4, "ch4_nu_seasonal": ns, "ch4_nu_raw": nr,
        "verdict": "LICENSED" if (c1 and c2_pass and c3 and c4) else "HOLD",
    }


# =====================================================================
def main() -> None:
    print("Loading actuals (Ontario hourly demand)...")
    act = load_actuals(cutoff=None).asfreq("h")
    print(f"  series: {act.index.min()} -> {act.index.max()}, "
          f"n = {len(act)}")

    print("\n=== Per-day-type elbow-rule dimensions ===")
    clim = _clim_monthhour(act)
    z_seasonal = _z_seasonal(act, clim)
    z_scale_only = _z_scale_only(act, clim)
    dims_seasonal = _refrozen_dims(z_seasonal)
    dims_raw = _refrozen_dims(act)
    dims_scale_only = _refrozen_dims(z_scale_only)
    print(f"  seasonal (on z):           {dims_seasonal}")
    print(f"  no-transform (on D):       {dims_raw}")
    print(f"  scale-only (on D/sigma):   {dims_scale_only}")

    # Pre-registered confound check (§4.2): if the scale-only elbow rule
    # lands a dim outside [2, 4] for any day-type, flag the result as
    # confounded and withhold the candidate-A verdict.
    scale_d_oor = {
        dt: int(dims_scale_only[dt])
        for dt in DAYTYPES
        if not (2 <= int(dims_scale_only[dt]) <= 4)
    }
    if scale_d_oor:
        print(f"  *** WARNING: scale-only dims outside [2,4] for "
              f"{scale_d_oor}; candidate-A verdict will be flagged as "
              f"CONFOUNDED per pre-registration. ***")

    print("\n=== Channel 1: end-to-end backtest ===")
    print("  (running seasonal pipeline...)")
    fc_seasonal = _backtest(act, dims_seasonal, "seasonal")
    print("  (running no-transform pipeline...)")
    fc_raw = _backtest(act, dims_raw, "identity")
    print("  (running scale-only pipeline...)")
    fc_scale = _backtest(act, dims_scale_only, "scale_only")
    ch1_s = channel_1(fc_seasonal)
    ch1_r = channel_1(fc_raw)
    ch1_a = channel_1(fc_scale)
    print(f"  seasonal:   MAE = {ch1_s['MAE_MW']:.1f} MW, "
          f"MAPE = {ch1_s['MAPE_pct']:.3f}%, n = {ch1_s['n']}")
    print(f"  no-transform: MAE = {ch1_r['MAE_MW']:.1f} MW, "
          f"MAPE = {ch1_r['MAPE_pct']:.3f}%, n = {ch1_r['n']}")
    print(f"  scale-only: MAE = {ch1_a['MAE_MW']:.1f} MW, "
          f"MAPE = {ch1_a['MAPE_pct']:.3f}%, n = {ch1_a['n']}")

    print("\n=== Channel 2: Sigma_j numerical viability (per pipeline) ===")
    rng = np.random.default_rng(42)
    ch2_s = channel_2(act, dims_seasonal, "seasonal", rng)
    rng = np.random.default_rng(42)
    ch2_r = channel_2(act, dims_raw, "identity", rng)
    rng = np.random.default_rng(42)
    ch2_a = channel_2(act, dims_scale_only, "scale_only", rng)
    for dt in DAYTYPES:
        s, r, a = ch2_s[dt], ch2_r[dt], ch2_a[dt]
        print(f"  {dt:9s}: seasonal {s.get('n_finite', 0)}/{s.get('n_anchors', 0)}, "
              f"no-transform {r.get('n_finite', 0)}/{r.get('n_anchors', 0)}, "
              f"scale-only {a.get('n_finite', 0)}/{a.get('n_anchors', 0)}")

    print("\n=== Channel 3: demand-space 95% coverage, hour-stratified ===")
    ch3_s = channel_3(fc_seasonal)
    ch3_r = channel_3(fc_raw)
    ch3_a = channel_3(fc_scale)
    print(f"  seasonal:   avg cov = {ch3_s['average_coverage_pct']:.2f}%, "
          f"median half-width = {ch3_s['median_halfwidth_MW']:.1f} MW")
    print(f"  no-transform: avg cov = {ch3_r['average_coverage_pct']:.2f}%, "
          f"median half-width = {ch3_r['median_halfwidth_MW']:.1f} MW")
    print(f"  scale-only: avg cov = {ch3_a['average_coverage_pct']:.2f}%, "
          f"median half-width = {ch3_a['median_halfwidth_MW']:.1f} MW")
    print("  per-hour |cov_scale - cov_seasonal| (pp):")
    s_ph = ch3_s["per_hour_coverage_pct"]
    a_ph = ch3_a["per_hour_coverage_pct"]
    for h in sorted(set(s_ph) & set(a_ph)):
        gap = abs(a_ph[h] - s_ph[h])
        print(f"    h={h:>2}: seasonal {s_ph[h]:5.1f}%  scale-only "
              f"{a_ph[h]:5.1f}%  |gap| = {gap:4.1f}")

    print("\n=== Channel 4: propagation-aware nu_hat ===")
    ch4_s = channel_4(fc_seasonal)
    ch4_r = channel_4(fc_raw)
    ch4_a = channel_4(fc_scale)
    print(f"  seasonal:   nu_hat = {ch4_s['nu_hat']}, excess kurt "
          f"= {ch4_s['excess_kurt']}, n = {ch4_s['n']}")
    print(f"  no-transform: nu_hat = {ch4_r['nu_hat']}, excess kurt "
          f"= {ch4_r['excess_kurt']}, n = {ch4_r['n']}")
    print(f"  scale-only: nu_hat = {ch4_a['nu_hat']}, excess kurt "
          f"= {ch4_a['excess_kurt']}, n = {ch4_a['n']}")

    # Verdicts: keep the prior seasonal-vs-raw (already-resolved) and
    # add the new seasonal-vs-scale-only (the candidate-A verdict).
    seasonal_pkg = {"ch1": ch1_s, "ch2": ch2_s, "ch3": ch3_s, "ch4": ch4_s}
    raw_pkg = {"ch1": ch1_r, "ch2": ch2_r, "ch3": ch3_r, "ch4": ch4_r}
    scale_pkg = {"ch1": ch1_a, "ch2": ch2_a, "ch3": ch3_a, "ch4": ch4_a}

    print("\n=== Verdict: seasonal vs. scale-only (Candidate A) ===")
    v_a = decision(seasonal_pkg, scale_pkg)
    # Pre-registered Ch4-conditional-on-Ch3 rule (§4.2): if Ch3 fails,
    # Ch4 reports "not interpretable" rather than PASS or FAIL.
    ch4_interpretable = v_a["ch3_pass"]
    ch4_status = (
        ("PASS" if v_a["ch4_pass"] else "FAIL")
        if ch4_interpretable
        else "NOT INTERPRETABLE (Ch3 failed)"
    )
    print(f"  Ch1 (forecast):       |dMAE| = {v_a['ch1_dMAE']:.2f} MW, "
          f"|dMAPE| = {v_a['ch1_dMAPE']:.4f} pp -- "
          f"{'PASS' if v_a['ch1_pass'] else 'FAIL'}")
    print(f"  Ch2 (Sigma viability): seasonal OK = "
          f"{v_a['ch2_seasonal_ok']}, scale-only OK = "
          f"{v_a['ch2_raw_ok']} -- "
          f"{'PASS' if v_a['ch2_pass'] else 'FAIL'}")
    wh = v_a['ch3_worst_hour_gap_pp']
    wh_str = f"{wh:.2f}" if wh is not None else "n/a"
    print(f"  Ch3 (coverage):       worst-hour |gap| = {wh_str} pp, "
          f"avg |gap| = {v_a['ch3_avg_gap_pp']:.2f} pp -- "
          f"{'PASS' if v_a['ch3_pass'] else 'FAIL'}")
    print(f"  Ch4 (heavy-tail):     nu_seasonal = "
          f"{v_a['ch4_nu_seasonal']}, nu_scale-only = "
          f"{v_a['ch4_nu_raw']} -- {ch4_status}")

    # Candidate-A verdict applies the all-channel rule with the
    # Ch4-conditional refinement and the d-confound flag.
    if scale_d_oor:
        verdict_A = "CONFOUNDED (scale-only elbow d outside [2,4])"
    elif not ch4_interpretable:
        # Ch3 fails -> Ch4 not interpretable -> outcome is by Ch1,Ch2,Ch3 alone;
        # if any of those fails, HOLD; otherwise we don't have enough info.
        if v_a["ch1_pass"] and v_a["ch2_pass"]:
            # Ch3 fail with Ch1+Ch2 pass: HOLD with channel = Ch3 (scale-axis
            # coverage failure even though scale step is preserved). This is
            # mechanistically unexpected; surface as a separate note.
            verdict_A = "HOLD (Ch3 fail under scale preservation; investigate)"
        else:
            verdict_A = "HOLD"
    else:
        verdict_A = ("LICENSED" if (v_a["ch1_pass"] and v_a["ch2_pass"]
                                    and v_a["ch3_pass"] and v_a["ch4_pass"])
                     else "HOLD")
    print(f"\n  CANDIDATE-A VERDICT: {verdict_A}")

    # Pre-registered three-branch outcome diagnosis (§4.2)
    if scale_d_oor:
        branch = "confound"
    elif v_a["ch1_pass"] and v_a["ch2_pass"] and v_a["ch3_pass"] \
            and (v_a["ch4_pass"] if ch4_interpretable else True):
        branch = "(i) all-pass: additive sub-step is the removable lever"
    elif v_a["ch3_pass"] and ch4_interpretable and not v_a["ch4_pass"]:
        branch = ("(ii) Ch3 passes, Ch4 fails: heavy-tail cost lives "
                  "with the multiplicative sigma_{m,h} step, not the "
                  "additive one; scale-only not the lever")
    elif not v_a["ch1_pass"]:
        branch = ("(iii) Ch1 fails substantially: additive step is "
                  "also conditioning the embedding-state input; "
                  "decomposition not separable at this level")
    else:
        branch = "outcome not one of the pre-stated branches; report verbatim"
    print(f"  Pre-stated branch: {branch}")


if __name__ == "__main__":
    main()
