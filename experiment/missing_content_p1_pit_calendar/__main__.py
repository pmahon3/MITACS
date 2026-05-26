"""P1 runner — PIT calendar fingerprint analysis.

Implements the metric pre-registered in
notes/preregistrations/2026-05-26_pit-calendar-fingerprint/phase_a.yaml.
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import ndtr

from config import PROJECT_ROOT, load_config
from experiment import freeze
from experiment._actuals import (
    load_actuals,
    zscore_params,
    zscore_transform,
)
from experiment.backtest import _complete_delivery_days
from experiment.predict import _daytype


# ---------------------------------------------------------------------------
# Pre-registered binning (phase_a §binning)
# ---------------------------------------------------------------------------


SEASON_BINS = {
    "winter":   {12, 1, 2},
    "summer":   {6, 7, 8},
    "shoulder": {3, 4, 5, 9, 10, 11},
}

HE_BINS = {
    "overnight":    set(range(1, 7)),         # HE1-6
    "morning_ramp": set(range(7, 11)),        # HE7-10
    "midday":       set(range(11, 17)),       # HE11-16
    "evening_ramp": set(range(17, 25)),       # HE17-24
}


def _season_of(month: int) -> str:
    for s, ms in SEASON_BINS.items():
        if month in ms:
            return s
    raise ValueError(f"no season for month {month}")


def _he_bin_of(he: int) -> str:
    for b, hs in HE_BINS.items():
        if he in hs:
            return b
    raise ValueError(f"no HE bin for hour {he}")


# ---------------------------------------------------------------------------
# Iterated diffusion (mirrors the v2 helper but inline here is acceptable —
# it's the discrete-time analogue of eq (★) of the seed, used in v2 itself;
# the formula is fixed by the registered metric definition in phase_a)
# ---------------------------------------------------------------------------


def _iterated_drift_and_var(
    C1: np.ndarray, Sigma1: np.ndarray, x_state: np.ndarray, h_max: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """For a fixed starting state x_state and per-day-type (C_1, Sigma_1),
    return:
      mu_iter[h]   for h = 1..h_max      (scalar; coord-0 of C_1^h x_state)
      var_iter[h]  for h = 1..h_max      (scalar; coord-(0,0) of cumulative sum)

    The iterated drift x_{h} = C_1^h x_state has its coord-0 as the
    iterated mean prediction; the iterated covariance is the sum
    sum_{j=0..h-1} (C_1^j Sigma_1 C_1^{jT}) coord-(0,0).
    """
    mu = np.zeros(h_max)
    var = np.zeros(h_max)
    Cj = np.eye(C1.shape[0])           # C_1^0 = I
    sum_cov = np.zeros_like(Sigma1)    # accumulator for sum_j C_j Sigma_1 C_j^T
    x = x_state.copy()
    for h in range(1, h_max + 1):
        # Iterated covariance update: accumulate BEFORE advancing Cj
        sum_cov = sum_cov + Cj @ Sigma1 @ Cj.T
        var[h - 1] = sum_cov[0, 0]
        # Advance the drift: x_{h} = C_1 x_{h-1}
        x = C1 @ x
        mu[h - 1] = x[0]
        Cj = Cj @ C1
    return mu, var


# ---------------------------------------------------------------------------
# Build the PIT table per delivery day x hour
# ---------------------------------------------------------------------------


def _build_pit_table(
    z_full: pd.Series,
    factors: dict,
    cutoff: pd.Timestamp,
    anchor_h: int,
    dims: dict,
) -> pd.DataFrame:
    """For every post-cutoff delivery day with complete window, compute
    PIT u(d, h) for h = 1..24.

    Output columns:
        delivery_day, target_dt, h, day_type,
        z_actual, mu_iter, var_iter, u_PIT,
        month, doy, dow, year, hour_of_day,
        season_bin, he_bin
    """
    days = _complete_delivery_days(z_full, anchor_h)
    days = days[days > cutoff]
    rows = []
    for D in days:
        D = pd.Timestamp(D.date())
        dt = _daytype(D + pd.Timedelta(hours=anchor_h), anchor_h)
        d = int(dims[dt])
        C1 = factors[dt][1]["C"]
        S1 = factors[dt][1]["Sigma"]
        # Issue-time anchor: hour before the delivery-day window begins
        issue_anchor = D + pd.Timedelta(hours=anchor_h - 1)
        # Build x_state = (z(issue), z(issue-1), ..., z(issue-d+1))
        lag_times = [issue_anchor - pd.Timedelta(hours=i) for i in range(d)]
        if not all(t in z_full.index for t in lag_times):
            continue
        x_state = z_full.reindex(lag_times).to_numpy()
        if not np.all(np.isfinite(x_state)):
            continue
        # Iterated forecasts for h=1..24
        mu_iter, var_iter = _iterated_drift_and_var(C1, S1, x_state, h_max=24)
        # Targets and actuals
        target_dts = [
            D + pd.Timedelta(hours=anchor_h + (h - 1)) for h in range(1, 25)
        ]
        actuals = z_full.reindex(target_dts).to_numpy()
        if not np.all(np.isfinite(actuals)):
            continue
        for h in range(1, 25):
            i = h - 1
            std = np.sqrt(var_iter[i]) if var_iter[i] > 0 else np.nan
            if not np.isfinite(std) or std <= 0:
                continue
            u = float(ndtr((actuals[i] - mu_iter[i]) / std))
            t = target_dts[i]
            rows.append({
                "delivery_day": D,
                "target_dt":    t,
                "h":            h,
                "day_type":     dt,
                "z_actual":     float(actuals[i]),
                "mu_iter":      float(mu_iter[i]),
                "var_iter":     float(var_iter[i]),
                "u_PIT":        u,
                "month":        t.month,
                "doy":          t.dayofyear,
                "dow":          t.dayofweek,           # 0=Mon, 6=Sun
                "year":         t.year,
                "hour_of_day":  h,                     # = HE for daytype anchored at anchor_h
                "season_bin":   _season_of(t.month),
                "he_bin":       _he_bin_of(h),
            })
    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# Candidate set (calendar-only; pre-registered)
# ---------------------------------------------------------------------------


# Federal/Ontario provincial statutory holidays (post-cutoff window).
# Hardcoded list (acquisition-free; matches phase_a §candidate_set.other).
_HOLIDAYS_2025_2026: set[pd.Timestamp] = {
    pd.Timestamp(d) for d in [
        "2025-01-01",  # New Year's Day
        "2025-02-17",  # Family Day (ON)
        "2025-04-18",  # Good Friday
        "2025-05-19",  # Victoria Day
        "2025-07-01",  # Canada Day
        "2025-08-04",  # Civic Holiday (ON)
        "2025-09-01",  # Labour Day
        "2025-10-13",  # Thanksgiving
        "2025-11-11",  # Remembrance Day (observed; varying)
        "2025-12-25",  # Christmas
        "2025-12-26",  # Boxing Day
        "2026-01-01",
        "2026-02-16",  # Family Day
        "2026-04-03",  # Good Friday
        "2026-05-18",  # Victoria Day
    ]
}


def _candidate_values(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Compute each calendar candidate at every row of df.

    Returns a dict {candidate_name: 1-D array aligned to df}.
    """
    out: dict[str, np.ndarray] = {}
    doy = df["doy"].to_numpy()
    out["cos_annual"] = np.cos(2 * np.pi * doy / 365.0)
    out["sin_annual"] = np.sin(2 * np.pi * doy / 365.0)
    out["cos_semi"]   = np.cos(2 * np.pi * doy / 182.5)
    out["sin_semi"]   = np.sin(2 * np.pi * doy / 182.5)

    # day-of-week indicators (0=Mon .. 6=Sun)
    dow = df["dow"].to_numpy()
    for i, name in enumerate(
        ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
    ):
        out[f"dow_indicator_{name}"] = (dow == i).astype(float)

    # holiday indicator (per phase_a)
    out["holiday_indicator"] = df["target_dt"].apply(
        lambda t: 1.0 if pd.Timestamp(t).normalize() in _HOLIDAYS_2025_2026 else 0.0
    ).to_numpy()

    # year_since_cutoff: continuous secular drift candidate
    yr = df["year"].to_numpy() + (df["doy"].to_numpy() - 1) / 365.0
    out["year_since_cutoff"] = yr - 2024.5

    # he_dev_from_daytype_mean: residual z_actual from mean within (daytype, h)
    # Compute the mean per (daytype, hour_of_day) on this dataframe, then
    # subtract per-row.
    df = df.copy()
    df["_dt_h"] = df["day_type"].astype(str) + "_" + df["hour_of_day"].astype(str)
    means = df.groupby("_dt_h")["z_actual"].transform("mean")
    out["he_dev_from_daytype_mean"] = (df["z_actual"] - means).to_numpy()

    # DoW x HE interaction (per phase_a) — encode as a 168-level categorical
    # whose mean PIT per level we'll project; for the candidate-vector form,
    # use the average PIT-residual at level (dow, hour_of_day) excluding the
    # row, but for simplicity in cell-aggregation we use a 1-D summary: the
    # mean z_actual at the (dow, hour_of_day) level, similar in shape to
    # he_dev_from_daytype_mean but cross-cutting DoW. This captures DoW*HE
    # fine-structure that day_type alone doesn't.
    df["_dow_h"] = df["dow"].astype(str) + "_" + df["hour_of_day"].astype(str)
    means_dh = df.groupby("_dow_h")["z_actual"].transform("mean")
    out["dow_he_interaction"] = (df["z_actual"] - means_dh).to_numpy()

    return out


# ---------------------------------------------------------------------------
# Cell-aggregated explanatory score
# ---------------------------------------------------------------------------


def _cell_key(row) -> tuple:
    return (row["season_bin"], row["he_bin"], row["day_type"])


def _explanatory_score(
    df: pd.DataFrame, candidate: np.ndarray, *, cell_T: dict | None = None,
) -> tuple[float, dict]:
    """Cell-weighted Spearman partial correlation.

    Returns (score, cell_T) where cell_T is the per-cell mean PIT
    deviation (computed once; passed in to reuse across candidates).
    """
    if cell_T is None:
        cell_T = (df.assign(pit_res=df["u_PIT"] - 0.5)
                    .groupby(["season_bin", "he_bin", "day_type"])["pit_res"]
                    .mean()
                    .to_dict())

    # Spearman per cell: rank-correlation within each (s, he, dt) cell
    df_w = df.copy()
    df_w["_cand"] = candidate
    df_w["_pit_res"] = df_w["u_PIT"] - 0.5
    df_w["_cell_key"] = list(zip(df_w["season_bin"], df_w["he_bin"], df_w["day_type"]))

    score = 0.0
    per_cell = {}
    for cell, sub in df_w.groupby("_cell_key", sort=False):
        if len(sub) < 5:
            per_cell[cell] = float("nan")
            continue
        # rank-correlation: Spearman; use pd's corr with method='spearman'
        rho = sub["_cand"].corr(sub["_pit_res"], method="spearman")
        if np.isnan(rho):
            per_cell[cell] = float("nan")
            continue
        T = cell_T.get(cell, 0.0)
        per_cell[cell] = float(rho)
        score += abs(T) * abs(rho)
    return score, per_cell


# ---------------------------------------------------------------------------
# Null distribution (random permutation of candidate's time index)
# ---------------------------------------------------------------------------


def _null_distribution(
    df: pd.DataFrame, candidate: np.ndarray, *, cell_T: dict, n_perm: int = 1000,
    seed: int = 0,
) -> np.ndarray:
    """Null distribution of explanatory_score under random permutation
    of the candidate's time index. The observed (PIT, cell) structure
    is held fixed; only the candidate's alignment with timestamps is
    scrambled."""
    rng = np.random.default_rng(seed)
    n = len(candidate)
    nulls = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(n)
        cand_perm = candidate[perm]
        s, _ = _explanatory_score(df, cand_perm, cell_T=cell_T)
        nulls[i] = s
    return nulls


# ---------------------------------------------------------------------------
# Paired-day bootstrap
# ---------------------------------------------------------------------------


def _bootstrap_scores(
    df: pd.DataFrame, candidates: dict[str, np.ndarray],
    n_bootstrap: int = 1000, seed: int = 1,
) -> dict[str, np.ndarray]:
    """Paired-day bootstrap on the explanatory_score per candidate.

    Resamples delivery days (with replacement); recomputes cell_T and
    score on each resample. Returns {candidate: array of n_bootstrap
    scores}.
    """
    rng = np.random.default_rng(seed)
    all_days = pd.DatetimeIndex(sorted(set(df["delivery_day"])))
    n_days = len(all_days)
    # Pre-index df by delivery_day for fast lookup
    grouped = {d: g for d, g in df.groupby("delivery_day", sort=False)}
    cand_names = list(candidates.keys())
    out = {name: np.empty(n_bootstrap) for name in cand_names}
    for b in range(n_bootstrap):
        idx = rng.integers(0, n_days, size=n_days)
        sampled_days = all_days[idx]
        df_b = pd.concat([grouped[d] for d in sampled_days], ignore_index=True)
        # Recompute candidates on the bootstrapped df (he_dev_from_daytype_mean
        # depends on the data distribution; recompute it correctly per bootstrap).
        cand_b = _candidate_values(df_b)
        cell_T_b = (df_b.assign(pit_res=df_b["u_PIT"] - 0.5)
                       .groupby(["season_bin", "he_bin", "day_type"])["pit_res"]
                       .mean()
                       .to_dict())
        for name in cand_names:
            s, _ = _explanatory_score(df_b, cand_b[name], cell_T=cell_T_b)
            out[name][b] = s
    return out


# ---------------------------------------------------------------------------
# Outcome firing
# ---------------------------------------------------------------------------


_SEASONAL = {"cos_annual", "sin_annual", "cos_semi", "sin_semi"}
_WEEKLY = (
    {f"dow_indicator_{d}" for d in ("mon", "tue", "wed", "thu", "fri", "sat", "sun")}
    | {"dow_he_interaction"}
)
_OTHER = {"holiday_indicator", "year_since_cutoff", "he_dev_from_daytype_mean"}


def _class_of(name: str) -> str:
    if name in _SEASONAL: return "seasonal"
    if name in _WEEKLY:   return "weekly"
    if name in _OTHER:    return "other"
    raise ValueError(f"unknown candidate {name}")


def _evaluate_outcome(
    observed_scores: dict[str, float],
    boot_scores: dict[str, np.ndarray],
    null_scores: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Per phase_a §shape_outcomes:
      O-A: top seasonal cleared null AND top_seasonal >= 2.0 * top_weekly AND >= 2.0 * top_other
      O-B: top weekly cleared null AND top_weekly >= 2.0 * top_seasonal AND >= 2.0 * top_other
      O-C: both seasonal and weekly cleared null AND ratio in (0.5, 2.0)
      O-D: no candidate cleared null
      AMBIGUOUS: anything else (borderline, multiple fire)
    """
    # CI lower bound per candidate
    ci_low = {name: float(np.percentile(boot_scores[name], 2.5))
              for name in observed_scores}
    null_95 = {name: float(np.percentile(null_scores[name], 95))
               for name in observed_scores}
    significant = {name: ci_low[name] > null_95[name]
                   for name in observed_scores}

    # Group by class
    by_class = {"seasonal": {}, "weekly": {}, "other": {}}
    for name, score in observed_scores.items():
        by_class[_class_of(name)][name] = score

    # Top per class
    def _top(class_name):
        items = [(n, s) for n, s in by_class[class_name].items() if significant[n]]
        if not items:
            return None, 0.0
        items.sort(key=lambda x: -x[1])
        return items[0]

    top_s_name, top_s_score = _top("seasonal")
    top_w_name, top_w_score = _top("weekly")
    top_o_name, top_o_score = _top("other")

    any_significant = any(significant.values())

    # Outcome decision
    if not any_significant:
        verdict = "O-D"
        reason = "No candidate cleared the null baseline at 95% CI lower bound > 95th null percentile."
    else:
        # Compute the ratio if both seasonal and weekly significant
        if top_s_name is not None and top_w_name is not None:
            # Top seasonal vs top weekly
            if top_s_score >= 2.0 * top_w_score and top_s_score >= 2.0 * max(top_o_score, 1e-12):
                verdict = "O-A"
                reason = f"Top seasonal ({top_s_name}, score={top_s_score:.4f}) >= 2x all others."
            elif top_w_score >= 2.0 * top_s_score and top_w_score >= 2.0 * max(top_o_score, 1e-12):
                verdict = "O-B"
                reason = f"Top weekly ({top_w_name}, score={top_w_score:.4f}) >= 2x all others."
            else:
                ratio = top_s_score / max(top_w_score, 1e-12)
                if 0.5 < ratio < 2.0:
                    verdict = "O-C"
                    reason = f"Both classes significant; ratio seasonal/weekly = {ratio:.2f} in (0.5, 2.0)."
                else:
                    verdict = "AMBIGUOUS"
                    reason = f"Ratio {ratio:.2f} borderline; neither outcome fires cleanly."
        elif top_s_name is not None and top_w_name is None:
            # Only seasonal significant
            if top_s_score >= 2.0 * max(top_o_score, 1e-12):
                verdict = "O-A"
                reason = f"Only seasonal significant; top seasonal {top_s_name} dominates."
            else:
                verdict = "AMBIGUOUS"
                reason = f"Seasonal significant but does not dominate other-class candidate."
        elif top_w_name is not None and top_s_name is None:
            if top_w_score >= 2.0 * max(top_o_score, 1e-12):
                verdict = "O-B"
                reason = f"Only weekly significant; top weekly {top_w_name} dominates."
            else:
                verdict = "AMBIGUOUS"
                reason = f"Weekly significant but does not dominate other-class candidate."
        else:
            # Only "other" significant
            verdict = "AMBIGUOUS"
            reason = "Only other-class candidates significant; not in the O-A/B/C/D taxonomy."

    return {
        "verdict": verdict,
        "reason": reason,
        "significant": significant,
        "ci_low": ci_low,
        "null_95": null_95,
        "top_seasonal": (top_s_name, top_s_score),
        "top_weekly":   (top_w_name, top_w_score),
        "top_other":    (top_o_name, top_o_score),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="in_path", type=Path,
                   default=Path("scratch/data/multiscale_factor/factors_v2_final.pkl"))
    p.add_argument("--out", type=Path,
                   default=Path("scratch/data/missing_content_p1/pit_calendar.pkl"))
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--n-perm", type=int, default=1000)
    p.add_argument("--skip-bootstrap", action="store_true",
                   help="point estimates + null only (for fast iteration)")
    args = p.parse_args(argv)

    cfg = load_config()
    spec = freeze.load_verified()
    cutoff = pd.Timestamp(spec["data_cutoff"])
    anchor_h = int(cfg.data.day_anchor_hours)
    dims = spec["predictor"]["embedding_dims"]
    clim_method = spec["predictor"].get("climatology_method", "month_hour")
    k_year = spec["predictor"].get("fourier_k_year")
    k_day = spec["predictor"].get("fourier_k_day")

    print(f"# P1 PIT calendar fingerprint")
    print(f"  factor pickle: {args.in_path}")
    print(f"  cutoff:        {cutoff}")
    print(f"  anchor_h:      {anchor_h}")
    print(f"  embedding:     {dict(dims)}")
    print()

    # Load v2 factors
    with args.in_path.open("rb") as f:
        v2 = pickle.load(f)
    factors = v2["factors"]

    # Build z_full from production (matches v2's pre-cutoff library + post-cutoff actuals)
    raw_full = load_actuals(cutoff=None).dropna()
    zp = zscore_params(cutoff, method=clim_method, k_year=k_year, k_day=k_day)
    z_full = zscore_transform(raw_full, zp)

    print(f"# building PIT table (post-cutoff delivery days x 24 hours) ...")
    t0 = time.time()
    df = _build_pit_table(z_full, factors, cutoff, anchor_h, dims)
    print(f"  rows: {len(df)}  ({time.time()-t0:.1f}s)")
    print(f"  daytypes: {df['day_type'].value_counts().to_dict()}")
    print()

    # Marginal PIT shape (DA's secondary check)
    pit = df["u_PIT"].to_numpy()
    pit_centered = pit - 0.5
    print(f"# marginal PIT (DA's metric-blindness check):")
    print(f"  mean(u):        {pit.mean():.4f}  (uniform: 0.5)")
    print(f"  std(u):         {pit.std(ddof=1):.4f}  (uniform: 0.289)")
    print(f"  mean(|u-0.5|):  {np.abs(pit_centered).mean():.4f}  (uniform: 0.25)")
    # Histogram counts to assess U-shape
    bins = np.linspace(0, 1, 11)
    hist, _ = np.histogram(pit, bins=bins)
    expected = len(pit) / 10
    chi2 = ((hist - expected) ** 2 / expected).sum()
    print(f"  chi2 vs uniform (10 bins): {chi2:.2f}  (df=9; critical 95% ~16.92)")
    print(f"  histogram (10 bins): {list(hist)}")
    print()

    # Per-cell mean PIT deviation
    cell_T = (df.assign(pit_res=df["u_PIT"] - 0.5)
                .groupby(["season_bin", "he_bin", "day_type"])["pit_res"]
                .mean()
                .to_dict())
    print(f"# per-cell mean PIT deviation (T_c = mean(u-0.5) per cell):")
    for cell, T in sorted(cell_T.items()):
        marker = " *" if abs(T) > 0.05 else ""
        print(f"  {cell[0]:>8s}/{cell[1]:>13s}/{cell[2]:>8s}  T = {T:+.4f}{marker}")
    print()

    # Compute candidate values
    candidates = _candidate_values(df)
    print(f"# {len(candidates)} candidates:")
    for name in candidates:
        cls = _class_of(name)
        print(f"  [{cls:>8s}]  {name}")
    print()

    # Observed scores
    print(f"# computing observed explanatory_scores ...")
    t0 = time.time()
    observed_scores = {}
    for name, vals in candidates.items():
        score, _ = _explanatory_score(df, vals, cell_T=cell_T)
        observed_scores[name] = score
    print(f"  ({time.time()-t0:.1f}s)")
    for name, s in sorted(observed_scores.items(), key=lambda x: -x[1]):
        cls = _class_of(name)
        print(f"  [{cls:>8s}]  {name:>30s}  score={s:.4f}")
    print()

    # Null distribution per candidate
    print(f"# null distributions ({args.n_perm} permutations per candidate) ...")
    t0 = time.time()
    null_scores = {}
    for name, vals in candidates.items():
        nulls = _null_distribution(df, vals, cell_T=cell_T, n_perm=args.n_perm, seed=hash(name) & 0xFFFFFFFF)
        null_scores[name] = nulls
    print(f"  ({time.time()-t0:.1f}s)")
    for name in sorted(observed_scores, key=lambda n: -observed_scores[n]):
        p95 = np.percentile(null_scores[name], 95)
        p_obs = (null_scores[name] >= observed_scores[name]).mean()
        cls = _class_of(name)
        print(f"  [{cls:>8s}]  {name:>30s}  obs={observed_scores[name]:.4f}  "
              f"null_95={p95:.4f}  p={p_obs:.3f}")
    print()

    # Bootstrap (optional)
    if args.skip_bootstrap:
        print(f"# bootstrap SKIPPED")
        boot_scores = None
    else:
        print(f"# paired-day bootstrap (n={args.n_bootstrap}) ...")
        t0 = time.time()
        boot_scores = _bootstrap_scores(df, candidates, n_bootstrap=args.n_bootstrap)
        print(f"  ({time.time()-t0:.1f}s)")
        for name in sorted(observed_scores, key=lambda n: -observed_scores[n]):
            ci_low = np.percentile(boot_scores[name], 2.5)
            ci_high = np.percentile(boot_scores[name], 97.5)
            median = np.percentile(boot_scores[name], 50)
            cls = _class_of(name)
            print(f"  [{cls:>8s}]  {name:>30s}  "
                  f"CI=[{ci_low:.4f}, {ci_high:.4f}]  median={median:.4f}")
        print()

    # Outcome firing
    if boot_scores is not None:
        outcome = _evaluate_outcome(observed_scores, boot_scores, null_scores)
        print(f"# pre-registered outcome: {outcome['verdict']}")
        print(f"  reason: {outcome['reason']}")
        if outcome["top_seasonal"][0]:
            print(f"  top_seasonal: {outcome['top_seasonal'][0]} (score={outcome['top_seasonal'][1]:.4f})")
        if outcome["top_weekly"][0]:
            print(f"  top_weekly:   {outcome['top_weekly'][0]} (score={outcome['top_weekly'][1]:.4f})")
        if outcome["top_other"][0]:
            print(f"  top_other:    {outcome['top_other'][0]} (score={outcome['top_other'][1]:.4f})")
        print()

    # Pickle
    out = args.out
    if not out.is_absolute():
        out = (Path.cwd() / out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pit_df": df,
        "cell_T": cell_T,
        "candidates": candidates,
        "observed_scores": observed_scores,
        "null_scores": null_scores,
        "boot_scores": boot_scores,
        "outcome": outcome if boot_scores is not None else None,
        "marginal_pit_stats": {
            "mean": float(pit.mean()),
            "std": float(pit.std(ddof=1)),
            "mean_abs_centered": float(np.abs(pit_centered).mean()),
            "chi2_uniform_10bin": float(chi2),
            "hist_10bin": list(hist.astype(int)),
        },
        "config": {
            "cutoff": cutoff,
            "anchor_h": anchor_h,
            "embedding_dims": dict(dims),
            "n_bootstrap": args.n_bootstrap if not args.skip_bootstrap else 0,
            "n_perm": args.n_perm,
        },
    }
    with out.open("wb") as f:
        pickle.dump(payload, f)
    try:
        rel = out.relative_to(PROJECT_ROOT)
        print(f"  wrote {rel}")
    except ValueError:
        print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
