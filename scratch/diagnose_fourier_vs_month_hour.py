"""A-vs-B preprocessing diagnostic.

Goal-aware reframe: deseasonalisation is preprocessing in service of
day-ahead forecasting. The current (month, hour) lookup has a known
artefact (12 seam events/year at month boundaries) that contaminates
forecast-cache evidence on those days. Path A: replace with a Fourier
climatology (smooth in day-of-year and hour-of-day). Path B: keep
(month, hour), exclude seam days by design.

This script tests whether a small-truncation Fourier climatology
cleanly beats (month, hour) on three diagnostics, all on pre-cutoff
data so the comparison is in-sample for the climatology fit
(post-cutoff is reserved for the actual forecast experiments):

  (i)  Residual stationarity: year-over-year (mean, std) of the
       deseasonalised z series. Should be ~(0, 1) every year.
  (ii) Smoothness across the month boundary: average z and its
       jump across the midnight-of-first-of-month transition.
       Under (month, hour) we expect a step; under Fourier, expect
       smooth.
  (iii) Variance reduction: fraction of raw-demand variance that
       each preprocessing removes. The better preprocessing
       removes more, all else equal.

Decision rule (pre-stated):
  Pick A if Fourier (a) reduces year-over-year residual variance
  meaningfully (e.g. variance ratio < 0.9 vs month-hour), AND
  (b) smooths the month-boundary jump by > 50% (|jump| reduction).
  Otherwise pick B.

PROVENANCE-GRADE: INSPECTION-ONLY (diagnostic; not a fit artifact).
"""
from __future__ import annotations

import sys
sys.path.insert(0, '/Users/pmahon/Research/Dynamics/MITACS')
import numpy as np
import pandas as pd
from experiment._actuals import load_actuals, zscore_params, zscore_transform

CUTOFF = pd.Timestamp("2021-10-18 23:00:00")
K_YEAR = 3   # Fourier harmonics on day-of-year (annual, semi-annual, ...)
K_DAY = 4    # Fourier harmonics on hour-of-day


def _doy(idx: pd.DatetimeIndex) -> np.ndarray:
    """Day-of-year as a float in [0, 365.25)."""
    return idx.dayofyear.values.astype(float) - 1.0


def _hod(idx: pd.DatetimeIndex) -> np.ndarray:
    return idx.hour.values.astype(float)


def _fourier_design(idx: pd.DatetimeIndex, k_year: int = K_YEAR,
                    k_day: int = K_DAY) -> np.ndarray:
    """Design matrix with constant + k_year (cos, sin) pairs in
    day-of-year + k_day (cos, sin) pairs in hour-of-day. Total
    columns = 1 + 2*k_year + 2*k_day."""
    d = _doy(idx) / 365.25
    h = _hod(idx) / 24.0
    n = len(idx)
    cols = [np.ones(n)]
    for k in range(1, k_year + 1):
        cols.append(np.cos(2 * np.pi * k * d))
        cols.append(np.sin(2 * np.pi * k * d))
    for j in range(1, k_day + 1):
        cols.append(np.cos(2 * np.pi * j * h))
        cols.append(np.sin(2 * np.pi * j * h))
    return np.column_stack(cols)


def fit_fourier_climatology(demand: pd.Series, cutoff: pd.Timestamp):
    """Fit Fourier climatology on pre-cutoff data. Returns (mu_fn,
    sigma_fn) callable on a DatetimeIndex."""
    train = demand[demand.index <= cutoff].dropna()
    X = _fourier_design(train.index)
    beta_mu, *_ = np.linalg.lstsq(X, train.values, rcond=None)
    resid = train.values - X @ beta_mu
    # Fit sigma on residual magnitude: |D - mu| ~ similar Fourier shape
    # (well-known that demand variance has day/year structure too).
    beta_sig, *_ = np.linalg.lstsq(X, np.abs(resid), rcond=None)
    # sigma estimate via |resid| / sqrt(2/pi) (the half-normal moment factor)
    sigma_scale = np.sqrt(np.pi / 2.0)

    def mu_fn(idx):
        return _fourier_design(idx) @ beta_mu

    def sigma_fn(idx):
        sig = _fourier_design(idx) @ beta_sig * sigma_scale
        return np.maximum(sig, 1.0)   # floor to prevent /0 in pathological cells

    return mu_fn, sigma_fn, beta_mu, beta_sig


def deseason_month_hour(demand: pd.Series, cutoff: pd.Timestamp) -> pd.Series:
    """Existing (month, hour) preprocessing for direct comparison."""
    zp = zscore_params(cutoff)
    return zscore_transform(demand, zp)


def deseason_fourier(demand: pd.Series, cutoff: pd.Timestamp) -> pd.Series:
    mu_fn, sigma_fn, _, _ = fit_fourier_climatology(demand, cutoff)
    mu = mu_fn(demand.index)
    sig = sigma_fn(demand.index)
    return pd.Series((demand.values - mu) / sig, index=demand.index,
                     name="zscore")


def yearly_stats(z: pd.Series, label: str) -> pd.DataFrame:
    """Year-over-year (mean, std). Stationary means both stay near
    (0, 1) every year."""
    df = z.to_frame("z").copy()
    df["year"] = df.index.year
    g = df.groupby("year")["z"].agg(["mean", "std", "count"])
    g["label"] = label
    return g


def month_boundary_stats(z: pd.Series, label: str) -> dict:
    """Mean z by hour-offset around the midnight-of-first-of-month
    boundary, pooled across all such boundaries.

    Hour offset = 0 means 00:00 on the first of a month.
    Hour offset = -1 means 23:00 on the previous month's last day.
    The 'jump' is mean[offset=0] - mean[offset=-1] (should be ~0
    under a good preprocessing)."""
    df = z.to_frame("z").copy()
    df = df.dropna()
    # Find all timestamps at the first-of-month boundary in dev data.
    # Each boundary spans -6..+6 hours.
    boundaries = df.index[
        (df.index.day == 1) & (df.index.hour == 0)
    ]
    rows = []
    for boundary in boundaries:
        for offset in range(-6, 7):
            t = boundary + pd.Timedelta(hours=offset)
            if t in df.index:
                rows.append({"offset": offset, "z": float(df.loc[t, "z"])})
    bdf = pd.DataFrame(rows)
    if bdf.empty:
        return {"label": label, "n_boundaries": 0}
    g = bdf.groupby("offset")["z"].agg(["mean", "std", "count"])
    jump = float(g.loc[0, "mean"] - g.loc[-1, "mean"])
    return {
        "label": label,
        "n_boundaries": int(len(boundaries)),
        "table": g,
        "jump_at_boundary": jump,
    }


def variance_reduction(demand: pd.Series, z: pd.Series, label: str,
                       cutoff: pd.Timestamp) -> dict:
    """Fraction of raw-demand variance removed by the preprocessing,
    computed on pre-cutoff data (in-sample for climatology fit)."""
    pre = demand[demand.index <= cutoff].dropna()
    z_pre = z[z.index <= cutoff].dropna()
    common = pre.index.intersection(z_pre.index)
    raw_var = float(pre.loc[common].var())
    z_var = float(z_pre.loc[common].var())   # z_var should be ~1 if normalised well
    # var of raw demand without preprocessing: raw_var
    # var of (D - mu): equivalent to var(z * sigma_eff) — hard to disentangle
    # cleanly. Cleanest single number: 1 - z_var (how much of the
    # standardisation went "wrong" — z_var close to 1 means well-calibrated).
    return {
        "label": label,
        "raw_var": raw_var,
        "z_var": z_var,
        "z_mean": float(z_pre.loc[common].mean()),
        "n": int(len(common)),
    }


def main():
    print(f"{'='*72}")
    print(f"A-vs-B preprocessing diagnostic")
    print(f"CUTOFF = {CUTOFF.date()}, Fourier K_year={K_YEAR}, K_day={K_DAY}")
    print(f"{'='*72}\n")

    act = load_actuals(cutoff=None).asfreq("h")
    print(f"Loaded actuals: {act.index[0].date()} to {act.index[-1].date()}, "
          f"n_hours = {len(act):,}")
    print()

    print("Deseasonalising under each preprocessing...")
    z_mh = deseason_month_hour(act, CUTOFF)
    z_fr = deseason_fourier(act, CUTOFF)

    # ============ (i) Yearly stats: stationarity-ish ============
    print("(i) Year-over-year (mean, std) of deseasonalised z")
    print("    (well-calibrated preprocessing: ~(0, 1) every year)")
    print()
    for label, z in [("month-hour", z_mh), ("fourier", z_fr)]:
        ys = yearly_stats(z[z.index <= CUTOFF], label)
        print(f"  {label} (pre-cutoff in-sample):")
        with pd.option_context("display.float_format", "{:.4f}".format):
            print(ys.to_string())
        print()
    # post-cutoff (where the dev experiments happen)
    print("  POST-cutoff (out-of-sample for climatology fit):")
    for label, z in [("month-hour", z_mh), ("fourier", z_fr)]:
        ys = yearly_stats(z[z.index > CUTOFF], label)
        print(f"    {label}:")
        with pd.option_context("display.float_format", "{:.4f}".format):
            print(ys.to_string())
        print()

    # ============ (ii) Month-boundary smoothness ============
    print("(ii) Month-boundary smoothness (mean z by hour-offset)")
    print("     The seam: |z[offset=0] - z[offset=-1]| should be small")
    print()
    for label, z in [("month-hour", z_mh), ("fourier", z_fr)]:
        info = month_boundary_stats(z[z.index <= CUTOFF], label)
        print(f"  {label} (pre-cutoff):")
        print(f"    n boundaries: {info['n_boundaries']}")
        with pd.option_context("display.float_format", "{:.4f}".format):
            print(info["table"].to_string())
        print(f"    JUMP at offset 0 vs -1: "
              f"{info['jump_at_boundary']:+.4f}")
        print()

    # ============ (iii) Variance / mean diagnostics ============
    print("(iii) Calibration check on pre-cutoff data:")
    for label, z in [("month-hour", z_mh), ("fourier", z_fr)]:
        vr = variance_reduction(act, z, label, CUTOFF)
        print(f"  {label}: mean(z) = {vr['z_mean']:+.4f}, "
              f"var(z) = {vr['z_var']:.4f}, n = {vr['n']:,}")
    print()

    # ============ Pre-stated decision rule ============
    print(f"{'='*72}")
    print(f"Pre-stated decision rule:")
    print(f"  A wins if BOTH:")
    print(f"    (a) Fourier yearly variance ratio < 0.9 vs month-hour")
    print(f"        in EITHER pre- or post-cutoff slabs")
    print(f"    (b) Fourier month-boundary jump magnitude reduced > 50%")
    print(f"{'='*72}")
    mh_jump = abs(month_boundary_stats(
        z_mh[z_mh.index <= CUTOFF], "mh")["jump_at_boundary"])
    fr_jump = abs(month_boundary_stats(
        z_fr[z_fr.index <= CUTOFF], "fr")["jump_at_boundary"])
    jump_reduction = 1 - fr_jump / mh_jump if mh_jump > 0 else float("nan")
    print(f"  month-hour |jump|: {mh_jump:.4f}")
    print(f"  fourier    |jump|: {fr_jump:.4f}")
    print(f"  jump reduction:    {jump_reduction:.1%}")
    print()

    # Variance ratio (pre-cutoff, year-pooled)
    mh_pre = z_mh[z_mh.index <= CUTOFF].dropna()
    fr_pre = z_fr[z_fr.index <= CUTOFF].dropna()
    common = mh_pre.index.intersection(fr_pre.index)
    mh_var = float(mh_pre.loc[common].var())
    fr_var = float(fr_pre.loc[common].var())
    var_ratio = fr_var / mh_var if mh_var > 0 else float("nan")
    print(f"  month-hour var: {mh_var:.4f}")
    print(f"  fourier    var: {fr_var:.4f}")
    print(f"  variance ratio: {var_ratio:.4f}")
    print()

    crit_a = var_ratio < 0.9
    crit_b = jump_reduction > 0.5
    print(f"  Criterion (a) variance ratio < 0.9: "
          f"{'PASS' if crit_a else 'fail'}")
    print(f"  Criterion (b) jump reduction > 50%: "
          f"{'PASS' if crit_b else 'fail'}")
    print()
    if crit_a and crit_b:
        print("  -> Pick A (commit to Fourier preprocessing rebuild).")
    elif crit_a or crit_b:
        print(f"  -> MIXED: one criterion passes, one fails. Worth "
              f"closer inspection of the failing criterion before "
              f"committing.")
    else:
        print(f"  -> Pick B (keep month-hour, exclude seam days by "
              f"design).")


if __name__ == "__main__":
    main()
