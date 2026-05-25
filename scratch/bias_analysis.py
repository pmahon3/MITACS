"""EXPLORATORY: mechanism tests for the mid-day forecast-high bias.

Runs three diagnostics, each making distinguishing predictions:

  (1) Mean iterated trajectory of predictor 2 (mean-iter, theta=0)
      from a representative midnight start, plotted against actual
      mean demand by hour, day-type-stratified.  Tells us whether
      the iterated trajectory traces the wrong diurnal shape (the
      "(ε) wrong-shape-trajectory" hypothesis) or merely damps
      toward the population mean (the "(α) eigenvalue-pull"
      hypothesis).

  (2) Eigenstructure of the global C(theta=0) operator per day-type.
      Spectral decomposition of the d×d drift matrix.  Eigenvalues
      tell us about damping (|λ| < 1: trajectory decays to a fixed
      point; |λ| ≈ 1: persistent), complex pairs tell us about
      oscillation periods.  A diurnal cycle requires a complex pair
      with period ≈ 24h; deviation explains diurnal shape error.

  (3) Climatology drift.  Compares mu_{m,h}(pre-cutoff library) vs.
      the corresponding hourly mean over the 500-day post-cutoff
      actuals.  If the bias visible in the signed-error figure
      tracks (mu_pre - mean_post) by hour, the bias is (γ)
      climatology mis-match -- NOT a property of the dynamical
      estimator at all.

PROVENANCE-GRADE: INSPECTION-ONLY.

Inputs:
  - The frozen spec (experiment.freeze.load_verified)
  - load_actuals(cutoff=None) for the full demand series
  - The persisted predictor 2 backtest pickle for the signed-error
    overlay in (3)

Outputs:
  PDFs in writeup/tex/figs/ (NOT yet referenced from the draft;
  these support the analysis only):
    fig_bias_trajectory.pdf
    fig_bias_eigenstructure.pdf  (table-like)
    fig_bias_climatology.pdf

  Headlines printed to stdout for the discussion.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from edynamics.modelling_tools import Embedding, Lag
from config import load_config
from experiment import freeze
from experiment._actuals import (
    load_actuals, zscore_params, zscore_transform, mu_at, sigma_at,
)
from experiment.predict import _daytype
from processing.innovations.estimator import _local_fit_at

# uniform style (mirrors writeup/figures.py)
sns.set_theme(context="paper", style="whitegrid", font="serif", font_scale=0.95)
plt.rcParams.update({
    "figure.dpi":          150,
    "savefig.dpi":         300,
    "savefig.bbox":        "tight",
    "pdf.fonttype":        42,
})

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "writeup" / "tex" / "figs"
SAMP = ROOT / "scratch" / "data" / "smc_smap_samples"
PALETTE_DT = {"weekday": "#2c7fb8", "saturday": "#d95f0e", "sunday": "#7b3294"}


# ---------------------------------------------------------------------------
# shared setup
# ---------------------------------------------------------------------------
def _setup():
    cfg = load_config()
    spec = freeze.load_verified()
    cutoff = pd.Timestamp(spec["data_cutoff"])
    anchor_h = spec["predictor"]["day_anchor_hour"]
    clim_method = spec["predictor"].get("climatology_method", "month_hour")
    k_year = spec["predictor"].get("fourier_k_year")
    k_day  = spec["predictor"].get("fourier_k_day")
    zp = zscore_params(cutoff, method=clim_method, k_year=k_year, k_day=k_day)
    raw_full = load_actuals(cutoff=None)
    z_full = zscore_transform(raw_full, zp)
    return cfg, spec, cutoff, anchor_h, raw_full, z_full, zp


def _zscore_to_mw_at(t: pd.Timestamp, z: float, zp: dict) -> float:
    """Invert the z-transform: D = mu + sigma * z at clock time t."""
    idx = pd.DatetimeIndex([t])
    mu = float(mu_at(zp, idx)[0])
    sigma = float(sigma_at(zp, idx)[0])
    return mu + sigma * z


def _fit_global_C(z_series: pd.Series, daytype: str, anchor_h: int,
                  d: int) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Fit predictor 2's global linear C on the pre-cutoff library for
    one day-type. Returns (C, Sigma, block_df).

    Mirrors experiment.predict._build_pre_cutoff: embed on the full
    hourly z-series so the Lag observer has a defined frequency, then
    restrict the *library anchors* to the chosen day-type. This is the
    same construction as production.
    """
    df = z_series.to_frame("zscore").asfreq("h")
    lags = [Lag(variable_name="zscore", tau=-i) for i in range(d)]
    # Restrict anchors to the chosen day-type (after the d-step warm-up
    # and before the final hour, mirroring _build_pre_cutoff).
    all_lib = df.index[d:-1]
    dtags = all_lib.map(lambda t: _daytype(t, anchor_h))
    lib = all_lib[dtags == daytype]
    emb = Embedding(data=df, observers=lags, library_times=lib)
    emb.compile()
    blk = emb.block.dropna()
    X = blk.iloc[:-1].values
    Y = blk.iloc[1:].values
    # Plain OLS (predictor 2 is theta=0 by definition: uniform weights).
    C, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ C
    mu_r = resid.mean(axis=0)
    rc = resid - mu_r
    Sigma = rc.T @ rc / len(rc)
    return C, Sigma, blk


# ---------------------------------------------------------------------------
# (1) iterated trajectory
# ---------------------------------------------------------------------------
def fig_bias_trajectory() -> Path:
    cfg, spec, cutoff, anchor_h, raw_full, z_full, zp = _setup()
    daytypes = ("weekday", "saturday", "sunday")
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    headlines = []
    for dt in daytypes:
        d = int(cfg.embedding_dim(dt))
        C, _Sig, blk = _fit_global_C(z_full[z_full.index <= cutoff],
                                     dt, anchor_h, d)
        # Pick a representative midnight start state: the median of
        # the library's "midnight anchor" rows.  HE1 == hour-of-day 0
        # under cfg.data.day_anchor_hours=0.
        z_index = blk.index
        midnight_mask = z_index.hour == 0
        x0_z = np.median(blk[midnight_mask].values, axis=0)
        # Iterate the trajectory in z-space for 24 hours
        traj_z = np.zeros((24, d))
        x = x0_z.copy()
        for h in range(24):
            x = x @ C
            traj_z[h] = x
        # Convert coord 0 back to MW at the corresponding clock hour.
        # Use a representative date for the climatology -- pick the
        # mid-point of the post-cutoff scoring window so demand
        # magnitudes are comparable.  Actually use the same clock
        # hours: use one representative day's mu, sigma at HE1..HE24
        # for the day-type. We pick the median delivery_date in the
        # actuals over the post-cutoff window.
        post = raw_full[raw_full.index > cutoff]
        dt_post = post.index.map(lambda t: _daytype(t, anchor_h))
        post_dt = post[dt_post == dt]
        # representative day = median of unique calendar dates
        days = pd.Series(post_dt.index.normalize().unique())
        rep_day = pd.Timestamp(days.median()).normalize()
        # x0 is the state at anchor_time = midnight (rep_day + 0h).
        # After one iteration step `x = x @ C`, state is at midnight + 1h.
        # So traj_z[h] corresponds to clock time rep_day + (h+1) hours,
        # which is HE(h+2) under the HE1=00:00 convention (HE1 = 0:00,
        # HE2 = 1:00, ..., HE24 = 23:00).
        # Note: traj_z[0] is at 01:00 = HE2, traj_z[23] is at 00:00 next day = HE1.
        target_hours = [(h + 1) % 24 for h in range(24)]   # 1, 2, ..., 23, 0
        rep_hours = pd.DatetimeIndex(
            [rep_day + pd.Timedelta(hours=h + 1) for h in range(24)])
        traj_mw_raw = np.array([_zscore_to_mw_at(rep_hours[h], traj_z[h, 0], zp)
                                for h in range(24)])
        # Reorder so x-axis runs HE1..HE24 (== hour-of-day 0..23).
        order = np.argsort(target_hours)   # HE1 first (= hour 0)
        target_HE = np.array(target_hours)[order] + 1     # HE label
        traj_mw = traj_mw_raw[order]
        # Actual mean demand by hour-of-day, for this day-type, over
        # the full post-cutoff scoring window
        actual_by_hour = (post_dt.groupby(post_dt.index.hour).mean()
                          .reindex(range(24)).values)
        # Plot both: actual is hour-of-day 0..23 = HE1..HE24
        ax.plot(range(1, 25), actual_by_hour,
                color=PALETTE_DT[dt], linewidth=1.4,
                label=f"{dt} actual mean")
        ax.plot(target_HE, traj_mw,
                color=PALETTE_DT[dt], linewidth=1.4, linestyle="--",
                label=f"{dt} iterated trajectory")
        diff_peak = traj_mw - actual_by_hour
        headlines.append((dt,
                          f"peak diff +{diff_peak.max():.0f} at HE{target_HE[np.argmax(diff_peak)]}",
                          f"min diff {diff_peak.min():.0f} at HE{target_HE[np.argmin(diff_peak)]}",
                          f"corr={np.corrcoef(traj_mw, actual_by_hour)[0,1]:.3f}"))
    ax.set_xlabel("hour of delivery day (HE)")
    ax.set_ylabel("Ontario demand (MW)")
    ax.set_xticks([1, 5, 9, 13, 17, 21, 24])
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
              fontsize=7.5, frameon=False)
    fig.subplots_adjust(right=0.74)
    out = FIGS / "fig_bias_trajectory.pdf"
    fig.savefig(out)
    plt.close(fig)
    print("\n=== (1) iterated trajectory vs actual diurnal ===")
    for h in headlines:
        print(f"  {h[0]:>9s}  {h[1]:<32s}  {h[2]:<32s}  {h[3]}")
    return out


# ---------------------------------------------------------------------------
# (2) eigenstructure
# ---------------------------------------------------------------------------
def report_eigenstructure() -> None:
    cfg, spec, cutoff, anchor_h, raw_full, z_full, _zp = _setup()
    daytypes = ("weekday", "saturday", "sunday")
    print("\n=== (2) eigenstructure of global C(theta=0) ===")
    for dt in daytypes:
        d = int(cfg.embedding_dim(dt))
        C, _Sig, _blk = _fit_global_C(z_full[z_full.index <= cutoff],
                                      dt, anchor_h, d)
        # We applied x_next = x @ C (row convention).  Eigenvalues of
        # the operator are eigenvalues of C^T (left action, same spec
        # as C).
        evs = np.linalg.eigvals(C)
        evs_sorted = sorted(evs, key=lambda z: -abs(z))
        print(f"\n  {dt}  (d = {d})")
        for k, lam in enumerate(evs_sorted):
            mag = abs(lam)
            if abs(lam.imag) < 1e-9:
                print(f"    lambda_{k+1} = {lam.real:+.4f}             "
                      f"|lambda|={mag:.4f}")
            else:
                period_steps = 2 * np.pi / abs(np.angle(lam))
                print(f"    lambda_{k+1} = {lam.real:+.4f} {lam.imag:+.4f}i   "
                      f"|lambda|={mag:.4f}  period={period_steps:.2f} steps")


# ---------------------------------------------------------------------------
# (3) climatology drift
# ---------------------------------------------------------------------------
def fig_bias_climatology() -> Path:
    cfg, spec, cutoff, anchor_h, raw_full, _z_full, zp = _setup()
    # mu_{m,h} as evaluated on the post-cutoff timestamps -- these are
    # the demand-space "centring" values the inverse z-transform uses
    # when scoring.  Compare against the empirical hourly mean of post-
    # cutoff actuals.
    daytypes = ("weekday", "saturday", "sunday")
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2),
                              sharex=True, sharey=True)
    rows = []
    for ax, dt in zip(axes, daytypes):
        post = raw_full[raw_full.index > cutoff]
        dt_tags = post.index.map(lambda t: _daytype(t, anchor_h))
        sub = post[dt_tags == dt]
        # actual hourly mean over the population
        actual_hourly = (sub.groupby(sub.index.hour).mean()
                            .reindex(range(24)).values)
        # climatology mu evaluated at every post-cutoff timestamp
        mu_arr = mu_at(zp, sub.index)
        mu_series = pd.Series(mu_arr, index=sub.index)
        mu_hourly = (mu_series.groupby(mu_series.index.hour).mean()
                              .reindex(range(24)).values)
        diff = mu_hourly - actual_hourly         # +ve = climatology high
        ax.axhline(0, color="black", linewidth=0.5)
        ax.plot(range(1, 25), diff, color=PALETTE_DT[dt],
                marker="o", linewidth=1.3)
        ax.set_title(dt, fontsize=9)
        ax.set_xlabel("hour of delivery day (HE)")
        ax.set_xticks([1, 5, 9, 13, 17, 21, 24])
        rows.append((dt, diff))
    axes[0].set_ylabel("clim $\\mu_{m,h}$ $-$ actual mean (MW)\n"
                       "(+ve = clim above actual)")
    fig.subplots_adjust(wspace=0.12)
    out = FIGS / "fig_bias_climatology.pdf"
    fig.savefig(out)
    plt.close(fig)
    print("\n=== (3) climatology drift: mu_{m,h}(pre-cutoff library) vs post-cutoff actual mean ===")
    for dt, diff in rows:
        print(f"\n  {dt}")
        for h in [1, 6, 9, 12, 15, 18, 21, 24]:
            print(f"    HE{h:>2d}: mu - actual = {diff[h-1]:+7.0f} MW")
    return out


# ---------------------------------------------------------------------------
# (A) trajectory shape invariance across midnight starts
# ---------------------------------------------------------------------------
def fig_trajectory_shape_invariance() -> Path:
    """Iterate from MANY library-midnight starts (not just the median)
    and overlay the trajectories per day-type.  If the operator's
    spectrum sets the shape, all trajectories share that shape (modulo
    initial-state projection on the slow eigenmode = scale/offset).
    If shapes vary wildly, the spectral story is incomplete."""
    cfg, spec, cutoff, anchor_h, raw_full, z_full, zp = _setup()
    daytypes = ("weekday", "saturday", "sunday")
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.0),
                             sharex=True, sharey=True)
    summaries = []
    for ax, dt in zip(axes, daytypes):
        d = int(cfg.embedding_dim(dt))
        C, _Sig, blk = _fit_global_C(z_full[z_full.index <= cutoff],
                                     dt, anchor_h, d)
        midnight_mask = blk.index.hour == 0
        x0_pool = blk[midnight_mask].values     # (n_midnights, d)
        # Sample up to 60 midnights uniformly spread over the library
        n_show = min(60, len(x0_pool))
        idx = np.linspace(0, len(x0_pool) - 1, n_show).astype(int)
        # Representative day for the z->MW mapping
        post = raw_full[raw_full.index > cutoff]
        dt_post = post.index.map(lambda t: _daytype(t, anchor_h))
        days = pd.Series(post[dt_post == dt].index.normalize().unique())
        rep_day = pd.Timestamp(days.median()).normalize()
        # x0 anchor is midnight (rep_day + 0h). After step h, state is
        # at clock time (h+1) mod 24. traj_z[h] → HE((h+1) mod 24 + 1).
        target_hours = [(h + 1) % 24 for h in range(24)]
        order = np.argsort(target_hours)
        target_HE = np.array(target_hours)[order] + 1
        rep_hours = pd.DatetimeIndex(
            [rep_day + pd.Timedelta(hours=h + 1) for h in range(24)])
        # Centred shape: trajectory minus its own 24h mean (factors out
        # the initial-state offset along the slow eigenvector)
        shapes = np.zeros((n_show, 24))
        for k, j in enumerate(idx):
            x = x0_pool[j].copy()
            for h in range(24):
                x = x @ C
                shapes[k, h] = _zscore_to_mw_at(rep_hours[h], x[0], zp)
        # Reorder to HE1..HE24
        shapes_ordered = shapes[:, order]
        shapes_centred = shapes_ordered - shapes_ordered.mean(axis=1, keepdims=True)
        # Plot all centred trajectories
        for k in range(n_show):
            ax.plot(target_HE, shapes_centred[k],
                    color=PALETTE_DT[dt], alpha=0.15, linewidth=0.7)
        # Median centred trajectory
        med = np.median(shapes_centred, axis=0)
        ax.plot(target_HE, med, color=PALETTE_DT[dt], linewidth=1.8)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(dt, fontsize=9)
        ax.set_xlabel("hour of delivery day (HE)")
        ax.set_xticks([1, 5, 9, 13, 17, 21, 24])
        # Pairwise correlation of centred shapes (Pearson)
        corrs = np.corrcoef(shapes_centred)
        # mean off-diagonal
        mask = ~np.eye(n_show, dtype=bool)
        mean_corr = float(corrs[mask].mean())
        amp = float(shapes_centred.std(axis=1).mean())
        summaries.append((dt, n_show, mean_corr, amp))
    axes[0].set_ylabel("centred trajectory (MW)")
    fig.subplots_adjust(wspace=0.10)
    out = FIGS / "fig_trajectory_shape_invariance.pdf"
    fig.savefig(out)
    plt.close(fig)
    print("\n=== (A) trajectory shape invariance across midnight starts ===")
    print(f"  {'dt':>9s}  {'n_starts':>9s}  {'mean pairwise corr':>20s}  {'mean amp (MW)':>15s}")
    for dt, n, corr, amp in summaries:
        print(f"  {dt:>9s}  {n:>9d}  {corr:>20.3f}  {amp:>15.0f}")
    return out


# ---------------------------------------------------------------------------
# (B) Fourier content: iterated trajectory vs actual demand
# ---------------------------------------------------------------------------
def report_fourier_content() -> None:
    """Tabulate sin/cos amplitudes at k = 1, 2, 3 cycles/day for the
    iterated trajectory and for actual mean demand, per day-type.

    Definitions (24-point series):
      a_k = (2/24) * sum_h x[h] * cos(2*pi*k*h/24)
      b_k = (2/24) * sum_h x[h] * sin(2*pi*k*h/24)
      A_k = sqrt(a_k^2 + b_k^2)     (peak-to-mean amplitude in MW)
    """
    cfg, spec, cutoff, anchor_h, raw_full, z_full, zp = _setup()
    daytypes = ("weekday", "saturday", "sunday")
    ks = (1, 2, 3)

    def fft_amps(x: np.ndarray) -> dict:
        h = np.arange(24)
        out = {}
        for k in ks:
            a = (2.0 / 24) * (x * np.cos(2 * np.pi * k * h / 24)).sum()
            b = (2.0 / 24) * (x * np.sin(2 * np.pi * k * h / 24)).sum()
            out[k] = float(np.hypot(a, b))
        return out

    print("\n=== (B) Fourier amplitudes (MW) at k cycles/day ===")
    print(f"  {'dt':>9s}  {'source':>14s}     A_1       A_2       A_3")
    for dt in daytypes:
        d = int(cfg.embedding_dim(dt))
        C, _Sig, blk = _fit_global_C(z_full[z_full.index <= cutoff],
                                     dt, anchor_h, d)
        # iterated trajectory from the median midnight start, in MW
        midnight_mask = blk.index.hour == 0
        x0 = np.median(blk[midnight_mask].values, axis=0)
        post = raw_full[raw_full.index > cutoff]
        dt_post = post.index.map(lambda t: _daytype(t, anchor_h))
        days = pd.Series(post[dt_post == dt].index.normalize().unique())
        rep_day = pd.Timestamp(days.median()).normalize()
        # As in (1): traj_z[h] is at midnight + (h+1) hours. Order to HE1..HE24.
        target_hours = [(h + 1) % 24 for h in range(24)]
        order = np.argsort(target_hours)
        rep_hours = pd.DatetimeIndex(
            [rep_day + pd.Timedelta(hours=h + 1) for h in range(24)])
        traj_mw_raw = np.zeros(24)
        x = x0.copy()
        for h in range(24):
            x = x @ C
            traj_mw_raw[h] = _zscore_to_mw_at(rep_hours[h], x[0], zp)
        traj_mw = traj_mw_raw[order]   # HE1..HE24 order
        # actual mean demand by hour
        sub = post[dt_post == dt]
        actual_h = (sub.groupby(sub.index.hour).mean()
                       .reindex(range(24)).values)
        # report
        a_traj = fft_amps(traj_mw)
        a_act = fft_amps(actual_h)
        print(f"  {dt:>9s}  {'iterated traj':>14s}  "
              f"{a_traj[1]:>7.0f}   {a_traj[2]:>7.0f}   {a_traj[3]:>7.0f}")
        print(f"  {dt:>9s}  {'actual demand':>14s}  "
              f"{a_act[1]:>7.0f}   {a_act[2]:>7.0f}   {a_act[3]:>7.0f}")
        print(f"  {dt:>9s}  {'ratio iter/act':>14s}  "
              f"{a_traj[1]/max(a_act[1],1e-9):>7.2f}   "
              f"{a_traj[2]/max(a_act[2],1e-9):>7.2f}   "
              f"{a_traj[3]/max(a_act[3],1e-9):>7.2f}")
        print()


# ---------------------------------------------------------------------------
# (C) trajectory from different starting hours: does the shape shift?
# ---------------------------------------------------------------------------
def fig_trajectory_starting_hour() -> Path:
    """Iterate from 4 different starting hours (HE0, HE6, HE12, HE18)
    for each day-type.  If the iterated trajectory is a 'time-reversed
    echo' of the lag coordinates, starting at HE12 (lags carry the
    morning ramp) should produce a different-shaped trajectory from
    starting at HE0 (lags carry the prior evening cycle).  The shift
    in shape across start hours diagnoses the imprint mechanism."""
    cfg, spec, cutoff, anchor_h, raw_full, z_full, zp = _setup()
    daytypes = ("weekday", "saturday", "sunday")
    start_hours = (0, 6, 12, 18)
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.0),
                             sharex=True, sharey=True)
    colours = sns.color_palette("viridis", n_colors=len(start_hours))
    for ax, dt in zip(axes, daytypes):
        d = int(cfg.embedding_dim(dt))
        C, _Sig, blk = _fit_global_C(z_full[z_full.index <= cutoff],
                                     dt, anchor_h, d)
        post = raw_full[raw_full.index > cutoff]
        dt_post = post.index.map(lambda t: _daytype(t, anchor_h))
        days = pd.Series(post[dt_post == dt].index.normalize().unique())
        rep_day = pd.Timestamp(days.median()).normalize()
        for c, sh in zip(colours, start_hours):
            # library starts at clock hour sh (HE(sh+1) in HE convention)
            mask = blk.index.hour == sh
            if mask.sum() == 0:
                continue
            x0 = np.median(blk[mask].values, axis=0)
            # x0 anchor is at clock-hour sh; after step h the state is
            # at hour (sh + h + 1) mod 24 → HE label = that+1
            rep_hours = pd.DatetimeIndex(
                [rep_day + pd.Timedelta(hours=sh + h + 1) for h in range(24)])
            traj_mw_raw = np.zeros(24)
            x = x0.copy()
            for h in range(24):
                x = x @ C
                traj_mw_raw[h] = _zscore_to_mw_at(rep_hours[h], x[0], zp)
            # centre so we compare shape
            centred_raw = traj_mw_raw - traj_mw_raw.mean()
            target_HE = [((sh + h + 1) % 24) + 1 for h in range(24)]
            order = np.argsort(target_HE)
            ax.plot(np.array(target_HE)[order], centred_raw[order],
                    color=c, linewidth=1.4, marker="o", markersize=3,
                    label=f"start HE{sh+1}")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(dt, fontsize=9)
        ax.set_xlabel("target hour (HE)")
        ax.set_xticks([1, 5, 9, 13, 17, 21, 24])
    axes[0].set_ylabel("centred iterated trajectory (MW)")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
                    fontsize=7.5, frameon=False)
    fig.subplots_adjust(right=0.82, wspace=0.10)
    out = FIGS / "fig_trajectory_starting_hour.pdf"
    fig.savefig(out)
    plt.close(fig)
    print("\n=== (C) trajectory shape vs starting hour ===")
    print("  See fig_trajectory_starting_hour.pdf.")
    print("  Prediction: if 'time-reversed echo' is the mechanism, the")
    print("  trajectory's hour-of-peak should shift with the start hour.")
    return out


# ---------------------------------------------------------------------------
# (D) trajectory in z-space (no climatology mapping)
# ---------------------------------------------------------------------------
def fig_trajectory_zspace() -> Path:
    """Re-plot the iterated trajectory in z-space, BEFORE inverse-z
    mapping back to MW.  This factors out the climatology mu_{m,h}.
    If the z-space trajectory is small / mostly flat, the diurnal
    shape we see in MW is the climatology talking through sigma_{m,h}.
    If the z-space trajectory has structure, the C operator is doing
    real work."""
    cfg, spec, cutoff, anchor_h, raw_full, z_full, zp = _setup()
    daytypes = ("weekday", "saturday", "sunday")
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.0),
                             sharex=True, sharey=True)
    for ax, dt in zip(axes, daytypes):
        d = int(cfg.embedding_dim(dt))
        C, _Sig, blk = _fit_global_C(z_full[z_full.index <= cutoff],
                                     dt, anchor_h, d)
        midnight_mask = blk.index.hour == 0
        x0_pool = blk[midnight_mask].values
        n_show = min(60, len(x0_pool))
        idx = np.linspace(0, len(x0_pool) - 1, n_show).astype(int)
        z_trajs = np.zeros((n_show, 24))
        for k, j in enumerate(idx):
            x = x0_pool[j].copy()
            for h in range(24):
                x = x @ C
                z_trajs[k, h] = x[0]
        # Midnight anchor → step h is at clock-hour (h+1) mod 24 → HE((h+1)%24+1)
        target_hours = [(h + 1) % 24 for h in range(24)]
        order = np.argsort(target_hours)
        target_HE = np.array(target_hours)[order] + 1
        z_ord = z_trajs[:, order]
        # Plot all + median
        for k in range(n_show):
            ax.plot(target_HE, z_ord[k],
                    color=PALETTE_DT[dt], alpha=0.12, linewidth=0.7)
        ax.plot(target_HE, np.median(z_ord, axis=0),
                color=PALETTE_DT[dt], linewidth=1.8)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(dt, fontsize=9)
        ax.set_xlabel("target HE")
        ax.set_xticks([1, 5, 9, 13, 17, 21, 24])
    axes[0].set_ylabel("iterated trajectory in z-space")
    fig.subplots_adjust(wspace=0.10)
    out = FIGS / "fig_trajectory_zspace.pdf"
    fig.savefig(out)
    plt.close(fig)
    # headline
    print("\n=== (D) iterated trajectory in z-space ===")
    print(f"  {'dt':>9s}  {'z-traj amp (std)':>18s}  {'z-traj range':>15s}  {'z-traj final h=24':>18s}")
    for dt in daytypes:
        d = int(cfg.embedding_dim(dt))
        C, _Sig, blk = _fit_global_C(z_full[z_full.index <= cutoff],
                                     dt, anchor_h, d)
        midnight_mask = blk.index.hour == 0
        x0_pool = blk[midnight_mask].values
        x0 = np.median(x0_pool, axis=0)
        z_traj = np.zeros(24)
        x = x0.copy()
        for h in range(24):
            x = x @ C
            z_traj[h] = x[0]
        print(f"  {dt:>9s}  {z_traj.std():>18.4f}  {z_traj.ptp():>15.4f}  {z_traj[-1]:>18.4f}")
    return out


# ---------------------------------------------------------------------------
def main() -> None:
    p1 = fig_bias_trajectory()
    print(f"\n  wrote {p1.relative_to(ROOT)}")
    report_eigenstructure()
    p3 = fig_bias_climatology()
    print(f"\n  wrote {p3.relative_to(ROOT)}")
    pA = fig_trajectory_shape_invariance()
    print(f"\n  wrote {pA.relative_to(ROOT)}")
    report_fourier_content()
    pC = fig_trajectory_starting_hour()
    print(f"\n  wrote {pC.relative_to(ROOT)}")
    pD = fig_trajectory_zspace()
    print(f"\n  wrote {pD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
