"""EXPLORATORY: optimal window length for the mu_{m,h} climatology.

Sweep W in {0.5, 1, 2, 3, 5, 10, 20} years.  For each W:

  - Fit (mu_{m,h}, sigma_{m,h}) on the W years immediately BEFORE the
    last 6 months of pre-cutoff data (the training slice).
  - Evaluate four metrics on the last 6 months of pre-cutoff data
    (the holdout):

      (i)   held-out NLL = -Sigma log N(y_t; mu(t), sigma(t)^2)
            sum over every observation in the holdout
      (ii)  AIC = 2 * NLL + 2 * p, p = number of (m,h) bins used
      (iii) rolling stability of mu_{m,h}: variance of mu(m,h) across
            non-overlapping W-year windows ending at successive months
            of the pre-cutoff history. Lower = more stable.
      (iv)  residual diurnal-z amplitude on the holdout: max_h(mean z_h)
            - min_h(mean z_h).  Directly measures the bias source.

Strict pre-cutoff: never touches post-cutoff actuals.

Output: a printed table of (W, NLL, AIC, stability, diurnal_amp) and
a PDF plot of each metric vs W.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from experiment import freeze
from experiment._actuals import load_actuals

sns.set_theme(context="paper", style="whitegrid", font="serif", font_scale=0.95)
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "scratch" / "data" / "climatology_window"
OUT.mkdir(parents=True, exist_ok=True)

# Holdout = last 12 months of pre-cutoff data (covers all month bins).
HOLDOUT_MONTHS = 12
WINDOWS_YEARS = (1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0)


# ---------------------------------------------------------------------------
def _fit_mh(series: pd.Series) -> tuple[pd.Series, pd.Series, int]:
    """Fit mu_{m,h}, sigma_{m,h} on a slice of raw demand.

    Returns (mu_table, sigma_table, p) where the tables are indexed by
    (month, hour) and p = number of populated bins.
    """
    df = pd.DataFrame({
        "y":     series.values,
        "month": series.index.month,
        "hour":  series.index.hour,
    })
    g = df.groupby(["month", "hour"])
    mu = g["y"].mean()
    # ddof=0 for ML estimate of sigma^2
    sd = g["y"].std(ddof=0)
    # Avoid sigma -> 0 (numerical floor at 1 MW; in practice sigma >> 1)
    sd = sd.where(sd > 1.0, 1.0)
    return mu, sd, int(len(mu))


def _eval_on(holdout: pd.Series, mu: pd.Series, sd: pd.Series) -> dict:
    """Evaluate climatology (mu, sd) on a holdout series. Returns
    dict with nll (total), n, missing_bins, z_residual."""
    df = pd.DataFrame({
        "y":     holdout.values,
        "month": holdout.index.month,
        "hour":  holdout.index.hour,
    })
    # Lookup: bring mu, sd into df by (month, hour)
    mu_lookup = pd.DataFrame({"mu": mu}).reset_index()
    sd_lookup = pd.DataFrame({"sd": sd}).reset_index()
    df = df.merge(mu_lookup, on=["month", "hour"], how="left")
    df = df.merge(sd_lookup, on=["month", "hour"], how="left")
    missing = df["mu"].isna().sum()
    df = df.dropna(subset=["mu", "sd"])
    # NLL = sum [ log(sigma) + 0.5 (y-mu)^2 / sigma^2 + 0.5 log(2 pi) ]
    half_log_2pi = 0.5 * np.log(2 * np.pi)
    z = (df["y"] - df["mu"]) / df["sd"]
    nll = float((np.log(df["sd"]) + 0.5 * z ** 2 + half_log_2pi).sum())
    return {
        "nll":          nll,
        "n":            int(len(df)),
        "missing_bins": int(missing),
        "z_resid":      z.to_numpy(),
        "df":           df.assign(z=z),
    }


def _diurnal_amplitude(z_series_with_hours: pd.DataFrame) -> float:
    """Compute the residual diurnal-z amplitude: max_h(mean z) - min_h(mean z)."""
    by_h = z_series_with_hours.groupby("hour")["z"].mean()
    return float(by_h.max() - by_h.min())


def _rolling_stability(raw: pd.Series, W_years: float,
                       n_windows: int = 10) -> float:
    """Across n non-overlapping W-year windows ending at evenly-spaced
    times across the pre-cutoff history, compute the standard deviation
    of mu_{m,h} across windows, averaged over (m,h) bins. Low = stable.

    Returns NaN if fewer than 2 windows fit (the dataset isn't long
    enough to host multiple non-overlapping W-year windows).
    """
    W_td = pd.Timedelta(days=int(W_years * 365.25))
    span = raw.index[-1] - raw.index[0]
    n_possible = int(span / W_td)
    if n_possible < 2:
        return float("nan")
    n_use = min(n_windows, n_possible)
    # Place n_use end-points evenly across [first + W_td, last]
    end_dates = pd.date_range(start=raw.index[0] + W_td,
                              end=raw.index[-1],
                              periods=n_use)
    mu_tables = []
    for end in end_dates:
        sub = raw[(raw.index > end - W_td) & (raw.index <= end)]
        if len(sub) < 24 * 30:
            continue
        mu, _sd, _p = _fit_mh(sub)
        mu_tables.append(mu)
    if len(mu_tables) < 2:
        return float("nan")
    common = mu_tables[0].index
    for t in mu_tables[1:]:
        common = common.intersection(t.index)
    arr = np.array([t.loc[common].values for t in mu_tables])
    return float(arr.std(axis=0).mean())


# ---------------------------------------------------------------------------
def sweep() -> pd.DataFrame:
    spec = freeze.load_verified()
    cutoff = pd.Timestamp(spec["data_cutoff"])
    raw_full = load_actuals(cutoff=None).dropna()
    pre = raw_full[raw_full.index <= cutoff]

    holdout_start = cutoff - pd.DateOffset(months=HOLDOUT_MONTHS)
    holdout = pre[pre.index > holdout_start]
    training_pool = pre[pre.index <= holdout_start]

    print(f"  pre-cutoff data range: {pre.index.min().date()} .. {pre.index.max().date()}")
    print(f"  holdout (last {HOLDOUT_MONTHS} months pre-cutoff): "
          f"{holdout.index.min().date()} .. {holdout.index.max().date()}  (n={len(holdout):,d})")
    print(f"  training pool ends at: {training_pool.index.max().date()}")
    print()

    rows = []
    for W in WINDOWS_YEARS:
        W_td = pd.Timedelta(days=int(W * 365.25))
        train = training_pool[
            training_pool.index > training_pool.index.max() - W_td]
        mu, sd, p = _fit_mh(train)
        ev = _eval_on(holdout, mu, sd)
        amp = _diurnal_amplitude(ev["df"])
        stab = _rolling_stability(training_pool, W)
        aic = 2 * ev["nll"] + 2 * p
        rows.append({
            "W_years":      W,
            "n_train":      len(train),
            "p_bins":       p,
            "missing_bins_on_holdout": ev["missing_bins"],
            "nll":          ev["nll"],
            "nll_per_pt":   ev["nll"] / ev["n"],
            "aic":          aic,
            "stability_mw": stab,
            "diurnal_amp":  amp,
        })
    return pd.DataFrame(rows)


def fig_metrics(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.5))
    cols = [
        ("nll_per_pt",   "held-out NLL per observation",  axes[0, 0]),
        ("aic",          "AIC (lower = better)",           axes[0, 1]),
        ("stability_mw", "rolling-fit stability  (MW, lower = more stable)",
         axes[1, 0]),
        ("diurnal_amp",  "residual diurnal-z amplitude  (z units)",
         axes[1, 1]),
    ]
    for col, lab, ax in cols:
        ax.plot(df["W_years"], df[col], color="#2c7fb8",
                marker="o", linewidth=1.4)
        # Mark argmin
        amin = df[col].idxmin()
        Wmin = df.loc[amin, "W_years"]
        ax.axvline(Wmin, color="#d95f0e", linestyle="--", linewidth=0.8,
                   alpha=0.7, label=f"min at W={Wmin}y")
        ax.set_xscale("log")
        ax.set_xlabel("window length $W$ (years, log)")
        ax.set_ylabel(lab)
        ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    out = OUT / "metrics.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
def main() -> None:
    df = sweep()
    print(f"  {'W (y)':>6s}  {'n_train':>9s}  {'p_bins':>7s}  "
          f"{'NLL/pt':>9s}  {'AIC':>12s}  {'stab MW':>9s}  {'diurn amp':>10s}")
    for _, r in df.iterrows():
        print(f"  {r.W_years:>6.1f}  {r.n_train:>9,.0f}  {int(r.p_bins):>7d}  "
              f"{r.nll_per_pt:>9.3f}  {r.aic:>12,.0f}  "
              f"{r.stability_mw:>9.0f}  {r.diurnal_amp:>10.3f}")
    print()
    for col, name in [("nll", "held-out NLL"),
                       ("aic", "AIC"),
                       ("stability_mw", "rolling stability"),
                       ("diurnal_amp", "diurnal-z amplitude")]:
        amin = df[col].idxmin()
        Wmin = df.loc[amin, "W_years"]
        print(f"  argmin {name:>20s}: W = {Wmin} years   value = {df.loc[amin, col]:.4f}")
    out_csv = OUT / "sweep.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n  wrote {out_csv.relative_to(ROOT)}")
    out_pdf = fig_metrics(df)
    print(f"  wrote {out_pdf.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
