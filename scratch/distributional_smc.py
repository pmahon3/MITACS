"""EXPLORATORY: distributional diagnostics on the persisted SMC samples.

Inputs are the per-cell sample arrays from `backtest_smc.py --save-samples`:
  scratch/data/smc_smap_samples/samples_<cell>.npz  with keys
    samples         (n_days, M, H) float32   -- de-z-scored MW per particle
    delivery_dates  (n_days,)      int64 ns

Outputs four kinds of diagnostic that the point-MAE / single-PI summary in
the backtest pickle cannot show:

  (i)   multi-sigma interval coverage at nominal {50,68,80,90,95,99}%
        from the M=200 sample quantiles, per (cell, horizon, daytype).
        Diagnoses tail miscalibration even when the central (~68%)
        coverage looks reasonable.

  (ii)  PIT histograms.  For each (target, horizon) the probability-
        integral transform is
            PIT = (rank of actual among M samples + U[0,1]) / (M+1)
        a randomised mid-rank correction for ties / discreteness.
        Pooled within (cell, daytype) and within (cell, horizon).
        Uniform == perfectly calibrated; U-shape == under-dispersed;
        inverted-U == over-dispersed; skewed == biased.

  (iii) Ensemble shape per (cell, horizon, daytype):
            mean of sample std (predictive spread),
            mean of sample skewness,
            mean of sample excess kurtosis.
        Diagnoses whether the M-particle distribution captures the
        non-Gaussian shape that the rebaseline (mitacs-rebaseline-facts)
        showed for the one-step innovation (excess kurt ~24-33).

  (iv)  CRPS, the proper scoring rule for the ensemble forecast.  We
        use the sample-quantile form
            CRPS_emp = (1/M) sum_i |x_i - y| - (1/(2 M^2)) sum_{i,j} |x_i - x_j|
        with the unbiased variance correction factor M/(M-1) applied to
        the second term (Zamo & Naveau 2018).  Lower = better; reported
        per (cell, horizon, daytype) and pooled.

Results are saved as CSVs in `scratch/data/smc_smap_samples/distrib/` and
printed as small tables.  PIT histograms are saved as PNGs (matplotlib Agg)
so the script is headless-safe.

PROVENANCE-GRADE: INSPECTION-ONLY.  These are diagnostics on the same
samples already produced under the production SMC code path; they do
not refit the estimator or rerun the SMC.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# nominal central coverage levels we test.  68% is the "1-sigma" band
# already reported by the backtest; the others widen progressively.
NOMINAL_LEVELS = (0.50, 0.6827, 0.80, 0.90, 0.95, 0.99)


# four SMC cells produced by backtest_smc.py --save-samples
CELLS = [
    ("C", "smc_global_gauss", "SMC global Gaussian"),
    ("D", "smc_global_emp",   "SMC global empirical"),
    ("E", "smc_prod_gauss",   "SMC prod Gaussian"),
    ("F", "smc_prod_emp",     "SMC prod empirical"),
]


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def _load_cell(samples_dir: Path, code: str, tag: str) -> tuple[np.ndarray, pd.DatetimeIndex, pd.DataFrame]:
    """Returns (samples, dates, summary_df) aligned by delivery_date."""
    npz = np.load(samples_dir / f"samples_{code}_{tag}.npz", allow_pickle=False)
    samples = npz["samples"]                                   # (D, M, H) float32
    dates = pd.to_datetime(npz["delivery_dates"]).normalize()  # (D,)
    pkl = pd.read_pickle(samples_dir / f"cell_{code}_{tag}.pkl")
    pkl["delivery_date"] = pkl["delivery_date"].dt.normalize()
    return samples, dates, pkl


def _align_actuals(samples: np.ndarray,
                   dates: pd.DatetimeIndex,
                   pkl: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Pivot the summary pickle to (D, H) actuals and daytype, in `dates` order."""
    actual = (pkl.pivot_table(index="delivery_date", columns="horizon_h",
                              values="actual_mw", aggfunc="first")
                .sort_index())
    actual = actual.reindex(dates).to_numpy()                  # (D, H) int64 -> float
    dt_first = pkl.sort_values("horizon_h").groupby("delivery_date")["daytype"].first()
    daytypes = dt_first.reindex(dates).to_numpy()              # (D,)
    return actual.astype(np.float64), daytypes


# ---------------------------------------------------------------------------
# multi-sigma coverage
# ---------------------------------------------------------------------------
def coverage_table(samples: np.ndarray, actual: np.ndarray,
                   daytypes: np.ndarray) -> pd.DataFrame:
    """Empirical interval coverage at each nominal level, per (daytype, horizon).

    samples : (D, M, H)
    actual  : (D, H)
    """
    D, M, H = samples.shape
    rows = []
    # broadcast actuals to (D, 1, H) for comparison
    a = actual[:, None, :]
    for level in NOMINAL_LEVELS:
        alpha = (1.0 - level) / 2.0
        lo = np.quantile(samples, alpha,        axis=1)        # (D, H)
        hi = np.quantile(samples, 1.0 - alpha,  axis=1)        # (D, H)
        hit = ((actual >= lo) & (actual <= hi))                # (D, H)
        # per-horizon
        for h in range(H):
            for dt in np.unique(daytypes):
                mask = daytypes == dt
                hit_h = hit[mask, h]
                if len(hit_h) == 0:
                    continue
                rows.append({
                    "level":   level,
                    "daytype": str(dt),
                    "horizon": h + 1,
                    "n_days":  int(len(hit_h)),
                    "cov_emp": float(hit_h.mean()),
                    "se":      float(np.sqrt(hit_h.mean() * (1 - hit_h.mean()) / len(hit_h))),
                })
            mask_all = slice(None)
            hit_h = hit[mask_all, h]
            rows.append({
                "level":   level,
                "daytype": "all",
                "horizon": h + 1,
                "n_days":  int(len(hit_h)),
                "cov_emp": float(hit_h.mean()),
                "se":      float(np.sqrt(hit_h.mean() * (1 - hit_h.mean()) / len(hit_h))),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# PIT
# ---------------------------------------------------------------------------
def pit_values(samples: np.ndarray, actual: np.ndarray,
               rng: np.random.Generator) -> np.ndarray:
    """Randomised mid-rank PIT.  Returns (D, H) in [0, 1]."""
    D, M, H = samples.shape
    # rank of actual among M samples (number strictly less + ties handled by U)
    less = (samples < actual[:, None, :]).sum(axis=1)          # (D, H)
    eq   = (samples == actual[:, None, :]).sum(axis=1)         # (D, H)
    u = rng.uniform(size=(D, H))
    pit = (less + u * (eq + 1.0)) / (M + 1.0)
    return pit


def pit_histogram(pit_flat: np.ndarray, n_bins: int = 20,
                  title: str = "", out_path: Path | None = None) -> dict:
    """Save a PIT histogram PNG and return summary stats.

    Returns dict with mean, std, chi2 (uniformity), n.
    """
    pit_flat = pit_flat[np.isfinite(pit_flat)]
    n = len(pit_flat)
    if n == 0:
        return {"n": 0}
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    counts, _ = np.histogram(pit_flat, bins=edges)
    expected = n / n_bins
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    if out_path is not None:
        fig, ax = plt.subplots(figsize=(4.2, 3.0))
        ax.bar((edges[:-1] + edges[1:]) / 2, counts / n,
               width=1.0 / n_bins, edgecolor="black", linewidth=0.5)
        ax.axhline(1.0 / n_bins, color="red", linestyle="--", linewidth=0.8,
                   label="uniform")
        ax.set_xlim(0, 1)
        ax.set_xlabel("PIT")
        ax.set_ylabel("density")
        ax.set_title(f"{title}  n={n}  chi2={chi2:.1f}")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
    return {
        "n":    int(n),
        "mean": float(pit_flat.mean()),
        "std":  float(pit_flat.std(ddof=1)),
        "chi2": chi2,
    }


# ---------------------------------------------------------------------------
# ensemble shape (sample-moment stats per anchor, then averaged)
# ---------------------------------------------------------------------------
def shape_stats(samples: np.ndarray, daytypes: np.ndarray) -> pd.DataFrame:
    """Mean of per-anchor sample std / skew / excess-kurt, per (daytype, h)."""
    D, M, H = samples.shape
    # per-anchor moments
    mu  = samples.mean(axis=1)                                 # (D, H)
    s2  = samples.var(axis=1, ddof=1)                          # (D, H)
    sd  = np.sqrt(s2)                                          # (D, H)
    # skew / excess kurt via central moments (Fisher's g1, g2)
    z = (samples - mu[:, None, :]) / np.maximum(sd[:, None, :], 1e-12)
    g1 = (z ** 3).mean(axis=1)                                 # (D, H)
    g2 = (z ** 4).mean(axis=1) - 3.0                           # (D, H)
    rows = []
    for h in range(H):
        for dt in list(np.unique(daytypes)) + ["all"]:
            mask = slice(None) if dt == "all" else (daytypes == dt)
            if not np.any(mask) if isinstance(mask, np.ndarray) else False:
                continue
            rows.append({
                "daytype":     str(dt),
                "horizon":     h + 1,
                "n_days":      int(np.sum(mask) if isinstance(mask, np.ndarray) else D),
                "mean_sd":     float(sd[mask, h].mean()),
                "mean_skew":   float(g1[mask, h].mean()),
                "mean_exkurt": float(g2[mask, h].mean()),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CRPS (sample / ensemble form, unbiased)
# ---------------------------------------------------------------------------
def crps_table(samples: np.ndarray, actual: np.ndarray,
               daytypes: np.ndarray) -> pd.DataFrame:
    """Per-anchor CRPS averaged by (daytype, horizon).

    sample CRPS = mean_i |x_i - y| - (1/(2 M (M-1))) sum_{i!=j} |x_i - x_j|
    """
    D, M, H = samples.shape
    # term 1: average |x_i - y|
    term1 = np.abs(samples - actual[:, None, :]).mean(axis=1)  # (D, H)
    # term 2: (1 / (M (M-1))) sum_{i<j} |x_i - x_j|   (no double count)
    # sort along M; closed-form: sum_{i<j} |x_i-x_j| = sum_k (2 k - M - 1) x_(k+1)
    xs = np.sort(samples, axis=1)                              # (D, M, H)
    k = np.arange(1, M + 1).reshape(1, M, 1)
    weights = (2 * k - M - 1).astype(np.float64)
    pair_sum = (weights * xs).sum(axis=1)                      # (D, H)
    term2 = pair_sum / (M * (M - 1))                            # one count per pair
    crps = term1 - term2                                       # (D, H)
    rows = []
    for h in range(H):
        for dt in list(np.unique(daytypes)) + ["all"]:
            mask = slice(None) if dt == "all" else (daytypes == dt)
            n = int(np.sum(mask) if isinstance(mask, np.ndarray) else D)
            if n == 0:
                continue
            rows.append({
                "daytype":  str(dt),
                "horizon":  h + 1,
                "n_days":   n,
                "crps_mw":  float(crps[mask, h].mean()),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------
def main(samples_dir: Path, out_dir: Path, seed: int = 0xC0FFEE) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    coverage_frames, pit_summary_rows, shape_frames, crps_frames = [], [], [], []
    headline_rows = []
    for code, tag, label in CELLS:
        print(f"\n=== cell {code} ({label}) ===")
        samples, dates, pkl = _load_cell(samples_dir, code, tag)
        actual, daytypes = _align_actuals(samples, dates, pkl)
        D, M, H = samples.shape
        print(f"  shape (D, M, H) = ({D}, {M}, {H})  daytypes: "
              f"{dict(zip(*np.unique(daytypes, return_counts=True)))}")

        # ---- (i) multi-sigma coverage ----
        cov = coverage_table(samples, actual, daytypes)
        cov["cell"] = code
        coverage_frames.append(cov)

        # 68.27% all-horizon all-daytype headline (should match backtest pickle)
        h_all = cov[(cov["level"] > 0.68) & (cov["level"] < 0.69) & (cov["daytype"] == "all")]
        head_cov = h_all["cov_emp"].mean()
        print(f"  68% coverage  (all daytypes, mean over h) = {head_cov:.3f}")

        # ---- (ii) PIT ----
        pit = pit_values(samples, actual, rng)                 # (D, H)
        # pooled all
        stats = pit_histogram(pit.ravel(), title=f"PIT cell {code} ({label})",
                              out_path=plot_dir / f"pit_{code}_{tag}_all.png")
        stats.update({"cell": code, "stratum": "all", "stratum_value": "all"})
        pit_summary_rows.append(stats)
        # per daytype
        for dt in np.unique(daytypes):
            mask = daytypes == dt
            stats = pit_histogram(pit[mask].ravel(),
                                  title=f"PIT cell {code} ({label}) — {dt}",
                                  out_path=plot_dir / f"pit_{code}_{tag}_{dt}.png")
            stats.update({"cell": code, "stratum": "daytype", "stratum_value": str(dt)})
            pit_summary_rows.append(stats)
        # per horizon (one figure with 24 mini-axes)
        for h in (1, 6, 12, 18, 24):
            stats = pit_histogram(pit[:, h - 1],
                                  title=f"PIT cell {code} h={h}",
                                  out_path=plot_dir / f"pit_{code}_{tag}_h{h:02d}.png")
            stats.update({"cell": code, "stratum": "horizon", "stratum_value": f"h={h}"})
            pit_summary_rows.append(stats)

        # ---- (iii) ensemble shape ----
        shape = shape_stats(samples, daytypes)
        shape["cell"] = code
        shape_frames.append(shape)
        # h=1 headline
        s1 = shape[(shape["horizon"] == 1) & (shape["daytype"] == "all")]
        if len(s1):
            print(f"  ensemble shape  h=1 (all)  sd={s1.iloc[0].mean_sd:6.1f}  "
                  f"skew={s1.iloc[0].mean_skew:+.3f}  "
                  f"exkurt={s1.iloc[0].mean_exkurt:+.3f}")

        # ---- (iv) CRPS ----
        crps = crps_table(samples, actual, daytypes)
        crps["cell"] = code
        crps_frames.append(crps)
        c_all = crps[crps["daytype"] == "all"]
        if len(c_all):
            print(f"  CRPS  (all daytypes, mean over h) = {c_all['crps_mw'].mean():6.1f} MW")

        headline_rows.append({
            "cell":  code,
            "label": label,
            "cov_68_all":         float(head_cov),
            "crps_mean_all":      float(c_all["crps_mw"].mean()) if len(c_all) else np.nan,
            "exkurt_h1_all":      float(s1.iloc[0].mean_exkurt) if len(s1) else np.nan,
        })

    # save big CSVs
    pd.concat(coverage_frames, ignore_index=True).to_csv(out_dir / "coverage_multisigma.csv", index=False)
    pd.DataFrame(pit_summary_rows).to_csv(out_dir / "pit_summary.csv", index=False)
    pd.concat(shape_frames,    ignore_index=True).to_csv(out_dir / "shape_stats.csv", index=False)
    pd.concat(crps_frames,     ignore_index=True).to_csv(out_dir / "crps.csv", index=False)
    pd.DataFrame(headline_rows).to_csv(out_dir / "headline.csv", index=False)

    print("\n" + "=" * 70)
    print("HEADLINE distributional summary")
    print("=" * 70)
    hdf = pd.DataFrame(headline_rows)
    print(hdf.to_string(index=False,
                        formatters={
                            "cov_68_all":   "{:.3f}".format,
                            "crps_mean_all":"{:6.1f}".format,
                            "exkurt_h1_all":"{:+.3f}".format,
                        }))

    # cross-cell multi-sigma table (pooled all daytypes, mean over h)
    print("\nMulti-sigma coverage (pooled all daytypes, mean over h)")
    full = pd.concat(coverage_frames, ignore_index=True)
    pivot = (full[full["daytype"] == "all"]
             .groupby(["cell", "level"])["cov_emp"].mean().unstack("level"))
    print(pivot.to_string(float_format=lambda x: f"{x:.3f}"))

    print(f"\nWrote CSVs and {len(list(plot_dir.glob('*.png')))} PNGs to {out_dir}")


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--samples-dir", type=Path,
                   default=Path("scratch/data/smc_smap_samples"),
                   help="Directory containing samples_*.npz and cell_*.pkl")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output dir (default: <samples-dir>/distrib)")
    p.add_argument("--seed", type=int, default=0xC0FFEE,
                   help="RNG seed for the PIT mid-rank correction")
    return p.parse_args()


if __name__ == "__main__":
    args = _cli()
    out_dir = args.out_dir or (args.samples_dir / "distrib")
    main(args.samples_dir, out_dir, seed=args.seed)
