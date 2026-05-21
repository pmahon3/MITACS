"""Fourier climatology preprocessing (Path A from diagnose_fourier_vs_month_hour.py).

Replaces the (month, hour) lookup-table climatology with a smooth
Fourier basis in (day-of-year, hour-of-day). The basis has K_YEAR
harmonics in day-of-year (annual + sub-annual periodicity) and
K_DAY harmonics in hour-of-day. No interactions; no year-trend
term (the multi-decade demand drift is left as a residual property,
handled implicitly by the local-linear forecaster's lag state).

Configured at the diagnostic-selected optimum K_year=8, K_day=8:
33 parameters total vs the (month, hour) lookup's 288.

Diagnostic findings (commit 6e73028):
  - per-month |bias| at first-of-month 00:00: 0.21 vs 0.42 (-49%)
  - per-boundary RMS jump |z[0] - z[-1]|: 0.18 vs 0.80 (-78%)
  - in-sample var(z): 0.974 vs 0.998 (parsimony cost)

This module is scratch-side: production experiment._actuals stays
untouched until A's empirical payoff confirms it's worth the
promotion. Used by sibling scratch scripts (k_capture_fourier.py,
s_capture_fourier.py) for the A vs B head-to-head.

PROVENANCE-GRADE: INSPECTION-ONLY.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

K_YEAR = 8
K_DAY = 8


def _fourier_design(idx: pd.DatetimeIndex, k_year: int = K_YEAR,
                    k_day: int = K_DAY) -> np.ndarray:
    """Design matrix: 1 + 2*k_year (day-of-year cos/sin pairs)
    + 2*k_day (hour-of-day cos/sin pairs)."""
    d = (idx.dayofyear.values.astype(float) - 1.0) / 365.25
    h = idx.hour.values.astype(float) / 24.0
    n = len(idx)
    cols = [np.ones(n)]
    for k in range(1, k_year + 1):
        cols.append(np.cos(2 * np.pi * k * d))
        cols.append(np.sin(2 * np.pi * k * d))
    for j in range(1, k_day + 1):
        cols.append(np.cos(2 * np.pi * j * h))
        cols.append(np.sin(2 * np.pi * j * h))
    return np.column_stack(cols)


@dataclass(frozen=True)
class FourierParams:
    """Fitted Fourier climatology parameters; method-tagged for
    downstream dispatch."""
    beta_mu: np.ndarray              # (1 + 2*K_YEAR + 2*K_DAY,)
    beta_sigma: np.ndarray           # same shape
    k_year: int
    k_day: int
    cutoff: str                      # ISO string of fit cutoff
    method: str = "fourier"

    def evaluate(self, idx: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
        """Return (mu, sigma) at each timestamp."""
        X = _fourier_design(idx, self.k_year, self.k_day)
        mu = X @ self.beta_mu
        # sigma fit was on |residual|; rescale by sqrt(pi/2) (half-normal moment)
        sigma_scale = np.sqrt(np.pi / 2.0)
        sigma = np.maximum(X @ self.beta_sigma * sigma_scale, 1.0)
        return mu, sigma


def fit_fourier_params(demand: pd.Series, cutoff: pd.Timestamp,
                       k_year: int = K_YEAR, k_day: int = K_DAY
                       ) -> FourierParams:
    """Fit Fourier mu and sigma on pre-cutoff demand. Leakage guard:
    only pre-cutoff data enters the fit, matching the existing
    (month, hour) climatology's spec discipline."""
    train = demand[demand.index <= cutoff].dropna()
    X = _fourier_design(train.index, k_year, k_day)
    beta_mu, *_ = np.linalg.lstsq(X, train.values, rcond=None)
    resid = train.values - X @ beta_mu
    beta_sigma, *_ = np.linalg.lstsq(X, np.abs(resid), rcond=None)
    return FourierParams(
        beta_mu=beta_mu, beta_sigma=beta_sigma,
        k_year=k_year, k_day=k_day,
        cutoff=str(cutoff),
    )


def fourier_transform(demand: pd.Series,
                      params: FourierParams) -> pd.Series:
    """Apply Fourier climatology -> z-score series."""
    mu, sigma = params.evaluate(demand.index)
    return pd.Series((demand.values - mu) / sigma, index=demand.index,
                     name="zscore")


def fourier_destandardise(z_value: float, ts: pd.Timestamp,
                          params: FourierParams) -> float:
    """Convert z forecast at a single timestamp back to demand MW.
    Mirror of mu_mh + sigma_mh * z used in predict.py for the
    (month, hour) path."""
    idx = pd.DatetimeIndex([ts])
    mu, sigma = params.evaluate(idx)
    return float(z_value * sigma[0] + mu[0])


# Convenience: a one-shot "load Ontario, fit, return both demand and z"
# that matches the typical scratch usage pattern.
def load_demand_and_z(cutoff: pd.Timestamp = None
                      ) -> tuple[pd.Series, pd.Series, FourierParams]:
    """Returns (raw_demand, z_fourier, params). cutoff=None uses
    the project's frozen CUTOFF."""
    from experiment._actuals import load_actuals
    if cutoff is None:
        from scratch.benchmark_2021_22_ieso import CUTOFF
        cutoff = CUTOFF
    demand = load_actuals(cutoff=None).asfreq("h")
    params = fit_fourier_params(demand, cutoff)
    z = fourier_transform(demand, params)
    return demand, z, params


if __name__ == "__main__":
    # Smoke test
    d, z, p = load_demand_and_z()
    print(f"Fourier climatology K_year={K_YEAR}, K_day={K_DAY}")
    print(f"  fit cutoff: {p.cutoff}")
    print(f"  n_params: mu={len(p.beta_mu)}, sigma={len(p.beta_sigma)}")
    print(f"  pre-cutoff z mean/std: "
          f"{z[z.index <= p.cutoff].mean():+.4f} / "
          f"{z[z.index <= p.cutoff].std():.4f}")
    print(f"  post-cutoff z mean/std: "
          f"{z[z.index > p.cutoff].mean():+.4f} / "
          f"{z[z.index > p.cutoff].std():.4f}")
    # Destandardise round-trip check
    t = pd.Timestamp("2022-04-01 00:00")
    if t in d.index:
        z_t = z.loc[t]
        d_back = fourier_destandardise(z_t, t, p)
        d_raw = float(d.loc[t])
        print(f"  round-trip @ {t}: raw={d_raw:.1f} -> z={z_t:+.4f} -> "
              f"raw={d_back:.1f}  (err={d_back - d_raw:.4f})")
