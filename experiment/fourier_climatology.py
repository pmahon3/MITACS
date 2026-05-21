"""Fourier climatology -- continuous (day-of-year, hour-of-day) basis.

Replaces the piecewise-constant (month, hour) lookup with a smooth
Fourier expansion in (day-of-year, hour-of-day). The basis has
``k_year`` harmonics in day-of-year (annual + sub-annual periodicity)
and ``k_day`` harmonics in hour-of-day. No interactions, no year-trend
(the multi-decade demand drift is left as a residual property,
handled implicitly by the local-linear forecaster's lag state).

Production constants K_YEAR = K_DAY = 8 (33 parameters total vs the
month-hour lookup's 288). Selected on pre-cutoff dev-set diagnostics:
per-month |bias| at first-of-month 00:00 reduced from 0.42 to 0.21
(-49%); per-boundary RMS jump |z[0] - z[-1]| reduced from 0.80 to
0.18 (-78%). The continuous basis eliminates the month-boundary
lookup discontinuity by construction.

Leakage guard: only ``<= cutoff`` data enters the fit, matching the
existing month-hour climatology's spec discipline.
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
    + 2*k_day (hour-of-day cos/sin pairs).

    Day-of-year uses 365.25 to handle leap years smoothly without
    a discontinuity at year boundaries.
    """
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
    cutoff: str                      # ISO string of the fit cutoff
    method: str = "fourier"

    def evaluate(self, idx: pd.DatetimeIndex
                 ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(mu, sigma)`` at each timestamp."""
        X = _fourier_design(idx, self.k_year, self.k_day)
        mu = X @ self.beta_mu
        # sigma fit was on |residual|; rescale by sqrt(pi/2) (half-normal
        # moment) so the result is a standard-deviation, not a mean-abs.
        sigma_scale = np.sqrt(np.pi / 2.0)
        sigma = np.maximum(X @ self.beta_sigma * sigma_scale, 1.0)
        return mu, sigma


def fit_fourier_params(demand: pd.Series, cutoff: pd.Timestamp,
                       k_year: int = K_YEAR, k_day: int = K_DAY
                       ) -> FourierParams:
    """Fit Fourier ``mu`` and ``sigma`` on pre-cutoff demand."""
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
