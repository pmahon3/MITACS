"""Shared pre-cutoff actuals loader + z-score derivation.

ONE implementation used by both ``freeze.py`` (to fingerprint the
pre-registered data) and ``predict.py`` (to derive the z-score
parameters at predict time). They must agree exactly, so they call the
same code -- not two parallel loaders. The hourly-index construction
mirrors ``pre_processing.year_wise_standardization`` (skiprows=3,
Date+Hour-1 -> timestamp, "Ontario Demand").

The cutoff is enforced as a HARD guard: any row dated strictly after it
is dropped before any statistic is computed, so post-cutoff data can
never enter the predictor's input representation (the leakage guard).

Two climatology methods are supported:

  ``method='month_hour'`` (default): piecewise-constant lookup keyed by
    (month, hour-of-day). The historical production method.

  ``method='fourier'``: continuous Fourier basis in (day-of-year,
    hour-of-day). Eliminates the month-boundary lookup discontinuity
    by construction. Parameters (K_YEAR, K_DAY) live in
    ``experiment.fourier_climatology``.

Both methods produce the same downstream interface (``params`` dict
tagged with a ``method`` key; ``mu_at``/``sigma_at`` helpers dispatch
internally) so call sites that need destandardisation -- ``predict.py``,
``predict_multistep.py`` -- are method-agnostic.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import load_config

from .fourier_climatology import FourierParams, fit_fourier_params


def _load_raw_actuals() -> pd.Series:
    """All hourly "Ontario Demand" from the git-tracked per-year
    ``PUB_Demand_<yr>.csv`` files (no cutoff). Single parser; mirrors
    ``pre_processing.year_wise_standardization``."""
    cfg = load_config()
    frames = []
    for f in sorted(Path(cfg.paths.historical_csvs).glob("PUB_Demand_*.csv")):
        df = pd.read_csv(f, skiprows=3).dropna(
            subset=["Date", "Hour", "Ontario Demand"]
        )
        df["Time"] = pd.to_datetime(
            df["Date"].astype(str)
            + " "
            + (df["Hour"] - 1).astype(int).astype(str)
            + ":00:00"
        )
        frames.append(df.set_index("Time")[["Ontario Demand"]])
    return pd.concat(frames).sort_index()["Ontario Demand"]


def load_actuals(cutoff: pd.Timestamp | None = None) -> pd.Series:
    """Hourly "Ontario Demand". With ``cutoff``, strictly ``<= cutoff``
    (the HARD leakage guard for fits / z-score params). Without, the full
    tracked series -- ONLY for building post-cutoff query states, never a
    fit."""
    hourly = _load_raw_actuals()
    if cutoff is not None:
        hourly = hourly[hourly.index <= cutoff]
        if hourly.empty:
            raise ValueError(f"no actuals <= {cutoff}")
    return hourly


def load_pre_cutoff_actuals(cutoff: pd.Timestamp) -> pd.Series:
    """Hourly "Ontario Demand" strictly ``<= cutoff`` (leakage guard)."""
    return load_actuals(cutoff)


def zscore_params(
    cutoff: pd.Timestamp,
    method: str = "month_hour",
    k_year: int | None = None,
    k_day: int | None = None,
) -> dict[str, Any]:
    """Climatology params derived strictly from ``<= cutoff`` data.

    Returns a dict tagged with ``method``. The downstream callers
    (``zscore_transform``, ``mu_at``, ``sigma_at``) dispatch on this
    tag, so call sites don't need to know which method is in use.

    ``method='month_hour'``: returns
        ``{"method": "month_hour", "mu_mh": pd.Series, "sigma_mh":
        pd.Series}``
    where ``mu_mh``/``sigma_mh`` are MultiIndex-keyed by (month, hour).

    ``method='fourier'``: returns
        ``{"method": "fourier", "fourier": FourierParams}``
    with ``k_year``/``k_day`` defaulting to the production constants
    from ``experiment.fourier_climatology`` (8, 8).
    """
    if method == "month_hour":
        h = load_pre_cutoff_actuals(cutoff)
        by = [h.index.month, h.index.hour]
        mu_mh = h.groupby(by).mean()
        sigma_mh = h.groupby(by).std()
        return {"method": "month_hour", "mu_mh": mu_mh, "sigma_mh": sigma_mh}
    if method == "fourier":
        h = load_pre_cutoff_actuals(cutoff)
        # Fit with defaults if k_year / k_day not supplied.
        kwargs: dict[str, int] = {}
        if k_year is not None:
            kwargs["k_year"] = k_year
        if k_day is not None:
            kwargs["k_day"] = k_day
        fp = fit_fourier_params(h, cutoff, **kwargs)
        return {"method": "fourier", "fourier": fp}
    raise ValueError(
        f"unknown climatology method {method!r}; "
        f"expected 'month_hour' or 'fourier'"
    )


def mu_at(params: dict[str, Any], idx: pd.DatetimeIndex) -> np.ndarray:
    """Climatology mean at each timestamp; dispatches on ``params['method']``."""
    method = params.get("method", "month_hour")
    if method == "month_hour":
        keys = list(zip(idx.month, idx.hour))
        return params["mu_mh"].reindex(keys).to_numpy()
    if method == "fourier":
        mu, _ = params["fourier"].evaluate(idx)
        return mu
    raise ValueError(f"unknown climatology method {method!r}")


def sigma_at(params: dict[str, Any], idx: pd.DatetimeIndex) -> np.ndarray:
    """Climatology std at each timestamp; dispatches on ``params['method']``."""
    method = params.get("method", "month_hour")
    if method == "month_hour":
        keys = list(zip(idx.month, idx.hour))
        return params["sigma_mh"].reindex(keys).to_numpy()
    if method == "fourier":
        _, sigma = params["fourier"].evaluate(idx)
        return sigma
    raise ValueError(f"unknown climatology method {method!r}")


def zscore_transform(
    raw: pd.Series, params: dict[str, Any]
) -> pd.Series:
    """Apply the frozen climatology to raw demand -> zscore."""
    idx = raw.index
    mu = mu_at(params, idx)
    sd = sigma_at(params, idx)
    return pd.Series((raw.to_numpy() - mu) / sd, index=idx, name="zscore")


def actuals_fingerprint(cutoff: pd.Timestamp) -> str:
    """Content hash of the exact pre-cutoff actuals used.

    Deterministic over (timestamp_ns, demand) rows so any later change to
    the historical CSVs (re-scrape, revision) is detected by the frozen
    spec's hash.
    """
    h = load_pre_cutoff_actuals(cutoff)
    buf = np.ascontiguousarray(
        np.column_stack([h.index.view("int64"), h.to_numpy(dtype="float64")])
    )
    return hashlib.sha256(buf.tobytes()).hexdigest()


def zscore_params_fingerprint(
    cutoff: pd.Timestamp,
    method: str = "month_hour",
    k_year: int | None = None,
    k_day: int | None = None,
) -> str:
    """Content hash of the derived climatology -- recorded with every
    forecast so post-hoc drift is detectable even though the values are
    not stored in the spec."""
    p = zscore_params(cutoff, method=method, k_year=k_year, k_day=k_day)
    parts: list[bytes] = [p["method"].encode()]
    if p["method"] == "month_hour":
        for name in ("mu_mh", "sigma_mh"):
            s = p[name].sort_index()
            parts.append(name.encode())
            parts.append(np.ascontiguousarray(
                s.to_numpy(dtype="float64")
            ).tobytes())
    elif p["method"] == "fourier":
        fp: FourierParams = p["fourier"]
        parts.append(f"k_year={fp.k_year},k_day={fp.k_day}".encode())
        parts.append(b"beta_mu")
        parts.append(np.ascontiguousarray(
            fp.beta_mu.astype("float64")
        ).tobytes())
        parts.append(b"beta_sigma")
        parts.append(np.ascontiguousarray(
            fp.beta_sigma.astype("float64")
        ).tobytes())
    return hashlib.sha256(b"".join(parts)).hexdigest()
