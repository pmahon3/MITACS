#!/usr/bin/env python3
"""
Processing interface for Rose-Operator forecasts.
Provides a single save_all() to write out all artifacts from a run.
"""
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Sequence, Any
import logging

logger = logging.getLogger(__name__)


def _save_residuals(
        residuals: torch.Tensor,  # shape (N, H, d)
        output_dir: Path,
        run_tag: str,
) -> Path:
    """Save the raw residuals tensor."""
    output_dir.mkdir(exist_ok=True, parents=True)
    path = output_dir / f"residuals_{run_tag}.pt"
    torch.save(residuals, path)
    return path


def _save_ensembles(
        ensembles: torch.Tensor,  # shape (N, M, H, d)
        output_dir: Path,
        run_tag: str,
) -> Path:
    """Save the raw ensemble tensor."""
    output_dir.mkdir(exist_ok=True, parents=True)
    path = output_dir / f"ensemble_{run_tag}.pt"
    torch.save(ensembles, path)
    return path


def _save_coefficients(
        coefficients: torch.Tensor,  # shape (N, H, d, d)
        output_dir: Path,
        run_tag: str,
) -> Path:
    """Save the local linear map coefficient tensor."""
    output_dir.mkdir(exist_ok=True, parents=True)
    path = output_dir / f"coeffs_{run_tag}.pt"
    torch.save(coefficients, path)
    return path


def _save_covariances(
        covariances: Optional[torch.Tensor],  # shape (N, H, d, d)
        output_dir: Path,
        run_tag: str,
) -> Optional[Path]:
    """Save the innovation covariance tensor, if provided."""
    if covariances is None:
        return None
    output_dir.mkdir(exist_ok=True, parents=True)
    path = output_dir / f"covs_{run_tag}.pt"
    torch.save(covariances, path)
    return path


def _save_anchor_times(
        anchor_times: Sequence[Any],  # e.g. numpy datetime64 or pandas Timestamp
        output_dir: Path,
        run_tag: str,
) -> Path:
    """
    Save the anchor times array so downstream dashboards can use real datetimes.
    Stored as NumPy .npy.
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    path = output_dir / f"times_{run_tag}.npy"
    arr = np.asarray(anchor_times)
    np.save(path, arr)
    return path


# --- new helper -------------------------------------------------------
def _save_times(times: np.ndarray, out_dir: Path, tag: str) -> Path:
    """
    Save anchor times (dtype=datetime64[ns]) to times_<tag>.npy.
    """
    out_dir.mkdir(exist_ok=True, parents=True)
    path = out_dir / f"times_{tag}.npy"
    np.save(path, times)
    return path


def save_all(
    *,
    residuals:   torch.Tensor,          # (N,H,d)   or (N,M,H,d)
    coefficients:torch.Tensor,          # (N,H,d,d)
    covariances: torch.Tensor | None,   # optional
    ensembles:   torch.Tensor,          # (N,M,H,d)
    anchor_times: np.ndarray,           # (N,) datetime64[ns]
    output_dir:  Path,
    run_tag:     str,
) -> None:
    """
    Save residuals, ensembles, coeffs, covs (optional), and anchor_times.
    """
    _save_residuals(residuals, output_dir, run_tag)
    _save_ensembles(ensembles, output_dir, run_tag)
    _save_coefficients(coefficients, output_dir, run_tag)
    _save_times(anchor_times, output_dir, run_tag)           # ← NEW
    if covariances is not None:
        _save_covariances(covariances, output_dir, run_tag)
    logger.info("Saved all outputs for %s in %s", run_tag, output_dir)