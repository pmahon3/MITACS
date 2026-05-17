"""Save/load layer for the local Gaussian semigroup (Pi_Delta) estimator.

Supersedes the previous broken ``save_all`` (which required an ``ensembles``
tensor the producer never passed and used mismatched keyword names). The
filename convention follows the established ``<artifact>_<tag>`` glob contract
so existing loaders/dashboards keep working:

    coeffs_<tag>.pt        (N, d, d) float32  -- drift C_j
    covs_<tag>.pt          (N, d, d) float32  -- diffusion Sigma_j
    resid_means_<tag>.pt   (N, d)    float32  -- residual mean mu_j
    resid_eigvals_<tag>.pt (N, d)    float32  -- ascending eig(Sigma_j)
    residuals_<tag>.pt     (N, 1, d) float32  -- mu_j as (N,1,d) for the
                                                 legacy dashboard's loader
    thetas_<tag>.npy       (N,) float64       -- per-anchor drift bandwidth
    sigmas_<tag>.npy       (N,) float64       -- per-anchor diffusion bandwidth
    times_<tag>.npy        (N,) int64 ns      -- anchor timestamps
    spectrum_<tag>.npz                        -- aggregated spectral diagnostic
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from .estimator import SemigroupEstimate
from .spectral import SpectralDiagnostic


def save_all(
    *,
    estimate: SemigroupEstimate,
    spectrum: SpectralDiagnostic,
    output_dir: Path,
    run_tag: str,
) -> None:
    """Persist all estimator + spectral artifacts for ``run_tag``."""
    output_dir.mkdir(parents=True, exist_ok=True)

    def _t(arr: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(arr, dtype=torch.float32)

    torch.save(_t(estimate.coefficients), output_dir / f"coeffs_{run_tag}.pt")
    torch.save(_t(estimate.covariances), output_dir / f"covs_{run_tag}.pt")
    torch.save(_t(estimate.resid_means), output_dir / f"resid_means_{run_tag}.pt")
    torch.save(_t(estimate.eigvals), output_dir / f"resid_eigvals_{run_tag}.pt")
    # legacy dashboard expects residuals_*.pt as (N, H, d); use H=1 with mu_j
    torch.save(
        _t(estimate.resid_means[:, None, :]), output_dir / f"residuals_{run_tag}.pt"
    )

    np.save(output_dir / f"thetas_{run_tag}.npy", estimate.theta_star.astype(np.float64))
    np.save(output_dir / f"sigmas_{run_tag}.npy", estimate.sigma_star.astype(np.float64))
    np.save(output_dir / f"times_{run_tag}.npy", estimate.anchor_times.astype(np.int64))

    np.savez(
        output_dir / f"spectrum_{run_tag}.npz",
        mean_spectrum=spectrum.mean_spectrum,
        energy_fraction=spectrum.energy_fraction,
        gap_ratios=spectrum.gap_ratios,
        r_hat=np.int64(spectrum.r_hat),
    )


def _one(run_dir: Path, pattern: str) -> Path:
    matches = sorted(run_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no file matching {pattern!r} in {run_dir}")
    return matches[0]


def load_all(run_dir: Path) -> Dict[str, Any]:
    """Load the estimator + spectral artifacts written by :func:`save_all`."""
    run_dir = Path(run_dir)
    spec = np.load(_one(run_dir, "spectrum_*.npz"))
    return {
        "coeffs": torch.load(_one(run_dir, "coeffs_*.pt")).numpy(),
        "covs": torch.load(_one(run_dir, "covs_*.pt")).numpy(),
        "resid_means": torch.load(_one(run_dir, "resid_means_*.pt")).numpy(),
        "resid_eigvals": torch.load(_one(run_dir, "resid_eigvals_*.pt")).numpy(),
        "thetas": np.load(_one(run_dir, "thetas_*.npy")),
        "sigmas": np.load(_one(run_dir, "sigmas_*.npy")),
        "times": np.load(_one(run_dir, "times_*.npy")),
        "spectrum": {k: spec[k] for k in spec.files},
    }
