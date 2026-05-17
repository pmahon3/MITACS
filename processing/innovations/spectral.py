"""Eigenmode diagnostics on the per-anchor diffusion family ``{Sigma_j}``.

The intent (per the project email) is to use the orthogonal decomposition of
the residual covariances to inform delay-lag selection: directions carrying
the most diffusion variance indicate the effective dimensionality of the
stochastic part.

[DEFER-RF] Two choices here defer to the Resolvent_Framework programme and
are *not* settled in code:

  1. Whether lag selection should key off the diffusion spectrum
     ``eig(Sigma_j)`` or the drift spectrum ``eig(C_j)``. This module
     computes the *diffusion* spectrum; the drift alternative is left open.
  2. The stopping rule for "how many modes". The programme's audit withdrew
     Paper III for rediscovery of elbow/gap stopping rules (Lepski 1991
     etc.), so no stopping rule is asserted as authoritative. We expose the
     full aggregated spectrum and a transparent gap-ratio ``r_hat`` as a
     *diagnostic*, explicitly labelled, for the programme to adjudicate.

Nothing here makes a novelty claim.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SpectralDiagnostic:
    """Aggregated diffusion-spectrum diagnostics across anchors.

    Shapes (``d`` embedding dimension):
        mean_spectrum    : (d,) descending mean eigenvalue per rank
        energy_fraction  : (d,) cumulative fraction of total diffusion energy
        gap_ratios       : (d-1,) consecutive descending eigenvalue ratios
        r_hat            : int   gap-ratio mode count (DIAGNOSTIC ONLY)
    """

    mean_spectrum: np.ndarray
    energy_fraction: np.ndarray
    gap_ratios: np.ndarray
    r_hat: int


def diffusion_spectrum(eigvals: np.ndarray) -> SpectralDiagnostic:
    """Aggregate per-anchor diffusion eigenvalues into a spectral diagnostic.

    Parameters
    ----------
    eigvals:
        ``(N, d)`` ascending eigenvalues of ``Sigma_j`` (as produced by
        :class:`estimator.SemigroupEstimate`). NaN rows (failed anchors) are
        dropped.
    """
    eig = np.asarray(eigvals, dtype=float)
    eig = eig[np.all(np.isfinite(eig), axis=1)]
    if eig.size == 0:
        raise ValueError("no finite eigenvalue rows to aggregate")

    # descending per anchor, then average across anchors
    desc = np.sort(eig, axis=1)[:, ::-1]
    mean_spectrum = desc.mean(axis=0)                       # (d,)
    mean_spectrum = np.clip(mean_spectrum, 0.0, None)       # tiny negatives -> 0

    total = mean_spectrum.sum()
    energy_fraction = (
        np.cumsum(mean_spectrum) / total
        if total > 0
        else np.zeros_like(mean_spectrum)
    )

    # consecutive descending ratios lambda_r / lambda_{r+1}
    denom = mean_spectrum[1:] + 1e-12
    gap_ratios = mean_spectrum[:-1] / denom

    # DIAGNOSTIC r_hat: rank of the largest spectral gap, +1 (number of
    # modes above the dominant gap). Transparent and reproducible; not an
    # endorsed stopping rule -- see [DEFER-RF] in the module docstring.
    r_hat = int(np.argmax(gap_ratios) + 1) if gap_ratios.size else len(mean_spectrum)

    return SpectralDiagnostic(
        mean_spectrum=mean_spectrum,
        energy_fraction=energy_fraction,
        gap_ratios=gap_ratios,
        r_hat=r_hat,
    )
