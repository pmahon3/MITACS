"""Artifact loader for the operator analysis dashboard.

Reads the CURRENT per-day-type estimator contract written by
``processing.innovations.interface.save_all`` — NOT the dead
ensemble/horizon contract the archived ``post_processing.innovations``
loader assumed (that one globbed a single ``coeffs_*.pt`` and
synthesised a fake ensemble; it could not express per-day-type or
"over state space" at all).

Per ``run_tag`` (= day-type) the on-disk arrays are::

    coeffs_<tag>.pt        (N, d, d)  drift C_j   (row conv: x' = x @ C_j)
    covs_<tag>.pt          (N, d, d)  diffusion Sigma_j
    resid_eigvals_<tag>.pt (N, d)     ascending eig(Sigma_j)
    thetas_<tag>.npy       (N,)       per-anchor bandwidth theta_j
    sigmas_<tag>.npy       (N,)       per-anchor sigma_star
    times_<tag>.npy        (N,)       int64 ns anchor timestamps

`d` VARIES by day-type (observed: weekday/saturday d=2, sunday d=7) —
nothing here may assume a fixed embedding dimension. The anchor
embedding *coordinates* are NOT persisted; only the operator + the
timestamp per anchor are. The dashboard therefore orders anchors along
the reconstructed trajectory BY TIME (the natural traversal of a
delay-embedded orbit); a true attractor scatter is a tracked follow-up
that would re-derive coordinates from the z-score series.

Provenance posture (decided 2026-05-18): a viewer DISPLAYS objects, it
does not ASSERT claims, so it does not refuse unprovenanced artifacts.
It reports the provenance state plainly (see ``provenance_state``) so
the surface is never mistaken for an authoritative result. A
verify-on-load / sidecar-manifest interface is a separate later task.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

DAYTYPES = ("weekday", "saturday", "sunday")


@dataclass(frozen=True)
class DaytypeArtifacts:
    """One day-type's per-anchor operator artifacts, time-sorted."""

    daytype: str
    times: np.ndarray  # (N,) datetime64[ns], ascending
    coeffs: np.ndarray  # (N, d, d) drift C_j
    covs: np.ndarray  # (N, d, d) diffusion Sigma_j
    eigvals: np.ndarray  # (N, d) ascending eig(Sigma_j)
    thetas: np.ndarray  # (N,) bandwidth theta_j
    sigmas: np.ndarray  # (N,) sigma_star

    @property
    def n(self) -> int:
        return self.times.shape[0]

    @property
    def d(self) -> int:
        return self.coeffs.shape[1]

    def drift_eigvals(self) -> np.ndarray:
        """(N, d) complex eigenvalues of C_j per anchor — the local
        linearised dynamics (the Koopman / kappa_Q conditional-mean
        content). Computed here, not persisted."""
        return np.linalg.eigvals(self.coeffs)


def _load_tensor(p: Path) -> np.ndarray:
    arr = torch.load(p, weights_only=True)
    return arr.numpy() if hasattr(arr, "numpy") else np.asarray(arr)


def load_daytype(output_dir: Path, daytype: str) -> DaytypeArtifacts:
    """Load + time-sort one day-type. Raises if the core contract files
    are absent (a real missing-artifact error, not silently faked)."""
    need = {
        "coeffs": output_dir / f"coeffs_{daytype}.pt",
        "covs": output_dir / f"covs_{daytype}.pt",
        "eigvals": output_dir / f"resid_eigvals_{daytype}.pt",
        "thetas": output_dir / f"thetas_{daytype}.npy",
        "sigmas": output_dir / f"sigmas_{daytype}.npy",
        "times": output_dir / f"times_{daytype}.npy",
    }
    missing = [str(p) for p in need.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"missing {daytype} artifacts: {missing} — run the "
            f"innovations stage to produce them"
        )

    times_ns = np.load(need["times"]).astype("int64")
    order = np.argsort(times_ns, kind="stable")
    times = times_ns[order].astype("datetime64[ns]")

    return DaytypeArtifacts(
        daytype=daytype,
        times=times,
        coeffs=_load_tensor(need["coeffs"])[order],
        covs=_load_tensor(need["covs"])[order],
        eigvals=_load_tensor(need["eigvals"])[order],
        thetas=np.load(need["thetas"]).astype("float64")[order],
        sigmas=np.load(need["sigmas"]).astype("float64")[order],
    )


def available_daytypes(output_dir: Path) -> list[str]:
    return [
        dt
        for dt in DAYTYPES
        if (output_dir / f"coeffs_{dt}.pt").exists()
    ]


def provenance_state(output_dir: Path) -> str:
    """Plain status string for the dashboard banner. The current
    save_all writes bare torch/np files with no provenance header, so
    this is honestly reported as UNPROVENANCED rather than implied
    trustworthy. (A real verify-on-load interface is a later task.)"""
    if list(output_dir.glob("manifest_*.json")):
        return "provenance manifest present (not yet verified by this viewer)"
    return (
        "UNPROVENANCED / EXPLORATORY — artifacts have no provenance "
        "header; this is a viewer, not an authoritative result surface"
    )
