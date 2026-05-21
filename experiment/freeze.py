"""Hash-stamped frozen predictor spec -- the experiment's pre-registration.

The spec captures *everything that determines a forecast*: the estimator
method identity, the resolved per-day-type embedding dimensions, the data
cutoff (the model-specification boundary), and code/library provenance.
``spec_hash`` is SHA256 over the canonical serialization (sorted keys,
excluding the hash field). The loader recomputes and verifies it on read:
the model cannot be silently changed after forecasts are issued, which is
what makes the experiment a *registered* prediction.

Honest scope: this freezes the **method + resolved hyperparameters +
data cutoff + code commit**. The estimator is local/lazy (it fits per
anchor at prediction time from library data up to the cutoff), so there
is no separate trained-weights blob to freeze -- the freeze pins the
function and its inputs, and `predict.py` enforces the cutoff so no
post-cutoff data ever enters a fit.

Usage::

    python -m experiment.freeze --create     # write the frozen spec
    python -m experiment.freeze --verify     # recompute + check the hash
"""
from __future__ import annotations

import argparse
import hashlib
import json
from importlib import metadata
from typing import Any

import pandas as pd

from config import PROJECT_ROOT, load_config

from ._actuals import actuals_fingerprint as _actuals_fingerprint
from ._prov_core import canonical as _prov_canonical
from ._prov_core import git_clean as _git_clean
from ._prov_core import git_sha as _git_sha

SPEC_PATH = PROJECT_ROOT / "experiment" / "frozen_spec.json"

# The model-specification boundary: nothing dated after this informs the
# predictor (fits or z-score representation). Single source of truth.
DATA_CUTOFF = pd.Timestamp("2024-12-31T23:00:00")


def _canonical(spec: dict[str, Any]) -> bytes:
    """Deterministic bytes for hashing the frozen spec (excludes the
    ``spec_hash`` field). Thin wrapper over the shared primitive."""
    return _prov_canonical(spec, exclude="spec_hash")


def build_spec(
    climatology_method: str = "month_hour",
    fourier_k_year: int | None = None,
    fourier_k_day: int | None = None,
) -> dict[str, Any]:
    cfg = load_config()
    predictor: dict[str, Any] = {
        # method identity -- the estimator's resolved behaviour
        "estimator": "build_local_gaussian_semigroup",
        "drift_bandwidth_rule": "true_loo_cv",
        "diffusion": "plain_mu_centred_residual_covariance_no_kernel",
        "day_anchor_hour": cfg.data.day_anchor_hours,
        "dim_selection": cfg.embedding.dim_selection,
        "elbow_tol": cfg.embedding.elbow_tol,
        "grid_spacing": cfg.theta.grid_spacing,
        "variable_name": cfg.data.variable_name,
        "estimand": "intra_day",  # full_process Sigma is singular
        "embedding_dims": {
            dt: cfg.embedding_dim(dt) for dt in cfg.data.daytypes
        },
        "horizon": "one_step",  # multi-step is the unbuilt semigroup
        # de-seasonalisation: month_hour is the historical default;
        # fourier is the smooth alternative.
        "climatology_method": climatology_method,
    }
    if climatology_method == "fourier":
        # Default to the production constants from
        # experiment.fourier_climatology if not overridden.
        from .fourier_climatology import K_YEAR, K_DAY
        predictor["fourier_k_year"] = (
            int(fourier_k_year) if fourier_k_year is not None else K_YEAR
        )
        predictor["fourier_k_day"] = (
            int(fourier_k_day) if fourier_k_day is not None else K_DAY
        )
    spec: dict[str, Any] = {
        "schema": "mitacs.experiment.frozen_spec/1",
        "predictor": predictor,
        "data_cutoff": DATA_CUTOFF.isoformat(),
        "provenance": {
            "git_sha": _git_sha(),
            "git_clean": _git_clean(),
            "edynamics_version": metadata.version("edynamics"),
        },
        # Content hash of the exact pre-cutoff actuals the predictor's
        # z-score representation is derived from. The z-score params are
        # NOT stored (they are reproducible from this pinned data + the
        # pinned code commit); this fingerprint makes any later change to
        # the historical CSVs tamper-evident via the spec hash.
        "data_fingerprint": {
            "pre_cutoff_actuals_sha256": _actuals_fingerprint(DATA_CUTOFF),
        },
    }
    spec["spec_hash"] = hashlib.sha256(_canonical(spec)).hexdigest()
    return spec


def create(
    force: bool = False,
    climatology_method: str = "month_hour",
    fourier_k_year: int | None = None,
    fourier_k_day: int | None = None,
) -> dict[str, Any]:
    if SPEC_PATH.exists() and not force:
        raise SystemExit(
            f"{SPEC_PATH} already exists. A frozen spec is immutable by "
            f"design; use --force only to deliberately re-register (this "
            f"invalidates all prior forecasts' provenance)."
        )
    spec = build_spec(
        climatology_method=climatology_method,
        fourier_k_year=fourier_k_year,
        fourier_k_day=fourier_k_day,
    )
    if not spec["provenance"]["git_clean"]:
        raise SystemExit(
            "refusing to freeze from a dirty tree: the recorded git_sha "
            "would not reproduce the predictor. Commit or stash first."
        )
    SPEC_PATH.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    return spec


def load_verified() -> dict[str, Any]:
    """Load the frozen spec, recomputing and checking its hash. Raises if
    the file was altered after freezing (tamper-evident)."""
    if not SPEC_PATH.exists():
        raise FileNotFoundError(
            f"no frozen spec at {SPEC_PATH}; run "
            f"`python -m experiment.freeze --create`"
        )
    spec = json.loads(SPEC_PATH.read_text())
    recomputed = hashlib.sha256(_canonical(spec)).hexdigest()
    if recomputed != spec.get("spec_hash"):
        raise ValueError(
            f"frozen spec hash mismatch: file says {spec.get('spec_hash')}, "
            f"content hashes to {recomputed}. The spec was modified after "
            f"freezing -- experiment integrity is compromised."
        )
    return spec


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--create", action="store_true", help="write frozen spec")
    ap.add_argument(
        "--force", action="store_true",
        help="overwrite an existing spec (deliberate re-registration)",
    )
    ap.add_argument(
        "--verify", action="store_true", help="recompute + check the hash"
    )
    ap.add_argument(
        "--climatology", default="month_hour",
        choices=["month_hour", "fourier"],
        help="de-seasonalisation method (default: month_hour)",
    )
    ap.add_argument(
        "--fourier-k-year", type=int, default=None,
        help="Fourier day-of-year harmonics (default: K_YEAR from "
             "experiment.fourier_climatology)",
    )
    ap.add_argument(
        "--fourier-k-day", type=int, default=None,
        help="Fourier hour-of-day harmonics (default: K_DAY)",
    )
    args = ap.parse_args()

    if args.create:
        s = create(
            force=args.force,
            climatology_method=args.climatology,
            fourier_k_year=args.fourier_k_year,
            fourier_k_day=args.fourier_k_day,
        )
        print(f"froze predictor spec -> {SPEC_PATH}")
        print(f"  spec_hash         = {s['spec_hash']}")
        print(f"  git_sha           = {s['provenance']['git_sha']}")
        print(f"  dims              = {s['predictor']['embedding_dims']}")
        print(f"  climatology_method = "
              f"{s['predictor']['climatology_method']}")
        if s['predictor']['climatology_method'] == 'fourier':
            print(f"  fourier_k_year    = "
                  f"{s['predictor']['fourier_k_year']}")
            print(f"  fourier_k_day     = "
                  f"{s['predictor']['fourier_k_day']}")
    elif args.verify:
        s = load_verified()
        print(f"frozen spec OK (hash verified): {s['spec_hash']}")
        print(f"  cutoff={s['data_cutoff']}  git={s['provenance']['git_sha'][:12]}")
    else:
        ap.print_help()
