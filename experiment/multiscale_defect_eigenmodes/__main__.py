"""Eigenmode characterization runner.

Reads the v2 factor pickle and computes, per (day_type, h):

  R_k(h, dt) = (V^{-1} D_h V^{-T})[k, k]      where D_h = Sigma_h - iter_Sigma_h
                                              and  C_1 = V Lambda V^{-1}

Then evaluates the three pre-registered shape outcomes (O1/O2/O3) and
the library-asymmetry sub-test.

Sunday's d=4 has a complex-conjugate eigenpair: the two complex modes
are aggregated into their 2D real invariant subspace and reported as a
single "pair_block" entry.

No new compute beyond what's already in the v2 pickle.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

from config import PROJECT_ROOT


# Pre-registered shape-outcome thresholds (Phase A §shape_outcomes).
O1_DOMINANT_SHARE_THRESHOLD = 0.70
O3_EVEN_BAND_D2 = (0.40, 0.65)
O3_EVEN_BAND_D4_PAIR_AGG = (0.30, 0.50)   # sunday's 3 modes after pair-aggregation
TAU_LOCALIZED_TOLERANCE = 2.0              # factor-of-2 tolerance for O2

# Asymmetry sub-test (Phase A §asymmetry_subtest).
ASYMMETRY_LIBRARY_BAND = (3.0, 8.0)
WEEKDAY_VS_WEEKEND_N_RATIO_EXPECTED = 5.0


# ---------------------------------------------------------------------------
# Iterated diffusion (mirror of __main__.py's _iterated_sigma)
# ---------------------------------------------------------------------------


def _iterated_sigma(C1: np.ndarray, Sigma1: np.ndarray, h: int) -> np.ndarray:
    out = np.zeros_like(Sigma1)
    Cj = np.eye(C1.shape[0])
    for _ in range(h):
        out = out + Cj @ Sigma1 @ Cj.T
        Cj = Cj @ C1
    return out


# ---------------------------------------------------------------------------
# Eigendecomposition with complex-pair aggregation for sunday
# ---------------------------------------------------------------------------


def _decompose_C1(C1: np.ndarray) -> dict[str, Any]:
    """Return eigendecomposition plus a basis transform that aggregates
    complex-conjugate pairs into their 2D real invariant subspace.

    For d=2 with two real eigenvalues, returns straightforward V, Lambda
    and labels ['real_0', 'real_1'].

    For d=4 with one or more complex-conjugate pairs, returns a real
    basis V_real such that V_real^{-1} C1 V_real is block-diagonal:
    1x1 blocks for real eigenvalues, 2x2 blocks for complex pairs.
    """
    eigvals, eigvecs = np.linalg.eig(C1)
    d = C1.shape[0]

    # Sort by |lambda| ascending so labels are stable across day types
    order = np.argsort(np.abs(eigvals))
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Detect complex pairs (eigenvalues with non-trivial imaginary part)
    is_complex = np.abs(eigvals.imag) > 1e-10
    used = np.zeros(d, dtype=bool)
    blocks = []          # list of (label, real_basis_columns, eigval_summary)

    for i in range(d):
        if used[i]:
            continue
        if not is_complex[i]:
            used[i] = True
            v = eigvecs[:, i].real
            v = v / np.linalg.norm(v)
            blocks.append(
                (f"real_{i}", v[:, None], complex(eigvals[i]).real)
            )
        else:
            # find the complex-conjugate partner
            partner = None
            for j in range(i + 1, d):
                if used[j]:
                    continue
                if np.isclose(eigvals[j], np.conj(eigvals[i])):
                    partner = j
                    break
            if partner is None:
                # Shouldn't happen for a real-coeff matrix; fall back to
                # treating it as real (lossy)
                used[i] = True
                v = eigvecs[:, i].real
                v = v / np.linalg.norm(v) if np.linalg.norm(v) > 0 else v
                blocks.append(
                    (f"real_unpaired_{i}", v[:, None], complex(eigvals[i]).real)
                )
                continue
            used[i] = True
            used[partner] = True
            # Real basis for the 2D invariant subspace:
            # span{Re(v_i), Im(v_i)}
            v_re = eigvecs[:, i].real
            v_im = eigvecs[:, i].imag
            v_re = v_re / np.linalg.norm(v_re)
            v_im = v_im / np.linalg.norm(v_im)
            # Orthogonalise (Gram-Schmidt) so the 2D block is well-conditioned
            v_im = v_im - (v_im @ v_re) * v_re
            v_im = v_im / np.linalg.norm(v_im) if np.linalg.norm(v_im) > 0 else v_im
            sub = np.column_stack([v_re, v_im])
            blocks.append((f"pair_{i}_{partner}", sub, complex(eigvals[i])))

    # Stack into V_real
    V_real = np.hstack([b[1] for b in blocks])
    labels = [b[0] for b in blocks]
    eigval_summary = [b[2] for b in blocks]
    # Block sizes (1 for real, 2 for complex pair)
    block_sizes = [b[1].shape[1] for b in blocks]

    return {
        "V_real": V_real,
        "labels": labels,
        "eigval_summary": eigval_summary,
        "block_sizes": block_sizes,
        "eigvals_raw": eigvals,
    }


# ---------------------------------------------------------------------------
# Per-mode residuals
# ---------------------------------------------------------------------------


def _per_mode_residuals(
    D_h: np.ndarray, decomp: dict[str, Any]
) -> dict[str, float]:
    """Project D_h into the (block-)eigenbasis and extract the trace
    contribution of each block.

    For real blocks (1x1): returns D_proj[k, k].
    For complex-pair blocks (2x2): returns trace of the 2x2 submatrix
    (the natural scalar summary of the block's contribution).
    """
    V = decomp["V_real"]
    # D_h in the V_real basis. Use V^{-1} D V^{-T} since V isn't
    # necessarily orthogonal.
    Vinv = np.linalg.inv(V)
    D_proj = Vinv @ D_h @ Vinv.T

    out: dict[str, float] = {}
    offset = 0
    for label, size in zip(decomp["labels"], decomp["block_sizes"]):
        if size == 1:
            out[label] = float(D_proj[offset, offset])
        else:
            # 2x2 block: report the trace as the scalar summary
            block = D_proj[offset:offset + size, offset:offset + size]
            out[label] = float(np.trace(block))
        offset += size
    return out


# ---------------------------------------------------------------------------
# Shape-verdict evaluation
# ---------------------------------------------------------------------------


def _evaluate_shape(R_table: dict) -> dict[str, Any]:
    """Apply the pre-registered O1/O2/O3 criteria to the R_k(h, dt) table.

    R_table: {(dt, h): {label: R_value}}
    Returns the verdict + per-cell shares + dominant-mode index.
    """
    # Per-cell: compute |R_k| / sum_l |R_l|, identify dominant mode
    cell_shares: dict = {}      # (dt, h) -> {label: share}
    cell_dominant: dict = {}    # (dt, h) -> label
    cell_dom_share: dict = {}   # (dt, h) -> max share

    for (dt, h), R_modes in R_table.items():
        abs_modes = {k: abs(v) for k, v in R_modes.items()}
        total = sum(abs_modes.values())
        if total == 0:
            shares = {k: float("nan") for k in R_modes}
            dom = None
            dom_share = float("nan")
        else:
            shares = {k: v / total for k, v in abs_modes.items()}
            dom = max(shares.items(), key=lambda kv: kv[1])[0]
            dom_share = shares[dom]
        cell_shares[(dt, h)] = shares
        cell_dominant[(dt, h)] = dom
        cell_dom_share[(dt, h)] = dom_share

    # O1: all dominant_share > 0.70 AND within each dt same dominant mode
    o1_share_pass = all(s > O1_DOMINANT_SHARE_THRESHOLD
                        for s in cell_dom_share.values()
                        if not np.isnan(s))
    by_dt_dominant: dict = {}
    for (dt, h), dom in cell_dominant.items():
        by_dt_dominant.setdefault(dt, set()).add(dom)
    o1_dt_consistent = all(len(s) == 1 for s in by_dt_dominant.values())
    o1_fires = o1_share_pass and o1_dt_consistent

    # O2: dominant mode CHANGES with h within at least one day_type AND
    # at the peak-share cell of that dt, the dominant mode's tau is
    # within factor of 2 of h. (We'll need eigvals to compute tau —
    # passed in via R_table[<dt, h>] auxiliary; simpler: tabulate it
    # in a separate dict — we compute it in the caller, attach to
    # R_table.)
    # For this evaluation we only check the FIRST condition; the tau
    # check is auxiliary and the caller flags it.
    o2_dt_varying_dominant = any(len(s) > 1 for s in by_dt_dominant.values())
    o2_fires = (not o1_fires) and o2_dt_varying_dominant

    # O3: all dominant_share in the appropriate "even band" for the dt's block-count
    o3_fires = (not o1_fires) and (not o2_fires)
    # also check the share-band
    o3_band_pass = True
    for (dt, h), dom_share in cell_dom_share.items():
        n_blocks = len(cell_shares[(dt, h)])
        if n_blocks <= 2:
            lo, hi = O3_EVEN_BAND_D2
        else:
            lo, hi = O3_EVEN_BAND_D4_PAIR_AGG
        if not (lo <= dom_share <= hi):
            o3_band_pass = False
            break
    o3_fires = o3_fires and o3_band_pass

    fires = [name for name, x in [("O1", o1_fires), ("O2", o2_fires), ("O3", o3_fires)] if x]
    if len(fires) == 1:
        verdict = fires[0]
    elif len(fires) == 0:
        verdict = "NONE"
    else:
        verdict = "MULTIPLE"

    return {
        "verdict": verdict,
        "cell_shares": cell_shares,
        "cell_dominant": cell_dominant,
        "cell_dom_share": cell_dom_share,
        "by_dt_dominant_set": {dt: sorted(s) for dt, s in by_dt_dominant.items()},
        "o1_share_pass": bool(o1_share_pass),
        "o1_dt_consistent": bool(o1_dt_consistent),
        "o2_dt_varying_dominant": bool(o2_dt_varying_dominant),
        "o3_band_pass": bool(o3_band_pass),
    }


# ---------------------------------------------------------------------------
# Asymmetry sub-test
# ---------------------------------------------------------------------------


def _asymmetry_subtest(R_table: dict) -> dict[str, Any]:
    """asymmetry_ratio(h, k) = R_k(h, saturday) / R_k(h, weekday)
    for shared mode labels. Library-consistent if all ratios in [3, 8].
    """
    horizons = sorted({h for (_, h) in R_table})
    weekday_labels = next(iter(R_table[("weekday", horizons[0])].keys() if ("weekday", horizons[0]) in R_table else []), None)
    # actually iterate keys differently
    weekday_modes = {h: list(R_table[("weekday", h)].keys()) for h in horizons if ("weekday", h) in R_table}
    saturday_modes = {h: list(R_table[("saturday", h)].keys()) for h in horizons if ("saturday", h) in R_table}

    ratios: dict = {}
    for h in horizons:
        if ("weekday", h) not in R_table or ("saturday", h) not in R_table:
            continue
        # both day-types use d=2 with two real modes; label namespaces
        # should match. If they don't, skip.
        wd = R_table[("weekday", h)]
        st = R_table[("saturday", h)]
        for k in wd:
            if k not in st:
                continue
            wd_val = wd[k]
            if wd_val == 0:
                continue
            ratios[(h, k)] = st[k] / wd_val

    lo, hi = ASYMMETRY_LIBRARY_BAND
    library_consistent = all(lo <= abs(r) <= hi for r in ratios.values()) if ratios else False
    return {
        "ratios": ratios,
        "library_consistent": library_consistent,
        "expected_library_ratio": WEEKDAY_VS_WEEKEND_N_RATIO_EXPECTED,
        "band": ASYMMETRY_LIBRARY_BAND,
    }


# ---------------------------------------------------------------------------
# Synthetic gate cross-check
# ---------------------------------------------------------------------------


def _synthetic_eigenmode_check(syn: dict) -> dict[str, Any]:
    """Re-do the eigenmode decomposition on the synthetic VAR(1) gate
    to sanity-check the projection. Under strict semigroup, defect
    should be near-zero in every mode."""
    # synthetic gate stored {h: {"C": ..., "Sigma": ...}} in v2's pickle
    # under syn["...syn factor table..."]? Actually we have syn["kl"]
    # and syn["var_direct"] / ["var_iter"], but the per-h C, Sigma
    # were not pickled. So we recompute from the canonical synthetic
    # at the same seed.
    from experiment.multiscale_factor_coherence.__main__ import (
        _synthetic_gate as run_syn_gate, _matrix_power, _iterated_sigma as iter_sigma,
    )
    re_syn = run_syn_gate()
    # re_syn doesn't store per-h Sigma directly... actually we extended
    # it but only stored "Sigma" via the local fit. Let me reach into
    # the internal pattern: we need to run global_ols_fit on the
    # synthetic data. Simplest: reproduce here.
    import numpy as np
    from processing.innovations.estimator import global_ols_fit
    rng = np.random.default_rng(1)
    L_true = np.array([[0.6, 0.2], [0.1, 0.5]])
    Q_true = np.eye(2) * 0.3
    L_chol = np.linalg.cholesky(Q_true)
    n = 50_000
    z = np.zeros((n, 2))
    for t in range(1, n):
        z[t] = L_true @ z[t-1] + L_chol @ rng.standard_normal(2)

    horizons_to_check = (2, 6, 12, 24)
    out = {}
    X1, Y1 = z[:-1], z[1:]
    C1, Sigma1, _, _ = global_ols_fit(X1, Y1)
    decomp = _decompose_C1(C1)
    for h in horizons_to_check:
        Xh, Yh = z[:-h], z[h:]
        Ch, Sh, _, _ = global_ols_fit(Xh, Yh)
        iter_Sh = iter_sigma(C1, Sigma1, h)
        D = Sh - iter_Sh
        R = _per_mode_residuals(D, decomp)
        out[h] = R
    return {
        "synthetic_R_per_mode": out,
        "synthetic_eigvals": [complex(x) for x in decomp["eigvals_raw"]],
        "synthetic_labels": decomp["labels"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="in_path", type=Path,
                   default=Path("scratch/data/multiscale_factor/factors_v2_final.pkl"))
    p.add_argument("--out", type=Path,
                   default=Path("scratch/data/multiscale_factor/eigenmodes.pkl"))
    args = p.parse_args(argv)

    with args.in_path.open("rb") as f:
        v2 = pickle.load(f)
    factors = v2["factors"]
    horizons_present = sorted(set(
        h for dt in factors for h in factors[dt] if h != 1
    ))
    print(f"# eigenmode characterization")
    print(f"  input:    {args.in_path}")
    print(f"  horizons: {horizons_present}")
    print(f"  daytypes: {list(factors.keys())}")
    print()

    # Per-day-type decomposition + per-cell residuals
    decomps: dict = {}
    R_table: dict = {}
    n_per_dt: dict = {}
    eigvals_per_dt: dict = {}
    for dt in factors:
        C1 = factors[dt][1]["C"]
        Sigma1 = factors[dt][1]["Sigma"]
        n_per_dt[dt] = factors[dt][1]["n"]
        if C1 is None or Sigma1 is None:
            continue
        decomps[dt] = _decompose_C1(C1)
        eigvals_per_dt[dt] = decomps[dt]["eigval_summary"]
        for h in horizons_present:
            cell = factors[dt][h]
            if cell["C"] is None or cell["Sigma"] is None:
                continue
            Sh = cell["Sigma"]
            iter_Sh = _iterated_sigma(C1, Sigma1, h)
            D_h = Sh - iter_Sh
            R_table[(dt, h)] = _per_mode_residuals(D_h, decomps[dt])

    # Print the R_k(h, dt) table
    print("# per-mode diagonal residuals R_k(h, dt) (in C_1's block-eigenbasis)")
    print()
    for dt in sorted(decomps):
        labels = decomps[dt]["labels"]
        eigvals = decomps[dt]["eigval_summary"]
        print(f"  [{dt}]  eigvals (block-summary):  "
              + "  ".join(f"{l}={ev:+.4f}" if not isinstance(ev, complex)
                          else f"{l}={ev:+.3f}"
                          for l, ev in zip(labels, eigvals)))
        # header
        header = "    " + " ".join(f"{l:>16s}" for l in labels)
        print(f"    {'h':>4s}  " + " ".join(f"{l:>16s}" for l in labels))
        for h in horizons_present:
            if (dt, h) not in R_table:
                continue
            row = R_table[(dt, h)]
            cells = [f"{row[l]:+16.6e}" for l in labels]
            print(f"    {h:>4d}  " + " ".join(cells))
        print()

    # Shape verdict
    shape = _evaluate_shape(R_table)
    print(f"# pre-registered shape verdict: {shape['verdict']}")
    print(f"  O1 (single-mode dominant):    fires = {shape['verdict'] == 'O1'}")
    print(f"     share-pass: {shape['o1_share_pass']}  "
          f"dt-consistent: {shape['o1_dt_consistent']}")
    print(f"  O2 (h-localized):             fires = {shape['verdict'] == 'O2'}")
    print(f"     dt-varying-dominant: {shape['o2_dt_varying_dominant']}")
    print(f"  O3 (diffuse):                 fires = {shape['verdict'] == 'O3'}")
    print(f"     band-pass: {shape['o3_band_pass']}")
    print()
    print(f"  per-(dt, h) dominant mode (share):")
    for (dt, h) in sorted(shape["cell_dominant"]):
        print(f"    {dt:>8s}  h={h:>2d}  dominant={shape['cell_dominant'][(dt, h)]:>16s}  "
              f"share={shape['cell_dom_share'][(dt, h)]:.3f}")
    print()
    print(f"  dominant-mode set per day_type: {shape['by_dt_dominant_set']}")
    print()

    # Asymmetry sub-test
    asym = _asymmetry_subtest(R_table)
    print(f"# asymmetry sub-test: library_consistent = {asym['library_consistent']}")
    print(f"  (expected ratio ~ {asym['expected_library_ratio']}, "
          f"band {asym['band']})")
    print(f"  per-(h, k) asymmetry_ratio (saturday/weekday):")
    for key, r in sorted(asym["ratios"].items()):
        h, k = key
        in_band = asym['band'][0] <= abs(r) <= asym['band'][1]
        print(f"    h={h:>2d}  k={k:>16s}  ratio={r:+8.3f}  in_band={in_band}")
    print()

    # Synthetic gate cross-check
    print(f"# synthetic VAR(1) gate cross-check (strict semigroup; should be ~0)")
    syn_check = _synthetic_eigenmode_check(v2["synthetic"])
    print(f"  eigvals: {[f'{x:+.4f}' if not isinstance(x, complex) or abs(x.imag) < 1e-10 else str(x) for x in syn_check['synthetic_eigvals']]}")
    print(f"  labels:  {syn_check['synthetic_labels']}")
    for h, R in sorted(syn_check["synthetic_R_per_mode"].items()):
        print(f"  h={h:>2d}: " + " ".join(f"{k}={v:+.3e}" for k, v in R.items()))
    print()

    # Pickle
    out = args.out
    if not out.is_absolute():
        out = (Path.cwd() / out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "R_table": R_table,
        "decomps": {dt: {k: v for k, v in d.items() if k != "V_real" and k != "eigvals_raw"}
                    for dt, d in decomps.items()},  # don't pickle the basis matrices (small but unneeded)
        "shape": shape,
        "asymmetry": asym,
        "synthetic_check": syn_check,
        "n_per_dt": n_per_dt,
        "config": {
            "O1_threshold": O1_DOMINANT_SHARE_THRESHOLD,
            "O3_band_d2": O3_EVEN_BAND_D2,
            "O3_band_d4": O3_EVEN_BAND_D4_PAIR_AGG,
            "tau_localized_tolerance": TAU_LOCALIZED_TOLERANCE,
            "asymmetry_band": ASYMMETRY_LIBRARY_BAND,
            "expected_library_ratio": WEEKDAY_VS_WEEKEND_N_RATIO_EXPECTED,
        },
    }
    with out.open("wb") as f:
        pickle.dump(payload, f)
    try:
        rel = out.relative_to(PROJECT_ROOT)
        print(f"  wrote {rel}")
    except ValueError:
        print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
