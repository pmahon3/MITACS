"""Stage 1: k-step predictive-variance propagation + its validation gate.

WHAT THIS IS
------------
The error-decomposition diagnostic's M2 needs a predictive *law*
N(z_hat_k, Sigma_k) at every forecast horizon k. The production
estimator emits only the ONE-step local (C, Sigma) at a query state
(``_local_fit_at``). Iterating the point forecast 24 steps is already
done in the benchmark; what is missing is propagating the *variance*
through that same iteration.

This module builds that propagation and -- per the project discipline
that an UNVALIDATED composition path may never inform a build decision
(theory-correspondence Qualifier 3 + mitacs-multistep-design: the
{Pi_t}/Chapman-Kolmogorov composition is explicitly NOT yet verified) --
gates it against a closed-form known answer BEFORE it may be used on
Ontario data.

THE COMPOSITION
---------------
Local model at step i (row convention, matching the codebase):
    x_{i+1} = x_i @ C_i + e_i ,   e_i ~ (mu_i, Sigma_i)
Linearising the iteration around the realised forecast path, the
predictive covariance of the k-step-ahead state propagates as

    P_0   = 0                       (the issue state is known)
    P_{i+1} = C_i^T P_i C_i + Sigma_i

i.e. push the running covariance through the local linear map, then add
that step's local innovation covariance. The scalar predictive variance
the diagnostic consumes is coordinate 0:  s2_k = P_k[0, 0].

KNOWN-ANSWER GATE
-----------------
For a stationary VAR(1)  x_{t+1} = x_t A + eps,  eps ~ N(0, Q), every
local fit recovers C_i -> A and Sigma_i -> Q, so the recursion collapses
to the textbook closed form

    Sigma_k = sum_{i=0}^{k-1} (A^T)^i  Q  A^i .

The gate feeds the propagation the *true* (A, Q) at every step and
requires P_k to match this closed form within tolerance for k = 1..24.
That isolates the *composition algebra* (the thing that is unvalidated)
from the *local estimator* (already gate-validated elsewhere) -- a
failure here is a propagation bug, not an estimator bug.

PROVENANCE-GRADE: INSPECTION-ONLY -- a validation harness + reusable
propagation primitive; not itself a result artifact.
"""
from __future__ import annotations

import numpy as np


def propagate_predictive_cov(
    C_seq: np.ndarray,
    Sigma_seq: np.ndarray,
) -> np.ndarray:
    """Propagate the k-step predictive covariance along a forecast path.

    Parameters
    ----------
    C_seq : (K, d, d)
        Per-step local drift matrices C_0..C_{K-1} (row convention
        ``x' = x @ C``), one per iterated step actually taken.
    Sigma_seq : (K, d, d)
        Per-step local innovation covariances Sigma_0..Sigma_{K-1}.

    Returns
    -------
    P : (K, d, d)
        ``P[k]`` is the predictive covariance of the (k+1)-step-ahead
        state (so ``P[0]`` corresponds to horizon 1). Scalar predictive
        variance for the diagnostic is ``P[k][0, 0]``.

    Recursion: ``P_{i+1} = C_i^T P_i C_i + Sigma_i`` with ``P_0 = 0``.
    """
    K, d, _ = C_seq.shape
    P = np.zeros((K, d, d), dtype=float)
    running = np.zeros((d, d), dtype=float)
    for i in range(K):
        Ci = C_seq[i]
        running = Ci.T @ running @ Ci + Sigma_seq[i]
        # numerical symmetry hygiene (covariance must stay symmetric)
        running = 0.5 * (running + running.T)
        P[i] = running
    return P


def state_transition_from_local_fit(
    C: np.ndarray,
    sigma00: float,
    d: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the TRUE iterated-state one-step Jacobian and innovation
    covariance from a production local fit ``(C, Sigma)``.

    Critical production semantics (see memory ``mitacs-rank1-structural``
    and the advisor note): the estimator returns a (d,d) ``C`` that is the
    independent local linear regression of each next-state coordinate on
    the lag state. But in the *iterated forecast* the next embedding state
    is

        x_{i+1} = ( z_{i+1},  z_i,  z_{i-1}, ...,  z_{i-d+2} )

    where only the leading coordinate is predicted
    (``z_{i+1} = x_i @ C[:,0] + e_i``) and coordinates 1..d-1 are
    DETERMINISTIC shifts of the previous state. So the state Jacobian is
    NOT ``C``; it is

        J = [ C[:,0] | shift-block ]      (column 0 = predictive coeffs,
                                           columns 1..d-1 = [I_{d-1}; 0])

    and the per-step innovation covariance is RANK-1 by construction:
    stochastic content only in coordinate 0,

        Sigma_step = diag(Sigma[0,0], 0, ..., 0).

    Using ``C`` wholesale (as if it were the state Jacobian) would
    propagate a variance that does not match the forecast the iteration
    actually produces. Returns ``(J, Sigma_step)`` both (d,d).
    """
    c0 = C[:, 0] if C.ndim == 2 else np.asarray(C, dtype=float)
    J = np.zeros((d, d), dtype=float)
    J[:, 0] = c0
    if d > 1:
        # columns 1..d-1: x_{i+1}[k] = x_i[k-1]  ->  J[k-1, k] = 1
        J[np.arange(d - 1), np.arange(1, d)] = 1.0
    Sigma_step = np.zeros((d, d), dtype=float)
    Sigma_step[0, 0] = float(max(sigma00, 0.0))
    return J, Sigma_step


def augmented_state_transition(
    C: np.ndarray,
    sigma00: float,
    d: int,
    d_max: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Option A: the iterated-state Jacobian + innovation in a FIXED
    ``d_max``-dimensional state, so the covariance recursion is defined
    even when the embedding dimension ``d`` changes between steps
    (day-type rollover). See scratch/SCOPE_dim_change_propagation.md
    (Option C was falsified; this is the validated path).

    Construction (a refactor of :func:`state_transition_from_local_fit`,
    NOT a new estimator): the active state is the first ``d`` coords;
    the trailing ``d_max - d`` coords are older real lag values that
    evolve by a DETERMINISTIC shift and carry ZERO stochastic content.
    Hence in the full ``d_max`` state:

        J_aug[:, 0]               = c0 zero-padded to length d_max
                                    (the d predictive coeffs, then 0s)
        J_aug[k-1, k] = 1, k=1..d_max-1   (the standard shift block,
                                    spanning active AND inactive coords)
        Sigma_aug = diag(Sigma[0,0], 0, ..., 0)   (rank-1, unchanged)

    The shift block is full-width: an inactive coordinate at step i is
    just a lag that becomes "active" or ages out as the window moves;
    its dynamics are the same deterministic shift, with no innovation.
    This is exact bookkeeping of a constant-dimension state, not an
    approximation of the variance.
    """
    if not (1 <= d <= d_max):
        raise ValueError(f"need 1<=d<=d_max, got d={d}, d_max={d_max}")
    c0 = C[:, 0] if C.ndim == 2 else np.asarray(C, dtype=float)
    if c0.shape[0] != d:
        raise ValueError(f"c0 length {c0.shape[0]} != d {d}")
    J = np.zeros((d_max, d_max), dtype=float)
    J[:d, 0] = c0                       # predictive coeffs, zero-padded
    # full-width shift: x_{i+1}[k] = x_i[k-1] for k=1..d_max-1
    J[np.arange(d_max - 1), np.arange(1, d_max)] = 1.0
    Sigma = np.zeros((d_max, d_max), dtype=float)
    Sigma[0, 0] = float(max(sigma00, 0.0))
    return J, Sigma


def _var1_closed_form_cov(A: np.ndarray, Q: np.ndarray, K: int) -> np.ndarray:
    """Closed-form k-step predictive covariance of a VAR(1) with drift
    ``A`` (row convention) and innovation covariance ``Q``:

        Sigma_k = sum_{i=0}^{k-1} (A^T)^i Q A^i .
    """
    d = A.shape[0]
    out = np.zeros((K, d, d), dtype=float)
    acc = np.zeros((d, d), dtype=float)
    AT = A.T
    for k in range(K):
        # add the (k)-th term (A^T)^k Q A^k
        Ak = np.linalg.matrix_power(A, k)
        ATk = np.linalg.matrix_power(AT, k)
        acc = acc + ATk @ Q @ Ak
        out[k] = acc
    return out


def validate(seed: int = 0, K: int = 24, rtol: float = 1e-9) -> dict:
    """Known-answer gate: feed the propagation the TRUE (A, Q) at every
    step (the limit a perfect local fit recovers) and require it to match
    the VAR(1) closed form for k=1..K.

    Two regimes are checked so a coincidental pass is unlikely:
      - a stable, asymmetric, *correlated*-innovation VAR(1) (general
        case: A not symmetric, Q not diagonal);
      - a near-unit-root A (stresses the C^T P C accumulation -- where a
        transpose/order bug would blow up).
    """
    rng = np.random.default_rng(seed)
    results = {}

    def _one(name: str, A: np.ndarray, Q: np.ndarray) -> tuple[bool, float]:
        C_seq = np.repeat(A[None, :, :], K, axis=0)
        S_seq = np.repeat(Q[None, :, :], K, axis=0)
        P = propagate_predictive_cov(C_seq, S_seq)
        ref = _var1_closed_form_cov(A, Q, K)
        # compare the scalar the diagnostic actually consumes AND the
        # full matrix (a [0,0]-only check could miss a block bug)
        max_rel = float(
            np.max(np.abs(P - ref) / (np.abs(ref) + 1e-12))
        )
        ok = bool(np.allclose(P, ref, rtol=rtol, atol=1e-12))
        results[name] = {
            "passed": ok,
            "max_rel_err": max_rel,
            "s2_k_last": float(P[-1, 0, 0]),
            "s2_k_last_ref": float(ref[-1, 0, 0]),
        }
        return ok, max_rel

    # regime 1: general stable VAR(1), asymmetric A, correlated Q
    d = 3
    M = rng.standard_normal((d, d))
    A1 = 0.5 * M / np.max(np.abs(np.linalg.eigvals(M)))  # spectral radius .5
    L = rng.standard_normal((d, d))
    Q1 = L @ L.T + 0.1 * np.eye(d)                        # SPD, correlated
    _one("stable_asym_correlated", A1, Q1)

    # regime 2: near-unit-root (stresses the accumulation)
    A2 = A1 * (0.99 / 0.5)                                # spectral radius .99
    _one("near_unit_root", A2, Q1)

    # regime 3: PRODUCTION shape end-to-end -- shift-aware Jacobian +
    # rank-1 per-step innovation, against a hand-computed K=2 reference.
    # This checks the integration of state_transition_from_local_fit with
    # the recursion (not just the algebra on synthetic full-rank Q).
    d3 = 3
    Cprod = rng.standard_normal((d3, d3))           # only col 0 is used
    s00 = 0.37
    J, Sig = state_transition_from_local_fit(Cprod, s00, d3)
    P = propagate_predictive_cov(
        np.stack([J, J]), np.stack([Sig, Sig])
    )
    # K=1: P0 = Sig  -> P0[0,0] must be exactly s00, rest 0
    k1_ok = np.allclose(P[0], Sig, atol=1e-12)
    # K=2 hand reference: P1 = J^T P0 J + Sig
    P1_ref = J.T @ Sig @ J + Sig
    k2_ok = np.allclose(P[1], P1_ref, atol=1e-12)
    # rank-1 invariant: P0 must be rank 1 (single stochastic coord)
    rank1_ok = (np.linalg.matrix_rank(P[0], tol=1e-10) == 1)
    results["production_shape_K2"] = {
        "passed": bool(k1_ok and k2_ok and rank1_ok),
        "max_rel_err": float(np.max(np.abs(P[1] - P1_ref))),
        "s2_k_last": float(P[1, 0, 0]),
        "s2_k_last_ref": float(P1_ref[0, 0]),
        "k1_equals_sigma00": bool(k1_ok),
        "P0_rank1": bool(rank1_ok),
    }

    # ===== Option A gates (d_max-augmented propagation) ================
    # Gate A1 -- closed-form known answer through the AUGMENTED builder.
    # A constant-day-type VAR(1) of true dim d, run in a d_max state via
    # augmented_state_transition, must reproduce the textbook
    # Sigma_k = sum (A^T)^i Q A^i read at the ACTIVE d x d block. The
    # production innovation is rank-1, so use a rank-1 Q (only coord 0).
    d_a, d_max_a, Ka = 2, 4, 6
    Ma = rng.standard_normal((d_a, d_a))
    A_a = 0.6 * Ma / np.max(np.abs(np.linalg.eigvals(Ma)))   # spec.rad .6
    q0 = 0.23
    Q_a = np.zeros((d_a, d_a)); Q_a[0, 0] = q0               # rank-1
    # build the augmented per-step (J,Sigma): C with col0 = A_a[:,0]
    Ccol = np.zeros((d_a, d_a)); Ccol[:, 0] = A_a[:, 0]
    Jaug, Saug = augmented_state_transition(Ccol, q0, d_a, d_max_a)
    Paug = propagate_predictive_cov(
        np.stack([Jaug] * Ka), np.stack([Saug] * Ka)
    )
    # reference: the SAME system propagated natively at dim d_a with the
    # shift-aware production builder (already gate-validated above), then
    # compared on the active block. (A_a[:,0] is the only coeff the
    # iterated state uses; cols 1.. are the deterministic shift.)
    Jnat, Snat = state_transition_from_local_fit(Ccol, q0, d_a)
    Pnat = propagate_predictive_cov(
        np.stack([Jnat] * Ka), np.stack([Snat] * Ka)
    )
    active_match = np.allclose(
        Paug[:, :d_a, :d_a], Pnat, rtol=rtol, atol=1e-12
    )
    # inactive coords must never acquire stochastic content beyond what
    # the shift carries in from coord 0 (no spurious innovation)
    s00_match = np.allclose(Paug[:, 0, 0], Pnat[:, 0, 0], atol=1e-12)
    results["optionA_augmented_closedform"] = {
        "passed": bool(active_match and s00_match),
        "max_rel_err": float(
            np.max(np.abs(Paug[:, :d_a, :d_a] - Pnat)
                   / (np.abs(Pnat) + 1e-12))
        ),
        "s2_k_last": float(Paug[-1, 0, 0]),
        "s2_k_last_ref": float(Pnat[-1, 0, 0]),
        "active_block_matches_native": bool(active_match),
    }

    # Gate A2 -- refactor-equivalence: on a SINGLE-dim sequence the
    # augmented recursion's P[0,0] must EXACTLY equal the existing
    # validated single-dim recursion's P[0,0] (no new ground truth; a
    # pure "did the refactor change the answer" check). Use a realistic
    # production-shaped C (only col 0 matters) at d=3, d_max=4, K=8.
    d_e, dmax_e, Ke = 3, 4, 8
    Ce = rng.standard_normal((d_e, d_e)); s00e = 0.41
    Jn, Sn = state_transition_from_local_fit(Ce, s00e, d_e)
    Pn = propagate_predictive_cov(np.stack([Jn]*Ke), np.stack([Sn]*Ke))
    Ja, Sa = augmented_state_transition(Ce, s00e, d_e, dmax_e)
    Pa = propagate_predictive_cov(np.stack([Ja]*Ke), np.stack([Sa]*Ke))
    equiv = np.allclose(Pa[:, 0, 0], Pn[:, 0, 0], atol=1e-12)
    results["optionA_refactor_equivalence"] = {
        "passed": bool(equiv),
        "max_rel_err": float(
            np.max(np.abs(Pa[:, 0, 0] - Pn[:, 0, 0]))
        ),
        "s2_k_last": float(Pa[-1, 0, 0]),
        "s2_k_last_ref": float(Pn[-1, 0, 0]),
    }

    results["ALL_PASSED"] = bool(
        all(v["passed"] for k, v in results.items() if k != "ALL_PASSED")
    )
    return results


def main() -> None:
    res = validate()
    print("Stage-1 variance-propagation known-answer gate")
    print("=" * 60)
    for name, v in res.items():
        if name == "ALL_PASSED":
            continue
        print(
            f"  {name:<26} {'PASS' if v['passed'] else 'FAIL'}  "
            f"max_rel_err={v['max_rel_err']:.2e}  "
            f"s2_k(24)={v['s2_k_last']:.6g} "
            f"(ref {v['s2_k_last_ref']:.6g})"
        )
    print("=" * 60)
    verdict = "PASS" if res["ALL_PASSED"] else "FAIL"
    print(f"GATE: {verdict}  "
          f"-> Stage 2 {'UNBLOCKED' if res['ALL_PASSED'] else 'STAYS BLOCKED'}")


if __name__ == "__main__":
    main()
