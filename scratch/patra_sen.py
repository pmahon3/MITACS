"""Patra-Sen 2016 two-component mixture estimator (arXiv:1204.5488).

Setting: data X_1,...,X_n ~ F(x) = α F_s(x) + (1-α) F_b(x), where F_b
is a fully-specified known CDF, F_s is unknown, and α ∈ [0,1] is the
unknown mixing proportion to be estimated.

For our project: F_b = N(0,1) (the predicted-Gaussian null for
standardised residuals), and α̂_0 is the minimum contamination
fraction compatible with the data.

The estimator (eq. 7 of Patra-Sen):
    α̂_0 = inf { γ ∈ (0,1] : γ·d_n(F̂γ_s,n, F̌γ_s,n) ≤ c_n/√n }

where:
  F̂γ_s,n = (F_n − (1−γ)·F_b) / γ           (naive estimator, eq. 2)
  F̌γ_s,n = isotonic projection of F̂γ_s,n   (PAVA, eq. 3)
  d_n²(g,h) = (1/n)·Σ_i {g(X_i) − h(X_i)}²  (L²(F_n) distance)
  c_n = 0.1·log(log n)  for the point estimate (§3.2 recommendation)
  c_n = 0.6792           for the 95% lower CI (§4 Anderson-Darling
                          quantile; Theorem 5)

Lemma 9: γ·d_n is non-increasing convex in γ ∈ (0,1], so the
inf-over-γ is a well-defined crossing point findable by bisection.

PROVENANCE-GRADE: INSPECTION-ONLY (implementation; validate against
synthetic before applying to Ontario).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression


@dataclass(frozen=True)
class PatraSenResult:
    """Output of the Patra-Sen estimator."""
    alpha_hat: float           # point estimate (c_n = 0.1·log log n)
    alpha_L_95: float          # 95% lower confidence bound (c_n = 0.6792)
    n: int
    c_n_point: float
    threshold_point: float     # c_n / sqrt(n) at the point estimate
    threshold_L95: float       # 0.6792 / sqrt(n)
    gamma_grid: np.ndarray     # the grid used
    criterion_curve: np.ndarray  # γ·d_n(γ) over the grid


def _criterion_at_gamma(gamma: float, F_n_vals: np.ndarray,
                        F_b_vals: np.ndarray, eps: float = 1e-12
                        ) -> float:
    """Compute γ·d_n(F̂γ_s,n, F̌γ_s,n) at a single γ.

    Inputs:
      gamma:    candidate mixing proportion in (0, 1]
      F_n_vals: empirical CDF evaluated at sorted data, i.e. i/n
      F_b_vals: known CDF F_b evaluated at the same sorted data
    """
    if gamma <= 0:
        # Limit case (eq. 9): γ·d_n → d_n(F_n, F_b)
        return float(np.sqrt(np.mean((F_n_vals - F_b_vals) ** 2)))
    if gamma >= 1.0 - eps:
        # F̂_n^1 = F_n exactly; isotonic projection is itself; distance = 0
        return 0.0

    # Naive estimator (eq. 2): F̂γ = (F_n − (1−γ)·F_b) / γ
    F_hat = (F_n_vals - (1.0 - gamma) * F_b_vals) / gamma

    # Isotonic projection onto monotone non-decreasing in [0,1].
    # The paper's F̌γ is the CDF minimising eq. (3); per Lemma 1 it's
    # the right-hand slope of the greatest convex minorant, computable
    # as a PAVA on the sorted data with monotonicity constraint, then
    # clipped to [0,1] (the paper notes: F̌γ = min(max(F̃γ, 0), 1)).
    x_idx = np.arange(len(F_hat))   # sorted data positions; iso reg on index
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True)
    F_check = iso.fit_transform(x_idx, F_hat)

    # γ·d_n = γ·sqrt(mean((F̂ - F̌)²))
    d_n = float(np.sqrt(np.mean((F_hat - F_check) ** 2)))
    return gamma * d_n


def _find_smallest_crossing(gamma_grid: np.ndarray,
                            criterion: np.ndarray,
                            threshold: float) -> float:
    """Smallest γ in the grid with criterion(γ) ≤ threshold.

    The criterion is non-increasing (Lemma 9), so this is the first γ
    where the threshold is met. Returns 1.0 if never met (shouldn't
    happen since γ=1 gives criterion 0).
    """
    below = criterion <= threshold
    if not np.any(below):
        return 1.0
    idx = int(np.argmax(below))   # first True
    return float(gamma_grid[idx])


def patra_sen(data: np.ndarray, F_b,
              gamma_grid_size: int = 600,
              beta: float = 0.05) -> PatraSenResult:
    """Estimate the mixing proportion α_0 in F = α·F_s + (1-α)·F_b.

    Args:
      data: 1-D array of n i.i.d. observations from F
      F_b:  callable; F_b(x) returns the known CDF evaluated at x
      gamma_grid_size: resolution of the γ grid in (0,1]
      beta: 1 - confidence level for the lower bound (default 0.05 -> 95% CI)

    Returns:
      PatraSenResult with alpha_hat (point), alpha_L_95 (lower CI), etc.
    """
    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 30:
        raise ValueError(f"need n >= 30, got {n}")

    # Sort + compute F_n and F_b values
    x_sorted = np.sort(x)
    F_n_vals = (np.arange(1, n + 1)) / n           # empirical CDF
    F_b_vals = np.asarray([F_b(xi) for xi in x_sorted], dtype=float)

    # Grid over γ in (0,1]. Use a logarithmic-near-zero grid since the
    # criterion is steepest near small γ.
    gamma_grid = np.linspace(1.0 / gamma_grid_size, 1.0, gamma_grid_size)

    # Evaluate criterion γ·d_n(γ) at each grid point
    criterion = np.empty(gamma_grid_size)
    for i, g in enumerate(gamma_grid):
        criterion[i] = _criterion_at_gamma(g, F_n_vals, F_b_vals)

    # Point estimate (c_n = 0.1·log log n)
    c_n_point = 0.1 * np.log(np.log(max(n, np.e + 1)))
    thresh_point = c_n_point / np.sqrt(n)
    alpha_hat = _find_smallest_crossing(gamma_grid, criterion, thresh_point)

    # 95% lower CI (c_n = Anderson-Darling 95% quantile of CvM = 0.6792)
    # If beta != 0.05, would need other quantiles; here we hard-code 95%.
    if abs(beta - 0.05) > 1e-9:
        # Asymptotic CvM quantiles (from Anderson-Darling 1952 table):
        # Could extend; for now insist on 95%.
        raise NotImplementedError(
            f"only beta=0.05 (95% CI) supported; got {beta}"
        )
    c_n_L95 = 0.6792
    thresh_L95 = c_n_L95 / np.sqrt(n)
    alpha_L_95 = _find_smallest_crossing(gamma_grid, criterion, thresh_L95)

    return PatraSenResult(
        alpha_hat=alpha_hat,
        alpha_L_95=alpha_L_95,
        n=int(n),
        c_n_point=float(c_n_point),
        threshold_point=float(thresh_point),
        threshold_L95=float(thresh_L95),
        gamma_grid=gamma_grid,
        criterion_curve=criterion,
    )


# ===================================================================
#  Synthetic validation
# ===================================================================

def _gen_mixture(n: int, alpha: float, F_s_sampler, F_b_sampler,
                 rng) -> np.ndarray:
    """Sample n from α·F_s + (1-α)·F_b."""
    n_signal = rng.binomial(n, alpha)
    return np.concatenate([
        F_s_sampler(n_signal, rng),
        F_b_sampler(n - n_signal, rng),
    ])


def _validate_synthetic():
    """Replicate the Patra-Sen Fig. 1 setting:
       Setting I:  α=0.1, F_s = N(2,1), F_b = N(0,1).  n=5000.
       Setting II: α=0.1, F_s = Beta(1,10), F_b = Uniform(0,1). n=5000."""
    print("=" * 76)
    print("Patra-Sen estimator: synthetic validation (paper's Fig. 1)")
    print("=" * 76)
    rng = np.random.default_rng(20260520)

    # Setting I: N(2,1) signal vs N(0,1) background
    print("\nSetting I: α=0.1, F_s=N(2,1), F_b=N(0,1), n=5000")
    print("  Paper Fig 1 (left panel): clear elbow at γ≈0.10")
    data_I = _gen_mixture(
        n=5000, alpha=0.10,
        F_s_sampler=lambda k, r: r.normal(2.0, 1.0, k),
        F_b_sampler=lambda k, r: r.normal(0.0, 1.0, k),
        rng=rng,
    )
    res_I = patra_sen(data_I, F_b=lambda x: norm.cdf(x, loc=0.0, scale=1.0))
    print(f"  → α̂ = {res_I.alpha_hat:.4f}  "
          f"α̂_L^{{95%}} = {res_I.alpha_L_95:.4f}  "
          f"(true α = 0.10)")

    # Setting II: Beta(1,10) signal vs Uniform(0,1) background
    print("\nSetting II: α=0.1, F_s=Beta(1,10), F_b=Uniform(0,1), n=5000")
    print("  Paper Fig 1 (right panel): clear elbow at γ≈0.10")
    data_II = _gen_mixture(
        n=5000, alpha=0.10,
        F_s_sampler=lambda k, r: r.beta(1.0, 10.0, k),
        F_b_sampler=lambda k, r: r.uniform(0.0, 1.0, k),
        rng=rng,
    )
    res_II = patra_sen(data_II, F_b=lambda x: max(0.0, min(1.0, x)))
    print(f"  → α̂ = {res_II.alpha_hat:.4f}  "
          f"α̂_L^{{95%}} = {res_II.alpha_L_95:.4f}  "
          f"(true α = 0.10)")

    # Null setting: no contamination
    print("\nNull: α=0.0 (no signal), all from N(0,1), n=5000")
    print("  Expectation: α̂ ≈ 0 and α̂_L^{95%} = 0 (with prob ≥ 0.95)")
    data_null = rng.normal(0, 1, 5000)
    res_null = patra_sen(data_null, F_b=lambda x: norm.cdf(x, loc=0.0, scale=1.0))
    print(f"  → α̂ = {res_null.alpha_hat:.4f}  "
          f"α̂_L^{{95%}} = {res_null.alpha_L_95:.4f}")

    # Heavy-tail setting (the regime that matters for us)
    print("\nHeavy-tail setting (our target regime):")
    print("  α=0.05, F_s=N(0,3) heavy-tailed contamination, F_b=N(0,1), n=10000")
    data_ht = _gen_mixture(
        n=10000, alpha=0.05,
        F_s_sampler=lambda k, r: r.normal(0.0, 3.0, k),   # same mean, 3x std
        F_b_sampler=lambda k, r: r.normal(0.0, 1.0, k),
        rng=rng,
    )
    res_ht = patra_sen(data_ht, F_b=lambda x: norm.cdf(x, loc=0.0, scale=1.0))
    print(f"  → α̂ = {res_ht.alpha_hat:.4f}  "
          f"α̂_L^{{95%}} = {res_ht.alpha_L_95:.4f}  "
          f"(true α = 0.05)")
    print(f"  Note: contamination here has same mean as F_b but heavier tails;")
    print(f"  identifiability is partial (ess inf f_s/f_b > 0), so α̂ < α expected.")

    return res_I, res_II, res_null, res_ht


if __name__ == "__main__":
    _validate_synthetic()
