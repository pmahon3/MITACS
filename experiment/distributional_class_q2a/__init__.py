"""Q2A of the distributional-class-thread: richer-family-mixture-or-
nonparametric predictive distributions vs Q1's Student-t baseline.

Pre-registered in
``notes/preregistrations/2026-05-26_q2a-richer-family-mixture-or-
nonparametric/phase_a.yaml``.
Thread parent: ``2026-05-26_distributional-class-thread/thread.yaml``
(node Q2A, parent_branch R-D).

Math reference: ``writeup/tex/missing_content_memo.tex`` §4.4 + §5
(distributional-class ladder beyond a single Student-t nu;
finite-mixture-of-Gaussians and KDE residual laws).

Tests (analogues of Q1's M0/M1/M2; only the residual law swaps):

  * M3  -- Drift (C_1) + symmetric centred mix-2-Gaussians residual
  * M4  -- Drift (C_1) + symmetric centred mix-3-Gaussians residual
  * M5  -- Drift (C_1) + KDE residual law (Silverman bandwidth)

Verdict metric (per phase_a):

    effective_chi2 = min(chi2_M3, chi2_M4, chi2_M5)

against the post-cutoff marginal PIT, with paired-day bootstrap CIs.

Cuts (inherited from the Q2A synthetic gate METHOD/DESIGN artifact,
body_sha256 ``2bab514e1049a89643ab69f45568400969c78c37939c3b188166e1a89c775c42``):

  R-A2 (corroboration): effective_chi2 <= 228.44
  R-B2 (ambiguous):     228.44 < effective_chi2 <= 2284.44
  R-C2 (falsification): effective_chi2 > 2284.44

CLI::

    python -m experiment.distributional_class_q2a \\
        --in scratch/data/multiscale_factor/factors_v2_final.pkl \\
        --out scratch/data/distributional_class_q2a/q2a.pkl \\
        [--emit-result]
"""
