"""Q1 of the distributional-class-thread: Student-t vs Gaussian
predictive distribution.

Pre-registered in
``notes/preregistrations/2026-05-26_q1-student-t-vs-gaussian/phase_a.yaml``.
Thread parent: ``2026-05-26_distributional-class-thread/thread.yaml``
(node Q1, root).

Math reference: ``writeup/tex/missing_content_memo.tex`` §3 (Gaussian
baseline M0) and §4.4 (Student-t extension M1, M2; SMC iteration;
1D Wasserstein-2 divergence).

Tests:

  * M0  -- Gaussian baseline (reads v2 factor pickle)
  * M1  -- Student-t kernel only (climatology unchanged from M0)
  * M2  -- Student-t kernel + Student-t climatology refit per (m, h)

Outputs:

  * marginal_PIT_chi2 per model on post-cutoff data
  * fitted nu per day-type (M1) and per (m, h) bin (M2)
  * D_RATIO_t at (h=12, weekday) under M1 via SMC + 1D W2
  * synthetic Student-t-VAR gate

CLI::

    python -m experiment.distributional_class_q1 \\
        --in scratch/data/multiscale_factor/factors_v2_final.pkl \\
        --out scratch/data/distributional_class_q1/q1.pkl
"""
