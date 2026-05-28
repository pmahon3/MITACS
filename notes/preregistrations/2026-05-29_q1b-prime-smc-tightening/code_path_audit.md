body_sha256: null
checked_by: code-path-auditor agent (.claude/agents/code-path-auditor.md)
dependent_variable_provenance:
  better_chi2:
    feeds_from: same chain as cumulative_chi2_improvement_above_Z_c
    formula: min(chi2_Z2_b, chi2_Z2_c)
    verdict: PRODUCTION-PATH
  cumulative_chi2_improvement_above_Z_c:
    feeds_from: "mixture_2_gaussian_mle_fit (line 356, per-cell head-to-head fits) → _pit_for_zc_z2_candidate (line 1527, M=args.n_particles=400) → _marginal_pit_chi2 (line 1529, Q2A production helper)"
    formula: Q1A_M3_Z_c_CHI2_SETTLED (952.5167; hardcoded line 127) - better_chi2_Z2_c
    verdict: PRODUCTION-PATH
  cumulative_chi2_improvement_above_Z_c_ci95:
    feeds_from: same per-resample chain as above; bootstrap loop (lines ~1575-1592) re-samples u_PIT rows per-delivery-day and recomputes chi^2 via _marginal_pit_chi2 (Q2A production helper)
    formula: 1000-rep paired-day bootstrap percentile [2.5, 97.5] on cumulative_improvement
    note: 'The bootstrap loop resamples PIT rows row-wise and recomputes
      chi^2; it does NOT re-fit M3 per resample (a pre-existing
      description-vs-implementation note inherited from Q1B''s
      audited script, which Q1B SETTLED on). Out of scope for this
      audit per the task framing (no computational change; only
      artifact-pinning metadata + docstring touched).'
    verdict: PRODUCTION-PATH
  repro_chi2_no_z:
    elevation_note: 'Phase_a (body_sha256 7ed3eea3...) elevates this from gate-
      only (Q1B''s treatment) to a DEPENDENT variable. The
      production-stamped provenance is identical; the elevation
      changes how the metric is reported (point + |diff| + tol-pass/
      fail flag), not how it is computed.'
    feeds_from: "mixture_2_gaussian_mle_fit (line 406) → _pit_for_family (Q2A production helper, called via _pit_no_z_reproduction line 528) → _marginal_pit_chi2 (Q2A production helper)"
    formula: marginal PIT chi^2 of _pit_no_z_reproduction output
    verdict: PRODUCTION-PATH
  repro_chi2_zc:
    elevation_note: 'Same as repro_chi2_no_z: elevated from gate-only to DEPENDENT
      in Q1B'' phase_a; provenance unchanged from Q1B''s audited
      script.'
    feeds_from: "mixture_2_gaussian_mle_fit (line 379) → _pit_zc_only_reproduction custom SMC loop (line 558+; orchestration over production samplers/fitters) → _marginal_pit_chi2 (Q2A production helper)"
    formula: marginal PIT chi^2 of _pit_zc_only_reproduction output
    verdict: PRODUCTION-PATH
flagged_symbols: []
git_clean: true
git_sha: 8cd2935764da1daf24e06efc317f61a56054e1e4
inheritance_note: "Q1B's script imports four helpers from\nexperiment.distributional_class_q2a.__main__ — a SETTLED\npredecessor experiment (S9, distributional-class-thread):\n\n  - _build_library_for_daytype\n  - _make_mixture_2_sampler\n  - _marginal_pit_chi2\n  - _smc_iterate_family\n  - _pit_for_family (function-local import)\n\nAnd one helper from a SETTLED predecessor in the same thread\n(S11, P1):\n\n  - _time_of_day_label (canonical Z2_c binning)\n\nThese names do NOT appear in the static auditor's whitelist\n(which is limited to canonical production surfaces:\nexperiment._actuals, experiment.backtest, experiment.predict,\nexperiment.freeze, experiment.provenance, and\nprocessing.innovations.estimator). Per Q1B's audit precedent\n(commit 1736e71 SETTLED), settled-predecessor experiment helpers\ncount as PRODUCTION for descendant nodes in the same thread; the\ninheritance is via the SETTLED finding's stamped result, not\nvia the whitelist proper. The custom SMC propagation loops in\n_pit_zc_only_reproduction and _pit_for_zc_z2_candidate are\norchestration over production-stamped components (per-cell fits\nvia mixture_2_gaussian_mle_fit, per-step draws via\n_make_mixture_2_sampler, terminal scoring via\n_marginal_pit_chi2) — NOT reimplementation of fit / sampler /\nchi^2. The static auditor surfaces zero inline definitions of\n*mle*, *sampler*, *chi2*, or *fit* helpers in the script.\n"
n_particles_branch_analysis: "Q1B' flips --n-particles from 200 to 400 at the CLI. There are\nNO N-conditional branches in the script (no `if args.n_particles\n== X` or similar). args.n_particles is a pure scalar threaded\nthrough the three SMC call sites (1430, 1460, 1524) as the `M`\nargument, used in the helpers as the tile-count for\n`np.tile(x_state, (M, 1))` and as the M-particle sample shape in\n`sampler(M, rng)`. Doubling M can only:\n\n  (a) increase the per-step computational cost roughly linearly,\n  (b) reduce the per-step SMC sampling variance by 1/sqrt(N) per\n      the central-limit-theorem argument the phase_a's\n      threshold-derivations cite, and\n  (c) drift the seeded SMC trajectory from Q1B's N=200 exact\n      realization (the per-cell mixture draws are not stable\n      under M-change with held seed — phase_a \xA7baselines.secondary\n      documents this as expected, hence the N-shift carve-out\n      semantics for the reproduction-check asserts).\n\nNone of (a)-(c) activate any dormant codepath. The catch-and-\nreport semantics for the BLOCKING asserts at lines 1442/1472\n(per phase_a \xA7baselines.secondary final paragraph) are a\nphase_a-level pre-commitment about how to report IF the assert\nhalts, NOT a script-level branch change. The asserts themselves\nare unchanged from Q1B's audited version.\n"
production_calls:
  blocking_reproduction_asserts:
  - asserts: '|repro_chi2_no_z - Q2A_M3_CHI2_SETTLED (953.84)| <= _REPRODUCTION_TOL_CHI2 (50.0)'
    line: 1442
    production_inputs:
    - "fits_per_dt_dt_parent (← _fit_mix2_per_daytype_parent ← mixture_2_gaussian_mle_fit line 406)"
    - "_pit_for_family (← experiment.distributional_class_q2a.__main__; Q2A SETTLED production helper, called via _pit_no_z_reproduction line 528)"
    - "_marginal_pit_chi2 (← experiment.distributional_class_q2a.__main__; Q2A SETTLED production helper)"
    status_in_substitution_diff: UNCHANGED (assert text + tol constant + production-input chain byte-identical vs Q1B's audited version at commit 1736e71)
  - asserts: '|repro_chi2_zc - Q1A_M3_Z_c_CHI2_SETTLED (952.5167)| <= _REPRODUCTION_TOL_CHI2 (50.0)'
    line: 1472
    production_inputs:
    - "fits_per_dt_zc_parent (← _fit_mix2_per_zc_parent ← mixture_2_gaussian_mle_fit line 379)"
    - "_make_mixture_2_sampler (← experiment.distributional_class_q2a.__main__; Q2A SETTLED production helper)"
    - "_marginal_pit_chi2 (← experiment.distributional_class_q2a.__main__; Q2A SETTLED production helper)"
    - custom SMC propagation loop in _pit_zc_only_reproduction (line 558+; orchestration over production-stamped components, NOT reimplementation of fit/sampler/chi^2)
    status_in_substitution_diff: UNCHANGED (assert text + tol constant + production-input chain byte-identical vs Q1B's audited version at commit 1736e71)
  mixture_2_gaussian_mle_fit_call_sites:
  - enclosing: _fit_mix2_per_zc_z2_cell
    feeds: head-to-head per-cell M3 fits (Z2_b / Z2_c branches); chi2_Z2_b, chi2_Z2_c, better_chi2, cumulative_chi2_improvement_above_Z_c
    line: 356
  - enclosing: _fit_mix2_per_zc_parent
    feeds: "(a) reproduction-check #2 baseline fit (repro_chi2_zc; DEPENDENT in Q1B' phase_a) AND (b) pool-to-parent fallback for sparse 3-tuple cells per phase_a \xA7metric step 2"
    line: 379
  - enclosing: _fit_mix2_per_daytype_parent
    feeds: 'reproduction-check #1 baseline fit (repro_chi2_no_z; DEPENDENT in Q1B'' phase_a)'
    line: 406
  - enclosing: _run_inline_synthetic_gate (per-cell synthetic fit)
    feeds: "BLOCKING inline synthetic gate per phase_a \xA7baselines.secondary; invoked before production-data path"
    line: 702
  - enclosing: _run_inline_synthetic_gate (false-Z2 companion fit)
    feeds: false-Z2-permutation companion gate (BLOCKING)
    line: 787
  smc_propagation_call_sites_using_args_n_particles:
  - M_argument: args.n_particles (= 400 at Q1B' invocation)
    callee: _pit_no_z_reproduction
    feeds_dependent: repro_chi2_no_z
    line: 1430
  - M_argument: args.n_particles (= 400 at Q1B' invocation)
    callee: _pit_zc_only_reproduction
    feeds_dependent: repro_chi2_zc
    line: 1460
  - M_argument: args.n_particles (= 400 at Q1B' invocation)
    callee: _pit_for_zc_z2_candidate
    feeds_dependent: chi2_Z2_b, chi2_Z2_c, better_chi2, cumulative_chi2_improvement_above_Z_c
    line: 1524
  whitelisted_imports:
  - experiment._actuals.load_actuals
  - experiment._actuals.load_pre_cutoff_actuals
  - experiment._actuals.zscore_params
  - experiment._actuals.zscore_transform
  - experiment.backtest._complete_delivery_days
  - experiment.freeze
  - experiment.predict._daytype
  - experiment.provenance.Grade
  - experiment.provenance.make_result
  - processing.innovations.estimator.mixture_2_gaussian_mle_fit
purpose: "Re-audit of experiment/q1b_z2_conditioning/__main__.py against\nQ1B' phase_a (body_sha256\n7ed3eea35553b9c5cd3e03d0003603ed4d56337d151362d4a11c2d1955b75e78)\nafter the inline-edit parameterization commit 8cd2935. The prior\naudit (at commit dd31839, when the script was BYTE-IDENTICAL to\nthe Q1B audited version at 1736e71) is superseded — the script\nnow contains the 10-string artifact-pinning substitution plus an\n8-line docstring expansion. Confirms three claims:\n\n  1. The substitution introduces NO new imports, NO new function\n     definitions, NO new executable code paths, and NO changes\n     to fitter / sampler / chi^2 / library / bootstrap / assert\n     logic. All 10 substitution sites are string literals embedded\n     in artifact-emission code (body header text, result.yaml\n     reference dict, pickle metadata dict, output path construction,\n     CLI --help text). The substitution affects only the\n     artifact-pinning metadata that goes into the result.yaml\n     header — not the values computed for that header.\n\n  2. All operations feeding the script's outputs (primary\n     cumulative_chi2_improvement_above_Z_c, secondary\n     better_chi2 / repro_chi2_no_z / repro_chi2_zc) flow through\n     production-stamped functions, exactly as in Q1B's audited\n     version. The dependent-variable provenance chains are byte-\n     identical to the prior audit modulo the +8 line shift from\n     the docstring expansion.\n\n  3. The --n-particles 400 invocation path introduces no\n     reimplementation in branches that were dormant at N=200.\n     Static AST trace confirms NO N-conditional branches exist\n     anywhere in the script: args.n_particles is a pure scalar\n     passed through unmodified to all three SMC propagation\n     call-sites (lines 1430, 1460, 1524) as the `M` parameter.\n     This finding carries forward unchanged from the prior\n     audit.\n"
references:
- body_sha256: 7ed3eea35553b9c5cd3e03d0003603ed4d56337d151362d4a11c2d1955b75e78
  file: phase_a.yaml
  relation: subject_of_audit
- body_sha256: c6c54096a593b20484f53175ae31431e7add0de0f91120f37e05dbe65b3d5980
  file: ../2026-05-27_q1b-z2-conditioning-second-axis/phase_fidelity_check_p.yaml
  relation: predecessor_node_check_p_precedent
- body_sha256: 0602f0c07c9468e3f5905e4730dc55515ae81ed0f2533db38b408a59212f5c10
  file: ../2026-05-27_q1b-z2-conditioning-second-axis/phase_a.yaml
  relation: predecessor_node_phase_a
- body_sha256: 5c8f5e7d5a3640018ea99010079498e92e5c79e86d187881a7f3d9eccdd18fb9
  file: ../2026-05-27_resolution-paths-thread/thread.yaml
  relation: thread
schema: code_path_audit
script_audited: experiment/q1b_z2_conditioning/__main__.py
script_lines: 1772
script_status_note: 'Q1B''s script is NO LONGER byte-identical to the version audited
  at commit 1736e71 (Q1B SETTLED R-B1B). Commit 8cd2935
  ("parameterize for Q1B'' sibling (inline-edit, 10 substitutions)")
  applied 10 string-literal substitutions plus an 8-line docstring
  expansion. The computational pipeline (imports, function bodies
  feeding fitters / samplers / chi^2 / library construction /
  bootstrap / asserts) is byte-identical; only artifact-pinning
  metadata strings changed. Diff stats: 1 file changed, 19
  insertions(+), 11 deletions(-), all confined to string-literal
  payloads inside artifact-emission code. The 8-line docstring
  growth shifts post-docstring line numbers by +8 vs the prior
  audit; the static auditor''s output has been re-verified against
  the current line layout.'
static_auditor_verdict:
  command: .venv/bin/python -m experiment.audit.code_path experiment/q1b_z2_conditioning/__main__.py
  output_summary: 'Production-path verified. The claim may be CITED subject to the
    other audits (arbiter, multiverse).'
  verdict: PRODUCTION-PATH
substitution_sites_enumeration:
  description: 'Enumeration of all 10 string substitutions from commit 8cd2935 plus
    the docstring expansion. Each site is classified by its enclosing context to
    confirm none touch computational logic. All sites confirmed to be string-literal
    payloads inside artifact-emission code (no executable change).'
  sites:
  - context: module docstring
    kind: docstring_expansion
    lines_old: 1-12
    lines_new: 1-20
    note: 8 new doc lines + 1 hash substitution + 1 prereg-dir substitution within
      docstring (already counted below)
  - context: module docstring (artifact-pin reference)
    kind: phase_a_hash
    line_new: 13
    payload: phase_a body_sha256 reference for human readers
  - context: module docstring (artifact-pin reference)
    kind: prereg_dir_path
    line_new: 15
    payload: prereg-dir path for human readers
  - context: _render_body (artifact body header text builder)
    kind: prereg_dir_path
    line_new: ~905
    payload: 'L.append("  2026-05-29_q1b-prime-smc-tightening/phase_a.yaml")'
  - context: _render_body (artifact body header text builder)
    kind: phase_a_hash
    line_new: ~907
    payload: phase_a body_sha256 string embedded in artifact body header
  - context: _render_result_yaml (result.yaml dict emitter)
    kind: phase_a_hash
    line_new: ~1112
    payload: references[0].body_sha256 field value
  - context: main() CLI --help text
    kind: prereg_dir_path
    line_new: ~1340
    payload: help string for --emit-result flag
  - context: main() pickle metadata dict
    kind: phase_a_hash
    line_new: ~1652
    payload: phase_a_body_sha256 field in primary pickle's metadata sub-dict
  - context: main() result.yaml provenance dict (passed to make_result body)
    kind: phase_a_hash
    line_new: ~1701
    payload: phase_a_body_sha256 field in result.yaml metadata sub-dict
  - context: main() result.yaml provenance dict (passed to make_result body)
    kind: thread_hash
    line_new: ~1703
    payload: thread_body_sha256 field in result.yaml metadata sub-dict (post-A2
      thread state)
  - context: main() output path construction
    kind: prereg_dir_path
    line_new: ~1760
    payload: 'out_yaml = PROJECT_ROOT / "notes" / "preregistrations" / "2026-05-29_q1b-prime-smc-tightening"
      / "result.yaml"'
  verification: 'git diff 1736e71..8cd2935 -- experiment/q1b_z2_conditioning/__main__.py
    confirms diff is exactly these 10 substitutions plus 8 added docstring lines
    (19 insertions, 11 deletions). No new imports, no new def lines, no new symbols
    referenced. The static auditor''s production-imports list is byte-identical to
    the prior audit''s.'
topic: q1b-prime-smc-tightening
verdict: PRODUCTION-PATH
verdict_summary: 'PRODUCTION-PATH. The inline-edit parameterization commit 8cd2935
  applies 10 string-literal substitutions to artifact-pinning metadata (5x phase_a
  hash, 1x thread hash, 4x prereg-dir path) plus an 8-line docstring expansion.
  All substitution sites are confined to string-literal payloads inside artifact-
  emission code (body header text builder, result.yaml dict emitter, pickle
  metadata dict, output path construction, CLI --help text). Zero new imports,
  zero new function definitions, zero new symbols, zero changes to fitter /
  sampler / chi^2 computer / library construction / bootstrap / assert logic. The
  static auditor''s production-imports list (10 whitelisted symbols) is byte-
  identical to the prior audit. The substitution affects only what the result.yaml
  header points AT, not what is computed for it. All five dependent variables in
  Q1B'' phase_a (cumulative_chi2_improvement_above_Z_c +
  cumulative_chi2_improvement_above_Z_c_ci95 + better_chi2 + repro_chi2_no_z +
  repro_chi2_zc) trace to whitelisted production fitters
  (mixture_2_gaussian_mle_fit at lines 356/379/406) and settled-predecessor
  production helpers (Q2A''s _pit_for_family, _make_mixture_2_sampler,
  _marginal_pit_chi2; P1''s _time_of_day_label). The --n-particles 400 invocation
  introduces no dormant-branch activation (no N-conditional branches exist). The
  BLOCKING reproduction-check asserts at lines 1442/1472 are byte-identical vs
  Q1B''s audited version and consume production-stamped fit objects. Q1B'' may
  proceed (the substitution-only change is ratified).'
written_at: '2026-05-30T08:00:00-07:00'
