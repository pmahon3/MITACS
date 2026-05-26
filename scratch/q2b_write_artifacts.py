"""Q2B preregister artifacts builder.

Writes phase_a.yaml, then re-reads it for body_sha256, then writes
proponent.yaml and devils_advocate.yaml. Each call to stamp() uses
self_path so the artifact's own untracked existence does not flip
git_clean to false.

INSPECTION-ONLY scratch — does not produce a result; only persists
pre-registration YAMLs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiment.audit.registry import (  # noqa: E402
    stamp,
    read_yaml,
    verify_file,
)

OUT_DIR = (
    PROJECT_ROOT
    / "notes"
    / "preregistrations"
    / "2026-05-26_q2b-non-distributional-decomposition"
)
THREAD_PATH = (
    PROJECT_ROOT
    / "notes"
    / "preregistrations"
    / "2026-05-26_distributional-class-thread"
    / "thread.yaml"
)
THREAD_BODY_SHA256_EXPECTED = (
    "05b732d41a26552121912412a8834a108175e811819916920c8b05c8b1d50d58"
)


def _write_yaml(path: Path, data: dict) -> None:
    """Write YAML in the same style as Q2A artifacts."""
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False, width=200)


def _verify_or_die(path: Path) -> None:
    ok, msg = verify_file(path)
    if not ok:
        raise SystemExit(f"verify_file failed for {path}: {msg}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Confirm thread.yaml is what we expect.
    thread_doc = read_yaml(THREAD_PATH)
    actual = thread_doc.get("body_sha256")
    if actual != THREAD_BODY_SHA256_EXPECTED:
        raise SystemExit(
            f"thread.yaml body_sha256 mismatch:\n  expected {THREAD_BODY_SHA256_EXPECTED}\n  got      {actual}"
        )
    if thread_doc.get("current_node_id") != "Q2B":
        raise SystemExit(
            f"thread.yaml current_node_id = {thread_doc.get('current_node_id')} (expected Q2B)"
        )
    _verify_or_die(THREAD_PATH)

    # ---------------- PHASE A ----------------
    phase_a = build_phase_a()
    phase_a_path = OUT_DIR / "phase_a.yaml"
    stamped_a = stamp(phase_a, self_path=phase_a_path)
    _write_yaml(phase_a_path, stamped_a)
    _verify_or_die(phase_a_path)
    on_disk_a = read_yaml(phase_a_path)
    phase_a_body_sha = on_disk_a["body_sha256"]
    print(f"phase_a.yaml      body_sha256={phase_a_body_sha} git_clean={on_disk_a['git_clean']} git_sha={on_disk_a['git_sha']}")

    # ---------------- PROPONENT ----------------
    proponent = build_proponent(phase_a_body_sha)
    proponent_path = OUT_DIR / "proponent.yaml"
    stamped_p = stamp(proponent, self_path=proponent_path)
    _write_yaml(proponent_path, stamped_p)
    _verify_or_die(proponent_path)
    on_disk_p = read_yaml(proponent_path)
    print(f"proponent.yaml    body_sha256={on_disk_p['body_sha256']} git_clean={on_disk_p['git_clean']} git_sha={on_disk_p['git_sha']}")

    # ---------------- DEVILS-ADVOCATE ----------------
    da = build_devils_advocate(phase_a_body_sha)
    da_path = OUT_DIR / "devils_advocate.yaml"
    stamped_d = stamp(da, self_path=da_path)
    _write_yaml(da_path, stamped_d)
    _verify_or_die(da_path)
    on_disk_d = read_yaml(da_path)
    print(f"devils_advocate.yaml body_sha256={on_disk_d['body_sha256']} git_clean={on_disk_d['git_clean']} git_sha={on_disk_d['git_sha']}")


def build_phase_a() -> dict:
    return {
        "schema": "phase_a",
        "written_at": "2026-05-26T23:30:00-04:00",
        "references": [
            {
                "file": "../2026-05-26_distributional-class-thread/thread.yaml",
                "body_sha256": THREAD_BODY_SHA256_EXPECTED,
                "relation": "thread",
            }
        ],
        "topic": "q2b-non-distributional-decomposition",
        "research_question": (
            "Of the ~954 chi^2 Q2A confirmed is non-distributional (post-cutoff marginal PIT under "
            "the best richer family M3, mixture-2-Gaussian), how does it decompose into three pre-registered "
            "mechanisms (iterated mean trajectory bias, iterated variance divergence at large h, day-anchor "
            "seam artifact) when each is ablated in isolation?"
        ),
        "hypothesis": {
            "proponent": (
                "R-A3 fires with mech_1 (iterated_mean_trajectory_bias) dominant at explained_share ~ 0.75. "
                "Per mitacs-clim-gap-nonstationary, the post-cutoff actual-mu gap is approximately +1000 MW "
                "hour-flat and non-stationary, driving systematic iterated-mean drift on the registered "
                "predictor; this MW-space bias propagates to z-space (modulo standardization) as a "
                "first-moment offset in the iterated kappa_1 marginal that no symmetric-centred distributional "
                "family (Student-t, mixture, KDE) can absorb. Subtracting an empirical per-(h, day_type) "
                "mean bias prior to PIT eval should account for most of the residual chi^2. mech_2 "
                "(iterated_variance_divergence) contributes secondarily (~0.15); mech_3 (seam) is a "
                "REGISTERED NULL with expected share ~ 0 because production estimator already excludes "
                "seam-crossing iterations via _complete_delivery_days."
            ),
            "counter": (
                "See devils_advocate.yaml. Most plausibly R-B3 with mech_1 (~0.40) and mech_2 (~0.35) both "
                "contributing and neither dominant. The +1000 MW MW-space gap may translate substantially "
                "attenuated to z-space (kappa_1 is fit in z-space against a fixed pre-cutoff climatology). "
                "Iterated heavy-tailed variance over h=24 in mixture-2-Gaussian is non-trivial and mech_2 may "
                "contribute more than the proponent expects. Q2A's Saturday-4x-weekday family-invariant pattern "
                "suggests day-type-specific structure that mean-bias alone will not fully capture."
            ),
        },
        "variables": {
            "dependent": [
                "chi2_baseline_M3",
                "chi2_after_mean_bias_fix",
                "chi2_after_variance_cap_fix",
                "chi2_after_seam_exclusion_fix",
                "explained_share_mean_bias",
                "explained_share_variance_div",
                "explained_share_seam",
                "ci95_chi2_baseline_M3",
                "ci95_chi2_after_mean_bias_fix",
                "ci95_chi2_after_variance_cap_fix",
                "ci95_chi2_after_seam_exclusion_fix",
                "ci95_explained_share_mean_bias",
                "ci95_explained_share_variance_div",
                "ci95_explained_share_seam",
                "per_h_day_type_empirical_mean_bias",
                "per_h_day_type_empirical_variance",
                "per_h_day_type_smc_ensemble_variance",
            ],
            "independent": [
                "mechanism",
                "day_type",
            ],
            "excluded": [
                {
                    "variable": "cumulative_or_sequential_ablation",
                    "reasoning": (
                        "Ablation is per-mechanism in isolation. Cumulative ablation (mech_1 then mech_2 etc.) "
                        "would make the decomposition order-dependent and conflate interactions with main "
                        "effects. Pre-registered methodology is 3 isolated runs, one per mechanism."
                    ),
                },
                {
                    "variable": "cross_mechanism_interactions",
                    "reasoning": (
                        "Interaction terms (e.g., mean-bias-fix-conditional-on-variance-cap) are outside the "
                        "pre-registered candidate set. If sum of shares != 1 (overlap or under-cover), that "
                        "is reported as a sanity check but does not modify the verdict; interactions are "
                        "a follow-up node, not Q2B."
                    ),
                },
                {
                    "variable": "deployable_corrections",
                    "reasoning": (
                        "Mech 1 and mech 2 use post-cutoff actuals to define the fix specifically for "
                        "attribution. This is NOT a deployable correction (the deployment context has no "
                        "post-cutoff actuals at issue time); it is an upper bound on what each mechanism "
                        "could explain if perfectly corrected. The distinction is documented and the verdict "
                        "is about attribution, not improvement."
                    ),
                },
                {
                    "variable": "kernel_family_lift",
                    "reasoning": (
                        "Q2A established M3 (mixture-2-Gaussian) as the binding richer family. Q2B holds the "
                        "kernel family fixed at M3 (and reuses Q2A's fits) so the decomposition is on the "
                        "non-distributional axis exclusively. Lifting kernel family inside Q2B would conflate "
                        "Q2A's axis with Q2B's."
                    ),
                },
            ],
        },
        "data": {
            "source": (
                "Pre-cutoff Ontario demand (load_actuals, cutoff = spec.data_cutoff) for kappa_1 + climatology "
                "+ M3 mixture-2-Gaussian fits (reused from Q2A); post-cutoff for chi^2 evaluation. Same "
                "post-cutoff population as Q1 / Q2A: all delivery days from 2025-01-01 through 2026-05-16."
            ),
            "partition": {
                "train": "pre-cutoff library (<= 2024-12-31 23:00) for fits (reused from Q2A)",
                "dev": "not used (no hyperparameter tuning in Q2B)",
                "test": "post-cutoff 2025-01-01 .. 2026-05-16 (whole population; no subsetting)",
            },
            "cutoff_handling": (
                "ATTRIBUTION-ONLY USE OF POST-CUTOFF ACTUALS: empirical_mean_bias(h, day_type) and "
                "sigma2_empirical(h, day_type) are computed from post-cutoff actuals to define each "
                "mechanism's fix. This intentionally violates strict leakage-guarding because the verdict is "
                "about attribution (upper bound on what each mechanism could explain), not about deployment. "
                "Standard leakage-guarded pre-cutoff library is used for kappa_1 + climatology + M3 fits. "
                "The attribution-vs-deployment distinction is explicit in stopping_criterion and in the "
                "verdict interpretation."
            ),
        },
        "metric": (
            "PRIMARY: explained_share per mechanism m, where\n"
            "  explained_share[m] = (chi2_baseline - chi2_after_fix[m]) / (chi2_baseline - R_A2_cut)\n"
            "  chi2_baseline      = chi2_M3 from Q2A baseline (= 953.84 SETTLED point, recomputed in Q2B "
            "with same SMC seeds as a self-consistency sanity check)\n"
            "  R_A2_cut           = 228.44 (Q2A gate-validated R-A2 cut; binding cell V3_rho10 KDE kurt=12)\n"
            "  denominator        = 953.84 - 228.44 = 725.40 (frozen constant in the formula)\n"
            "\n"
            "For each mechanism m in {mean_bias, variance_div, seam}:\n"
            "  1. Reuse Q2A's M3 (mixture-2-Gaussian) fits (per-day-type C_1, mu, weights, scales).\n"
            "  2. Run SMC iteration over all post-cutoff issue-times, M=200 particles per state, "
            "iterating to the same horizon Q1/Q2A used.\n"
            "  3. Apply mechanism m's fix to the SMC particle ensemble:\n"
            "     mech_1 (mean_bias):  for each (h, day_type), subtract empirical_mean_bias(h, day_type) "
            "= mean over post-cutoff issue-times of (actual_z(h) - iterated_mean_z(h)) from each particle's "
            "z value at horizon h.\n"
            "     mech_2 (variance_div): for each (h, day_type), compute sigma2_iter(h, day_type) = mean "
            "over post-cutoff of sample variance of M particles at h; compute sigma2_empirical(h, day_type) "
            "= mean over post-cutoff of (actual_z - empirical_mean_z)^2 grouped by day_type. If sigma2_iter "
            "> sigma2_empirical, rescale all particles toward the ensemble mean by factor "
            "sqrt(sigma2_empirical / sigma2_iter); else no change.\n"
            "     mech_3 (seam): restrict PIT aggregation domain to exclude issue-times whose horizon-h target "
            "lands on the day-anchor hour. Production estimator already excludes seam-crossing iteration via "
            "_complete_delivery_days; mech_3 layers an additional aggregation-domain exclusion for the "
            "REGISTERED NULL.\n"
            "  4. Recompute marginal PIT: u = empirical CDF of particle distribution at z_actual; aggregate "
            "u over all post-cutoff (day, hour-of-day) pairs.\n"
            "  5. chi2_after_fix[m] = sum over 10 bins of (observed - expected)^2 / expected, expected = N/10.\n"
            "  6. explained_share[m] computed per formula above.\n"
            "\n"
            "SECONDARY (sanity-check; not part of the verdict):\n"
            "  - chi2_after_fix[m] stratified by day_type.\n"
            "  - sum_m explained_share[m]: reported and discussed. Sum substantially exceeding 1 indicates "
            "overlap (mechanisms address the same chi^2); sum substantially below max[m] explained_share[m] "
            "indicates under-attribution.\n"
            "  - Self-consistency: chi2_baseline_M3 recomputed in Q2B should match Q2A's M3 chi^2 = 953.84 "
            "within paired-day-bootstrap CI; mismatch indicates pipeline drift between Q2A and Q2B.\n"
            "  - Synthetic single-mechanism sanity (small N=500, M=50; ~10 sec): generate VAR(1) data with a "
            "known mean-bias offset, confirm mech_1 ablation attributes ~100% share to mean_bias.\n"
        ),
        "models": {
            "M3_baseline": {
                "description": (
                    "Q2A's M3 mixture-2-Gaussian SMC pipeline at its registered config (per-day-type C_1 + mu "
                    "from pre-cutoff library; symmetric two-component centred Gaussian mixture residual law "
                    "fit by mixture_2_gaussian_mle_fit; M=200 particles; same per-step independence "
                    "assumption; same day-anchor handling). chi2_baseline_M3 is recomputed in Q2B as a "
                    "self-consistency sanity check against Q2A's settled chi^2_M3 = 953.84."
                ),
                "free_parameters": "reused from Q2A; no refit",
                "notes": (
                    "M3 is the binding family from Q2A (best of M3/M4/M5 for the binding cell). Q2B holds "
                    "kernel family fixed at M3 and varies only the non-distributional mechanism."
                ),
            },
            "M3_mech1_mean_bias_fix": {
                "description": (
                    "M3 baseline with mean-bias fix applied to SMC particles: subtract per-(h, day_type) "
                    "empirical_mean_bias from each particle's z value at horizon h, where empirical_mean_bias "
                    "is computed from post-cutoff actuals (attribution-only use of post-cutoff data)."
                ),
                "free_parameters": (
                    "per (h, day_type): 1 empirical_mean_bias = ~24 x 3 = 72 attribution-only parameters"
                ),
                "notes": (
                    "Fix is non-deployable (uses post-cutoff actuals); pre-registered as an attribution upper "
                    "bound on the mean-bias mechanism's contribution to chi^2."
                ),
            },
            "M3_mech2_variance_cap_fix": {
                "description": (
                    "M3 baseline with variance-cap fix applied to SMC particles: if sigma2_iter(h, day_type) "
                    "exceeds sigma2_empirical(h, day_type), rescale all particles toward the ensemble mean "
                    "by factor sqrt(sigma2_empirical / sigma2_iter); else no change. Both quantities are "
                    "computed from post-cutoff data (attribution-only)."
                ),
                "free_parameters": (
                    "per (h, day_type): 1 sigma2_empirical (target) = ~24 x 3 = 72 attribution-only parameters"
                ),
                "notes": (
                    "One-sided cap (does not scale up if iterated variance is under-dispersed); pre-registered "
                    "as an attribution upper bound on the variance-divergence mechanism's contribution."
                ),
            },
            "M3_mech3_seam_exclusion_fix": {
                "description": (
                    "M3 baseline with PIT aggregation domain restricted to exclude issue-times whose horizon-h "
                    "target lands on the day-anchor hour. Production estimator already excludes seam-crossing "
                    "iteration via _complete_delivery_days (day_anchor_hour mask kwarg was removed in current "
                    "production); mech_3 adds a redundant aggregation-domain filter."
                ),
                "free_parameters": "0 (deterministic domain filter)",
                "notes": (
                    "REGISTERED NULL. Expected explained_share ~ 0 by construction. Included as a pre-"
                    "registered null check: if mech_3 reads > 0.15 the production seam handling has a leak "
                    "and the diagnostic is itself informative."
                ),
            },
        },
        "shape_outcomes": {
            "R-A3": {
                "criterion": (
                    "max_m explained_share[m] > 0.50 AND max_m explained_share[m] > 2 x second_m explained_share[m]"
                ),
                "interpretation": (
                    "One mechanism dominates the non-distributional chi^2 and is fixable independent of "
                    "kernel family (modulo the attribution-vs-deployment caveat). Points to a concrete "
                    "remediation in the next thread node (e.g., deployable analogue of the dominant "
                    "mechanism: state-dependent mean-bias correction without post-cutoff actuals)."
                ),
            },
            "R-B3": {
                "criterion": (
                    "NOT R-A3 AND NOT R-C3 (i.e., either max share is below 0.50, or max share exceeds 0.50 "
                    "but is within 2x of the second; in either case at least one share is >= 0.15)"
                ),
                "interpretation": (
                    "Multiple mechanisms contribute to the non-distributional chi^2; no single fix suffices. "
                    "Triggers a thread amendment by thread-coordinator (NOT an automatic next-node fire) "
                    "to design a combined-mechanism or interaction-aware investigation."
                ),
            },
            "R-C3": {
                "criterion": "max_m explained_share[m] < 0.15",
                "interpretation": (
                    "No pre-registered mechanism captures the chi^2. The marginal PIT pathology is "
                    "irreducibly pathological under the Q2B candidate set; new candidates (weather, state-"
                    "space dimension, day-type partition refinement) are needed via amendment."
                ),
            },
        },
        "falsification_criterion": {
            "numeric": "R-C3 fires (max_m explained_share[m] < 0.15)",
            "reasoning": (
                "If none of the three pre-registered mechanisms (mean_bias, variance_div, seam) accounts for "
                "more than 15% of the gap between baseline and the R-A2 corroboration cut, the Q2B candidate "
                "set is empirically inadequate to decompose the non-distributional chi^2. The 0.15 threshold "
                "is a generous noise floor above mech_3's REGISTERED NULL expected ~0 share; mech_1 and "
                "mech_2 falling below 0.15 means the substantive non-distributional content lives elsewhere "
                "(weather covariates, embedding dimension, day-type partition refinement). Routes to thread "
                "amendment for a new candidate set."
            ),
        },
        "corroboration_criterion": {
            "numeric": (
                "R-A3 fires (max_m explained_share[m] > 0.50 AND max_m > 2 x second_m)"
            ),
            "reasoning": (
                "If one mechanism dominates (> 50% of the gap AND > 2x the second), it identifies the "
                "leading non-distributional content and provides a concrete target for the next thread node. "
                "The 0.50 threshold is the formal dominance bar (top share exceeds the sum of all others); "
                "the > 2x next clause enforces clear separation from a 50/50 tie. Under the proponent's "
                "forecast (mech_1 ~ 0.75, mech_2 ~ 0.15) R-A3 fires; under the DA's forecast (mech_1 ~ 0.40, "
                "mech_2 ~ 0.35) R-A3 does not fire."
            ),
        },
        "ambiguous_region": {
            "range": (
                "R-B3: max share between 0.15 and 0.50, OR max share above 0.50 but within 2x of second"
            ),
            "action": (
                "Report and characterize. R-B3 triggers a thread amendment via thread-coordinator (NOT an "
                "automatic next-node fire) to design a combined-mechanism or interaction-aware investigation. "
                "Same R-B-mandates-amendment pattern as Q2A's R-B2."
            ),
        },
        "baselines": {
            "primary": (
                "Q2A's chi2_M3 = 953.84 (point estimate; CI [685, 1357] from Q2A's paired-day bootstrap). "
                "Q2B's chi2_baseline_M3 must match within bootstrap CI overlap; mismatch indicates pipeline "
                "drift between Q2A and Q2B and is BLOCKING for the Q2B verdict."
            ),
            "secondary": (
                "Per-mechanism synthetic sanity check (small N=500, M=50; ~10 sec wall-clock): generate a "
                "VAR(1)-class synthetic system with a known single-mechanism defect (e.g., constant mean-"
                "bias offset), run the Q2B ablation pipeline with all three mechanism fixes, confirm that "
                "the corresponding mechanism's explained_share is ~ 1.0 (within Monte Carlo noise) and the "
                "other two are ~ 0. Validates the ablation attribution logic against ground truth."
            ),
        },
        "stopping_criterion": {
            "rule": (
                "Run: M3 baseline SMC propagation (reusing Q2A fits); empirical_mean_bias and "
                "sigma2_empirical computation from post-cutoff actuals; mech_1 / mech_2 / mech_3 ablation "
                "PIT chi^2; 1000-resample paired-day bootstrap CI on each chi^2 and each explained_share; "
                "evaluate R-A3 / R-B3 / R-C3 against pre-registered cuts; arbiter. STOP. No iteration; no "
                "parameter tuning; no post-hoc mechanism redefinition."
            ),
            "reasoning": (
                "Pre-registration is complete; verdict is determined by data once the experiment runs. The "
                "attribution-only use of post-cutoff actuals is intentional and scoped to upper-bounding "
                "each mechanism's contribution; converting any mechanism to a deployable correction is a "
                "separate next-node design question, not a Q2B step."
            ),
        },
        "ci_method": {
            "type": "paired_day_bootstrap",
            "details": (
                "1000 paired-day bootstrap resamples of post-cutoff delivery days. Per resample, recompute "
                "chi^2_baseline and chi^2_after_fix[m] for m in {mean_bias, variance_div, seam} from the "
                "resampled day population, and compute explained_share[m] per resample using the same "
                "denominator constant 725.40 (953.84 - 228.44). Report 95% percentile CI on every chi^2 "
                "and every explained_share."
            ),
            "width": "95 percent percentile interval",
        },
        "affects_registered_spec": False,
        "thread": {
            "thread_topic": "distributional-class-thread",
            "thread_body_sha256": THREAD_BODY_SHA256_EXPECTED,
            "node_id": "Q2B",
            "parent_branch": "R-C",
            "threshold_derivations": (
                "Numeric thresholds (X=0.50 dominance, Y=0.15 null) are NOT derived from a synthetic gate "
                "(unlike Q2A's gate-derived 228.44 and 2284.44). They are set on non-data principled bases:\n"
                "  X=0.50: formal threshold for dominance (top mechanism > sum of all others); the > 2x next "
                "clause enforces clear separation from 50/50.\n"
                "  Y=0.15: generous noise floor above the seam mechanism's REGISTERED NULL expected ~0 share. "
                "If mech_3 reads > 0.15 the production seam handling has a leak; the 0.15 floor is the same "
                "bar applied to mech_1 / mech_2 so R-C3 fires iff none of the three captures > 15% of the gap.\n"
                "chi2_baseline = 953.84 from Q2A SETTLED finding (mitacs-q2a-richer-family, M3 point estimate).\n"
                "R-A2 corroboration cut = 228.44 from Q2A pre-registered gate (gate body_sha256 2bab514e..., "
                "binding cell V3_rho10 KDE kurt=12, gate-validated under variant-max conservative rule).\n"
                "Denominator 725.40 = 953.84 - 228.44 is frozen for all explained_share computations including "
                "per-bootstrap-resample evaluations."
            ),
        },
        "related_memory": [
            "mitacs-q2a-richer-family",
            "mitacs-q1-student-t-vs-gaussian",
            "mitacs-clim-gap-nonstationary",
            "mitacs-dayanchor-seam",
            "mitacs-multiscale-defect-eigenmodes-v1",
            "mitacs-rebaseline-facts",
        ],
        "related_seed": [
            "notes/seeds/multiscale_factor_coherence.md",
            "writeup/tex/missing_content_memo.tex",
        ],
    }


def build_proponent(phase_a_body_sha: str) -> dict:
    return {
        "schema": "proponent",
        "written_at": "2026-05-26T23:35:00-04:00",
        "references": [
            {
                "file": "phase_a.yaml",
                "body_sha256": phase_a_body_sha,
            }
        ],
        "forecast": {
            "value": {
                "outcome": "R-A3",
                "dominant_mechanism": "mean_bias",
                "explained_share_mean_bias": 0.75,
                "explained_share_variance_div": 0.15,
                "explained_share_seam": 0.0,
                "chi2_after_mean_bias_fix": 409,
                "chi2_after_variance_cap_fix": 845,
                "chi2_after_seam_exclusion_fix": 954,
                "chi2_baseline_M3_self_consistency": 953.84,
            },
            "units": (
                "explained_share dimensionless in [0, 1]; chi^2 dimensionless for 10-bin uniform; "
                "self-consistency baseline matches Q2A's M3 point estimate."
            ),
            "reasoning": (
                "Three lines of evidence converge on R-A3 with mech_1 dominant at share ~ 0.75 and "
                "chi2_after_mean_bias_fix ~ 409.\n\n"
                "(a) The prior cited in mitacs-clim-gap-nonstationary is load-bearing here: the post-"
                "cutoff actual-mu gap on the registered predictor is approximately +1000 MW HOUR-FLAT "
                "and growing through 2025-2026, with no recency-window choice closing it. This is a "
                "first-moment offset in the iterated forecast trajectory by construction; the operator's "
                "sigma*z_next contribution carries the diurnal shape but cannot compensate for a flat MW-"
                "space bias of that magnitude. In z-space (kappa_1's representation) this maps to a "
                "systematic mean offset of order sigma_iter*z (where sigma_iter is the iterated SMC "
                "standard deviation), which is precisely what mech_1's empirical_mean_bias subtraction "
                "targets. Subtracting per-(h, day_type) empirical mean bias is an upper-bound attribution "
                "on this driver and should account for the majority of the residual chi^2 above the R-A2 "
                "cut.\n\n"
                "(b) Q1 (S8d, mitacs-q1-student-t-vs-gaussian) established that heavy-tailedness lives "
                "in the kappa_1 INNOVATION, not the seasonal climatology (M1 outperformed M2 in Q1). The "
                "iterated SMC variance under M3 mixture-2-Gaussian compounds the per-step kappa_1 innovation "
                "variance over horizon h; this is non-trivial but well-behaved (no random-walk explosion at "
                "h=24 since the C_1 drift is contractive in expectation). Mech_2's variance cap targets only "
                "the upper tail of iterated variance dispersion; I forecast its share around 0.15 - it does "
                "real work where iterated variance modestly exceeds empirical, but does not dominate.\n\n"
                "(c) Production already excludes seam-crossing iteration via _complete_delivery_days "
                "(see processing/innovations/estimator.py line 763 docstring confirming day_anchor_hour "
                "mask kwarg was removed). Mech_3 layers a redundant aggregation-domain filter; its "
                "expected share is ~0 by construction. If it reads > 0.05, there is an undetected seam leak "
                "in production and the diagnostic is itself informative; I forecast 0.0 (no leak)."
            ),
        },
        "forecast_ci": {
            "width": 0.95,
            "low": {
                "explained_share_mean_bias": 0.50,
                "explained_share_variance_div": 0.05,
                "explained_share_seam": 0.0,
                "chi2_after_mean_bias_fix": 590,
                "chi2_after_variance_cap_fix": 918,
            },
            "high": {
                "explained_share_mean_bias": 0.95,
                "explained_share_variance_div": 0.30,
                "explained_share_seam": 0.05,
                "chi2_after_mean_bias_fix": 591,
                "chi2_after_variance_cap_fix": 736,
            },
            "reasoning": (
                "CI on mech_1 share [0.50, 0.95] keeps R-A3 in central call but crosses the R-A3/R-B3 "
                "boundary (mech_1 share = 0.50 with mech_2 = 0.30 would tip to R-B3 since 0.50 is not "
                "> 2x 0.30 = 0.60). CI on mech_2 share [0.05, 0.30] reflects uncertainty about how often "
                "iterated SMC variance exceeds empirical at the (h, day_type) cell level. CI on mech_3 "
                "share [0.0, 0.05] is asymmetric (one-sided, bounded below by 0) reflecting the registered "
                "null prior. chi2_after_fix CIs computed from the explained_share formula inverted; "
                "non-monotone bounds reflect the formula's inversion (high share -> low chi^2 and vice versa)."
            ),
        },
        "what_would_change_my_mind": [
            (
                "If mech_1 explained_share < 0.30, the MW-space +1000 MW gap does not translate cleanly "
                "to z-space chi^2 contribution. The clim-gap-nonstationary signal is in MW-space and the "
                "kappa_1 marginal is in z-space; the proponent assumes a roughly direct translation, but "
                "the standardization step may attenuate it more than expected."
            ),
            (
                "If R-C3 fires (all shares < 0.15), none of the three pre-registered mechanisms is the "
                "right level of decomposition. The substantive non-distributional content lives in covariates "
                "outside the Q2B candidate set (weather, state-space dimension, day-type partition "
                "refinement) and the thread needs amendment for a richer mechanism set."
            ),
            (
                "If mech_2 explained_share > mech_1 explained_share, iterated variance divergence is "
                "doing more work than the proponent's first-moment-dominant prior expects. Picture (C) "
                "may have a second-moment failure mode the proponent under-weighted."
            ),
            (
                "If mech_3 explained_share > 0.10, production seam handling has a leak and the Q1/Q2A "
                "results need re-audit; the diagnostic is then informative beyond Q2B's primary verdict."
            ),
        ],
        "confidence": 0.55,
    }


def build_devils_advocate(phase_a_body_sha: str) -> dict:
    return {
        "schema": "devils_advocate",
        "written_at": "2026-05-26T23:40:00-04:00",
        "references": [
            {
                "file": "phase_a.yaml",
                "body_sha256": phase_a_body_sha,
            }
        ],
        "strongest_argument": (
            "The proponent's central claim - that the +1000 MW MW-space gap translates to a dominant "
            "(~0.75) z-space mech_1 share - hinges on a translation step the proponent does not justify. "
            "The MW-space gap in mitacs-clim-gap-nonstationary is on the REGISTERED PREDICTOR (full demand "
            "iteration through forecast = mu + sigma*z_next). The Q2B kappa_1 marginal is in z-space, "
            "standardized against a FIXED pre-cutoff climatology that does not see the post-cutoff drift. "
            "The kappa_1 INNOVATION is the variable being PIT-evaluated, not the full demand forecast; "
            "a +1000 MW MW-space drift maps to a z-space first-moment offset of roughly 1000/sigma_clim ~ "
            "0.3-0.5 standard deviations (sigma_clim ~ 2000-3000 MW depending on hour). This is non-trivial "
            "but attenuated relative to the proponent's implicit assumption that the MW-space dominance "
            "translates 1-to-1 to z-space chi^2 dominance.\n\n"
            "Second, Q2A's Finding #2 (Saturday chi^2 ~ 4x weekday chi^2, family-invariant across M1/M3/M4/M5) "
            "indicates day-type-specific structure that survives every distributional family tested. This "
            "structure could be EITHER a mean-bias asymmetry (Saturday's empirical_mean_bias differs from "
            "weekday's by more than mech_1's per-day-type subtraction can absorb at the aggregate marginal "
            "level) OR a variance-divergence asymmetry (Saturday's iterated SMC variance compounds differently "
            "due to lower-data-density library cells). Either way the day-type pattern is unlikely to be "
            "fully captured by mech_1 alone; mech_2 is plausibly capturing structurally distinct content.\n\n"
            "Third, the iterated heavy-tailed variance over h=24 under M3 mixture-2-Gaussian is non-trivial "
            "by construction. M3's symmetric two-component centred mixture has higher per-step kurtosis than "
            "Gaussian; iterated over 24 steps the central limit pull toward Gaussian competes with the per-"
            "step kurtosis injection, and the resulting iterated marginal variance can substantially exceed "
            "the empirical (which is the actual residual against actuals, not against a forecast trajectory). "
            "Mech_2's variance-cap is one-sided (only fires when sigma2_iter > sigma2_empirical), so when it "
            "fires it does real attribution work; I forecast it captures roughly 0.35 of the gap.\n\n"
            "Net: R-B3 fires with mech_1 ~ 0.40 and mech_2 ~ 0.35, neither dominant under the > 2x clause. "
            "Mech_3 ~ 0.02 (registered null holds; near zero but not exactly zero due to small library-"
            "composition noise at the seam boundary)."
        ),
        "specific_failure_modes": [
            {
                "rank": 1,
                "mode": (
                    "z_space_vs_MW_space_attenuation_of_mean_bias_share_proponent_assumes_1to1_translation"
                ),
                "memory_ref": "mitacs-clim-gap-nonstationary",
                "likelihood": 0.45,
            },
            {
                "rank": 2,
                "mode": (
                    "iterated_variance_divergence_under_M3_heavy_tail_mixture_underestimated_by_proponent"
                ),
                "memory_ref": "mitacs-q1-student-t-vs-gaussian",
                "likelihood": 0.35,
            },
            {
                "rank": 3,
                "mode": (
                    "day_type_asymmetry_not_absorbed_by_either_mech1_or_mech2_in_aggregate_marginal"
                ),
                "memory_ref": "mitacs-multiscale-defect-eigenmodes-v1",
                "likelihood": 0.25,
            },
            {
                "rank": 4,
                "mode": (
                    "attribution_only_use_of_post_cutoff_actuals_inflates_mech1_apparent_share_because_"
                    "empirical_mean_bias_per_h_day_type_subtraction_is_a_72_parameter_fit_that_overfits"
                ),
                "memory_ref": None,
                "likelihood": 0.15,
            },
        ],
        "severity": "serious",
        "counter_prediction": {
            "value": {
                "outcome": "R-B3",
                "dominant_mechanism": None,
                "explained_share_mean_bias": 0.40,
                "explained_share_variance_div": 0.35,
                "explained_share_seam": 0.02,
                "chi2_after_mean_bias_fix": 664,
                "chi2_after_variance_cap_fix": 700,
                "chi2_after_seam_exclusion_fix": 939,
                "chi2_baseline_M3_self_consistency": 953.84,
            },
            "units": (
                "explained_share dimensionless in [0, 1]; chi^2 dimensionless for 10-bin uniform; "
                "outcome category in {R-A3, R-B3, R-C3}; dominant_mechanism null when no single dominance"
            ),
            "reasoning": (
                "Disagreeing with the proponent on outcome category (R-B3, not R-A3) and dominance "
                "structure (no single dominant mechanism, not mech_1 dominant). My central call is mech_1 "
                "~ 0.40 and mech_2 ~ 0.35 with mech_3 ~ 0.02; max share 0.40 is below the 0.50 dominance "
                "bar, and even setting that aside 0.40 is not > 2 x 0.35 = 0.70 so R-A3's separation clause "
                "fails on its own. R-B3 fires by complement. Reasoning: the MW-space +1000 MW gap attenuates "
                "to z-space at ~0.3-0.5 sigma_clim, doing real work but not dominant; iterated variance "
                "under M3's heavy-tail mixture at h=24 compounds non-trivially and mech_2's one-sided cap "
                "captures comparable share; mech_3 is near zero (registered null holds)."
            ),
        },
        "forecast_ci": {
            "width": 0.95,
            "low": {
                "explained_share_mean_bias": 0.25,
                "explained_share_variance_div": 0.20,
                "explained_share_seam": 0.0,
                "chi2_after_mean_bias_fix": 591,
                "chi2_after_variance_cap_fix": 591,
            },
            "high": {
                "explained_share_mean_bias": 0.60,
                "explained_share_variance_div": 0.55,
                "explained_share_seam": 0.05,
                "chi2_after_mean_bias_fix": 772,
                "chi2_after_variance_cap_fix": 809,
            },
            "reasoning": (
                "CI on mech_1 share [0.25, 0.60] crosses the R-A3 dominance bar (0.50) on the upper side. "
                "CI on mech_2 share [0.20, 0.55] also crosses 0.50 on the upper side. The CIs reflect "
                "asymmetric uncertainty: my central call is R-B3, but R-A3 fires in the upper-tail joint "
                "region where one share exceeds 0.50 AND > 2x the other. The chi^2 CIs are derived from the "
                "explained_share formula inverted; the bounds map non-monotonically because the formula "
                "decreases in share."
            ),
        },
        "what_would_change_my_mind": [
            (
                "If mech_1 explained_share > 0.65 with tight CI (lower bound > 0.50), the z-space "
                "attenuation concern is empirically minor and the proponent's MW-space-to-z-space "
                "translation assumption holds. The day-type-asymmetry concern (failure mode 3) would also "
                "be subordinated since mech_1 evidently absorbs it at the aggregate."
            ),
            (
                "If R-A3 fires for mech_2 (mech_2 > 0.50 AND > 2x mech_1), the iterated-variance-divergence "
                "mechanism dominates and my prediction is wrong on direction (which mechanism dominates) "
                "but right on category (no, actually wrong on both; R-A3 vs R-B3). However, this would "
                "validate my underlying concern that the proponent under-weighted mech_2."
            ),
            (
                "If chi2_baseline_M3 self-consistency check fails (Q2B re-run does not match Q2A's 953.84 "
                "within bootstrap CI overlap), the verdict is BLOCKED regardless of outcome category. "
                "The disagreement on shares would be moot."
            ),
        ],
        "opportunity_cost": {
            "alternative": (
                "A z-space-to-MW-space sensitivity quantification of the clim-gap-nonstationary signal "
                "(one-shot computation: what is the empirical_mean_bias in z-space, computed directly "
                "from post-cutoff data, and how does its magnitude compare to the iterated SMC standard "
                "deviation at each (h, day_type)?) would have been cheap (no SMC propagation needed) and "
                "would have grounded the proponent's translation assumption in a single empirical number. "
                "Pre-registering Q2B without this number means the proponent's dominance forecast is "
                "speculative on a quantity easily measured."
            ),
            "ranking": "complementary",
            "notes": (
                "The pre-registration is well-formed AS WRITTEN. The opportunity cost is a one-shot insight "
                "that would have informed forecast width without changing the experiment design; Q2B "
                "produces the empirical_mean_bias matrix as a side product, so the question is answered "
                "by Q2B's secondary outputs even if the central verdict is on explained_share."
            ),
        },
    }


if __name__ == "__main__":
    main()
