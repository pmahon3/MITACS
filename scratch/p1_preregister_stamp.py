"""One-shot script to write and stamp P1 phase_a + proponent + devils_advocate.

Run from repo root with .venv/bin/python.

Mirrors Q2B's preregistration structure. References thread.yaml at
3fe3573938b0aa24e7332e090a578401722ed225330b12395101b66a58c8c2ab.
"""

from __future__ import annotations

from pathlib import Path
import sys

import yaml

from experiment.audit import registry

REPO = Path(__file__).resolve().parents[1]
DIR = REPO / "notes" / "preregistrations" / "2026-05-27_p1-patra-sen-per-stratum-localization"
THREAD_HASH = "3fe3573938b0aa24e7332e090a578401722ed225330b12395101b66a58c8c2ab"

PHASE_A_PATH = DIR / "phase_a.yaml"
PROPONENT_PATH = DIR / "proponent.yaml"
DA_PATH = DIR / "devils_advocate.yaml"

PHASE_A = {
    "schema": "phase_a",
    "written_at": "2026-05-27T09:00:00-07:00",
    "references": [
        {
            "file": "../2026-05-27_resolution-paths-thread/thread.yaml",
            "body_sha256": THREAD_HASH,
            "relation": "thread",
        }
    ],
    "topic": "p1-patra-sen-per-stratum-localization",
    "research_question": (
        "Does the non-Gaussian mixture fraction alpha_L in the kappa_1 predictor's "
        "residuals vary systematically across natural conditioning strata "
        "(demand quantile, time-of-day, season, day-type, hour-of-week)? If yes, "
        "the variation localizes where the missing Z is concentrated, informing "
        "which exogenous variable to test in Q1A/B."
    ),
    "hypothesis": {
        "proponent": (
            "R-A fires with the season axis binding at range ~0.30 (max over the five "
            "axes), driven by temperature being the most plausible missing Z per "
            "mitacs-clim-gap-nonstationary and mitacs-postcovid-probe. Season acts as a "
            "temperature proxy in the absence of exogenous data and concentrates the "
            "heavy-tailed contamination at winter/summer extremes vs shoulder months. "
            "alpha_L_marginal lands in the qualifier-exhaustion-memo's 0.49-0.74 prior "
            "(point ~0.60); binding axis identifies the conditioning variable Q1A uses "
            "to construct its Z."
        ),
        "counter": (
            "See devils_advocate.yaml. R-C: alpha_L is approximately constant "
            "(range < 0.10) across ALL five natural axes. The missing Z is exogenous "
            "and orthogonal to time/calendar/demand stratifications; only temperature "
            "(Q1C) can detect it because no natural axis in P1's candidate set carries "
            "the temperature signal cleanly. The qualifier-exhaustion-memo's "
            "z-state-orthogonality finding generalizes to all natural conditioning."
        ),
    },
    "variables": {
        "dependent": [
            "alpha_L_per_stratum_per_axis",
            "range_alpha_L_per_axis",
            "ci95_alpha_L_per_stratum",
            "ci95_range_alpha_L_per_axis",
            "alpha_L_marginal",
            "binding_axis",
            "binding_axis_strata_ranking",
        ],
        "independent": [
            "stratification_axis",
            "stratum",
        ],
        "excluded": [
            {
                "variable": "covariate_dependent_joint_NPMLE_NPMLEmix",
                "reasoning": (
                    "The verdict architecture is range-threshold (R-A: range > 0.20; "
                    "R-C: < 0.10), not formal distance-covariance test. Plain "
                    "Patra-Sen 2016 per stratum suffices. Avoiding NPMLEmix's Mosek "
                    "dependency. Cite Deb et al. 2022 as the more rigorous joint "
                    "approach we deliberately don't use; cite Patra-Sen 2016 as the "
                    "per-stratum method. See thread scope_limits and "
                    "notes/literature/2026-05-26_stratified-patra-sen-scout.md."
                ),
            },
            {
                "variable": "temperature_stratification",
                "reasoning": (
                    "P1 stratifies on quantities already in hand (no exogenous data). "
                    "Temperature stratification belongs to Q1C (conditional experiment) "
                    "not P1 (diagnostic). If P1's R-C fires (all natural axes constant), "
                    "Q1C tests temperature as the prior-supported candidate even "
                    "without P1's pointer."
                ),
            },
            {
                "variable": "cross_axis_interactions",
                "reasoning": (
                    "P1 reports per-axis range only. Cross-axis interactions (e.g., "
                    "demand_quantile x time_of_day joint stratification) would require "
                    ">25 strata, sample-size issues, and a more complex verdict. "
                    "Reserved for thread amendment if marginal axes are inconclusive."
                ),
            },
        ],
    },
    "data": {
        "source": (
            "Q1's M1 (Student-t kernel) residuals on post-cutoff Ontario data. Loaded "
            "from scratch/data/distributional_class_q1/q1.pkl (per Q1's run); falls "
            "back to refit via the production fitter if the pickle is insufficient. "
            "Same post-cutoff population as Q1/Q2A/Q2B: all delivery days from "
            "2025-01-01 through 2026-05-16."
        ),
        "partition": {
            "train": (
                "pre-cutoff library was used by Q1 to fit M1; P1 does NOT refit. "
                "P1 operates on Q1's already-fit residuals."
            ),
            "dev": "not used",
            "test": "post-cutoff 2025-01-01 .. 2026-05-16 (same population as Q1/Q2A/Q2B)",
        },
        "cutoff_handling": (
            "Standard; P1 uses Q1's already-standardized residuals (residuals divided "
            "by M1's per-anchor Student-t scale parameter). Stratification axes are "
            "computed from the existing time index + actuals (no exogenous data, no "
            "leakage)."
        ),
    },
    "metric": (
        "PRIMARY: for each of 5 stratification axes, compute per-stratum alpha_L (lower 95% "
        "Patra-Sen 2016 bound on non-Gaussian mixture fraction), then range_axis = "
        "max(alpha_L_point) - min(alpha_L_point) across that axis's strata. The verdict "
        "metric is max_axis(range_axis).\n\n"
        "Stratification axes (5, pre-locked, matching thread skeleton candidate_set):\n"
        "  1. demand_quantile: terciles of concurrent actual demand (low / medium / high). 3 strata.\n"
        "  2. time_of_day: 4 bins of hour-of-day: overnight (0-5), morning ramp (6-10), "
        "afternoon (11-16), evening (17-23). 4 strata.\n"
        "  3. season: 3 bins of month: winter (Nov-Mar), summer (Jun-Aug), shoulder "
        "(Apr-May, Sep-Oct). 3 strata.\n"
        "  4. day_type: weekday, saturday, sunday. 3 strata. Included as reference "
        "(already used as conditioning in M1's fit, so any alpha_L variation across these "
        "strata is sanity-baseline, not novel signal).\n"
        "  5. hour_of_week: 168 hours collapsed to 8 bins of 21 hours each. Captures "
        "finer temporal structure than day_type+time_of_day separately.\n\n"
        "For each (axis, stratum):\n"
        "  - Collect M1's standardized residuals u_i = (z_actual - z_iter) / s_iter that "
        "fall in that stratum.\n"
        "  - Apply plain Patra-Sen (2016) two-component mixture estimator: treats empirical "
        "CDF F_n as alpha F_+ + (1-alpha) F_0 where F_0 is Gaussian(0,1) and F_+ is "
        "unspecified. Returns alpha_L^{95%} = lower 95% confidence bound on alpha.\n"
        "  - Bootstrap: 1000 paired-day resamples; recompute alpha_L per resample; "
        "report 95% CI on alpha_L per stratum.\n\n"
        "Per axis: range_axis = max(alpha_L_point) - min(alpha_L_point) across strata. "
        "CI on range_axis via the range statistic recomputed on each bootstrap resample.\n\n"
        "Verdict: binding_axis = argmax_axis range_axis. Verdict outcome (R-A/B/C) "
        "determined by max range value vs thresholds 0.20 / 0.10.\n\n"
        "SECONDARY / informational:\n"
        "  - alpha_L_marginal (unstratified Patra-Sen on full residual pool) as sanity "
        "check against qualifier-exhaustion-memo prior of 0.49-0.74. Large deviation "
        "indicates pipeline mismatch.\n"
        "  - Per-axis full strata ranking table for use by Q1A's Z construction.\n"
    ),
    "shape_outcomes": {
        "R-A": {
            "criterion": "max over 5 axes of range(alpha_L) > 0.20",
            "interpretation": (
                "Z detectable from current data; binding axis identifies where missing Z "
                "is concentrated. Routes to Q1A (Z-conditioning experiment using the "
                "strongest axis)."
            ),
        },
        "R-B": {
            "criterion": "max range in (0.10, 0.20]",
            "interpretation": (
                "Marginal variation; Z hint present but not cleanly localized. Routes to "
                "Q1A (use the marginal-strongest axis anyway)."
            ),
        },
        "R-C": {
            "criterion": "max range < 0.10 (all 5 axes return approximately constant alpha_L)",
            "interpretation": (
                "Z orthogonal to all 5 natural conditioning axes. Routes to Q1C "
                "(temperature fallback, the prior-supported Z that may be orthogonal to "
                "all our current stratifications)."
            ),
        },
    },
    "falsification_criterion": {
        "numeric": "R-C fires (max range < 0.10 across all 5 axes)",
        "reasoning": (
            "R-C is the substantive falsification of the noisy-structure hypothesis under "
            "the natural stratifications we have. If none of demand / time / season / "
            "day-type / hour-of-week shows systematic alpha_L variation, the missing "
            "content is orthogonal to all these natural axes - either intrinsic structured "
            "noise OR Z that requires data we don't have (temperature, weather, grid "
            "events). This makes the next step less informed and increases P2's relevance."
        ),
    },
    "corroboration_criterion": {
        "numeric": "R-A fires (max range > 0.20)",
        "reasoning": (
            "R-A localizes where the missing Z is concentrated. The binding axis becomes "
            "Q1A's conditioning variable. Most likely binding axes per priors: season "
            "(temperature proxy without temperature data) or hour-of-week (Q2A Finding #2 "
            "saturday 4x weekday is consistent with hour-of-week structure)."
        ),
    },
    "ambiguous_region": {
        "range": "R-B fires (max range in (0.10, 0.20])",
        "action": (
            "Q1A still fires using the marginal-strongest axis; the verdict is informative "
            "but not decisive. A subsequent thread amendment may be needed if Q1A's R-A1A "
            "doesn't fire."
        ),
    },
    "baselines": {
        "primary": (
            "qualifier-exhaustion-memo's alpha_L >= 0.49-0.74 prior for marginal "
            "(unstratified) sanity check. P1's alpha_L_marginal should land in or near "
            "this range; large deviation indicates pipeline mismatch."
        ),
        "secondary": (
            "Synthetic recovery gate (METHOD/DESIGN). Generate synthetic data with two "
            "known true-alpha settings: (i) alpha_true = 0.0 (pure Gaussian, expect "
            "alpha_L = 0 within Patra-Sen CI); (ii) alpha_true = 0.50 (50/50 Gaussian + "
            "mixture, expect alpha_L = 0.45-0.55 within Patra-Sen CI). Confirm the "
            "production fitter recovers both before the Ontario diagnostic runs. ~30 sec "
            "wall-clock; runs every invocation as in-script self-test."
        ),
    },
    "stopping_criterion": {
        "rule": (
            "Run synthetic recovery gate; load Q1 M1 residuals (or refit if pickle "
            "insufficient); stratify on 5 axes; apply Patra-Sen per stratum; bootstrap; "
            "compute range per axis; evaluate R-A/B/C; emit result. STOP. No iteration; "
            "no parameter tuning; no cross-axis interaction."
        ),
        "reasoning": (
            "Pre-registration complete; verdict determined by realized range values."
        ),
    },
    "ci_method": {
        "type": "paired_day_bootstrap",
        "details": (
            "1000 paired-day bootstraps of post-cutoff days. Per resample: recompute "
            "alpha_L per stratum per axis, recompute range_axis. Report 95% percentile "
            "CI on each alpha_L and each range."
        ),
        "width": "95 percent percentile interval",
    },
    "affects_registered_spec": False,
    "thread": {
        "thread_topic": "resolution-paths-thread",
        "thread_body_sha256": THREAD_HASH,
        "node_id": "P1",
        "parent_branch": None,
        "threshold_derivations": (
            "Numeric thresholds (R-A: range > 0.20; R-C: < 0.10) are NOT derived from a "
            "synthetic gate (unlike Q2A's gate-derived 228/2284). They are set on non-data "
            "principled bases:\n"
            "  - 0.20: substantive variation threshold. alpha_L spanning 20 percentage "
            "points across strata of a single axis is large enough that the binding "
            "stratum carries meaningfully more non-Gaussian mass than the lightest "
            "stratum; small enough to be detectable at n ~ 1000-3000 per stratum.\n"
            "  - 0.10: noise floor. Bootstrap CI width on alpha_L at the sample sizes we "
            "have is ~0.05-0.10 per stratum (extrapolating from Patra-Sen 2016 paper's "
            "reported behavior at n ~ 2000); range < 0.10 is within sampling noise.\n"
            "The qualifier-exhaustion-memo alpha_L >= 0.49-0.74 marginal range provides "
            "the prior for what unstratified alpha_L looks like; the stratified variation "
            "question is orthogonal."
        ),
    },
    "related_memory": [
        "mitacs-q2b-non-distributional",
        "mitacs-q2a-richer-family",
        "mitacs-q1-student-t-vs-gaussian",
        "mitacs-qualifier-exhaustion",
        "mitacs-clim-gap-nonstationary",
        "mitacs-postcovid-probe",
        "mitacs-multiscale-defect-eigenmodes-v1",
    ],
    "related_seed": [
        "notes/seeds/multiscale_factor_coherence.md",
        "writeup/tex/missing_content_memo.tex",
        "~/Research/Mathematics/Resolvent_Framework/notes/unsorted/disintegration_diagnostic.md",
        "~/Research/Mathematics/Resolvent_Framework/notes/unsorted/disintegration_diagnostic_audit_correction.md",
        "notes/literature/2026-05-26_stratified-patra-sen-scout.md",
    ],
}


def dump_yaml(data: dict, path: Path) -> None:
    path.write_text(
        yaml.safe_dump(data, default_flow_style=False, sort_keys=True, width=100),
        encoding="utf-8",
    )


def stamp_and_write(data: dict, path: Path) -> dict:
    stamped = registry.stamp(data, self_path=path)
    dump_yaml(stamped, path)
    return stamped


def main() -> int:
    DIR.mkdir(parents=True, exist_ok=True)

    print("--- stamping phase_a ---")
    phase_a_stamped = stamp_and_write(PHASE_A, PHASE_A_PATH)
    phase_a_hash = phase_a_stamped["body_sha256"]
    print(f"phase_a body_sha256 = {phase_a_hash}")
    print(f"phase_a git_sha     = {phase_a_stamped['git_sha']}")
    print(f"phase_a git_clean   = {phase_a_stamped['git_clean']}")

    proponent = {
        "schema": "proponent",
        "written_at": "2026-05-27T09:05:00-07:00",
        "references": [
            {"file": "phase_a.yaml", "body_sha256": phase_a_hash},
        ],
        "forecast": {
            "value": {
                "outcome": "R-A",
                "binding_axis": "season",
                "range_per_axis": {
                    "season": 0.30,
                    "hour_of_week": 0.25,
                    "time_of_day": 0.15,
                    "demand_quantile": 0.10,
                    "day_type": 0.05,
                },
                "alpha_L_marginal": 0.60,
                "max_range": 0.30,
            },
            "units": (
                "alpha_L and range dimensionless in [0, 1]; outcome category in "
                "{R-A, R-B, R-C}; binding_axis is one of {demand_quantile, "
                "time_of_day, season, day_type, hour_of_week}."
            ),
            "reasoning": (
                "Three lines of evidence point to R-A with season binding at "
                "range ~0.30 (max-over-axis).\n\n"
                "(a) The strongest a priori candidate Z is temperature "
                "(mitacs-clim-gap-nonstationary's +1000 MW gap, mitacs-postcovid-"
                "probe's 'exogenous info is the lever'). In the absence of exogenous "
                "data, the only natural axis that correlates strongly with temperature "
                "is season; the season axis should therefore inherit a temperature "
                "signal as a proxy. Winter (Nov-Mar) and summer (Jun-Aug) are "
                "high-load, high-volatility regimes where heavy-tailed residuals "
                "concentrate; shoulder (Apr-May, Sep-Oct) is the quiescent regime "
                "with closer-to-Gaussian residuals. A 0.30-point spread in alpha_L "
                "between winter/summer and shoulder is consistent with the magnitude "
                "of the climatology-gap effect.\n\n"
                "(b) Q2A's Finding #2 (saturday chi^2 ~ 4x weekday chi^2, family-"
                "invariant across M1/M3/M4/M5) suggests day-type or hour-of-week "
                "structure. Day-type is already conditioning M1's fit, so its alpha_L "
                "variation should be small (predicted range ~0.05). Hour-of-week is "
                "the finer-grained axis that captures within-week heterogeneity that "
                "day-type aggregates over; predicted range ~0.25, second-strongest.\n\n"
                "(c) qualifier-exhaustion-memo's marginal alpha_L >= 0.49-0.74 sets "
                "the unstratified prior; my point forecast 0.60 is mid-range and "
                "consistent with M1 residuals (which are NOT the corrected predictor "
                "but the diagnostic-level kappa_1 residual against actuals). Demand "
                "quantile (0.10) is intermediate: high-demand strata likely have "
                "fatter tails than low-demand strata, but the per-stratum sample "
                "size is reduced (~33%) and bootstrap noise widens the noise floor."
            ),
        },
        "forecast_ci": {
            "width": 0.95,
            "low": {
                "max_range": 0.18,
                "season_range": 0.18,
                "hour_of_week_range": 0.12,
                "alpha_L_marginal": 0.45,
            },
            "high": {
                "max_range": 0.45,
                "season_range": 0.45,
                "hour_of_week_range": 0.40,
                "alpha_L_marginal": 0.78,
            },
            "reasoning": (
                "CI on max_range [0.18, 0.45] straddles the R-A / R-B boundary (0.20). "
                "Central call R-A but the lower tail crosses into R-B. CI on season_range "
                "matches max_range (season is forecast to be the binding axis). "
                "hour_of_week_range CI [0.12, 0.40] also straddles the R-A bar; if "
                "hour_of_week binds instead of season the verdict is still R-A but with "
                "different downstream Z construction. alpha_L_marginal CI [0.45, 0.78] "
                "spans the qualifier-exhaustion-memo prior range."
            ),
        },
        "what_would_change_my_mind": [
            (
                "If max_range < 0.10 (R-C fires), no natural axis carries a localizing "
                "signal. Either the missing Z is exogenous (temperature) and orthogonal "
                "to all natural axes, or the noisy-structure / structured-noise tie "
                "leans toward intrinsic structured noise. The temperature prior is "
                "still defensible but season being a temperature proxy fails."
            ),
            (
                "If max_range > 0.20 but binding_axis != season (e.g., hour_of_week or "
                "time_of_day binds with > 0.25 range and season is < 0.15), the "
                "temperature-via-season interpretation is wrong on the proxy choice. "
                "R-A still fires but Q1A's downstream Z construction targets a "
                "temporal-cycle variable, not temperature."
            ),
            (
                "If alpha_L_marginal is outside [0.40, 0.85], there is a pipeline "
                "mismatch between Q1 M1 residuals and the qualifier-exhaustion baseline. "
                "The verdict is BLOCKED for diagnosis before proceeding."
            ),
            (
                "If day_type range > 0.15, M1's per-day-type conditioning has not "
                "absorbed the day-type asymmetry in non-Gaussian mass. This would be a "
                "Q1 / Q2A pipeline issue, not a P1 finding; arbiter should flag it."
            ),
        ],
        "confidence": 0.50,
    }

    da = {
        "schema": "devils_advocate",
        "written_at": "2026-05-27T09:10:00-07:00",
        "references": [
            {"file": "phase_a.yaml", "body_sha256": phase_a_hash},
        ],
        "strongest_argument": (
            "The proponent's central claim - that season acts as a temperature proxy "
            "with range ~0.30 alpha_L variation - rests on a chain of assumptions that "
            "the qualifier-exhaustion-memo's most-cited empirical finding directly "
            "contradicts. The memo's reported result is that alpha_L is ORTHOGONAL to "
            "z-state across the delay-embedding coordinates of the marginal kappa_1 "
            "residual. Z-state is a NATURAL CONDITIONING AXIS - it is exactly the "
            "kind of derived-from-existing-data stratification P1 is testing. The "
            "natural extension of that orthogonality finding is that alpha_L is "
            "orthogonal to ALL natural conditioning axes constructible from existing "
            "data (time index + actuals), because all such axes are themselves derived "
            "from the same underlying time series. The proponent's claim that season "
            "acts as a temperature proxy assumes that season carries information NOT "
            "already in the z-state (or the residual conditioning that produces it); "
            "but the season variable is a deterministic function of the time index, "
            "and the time index is already encoded in the z-state via the "
            "month-of-year and hour-of-day standardization. Season is therefore not a "
            "real independent axis - it is a re-encoding of information M1 has "
            "already conditioned on (the per-(month, hour) climatology mu_{m,h}, "
            "sigma_{m,h}).\n\n"
            "Second, the proponent's hour_of_week ~0.25 forecast is similarly "
            "vulnerable: hour-of-week = day-of-week x hour-of-day; day-of-week is "
            "already partitioned by day_type (weekday / saturday / sunday); "
            "hour-of-day is already partitioned by time_of_day. The 8-bin hour-of-"
            "week collapse re-aggregates information that the other natural axes "
            "already encode, just on a different grid. Q2A's saturday 4x weekday "
            "pattern survived FAMILY-INVARIANT - it is structural, not distributional - "
            "so it is structural at the level of the conditioning, not at the level "
            "of mixture-fraction asymmetry. P1's alpha_L on hour-of-week strata is "
            "consequently expected to be small.\n\n"
            "Third, the most-cited candidate Z (temperature) is NOT in any natural "
            "axis. Temperature varies within a season; temperature varies within a "
            "day; temperature varies within an hour. None of P1's 5 axes carries "
            "temperature as a separable signal. If the missing Z is temperature, it "
            "is orthogonal to ALL 5 natural axes by construction, and Q1C is the "
            "only branch that can detect it.\n\n"
            "Net: R-C fires with all five axis ranges < 0.10 (point ~0.05 each), "
            "max_range ~ 0.07. alpha_L_marginal lands at the upper end of the "
            "qualifier-exhaustion prior (~0.65) because the marginal mass is high but "
            "spatially uniform across natural axes."
        ),
        "specific_failure_modes": [
            {
                "rank": 1,
                "mode": (
                    "alpha_L_orthogonal_to_all_natural_axes_per_qualifier_exhaustion_"
                    "memo_extension_temperature_is_the_missing_Z_and_lies_outside_P1_"
                    "candidate_set"
                ),
                "memory_ref": "mitacs-qualifier-exhaustion",
                "likelihood": 0.55,
            },
            {
                "rank": 2,
                "mode": (
                    "season_is_not_an_independent_axis_because_it_is_a_function_of_"
                    "time_index_already_encoded_in_M1s_per_month_hour_climatology"
                ),
                "memory_ref": "mitacs-clim-gap-nonstationary",
                "likelihood": 0.40,
            },
            {
                "rank": 3,
                "mode": (
                    "hour_of_week_re_aggregates_day_type_x_time_of_day_information_"
                    "already_in_other_axes_no_independent_variation_signal"
                ),
                "memory_ref": "mitacs-q2a-richer-family",
                "likelihood": 0.35,
            },
            {
                "rank": 4,
                "mode": (
                    "demand_quantile_is_the_closest_to_an_independent_axis_high_demand_"
                    "strata_may_have_fatter_tails_but_per_stratum_sample_size_collapses_"
                    "to_n_around_1000_and_patra_sen_alpha_L_CI_width_swamps_the_point_"
                    "estimate_so_observed_range_stays_below_the_noise_floor_even_when_"
                    "true_range_is_nonzero"
                ),
                "memory_ref": None,
                "likelihood": 0.25,
            },
        ],
        "severity": "serious",
        "counter_prediction": {
            "value": {
                "outcome": "R-C",
                "binding_axis": None,
                "range_per_axis": {
                    "season": 0.05,
                    "hour_of_week": 0.05,
                    "time_of_day": 0.04,
                    "demand_quantile": 0.07,
                    "day_type": 0.03,
                },
                "alpha_L_marginal": 0.65,
                "max_range": 0.07,
            },
            "units": (
                "alpha_L and range dimensionless in [0, 1]; outcome category in "
                "{R-A, R-B, R-C}; binding_axis null when no axis exceeds the 0.10 noise "
                "floor."
            ),
            "reasoning": (
                "Disagreeing with the proponent on outcome category (R-C, not R-A). "
                "My central call is that all 5 natural axis ranges sit below the 0.10 "
                "noise floor (point ~0.03-0.06 each, max ~0.07). Reasoning: qualifier-"
                "exhaustion-memo's alpha_L-orthogonal-to-z-state finding extends to all "
                "natural axes derived from time index + actuals; season / hour-of-week / "
                "time-of-day / day-type are all functions of the time index already "
                "encoded in M1's per-(month, hour) standardization; demand_quantile is "
                "the closest to a real independent signal but sample-size collapse to "
                "tercile bins widens the Patra-Sen alpha_L CI and likely keeps range "
                "below noise floor. alpha_L_marginal ~0.65 is upper-mid of the "
                "qualifier-exhaustion prior, consistent with high marginal "
                "non-Gaussianity that is spatially uniform across natural strata. R-C "
                "routes to Q1C (temperature fallback), which is the prior-supported Z "
                "exogenous to P1's candidate set."
            ),
        },
        "forecast_ci": {
            "width": 0.95,
            "low": {
                "max_range": 0.03,
                "alpha_L_marginal": 0.55,
            },
            "high": {
                "max_range": 0.18,
                "alpha_L_marginal": 0.78,
            },
            "reasoning": (
                "CI on max_range [0.03, 0.18] mostly sits below 0.20 (R-A) but crosses "
                "into the R-B band (0.10, 0.20] in the upper tail. Central call R-C; "
                "the upper-tail R-B region would require demand_quantile to deliver "
                "a real signal that survives the noise floor. R-A is not in my 95% CI; "
                "I am committed to alpha_L being approximately constant across natural "
                "axes."
            ),
        },
        "what_would_change_my_mind": [
            (
                "If max_range > 0.25 with binding_axis = season AND CI lower bound > 0.20, "
                "season carries a temperature-proxy signal that is large enough to "
                "survive the noise floor with margin. The proponent's MW-space-to-z-space "
                "translation of clim-gap-nonstationary holds and my qualifier-exhaustion-"
                "memo extension is empirically refuted."
            ),
            (
                "If R-A fires for hour_of_week OR demand_quantile with binding range > "
                "0.25 (not season), the temperature-proxy reading is wrong on the proxy "
                "choice but R-A as a class is correct - some natural axis does carry "
                "localizing information. My pure-orthogonality position is refuted."
            ),
            (
                "If alpha_L_marginal lands at the low end of the qualifier-exhaustion "
                "prior (< 0.50) AND R-C still fires, the marginal pathology is less "
                "severe than the memo prior suggests; the structured-noise vs noisy-"
                "structure tie may be soluble within the natural-axis candidate set "
                "after all, contradicting my position that exogenous Z is required."
            ),
        ],
        "opportunity_cost": {
            "alternative": (
                "A direct one-shot test of season-as-temperature-proxy: compute the "
                "Pearson correlation between season (3-level ordinal) and historical "
                "Toronto Pearson hourly temperature (free ECCC archive) over the post-"
                "cutoff window. If the correlation is high (|rho| > 0.7), season is a "
                "decent proxy and the proponent's prior translation is defensible. If "
                "low (|rho| < 0.3), season is a poor proxy and Q1C's temperature direct "
                "test is the only viable detection branch. Pre-registering P1 without "
                "this number means the proponent's binding_axis = season forecast is "
                "speculative on a quantity easily measured."
            ),
            "ranking": "complementary",
            "notes": (
                "The pre-registration is well-formed AS WRITTEN. The opportunity cost is "
                "a one-shot empirical anchor that would have informed the proponent's "
                "binding_axis choice; P1 reports per-axis alpha_L regardless, so the "
                "empirical question is answered as a side effect even if my counter-"
                "prediction is the central call."
            ),
        },
    }

    print("--- stamping proponent ---")
    proponent_stamped = stamp_and_write(proponent, PROPONENT_PATH)
    print(f"proponent body_sha256 = {proponent_stamped['body_sha256']}")
    print(f"proponent git_sha     = {proponent_stamped['git_sha']}")
    print(f"proponent git_clean   = {proponent_stamped['git_clean']}")

    print("--- stamping devils_advocate ---")
    da_stamped = stamp_and_write(da, DA_PATH)
    print(f"da body_sha256 = {da_stamped['body_sha256']}")
    print(f"da git_sha     = {da_stamped['git_sha']}")
    print(f"da git_clean   = {da_stamped['git_clean']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
