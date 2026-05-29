# Within-stratum residual diagnostic — prose summary

**PROVENANCE-GRADE: INSPECTION-ONLY**

Run: 2026-05-29. Source data: `scratch/data/distributional_class_q1/q1.pkl`
(Q1's pit_M1 DataFrame, 12,000 post-cutoff PIT residuals, the same series
P1 ran on). Stratification: P1's production binning re-applied to give
(day_type, Z_c, Z_2) cells — the same 32-cell partition Q1B used.
Diagnostic script: `scratch/within_stratum_residual_diagnostic.py`.
Figure: `scratch/within_stratum_residual_diagnostic.pdf`.

## What the diagnostic asked

P1 (S11) localised across-stratum: the missing-Z signal sits on
`hour_of_week`, hot zone Fri-Sun, day-type Sat > Sun > Wkdy. Q1A/Q1B
showed that within-data σ-algebra refinement on those strata reduces
χ² only modestly (single-axis 0.18%; joint multi-axis 40% AT POINT
but CI 60×; Q1B' confirmed eval-window-noise dominates the CI). The
identifiability obstruction (Bergna et al. 2026 Prop 1; Heckman-Singer
1984; Allahverdyan 2020) means κ_Q alone cannot certify whether the
residual reflects intrinsic heavy tails or a latent-Z mixture.

But the framework *does* permit characterising **what properties any
successful exogenous Z must have** under the latent-Z reading. This
diagnostic asks the **within-stratum** complement to P1's
**across-stratum** localisation: for each of the 32 cells, what does
the residual structure look like *within* the cell? White (any
missing factor is fast-varying and hour_of_week-binned-out)? Strongly
autocorrelated (slow-varying factor signature)? Systematically
non-stationary across days?

## What the diagnostic found

The result is unambiguous and surprising in its uniformity.

**Within-cell lag-1 autocorrelation is strong across ALL 32 cells.**
Mean +0.669; median +0.699; range +0.33 to +0.82. Every cell clears
the Bartlett 95% null bound by a wide margin (typically the bound is
±0.1 at the cells' sample sizes; the realised values are 4-7×
above). This is decisively NOT white noise inside the strata Q1B's
finest σ-algebra produces. Residuals carry strong continuity from
each in-cell observation to the next, where "next" means "next
qualifying hour" — for saturday__zc6__morning_ramp, that's the next
HE-position on the same morning.

**Lag-5 autocorrelation is much weaker.** Mean ~+0.15; many cells
near zero or slightly negative. The autocorrelation is concentrated
at the within-day timescale (lag-1 = next hour in cell), not at the
week-to-week timescale (lag-5 = next week's same hour-of-week
position). This is the **slow-varying within-day continuity
signature**: the missing factor varies on a timescale shorter than
a week but longer than a single hour. Temperature carries exactly
this signature on a continental land-mass: hourly autocorrelation is
high (next hour's temperature ≈ this hour's); week-on-week
autocorrelation is much lower (next Saturday's morning temperature
need not resemble this Saturday's).

**Variance is systematically below uniform expectation.** Mean
variance ratio 0.66 (uniform would be 1.0; under-dispersion < 1).
This is the per-cell counterpart of S10c's under-dispersion finding
at the day-type level — the residuals fall on the conservative side
of the predictive distribution because the predictor's
σ-algebra-implied variance is too narrow. Combined with the
strong lag-1 ACF, this picture says the predictor is *both*
systematically narrow *and* the prediction error has temporal
continuity within strata.

**Between-day systematic shape is substantial.** SD of per-delivery-
day means within-cell is ~0.21 across cells (relative to the centred-
residual scale of ~0.29 = SD of Uniform[0,1]-0.5). Specific delivery
days sit systematically on one side of the predictive — saturday__
zc6__morning_ramp shows this most starkly with a per-day-mean SD of
0.10 (relatively tight given cell size, but still substantial).
Different Saturdays produce *systematically* different morning-ramp
forecast errors; this is the across-days variability the predictor
fails to capture. **A slow-varying exogenous Z that changes across
delivery days** is the natural candidate for this signature.

## What this implies for the candidate Z (temperature)

The three signatures together — strong within-day continuity (lag-1
ACF), weak week-on-week continuity (lag-5 ACF much smaller),
substantial between-day shape variance — describe the missing factor
as **a slow-varying continuous covariate whose hour-to-hour values
within a delivery day are highly correlated, but whose day-to-day
values are not strongly tied to the day-of-week.**

Temperature matches this description directly:

1. *Within-day continuity*: hourly temperature is one of the most
   autocorrelated atmospheric variables; HE7 → HE8 temperature
   correlation typically > 0.95 in continental climates.
2. *Weak week-on-week tie*: temperature evolves on synoptic-scale
   (3-5 day) timescales unrelated to the day-of-week. This Saturday
   morning's temperature need not resemble next Saturday morning's.
3. *Between-day variability of predictive bias*: cold or hot days
   produce systematically different demand-vs-forecast residuals;
   the predictor's σ-algebra has no information about *which days
   are cold or hot.*

The diagnostic does NOT certify that the missing factor IS
temperature (identifiability obstruction). It says the missing
factor is *consistent with* a slow-varying continuous covariate of
temperature's class. Humidity, wind, day-length, secular load
growth, and tariff schedule all have some properties of this class.
Temperature has the strongest combination per the prior-supported-
candidate framing.

## What this rules out

If the missing factor were *fast-varying* (load shocks, equipment
events), within-cell residuals would be white because hour_of_week
binning would absorb the average; observed ACF would be ~0. The
diagnostic rules this out at high confidence.

If the missing factor were *day-of-week structured but with no
within-day continuity* (e.g., a discrete categorical with weekday
flavour beyond saturday/sunday), lag-1 ACF inside z_c=6 cells
would be low; lag-5 (next week's same z_c position) would be
high. We see the opposite. Discrete-day-of-week-categorical also
ruled out.

If the missing factor were *purely seasonal* (slow drift over
months, no within-day shape), variance ratio would be ~1 within
cells (residuals would be uniform-like inside each cell, the
seasonal drift averaging out within hour-of-week strata). We see
under-dispersion. Pure seasonal also ruled out — or at least, not
the dominant component.

## Caveats

- **Identifiability obstruction**: this diagnostic operates under
  the latent-Z reading of the residual structure. The same
  signatures are consistent with intrinsic heavy-tailedness of the
  process (specifically, an AR(1)-with-heavy-tails innovation
  whose autoregressive coefficient happens to match the
  within-cell timescale). The diagnostic CANNOT distinguish; it
  characterises what a hypothetical Z would have to look like
  *if the latent-Z reading is correct.*
- **Spectral analysis is noisy at these cell sizes** (n=200-500
  per cell). The peak-frequency column in the per-cell table
  is reported but most peaks are at very low frequencies (slow
  drift), consistent with the lag-ACF picture but not adding
  independent information.
- **Lag-1 within-cell ≠ lag-1 in clock time**. For most cells the
  next-in-cell observation IS the next clock hour (consecutive
  HE-positions in z_2's time-of-day window). For the small
  z_2=overnight (h ∈ 1..6) cells, lag-1 is also next-clock-hour.
  But the period between cells differs across day-types — there
  are no saturday rows on weekdays — so cross-cell autocorrelation
  was deliberately not computed.
- **Sample size**: 12,000 rows total; 32 cells; ~150-500 rows per
  cell. ACF estimates are stable but not bootstrap-bounded here
  (descriptive scratch; bootstrap CIs would be a claim-grade
  follow-on, not appropriate for INSPECTION-ONLY).

## Falsifiable target for T1 (ECCC temperature acquisition)

T1's pre-registered hypothesis can now carry a sharper claim than
"temperature is the prior-supported candidate":

> Conditioning M3 on σ = (day_type, Z_c, temperature_decile) should
> reduce the within-cell lag-1 ACF of u_PIT from ~+0.67 toward 0,
> if temperature is the missing factor characterised by this
> diagnostic. The reduction in lag-1 ACF (in addition to the
> reduction in marginal-PIT χ² registered in T1's verdict cuts)
> is the discriminating test: a successful T1 should not just close
> the χ² gap but should also whiten the residuals at the
> hour-to-hour timescale.

**Suggested T1 secondary metric**: post-conditioning within-cell
lag-1 ACF, reported alongside the marginal-PIT χ² primary metric.
A falsifiable temperature outcome: χ² closes but lag-1 ACF stays
high → temperature absorbed marginal but not within-day continuity
→ wrong Z (or right Z at wrong granularity). χ² closes AND lag-1
ACF drops → temperature absorbed both → consistent with
temperature-is-the-Z reading at this paper's data resolution.

## ECCC scope implications

The diagnostic suggests **hourly temperature with high temporal
resolution is needed**, not daily-mean or seasonal-monthly. If the
data only carries daily-mean temperature, the within-day continuity
won't be captured and T1 will plausibly fail on the χ² side even
if temperature is the right Z at the daily level. Station selection
matters less than temporal resolution: any single station's hourly
temperature in the IESO zone will carry the within-day continuity;
multi-station averaging is a secondary concern.

The lag-1 ACF signature also justifies investing in **forecast
temperature** (not just observed) for the registered forward
experiment — the predictor needs to condition on the same-hour
temperature it would have at prediction time, which on a horizon ≥ 1
hour is the *forecast* temperature, not the realised. ECCC's
historical forecasts (not just historical observations) become a
relevant data target.

## Status

INSPECTION-ONLY diagnostic; the prose statements above are
descriptive characterisations, not claim-grade findings. The
deliverable is the falsifiable T1 target above. Not for inclusion
in the writeup. Suitable for informing T1's phase_a author at fill
time and for scoping the ECCC acquisition session.

Saved alongside the script + diagnostic plot in `scratch/`. Not
gitignored (text + scratch directory is tracked); not committed
to writeup. Referenceable from T1's phase_a as
`scratch/within_stratum_residual_diagnostic_summary.md` if and
when T1 phase_a is filled.
