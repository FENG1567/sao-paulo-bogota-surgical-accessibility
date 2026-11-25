# Reproducing the primary estimates

From the release root, run:

    python scripts/reproduce_estimates.py

The input defaults to data/estimation/times_primary.csv. Results are written to artifacts/estimation/ so generated files stay separate from the distributed inputs. Override paths with --times and --output-dir; --out is accepted as an alias for --output-dir. The script requires Python, NumPy, and pandas. It reproduces the primary estimates from the released origin-arm table and does not rerun routing or require the original GTFS, OSM, or facility archives.

## Input schema

Each row is one origin and analysis arm. uid is a pseudonymous area-level key shared across arms for pairing. Direct source geographic identifiers and coordinates are removed. Population and stratum values may still be linkable to public area-level records, so uid is not described as unlinkable.

| Field | Meaning |
| --- | --- |
| city | Study city |
| uid | Pseudonymous area-level origin key used to pair arms |
| stratum | Sampling-design stratum |
| pop | Population assigned to the origin area; required for ratio-estimator variance |
| design_weight | pop divided by inclusion_prob; equivalently pop times N_h divided by n_h |
| N_h | Eligible frame size in stratum h |
| n_h | Number of sampled origins in stratum h |
| inclusion_prob | Origin inclusion probability, n_h divided by N_h |
| scope | Facility-scenario code; names and definitions are in the source-data tables |
| mode | Travel mode or mode combination |
| profile | Routing time/profile definition |
| time_min | Minimum returned candidate-route time in minutes |
| status | Route-search status |
| finite_unknown | Indicates a technical/search failure that could hide a faster candidate route |

The São Paulo population field uses IBGE Census 2022 CD2022 whole-tract V0001 counts assigned to tract representative points. São Paulo income strata derive from V06004. IBGE attribution and applicable provider terms apply; the original census archive is not distributed or relicensed here.

The Bogotá population field uses positive full-cell GHS-POP R2023A values at 30 arc-seconds for the projected 2025 epoch, retained by center-in inclusion. GHS-POP is a European Commission / Joint Research Centre source under CC BY 4.0 and requires attribution. The original population-grid archive is not distributed here. Bogota design strata derive from Secretaria Distrital de Planeacion ManzanaEstratificacion, with the mapped reference year not established. Further source and rights details are in metadata/external_data_manifest.json and data/estimation/PROVENANCE.md.

## Estimands and outputs

The script evaluates thresholds from 10 through 60 minutes in five-minute steps.

- coverage_curves.csv reports the design-weighted percentage of the study-frame population represented by origins with a returned candidate route at or below each threshold. The lower estimate treats unresolved technical failures as uncovered; the upper bound treats those failures as potentially covered.
- mode_scope_contrasts.csv reports paired additive mode-by-scope contrasts on the same origins, with a pointwise design-based interval.
- ANALYSIS_NOTES.json records the estimator and interpretation.

Coverage uses the original stratified-SRS ratio estimator with finite-population correction and linearization variance. Census strata contribute zero sampling variance. Intervals are suppressed for a census or zero standard error. The paired contrast is descriptive and is not a causal difference-in-differences estimate. Results are conditional on the frozen candidate routes and network inputs; they do not represent observed clinical access.

The distributed fields contain author-computed route minima, statuses and sampling-design metadata, together with the attributed source area-population values required by the estimator. They contain no patient-level records, facility addresses, staff identifiers, or coordinates. The per-file SHA-256 values for the input and executable script are in data/estimation/SHA256SUMS.txt.
