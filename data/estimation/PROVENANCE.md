# Primary estimate input

times_primary.csv contains author-derived origin-arm route minima and sampling-design fields used by scripts/reproduce_estimates.py. Each row is one pseudonymous area-level origin and route arm. The uid values are newly generated opaque identifiers; direct source geographic identifiers and coordinates are removed. No mapping from source identifiers to these pseudonyms is included.

## Fields and design

- city, uid, stratum, scope, mode, profile: analysis arm and design strata.
- time_min, status, finite_unknown: returned candidate route minimum and route-search status used to compute point coverage and the technical-failure upper bound.
- pop: origin-area population, required by the stratified ratio-linearization variance.
- design_weight: pop / inclusion_prob (equivalently pop * N_h / n_h); N_h, n_h, and inclusion_prob preserve the sample-design metadata and finite-population correction.

For São Paulo, pop is the IBGE 2022 Census whole-tract V0001 count assigned to the tract representative point; São Paulo income-stratum information derives from IBGE 2022 Census V06004. For Bogotá, pop is the positive GHS-POP R2023A 30-arc-second full-cell population (projected 2025 epoch) retained by center-in inclusion. GHS-POP is a European Commission / Joint Research Centre source under CC BY 4.0 and requires attribution. Bogota design strata derive from Secretaria Distrital de Planeacion ManzanaEstratificacion. The input includes only route/design derivatives and attributed source population/stratum values needed for the estimator; it does not include source census or population-grid archives. IBGE source attribution and applicable provider terms are retained; the original census database is not relicensed by the authors.

The origin IDs are pseudonymous area-level identifiers. Population and stratum values can remain linkable to public area-level sources; the identifiers are not represented as unlinkable. No patient-level, facility-address, staff, or coordinate fields are included.

Estimates are conditional on the frozen queried candidate routes and network inputs. They do not measure observed clinical access.
