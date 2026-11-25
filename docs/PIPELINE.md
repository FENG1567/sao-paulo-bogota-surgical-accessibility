# Reproduction levels and original pipeline

1. **Frozen-results rendering:** the CSVs in `data/source_data` are the
   scientific inputs to the current figure renderer. `reference/figures`
   contains eight comparison PDFs, not inputs used as replacement outputs.
   `data/tables` contains the displayed main and supplemental tables. Table
   exports preserve every supplied cell; they do not estimate statistics.
2. **Re-estimation from computed route minima:** if `data/estimation` and
   `scripts/reproduce_estimates.py` are present, see `ESTIMATION.md` and the
   validation report for the exact result families tested. These inputs start
   after routing; this is not a fresh transit-network calculation.
3. **Raw-source to routes:** original authors' analysis code is retained in
   `code/analysis` for inspection. Its historical scripts require their stated
   inputs, and some retain historical project-layout assumptions. They are
   not the portable quick-start entry points. The chain is activity/facility
   definition -> OTP routes -> route-minimum aggregation -> survey-weighted
   estimates -> publication outputs. Consult the manifest for external data.

## Raw-source requirements

The routing stage requires OpenTripPlanner 2.8.0, matching built graphs,
network/build configuration, OSM and GTFS snapshots, destination inventories,
and sampled origins. The original A118 historical classifier and several
precise historical input identifiers have not been fully recovered. The
later activity reconstruction is not asserted to be the lost original
classifier. Consequently the package does not claim one-command exact
raw-source-to-paper reproduction. Downloading present-day sources would not
resolve historical snapshot uncertainty.

Original R claims extraction additionally needs `read.dbc`, `data.table` and
`arrow`; these are not required by the Python frozen-result quick start.
No raw SIH claims are supplied. All published analyses use facility-level or
area-level records, not identifiable participant records.

## Interpretation

The primary comparison is two cities, Sao Paulo and Bogota. R/E/A are facility
definitions, not randomized interventions. WT includes transit with walking
access/egress and a direct-walk alternative. The design confidence intervals
and the route technical-failure bounds answer different questions and must
not be combined as total model uncertainty. Unknown socioeconomic groups
remain in the overall denominator but are excluded from known-group contrasts.

## Configuration templates

`config/routing_baseline_template.json` preserves the baseline query
parameters and frozen graph hashes, with worker concurrency reduced from 8
to 1. City build/router configurations are provided where recovered. They
are templates requiring externally supplied matching input archives, origin
and facility sets and OTP graphs. They are not an auto-download workflow.
Recorded historical raw-input fingerprints are in
`metadata/raw_input_fingerprints.json`; missing snapshots are not supplied
by substituting newer data.
