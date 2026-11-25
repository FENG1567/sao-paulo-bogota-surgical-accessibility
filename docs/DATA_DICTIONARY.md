# Data organisation and units

`metadata/data_dictionary.json` lists every included CSV, its columns and row
count. File-level SHA-256 values are in `metadata/release_manifest.json`.
CSV files use UTF-8 (some have a BOM); loaders use `utf-8-sig` as needed.

`data/source_data`: 24 CSVs containing frozen analytical inputs to the published
figures, longer-form numerical sources for the tables, and supporting reported
summaries. `data/tables`: 3 main display tables and 15 supplementary CSVs. The
18 table inputs comprise 17 publication displays (including S3a-c and S4a-c)
plus one auxiliary S9 full-field table. `metadata/table_crosswalk.json` maps
each display to its table label and, for Tables 3 and S10, to its source-data CSV.
The current Table 3 display is mirrored by `Table3_current_display_source.csv`;
the former Table 3 display is retained byte-for-byte as Table S10 in
`table_s10_sensitivity_summary.csv` and `TableS10_sensitivity_summary_source.csv`.

Time thresholds and travel/walk limits are in minutes. Coverage estimates
stored as fractions use 0-1; *_pct and explicitly percent-labelled fields use
0-100; contrasts labelled pp are percentage points. Do not infer a unit by
magnitude alone: the renderer's transformations are the executable mapping
from each input field to the displayed axis. Population is persons, facility
counts are numbers of sites, coordinates are longitude/latitude in degrees.
Sao Paulo origins are representative points of whole IBGE census sectors;
Bogota origins are GHSL raster cell centres. These spatial units differ.
The per-file schema records field names rather than inventing definitions for
undocumented historical columns. Blank/NA/nan values represent missing or
nonapplicable values; do not convert them to zero. Display-table cells can
contain already formatted text and confidence intervals.

R = registry definition; E = equipment definition; A = historical activity
definition in Sao Paulo. Bogota's linked public-provider subset is a distinct
local construct. Baseline, weekday and weekend profiles are separate routing
scenarios. Design confidence intervals quantify the specified origin-sampling
design; technical bounds concern route-search failures.

Mixed derivative geography/population/classification fields keep their
third-party provenance. See THIRD_PARTY_NOTICES.md and metadata/data_rights.json.
