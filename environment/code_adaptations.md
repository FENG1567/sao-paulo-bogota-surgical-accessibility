# Source adaptations

- `code/analysis/activity_definition/01_build_hospitals.py`: replaced two private absolute paths with a release-relative `data` default and `URBAN_SURG_DATA_DIR` / `SIH_CHOLE_PARQUET` environment overrides. Hospital selection and output calculations are unchanged.
- `code/analysis/estimation/v13_analyse.py`: added a `np.trapezoid` / `np.trapz` compatibility alias because the locked runtime uses NumPy 1.26.4. The existing trapezoidal integration operation is unchanged.
- The four `reader_layout_20261125` wrapper sources were copied from the source ZIP. The main wrapper's source-data and reference defaults are repository-relative; its only difference from the reader-audit original is those path defaults.
- `code/analysis/aggregation/v13_commonpool_sensitivity.py` and `v13_aggregate_routes.py`: added sibling-`routing` import path adapters; the route engine implementation is unchanged.
- `code/analysis/estimation/v13_threshold_interval_summary.py`: restored its missing `v13_final_scientific_qa.py` helper byte-for-byte from the authoritative `core` source (SHA-256 9076cbd08bd8628187b117fd62cdfb493e9251790dfd13d54832839381c7de30) and resolved the import beside the threshold script. The QA checks are unchanged.
