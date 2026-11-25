# Validation

The candidate uses the revised presentation02 figure sources and the frozen numerical inputs.

- `scripts/reproduce_estimates.py` and `scripts/verify_estimates.py`: PASS with unchanged pseudonymous route inputs and design metadata.
- `scripts/reproduce_tables.py`: PASS; all 18 table inputs exported with every CSV cell preserved.
- `scripts/reproduce_figures.py --font-dir <licensed Arial directory>`: PASS; five main figures and three supplementary figures redrawn from frozen source CSVs.
- `scripts/verify_release.py --check-artifacts`: PASS; 18 table inputs (17 publication displays and one auxiliary S9 source), eight one-page figure PDFs, 32 panel labels.

The figure PDFs in `reference/figures/` are the current presentation02 references. Validation does not compare against the superseded 35-panel presentation.

## Current figure comparison

Compared the eight `artifacts/figures/*.pdf` files with their corresponding `reference/figures/*.pdf` files at 72 dpi. Dimensions, extracted text, pixel differences, panel letters read from both PDFs, and out-of-page text counts are recorded in `figure_comparison.json`. The comparison completed for all eight figures.

Changed-pixel fractions: Figure1_population_facilities: 0.0000 changed pixels, Figure2_mode_scope_threshold: 0.0000 changed pixels, Figure3_walk_caps: 0.0000 changed pixels, Figure4_income_geography: 0.0000 changed pixels, Figure5_validation_sensitivity: 0.0000 changed pixels, FigureS1_car_time_sensitivity: 0.0000 changed pixels, FigureS2_paired_sensitivity: 0.0000 changed pixels, FigureS3_bogota_classification: 0.0000 changed pixels.
