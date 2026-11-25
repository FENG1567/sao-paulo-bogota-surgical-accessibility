# Surgical facility accessibility: reproduction package

Code and derived results for the study of surgical-facility accessibility in
São Paulo and Bogotá. The package contains 43 CSV files: 24 source-data, 18 table, and one estimation CSV. The release contains the numerical sources for five
main figures, three supplementary figures, three main tables and Tables
S1–S10, including their labelled sub-tables. The historical project directory
name mentions four cities; the present paper's primary comparison uses two.

## Start here

Use Python 3.12. Create an isolated environment and install the pinned Python
requirements from the repository root:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead:
# .\.venv\Scripts\Activate.ps1
python -m pip install -r environment/requirements-lock.txt
python scripts/verify_release.py
python scripts/reproduce_estimates.py
python scripts/verify_estimates.py
python scripts/reproduce_tables.py
python scripts/reproduce_figures.py --font-dir /path/to/licensed/Arial
python scripts/verify_release.py --check-artifacts
```

On Windows the renderer can use Arial from the installed Windows font
directory; omit `--font-dir` if the automatic font lookup succeeds. On other
systems point to lawfully installed `arial.ttf` and `arialbd.ttf`. No font
download or redistribution is included. The publication figures use Arial
Bold throughout and 18-point uppercase panel labels. Substituting a font
changes the layout and is not the publication-matching validation path.

Outputs go to `artifacts/figures` and `artifacts/tables` by default. The
figure command genuinely redraws from CSVs; it does not copy the reference
PDFs. Table export preserves the supplied display cells and adds readable
Markdown tables. Both commands accept an alternative output directory; use
their `--help` for the exact syntax.

If `scripts/reproduce_estimates.py` is present, see
[`docs/ESTIMATION.md`](docs/ESTIMATION.md) for the separate computation of
estimates from pseudonymous area-level route minima and sampling-design metadata. The
validation report states which result families were matched. Re-rendering
figures, exporting table cells and re-estimating statistics are distinct steps.

## Contents

| Location | Purpose |
|---|---|
| `scripts/` | Portable rendering, export, integrity and re-estimation entry points |
| `code/figures/` | Current publication figure implementations |
| `code/analysis/` | Original activity, routing, aggregation and estimation code |
| `data/source_data/` | 24 frozen numerical and classification source CSVs for figures/results |
| `data/tables/` | 18 table input CSVs: 17 publication displays and one S9 auxiliary source (15 supplementary CSVs) |
| `data/estimation/` | One minimal pseudonymous computed-route CSV, when validated |
| `reference/figures/` | Eight current PDFs for output comparison |
| `environment/` | Dependency specification and version/source records |
| `metadata/` | File hashes, schemas, figure/table crosswalks and source/rights metadata |
| `docs/` | Data dictionary, pipeline, reproduction scope and validation evidence |

The original analysis code is retained for methodological inspection and
advanced reuse. Its historical entry points are not interchangeable with the
portable scripts. They need external inputs and their expected project
layout. See [`docs/PIPELINE.md`](docs/PIPELINE.md).

## Reproduction scope

This package supports frozen-result figure reconstruction and table export.
Any verified re-estimation step starts from already computed route minima,
not from fresh network routing. Exact raw-source-to-paper reconstruction
would additionally need matching OTP graphs, OSM/GTFS snapshots, sampled
origins and destination inputs. Some historical source identifiers and the
original A118 classification command remain incompletely recovered; the
manifest records these gaps. Current online downloads must not be presented
as the original historical snapshots.

`docs/figure_comparison.json` records fresh PDF-to-PDF raster, text, dimension, and panel-letter comparisons for the current eight figures. Panel letters are extracted from the PDFs themselves.

All eight current figure inputs are mapped in
`metadata/figure_crosswalk.json`. Figure 4 includes area/grid-origin
classification records required to reconstruct the map. These are not
patient-level data. Third-party origin geography/population contributions
retain their provenance and terms.

## Licences and citation

Original authors' code and documentation: **MIT**, see [`LICENSE`](LICENSE).
Authors' own summary results, estimates and figure designs: **CC BY 4.0**,
see [`LICENSE-DATA.md`](LICENSE-DATA.md). Mixed derivative source fields,
upstream databases and software are covered separately in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and
`metadata/data_rights.json`. This is not blanket relicensing of third-party
databases. Standalone font files and original third-party archives are excluded. Reference PDFs retain their embedded font-subset rights.

Use `CITATION.cff` to cite this package and cite the associated article when
its bibliographic record becomes available. Repository: https://github.com/FENG1567/sao-paulo-bogota-surgical-accessibility.
The manuscript version is release [`v2026.11.25`](https://github.com/FENG1567/sao-paulo-bogota-surgical-accessibility/releases/tag/v2026.11.25).

## 中文使用说明

本目录是可上传 GitHub 的复现材料，不是投稿正文文件夹。自编代码采用 MIT；
作者自有汇总结果采用 CC BY 4.0。第三方基础数据不因进入分析而转为作者所有。

先运行完整性检查，再导出表格和重绘图件。表格导出不能称为重新估计；图件
重绘也不能称为重新运行公共交通模型。统计重算的具体范围见 ESTIMATION.md。
未打包患者记录、原始名录、原始交通/地理档案、字体或内部审稿聊天。

各源文件的来源缺口、许可边界和实际验证结果均保留在元数据及说明文件中。
请整目录上传并保留目录结构，不要只上传一个脚本或只上传结果 CSV。
