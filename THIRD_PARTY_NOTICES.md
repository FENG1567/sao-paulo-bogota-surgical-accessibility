# Third-party sources and notices

The package contains author-derived research results. It does not bundle the
original databases, OpenTripPlanner binaries/graphs, GTFS/OSM archives, or
standalone Arial font files. Software dependencies retain their own licences. The MIT grant
is only for the authors' code; CC BY 4.0 is only for the authors' data/design
contributions and never overrides source-provider terms.

## Population and geography

GHS-POP R2023A, projected 2025 epoch, 30 arc seconds: European Commission,
Joint Research Centre, DOI https://doi.org/10.2905/2FF68A52-5B5B-4A22-8F40-C41DA8332CFE .
The official use conditions specify CC BY 4.0 and attribution:
https://human-settlement.emergency.copernicus.eu/GHSLhowToCite.php .
Product: https://human-settlement.emergency.copernicus.eu/ghs_pop2023.php .
Methodological reference: Pesaresi et al. (2024), Advances on the Global Human
Settlement Layer, DOI https://doi.org/10.1080/17538947.2024.2390454 .
Bogota-derived files include grid-centre coordinates, cell populations and
author route/classification results; the original raster is not included.
Sao Paulo uses IBGE census sectors and must not be attributed to GHSL.

Sao Paulo uses IBGE Census 2022 CD2022 sectors selected by CD_MUN=3550308.
Coordinates are EPSG:4326 Shapely representative points, not centroids. The
source population is whole-tract V0001 (total persons). V06004 is nominal
monthly mean income among household responsible persons with income, with
July 2022 as the income reference month. Derived quintiles are tract-income
groups, not individual income quintiles. IBGE must be credited for the source
sector keys, population and income fields. Official census download portal:
https://www.ibge.gov.br/estatisticas/sociais/populacao/22827-censo-demografico-2022.html?edicao=41852&t=downloads . IBGE municipal mesh terms are documented
at https://www.ibge.gov.br/biblioteca/visualizacao/livros/liv102152.pdf .
This mesh notice is not asserted to license every census input. No original
IBGE polygons or census archive are included.

Bogota residential-block estrato classifications and historical hospital
coordinates were supplied as local inputs. The estrato source is ManzanaEstratificacion; the recorded update is
2025-11-20, which does not establish the classification reference year:
https://datosabiertos.bogota.gov.co/dataset/estratificacion-para-bogota .
The historical hospital-point source URL/date were not fully retained. This package distributes
the authors' analytical classifications and summaries, not the original
polygons or hospital-point database. These metadata gaps prevent a claim of
exact raw-source reconstruction; see the external-data manifest.

OpenStreetMap: copyright OpenStreetMap contributors, source database under
ODbL; https://www.openstreetmap.org/copyright . No original OSM extracts or
administrative polygon database are bundled.

## Facilities, claims and transit

CNES, DATASUS SIH/SUS, REPS and Colombia installed-capacity sources remain
subject to their providers' terms; official links and recorded dates are in
`metadata/external_data_manifest.json`. No patient-level records, original
facility name/address directory, or claims archive is included. Reported
facility counts and estimates are derived analytical results.

SPTrans and SITP transit inputs are not redistributed. Feed calendars do not
identify the download snapshot; a fresh feed is not an exact replacement.
OpenTripPlanner 2.8.0 is upstream LGPL-3.0 software and is not bundled:
https://github.com/opentripplanner/OpenTripPlanner .

Arial is proprietary. Install a lawfully licensed Arial font locally, or use
an existing licensed installation. The rendering command accepts a font
directory; it does not download fonts and the repository contains none.

## Regional transit sensitivity

Supplementary Figure S2B uses the ARTESP regional bus feed, version 20261109,
retrieved 15 November 2026. Provider: https://dadosabertos.artesp.sp.gov.br/dataset/gtfs .
The frozen provider record states Creative Commons Atribuicao without a
version. The original feed is not redistributed; its recorded fingerprint
is supplied to distinguish it from later downloads.

The official Bogota portal identifies CC BY 4.0 for Manzana Estratificacion
(Secretaria Distrital de Planeacion); its original polygons are not bundled.
Publication PDF/SVG outputs contain font subsets or references as produced by
the original scientific renderer. These embedded subsets retain their font
provider rights; the author licences do not cover them. No standalone font
file is included or offered for extraction/redistribution.
