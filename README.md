# ERA5 ABL

Utilities for processing ERA5 137-model-level and surface data and diagnosing atmospheric-boundary-layer properties.

This package is a work in progress and will be used for a future publication.

The package currently contains functions for:

- preparation of spatially averaged ERA5 datasets;
- reconstruction of model-level pressure and height AGL (full levels, as in the IFS);
- gradient and bulk Richardson numbers;
- comparison of diagnosed and ERA5 boundary-layer height;
- cloud, stability, and wind-direction filtering;
- Gryanik and Lüpkes stability functions;
- basic diagnostics and plotting.


## Installation

From the project root:

```bash
python -m pip install -e .
```

NOTE: The preprocessing code also calls the external `cdo` executable. CDO must be installed separately and available on the system `PATH`.

The dependencies needed for the package to work can be ofund under `scripts/setup.py`.
The user can also run `setup.py`to install all the required packages.


## Configuration

Stable site metadata and physical constants are stored in
`era5_abl.config` (modifiable in the `era5_abl/config.py` file).

For example:

```python
from era5_abl.config import SITE_CONFIGS

site = SITE_CONFIGS["Mace Head"]

print(site.area)
print(site.wind_sector)

## Data retrieval

Surface data are requested from the CDS site by site. Model level data are instead requested for a few areas
(`ERA5_RETRIEVAL_AREAS` in `era5_abl/config.py`) in pieces of about 15 days, because the CDS rejects longer
requests and handles few big requests better than many small ones. Each location is then cut out of the area
files locally, so a new location needs no new download if an area already contains it.

`scripts/download_era5.py` does all of this. The requests are first sent to the CDS, which works on them
(hours to days) whether the computer is on or not, and the results are downloaded afterwards:

```bash
python download_era5.py   # SUBMIT_REQUESTS = True, FETCH_REQUESTS = False: send the requests, then switch off
python download_era5.py   # SUBMIT_REQUESTS = False, FETCH_REQUESTS = True: download, extract, delete
```

Each downloaded area file is deleted as soon as the locations are extracted from it (`DELETE_AREA_FILES`), so
the disk only ever holds one area file plus the locations. Both steps can be interrupted and started again: the
request ids are in `ERA5_data/raw_lvls_data/cds_requests.json`, and finished files are never downloaded twice.
The CDS does not keep results forever, so fetch them within a few days.

The same functions can also be called directly:

```python
import era5_abl as eabl
from era5_abl.config import SITE_CONFIGS

eabl.submit_model_level_requests(dates="2020-01-01/2021-12-31", output_dir=data_dir)
eabl.fetch_model_level_requests(dates="2020-01-01/2021-12-31", output_dir=data_dir)
eabl.extract_site_data(SITE_CONFIGS["Mace Head"], dates="2020-01-01/2021-12-31",
                       data_dir=data_dir, output_path=model_level_path)
```

`parallel_retrieval` still downloads everything in one blocking go (used for the surface data), but for model
levels the two steps above are the better route.

### Checking the files

`python scripts/validate_era5.py` reads the headers of every site file (seconds, not minutes) and checks that each
hour of the period is there, **once**, in order and with all its fields. An hour that is present twice (it can
happen when a download is interrupted and resumed, or run twice at the same time) makes CDO see extra time
steps, so the level data no longer line up with the surface data and `prepare_dataset` stops. 
`python scripts/fix_repeated_hours.py` removes the later copies and keeps every original as `<name>.orig`
(`--dry-run` only reports).

## Basic usage

```python
import era5_abl as eabl

ds_ml, ds_srf = eabl.prepare_dataset(
    model_level_path,
    surface_path,
    location="Mace Head",
)

ds_ml = eabl.compute_grad_Ri(ds_ml)
ds_ml = eabl.compute_bulk_Ri(ds_ml, ds_srf, reference_height=None)
ds_ml = eabl.compute_BLH_from_Ri_b(ds_ml, Ri_c=0.25)

ds_ml, ds_srf = eabl.filter_clouds(ds_ml, ds_srf)
ds_ml, ds_srf = eabl.filter_stability(ds_ml, ds_srf)
```

Functions may also be imported from their defining modules:

```python
from era5_abl.operations import compute_bulk_Ri
from era5_abl.filters import filter_stability
```

## Module responsibilities

- `data_retrieve.py`: CDS requests (surface data per site, model level data per area and ~15 days).
- `preprocessing.py`: extraction of the sites from the area files, file conversion, dataset preparation, variable names.
- `gribpoint.py`: reads the single-site model level GRIB files directly (about 10 seconds instead of 20-30 minutes in CDO, identical values), finds and removes repeated hours.
- `transfer_functions.py`: normalised transfer functions from Gryanik and Lüpkes 2018 paper.
- `operations.py`: diagnosed BLH and BLH comparisons. Interpolation. Richardson number computation. PDFs and retention statistics. Some numerical diagnostics.
- `filters.py`: cloud, stability, and wind-direction filters.
- `plotting.py`: plotting only.

## Status

This package is under active development. 

The usage of CDO is debatable but much faster than Metview.
