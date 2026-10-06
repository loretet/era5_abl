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

Surface data are requested from the CDS site by site. Model level data are instead requested for a few big
areas (`ERA5_RETRIEVAL_AREAS` in `era5_abl/config.py`) in pieces of about 15 days, because the CDS rejects longer
requests and handles a few big requests better than many small ones. The sites are then extracted locally:

```python
import era5_abl as eabl
from era5_abl.config import SITE_CONFIGS

# 1. Download (hours to days): better with `nohup python scripts/download_era5.py > download.log 2>&1 &`
eabl.parallel_retrieval(dates="2020-01-01/2021-12-31", output_dir=data_dir, retrieve_srf_data=False)

# 2. One GRIB file per site (CDO sellonlatbox + mergetime), which is the input of prepare_dataset
eabl.extract_site_data(SITE_CONFIGS["Mace Head"], dates="2020-01-01/2021-12-31",
                       data_dir=data_dir, output_path=model_level_path)
```

The area files are in `data_dir/raw_lvls_data/<area name>/`. A new site only needs a download if no area contains it.

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
- `transfer_functions.py`: normalised transfer functions from Gryanik and Lüpkes 2018 paper.
- `operations.py`: diagnosed BLH and BLH comparisons. Interpolation. Richardson number computation. PDFs and retention statistics. Some numerical diagnostics.
- `filters.py`: cloud, stability, and wind-direction filters.
- `plotting.py`: plotting only.

## Status

This package is under active development. 

The usage of CDO is debatable but much faster than Metview.
