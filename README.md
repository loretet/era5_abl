# ERA5 ABL

Utilities for processing ERA5 137-model-level and surface data and diagnosing atmospheric-boundary-layer properties.

This package is a work in progress and will be used for a future publication.

The package currently contains functions for:

- preparation of spatially averaged ERA5 datasets;
- reconstruction of model-level pressure and approximate height AGL;
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

## Basic usage

```python
import era5_abl as eabl

ds_ml, ds_srf = eabl.prepare_dataset(
    model_level_path,
    surface_path,
    location="Mace Head",
)

ds_ml = eabl.compute_grad_Ri_z(ds_ml)
ds_ml = eabl.compute_bulk_Ri(ds_ml)
ds_ml = eabl.compute_BLH_from_Ri_b(ds_ml)

ds_ml, ds_srf = eabl.filter_clouds(ds_ml, ds_srf)
ds_ml, ds_srf = eabl.filter_stability(ds_ml, ds_srf)
```

Functions may also be imported from their defining modules:

```python
from era5_abl.operations import compute_bulk_Ri
from era5_abl.filters import filter_stability
```

## Module responsibilities

- `preprocessing.py`: file conversion, dataset preparation, variable names.
- `transfer_functions.py`: normalised transfer functions from Gryanik and Lüpkes 2018 paper.
- `operations.py`: diagnosed BLH and BLH comparisons. Interpolation. Richardson number computation. PDFs and retention statistics. Some numerical diagnostics.
- `filters.py`: cloud, stability, and wind-direction filters.
- `plotting.py`: plotting only.

## Status

This package is under active development. 

The reconstructed model-level height is approximate and should be validated before using it for high-accuracy near-surface gradients.

The usage of CDO is debatable but much faster than Metview.
