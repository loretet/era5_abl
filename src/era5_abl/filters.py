import xarray as xr
import numpy as np
from .operations import interpolate_to_height
from pathlib import Path
from datetime import datetime, timezone


def filter_clouds(ds: xr.Dataset, ds_srf: xr.Dataset, lcc_thresh: float = 0.1, window_hours: int = 2) -> tuple[xr.Dataset, xr.Dataset]:
    """
    Filters input dataset based on Low Cloud Cover. The assumption is that middle to high clouds do
    not influence ABL turbulence (which is debatable).
    """ 
    # Get low cloud covers (potentially, the datasets also have middle nad high loud cover)
    lcc = ds_srf["lcc"] # shape (time,)

    # Check LCC on the current hour and the previous window_hours (assumes this is enough to ignore effect of cloud cover on turbulence)
    window_size = window_hours + 1
    lcc_rolling = lcc.rolling(time=window_size, min_periods=window_size).max()

    # Construct mask 
    mask_clouds = (lcc_rolling <= lcc_thresh)
    ds_ml_filtered = ds.where(mask_clouds, drop=True)
    ds_srf_filtered = ds_srf.sel(time=ds_ml_filtered["time"])

    return ds_ml_filtered, ds_srf_filtered

def filter_stability(ds: xr.Dataset, ds_srf: xr.Dataset, ri_surf_min: float = 0.0, 
                     ri_surf_min_height: float = 20.0, dtheta_tol: float = 0.0,
                     ) -> tuple[xr.Dataset, xr.Dataset]:
    """
    Retain times where:
    1. The surface-layer bulk Richardson number (Ri_b_srf, i.e. the same quantity used by the 
       transfer functions) at ri_surf_min_height is at least ri_surf_min.
    2. The whole layer below BLH is stable with respect to the surface ("parcel" criterion):
       theta_v(z) - theta_v_2m >= -dtheta_tol [K] at every model level below BLH. A surface parcel is 
       then negatively buoyant everywhere in the ABL, which removes convective and transition 
       (partly mixed) layers without using noisy vertical derivatives. dtheta_tol > 0 (e.g. 0.1K) keeps near-neutral layers.
    3. BLH is above ri_surf_min_height (the reference height lies inside the ABL).
    Requires Ri_b_srf (compute_bulk_Ri with reference_height=None) and theta_v_2m.
    """

    # 1. Surface layer: bulk Ri at the reference height
    ri_ref = interpolate_to_height(ds, "Ri_b_srf", None, ri_surf_min_height)
    mask_surf = (ri_ref >= ri_surf_min)

    # 2. Whole ABL: theta_v excess with respect to the surface at every level below BLH
    blh = ds_srf["blh"]
    mask_sub_blh = ds["z"] < blh
    dtheta_sub_blh = (ds["theta_v"] - ds_srf["theta_v_2m"]).where(mask_sub_blh)
    mask_abl = dtheta_sub_blh.min(dim="model_level") >= -dtheta_tol   # False if no level is below BLH

    # 3. Reference height inside the ABL
    mask_blh = blh > ri_surf_min_height

    # Apply final masking
    ds_ml_filtered = ds.where(mask_surf & mask_abl & mask_blh, drop=True)
    ds_srf_filtered = ds_srf.sel(time=ds_ml_filtered["time"])

    return ds_ml_filtered, ds_srf_filtered


def filter_wind_dir(ds: xr.Dataset, ds_srf: xr.Dataset, dir_min: float, dir_max: float) -> tuple[xr.Dataset, xr.Dataset]:
    """
    Filters dataset so that wind direction at all levels below BLH falls within
    the angular sector [dir_min, dir_max] in degrees. Handles sector boundaries crossing 0/360 degrees.
    """
    wind_dir = ds["wind_dir"]
    sub_blh_mask = (ds["z"] <= ds_srf.sel(time=ds.time)["blh"])

    # Evaluate sector condition (handles 0/360 wrap, e.g., [330, 30])
    if dir_min <= dir_max:
        in_sector = (wind_dir >= dir_min) & (wind_dir <= dir_max)
    else:
        in_sector = (wind_dir >= dir_min) | (wind_dir <= dir_max)

    # Levels below BLH must fall inside sector; levels above BLH pass automatically
    valid_levels = (~sub_blh_mask) | in_sector

    # Collapse along model_level to obtain 1D time mask and filter the dataset
    mask_time = valid_levels.all(dim="model_level")
    ds_ml_filtered = ds.where(mask_time, drop=True)
    ds_srf_filtered = ds_srf.sel(time=ds_ml_filtered.time)

    return ds_ml_filtered, ds_srf_filtered

def filter_ds_below_BLH(ds: xr.Dataset, ds_srf: xr.Dataset) -> xr.Dataset:
    """
    Filters dataset so that values above the ERA5-computed BLH are NaNs 
    """
    # Mask levels below BLH
    sub_blh_mask = (ds["z"] <= ds_srf["blh"].sel(time=ds.time))

    # Levels above BLH become NaN
    ds_ml_filtered = ds.where(sub_blh_mask, np.nan)

    return ds_ml_filtered

def print_filter_output(ds_initial: xr.Dataset, ds_filtered: xr.Dataset, step_name: str) -> dict:
    """
    Calculates and outputs the absolute number and percentage of retained timesteps
    relative to the initial dataset.
    """
    n_initial = ds_initial.sizes["time"]
    n_filtered = ds_filtered.sizes["time"]

    pct = (n_filtered / n_initial) * 100.0 if n_initial > 0 else 0.0

    print(
        f"[{step_name}] Retained: {n_filtered} / {n_initial} timesteps ({pct:.2f}%)"
    )

    return {
        "step": step_name,
        "n_initial": n_initial,
        "n_filtered": n_filtered,
        "percentage": pct,
    }

def save_filtered_dataset(
    ds: xr.Dataset, location: str,
    dataset_type: str, output_dir: str | Path,
    filter_params: dict,
) -> Path:
    """
    Save a filtered ERA5 dataset to NetCDF with filtering metadata via the
    'filter_params' dictionary.
    """

    if dataset_type not in {"lvls", "srf"}:
        raise ValueError(
            "dataset_type must be either 'lvls' or 'srf'."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # General info
    ds_out = ds.copy()
    ds_out.attrs.update({
        "location": location,
        "dataset_type": dataset_type,
        "filtering_applied": "cloud, stability, wind_direction",
        "date_saved_utc": datetime.now(timezone.utc).isoformat(),
    })

    # Add each filtering parameter as a separate NetCDF attribute
    for key, value in filter_params.items():
        ds_out.attrs[f"filter_{key}"] = value
    site_name = location.replace(" ", "") 
    filename = f"{site_name}_{dataset_type}_filtered.nc"

    output_path = output_dir / filename
    ds_out.to_netcdf(output_path, 
        encoding={var: {"_FillValue": None} for var in ds.data_vars},
    )

    print(f"Saved processed dataset: {output_path}")
    return output_path