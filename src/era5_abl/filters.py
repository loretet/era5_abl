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

    # Check LCC on the current hour and the previous 3 (this assumes 3/4 hours are enough to ignore effect of cloud cover on turbulence)
    window_size = window_hours + 1
    lcc_rolling = lcc.rolling(time=window_size, min_periods=window_size).max()

    # Construct mask 
    mask_clouds = (lcc_rolling <= lcc_thresh)
    ds_ml_filtered = ds.where(mask_clouds, drop=True)
    ds_srf_filtered = ds_srf.sel(time=ds_ml_filtered["time"])

    return ds_ml_filtered, ds_srf_filtered

def filter_stability(ds: xr.Dataset, ds_srf: xr.Dataset, ri_surf_min: float = 0.0, 
                     ri_surf_min_height: float = 20.0, grad_tol: float = -2e-4, min_valid_fraction: float = 0.8, 
                     smooth_window: int = 3
                     ) -> tuple[xr.Dataset, xr.Dataset]:
    """
    Retain times where:
    1. Near-surface Ri is at least ri_surface_min.
    2. At least required_valid_fraction % of sub-BLH layers satisfy
       smoothened( d(theta_v)/dz ) >= gradient_tolerance [K/m].
    """

    # Get surface quantities
    ri_20m = interpolate_to_height(ds,"Ri_g", None, ri_surf_min_height)
    mask_surf = (ri_20m >= ri_surf_min)

    # Smooth out theta nad compute gradient
    theta_smooth = ds.theta_v.rolling(model_level = smooth_window, 
                                      center = True, 
                                      min_periods = 1  # to avoid smothign from edge Nans
                                    ).mean()
    dt_dz = np.full_like(ds["theta_v"].values, np.nan)

    for t in range(ds.sizes["time"]):
        dt_dz[t,:] = np.gradient(
                                theta_smooth.isel(time=t).values,
                                ds.z.isel(time=t).values
                            )

    # Assign the smoothed gradient to the dataset
    ds = ds.assign(dt_dz=(("time","model_level"), dt_dz))

    # Mask to consider only layers below BLH
    blh = ds_srf["blh"]
    mask_sub_blh = ds["z"] < blh

    # Loop through time to find valid timesteps
    dtdz_sub_blh = ds.dt_dz.where(mask_sub_blh)

    # Compute fraction of retained timesteps and obtain mask
    n_sub_blh = dtdz_sub_blh.notnull().sum(dim="model_level")  # total 
    n_valid = (dtdz_sub_blh >= grad_tol).sum(dim="model_level") # respecting grad_tol
    valid_times = (n_sub_blh > 0) & ((n_valid / n_sub_blh) >= min_valid_fraction)

    # Apply final masking
    ds_ml_filtered = ds.where(mask_surf & valid_times, drop=True)
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
    ds_out.to_netcdf(output_path)

    print(f"Saved filtered dataset: {output_path}")
    return output_path