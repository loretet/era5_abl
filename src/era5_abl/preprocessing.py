import xarray as xr
import numpy as np
import os, subprocess
from .config import (
    GRAVITY,
    DRY_AIR_GAS_CONSTANT,
    REFERENCE_PRESSURE,
    POISSON_EXPONENT
)


def compute_ecmwf_pressure_and_height(ds_ml: xr.Dataset, ds_srf: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Computes 3D/2D pressure [Pa] and height AGL [m] on ECMWF hybrid model levels
    using A_k and B_k coefficients embedded in the NetCDF dataset (from GRIB VCT).
    Built with help of AI. NOTE: this reconstruction is APPROXIMATE.
    """
    g = GRAVITY
    R_d = DRY_AIR_GAS_CONSTANT

    # Extract surface pressure [Pa]
    if "lnsp" in ds_ml:
        sp = np.exp(ds_ml["lnsp"])
    elif "sp" in ds_srf:
        sp = ds_srf["sp"]
    elif "sp" in ds_ml:
        sp = ds_ml["sp"]
    else:
        raise KeyError("Could not find surface pressure ('lnsp' or 'sp') in datasets.")

    # Identify vertical dimension name and extract model level numbers
    lev_dim = (
        "model_level"
        if "model_level" in ds_ml.dims
        else ("lev" if "lev" in ds_ml.dims else ("hybrid" if "hybrid" in ds_ml.dims else "level"))
    )
    levels = ds_ml[lev_dim].values.astype(int)  # e.g., array([110, 111, ..., 137])
    n_levels = len(levels)

    # Extract A and B hybrid coefficients
    if "hyai" in ds_ml and "hybi" in ds_ml:
        a_full = ds_ml["hyai"].values
        b_full = ds_ml["hybi"].values
    elif "vct" in ds_ml:
        vct = ds_ml["vct"].values
        n_half = len(vct) // 2
        a_full = vct[:n_half]
        b_full = vct[n_half:]
    else:
        # Fallback: Default ECMWF L137 coefficient tables if missing from GRIB header
        raise ValueError(
            "Hybrid level coefficients (vct/hyai/hybi) not found in NetCDF dataset."
        )

    # Dynamic slicing of coefficients for levels present in ds_ml
    k_min = int(levels.min())
    k_max = int(levels.max())
    
    a_half = a_full[k_min - 1 : k_max + 1]  # Length: n_levels + 1 (e.g., 29)
    b_half = b_full[k_min - 1 : k_max + 1]  # Length: n_levels + 1

    # Compute half-level pressure: shape (time, model_level_half)
    # broadcasting sp (time) with a_half/b_half (model_level_half)
    sp_vals = sp.values[..., np.newaxis]  # (time, 1)
    p_half = a_half + b_half * sp_vals  # (time, 138)

    # Full-level pressure: midpoint of half-levels
    p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])  # (time, 137)

    # Compute virtual temperature
    q = ds_ml["q"].values
    t = ds_ml["t"].values
    t_v = t * (1.0 + 0.608 * q)

    # Hydrostatic integration from surface (TOA = level 0, Surface = level N-1)
    # dln_p = ln(p_{k+1/2} / p_{k-1/2})
    dln_p = np.log(p_half[:, 1:] / p_half[:, :-1]) # WARNING: this is an approximation of the actual ECMWF algorithm!
    dphi = R_d * t_v * dln_p  # geopotential increment per layer

    # Integrate from surface upwards (reverse sum along level axis)
    phi_agl = np.cumsum(dphi[:, ::-1], axis=1)[:, ::-1]
    z_agl_vals = phi_agl / g  # Height AGL [m]

    # Format as xarray DataArrays matching ds_ml dimensions
    time_dim = "time" if "time" in ds_ml.dims else "valid_time"

    pressure_da = xr.DataArray(
        p_full,
        dims=[time_dim, lev_dim],
        coords={time_dim: ds_ml[time_dim], lev_dim: ds_ml[lev_dim]},
        attrs={"units": "Pa", "long_name": "Pressure on model levels"},
    )

    z_agl_da = xr.DataArray(
        z_agl_vals,
        dims=[time_dim, lev_dim],
        coords={time_dim: ds_ml[time_dim], lev_dim: ds_ml[lev_dim]},
        attrs={
            "units": "m",
            "long_name": "Height Above Ground Level",
        },
    )

    return pressure_da, z_agl_da

def standardize_surface_varnames(ds: xr.Dataset) -> xr.Dataset:
    """
    Maps ECMWF GRIB ParamIDs (varXXX) to standard short names in an xarray Dataset
    and updates variable 'long_name' attributes to full CDS variable descriptions.
    """
    param_info = {
        "var34": {"short_name": "sst", "long_name": "Sea surface temperature"},
        "var167": {"short_name": "t2m", "long_name": "2 metre temperature"},
        "var134": {"short_name": "sp", "long_name": "Surface pressure"},
        "var23": {"short_name": "cbh", "long_name": "Cloud base height"},
        "var188": {"short_name": "hcc", "long_name": "High cloud cover"},
        "var186": {"short_name": "lcc", "long_name": "Low cloud cover"},
        "var187": {"short_name": "mcc", "long_name": "Medium cloud cover"},
        "var159": {"short_name": "blh", "long_name": "Boundary layer height"},
        "var129": {"short_name": "z", "long_name": "Surface geopotential"},
    }

    # Update long_name attributes for present variables prior to renaming
    for var_id, info in param_info.items():
        if var_id in ds.data_vars:
            ds[var_id].attrs["long_name"] = info["long_name"]

    # Rename varXXX to short_name
    rename_dict = {
        var_id: info["short_name"]
        for var_id, info in param_info.items()
        if var_id in ds.data_vars
    }

    return ds.rename(rename_dict)

def prepare_dataset(grib_ml_path: str, grib_srf_path: str, location: str = None) -> tuple[xr.Dataset, xr.Dataset]:
    """
    Spatially averages GRIB files via CDO (-f nc4), explicitly computes
    3D pressure p(t, k) and hydrostatic height z_agl(t, k) from hybrid coefficients,
    derives theta_v, and reverses the vertical axis so k=0 is the surface.
    """
    P0 = REFERENCE_PRESSURE

    # Output paths
    ml_nc_path = grib_ml_path.replace(".grib", "_CDO_processed.nc")
    srf_nc_path = grib_srf_path.replace(".grib", "_CDO_processed.nc")

    # Run CDO fldmean with explicit -f nc4 flag
    for grib_path, nc_path in [
        (grib_ml_path, ml_nc_path),
        (grib_srf_path, srf_nc_path),
    ]:
        if not os.path.exists(nc_path) or os.path.getsize(nc_path) == 0:
            cmd = f'cdo -f nc4 fldmean "{grib_path}" "{nc_path}"'
            print(f"Executing: {cmd}")
            subprocess.run(cmd, shell=True, check=True)

    # Open spatially averaged NetCDF files
    ds_ml = xr.open_dataset(ml_nc_path, engine="netcdf4")
    ds_srf = xr.open_dataset(srf_nc_path, engine="netcdf4")

    # Standardize dimension names
    dim_map = {}
    if "valid_time" in ds_ml.dims:
        dim_map["valid_time"] = "time"
    if "hybrid" in ds_ml.dims:
        dim_map["hybrid"] = "model_level"
    elif "lev" in ds_ml.dims:
        dim_map["lev"] = "model_level"

    if dim_map:
        ds_ml = ds_ml.rename(dim_map)
        ds_srf = ds_srf.rename({k: v for k, v in dim_map.items() if k in ds_srf.dims})

    # Strip residual scalar lat/lon dimensions
    ds_ml = ds_ml.squeeze(drop=True)
    ds_srf = ds_srf.squeeze(drop=True)

    # Standardise surface dataset variable names
    ds_srf = standardize_surface_varnames(ds_srf)

    # Compute pressure [Pa] and height AGL [m] from hybrid coefficients
    p_da, z_agl_da = compute_ecmwf_pressure_and_height(ds_ml, ds_srf)

    # Assign pressure as a dataset variable and height AGL as a 2D coordinate
    ds_ml["pressure"] = p_da
    ds_ml = ds_ml.assign_coords(z=(("time", "model_level"), z_agl_da.values))

    # Compute Virtual Potential Temperature 
    t_v = ds_ml["t"] * (1.0 + 0.608 * ds_ml["q"])
    ds_ml["theta_v"] = t_v * (P0 / ds_ml["pressure"]) ** POISSON_EXPONENT

    # Compute wind speed 
    ds_ml["wind_speed"] = np.sqrt(ds_ml["u"]**2 + ds_ml["v"]**2)

    # Reindex vertical axis so k=0 is Surface and k=N-1 is Top of Atmosphere
    ds_ml = ds_ml.reindex(model_level=ds_ml.model_level[::-1])

    # Add location metadata
    if location:
        ds_ml.attrs["Location"] = location
        ds_srf.attrs["Location"] = location

    return ds_ml, ds_srf