#%% Imports
import xarray as xr
import numpy as np
import metview as mv
import seaborn as sns
import os, subprocess
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
# also requires CDO installed in the python environment

# surface data for one year over Cabauw took roughly 2 hours/year (8.6 Mb) to download
# model level data takes around 30 hour/year (1.28 Gb) to download
# CDO then processes the in roughly 8 min/year per file (whereas metview processing would hang on my laptop - hence CDO)'


#%%
################################################
###############  FUNCTIONS  ####################
################################################

#%% Preporcessing functions
def compute_ecmwf_pressure_and_height(ds_ml: xr.Dataset, ds_srf: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Computes 3D/2D pressure [Pa] and height AGL [m] on ECMWF hybrid model levels
    using A_k and B_k coefficients embedded in the NetCDF dataset (from GRIB VCT).
    Built with help of AI.
    """
    g = 9.80665
    R_d = 287.058

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
    g = 9.80665
    P0 = 100000.0

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
    ds_ml["theta_v"] = t_v * (P0 / ds_ml["pressure"]) ** 0.2854

    # Compute wind speed 
    ds_ml["wind_speed"] = np.sqrt(ds_ml["u"]**2 + ds_ml["v"]**2)

    # Reindex vertical axis so k=0 is Surface and k=N-1 is Top of Atmosphere
    ds_ml = ds_ml.reindex(model_level=ds_ml.model_level[::-1])

    # Add location metadata
    if location:
        ds_ml.attrs["Location"] = location
        ds_srf.attrs["Location"] = location

    return ds_ml, ds_srf

#%% Operational functions

def interpolate_to_height(ds: xr.Dataset, var_name: str, ds_srf: xr.Dataset = None, target_height: float = None) -> xr.DataArray:
    """
    Linearly interpolate var_name to a constant physical height or to the time-dependent BLH.
    If target_height is None, interpolates to BLH. Otherwise interpolates to the specified height.
    Assumes z(time, model_level) is ordered monotonically from the surface upward.
    """
    values = np.empty(ds.sizes["time"], dtype=float)
    use_blh = target_height is None

    for t in range(ds.sizes["time"]):
        z_t = ds["z"].isel(time=t).values
        var_t = ds[var_name].isel(time=t).values
        
        if use_blh:
            if ds_srf != None:
                height = float(ds_srf["blh"].isel(time=t))
            else:
                raise ValueError("interpolate_to_height: ds_srf not provided (BLH impossible to retrieve)")
        else:
            height = target_height

        # Clamp values to lowest model level if height is lower (higher) than z_min (z_max)
        if height < z_t[0]:
            values[t] = var_t[0]
        elif height > z_t[-1]:
            values[t] = var_t[-1]
        else:
            values[t] = np.interp(height, z_t, var_t)

    if use_blh:
        name = f"{var_name}_at_BLH"
    else:
        name = f"{var_name}_{target_height:g}m"

    return xr.DataArray(
        values,
        dims="time",
        coords={"time": ds["time"]},
        name=name,
        attrs=ds[var_name].attrs,
    )

def compute_grad_Ri_z(ds: xr.Dataset) -> xr.Dataset:
    """
    Computes Gradient Richardson Number profiles over non-uniform physical height
    z(t, k) using central differences.
    """

    g = 9.80665

    z = ds["z"].values  # (time, level)
    theta_v = ds["theta_v"].values  # (time, level)
    u = ds["u"].values  # (time,level)
    v = ds["v"].values  # (time, level)

    n_times, n_levels = z.shape
    dtheta_v_dz = np.full((n_times, n_levels), np.nan)
    du_dz = np.full((n_times, n_levels), np.nan)
    dv_dz = np.full((n_times, n_levels), np.nan)

    # Differentiate along level axis for each timestep against variable height z(t, k)
    for t in range(n_times):
        valid = (
            np.isfinite(z[t])
            & np.isfinite(theta_v[t])
            & np.isfinite(u[t])
            & np.isfinite(v[t])
        )
        # z[t, :] is a 1D array of non-uniform coordinates for timestep t -> np.gradient can be used (only when iterating in time)
        z_t = z[t, valid]
        dtheta_v_dz[t, valid] = np.gradient(
            theta_v[t, valid],
            z_t,
            edge_order=2,
        )
        du_dz[t, valid] = np.gradient(
            u[t, valid],
            z_t,
            edge_order=2,
        )
        dv_dz[t, valid] = np.gradient(
            v[t, valid],
            z_t,
            edge_order=2,
        )
    shear_sq = du_dz**2 + dv_dz**2

   # Avoid division by zero and compute Richardson
    valid_ri = (
        np.isfinite(dtheta_v_dz)
        & np.isfinite(shear_sq)
        & (shear_sq >= 1e-6)
        & (theta_v > 0)
    )
    Ri = np.full_like(theta_v, np.nan, dtype=float)
    Ri[valid_ri] = (
        (g / theta_v[valid_ri])
        * dtheta_v_dz[valid_ri]
        / shear_sq[valid_ri]
    )

    # Assign to the starting dataset
    ds = ds.assign(
        Ri_g=(("time", "model_level"), Ri),
        shear_sq=(("time", "model_level"), shear_sq),
        dtheta_v_dz=(("time", "model_level"), dtheta_v_dz),
    )
    ds["Ri_g"].attrs.update({
        "long_name": "Gradient Richardson number",
        "shear_sq_min": 1e-6,
    })

    return ds

def compute_bulk_Ri(ds: xr.Dataset, reference_height: float = 20.0, velocity_offset: float = 0.0) -> xr.Dataset:
    """
    Compute a bulk Richardson-number profile relative to a near-surface
    reference height.
    """

    g = 9.80665

    # Get reference va,lues
    theta_ref = interpolate_to_height(ds, "theta_v", None, reference_height)
    u_ref = interpolate_to_height(ds, "u", None, reference_height)
    v_ref = interpolate_to_height(ds, "v", None, reference_height)
    z = ds["z"]

    # Get differences
    delta_z = z - reference_height
    delta_theta = ds["theta_v"] - theta_ref
    delta_u = ds["u"] - u_ref
    delta_v = ds["v"] - v_ref

    # Compute Ri_b
    denominator = (
        delta_u**2
        + delta_v**2
        + velocity_offset**2
    )
    Ri_bulk = (
        g
        * delta_theta
        * delta_z
        / (theta_ref * denominator)
    )
    # Bulk Ri below/reference height is not used.
    Ri_bulk = Ri_bulk.where(delta_z > 0)

    # Assign to dataset
    ds = ds.assign(Ri_b=Ri_bulk)
    ds["Ri_b"].attrs.update({
        "long_name": "Bulk Richardson number",
        "reference_height": reference_height,
        "velocity_offset": velocity_offset,
    })

    return ds

def compute_BLH_from_Ri_b(ds: xr.Dataset, Ri_c: float = 0.25, z_min: float = 20.0, 
                          z_max: float | None = 3000.0, persistence: int = 2
                          ) -> xr.Dataset:
    """
    Computes the boundary layer height (BLH) from the first upward crossing of Ri_bulk = Ri_c.
    persistence=int means the threshold must remain exceeded for at least X consecutive model levels.    
    """

    # Initiate BLH array
    BLH = np.full(ds.sizes["time"], np.nan)
    n_times = ds.sizes["time"]

    # Search first value of z where Ri exceeds critical value for each time step
    fail_count = 0
    for t in range(n_times):
        z = ds["z"].isel(time=t).values
        Ri = ds["Ri_b"].isel(time=t).values
        # only considers values within the heights of choice
        z_mask = (z > z_min)
        if z_max is not None:
            z_mask &= (z < z_max)
        Ri = Ri[z_mask]
        z = z[z_mask]
        # Check that if a value exceeds Ri_c, it does so for #'persistence' layers
        crossing_found = False
        consec_count = 0
        for k,zz in enumerate(z):
            if Ri[k] >= Ri_c:
                if consec_count == 0:
                    start_z = zz
                consec_count += 1
                if consec_count == persistence:
                    BLH[t] = start_z
                    break 
            else: 
                consec_count = 0

        if consec_count < persistence:
            fail_count += 1

    if fail_count >= 1:
        print(f"WARNING - compute_BLH_from_Ri_b: Threshold Ri_b was not crossed more than \
            persistence={persistence} times in a row\nfor {fail_count} timesteps. For these, BLH is NaN")

    # Assign to BLH variable in the dataset
    ds = ds.assign(BLH_Ri=("time", BLH))

    return ds 

def compare_diagnosed_and_era5_blh(
    ds_ml: xr.Dataset,
    ds_srf: xr.Dataset,
) -> None:
    diagnosed = ds_ml["BLH_Ri"]
    era5 = ds_srf["blh"].sel(
        time=diagnosed["time"]
    )

    difference = diagnosed - era5

    print(
        "Diagnosed BLH quantiles:",
        diagnosed.quantile(
            [0.05, 0.5, 0.95]
        ).values,
    )

    print(
        "ERA5 BLH quantiles:",
        era5.quantile(
            [0.05, 0.5, 0.95]
        ).values,
    )

    print(
        "Difference quantiles [m]:",
        difference.quantile(
            [0.05, 0.5, 0.95]
        ).values,
    )

def compute_wind_dir(ds: xr.Dataset) -> xr.Dataset:
    """
    Computes the meteorological wind direction based on u,v on model level data and adds it to the dataset.     
    """
    u = ds["u"]
    v = ds["v"]

    # Computes direction from which the wind is blowing 
    windir_rad = np.arctan2(-u,-v)
    windir_deg = np.degrees(windir_rad) % 360

    # Assigns to dataset
    ds = ds.assign(wind_dir = windir_deg)

    return ds

def compute_PDF(ds: xr.Dataset, var_name: str, bins: int = 50) -> tuple:
    """
    Computes the Probability Density Function (PDF) of a given variable in the dataset.
    By default, the variable is flattened across all dimensions.
    Returns the histogram counts and bin edges.
    """

    data = ds[var_name].values.flatten()
    data = data[np.isfinite(data)]  # take out NaNs
    hist, bin_edges = np.histogram(data, bins=bins, density=True)

    return hist, bin_edges

def compute_difference_surface_top_ABL(ds_ml: xr.Dataset, ds_srf: xr.Dataset, var_name: str) -> xr.DataArray:
    """
    Compute the difference of a model level variable between the surface and the top of the ABL as a
    function of time. The top of the ABL is defined by the BLH (Boundary Layer Height) variable.
    """
    # Top of ABL variable (interpolate/extract at BLH)
    toa_var = interpolate_to_height(ds_ml, var_name, ds_srf)

    # Surface variable
    if var_name == "t":
        surf_var = ds_srf["t2m"] # no proper surface temperature available
    elif var_name == "wind_speed":
        surf_var = interpolate_to_height(ds_ml, var_name, None, 2.0)# wind speed at "surface" (2m)
    else:
        surf_var = ds_srf[var_name]

    return toa_var - surf_var 

def compute_epsilon(ds: xr.Dataset, location: str) ->  tuple[float, float]:
    """ 
    Computes ration between z (20 m) and z0 or zt from surface data.
    """
    # One can either select time-varying values (based on seasonality and land cover) from ERA5 or assume constant
    # values depending on the location. Here, I choose to keep them constant (seemed suitable to the scope of this work).
    # Bear in mind that epsilon only appears in logarithms in GL18 functions, meaning that the roughenss lengths variations 
    # are compressed (e.g. 100% variation in z0 leads to 13% variation in log(z/z0) with z=20m)
    if ds.attrs['Location'] == "Mace Head": # Roughly based on He et al. (2021). Reduced Sea-Surface Roughness Length at a Coastal Site. Atmosphere, 12(8):991. 
        z0 = 0.005  # marine sector 
        zt = 0.0005 # depends on wind direction. Might need additional filtering
    elif ds.attrs['Location'] == "Cabauw": # Beljaars & Bosveld (1997). Cabauw data for the verification of land surface schemes. Journal of Climate, 10(6), 1194–1207.
        z0 = 0.15
        zt = z0*0.1
    elif ds.attrs['Location'] == "Southern Great Plains": # Jacobs & Brutsaert. (1998). Momentum roughness and view-angle dependent heat roughness at a Southern Great Plains test-site. Journal of Hydrology, 211(1), 62-68.
        z0 = 0.15  # quite variable during the year (crops)
        zt = 0.003
    elif ds.attrs['Location'] == "Summit Station": # Miller et al. (2017). Surface energy budget responses to radiative forcing at Summit, Greenland. The Cryosphere. 11(1), 497-516.
        z0 = 0.0004
        zt = 0.0001
    else:
        print("No location specified: assigning default roughness lengths values")
        z0 = 0.1
        zt = z0*0.2
    return 20/z0, 20/zt

def compute_zeta_GL18(ds: xr.Dataset, epsilon: float, epsilon_t: float) -> xr.Dataset:
    """
    Computes the atmospheric stability parameter zeta (z/L) from Eq. 21 of Gryanik & Lüpkes 2018 (GL18).
    Considers z = 20 m. NB: ONLY valid for Ri >= 0! NB2: They used 10 m as reference height
    """
    ln_eps = np.log(epsilon)
    ln_epst = np.log(epsilon_t)

    A = (ln_eps**2) / ln_epst
    B = ln_eps + 11.3
    C = ln_epst + 6.4

    coeff_linear = A 
    coeff_nonlinear = (B**3.82) / (11.5 * (C**1.91))
    bracket = (B**2) / C - A

    # Extract (interpolate) Ri at z = 20 m AGL per timestep
    Ri_20 = interpolate_to_height(ds, "Ri_g", None, 20.0)

    Ri_20_pos = np.maximum(Ri_20, 0.0)  # Restrict to positive Ri (enutral/stable)
    zeta = (coeff_linear * Ri_20_pos) + (
        coeff_nonlinear * bracket * (Ri_20_pos**2.91)
    )
    # Assign to the starting dataset
    ds = ds.assign(zeta_20=zeta)

    return ds

def compute_fm(ds: xr.Dataset, epsilon: float ) -> xr.DataArray:
    """
    Computes the momentum stability correction function f_m (Eq. 22 of Gryanik and Lüpkes)
    Considers z = 20 m. NB: ONLY valid for Ri >= 0! NB2: They used 10 m as reference height
    """
    zeta_20_pos = np.maximum(ds.zeta_20, 0.0) # Restrict to positive Richardson (neutral/stable)
    x = np.cbrt(1.0 + zeta_20_pos)
    ln_eps = np.log(epsilon)

    num_log = (x + 0.67) ** 2
    den_log = (x**2) - (0.67 * x) + 0.45

    term1 = 10.29 - (19.5 * x)
    term2 = 2.18 * np.log(num_log / den_log)
    term3 = 7.54 * np.arctan((1.725 * x) - 0.58)

    bracket = (term1 + term2 + term3) / ln_eps
    fm = (1.0 - bracket) ** (-2.0)

    return fm

def compute_fh(fm: xr.DataArray, ds: xr.Dataset, epsilon_t: float) -> xr.DataArray:
    """
    Computes the momentum stability correction function f_m (Eq. 22 of Gryanik and Lüpkes)
    Considers z = 10 m. NB: ONLY valid for Ri >= 0!
    """
    ln_epst = np.log(epsilon_t)

    zeta_20_pos = np.maximum(ds.zeta_20, 0.0) # Restrict to positive values (neutral/stable)
    term1 = 2.16
    term2 = 2.5 * np.log(1.0 + (3.0 * zeta_20_pos) + (zeta_20_pos**2))
    term3 = 1.12 * np.log((zeta_20_pos + 0.38) / (zeta_20_pos + 2.62))

    bracket = (term1 - term2 + term3) / ln_epst
    fh = np.sqrt(fm) * ((1.0 - bracket) ** (-1.0))

    return fh

#%% Filtering functions

#  See https://share.gemini.google/oecHx8O5us47 or https://gemini.google.com/app/7d5b2894ee35dc71?pageId=none 
#  Testing with the proper data is needed.

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

def filter_stability(ds: xr.Dataset, ds_srf: xr.Dataset, ri_surface_min: float = 0.0, 
                     grad_tol: float = -2e-4, min_valid_fraction: float = 0.8, 
                     smooth_window: int = 3
                     ) -> tuple[xr.Dataset, xr.Dataset]:
    """
    Retain times where:
    1. Near-surface Ri is at least ri_surface_min.
    2. At least required_valid_fraction % of sub-BLH layers satisfy
       smoothened( d(theta_v)/dz ) >= gradient_tolerance [K/m].
    """

    # Get surface quantities
    ri_20m = interpolate_to_height(ds,"Ri_g", None, 20.0)
    mask_surf = (ri_20m >= ri_surface_min)

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

#%% Plotting functions    ((STILL TESTING! - done with help of AI))

# Define consistent aesthetic mappings across the 4 datasets
DATASET_STYLES = [
    {"color": "#1f77b4", "linestyle": "-", "marker": "o"},   # CABAUW
    {"color": "#ff7f0e", "linestyle": "--", "marker": "s"},  # MACE HEAD
    {"color": "#2ca02c", "linestyle": "-.", "marker": "^"},  # SUMMIT STATION
    {"color": "#d62728", "linestyle": ":", "marker": "D"},   # SOUTHER GREAT PLAINS
]

def plot_in_time(
        ds: xr.Dataset, 
        var_name: str
    ):
    """
    Plots vertical profiles of var_name against height AGL (z) across time.
    Uses predefined dataset styles for consistent line and marker aesthetics.
    """
    time_dim = "valid_time" if "valid_time" in ds.dims else "time"
    times = ds[time_dim].values
    N = len(times)

    fig = plt.figure(figsize=(6, 8))
    for i, t in enumerate(times):
        ds_time = ds.sel({time_dim: t})
        style = DATASET_STYLES[i % len(DATASET_STYLES)]
        lbl = str(t)[:16] if (i == 0 or i == N - 1) else None
        plt.plot(
            ds_time[var_name],
            ds_time["z"],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            label=lbl,
        )

    plt.xlabel(var_name)
    plt.ylabel("Height AGL [m]")
    plt.grid(True, linestyle="--", alpha=0.6)
    if N > 1:
        plt.legend()
    plt.title(f"Vertical Profile Evolution: {var_name}")

    return fig

def plot_abl_top_vs_surface_scatter_contour(
    ds_dict: dict[str, xr.Dataset],
    ds_srf_dict: dict[str, xr.Dataset],
    temp_var: str = "t",
):
    """
    Plots a combined scatter and 2D KDE contour plot of Delta T (ABL Top - Surface)

    vs. Wind Speed at the ABL Top across 4 datasets.
    """
    fig, ax = plt.subplots(figsize=(9, 7))

    for idx, (name, ds) in enumerate(ds_dict.items()):
        style = DATASET_STYLES[idx % len(DATASET_STYLES)]
        ds_srf = ds_srf_dict[name]

        # 1. Compute Temperature Difference (ABL Top - Surface)
        delta_T = compute_difference_surface_top_ABL(
            ds, ds_srf, var_name=temp_var
        ).values

        # 2. Extract Wind Speed at BLH for each timestep
        n_times = ds.dims["time"]
        u_blh = np.zeros(n_times)

        for t in range(n_times):
            z_t = ds["z"].isel(time=t).values
            blh_t = ds_srf["blh"].isel(time=t).values
            k_idx = np.abs(z_t - blh_t).argmin()
            u_blh[t] = ds["wind_speed"].isel(time=t, model_level=k_idx).values

        # Filter out NaN/Inf values if present
        valid_mask = np.isfinite(u_blh) & np.isfinite(delta_T)
        x_val = u_blh[valid_mask]
        y_val = delta_T[valid_mask]

        # Scatter plot (low opacity)
        ax.scatter(
            x_val,
            y_val,
            color=style["color"],
            alpha=0.25,
            s=20,
            edgecolors="none",
        )

        # Overlaid 2D Density Contours (KDE)
        sns.kdeplot(
            x=x_val,
            y=y_val,
            ax=ax,
            color=style["color"],
            levels=4,
            linewidths=1.5,
            linestyles=style["linestyle"],
            label=f"{name}",
        )

    ax.set_xlabel("Wind Speed at BLH [m/s]", fontsize=11)
    ax.set_ylabel(
        f"$\Delta {temp_var.upper()}$ (ABL Top - Surface) [K]", fontsize=11
    )
    ax.set_title(
        "ABL Top Wind Speed vs. Temperature Difference across Locations",
        fontsize=12,
    )
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")
    plt.tight_layout()

    return fig, ax

def plot_multi_dataset_pdf(
    ds_dict: dict[str, xr.Dataset],
    var_name: str,
    target_height: float | None = None,
    bins: int = 50,
):
    """
    Plots Probability Density Functions (PDFs) of a specified variable across 4 datasets.
    If the variable has a 'model_level' dimension, it extracts values at the
    level closest to target_height [m AGL]. Otherwise, it flattens the 1D/2D
    field.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for idx, (name, ds) in enumerate(ds_dict.items()):
        style = DATASET_STYLES[idx % len(DATASET_STYLES)]

        if var_name not in ds:
            raise KeyError(f"Variable '{var_name}' not found in dataset '{name}'.")

        data_array = ds[var_name]

        # Check if variable depends on vertical model levels
        if "model_level" in data_array.dims:
            if target_height is None:
                raise ValueError(
                    f"Variable '{var_name}' varies with model level. You must specify 'target_height' [m AGL]."
                )

            # Extract nearest index to target_height per timestep
            k_indices = np.abs(ds["z"] - target_height).argmin(
                dim="model_level"
            )
            extracted_vals = ds[var_name].isel(model_level=k_indices).values.flatten()
            height_str = fr" at $z \approx {target_height}$ m AGL"
        else:
            extracted_vals = data_array.values.flatten()
            height_str = ""

        # Remove NaNs for histogram evaluation
        extracted_vals = extracted_vals[np.isfinite(extracted_vals)]

        # Compute empirical PDF step-histogram
        counts, bin_edges = np.histogram(
            extracted_vals, bins=bins, density=True
        )
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        ax.plot(
            bin_centers,
            counts,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2,
            label=f"{name}",
        )

    ax.set_xlabel(f"{var_name}", fontsize=11)
    ax.set_ylabel("Probability Density", fontsize=11)
    ax.set_title(f"PDF Comparison: {var_name}{height_str}", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")
    plt.tight_layout()

    return fig, ax


def plot_Ri_vs_stability_function(
    ds_dict: dict[str, xr.Dataset], func_type: str = "fm"
):
    """
    Plots Gradient Richardson Number (Ri_20) on the y-axis against the GL18 stability

    functions f_m or f_h on the x-axis for all four datasets.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    target_var = "fm_20" if func_type == "fm" else "fh_20"

    for idx, (name, ds) in enumerate(ds_dict.items()):
        style = DATASET_STYLES[idx % len(DATASET_STYLES)]

        # Extract Ri_20 and stability function arrays
        ri_20 = interpolate_to_height(ds, "Ri_g", None, 20.0).values.flatten()
        f_val = ds[target_var].values.flatten()

        # Filter to valid stable regime (Ri >= 0)
        valid_mask = np.isfinite(ri_20) & np.isfinite(f_val) & (ri_20 >= 0)
        ri_plot = ri_20[valid_mask]
        f_plot = f_val[valid_mask]

        # Sort along Ri axis to guarantee clean line rendering
        sort_idx = np.argsort(ri_plot)

        ax.plot(
            f_plot[sort_idx],
            ri_plot[sort_idx],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2,
            label=f"{name}",
        )

    x_label_str = "$f_m(z/L)$" if func_type == "fm" else "$f_h(z/L)$"
    ax.set_xlabel(f"Stability Correction Factor {x_label_str}", fontsize=11)
    ax.set_ylabel("Gradient Richardson Number $Ri_{20m}$", fontsize=11)
    ax.set_title(
        f"Surface Layer Stability Function ({x_label_str}) vs. $Ri_{{20m}}$",
        fontsize=12,
    )
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")
    plt.tight_layout()

    return fig, ax

def plot_vertical_profile(
    ds_dict: dict[str, xr.Dataset],
    var_name: str,
    time: str | np.datetime64 | None = None,
    time_range: tuple[str, str] | slice | None = None,
    cmap_name: str = "viridis",
):
    """
    Plots vertical profiles (variable vs. height) across datasets. Either at a specific
    time or for a time range (using a consistent colormap)
    """
    if time is not None and time_range is not None:
        raise ValueError(
            "Specify either 'time' or 'time_range', not both."
        )

    fig, ax = plt.subplots(figsize=(7, 8))

    for idx, (name, ds) in enumerate(ds_dict.items()):
        style = DATASET_STYLES[idx % len(DATASET_STYLES)]

        if var_name not in ds or "z" not in ds:
            raise KeyError(
                f"Variable '{var_name}' or height 'z' missing in dataset '{name}'."
            )

        # Case 1: Single timestamp plotting
        if time is not None:
            ds_sel = ds.sel(time=time, method="nearest")

            # Convert to 1D NumPy arrays to prevent coordinate indexing conflicts
            var_vals = ds_sel[var_name].values.flatten()
            z_vals = ds_sel["z"].values.flatten()

            # Filter valid numerical entries
            valid_mask = np.isfinite(var_vals) & np.isfinite(z_vals)
            sort_idx = np.argsort(z_vals[valid_mask])

            ax.plot(
                var_vals[valid_mask][sort_idx],
                z_vals[valid_mask][sort_idx],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2,
                label=f"{name}",
            )

        # Case 2: Time range (Colormap per Dataset / Time Increment)
        elif time_range is not None:
            if isinstance(time_range, tuple):
                t_slice = slice(time_range[0], time_range[1])
            else:
                t_slice = time_range

            ds_sel = ds.sel(time=t_slice)
            n_times = ds_sel.sizes["time"]

            if n_times == 0:
                raise ValueError(
                    f"No timesteps found in range {time_range} for dataset '{name}'."
                )

            # Generate colormap norm across selected timesteps
            cmap = plt.get_cmap(cmap_name)
            colors = [cmap(i) for i in np.linspace(0.2, 1.0, n_times)]

            for t_idx in range(n_times):
                ds_step = ds_sel.isel(time=t_idx)

                var_vals = ds_step[var_name].values.flatten()
                z_vals = ds_step["z"].values.flatten()

                valid_mask = np.isfinite(var_vals) & np.isfinite(z_vals)
                sort_idx = np.argsort(z_vals[valid_mask])

                lbl = (
                    f"{name} ({n_times} steps)"
                    if t_idx == 0 and len(ds_dict) > 1
                    else None
                )

                ax.plot(
                    var_vals[valid_mask][sort_idx],
                    z_vals[valid_mask][sort_idx],
                    color=colors[t_idx],
                    linestyle=style["linestyle"],
                    linewidth=1.5,
                    alpha=0.7,
                    label=lbl,
                )

        # Case 3: Time-avgd mean profile (Fallback)
        else:
            var_mean = ds[var_name].mean(dim="time").values.flatten()
            z_mean = ds["z"].mean(dim="time").values.flatten()

            valid_mask = np.isfinite(var_mean) & np.isfinite(z_mean)
            sort_idx = np.argsort(z_mean[valid_mask])

            ax.plot(
                var_mean[valid_mask][sort_idx],
                z_mean[valid_mask][sort_idx],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2,
                label=f"{name} (Time Mean)",
            )

    # Adding a colorbar for the time range case
    if time_range is not None:
        norm = mcolors.Normalize(vmin=0, vmax=n_times - 1)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label("Timestep Index within Range", fontsize=10)

    title_suffix = (
        f" at t = {time}"
        if time
        else (f" (Range: {time_range})" if time_range else " (Time Mean)")
    )
    ax.set_xlabel(f"{var_name}", fontsize=11)
    ax.set_ylabel("Height AGL $z$ [m]", fontsize=11)
    ax.set_title(f"Vertical Profile: {var_name}{title_suffix}", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")
    plt.tight_layout()

    return fig, ax

#%%
################################################
##################  MAIN  ######################
################################################

#%% Configuration and file paths
data_dir = "/Users/lodo0477/Documents/PhD/Research/Entrainment_with_Palli/ERA5_data"

# Define location metadata maps
site_configs = {
    "Mace Head": {
        "ml_grib": os.path.join(data_dir, "MaceHead_lvls.grib"),
        "srf_grib": os.path.join(data_dir, "MaceHead_surface.grib"),
        "wind_sector": (180.0, 360.0),  # Marine sector 
    },
    "Cabauw": {
        "ml_grib": os.path.join(data_dir, "Cabauw_lvls.grib"),
        "srf_grib": os.path.join(data_dir, "Cabauw_surface.grib"),
        "wind_sector": (0.0, 360.0)
    },
    "Summit Station": {
        "ml_grib": os.path.join(data_dir, "SummitStation_lvls.grib"),
        "srf_grib": os.path.join(data_dir, "SummitStation_surface.grib"),
        "wind_sector": (0.0, 360.0)
    },
    "Southern Great Plains": {
        "ml_grib": os.path.join(data_dir, "ARMGreatPlains_lvls.grib"),
        "srf_grib": os.path.join(data_dir, "ARMGreatPlains_surface.grib"),
        "wind_sector": (0.0, 360.0)
    },
}

ds_ml_dict = {}
ds_srf_dict = {}

#%% Main pipeline
for loc, cfg in site_configs.items():
    print(f"\n--- Processing Location: {loc} ---")

    # File prep and spatial averaging
    ds_ml, ds_srf = prepare_dataset(cfg["ml_grib"], cfg["srf_grib"], location=loc)

    # Cloud filtering
    ds_ml_f0, ds_srf_f0 = ds_ml.copy(), ds_srf.copy()
    ds_ml_f1, ds_srf_f1 = filter_clouds(ds_ml_f0, ds_srf_f0, lcc_thresh=0.1, window_hours=3)
    _ = print_filter_output(ds_ml_f0, ds_ml_f1, "Low cloud cover filtering")

    # Stability filtering
    ds_ml_f1 = compute_grad_Ri_z(ds_ml_f1)
    ds_ml_f1 = compute_BLH_from_Ri_b(ds_ml_f1, Ri_c=0.25)
    ds_ml_f2, ds_srf_f2 = filter_stability(ds_ml_f1, ds_srf_f1, tol=1e-5)
    _ = print_filter_output(ds_ml_f1, ds_ml_f2, "Stability filtering")

    # Wind direction filtering
    ds_ml_f2 = compute_wind_dir(ds_ml_f2)
    if "wind_sector" in cfg:
        dir_min, dir_max = cfg["wind_sector"]
        ds_ml_f3, ds_srf_f3 = filter_wind_dir(ds_ml_f2, dir_min, dir_max)
        _ = print_filter_output(ds_ml_f2, ds_ml_f3, "Wind direction sector filtering")
        ds_ml_filtered, ds_srf_filtered = ds_ml_f3, ds_srf_f3
    else:
        ds_ml_filtered, ds_srf_filtered = ds_ml_f2, ds_srf_f2
    _ = print_filter_output(ds_ml_f0, ds_ml_filtered, "Total filtering")


    # Stability functions computation
    eps, eps_t = compute_epsilon(ds_ml_filtered, location=loc)
    ds_ml_filtered = compute_zeta_GL18(ds_ml_filtered, epsilon=eps, epsilon_t=eps_t)
    fm_20 = compute_fm(ds_ml_filtered, epsilon=eps)
    fh_20 = compute_fh(fm_20, ds_ml_filtered, epsilon_t=eps_t)
    ds_ml_filtered = ds_ml_filtered.assign(fm_20=fm_20, fh_20=fh_20)

    # Store for multi-site comparisons
    ds_ml_dict[loc] = ds_ml_filtered
    ds_srf_dict[loc] = ds_srf_filtered

# %% Plotting & Analysis
# Example 1 (time variable): PDF comparison of BLH across locations and ocmpare to diagnosed one
plot_multi_dataset_pdf(ds_ml_dict, var_name="BLH", bins=40) 
plot_multi_dataset_pdf(ds_srf_dict, var_name="blh", bins=40)    

# Example 2 (time-height variable): PDF comparison of Theta_v at 100m AGL
plot_multi_dataset_pdf(ds_ml_dict, var_name="Ri_g", target_height=100.0, bins=50)

# Example 3: Scatter/KDE of Delta T vs Wind speed at BLH
plot_abl_top_vs_surface_scatter_contour(ds_ml_dict, ds_srf_dict, temp_var="t")

# Example 4: Stability correction function curves
plot_Ri_vs_stability_function(ds_ml_dict, func_type="fm")

# Example 5: Single timestamp vertical profile
plot_vertical_profile(ds_ml_dict, "theta_v", time="2020-07-15T12:00:00")

# Example 6: Time-Range variable profile sequence 
plot_vertical_profile(ds_ml_dict, "Ri_g", time_range=("2020-07-15T06:00:00", "2020-07-25T12:00:00"))


#%%