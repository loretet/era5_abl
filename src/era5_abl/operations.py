import xarray as xr
import numpy as np


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

def compare_diagnosed_and_era5_blh(ds_ml: xr.Dataset, ds_srf: xr.Dataset) -> None:
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
