import xarray as xr
import numpy as np


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
        print(f"WARNING - compute_BLH_from_Ri_b: Threshold Ri_b was not crossed more than")
        print(f"persistence={persistence} times in a row for {fail_count} timesteps. For these, BLH is NaN")

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
