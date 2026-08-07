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
