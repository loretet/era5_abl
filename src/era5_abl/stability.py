import numpy as np
import xarray as xr
from .operations import interpolate_to_height
from .preprocessing import compute_thetav
from .config import (
    GRAVITY
)

def compute_grad_Ri(ds: xr.Dataset) -> xr.Dataset:
    """
    Computes Gradient Richardson Number profiles over non-uniform physical height
    z(t, k) using central differences.
    """

    g = GRAVITY

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

    g = GRAVITY

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
    elif var_name == "theta_v":
        surf_var = compute_thetav(ds_srf)
    elif var_name == "wind_speed":
        surf_var = interpolate_to_height(ds_ml, var_name, None, 2.0)# wind speed at "surface" (2m)
    else:
        surf_var = ds_srf[var_name]

    return toa_var - surf_var 