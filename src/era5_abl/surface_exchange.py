import numpy as np
import xarray as xr
from .config import get_site_config
from .operations import interpolate_to_height



def compute_epsilon(ds_srf: xr.Dataset, reference_height: float = 20.0,) -> tuple[float, float]:
    """ 
    Computes ration between z (at ref. height) and z0 or zt from surface data.
    """

    epsilon = reference_height / ds.roughness_length_momentum
    epsilon_t = reference_height / ds.roughness_length_heat

    return epsilon, epsilon_t

def compute_zeta_GL18(ds: xr.Dataset, epsilon: float, epsilon_t: float, reference_height: float) -> xr.Dataset:
    """
    Computes the atmospheric stability parameter zeta (z/L) from Eq. 21 of Gryanik & Lüpkes 2018 (GL18).
    Considers z at reference_height. NB: ONLY valid for Ri_b >= 0! NB2: They used 10 m as reference height
    """
    ln_eps = np.log(epsilon)
    ln_epst = np.log(epsilon_t)

    A = (ln_eps**2) / ln_epst
    B = ln_eps + 11.3
    C = ln_epst + 6.4

    coeff_linear = A 
    coeff_nonlinear = (B**3.82) / (11.5 * (C**1.91))
    bracket = (B**2) / C - A

    # Extract (interpolate) Ri at z = reference_height AGL per timestep
    Ri_ref = interpolate_to_height(ds, "Ri_b_srf", None, reference_height)

    Ri_ref_pos = Ri_ref.where(Ri_ref >= 0.0, np.nan)  # Restrict to positive Ri (enutral/stable)
    zeta = (coeff_linear * Ri_ref_pos) + (
        coeff_nonlinear * bracket * (Ri_ref_pos**2.91)
    )
    # Assign to the starting dataset
    ds = ds.assign(zeta_ref_GL18=zeta)

    return ds

def compute_fm_GL18(ds: xr.Dataset, epsilon: float) -> xr.DataArray:
    """
    Computes the momentum stability correction function f_m (Eq. 22 of Gryanik and Lüpkes)
    Considers zeta at reference_height. NB: ONLY valid for z/L >= 0! NB2: They used 10 m as reference height
    """
    zeta_ref_pos = np.maximum(ds.zeta_ref_GL18, 0.0) # Restrict to positive Richardson (neutral/stable)
    x = np.cbrt(1.0 + zeta_ref_pos)
    ln_eps = np.log(epsilon)

    num_log = (x + 0.67) ** 2
    den_log = (x**2) - (0.67 * x) + 0.45

    term1 = 10.29 - (19.5 * x)
    term2 = 2.18 * np.log(num_log / den_log)
    term3 = 7.54 * np.arctan((1.725 * x) - 0.58)

    bracket = (term1 + term2 + term3) / ln_eps
    fm = (1.0 - bracket) ** (-2.0)

    return fm

def compute_fh_GL18(fm: xr.DataArray, ds: xr.Dataset, epsilon_t: float) -> xr.DataArray:
    """
    Computes the heat stability correction function f_m (Eq. 22 of Gryanik and Lüpkes)
    Considers zeta at reference_height. NB: ONLY valid for z/L >= 0!
    """
    ln_epst = np.log(epsilon_t)

    zeta_ref_pos = np.maximum(ds.zeta_ref_GL18, 0.0) # Restrict to positive values (neutral/stable)
    term1 = 2.16
    term2 = 2.5 * np.log(1.0 + (3.0 * zeta_ref_pos) + (zeta_ref_pos**2))
    term3 = 1.12 * np.log((zeta_ref_pos + 0.38) / (zeta_ref_pos + 2.62))

    bracket = (term1 - term2 + term3) / ln_epst
    fh = np.sqrt(fm) * ((1.0 - bracket) ** (-1.0))

    return fh

def compute_fm_IFS(ds: xr.Dataset, reference_height: float) -> xr.DataArray:
    """
    Computes the original IFS momentum stability correction function f_m (Louis, Tiedtke, Geleyn)
    Considers Ri at the reference_height. NB: ONLY valid for Ri >= 0! 
    """
    b, d = 5, 1
    Ri_ref = interpolate_to_height(ds, "Ri_b_srf", None, reference_height)
    return 1 / ( 1 + 2*b*Ri_ref * (1 + d*Ri_ref)**(-1/2) )
    

def compute_fh_IFS(ds: xr.Dataset, reference_height: float) -> xr.DataArray:
    """
    Computes the original IFS heat stability correction function f_m (Louis, Tiedtke, Geleyn)
    Considers Ri at the reference_height. NB: ONLY valid for Ri >= 0!
    """
    b, d = 5, 1
    Ri_ref = interpolate_to_height(ds, "Ri_b_srf", None, reference_height)
    return 1 / ( 1 + 2*b*Ri_ref * (1 + d*Ri_ref)**(+1/2) )


# def compute_zeta_IFS(ds: xr.Dataset, epsilon: float, epsilon_t: float, reference_height: float, tol: float = 1e-6) -> xr.Dataset:

#     def compute_psi(zeta, phi_type):
#         a, b, c, d = 1, 2/3, 5, 0.35
#         term1 = -b*(zeta - c/d) * np.exp(-d*zeta) 
#         term3 = -b*c/d 
#         if phi_type == "m":
#             return term1 - a*zeta + term3
#         elif phi_type == "h":
#             return term1 - (1+2/3*a*zeta)**1.5 + term3 + 1
#         else:
#             raise ValueError

#     def compute_g_IFS(zeta, epsilon, epsilon_t, psi_m, psi_h):
#         return zeta * (np.log(epsilon_t) - psi_h) / (np.log(epsilon) - psi_m)**2
    
#     # Extract (interpolate) Ri at z = reference_height AGL per timestep
#     Ri_ref = interpolate_to_height(ds, "Ri_b_srf", None, reference_height)
#     Ri_ref_pos = np.maximum(Ri_ref, 0.0)  # Restrict to positive Ri (enutral/stable)

#     zeta = Ri_ref_pos * (np.log(epsilon)**2 / np.log(epsilon_t))

#     # Iteratively solve for z
#     while res > tol:
#         psi_m, psi_h = compute_psi(zeta,"m"), compute_psi(zeta,"h")
#         g_IFS = compute_g_IFS(zeta, epsilon, epsilon_t, psi_m, psi_h)

#         res = g_IFS - Ri_ref_pos
#         dgdz = (g_IFS + g_IFS_old) / 2*(zeta-zeta_old)

#         zeta = zeta - res / dgdz
#         zeta_pos = np.maximum(ds.zeta, 0.0)       wrong! Should filter out the wrong values

#         g_IFS_old = g_IFS
#         zeta_old = zeta
#     return zeta_pos

