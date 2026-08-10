import numpy as np
import xarray as xr
from .config import get_site_config
from .operations import interpolate_to_height

def compute_epsilon(location: str, reference_height: float = 20.0,) -> tuple[float, float]:
    """ 
    Computes ration between z (20 m) and z0 or zt from surface data.
    """

    site = get_site_config(location)

    epsilon = reference_height / site.roughness_length_momentum
    epsilon_t = reference_height / site.roughness_length_heat

    return epsilon, epsilon_t

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