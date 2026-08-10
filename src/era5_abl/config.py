from dataclasses import dataclass

# ---------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------

GRAVITY = 9.80665                # m s^-2
DRY_AIR_GAS_CONSTANT = 287.058   # J kg^-1 K^-1
REFERENCE_PRESSURE = 100000.0    # Pa
POISSON_EXPONENT = 0.2854        # R_d / c_p approximation

# ---------------------------------------------------------------------
# Site metadata
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class SiteConfig:
    area: str
    wind_sector: tuple[float, float]
    roughness_length_momentum: float
    roughness_length_heat: float
    model_level_filename: str
    surface_filename: str


SITE_CONFIGS = {
    "Mace Head": SiteConfig(
        area="53.25/9.5/53.00/9.75",
        wind_sector=(180.0, 360.0),
        roughness_length_momentum=0.005,
        roughness_length_heat=0.0005,   # Roughly based on He et al. (2021). Reduced Sea-Surface Roughness Length at a Coastal Site. Atmosphere, 12(8):991. 
        model_level_filename="MaceHead_lvls.grib",
        surface_filename="MaceHead_surface.grib",
    ),
    "Cabauw": SiteConfig(
        area="52/4.75/51.75/5",
        wind_sector=(0.0, 360.0),
        roughness_length_momentum=0.15,
        roughness_length_heat=0.015,    # Beljaars & Bosveld (1997). Cabauw data for the verification of land surface schemes. Journal of Climate, 10(6), 1194–1207.
        model_level_filename="Cabauw_lvls.grib",
        surface_filename="Cabauw_surface.grib",
    ),
    "Summit Station": SiteConfig(
        area="72.75/38.25/72.5/38.5",
        wind_sector=(0.0, 360.0),
        roughness_length_momentum=0.0004,
        roughness_length_heat=0.0001,   # Miller et al. (2017). Surface energy budget responses to radiative forcing at Summit, Greenland. The Cryosphere. 11(1), 497-516.
        model_level_filename="SummitStation_lvls.grib",
        surface_filename="SummitStation_surface.grib",
    ),
    "Southern Great Plains": SiteConfig(
        area="36.75/97.25/36.5/97.5",
        wind_sector=(0.0, 360.0),
        roughness_length_momentum=0.15, # quite variable due to crops
        roughness_length_heat=0.003,    # Jacobs & Brutsaert. (1998). Momentum roughness and view-angle dependent heat roughness at a Southern Great Plains test-site. Journal of Hydrology, 211(1), 62-68.
        model_level_filename="ARMGreatPlains_lvls.grib",
        surface_filename="ARMGreatPlains_surface.grib",    
    ),
}


def get_site_config(location: str) -> SiteConfig:
    return SITE_CONFIGS[location]

# ---------------------------------------------------------------------
# ERA5 retrieval defaults
# ---------------------------------------------------------------------

ERA5_SURFACE_VARIABLES = [
    "sea_surface_temperature",
    "2m_temperature",
    "surface_pressure",
    "cloud_base_height",
    "high_cloud_cover",
    "low_cloud_cover",
    "medium_cloud_cover",
    "boundary_layer_height",
    "geopotential",
    "2m_dewpoint_temperature"
]

ERA5_MODEL_LEVELS = "110/to/137"

# Param IDs (see more here: https://codes.ecmwf.int/grib/param-db?encoding=grib2&ordering=id&limit=20&page=1):
# 130 = temperature (t)
# 131 = u-component of wind (u)
# 132 = v-component of wind (v)
# 133 = specific humidity (q)
# 152 = logarithm of surface pressure (lnsp)
ERA5_MODEL_LEVEL_PARAMS = "130/131/132/133/152"
ERA5_DELTA_TIME = "00/to/23/by/1"
ERA5_GRID = "0.25/0.25"
ERA5_FORMAT = "grib"
