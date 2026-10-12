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
    area: str    # "N/W/S/E" (longitudes from -180 to 180): grid point(s) of the site, extracted from its retrieval area
    wind_sector: tuple[float, float]
    model_level_filename: str
    surface_filename: str


SITE_CONFIGS = {
    "Mace Head": SiteConfig(
        area="53.25/-10.0/53.25/-10.0",
        wind_sector=(180.0, 360.0),
        model_level_filename="MaceHead_lvls.grib",
        surface_filename="MaceHead_surface.grib",
    ),
    "Cabauw": SiteConfig(
        area="52.0/5.0/52.0/5.0",
        wind_sector=(0.0, 360.0),
        model_level_filename="Cabauw_lvls.grib",
        surface_filename="Cabauw_surface.grib",
    ),
    "Summit Station": SiteConfig(
        area="72.5/-38.5/72.5/-38.5",
        wind_sector=(0.0, 360.0),
        model_level_filename="SummitStation_lvls.grib",
        surface_filename="SummitStation_surface.grib",
    ),
    "ARM Southern Great Plains": SiteConfig(
        area="36.5/-97.5/36.5/-97.5",
        wind_sector=(0.0, 360.0),
        model_level_filename="ARMGreatPlains_lvls.grib",
        surface_filename="ARMGreatPlains_surface.grib",    
    ),
    # Dropped (Oct 2026): the CDS allows at most 150 queued requests, and Concordia is too far
    # from the other sites to share a retrieval area. Re-enable here and in ERA5_RETRIEVAL_AREAS.
    #    "Concordia Dome C": SiteConfig(
    #        area="-75.0/123.25/-75.0/123.25",
    #        wind_sector=(0.0, 360.0),
    #        model_level_filename="ConcordiaDomeC_lvls.grib",
    #        surface_filename="ConcordiaDomeC_surface.grib",    
    #    ),
    "ARM Eastern North Atlantic": SiteConfig(
        area="39.0/-28.0/39.0/-28.0",
        wind_sector=(0.0, 360.0),
        model_level_filename="ARMEasternNorthAtlantic_lvls.grib",
        surface_filename="ARMEasternNorthAtlantic_surface.grib",    
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
    "2m_dewpoint_temperature",
    "skin_temperature", 
    "instantaneous_surface_sensible_heat_flux", 
    "friction_velocity", 
    "land_sea_mask",
    "fsr",  # forecast roughness length for momentum
    "flsr"  # log of forecast roughness length for heat
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

# Areas ("N/W/S/E") requested from the CDS for the model level data. A few requests over a big area are more
# efficient than many requests over small ones (CDS support, Oct 2026), so the sites are not requested one by one:
# each site is extracted locally from the area that contains it (see preprocessing.extract_site_data).
ERA5_RETRIEVAL_AREAS = {
    # One single box around all the sites. ERA5 model level fields are archived globally and the "area" is only
    # applied afterwards, in the post-processing, so a small area costs MARS exactly as much tape reading as a
    # big one: asking for several areas over the same dates just makes the robot fetch the same tape again and
    # again, which lowers the priority of the requests (ECMWF support, Oct 2026). One box over all the sites is
    # therefore one tape read per piece instead of three, and the sites are cut out of it locally afterwards
    # (see preprocessing.extract_sites_from_file). ~5 GB per request, well under the 20 GB the CDS asks for.
    "AllSites": "72.5/-97.5/36.5/5.0",    # Summit Station, Mace Head, Cabauw, ARM ENA, ARM Great Plains
    # Previous split, kept in case the single box turns out to be too much to transfer (it is ~10 times more
    # data in total, even though it is fewer requests):
    # "Europe": "53.25/-10.0/52.0/5.0",                 # Mace Head, Cabauw
    # "GreenlandAzores": "72.5/-38.5/39.0/-28.0",       # Summit Station, ARM Eastern North Atlantic
    # "ARMGreatPlains": "36.5/-97.5/36.5/-97.5",        # too far from the others: single grid point
    # "ConcordiaDomeC": "-75.0/123.25/-75.0/123.25",    # dropped: the CDS allows at most 150 queued requests
}
# The CDS rejects model level requests longer than ~15 days ("cost limits exceeded"). 17 days are tested to work.
ERA5_MAX_DAYS_PER_REQUEST = 16
# With more than ~15 requests at the same time the downloads get slower.
ERA5_MAX_PARALLEL_REQUESTS = 15


def get_retrieval_area(site: SiteConfig) -> str:
    """Name of the retrieval area (in ERA5_RETRIEVAL_AREAS) that contains the site."""
    north, west, south, east = (float(x) for x in site.area.split("/"))
    for name, area in ERA5_RETRIEVAL_AREAS.items():
        area_north, area_west, area_south, area_east = (float(x) for x in area.split("/"))
        if area_south <= south and north <= area_north and area_west <= west and east <= area_east:
            return name
    raise ValueError(f"No area in ERA5_RETRIEVAL_AREAS contains the site area {site.area}: add one in config.py.")
