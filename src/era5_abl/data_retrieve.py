import cdsapi
from .config import (
    ERA5_FORMAT,
    ERA5_GRID,
    ERA5_MODEL_LEVEL_PARAMS,
    ERA5_MODEL_LEVELS,
    ERA5_SURFACE_VARIABLES,
    ERA5_DELTA_TIME
)

# CDS API documentation: https://cds.climate.copernicus.eu/how-to-api


def retrieve_surface_data(
    area: str,
    dates: str,
    output_path: str,
    variables: list[str] = ERA5_SURFACE_VARIABLES,
    dt: str = ERA5_DELTA_TIME,
    format: str = ERA5_FORMAT,
) -> None:
    """Retrieve ERA5 single-level reanalysis data through the CDS API."""

    client = cdsapi.Client()

    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": ["reanalysis"],
            "variable": variables,
            "date": dates,
            "time": dt,
            "area": area,
            "format": format,
        },
        output_path,
    )

def retrieve_model_level_data(
    area: str,
    dates: str,
    output_path: str,
    params: str = ERA5_MODEL_LEVEL_PARAMS,
    dt: str = ERA5_DELTA_TIME,
    mls: str = ERA5_MODEL_LEVELS,
    res: str = ERA5_GRID,
    format: str = ERA5_FORMAT,
) -> None:
    """Retrieve ERA5 reanalysis data on model levels through the CDS API."""

    client = cdsapi.Client()

    client.retrieve(
        "reanalysis-era5-complete",
        {
            "date": dates,
            "levelist": mls,
            "levtype": "ml",
            "param": params,
            "stream": "oper",
            "time": dt,
            "type": "an",
            "area": area,
            "grid": res,
            "format": format,
        },
        output_path,
    )
