import cdsapi, os
from .config import (
    ERA5_FORMAT,
    ERA5_GRID,
    ERA5_MODEL_LEVEL_PARAMS,
    ERA5_MODEL_LEVELS,
    ERA5_SURFACE_VARIABLES,
    ERA5_DELTA_TIME,
    SITE_CONFIGS
)
from .config import SiteConfig
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


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

def parallel_retrieval(    
    dates: str,
    output_dir: str | Path,
    max_workers: int = 4,
    surface_variables: list[str] = ERA5_SURFACE_VARIABLES,
    params: str = ERA5_MODEL_LEVEL_PARAMS,
    dt: str = ERA5_DELTA_TIME,
    mls: str = ERA5_MODEL_LEVELS,
    res: str = ERA5_GRID,
    format: str = ERA5_FORMAT,
) -> None:
    """
    Retrieves ERA5 data in parallel for each distinct site. However, within each site,
    the surface and model level data are requested sequentially.
    The function still awaits for the end of the download (via "future.result()") before
    proceeding, to avoid working on files that do not exist yet.
    """

    output_dir = Path(output_dir)

    def retrieve_site(name, site):

        surface_path = output_dir / site.surface_filename
        model_level_path = output_dir / site.model_level_filename

        print(f"Starting retrieval for {name}")

        retrieve_surface_data(
            area=site.area,
            dates=dates,
            output_path=str(surface_path),
            variables=surface_variables,
            dt=dt,
            format=format,
        )

        retrieve_model_level_data(
            area=site.area,
            dates=dates,
            output_path=str(model_level_path),
            params=params,
            dt=dt,
            mls=mls,
            res=res,
            format=format,
        )

        print(f"Finished retrieval for {name}")
        
    print("--- Retrieving ERA5 data in parallel ---")
    print("WARNING: this might take a while.") 
    print("Expect around 2h per year for surface data (nine variables over 0.25x0.25 deg area)")
    print("and 24h per year for model level data (five variables over one 0.25x0.25 deg area).")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                retrieve_site,
                name,
                site,
            )
            for name, site in SITE_CONFIGS.items()
        ]

        for future in futures:
            future.result()
