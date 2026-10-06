import cdsapi, os
import calendar, math
from .config import (
    ERA5_FORMAT,
    ERA5_GRID,
    ERA5_MODEL_LEVEL_PARAMS,
    ERA5_MODEL_LEVELS,
    ERA5_SURFACE_VARIABLES,
    ERA5_DELTA_TIME,
    ERA5_RETRIEVAL_AREAS,
    ERA5_MAX_DAYS_PER_REQUEST,
    ERA5_MAX_PARALLEL_REQUESTS,
    SITE_CONFIGS,
    get_retrieval_area,
)
from .config import SiteConfig
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
from threading import Event


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
            "date": dates.replace("/", "/to/") if "/to/" not in dates else dates,
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

def split_dates(dates: str, max_days: int = ERA5_MAX_DAYS_PER_REQUEST) -> list[str]:
    """
    Splits a date interval "YYYY-MM-DD/YYYY-MM-DD" in consecutive pieces of at most max_days days, e.g.
    "2020-01-01/2020-01-15", "2020-01-16/2020-01-31", "2020-02-01/2020-02-14", "2020-02-15/2020-02-29", ...
    Every month is split on its own, in pieces of (almost) the same length. In this way months with a different
    number of days never leave a piece of one or two days, and a piece never crosses the end of a month
    (ECMWF stores every month on a different tape, so this is also the fastest way to get the data).
    """
    start, end = (date.fromisoformat(d) for d in dates.split("/"))

    pieces = []
    first = start
    while first <= end:
        # Days of this month that are within the interval (the first and the last month can be incomplete)
        last = min(end, date(first.year, first.month, calendar.monthrange(first.year, first.month)[1]))
        n_days = (last - first).days + 1
        n_pieces = math.ceil(n_days / max_days)
        for i in range(n_pieces):
            piece_first = first + timedelta(days=i * n_days // n_pieces)
            piece_last = first + timedelta(days=(i + 1) * n_days // n_pieces - 1)
            pieces.append(f"{piece_first}/{piece_last}")
        first = last + timedelta(days=1)

    return pieces

def model_level_paths(
    area_name: str,
    dates: str,
    output_dir: str | Path,
    max_days: int = ERA5_MAX_DAYS_PER_REQUEST,
) -> list[Path]:
    """
    Paths of the model level files of one retrieval area (one file per piece of split_dates), e.g.
    output_dir/raw_lvls_data/NorthernSites/NorthernSites_lvls_20200101-20200115.grib
    """
    area_dir = Path(output_dir) / "raw_lvls_data" / area_name
    return [
        area_dir / f"{area_name}_lvls_{piece.replace('-', '').replace('/', '-')}.grib"
        for piece in split_dates(dates, max_days)
    ]

def parallel_retrieval(    
    dates: str,
    site_names: list[str] | None = None,
    output_dir: str | Path = ".",
    max_workers: int = ERA5_MAX_PARALLEL_REQUESTS,
    max_days: int = ERA5_MAX_DAYS_PER_REQUEST,
    surface_variables: list[str] = ERA5_SURFACE_VARIABLES,
    ml_params: str = ERA5_MODEL_LEVEL_PARAMS,
    dt: str = ERA5_DELTA_TIME,
    mls: str = ERA5_MODEL_LEVELS,
    res: str = ERA5_GRID,
    format: str = ERA5_FORMAT,
    retrieve_srf_data: bool = True,
    retrieve_ml_data: bool = True
) -> None:
    """
    Retrieves ERA5 data in parallel (at most max_workers requests at the same time).
    It either considers all the locations in CONFIG, or only those listed in "site_names".
    However, they stillhave to be present in CONFIG.
    - Surface data: one request per site, for the whole period
    - Model level data: the CDS does not accept long requests anymore, so the period is split in pieces of
      at most max_days days (see split_dates). Moreover, the sites are not requested one by one: one request
      per piece is sent for every retrieval area (ERA5_RETRIEVAL_AREAS in CONFIG) that contains at least one
      of the sites. The sites are extracted from these files afterwards (preprocessing.extract_site_data).
      Files that are already there are skipped, so after an interruption or a failed request it is enough to
      call the function again.
    The function still awaits for the end of the download (via "future.result()") before
    proceeding, to avoid working on files that do not exist yet.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if site_names is None:
        # Retrieve all configured sites
        selected_sites = SITE_CONFIGS
    else:
        # Retrieve only selected sites
        selected_sites = {name: SITE_CONFIGS[name] for name in site_names}

    # List of requests: (label, retrieval function, arguments of the function, output path)
    requests = []

    if retrieve_srf_data:
        for name, site in selected_sites.items():
            requests.append((
                f"surface data for {name}",
                retrieve_surface_data,
                dict(area=site.area, dates=dates, variables=surface_variables, dt=dt, format=format),
                output_dir / site.surface_filename,
            ))

    if retrieve_ml_data:
        # Retrieval areas needed for the selected sites
        area_names = sorted({get_retrieval_area(site) for site in selected_sites.values()})
        pieces = split_dates(dates, max_days)
        paths = {name: model_level_paths(name, dates, output_dir, max_days) for name in area_names}
        n_skipped = 0
        # Loop on the pieces first, so that requests for the same month (same tape) are sent together
        for i, piece in enumerate(pieces):
            for area_name in area_names:
                path = paths[area_name][i]
                if path.exists():
                    n_skipped += 1
                    continue
                requests.append((
                    f"model level data for {area_name}, {piece}",
                    retrieve_model_level_data,
                    dict(area=ERA5_RETRIEVAL_AREAS[area_name], dates=piece, params=ml_params,
                         dt=dt, mls=mls, res=res, format=format),
                    path,
                ))
        print(f"Model level data: {len(pieces)} pieces of at most {max_days} days for each of the areas {area_names}.")
        print(f"{n_skipped} files are already there, {len(pieces) * len(area_names) - n_skipped} to retrieve.")

    failed = []
    too_large = Event()

    def retrieve(label, function, kwargs, path):

        if too_large.is_set():
            # A request was rejected for its size: the others would be rejected too
            return

        print(f"Starting retrieval of {label}", flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Download to a temporary file, renamed only when complete: an interrupted download never leaves a broken file
        tmp_path = path.with_name(path.name + ".part")
        try:
            function(output_path=str(tmp_path), **kwargs)
            tmp_path.replace(path)
            print(f"Finished retrieval of {label}", flush=True)
        except Exception as error:
            tmp_path.unlink(missing_ok=True)
            failed.append(label)
            print(f"FAILED retrieval of {label}: {error}", flush=True)
            if "cost limits exceeded" in str(error) or "too large" in str(error):
                too_large.set()

    if requests:
        print("--- Retrieving ERA5 data in parallel ---")
        print("WARNING: this might take a while (the requests wait in the CDS queue, hours to days in total).", flush=True)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:

            futures = []

            for request in requests:
                future = executor.submit(retrieve, *request)
                futures.append(future)

            # Wait until all retrievals finish
            for future in futures:
                future.result()

        if too_large.is_set():
            print("The CDS rejected a request because it is too large: reduce max_days "
                  "(ERA5_MAX_DAYS_PER_REQUEST in CONFIG) and retrieve again.")
        elif failed:
            print(f"{len(failed)} requests failed (call the function again to retry them):")
            for label in failed:
                print(f"  {label}")
        else:
            print("--- All the ERA5 data were retrieved ---")
