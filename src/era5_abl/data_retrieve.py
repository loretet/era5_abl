import cdsapi, os
import calendar, json, math, time
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

def build_model_level_request(
    area: str,
    dates: str,
    params: str = ERA5_MODEL_LEVEL_PARAMS,
    dt: str = ERA5_DELTA_TIME,
    mls: str = ERA5_MODEL_LEVELS,
    res: str = ERA5_GRID,
    format: str = ERA5_FORMAT,
) -> dict:
    """The MARS request for ERA5 model level data (defined here only, so that all the retrieval modes send the same one)."""

    return {
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
    }

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
        build_model_level_request(area, dates, params, dt, mls, res, format),
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
    output_dir/raw_lvls_data/areas/Europe/Europe_lvls_20200101-20200115.grib
    (in their own "areas" folder: an area and a site can have the same name, e.g. ARMGreatPlains)
    """
    area_dir = Path(output_dir) / "raw_lvls_data" / "areas" / area_name
    return [
        area_dir / f"{area_name}_lvls_{piece.replace('-', '').replace('/', '-')}.grib"
        for piece in split_dates(dates, max_days)
    ]

def model_level_requests(
    dates: str,
    site_names: list[str] | None = None,
    output_dir: str | Path = ".",
    max_days: int = ERA5_MAX_DAYS_PER_REQUEST,
    params: str = ERA5_MODEL_LEVEL_PARAMS,
    dt: str = ERA5_DELTA_TIME,
    mls: str = ERA5_MODEL_LEVELS,
    res: str = ERA5_GRID,
    format: str = ERA5_FORMAT,
) -> list[dict]:
    """
    One entry {"label", "area_name", "piece", "request", "path"} per (retrieval area, piece of dates), for the
    areas that contain the selected sites. Ordered by piece first, so that requests for the same month
    (= the same tape at ECMWF) come one after the other.
    """
    if site_names is None:
        selected_sites = SITE_CONFIGS
    else:
        selected_sites = {name: SITE_CONFIGS[name] for name in site_names}
    area_names = sorted({get_retrieval_area(site) for site in selected_sites.values()})
    paths = {name: model_level_paths(name, dates, output_dir, max_days) for name in area_names}

    requests = []
    for i, piece in enumerate(split_dates(dates, max_days)):
        for area_name in area_names:
            requests.append({
                "label": f"model level data for {area_name}, {piece}",
                "area_name": area_name,
                "piece": piece,
                "request": build_model_level_request(ERA5_RETRIEVAL_AREAS[area_name], piece, params, dt, mls, res, format),
                "path": paths[area_name][i],
            })
    return requests

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
    However, they still have to be present in CONFIG.
    - Surface data: one request per site, for the whole period.
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
        ml_requests = model_level_requests(dates, site_names, output_dir, max_days, ml_params, dt, mls, res, format)
        todo = [job for job in ml_requests if not job["path"].exists()]
        for job in todo:
            requests.append((
                job["label"],
                retrieve_model_level_data,
                dict(area=job["request"]["area"], dates=job["piece"], params=ml_params,
                     dt=dt, mls=mls, res=res, format=format),
                job["path"],
            ))
        print(f"Model level data: {len(ml_requests)} requests of at most {max_days} days, "
              f"{len(ml_requests) - len(todo)} already there, {len(todo)} to retrieve.")

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


#  Requests that keep running on the CDS while this computer is off  ---------------------------------------
#  submit_model_level_requests() only sends the requests (minutes), fetch_model_level_requests() downloads the
#  results whenever this computer is on. The ids are stored on disk, so both can be stopped and started again.

def request_state_file(output_dir: str | Path) -> Path:
    """File where the state of the requests sent to the CDS is stored, so that it survives a restart."""
    return Path(output_dir) / "raw_lvls_data" / "cds_requests.json"

def read_request_state(output_dir: str | Path) -> dict:
    """
    {"pending": {file name: request id}, "done": [file names]}: the requests that are queued or running on the
    CDS, and the ones that are already downloaded (they stay in "done" also when the file is deleted after the
    sites are extracted from it, so that they are not requested again).
    """
    state_file = request_state_file(output_dir)
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    return {"pending": state.get("pending", {}), "done": state.get("done", [])}

def write_request_state(state: dict, output_dir: str | Path) -> None:
    state_file = request_state_file(output_dir)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(state, indent=1))

def is_finished(job: dict, state: dict) -> bool:
    """True if the file of this request is on disk, or was downloaded and then deleted after the extraction."""
    return job["path"].exists() or job["path"].name in state["done"]

def cds_client(fast_fail: bool = False):
    """
    The client underneath cdsapi, which can submit a request without waiting for the result.
    The client contacts the CDS as soon as it is created, and cdsapi retries a failing connection up to 500
    times by default, pausing up to 2 minutes each time: a scheduled run that starts while this computer is
    offline would hang for most of a day (and launchd would not start the next one until it ends).
    With fast_fail (used by the --once runs) it gives up after about a minute instead, and the next run tries
    again.
    """
    options = {"retry_max": 3, "sleep_max": 20} if fast_fail else {}
    return cdsapi.Client(**options).client

def submit_model_level_requests(
    dates: str,
    site_names: list[str] | None = None,
    output_dir: str | Path = ".",
    max_days: int = ERA5_MAX_DAYS_PER_REQUEST,
    fast_fail: bool = False,
    **request_kwargs,
) -> dict:
    """
    Sends the model level requests to the CDS WITHOUT downloading anything, and stores their ids in
    raw_lvls_data/cds_requests.json. The CDS then works on them (hours to days) whether this computer is on or
    not, and fetch_model_level_requests() collects the results later.
    Requests that are already submitted or already downloaded are not sent again, so the function can be called
    again at any time (e.g. if the CDS refuses new requests because too many of them are queued).
    NOTE: the CDS does not keep the results forever, so do not wait too long before fetching them.
    fast_fail: give up quickly if the CDS cannot be reached (for scheduled runs, see cds_client).
    """
    state = read_request_state(output_dir)
    requests = model_level_requests(dates, site_names, output_dir, max_days, **request_kwargs)

    client = cds_client(fast_fail)
    n_new = 0
    for job in requests:
        name = job["path"].name
        if is_finished(job, state) or name in state["pending"]:
            continue
        try:
            remote = client.submit("reanalysis-era5-complete", job["request"])
        except Exception as error:
            print(f"Could not submit {job['label']}: {error}")
            print("--> Stopping here. The requests sent so far are stored: call this function again later "
                  "to send the remaining ones.")
            break
        state["pending"][name] = remote.request_id
        write_request_state(state, output_dir)    # written every time: nothing is lost if the script is killed
        n_new += 1
        print(f"Submitted {job['label']} ({remote.request_id})", flush=True)

    n_done = sum(1 for job in requests if is_finished(job, state))
    print(f"\n{n_new} new requests sent, {len(state['pending'])} queued or running on the CDS, {n_done} already "
          f"downloaded, {len(requests) - n_done - len(state['pending'])} still to send.")
    return state

def fetch_model_level_requests(
    dates: str,
    site_names: list[str] | None = None,
    output_dir: str | Path = ".",
    max_days: int = ERA5_MAX_DAYS_PER_REQUEST,
    on_file=None,
    wait_minutes: int = 5,
    once: bool = False,
    **request_kwargs,
) -> bool:
    """
    Downloads the results of the requests sent by submit_model_level_requests(), one by one, as they become ready,
    and returns True if everything was downloaded.
    If on_file is given, it is called as on_file(path, area_name, piece) right after each download: this is where
    the sites are extracted and the (big) downloaded file can be deleted, so that the disk never fills up.
    It can be interrupted with Ctrl-C and started again later: what is on disk is kept, and the rest is still
    waiting on the CDS.
    Requests that failed, or that the CDS does not know anymore (results are not kept forever), are removed from
    the list, so that submit_model_level_requests() sends them again.

    once=False (the default) keeps checking every wait_minutes until everything is downloaded. This needs the
             computer to stay on for hours or days, so it is meant to be run by hand (with caffeinate).
    once=True  downloads whatever is ready NOW and returns. This is the mode for a scheduled job (launchd,
             cron): it ends in minutes instead of sleeping, so the computer is free to go to sleep again, and
             the next run collects the next batch.
    """
    requests = model_level_requests(dates, site_names, output_dir, max_days, **request_kwargs)
    client = cds_client(fast_fail=once)

    while True:
        state = read_request_state(output_dir)
        n_waiting = 0

        for job in requests:
            name = job["path"].name
            if is_finished(job, state) or name not in state["pending"]:
                continue
            try:
                remote = client.get_remote(state["pending"][name])
                status = remote.status
            except Exception as error:
                print(f"{job['label']}: the CDS does not know this request anymore ({error}): it will be sent again.")
                state["pending"].pop(name)
                write_request_state(state, output_dir)
                continue

            if status == "successful":
                job["path"].parent.mkdir(parents=True, exist_ok=True)
                # Download to a temporary file, renamed only when complete: an interruption never leaves a broken file
                tmp_path = job["path"].with_name(name + ".part")
                print(f"Downloading {job['label']}", flush=True)
                remote.download(str(tmp_path))
                tmp_path.replace(job["path"])
                print(f"Downloaded {job['label']} ({job['path'].stat().st_size / 1e9:.2f} GB)", flush=True)
                try:
                    remote.delete()    # the result is on disk now: free it on the CDS
                except Exception:
                    pass
                if on_file is not None:
                    on_file(job["path"], job["area_name"], job["piece"])
                # Marked as done only now: on_file may have deleted the file after extracting the sites
                state["pending"].pop(name)
                state["done"].append(name)
                write_request_state(state, output_dir)
            elif status in ("failed", "rejected", "dismissed"):
                print(f"{job['label']}: {status} on the CDS: it will be sent again.")
                state["pending"].pop(name)
                write_request_state(state, output_dir)
            else:
                n_waiting += 1

        missing = [job for job in requests if not is_finished(job, state)]
        print(f"-- {len(requests) - len(missing)}/{len(requests)} files downloaded, {n_waiting} queued or "
              f"running on the CDS, {len(missing) - n_waiting} still to send", flush=True)
        if not missing:
            print("--- All the model level files were downloaded ---")
            return True
        if n_waiting == 0:
            print("Nothing is running on the CDS: send the missing requests (SUBMIT_REQUESTS) and fetch again.")
            return False
        if once:
            # One pass only: the rest is still being worked on at the CDS and is collected by the next run,
            # so this process ends now instead of keeping the computer awake while it waits.
            print(f"({n_waiting} still at the CDS: they are collected by the next run.)")
            return False
        time.sleep(wait_minutes * 60)

