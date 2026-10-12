# 2026, L. Donati
# Checks that the merged ERA5 model level files (one per location, in ERA5_data/raw_lvls_data) cover the whole
# period hour by hour, with no missing day. Run it once the downloads are finished:
#
#   python validate_era5.py
#
# It only reads the time axis of each file, so it is quick and it changes nothing.

from pathlib import Path
import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import xarray as xr
from era5_abl.config import SITE_CONFIGS

# Directory with data (same as download_era5.py):
# (override on another machine, e.g. a supercomputer, with:  export ERA5_DATA_DIR=/scratch/.../ERA5_data)
DATA_DIR = Path(os.environ.get(
    "ERA5_DATA_DIR",
    "/Users/lodo0477/Documents/PhD/Research/Entrainment_with_Palli/ERA5_data/",
))
# Period that the files are expected to cover (same as DATES in download_era5.py):
START = datetime(2020, 1, 1, 0)
END = datetime(2020, 12, 31, 23)
# Locations from the SITE_CONFIGS dictionary (same as download_era5.py):
SELECTED_LOCATIONS = [
    "Cabauw",
    "Mace Head",
    "ARM Southern Great Plains",
    "Summit Station",
    # "Concordia Dome C",
    "ARM Eastern North Atlantic",
]


def file_times(path):
    """All the timestamps in a GRIB file, sorted, as a DatetimeIndex."""
    ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
    try:
        if "valid_time" in ds.coords:
            values = np.atleast_1d(ds.valid_time.values).ravel()
        elif "time" in ds.coords:
            values = np.atleast_1d(ds.time.values).ravel()
            if "step" in ds.coords:
                values = (ds.time + ds.step).values.ravel()
        else:
            raise KeyError("no time coordinate in the file")
    finally:
        ds.close()
    return pd.DatetimeIndex(pd.unique(pd.DatetimeIndex(values))).sort_values()


def check(name, site):
    path = DATA_DIR / "raw_lvls_data" / site.model_level_filename
    if not path.exists():
        print(f"  MISSING FILE: {path}")
        return False

    try:
        times = file_times(path)
    except Exception as exc:
        print(f"  COULD NOT BE READ: {type(exc).__name__}: {exc}")
        return False

    expected = pd.date_range(START, END, freq="h")
    missing = expected.difference(times)
    extra_before = times[times < pd.Timestamp(START)]
    extra_after = times[times > pd.Timestamp(END)]

    print(f"  {len(times)} timesteps, {times[0]} -> {times[-1]}")
    if len(extra_before) or len(extra_after):
        print(f"  {len(extra_before) + len(extra_after)} timesteps outside the period (harmless)")

    if not len(missing):
        print(f"  complete: all {len(expected)} hours are there")
        return True

    missing_days = sorted({t.date() for t in missing})
    print(f"  {len(missing)} MISSING HOURS, on {len(missing_days)} day(s):")
    shown = missing_days[:10]
    for day in shown:
        n = sum(1 for t in missing if t.date() == day)
        print(f"    {day}  ({n} h)")
    if len(missing_days) > len(shown):
        print(f"    ... and {len(missing_days) - len(shown)} more days")
    return False


if __name__ == "__main__":
    expected_hours = int((END - START) / timedelta(hours=1)) + 1
    print(f"Period: {START} -> {END}  ({expected_hours} hourly timesteps expected)")
    print(f"Files:  {DATA_DIR / 'raw_lvls_data'}\n")

    results = {}
    for name in SELECTED_LOCATIONS:
        print(f"{name}:")
        results[name] = check(name, SITE_CONFIGS[name])
        print()

    good = [n for n, ok in results.items() if ok]
    bad = [n for n, ok in results.items() if not ok]
    print("-" * 70)
    if not bad:
        print(f"All {len(good)} locations are complete.")
    else:
        print(f"{len(good)}/{len(results)} locations complete. Incomplete: {', '.join(bad)}")
        print("Re-run download_era5.py (without flags) to request and fetch what is missing.")
