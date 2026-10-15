# 2026, L. Donati
# Checks that the merged ERA5 model level files (one per location, in ERA5_data/raw_lvls_data) are good, hour by
# hour. Run it once the downloads are finished, and again after anything that rewrites the files:
#
#   python validate_era5.py
#
# For every file it checks that
#   - every hour of the period is there (and lists the missing days if not),
#   - every hour has ALL its fields (a half-written day shows up here),
#   - no hour is there twice and the hours are in time order: a repeated hour makes CDO see extra time steps, so
#     the level data no longer line up with the surface data (see fix_repeated_hours.py),
#   - the file is not cut short.
# It reads only the GRIB message headers (a few seconds per file) and changes nothing.

from pathlib import Path
import os
from datetime import datetime, timedelta
import pandas as pd
from era5_abl.config import SITE_CONFIGS
from era5_abl.gribpoint import scan_hours

# Directory with data (same as download_era5.py):
# (override on another machine, e.g. a supercomputer, with:  export ERA5_DATA_DIR=/scratch/.../ERA5_data)
DATA_DIR = Path(os.environ.get(
    "ERA5_DATA_DIR",
    "/Users/lodo0477/Documents/PhD/Research/Entrainment_with_Palli/ERA5_data/",
))
# Period that the files are expected to cover (same as DATES in download_era5.py):
START = datetime(2020, 1, 1, 0)
END = datetime(2021, 12, 31, 23)
# Locations from the SITE_CONFIGS dictionary (same as download_era5.py):
SELECTED_LOCATIONS = [
    "Cabauw",
    "Mace Head",
    "ARM Southern Great Plains",
    "Summit Station",
    # "Concordia Dome C",
    "ARM Eastern North Atlantic",
]


def _hours(timestamps, limit=6):
    shown = ", ".join(f"{t:%Y-%m-%d %H:%M}" for t in timestamps[:limit])
    return shown + (f", ... and {len(timestamps) - limit} more" if len(timestamps) > limit else "")


def check(name, site):
    path = DATA_DIR / "raw_lvls_data" / site.model_level_filename
    if not path.exists():
        print(f"  MISSING FILE: {path}")
        return False

    try:
        scan = scan_hours(path)
    except Exception as exc:
        print(f"  COULD NOT BE READ: {type(exc).__name__}: {exc}")
        return False

    hours = sorted(scan["hours"])
    if not hours:
        print("  EMPTY FILE")
        return False
    times = pd.DatetimeIndex(hours)
    expected = pd.date_range(START, END, freq="h")
    missing = expected.difference(times)
    outside = times[(times < pd.Timestamp(START)) | (times > pd.Timestamp(END))]
    n_fields = scan["fields"]
    repeated = [t for t in hours if scan["hours"][t] > scan["present"][t]]
    incomplete = [t for t in hours if scan["present"][t] < n_fields]

    print(f"  {len(hours)} distinct hours, {times[0]} -> {times[-1]}, {n_fields} fields per hour")
    if len(outside):
        print(f"  {len(outside)} hours outside the period (harmless)")
    ok = True
    if repeated:
        ok = False
        print(f"  {len(repeated)} HOUR(S) PRESENT MORE THAN ONCE: {_hours(repeated)}")
        print("    -> prepare_dataset would stop with a size mismatch; run fix_repeated_hours.py")
    if scan["unordered"]:
        ok = False
        print("  THE HOURS ARE NOT IN TIME ORDER (a repeated hour or a bad merge)")
    if incomplete:
        ok = False
        print(f"  {len(incomplete)} HOUR(S) WITH MISSING FIELDS: {_hours(incomplete)}")
    if len(missing):
        ok = False
        missing_days = sorted({t.date() for t in missing})
        print(f"  {len(missing)} MISSING HOURS, on {len(missing_days)} day(s):")
        shown = missing_days[:10]
        for day in shown:
            n = sum(1 for t in missing if t.date() == day)
            print(f"    {day}  ({n} h)")
        if len(missing_days) > len(shown):
            print(f"    ... and {len(missing_days) - len(shown)} more days")
    if ok:
        print(f"  complete: all {len(expected)} hours are there, once each and in order")
    return ok


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
        print("Repeated hours: run fix_repeated_hours.py. Missing hours: re-run download_era5.py (without flags) "
              "to request and fetch what is missing.")
