# 2026, L. Donati
# Finds and removes hours that appear more than once in the model level GRIB files.
#
# Why: if the same hour is in a file twice, CDO reads it as extra time steps, the level data then have more time
# steps than the surface data, and prepare_dataset stops with a "sizes" error in compute_ecmwf_pressure_and_height.
# This happened to the first piece of January 2021 (a download that was interrupted and resumed, or run twice at
# the same time, appended hours after the end of the piece). validate_era5.py reports such files too.
#
# What it does: for every file with repeated hours, keeps the FIRST message of each (hour, field) and drops the
# later copies. The original is kept next to the new file as <name>.orig and is never overwritten.
#
#   python fix_repeated_hours.py --dry-run          # only report, change nothing
#   python fix_repeated_hours.py                    # fix every file under <ERA5_data>/raw_lvls_data
#   python fix_repeated_hours.py FILE [FILE ...]    # fix only these files
#
# A merged file (<site>_lvls.grib) is built from the pieces, so fix both: the script does when it is run over the
# whole folder. Re-merging the pieces afterwards would also work, but takes ~20 minutes per site in CDO.

import argparse
import os
import shutil
import sys
from pathlib import Path

from era5_abl.gribpoint import drop_repeated_messages, find_repeated_messages

DATA_DIR = Path(os.environ.get(
    "ERA5_DATA_DIR",
    "/Users/lodo0477/Documents/PhD/Research/Entrainment_with_Palli/ERA5_data/",
))


def describe(repeated, limit=6):
    hours = sorted({v for v, _, _ in repeated})
    shown = ", ".join(f"{h:%Y-%m-%d %H:%M}" for h in hours[:limit])
    return f"{len(hours)} hour(s) appear more than once ({shown}{', ...' if len(hours) > limit else ''})"


def main():
    parser = argparse.ArgumentParser(description="Remove repeated hours from model level GRIB files.")
    parser.add_argument("files", nargs="*", help="files to check (default: every .grib under raw_lvls_data)")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="the ERA5_data directory")
    parser.add_argument("--dry-run", action="store_true", help="only report, change nothing")
    args = parser.parse_args()

    files = [Path(f) for f in args.files] or sorted((args.data_dir / "raw_lvls_data").rglob("*.grib"))
    if not files:
        sys.exit(f"No .grib files found under {args.data_dir / 'raw_lvls_data'}")

    n_fixed = 0
    for path in files:
        info = find_repeated_messages(path)
        if not info["repeated"]:
            print(f"ok      {path.name}")
            continue
        print(f"REPEATS {path.name}: {describe(info['repeated'])}, {len(info['repeated'])} fields repeated")
        if args.dry_run:
            continue

        orig, fixed = path.with_name(path.name + ".orig"), path.with_name(path.name + ".fixed")
        if orig.exists():
            sys.exit(f"{orig.name} already exists: move it away first (the original is never overwritten).")
        if shutil.disk_usage(path.parent).free < path.stat().st_size * 1.1:
            sys.exit(f"Not enough free disk space to write a corrected copy of {path.name}.")
        stats = drop_repeated_messages(path, fixed)
        check = find_repeated_messages(fixed)
        if check["repeated"] or check["unordered"]:
            fixed.unlink()
            sys.exit(f"The corrected copy of {path.name} is still not right; nothing was changed.")
        path.rename(orig)           # the original is kept...
        fixed.rename(path)          # ...and the corrected file takes its place
        n_fixed += 1
        print(f"        fixed: kept {stats['kept']:,} messages, dropped {stats['dropped']:,}; "
              f"the original is {orig.name}")

    print(f"\n{n_fixed} file(s) fixed." if not args.dry_run else "\nDry run: nothing was changed.")
    if n_fixed:
        print("Once you are happy with the result, the *.orig files can be deleted to free the disk.")


if __name__ == "__main__":
    main()
