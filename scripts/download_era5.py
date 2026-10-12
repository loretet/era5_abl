# 2026, L. Donati
# Downloads the ERA5 model level data of the areas that contain the selected locations (ERA5_RETRIEVAL_AREAS
# in era5_abl/config.py), and extracts the locations from every file as soon as it is downloaded.
#
# The requests are first SENT to the CDS, which then works on them (hours to days) whether this computer is on
# or not, and are FETCHED afterwards. So the computer only has to be on while the files are downloaded:
#
#   python download_era5.py --submit-only   # only send the requests (minutes), then the computer can be
#                                           # switched off while the CDS works on them
#   python download_era5.py --fetch-only    # download the results that are ready, extract the locations,
#                                           # delete the big files, then KEEP WAITING for the rest
#   python download_era5.py                 # both: send whatever is missing, then fetch what is ready
#
# Add --once to any of these to make it do one pass and stop, instead of waiting for the requests that are
# still being worked on at the CDS. This is the mode for a SCHEDULED run (launchd, cron): without it the
# script sleeps until everything is downloaded, which keeps this computer awake for hours or days.
#
#   python download_era5.py --once               # one pass: send what is missing, collect what is ready, exit
#   python download_era5.py --fetch-only --once  # one pass, downloading only
#
# Both steps can be interrupted (Ctrl-C) and started again at any time: the ids of the requests are stored in
# ERA5_data/raw_lvls_data/cds_requests.json and the files that are already there are never downloaded again.
# To let it run in the background: nohup caffeinate -is python download_era5.py > download.log 2>&1 &
# (caffeinate is NOT needed with --once, which is the point of it.)
# NOTE: the CDS does not keep the results forever, so fetch them within a few days.
# The CDS API key must be in ~/.cdsapirc

from pathlib import Path
import os
import argparse
import era5_abl as era
from era5_abl.config import SITE_CONFIGS

# Directory with data (on the machine where this script runs):
# (override on another machine, e.g. a supercomputer, with:  export ERA5_DATA_DIR=/scratch/.../ERA5_data)
DATA_DIR = Path(os.environ.get(
    "ERA5_DATA_DIR",
    "/Users/lodo0477/Documents/PhD/Research/Entrainment_with_Palli/ERA5_data/",
))
# Date interval considered (whole months, so that any sub-period can be processed later):
DATES = "2021-01-01/2021-12-31"
# Whether to delete each downloaded area file once the locations are extracted from it (keeps the disk free:
# only the locations are kept, which is ~100 times less):
DELETE_AREA_FILES = True
# Locations from the SITE_CONFIGS dictionary (they decide which areas are downloaded):
SELECTED_LOCATIONS = [
    "Cabauw",
    "Mace Head",
    "ARM Southern Great Plains",
    "Summit Station",
    # "Concordia Dome C",
    "ARM Eastern North Atlantic",
]
site_configs = {key: SITE_CONFIGS[key] for key in SELECTED_LOCATIONS}


def extract_and_clean(path, area_name, piece):
    """Called right after each download: cuts the locations out of the file and deletes it."""
    era.extract_sites_from_file(path, area_name, piece, site_configs, DATA_DIR,
                                delete_area_file=DELETE_AREA_FILES)


def main():
    parser = argparse.ArgumentParser(
        description="Send the ERA5 model level requests to the CDS and/or fetch the ones that are ready.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--submit-only", action="store_true",
                      help="only send the requests, do not download anything")
    mode.add_argument("--fetch-only", action="store_true",
                      help="only download and extract the results that are ready")
    parser.add_argument("--once", action="store_true",
                       help="do one pass and stop, instead of waiting for the requests that are still running "
                            "on the CDS (use this for a scheduled run: it does not keep the computer awake)")
    args = parser.parse_args()

    submit_requests = not args.fetch_only
    fetch_requests = not args.submit_only

    if submit_requests:
        era.submit_model_level_requests(
            dates=DATES,
            site_names=SELECTED_LOCATIONS,
            output_dir=DATA_DIR,
            fast_fail=args.once,    # a scheduled run must not hang for hours if the network is down
        )

    if fetch_requests:
        complete = era.fetch_model_level_requests(
            dates=DATES,
            site_names=SELECTED_LOCATIONS,
            output_dir=DATA_DIR,
            on_file=extract_and_clean,
            once=args.once,
        )
        if complete:
            # One GRIB file per location for the whole period (the input of prepare_dataset in main.py)
            for name, site in site_configs.items():
                era.extract_site_data(site, dates=DATES, data_dir=DATA_DIR,
                                      output_path=DATA_DIR / "raw_lvls_data" / site.model_level_filename)
            print("\nThe locations are ready in raw_lvls_data: run main.py (ML_SITE_EXTRACTION is not needed).")
        elif args.once:
            print("\nSome requests are still missing: the next run collects them.")
        else:
            print("\nSome requests are still missing: run this script again (--fetch-only) in a few hours.")


if __name__ == "__main__":
    main()
