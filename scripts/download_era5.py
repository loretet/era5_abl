# Downloads the ERA5 model level data from the CDS, for the areas that contain the selected locations.
# Same retrieval as in main.py, but as a plain python script that can run in the background on a machine
# that stays on (the whole download takes hours to days, and the script has to run until the end):
#
#   nohup python download_era5.py > download.log 2>&1 &
#
# Progress:  tail -f download.log
# The files that are already downloaded are skipped: after an interruption or failed requests, just run it again.
# The CDS API key must be in ~/.cdsapirc

from pathlib import Path
import era5_abl as era

# Directory with data (on the machine where this script runs):
DATA_DIR = Path(
    "/Users/lodo0477/Documents/PhD/Research/"
    "Entrainment_with_Palli/ERA5_data/"
)
# Date interval considered (whole months, so that any sub-period can be processed later):
DATES = "2020-01-01/2021-12-31"
# Locations from the SITE_CONFIGS dictionary (they decide which areas are downloaded):
SELECTED_LOCATIONS = [
    "Cabauw",
    "Mace Head",
    "ARM Southern Great Plains",
    "Summit Station",
    "Concordia Dome C",
    "ARM Eastern North Atlantic",
]

era.parallel_retrieval(
    site_names=SELECTED_LOCATIONS,
    dates=DATES,
    output_dir=DATA_DIR,
    max_workers=15,     # with more requests at the same time the downloads get slower
    retrieve_srf_data=False,
    retrieve_ml_data=True,
)
