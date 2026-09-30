# 2026, L. Donati
# Performs analysis of ERA5 surface and model level datasets for four locations.

#%% Imports
%load_ext autoreload
%autoreload 2
from pathlib import Path
import era5_abl as era
from era5_abl.config import SITE_CONFIGS
from era5_abl.plotting import (plot_multi_dataset_pdf,
    plot_abl_top_vs_surface_scatter_contour,
    plot_Ri_vs_stability_function,
    plot_vertical_profile,
    plot_abl_top_vs_surface_hexbin
)
import xarray as xr

#%% User configuration

# Directory with data:
DATA_DIR = Path(
    "/Users/lodo0477/Documents/PhD/Research/"
    "Entrainment_with_Palli/ERA5_data/"
)
# Whether to retrieve data with CDS API or not:
SRF_DATA_RETRIEVAL = False
ML_DATA_RETRIEVAL = False
# Wether to process the data (filter, add multiple variables...)
PROCESS_DATASETS = True
# Whether to carry out the CDO processing (fldmean):
CDO_PROCESSING = False
# Whether to save CDO-processed datasets. 
SAVE_CDO = False
# Whether to save the filtered datasets or not:
SAVE_FILTERED_DATASETS = True
# Date interval considered:
DATES = "2020-01-01/2021-12-31"
# Number of levels with Ri higher than 0.25 to compute BLH
RIb_PERSISTENCE = 2
# Set filtering parameters (to filter for neutral and stable cloud-free layers, in this example)
filter_params = {
    "lcc_threshold": 0.15,                # Maximum amount of Low Level Clouds allowed by the cloud filtering
    "cloud_window_hours": 2,              # Amount of hours where the lcc_threshold must be maintained in cloud filtering
    "ri_surf_min": 0.0,                   # Minimum bulk Richardson number (Ri_b_srf) at height ri_surf_min_height retained by the stability filtering
    "ri_surf_min_height": 20.0,           # Reference height [m] for the surface-layer Ri_b (stability filtering and transfer functions)
    "dtheta_tol": 0.0,                    # [K] theta_v(z) - theta_v_2m must be >= -dtheta_tol at all levels below BLH (>0 keeps near-neutral layers)
    "Ri_c": 0.25,               # Critical Richardson number for stability filtering and computation
    "wind_dir_min_deg": 0,      # Wind direction filtering,  lower value in deg  (placeholder! Updated later where needed)
    "wind_dir_max_deg": 360,    # Wind direction filtering, higher value in deg (placeholder! Updated later where needed)
}
filtered_dir = DATA_DIR / "filtered_data"
# Optionally, only select some locations from the SITE_CONFIGS dictionary:
# (Side note: every new location has to be added to SITE_CONFIGS in era5_abl/src/config.py !)
SELECTED_LOCATIONS = [
    "Cabauw",
    "Mace Head",
    "ARM Southern Great Plains",
    "Summit Station",
    "Concordia Dome C",
    "ARM Eastern North Atlantic",
]
site_configs = {key : SITE_CONFIGS[key] for key in SELECTED_LOCATIONS}


#%% Retrieve ERA5 data
era.parallel_retrieval(
    site_names=SELECTED_LOCATIONS,
    dates=DATES,
    output_dir=DATA_DIR,
    max_workers=len(SELECTED_LOCATIONS)*2,
    retrieve_srf_data=SRF_DATA_RETRIEVAL,
    retrieve_ml_data=ML_DATA_RETRIEVAL,
)

#%% Main pipeline
ds_ml_dict = {}
ds_srf_dict = {}

# Process each dataset
if PROCESS_DATASETS:
    for loc, site in site_configs.items():
        print(f"\n--- Processing Location: {loc} ---")
        srf_path = DATA_DIR / "raw_surface_data" / site.surface_filename
        ml_path = DATA_DIR / "raw_lvls_data" / site.model_level_filename

        # File prep and spatial averaging
        if CDO_PROCESSING:
            ds_ml, ds_srf = era.prepare_dataset(str(ml_path), str(srf_path), location=loc)
        else:
            # NB: files processed before the height fix (Sept 2026) carry the old z -> re-run with CDO_PROCESSING = True once
            ds_srf = xr.open_dataset(f"{DATA_DIR}/{site.surface_filename.replace('.grib', '_CDO_processed.nc')}")
            ds_ml = xr.open_dataset(f"{DATA_DIR}/{site.model_level_filename.replace('.grib', '_CDO_processed.nc')}")
            ds_ml = ds_ml.drop_vars(["hyai", "hybi", "hyam", "hybm"], errors="ignore")  # avoids ~1 GB filtered files

        # Save unfiltered CDO-processed datasets
        if SAVE_CDO:
            ds_srf.to_netcdf(f"{DATA_DIR}/{site.surface_filename.replace('.grib', '_CDO_processed.nc')}")
            ds_ml.to_netcdf(f"{DATA_DIR}/{site.model_level_filename.replace('.grib', '_CDO_processed.nc')}")

        # Cloud filtering
        ds_ml_f0, ds_srf_f0 = ds_ml.copy(), ds_srf.copy()
        ds_ml_f1, ds_srf_f1 = era.filter_clouds(
            ds_ml_f0, ds_srf_f0, 
            lcc_thresh=filter_params["lcc_threshold"], window_hours=filter_params["cloud_window_hours"]
        )
        era.print_filter_output(ds_ml_f0, ds_ml_f1, "Low cloud cover filtering")

        # Stability filtering
        ds_ml_f1 = era.compute_grad_Ri(ds_ml_f1)
        ds_ml_f1 = era.compute_bulk_Ri(ds_ml_f1, ds_srf_f1, reference_height=None)
        ds_ml_f1 = era.compute_BLH_from_Ri_b(ds_ml_f1, Ri_c=filter_params["Ri_c"], persistence=RIb_PERSISTENCE)
        ds_ml_f2, ds_srf_f2 = era.filter_stability(ds_ml_f1, ds_srf_f1,
            ri_surf_min=filter_params["ri_surf_min"], ri_surf_min_height=filter_params["ri_surf_min_height"], 
            dtheta_tol=filter_params["dtheta_tol"],
        )
        era.print_filter_output(ds_ml_f1, ds_ml_f2, "Stability filtering")

        # Wind direction filtering
        ds_ml_f2 = era.compute_wind_dir(ds_ml_f2)
        dir_min, dir_max = site.wind_sector
        ds_ml_f3, ds_srf_f3 = era.filter_wind_dir(
            ds_ml_f2, ds_srf_f2,
            dir_min, dir_max,
        )
        era.print_filter_output(ds_ml_f2, ds_ml_f3, "Wind direction filtering")

        # Filter model level dataset to only retain values below BLH
        ds_ml_f4 = era.filter_ds_below_BLH(ds_ml_f3, ds_srf_f3)

        # Filtering is finished
        ds_ml_filtered = ds_ml_f4
        ds_srf_filtered = ds_srf_f3
        era.print_filter_output(ds_ml_f0, ds_ml_f3, "Total filtering results from the initial dataset")

        # Stability functions computation - GL18
        eps, eps_t = era.compute_epsilon(ds_srf_filtered, reference_height=filter_params["ri_surf_min_height"])
        ds_ml_filtered = era.compute_zeta_GL18(ds_ml_filtered, epsilon=eps, epsilon_t=eps_t, 
                                               reference_height=filter_params["ri_surf_min_height"])
        fm_20_GL18 = era.compute_fm_GL18(ds_ml_filtered, epsilon=eps)
        fh_20_GL18 = era.compute_fh_GL18(fm_20_GL18, ds_ml_filtered, epsilon_t=eps_t)
        ds_ml_filtered = ds_ml_filtered.assign(fm_20_GL18=fm_20_GL18, fh_20_GL18=fh_20_GL18)

        # Stability functions computation - IFS
        fm_20_IFS = era.compute_fm_IFS(ds_ml_filtered, reference_height=filter_params["ri_surf_min_height"])
        fh_20_IFS = era.compute_fh_IFS(ds_ml_filtered, reference_height=filter_params["ri_surf_min_height"])
        ds_ml_filtered = ds_ml_filtered.assign(fm_20_IFS=fm_20_IFS, fh_20_IFS=fh_20_IFS)

        # Store for multi-site comparisons
        ds_ml_dict[loc] = ds_ml_filtered
        ds_srf_dict[loc] = ds_srf_filtered

        # Save filtered datasets
        if SAVE_FILTERED_DATASETS:
            filter_params.update(wind_dir_min_deg=dir_min, wind_dir_max_deg=dir_max)
            era.save_filtered_dataset(
                ds_ml_filtered,
                location=loc,
                dataset_type="lvls",
                output_dir=filtered_dir,
                filter_params=filter_params,
            )
            era.save_filtered_dataset(
                ds_srf_filtered,
                location=loc,
                dataset_type="srf",
                output_dir=filtered_dir,
                filter_params=filter_params,
            )
else:
    for loc, site in site_configs.items():
        # Open already-stored datasets for multi-site comparisons
        ds_ml_dict[loc] = xr.open_dataset(filtered_dir / f"{loc.replace(' ','')}_lvls_filtered.nc")
        ds_srf_dict[loc] = xr.open_dataset(filtered_dir / f"{loc.replace(' ','')}_srf_filtered.nc")


# %% Plotting & Analysis
# Example 1 (time variable): PDF comparison of BLH across locations and ocmpare to diagnosed one
plot_multi_dataset_pdf(ds_ml_dict, var_name="BLH_Ri", bins="fd", density=True) 
_,ax=plot_multi_dataset_pdf(ds_srf_dict, var_name="blh", bins="fd", density=True)    
ax.set_xlim(left=0,right=2000)

# Example 2 (time-height variable): PDF comparison 
_,ax=plot_multi_dataset_pdf(ds_ml_dict, var_name="Ri_g", target_height=20.0, bins="fd", density=True)
ax.set_xlim(left=0,right=0.5)
ax.axvline(0.25,c="k")
_,ax=plot_multi_dataset_pdf(ds_ml_dict, var_name="Ri_g", target_height=None, bins="fd", density=True)
ax.set_xlim(left=0,right=1.5)
ax.axvline(0.25,c="k")


# Example 3: Scatter/KDE of Delta theta_v vs Wind speed at BLH (use theta_v, not t: t decreases with height even in neutral layers)
_,ax=plot_abl_top_vs_surface_scatter_contour(ds_ml_dict, ds_srf_dict, temp_var="theta_v")
ax.set_ylim(bottom=0)
ax.set_xlim(left=0)

# Example 4: Hexbin plot showing the toa-surface difference for every location separately
_,axs = plot_abl_top_vs_surface_hexbin(ds_ml_dict, ds_srf_dict, temp_var="theta_v", gridsize=40)
for ax in axs:
    ax.set_xlim(left=0, right=30)
    ax.set_ylim(top=6.1, bottom=-6)

# Example 5: Stability correction function curves
for f in ["fm","fh"]:
    _,ax = plot_Ri_vs_stability_function(ds_ml_dict, ds_srf_dict, target_var=f"{f}_20_GL18", 
                                        ref_IFS_profile=f"{f}_20_IFS", reference_height=filter_params["ri_surf_min_height"])
    ax.set_ylim(top=1.0,bottom=0.0)
    ax.set_xlim(left=0.0,right=0.5)

# Example 6: Single timestamp vertical profile
plot_vertical_profile(ds_ml_dict, "t", time="2020-07-15T12:00:00")

# Example 7: Time-Range variable profile sequence 
# Create one-entry dict to oplot only one location (can also be used with the entire dictionary, but might be confusing)
target_loc = "Mace Head"  
dict_from_loc = {target_loc: ds_ml_dict[target_loc]} 
plot_vertical_profile(dict_from_loc, "t", time_range=("2020-07-15T06:00:00", "2020-07-25T12:00:00"))

#%%

###############################
#### work in progress part ####
###############################

#%%

import matplotlib.pyplot as plt
import numpy as np

for v,var in enumerate(["z0", "z0h"]):
    fig,axs = plt.subplots(3,2,figsize=(10,9))
    axs = axs.flatten()
    fig.suptitle(fr"PDF of {['','thermal'][v]} roughness length ${var}$")
    for k,key in enumerate(ds_srf_dict.keys()):
        ax = axs[k]
        ds = ds_srf_dict[key]
        ax.hist(ds[var])
        ax.set_title(f"{key},  median: {np.median(ds[var].values.flatten()):.1e} m") 
    plt.tight_layout()

_,ax=plot_multi_dataset_pdf(ds_ml_dict, var_name="wind_speed", target_height=None, bins="fd", density=True)
_,ax=plot_multi_dataset_pdf(ds_ml_dict, var_name="Ri_b_srf", target_height=None, bins="fd", density=False)
ax.set_xlim(left=0,right=1)
ax.axvline(0.25,c="k")

# %%
