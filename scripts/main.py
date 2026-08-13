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
    "Entrainment_with_Palli/ERA5_data"
)
# Whether to retrieve data with CDS API or not:
SRF_DATA_RETRIEVAL = False
ML_DATA_RETRIEVAL = False
# Whether to filter the datasets or not:
FILTER_DATASETS = True
# Dates considered:
DATES = "2020-01-01/to/2020-12-31"
# Critical Richardson:
RI_C = 0.25
# Reference height for some transfer functions/stability computations:
REFERENCE_HEIGHT = 20.0
# Number of levels with Ri higher than 0.25 to compute BLH
RIb_PERSISTENCE = 5
# Set filtering parameters (to filter for neutral and stable cloud-free layers, in this example)
filter_params = {
    "lcc_threshold": 0.15,                # Maximum amount of Low Level Clouds allowed by the cloud filtering
    "cloud_window_hours": 2,              # Amount of hours where the lcc_threshold must be maintained in cloud filtering
    "ri_surf_min": -2e-4,                 # Minimum/maximum Richardson number at height ri_surf_min_height retained by the stability filtering
    "ri_surf_min_height": REFERENCE_HEIGHT,       # Height at which the minimum/maximum surface Richardson number is computed for stability filtering
    "grad_tol": -2e-4,                            # Minimum/maximum Richardson number retained by the stability filtering
    "min_valid_grad_fraction": 0.8,      # Minimum fraction of d(theta_v)/dz that must be above grad_tol
    "grad_smooth_window": 3,             # Gradient smoothing windows for stability filtering 
    "Ri_c": RI_C,               # Critical Richardson number for stability filtering and computation
    "wind_dir_min_deg": 0,      # Wind direction filtering,  lower value in deg  (placeholder! Updated later)
    "wind_dir_max_deg": 360,    # Wind direction filtering, higher value in deg (placeholder! Updated later)
}
filtered_dir = DATA_DIR / "filtered_data"


#%% Retrieve ERA5 data
era.parallel_retrieval(
    dates=DATES,
    output_dir=DATA_DIR,
    max_workers=4,
    retrieve_srf_data=SRF_DATA_RETRIEVAL,
    retrieve_ml_data=ML_DATA_RETRIEVAL,
)

#%% Main pipeline
ds_ml_dict = {}
ds_srf_dict = {}

# Process each dataset
if FILTER_DATASETS:
    for loc, site in SITE_CONFIGS.items():
        print(f"\n--- Processing Location: {loc} ---")
        ml_path = DATA_DIR / site.model_level_filename
        srf_path = DATA_DIR / site.surface_filename

        # File prep and spatial averaging
        ds_ml, ds_srf = era.prepare_dataset(str(ml_path), str(srf_path), location=loc)

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
            grad_tol=filter_params["grad_tol"], min_valid_fraction=filter_params["min_valid_grad_fraction"], 
            smooth_window=filter_params["grad_smooth_window"]
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

        # Filtering is finished
        ds_ml_filtered = ds_ml_f3
        ds_srf_filtered = ds_srf_f3
        era.print_filter_output(ds_ml_f0, ds_ml_f3, "Total filtering results from the initial dataset")

        # Stability functions computation
        eps, eps_t = era.compute_epsilon(location=loc, reference_height=REFERENCE_HEIGHT)
        ds_ml_filtered = era.compute_zeta_GL18(
            ds_ml_filtered,
            epsilon=eps,
            epsilon_t=eps_t,
            reference_height=REFERENCE_HEIGHT
        )
        fm_20 = era.compute_fm(
            ds_ml_filtered,
            epsilon=eps,
        )
        fh_20 = era.compute_fh(
            fm_20,
            ds_ml_filtered,
            epsilon_t=eps_t,
        )
        ds_ml_filtered = ds_ml_filtered.assign(
            fm_20=fm_20,
            fh_20=fh_20,
        )

        # Store for multi-site comparisons
        ds_ml_dict[loc] = ds_ml_filtered
        ds_srf_dict[loc] = ds_srf_filtered

        # Save datasets
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
    for loc, site in SITE_CONFIGS.items():
        # Open already-stored datasets for multi-site comparisons
        ds_ml_dict[loc] = xr.open_dataset(filtered_dir / f"{loc.replace(" ","")}_lvls_filtered.nc")
        ds_srf_dict[loc] = xr.open_dataset(filtered_dir / f"{loc.replace(" ","")}_srf_filtered.nc")


# %% Plotting & Analysis
# Example 1 (time variable): PDF comparison of BLH across locations and ocmpare to diagnosed one
plot_multi_dataset_pdf(ds_ml_dict, var_name="BLH_Ri", bins="fd", density=True) 
_,ax=plot_multi_dataset_pdf(ds_srf_dict, var_name="blh", bins="fd", density=True)    
ax.set_xlim(left=0,right=2000)

# Example 2 (time-height variable): PDF comparison 
_,ax=plot_multi_dataset_pdf(ds_ml_dict, var_name="Ri_g", target_height=20.0, bins="fd", density=True)
ax.set_xlim(left=0,right=0.5)
# ax.set_ylim(top=5,bottom=0.0)

# Example 3: Scatter/KDE of Delta T vs Wind speed at BLH
_,ax=plot_abl_top_vs_surface_scatter_contour(ds_ml_dict, ds_srf_dict, temp_var="theta_v")
ax.axhline(0.0,c="k",alpha=0.8)

# Example 4: Hexbin plot showing the toa-surface difference for every location separately
_,axs = plot_abl_top_vs_surface_hexbin(ds_ml_dict, ds_srf_dict, temp_var="theta_v", gridsize=40)
for ax in axs:
    ax.set_xlim(left=0, right=30)
    ax.set_ylim(top=6.1, bottom=-6)

# Example 5: Stability correction function curves
_,ax = plot_Ri_vs_stability_function(ds_ml_dict, target_var="fm_20", reference_height=REFERENCE_HEIGHT)
ax.set_ylim(top=1.0,bottom=0.0)
ax.set_xlim(left=0.0,right=0.5)
_,ax = plot_Ri_vs_stability_function(ds_ml_dict, target_var="fh_20", reference_height=REFERENCE_HEIGHT)
ax.set_ylim(top=1.0,bottom=0.0)
ax.set_xlim(left=0.0,right=0.5)  

# Example 6: Single timestamp vertical profile
plot_vertical_profile(ds_ml_dict, "theta_v", time="2020-07-15T12:00:00")

# Example 7: Time-Range variable profile sequence 
# Create one-entry dict to oplot only one location (can also be used with the entire dictionary, but might be confusing)
target_loc = "Mace Head"  
dict_from_loc = {target_loc: ds_ml_dict[target_loc]} 
plot_vertical_profile(dict_from_loc, "theta_v", time_range=("2020-07-15T06:00:00", "2020-07-25T12:00:00"))

#%%