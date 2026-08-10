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
    plot_vertical_profile
)

# %% User configuration

# Directory with data:
DATA_DIR = Path(
    "/Users/lodo0477/Documents/PhD/Research/"
    "Entrainment_with_Palli/ERA5_data"
)
# Whether to retrieve data with CDS API or not:
SRF_DATA_RETRIEVAL = True
ML_DATA_RETRIEVAL = False
# Dates considered:
DATES = "2020-01-01/2021-12-31"
# Cloud filtering thresholds:
LCC_THRESH = 0.15
CLOUD_WINDOW_HOURS = 2
# Critical Richardson:
RI_C = 0.25
# Stability filtering thresholds:
GRAD_TOL = -2e-4
MIN_VALID_FRACTION = 0.8
SMOOTH_WINDOW = 3
# Reference height for some transfer functions/stability computations:
REFERENCE_HEIGHT = 20.0


#%% Retrieve ERA5 data
if DATA_RETRIEVAL:
    era.parallel_retrieval(
        dates=DATES,
        output_dir=DATA_DIR,
        max_workers=4,
        surface_variables="2m_dewpoint_temperature"
        retrieve_srf_data=SRF_DATA_RETRIEVAL,
        retrieve_srf_data=ML_DATA_RETRIEVAL,
    )

# If one only wants one site:
# site_name = "Mace Head"
# site = SITE_CONFIGS[site_name]


#%% Main pipeline
ds_ml_dict = {}
ds_srf_dict = {}

# Process each dataset
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
        lcc_thresh=LCC_THRESH, window_hours=CLOUD_WINDOW_HOURS
    )
    era.print_filter_output(ds_ml_f0, ds_ml_f1, "Low cloud cover filtering")

    # Stability filtering
    ds_ml_f1 = era.compute_grad_Ri(ds_ml_f1)
    ds_ml_f1 = era.compute_bulk_Ri(ds_ml_f1)
    ds_ml_f1 = era.compute_BLH_from_Ri_b(ds_ml_f1, Ri_c=RI_C)
    ds_ml_f2, ds_srf_f2 = era.filter_stability(ds_ml_f1, ds_srf_f1,
        ri_surface_min=0.0, grad_tol=GRAD_TOL,
        min_valid_fraction=MIN_VALID_FRACTION, smooth_window=SMOOTH_WINDOW,
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


# %% Plotting & Analysis
# Example 1 (time variable): PDF comparison of BLH across locations and ocmpare to diagnosed one
plot_multi_dataset_pdf(ds_ml_dict, var_name="BLH_Ri", bins=40) 
plot_multi_dataset_pdf(ds_srf_dict, var_name="blh", bins=40)    

# Example 2 (time-height variable): PDF comparison of Theta_v at 100m AGL
plot_multi_dataset_pdf(ds_ml_dict, var_name="Ri_g", target_height=100.0, bins=50)

# Example 3: Scatter/KDE of Delta T vs Wind speed at BLH
plot_abl_top_vs_surface_scatter_contour(ds_ml_dict, ds_srf_dict, temp_var="t")

# Example 4: Stability correction function curves
plot_Ri_vs_stability_function(ds_ml_dict, func_type="fm", reference_height=REFERENCE_HEIGHT)

# Example 5: Single timestamp vertical profile
plot_vertical_profile(ds_ml_dict, "theta_v", time="2020-07-15T12:00:00")

# Example 6: Time-Range variable profile sequence 
plot_vertical_profile(ds_ml_dict, "Ri_g", time_range=("2020-07-15T06:00:00", "2020-07-25T12:00:00"))


#%%