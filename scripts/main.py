# 2026, L. Donati
# Performs analysis of ERA5 datasets for four locations.
# Work done for the paper X

#%% Imports
import os
import era5_abl as era

#%% Configuration
data_dir = "/Users/lodo0477/Documents/PhD/Research/Entrainment_with_Palli/ERA5_data"

# Define location metadata maps
site_configs = {
    "Mace Head": {
        "ml_grib": os.path.join(data_dir, "MaceHead_lvls.grib"),
        "srf_grib": os.path.join(data_dir, "MaceHead_surface.grib"),
        "wind_sector": (180.0, 360.0),  # Marine sector 
    },
    "Cabauw": {
        "ml_grib": os.path.join(data_dir, "Cabauw_lvls.grib"),
        "srf_grib": os.path.join(data_dir, "Cabauw_surface.grib"),
        "wind_sector": (0.0, 360.0)
    },
    "Summit Station": {
        "ml_grib": os.path.join(data_dir, "SummitStation_lvls.grib"),
        "srf_grib": os.path.join(data_dir, "SummitStation_surface.grib"),
        "wind_sector": (0.0, 360.0)
    },
    "Southern Great Plains": {
        "ml_grib": os.path.join(data_dir, "ARMGreatPlains_lvls.grib"),
        "srf_grib": os.path.join(data_dir, "ARMGreatPlains_surface.grib"),
        "wind_sector": (0.0, 360.0)
    },
}

ds_ml_dict = {}
ds_srf_dict = {}

#%% Main pipeline
for loc, cfg in site_configs.items():
    print(f"\n--- Processing Location: {loc} ---")

    # File prep and spatial averaging
    ds_ml, ds_srf = era.prepare_dataset(cfg["ml_grib"], cfg["srf_grib"], location=loc)

    # Cloud filtering
    ds_ml_f0, ds_srf_f0 = ds_ml.copy(), ds_srf.copy()
    ds_ml_f1, ds_srf_f1 = era.filter_clouds(ds_ml_f0, ds_srf_f0, lcc_thresh=0.1, window_hours=3)
    _ = era.print_filter_output(ds_ml_f0, ds_ml_f1, "Low cloud cover filtering")

    # Stability filtering
    ds_ml_f1 = era.compute_grad_Ri_z(ds_ml_f1)
    ds_ml_f1 = era.compute_BLH_from_Ri_b(ds_ml_f1, Ri_c=0.25)
    ds_ml_f2, ds_srf_f2 = era.filter_stability(ds_ml_f1, ds_srf_f1, tol=1e-5)
    _ = era.print_filter_output(ds_ml_f1, ds_ml_f2, "Stability filtering")

    # Wind direction filtering
    ds_ml_f2 = era.compute_wind_dir(ds_ml_f2)
    if "wind_sector" in cfg:
        dir_min, dir_max = cfg["wind_sector"]
        ds_ml_f3, ds_srf_f3 = era.filter_wind_dir(ds_ml_f2, dir_min, dir_max)
        _ = era.print_filter_output(ds_ml_f2, ds_ml_f3, "Wind direction sector filtering")
        ds_ml_filtered, ds_srf_filtered = ds_ml_f3, ds_srf_f3
    else:
        ds_ml_filtered, ds_srf_filtered = ds_ml_f2, ds_srf_f2
    _ = era.print_filter_output(ds_ml_f0, ds_ml_filtered, "Total filtering")


    # Stability functions computation
    eps, eps_t = era.compute_epsilon(ds_ml_filtered, location=loc)
    ds_ml_filtered = era.compute_zeta_GL18(ds_ml_filtered, epsilon=eps, epsilon_t=eps_t)
    fm_20 = era.compute_fm(ds_ml_filtered, epsilon=eps)
    fh_20 = era.compute_fh(fm_20, ds_ml_filtered, epsilon_t=eps_t)
    ds_ml_filtered = ds_ml_filtered.assign(fm_20=fm_20, fh_20=fh_20)

    # Store for multi-site comparisons
    ds_ml_dict[loc] = ds_ml_filtered
    ds_srf_dict[loc] = ds_srf_filtered

# %% Plotting & Analysis
# Example 1 (time variable): PDF comparison of BLH across locations and ocmpare to diagnosed one
era.plot_multi_dataset_pdf(ds_ml_dict, var_name="BLH", bins=40) 
era.plot_multi_dataset_pdf(ds_srf_dict, var_name="blh", bins=40)    

# Example 2 (time-height variable): PDF comparison of Theta_v at 100m AGL
era.plot_multi_dataset_pdf(ds_ml_dict, var_name="Ri_g", target_height=100.0, bins=50)

# Example 3: Scatter/KDE of Delta T vs Wind speed at BLH
era.plot_abl_top_vs_surface_scatter_contour(ds_ml_dict, ds_srf_dict, temp_var="t")

# Example 4: Stability correction function curves
era.plot_Ri_vs_stability_function(ds_ml_dict, func_type="fm")

# Example 5: Single timestamp vertical profile
era.plot_vertical_profile(ds_ml_dict, "theta_v", time="2020-07-15T12:00:00")

# Example 6: Time-Range variable profile sequence 
era.plot_vertical_profile(ds_ml_dict, "Ri_g", time_range=("2020-07-15T06:00:00", "2020-07-25T12:00:00"))
