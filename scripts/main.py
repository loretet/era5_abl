
ds_ml_dict = {}
ds_srf_dict = {}

#%% Main pipeline
for loc, cfg in site_configs.items():
    print(f"\n--- Processing Location: {loc} ---")

    # File prep and spatial averaging
    ds_ml, ds_srf = prepare_dataset(cfg["ml_grib"], cfg["srf_grib"], location=loc)

    # Cloud filtering
    ds_ml_f0, ds_srf_f0 = ds_ml.copy(), ds_srf.copy()
    ds_ml_f1, ds_srf_f1 = filter_clouds(ds_ml_f0, ds_srf_f0, lcc_thresh=0.1, window_hours=3)
    _ = print_filter_output(ds_ml_f0, ds_ml_f1, "Low cloud cover filtering")

    # Stability filtering
    ds_ml_f1 = compute_grad_Ri_z(ds_ml_f1)
    ds_ml_f1 = compute_BLH_from_Ri_b(ds_ml_f1, Ri_c=0.25)
    ds_ml_f2, ds_srf_f2 = filter_stability(ds_ml_f1, ds_srf_f1, tol=1e-5)
    _ = print_filter_output(ds_ml_f1, ds_ml_f2, "Stability filtering")

    # Wind direction filtering
    ds_ml_f2 = compute_wind_dir(ds_ml_f2)
    if "wind_sector" in cfg:
        dir_min, dir_max = cfg["wind_sector"]
        ds_ml_f3, ds_srf_f3 = filter_wind_dir(ds_ml_f2, dir_min, dir_max)
        _ = print_filter_output(ds_ml_f2, ds_ml_f3, "Wind direction sector filtering")
        ds_ml_filtered, ds_srf_filtered = ds_ml_f3, ds_srf_f3
    else:
        ds_ml_filtered, ds_srf_filtered = ds_ml_f2, ds_srf_f2
    _ = print_filter_output(ds_ml_f0, ds_ml_filtered, "Total filtering")


    # Stability functions computation
    eps, eps_t = compute_epsilon(ds_ml_filtered, location=loc)
    ds_ml_filtered = compute_zeta_GL18(ds_ml_filtered, epsilon=eps, epsilon_t=eps_t)
    fm_20 = compute_fm(ds_ml_filtered, epsilon=eps)
    fh_20 = compute_fh(fm_20, ds_ml_filtered, epsilon_t=eps_t)
    ds_ml_filtered = ds_ml_filtered.assign(fm_20=fm_20, fh_20=fh_20)

    # Store for multi-site comparisons
    ds_ml_dict[loc] = ds_ml_filtered
    ds_srf_dict[loc] = ds_srf_filtered

# %% Plotting & Analysis
# Example 1 (time variable): PDF comparison of BLH across locations and ocmpare to diagnosed one
plot_multi_dataset_pdf(ds_ml_dict, var_name="BLH", bins=40) 
plot_multi_dataset_pdf(ds_srf_dict, var_name="blh", bins=40)    

# Example 2 (time-height variable): PDF comparison of Theta_v at 100m AGL
plot_multi_dataset_pdf(ds_ml_dict, var_name="Ri_g", target_height=100.0, bins=50)

# Example 3: Scatter/KDE of Delta T vs Wind speed at BLH
plot_abl_top_vs_surface_scatter_contour(ds_ml_dict, ds_srf_dict, temp_var="t")

# Example 4: Stability correction function curves
plot_Ri_vs_stability_function(ds_ml_dict, func_type="fm")

# Example 5: Single timestamp vertical profile
plot_vertical_profile(ds_ml_dict, "theta_v", time="2020-07-15T12:00:00")

# Example 6: Time-Range variable profile sequence 
plot_vertical_profile(ds_ml_dict, "Ri_g", time_range=("2020-07-15T06:00:00", "2020-07-25T12:00:00"))
