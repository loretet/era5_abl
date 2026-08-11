import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D   
from matplotlib.colors import LinearSegmentedColormap
from .operations import interpolate_to_height
from .stability import compute_difference_surface_top_ABL
from .config import SITE_CONFIGS


DATASET_STYLES = [
    {"color": "#1f77b4", "linestyle": "-", "marker": "o"},   
    {"color": "#ff7f0e", "linestyle": "--", "marker": "s"},  
    {"color": "#2ca02c", "linestyle": "-.", "marker": "^"}, 
    {"color": "#d62728", "linestyle": ":", "marker": "D"},   
]


def approx_scientific_notation(value: float) -> str:
    """Format a float in approximate scientific notation. E.g. 0.00234 => 2e-3."""
    formatted = f"{value:.0e}"
    if "e" in formatted:
        base, exp = formatted.split("e")
        exp = exp.lstrip("+0") or "0"
        return f"{base}e{exp}"
    return formatted


def plot_in_time(
        ds: xr.Dataset, 
        var_name: str,
        style: list[dict[str,str]] = DATASET_STYLES
    ):
    """
    Plots vertical profiles of var_name against height AGL (z) across time.
    Uses predefined dataset styles for consistent line and marker aesthetics.
    """
    time_dim = "valid_time" if "valid_time" in ds.sizes else "time"
    times = ds[time_dim].values
    N = len(times)

    fig = plt.figure(figsize=(6, 8))
    for i, t in enumerate(times):
        ds_time = ds.sel({time_dim: t})
        ds_style = style[i % len(style)]
        lbl = str(t)[:16] if (i == 0 or i == N - 1) else None
        plt.plot(
            ds_time[var_name],
            ds_time["z"],
            color=ds_style["color"],
            linestyle=ds_style["linestyle"],
            marker=ds_style["marker"],
            label=lbl,
        )

    plt.xlabel(var_name)
    plt.ylabel("Height AGL [m]")
    plt.grid(True, linestyle="--", alpha=0.6)
    if N > 1:
        plt.legend()
    plt.title(f"Vertical Profile Evolution: {var_name}")

    return fig

def plot_abl_top_vs_surface_scatter_contour(
    ds_dict: dict[str, xr.Dataset],
    ds_srf_dict: dict[str, xr.Dataset],
    temp_var: str = "t",
    style: list[dict[str,str]] = DATASET_STYLES
):
    """
    Plots a combined scatter and 2D KDE contour plot of Delta T (ABL Top - Surface)

    vs. Wind Speed at the ABL Top across 4 datasets.
    """
    fig, ax = plt.subplots(figsize=(9, 7))

    legend_handles = []
    for idx, (name, ds) in enumerate(ds_dict.items()):
        ds_style = style[idx % len(style)]
        ds_srf = ds_srf_dict[name]

        # 1. Compute Temperature Difference (ABL Top - Surface)
        delta_T = compute_difference_surface_top_ABL(
            ds, ds_srf, var_name=temp_var
        ).values

        # 2. Extract Wind Speed at BLH for each timestep
        n_times = ds.sizes["time"]
        u_blh = np.zeros(n_times)

        for t in range(n_times):
            z_t = ds["z"].isel(time=t).values
            blh_t = ds_srf["blh"].isel(time=t).values
            k_idx = np.abs(z_t - blh_t).argmin()
            u_blh[t] = ds["wind_speed"].isel(time=t, model_level=k_idx).values

        # Filter out NaN/Inf values if present
        valid_mask = np.isfinite(u_blh) & np.isfinite(delta_T)
        x_val = u_blh[valid_mask]
        y_val = delta_T[valid_mask]

        # Scatter plot (low opacity)
        ax.scatter(
            x_val,
            y_val,
            color=ds_style["color"],
            alpha=0.01,
            s=20,
            edgecolors="none",
        )

        # Overlaid 2D Density Contours (KDE)
        sns.kdeplot(
            x=x_val,
            y=y_val,
            ax=ax,
            color=ds_style["color"],
            levels=4,
            linewidths=1.5,
            linestyles=ds_style["linestyle"],
            cut=0,                          # cut=0 takes out contour artifact near zero. Note: this is an ok approximation since we only want
            clip=((0, None), (None, None)), # a qualitative indication of the cluster, but shouldn't be used to estimate the probability
        )                                   # density near U=0 quantiatively!
        handle = Line2D(
            [],[], color=ds_style["color"], linestyle=ds_style["linestyle"],
            label=f"{name} (n={len(y_val)})"
        )
        legend_handles.append(handle)

    ax.set_xlabel("Wind Speed at BLH [m/s]", fontsize=11)
    ax.set_ylabel(
        f"$\Delta {temp_var.upper()}$ (ABL Top - Surface) [K]", fontsize=11
    )
    ax.set_title(
        "ABL Top Wind Speed vs. Temperature Difference across Locations",
        fontsize=12,
    )
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", handles=legend_handles)
    plt.tight_layout()

    return fig, ax

def plot_multi_dataset_pdf(
    ds_dict: dict[str, xr.Dataset],
    var_name: str,
    target_height: float | None = None,
    bins: int | str = "fd",
    style: list[dict[str,str]] = DATASET_STYLES,
    density: bool = True
):
    """
    Plots Probability Density Functions (PDFs) of a specified variable across 4 datasets.
    If the variable has a 'model_level' dimension, it extracts values at the
    level closest to target_height [m AGL]. Otherwise, it flattens the 1D/2D
    field.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for idx, (name, ds) in enumerate(ds_dict.items()):
        ds_style = style[idx % len(style)]

        if var_name not in ds:
            raise KeyError(f"Variable '{var_name}' not found in dataset '{name}'.")

        data_array = ds[var_name]

        # Check if variable depends on vertical model levels
        if "model_level" in data_array.sizes:
            if target_height is None:
                raise ValueError(
                    f"Variable '{var_name}' varies with model level. You must specify 'target_height' [m AGL]."
                )

            # Extract nearest index to target_height per timestep
            k_indices = np.abs(ds["z"] - target_height).argmin(
                dim="model_level"
            )
            extracted_vals = ds[var_name].isel(model_level=k_indices).values.flatten()
            height_str = fr" at $z \approx {target_height}$ m AGL"
        else:
            extracted_vals = data_array.values.flatten()
            height_str = ""

        # Remove NaNs for histogram evaluation
        extracted_vals = extracted_vals[np.isfinite(extracted_vals)]

        # Compute empirical PDF step-histogram
        counts, bin_edges = np.histogram(
            extracted_vals, bins=bins, density=density
        )
        n_bins = len(bin_edges) - 1
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        ax.plot(
            bin_centers,
            counts,
            color=ds_style["color"],
            linestyle=ds_style["linestyle"],
            linewidth=2,
            label=f"{name}, bins={n_bins}",
        )

    ax.set_xlabel(f"{var_name}", fontsize=11)
    ax.set_ylabel("Probability Density", fontsize=11)
    ax.set_title(f"PDF Comparison: {var_name}{height_str}", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")
    plt.tight_layout()

    return fig, ax


def plot_Ri_vs_stability_function(
    ds_dict: dict[str, xr.Dataset], target_var: str = "fm",
    reference_height: float = 20.0, style: list[dict[str,str]] = DATASET_STYLES
):
    """
    Plots Bulk Richardson Number on the x-axis against the GL18 normalised
    transfer coefficients f_m or f_h on the y-axis for all four datasets.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for idx, (name, ds) in enumerate(ds_dict.items()):
        ds_style = style[idx % len(style)]

        # Extract Ri at reference_height and stability function arrays
        ri_ref = interpolate_to_height(ds, "Ri_b_srf", target_height=reference_height).values.flatten()
        f_val = ds[target_var].values.flatten()

        # Filter to valid stable regime (Ri >= 0)
        valid_mask = np.isfinite(ri_ref) & np.isfinite(f_val) & (ri_ref >= 0)
        ri_plot = ri_ref[valid_mask]
        f_plot = f_val[valid_mask]

        # Sort along Ri axis to guarantee clean line rendering
        sort_idx = np.argsort(ri_plot)

        # Write down labels for legend
        if "fm" in target_var:
            label_text = fr"{name}, $\epsilon = {approx_scientific_notation(reference_height/SITE_CONFIGS[name].roughness_length_momentum)}$ " + \
                                 fr"$\alpha = {approx_scientific_notation(SITE_CONFIGS[name].roughness_length_momentum/SITE_CONFIGS[name].roughness_length_heat)}$" 
        else:
            label_text = fr"{name}, $\epsilon_t = {approx_scientific_notation(reference_height/SITE_CONFIGS[name].roughness_length_heat)}$ " + \
                                 fr"$\alpha = {approx_scientific_notation(SITE_CONFIGS[name].roughness_length_momentum/SITE_CONFIGS[name].roughness_length_heat)}$" 
        ax.plot(
            ri_plot[sort_idx],
            f_plot[sort_idx],
            color=ds_style["color"],
            linestyle=ds_style["linestyle"],
            linewidth=2,
            label=label_text,
        )

    y_label_str = fr"$f_m(z/L)_{{{int(reference_height)}}}$" if "fm" in target_var else fr"$f_h(z/L)_{{{int(reference_height)}}}$"
    ax.set_ylabel(fr"Normalised transfer coefficient {y_label_str}", fontsize=11)
    ax.set_xlabel(fr"Bulk Richardson $Ri_{{{int(reference_height)}}}$", fontsize=11)
    ax.set_title(
        fr"Normalised transfer coefficient ({y_label_str}) vs. $Ri_{{{int(reference_height)}}}$",
        fontsize=12,
    )
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")
    plt.tight_layout()

    return fig, ax

def plot_vertical_profile(
    ds_dict: dict[str, xr.Dataset],
    var_name: str,
    time: str | np.datetime64 | None = None,
    time_range: tuple[str, str] | slice | None = None,
    cmap_name: str = "viridis",
    style: list[dict[str,str]] = DATASET_STYLES
):
    """
    Plots vertical profiles (variable vs. height) across datasets. Either at a specific
    time or for a time range (using a consistent colormap)
    """
    if time is not None and time_range is not None:
        raise ValueError(
            "Specify either 'time' or 'time_range', not both."
        )

    fig, ax = plt.subplots(figsize=(7, 8))

    for idx, (name, ds) in enumerate(ds_dict.items()):
        ds_style = style[idx % len(style)]

        if var_name not in ds or "z" not in ds:
            raise KeyError(
                f"Variable '{var_name}' or height 'z' missing in dataset '{name}'."
            )

        # Case 1: Single timestamp plotting
        if time is not None:
            ds_sel = ds.sel(time=time, method="nearest")

            # Convert to 1D NumPy arrays to prevent coordinate indexing conflicts
            var_vals = ds_sel[var_name].values.flatten()
            z_vals = ds_sel["z"].values.flatten()

            # Filter valid numerical entries
            valid_mask = np.isfinite(var_vals) & np.isfinite(z_vals)
            sort_idx = np.argsort(z_vals[valid_mask])

            ax.plot(
                var_vals[valid_mask][sort_idx],
                z_vals[valid_mask][sort_idx],
                color=ds_style["color"],
                linestyle=ds_style["linestyle"],
                linewidth=2,
                label=f"{name}",
            )

        # Case 2: Time range (Colormap per Dataset / Time Increment)
        elif time_range is not None:
            if isinstance(time_range, tuple):
                t_slice = slice(time_range[0], time_range[1])
            else:
                t_slice = time_range

            ds_sel = ds.sel(time=t_slice)
            n_times = ds_sel.sizes["time"]

            if n_times == 0:
                raise ValueError(
                    f"No timesteps found in range {time_range} for dataset '{name}'."
                )

            # Generate colormap norm across selected timesteps
            cmap = plt.get_cmap(cmap_name)
            colors = [cmap(i) for i in np.linspace(0.2, 1.0, n_times)]

            for t_idx in range(n_times):
                ds_step = ds_sel.isel(time=t_idx)

                var_vals = ds_step[var_name].values.flatten()
                z_vals = ds_step["z"].values.flatten()

                valid_mask = np.isfinite(var_vals) & np.isfinite(z_vals)
                sort_idx = np.argsort(z_vals[valid_mask])

                lbl = (
                    f"{name} ({n_times} steps)"
                    if t_idx == 0 and len(ds_dict) > 1
                    else None
                )

                ax.plot(
                    var_vals[valid_mask][sort_idx],
                    z_vals[valid_mask][sort_idx],
                    color=colors[t_idx],
                    linestyle=ds_style["linestyle"],
                    linewidth=1.5,
                    alpha=0.7,
                    label=lbl,
                )

        # Case 3: Time-avgd mean profile (Fallback)
        else:
            var_mean = ds[var_name].mean(dim="time").values.flatten()
            z_mean = ds["z"].mean(dim="time").values.flatten()

            valid_mask = np.isfinite(var_mean) & np.isfinite(z_mean)
            sort_idx = np.argsort(z_mean[valid_mask])

            ax.plot(
                var_mean[valid_mask][sort_idx],
                z_mean[valid_mask][sort_idx],
                color=ds_style["color"],
                linestyle=ds_style["linestyle"],
                linewidth=2,
                label=f"{name} (Time Mean)",
            )

    # Adding a colorbar for the time range case
    if time_range is not None:
        norm = mcolors.Normalize(vmin=0, vmax=n_times - 1)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label("Timestep Index within Range", fontsize=10)

    title_suffix = (
        f" at t = {time}"
        if time
        else (f" (Range: {time_range})" if time_range else " (Time Mean)")
    )
    ax.set_xlabel(f"{var_name}", fontsize=11)
    ax.set_ylabel("Height AGL $z$ [m]", fontsize=11)
    ax.set_title(f"Vertical Profile: {var_name}{title_suffix}", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")
    plt.tight_layout()

    return fig, ax

def plot_abl_top_vs_surface_hexbin(
    ds_dict: dict[str, xr.Dataset],
    ds_srf_dict: dict[str, xr.Dataset],
    temp_var: str = "t",
    style: list[dict[str, str]] = DATASET_STYLES,
    gridsize: int = 40,
):
    """
    Plots hexbin distributions of Delta T (ABL Top - Surface)
    vs. Wind Speed at the ABL Top.
    One subplot is created for each dataset.
    """

    n_datasets = len(ds_dict)

    # Smallest square grid that contains all datasets
    N = int(np.ceil(np.sqrt(n_datasets)))
    fig, axes = plt.subplots(N, N, figsize=(6 * N, 5 * N), squeeze=False)

    axes = axes.flatten()

    for idx, (name, ds) in enumerate(ds_dict.items()):
        ax = axes[idx]
        ds_style = style[idx % len(style)]
        ds_srf = ds_srf_dict[name]

        # Compute Temperature Difference (ABL Top - Surface)
        delta_T = compute_difference_surface_top_ABL(
            ds,
            ds_srf,
            var_name=temp_var
        ).values

        # Extract Wind Speed at BLH for each timestep
        n_times = ds.sizes["time"]
        u_blh = np.zeros(n_times)

        for t in range(n_times):
            z_t = ds["z"].isel(time=t).values
            blh_t = ds_srf["blh"].isel(time=t).values
            k_idx = np.abs(z_t - blh_t).argmin()
            u_blh[t] = ds["wind_speed"].isel(
                time=t,
                model_level=k_idx,
            ).values

        # Filter NaN / Inf values
        valid_mask = (
            np.isfinite(u_blh)
            & np.isfinite(delta_T)
        )
        x_val = u_blh[valid_mask]
        y_val = delta_T[valid_mask]

        # Dataset-specific colormap:
        # low density -> white
        # high density -> DATASET_STYLES color
        cmap = LinearSegmentedColormap.from_list(
            f"{name}_cmap",
            ["white", ds_style["color"]],
        )

        # Hexbin distribution
        hb = ax.hexbin(
            x_val,
            y_val,
            gridsize=gridsize,
            mincnt=1,
            bins="log",
            cmap=cmap,
        )

        # Colorbar for this dataset
        cbar = fig.colorbar(hb, ax=ax)
        cbar.set_label("Number of observations")
        ax.set_xlabel("Wind Speed at BLH [m/s]", fontsize=11)
        ax.set_ylabel(f"$\\Delta {temp_var.upper()}$ (ABL Top - Surface) [K]",fontsize=11)
        ax.set_title(f"{name} (n={len(y_val)})", fontsize=12)
        ax.set_xlim(left=0)
        ax.grid(True, linestyle="--", alpha=0.3)

    # Remove unused panels if number of datasets is not a perfect square
    for idx in range(n_datasets, len(axes)):
        axes[idx].remove()
    fig.suptitle(
        "ABL Top Wind Speed vs. Temperature Difference",
        fontsize=14,
    )
    plt.tight_layout()

    return fig, axes