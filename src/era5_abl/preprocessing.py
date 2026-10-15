import xarray as xr
import numpy as np
import os, shutil, subprocess, tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from .config import (
    GRAVITY,
    DRY_AIR_GAS_CONSTANT,
    ERA5_RETRIEVAL_AREAS,
    SiteConfig,
    get_retrieval_area,
)
from .data_retrieve import model_level_paths, split_dates
from .gribpoint import NotAPointFile, RepeatedHours, read_point_grib
from .operations import compute_thetav


def compute_ecmwf_pressure_and_height(ds_ml: xr.Dataset, ds_srf: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Computes 3D/2D pressure [Pa] and height AGL [m] on ECMWF hybrid model levels
    using A_k and B_k coefficients embedded in the NetCDF dataset (from GRIB VCT).
    Heights are geopotential heights of the FULL model levels, computed as in the IFS
    (see ECMWF's compute_geopotential_on_ml.py). Built with help of AI.
    """
    g = GRAVITY
    R_d = DRY_AIR_GAS_CONSTANT

    # Extract surface pressure [Pa]
    if "lnsp" in ds_ml:
        sp = np.exp(ds_ml["lnsp"])
    elif "sp" in ds_srf:
        sp = ds_srf["sp"]
    elif "sp" in ds_ml:
        sp = ds_ml["sp"]
    else:
        raise KeyError("Could not find surface pressure ('lnsp' or 'sp') in datasets.")

    # Identify vertical dimension name and extract model level numbers
    lev_dim = (
        "model_level"
        if "model_level" in ds_ml.sizes
        else ("lev" if "lev" in ds_ml.sizes else ("hybrid" if "hybrid" in ds_ml.sizes else "level"))
    )
    levels = ds_ml[lev_dim].values.astype(int)  # e.g., array([110, 111, ..., 137])

    # Extract A and B hybrid coefficients
    if "hyai" in ds_ml and "hybi" in ds_ml:
        a_full = ds_ml["hyai"].values
        b_full = ds_ml["hybi"].values
    elif "vct" in ds_ml:
        vct = ds_ml["vct"].values
        n_half = len(vct) // 2
        a_full = vct[:n_half]
        b_full = vct[n_half:]
    else:
        raise ValueError(
            "Hybrid level coefficients (vct/hyai/hybi) not found in NetCDF dataset."
        )

    # Dynamic slicing of coefficients for levels present in ds_ml
    k_min = int(levels.min())
    k_max = int(levels.max())
    
    a_half = a_full[k_min - 1 : k_max + 1]  # Length: n_levels + 1 (e.g., 29)
    b_half = b_full[k_min - 1 : k_max + 1]  # Length: n_levels + 1

    # Compute half-level pressure: shape (time, model_level_half)
    # broadcasting sp (time) with a_half/b_half (model_level_half)
    sp_vals = sp.values[..., np.newaxis]  # (time, 1)
    p_half = a_half + b_half * sp_vals  # (time, 138)

    # Full-level pressure: midpoint of half-levels
    p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])  # (time, 137)

    # Compute virtual temperature (same constant as ECMWF, Rv/Rd - 1 = 0.609133)
    q = ds_ml["q"].values
    t = ds_ml["t"].values
    t_v = t * (1.0 + 0.609133 * q)

    # Hydrostatic integration from surface (TOA = level 0, Surface = level N-1), following
    # IFS documentation (Part III, Eqs. 2.20-2.23) and ECMWF's compute_geopotential_on_ml.py (https://github.com/tkschuler/ERA5-Utils/blob/main/compute_geopotential_on_ml_updated.py)
    # dln_p = ln(p_{k+1/2} / p_{k-1/2})
    dln_p = np.log(p_half[:, 1:] / p_half[:, :-1])
    dphi = R_d * t_v * dln_p  # geopotential increment across each layer (half level to half level)

    # Geopotential AGL at the UPPER half level of each layer (k-1/2): integrate from surface upwards
    phi_half_top = np.cumsum(dphi[:, ::-1], axis=1)[:, ::-1]
    phi_half_bottom = phi_half_top - dphi  # lower half level (k+1/2); = 0 at the surface

    # Full level k sits above the lower half level by alpha_k * R_d * T_v,k
    alpha = 1.0 - p_half[:, :-1] / (p_half[:, 1:] - p_half[:, :-1]) * dln_p  # (only valid for k > 1, where alpha_1 = ln 2)
    phi_full = phi_half_bottom + alpha * R_d * t_v
    z_agl_vals = phi_full / g  # Height AGL of full model levels [m]

    # Format as xarray DataArrays matching ds_ml dimensions
    time_dim = "time" if "time" in ds_ml.sizes else "valid_time"

    pressure_da = xr.DataArray(
        p_full,
        dims=[time_dim, lev_dim],
        coords={time_dim: ds_ml[time_dim], lev_dim: ds_ml[lev_dim]},
        attrs={"units": "Pa", "long_name": "Pressure on model levels"},
    )

    z_agl_da = xr.DataArray(
        z_agl_vals,
        dims=[time_dim, lev_dim],
        coords={time_dim: ds_ml[time_dim], lev_dim: ds_ml[lev_dim]},
        attrs={
            "units": "m",
            "long_name": "Height Above Ground Level",
        },
    )

    return pressure_da, z_agl_da

def standardize_surface_varnames(ds: xr.Dataset) -> xr.Dataset:
    """
    Maps ECMWF GRIB ParamIDs (varXXX) to standard short names in an xarray Dataset
    and updates variable 'long_name' attributes to full CDS variable descriptions.
    """
    param_info = {
        "var34": {"short_name": "sst", "long_name": "Sea surface temperature"},
        "var167": {"short_name": "t2m", "long_name": "2 metre temperature"},
        "var134": {"short_name": "sp", "long_name": "Surface pressure"},
        "var23": {"short_name": "cbh", "long_name": "Cloud base height"},
        "var188": {"short_name": "hcc", "long_name": "High cloud cover"},
        "var186": {"short_name": "lcc", "long_name": "Low cloud cover"},
        "var187": {"short_name": "mcc", "long_name": "Medium cloud cover"},
        "var159": {"short_name": "blh", "long_name": "Boundary layer height"},
        "var129": {"short_name": "z", "long_name": "Surface geopotential"},
        "var168": {"short_name": "2d", "long_name": "2 metre dewpoint temperature"},
        "var244": {"short_name": "fsr", "long_name": "Forecast roughness length for momentum"},
        "var245": {"short_name": "flsr", "long_name": "Log of forecast roughness length for heat"},
    }

    # Update long_name attributes for present variables prior to renaming
    for var_id, info in param_info.items():
        if var_id in ds.data_vars:
            ds[var_id].attrs["long_name"] = info["long_name"]

    # Rename varXXX to short_name
    rename_dict = {
        var_id: info["short_name"]
        for var_id, info in param_info.items()
        if var_id in ds.data_vars
    }

    return ds.rename(rename_dict)

def site_folder_name(site: SiteConfig) -> str:
    """'MaceHead' from the file name 'MaceHead_lvls.grib'."""
    return site.model_level_filename.split("_lvls")[0]

def site_piece_path(site: SiteConfig, piece: str, data_dir: str | Path) -> Path:
    """
    Path of ONE piece of one site, cut out of the file of its retrieval area, e.g.
    data_dir/raw_lvls_data/MaceHead/MaceHead_lvls_20200101-20200115.grib
    """
    folder = site_folder_name(site)
    name = f"{folder}_lvls_{piece.replace('-', '').replace('/', '-')}.grib"
    return Path(data_dir) / "raw_lvls_data" / folder / name

def site_piece_paths(site: SiteConfig, dates: str, data_dir: str | Path) -> list[Path]:
    """Paths of all the pieces of one site (one per piece of data_retrieve.split_dates)."""
    return [site_piece_path(site, piece, data_dir) for piece in split_dates(dates)]

def extract_sites_from_file(
    area_path: str | Path,
    area_name: str,
    piece: str,
    site_configs: dict,
    data_dir: str | Path,
    delete_area_file: bool = False,
    max_workers: int = 4,
) -> list[Path]:
    """
    Cuts out of ONE downloaded file (one retrieval area, one piece of dates) the grid point(s) of every selected
    site that lies in that area, via CDO. Called by scripts/download_era5.py right after each download:
    with delete_area_file the big file is removed as soon as the sites are extracted, so that the whole download
    needs little disk (only the sites are kept).
    """
    area_path = Path(area_path)
    sites = {name: site for name, site in site_configs.items() if get_retrieval_area(site) == area_name}
    area_box = [float(x) for x in ERA5_RETRIEVAL_AREAS[area_name].split("/")]

    def extract_site(item):
        name, site = item
        site_path = site_piece_path(site, piece, data_dir)
        site_path.parent.mkdir(parents=True, exist_ok=True)
        if site_path.exists():
            return site_path
        tmp_path = site_path.with_name(site_path.name + ".part")
        if [float(x) for x in site.area.split("/")] == area_box:
            # The area is the site itself: nothing to select (and CDO cannot select from a single grid point)
            shutil.copy(area_path, tmp_path)
        else:
            # CDO wants lon1,lon2,lat1,lat2. The small margin keeps the grid points on the edge of the box
            north, west, south, east = (float(x) for x in site.area.split("/"))
            box = f"{west - 0.01},{east + 0.01},{south - 0.01},{north + 0.01}"
            cmd = f'cdo -s sellonlatbox,{box} "{area_path}" "{tmp_path}"'
            subprocess.run(cmd, shell=True, check=True)
        tmp_path.replace(site_path)
        return site_path

    print(f"Extracting {len(sites)} sites from {area_path.name}: {', '.join(sites)}")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        site_paths = list(executor.map(extract_site, sites.items()))

    if delete_area_file:
        area_path.unlink()
    return site_paths

def extract_site_data(
    site: SiteConfig,
    dates: str,
    data_dir: str | Path,
    output_path: str | Path,
    max_workers: int = 4,
) -> None:
    """
    Merges in time (CDO mergetime) the pieces of one site, into one GRIB file for the whole period, as if
    site.area had been requested directly from the CDS. The pieces that are not there yet are cut out of the
    files of the retrieval area (extract_sites_from_file), so this also works if the area files were downloaded
    without extracting the sites along the way.
    """
    area_name = get_retrieval_area(site)
    area_paths = model_level_paths(area_name, dates, data_dir)
    site_paths = site_piece_paths(site, dates, data_dir)

    # Cut out the pieces that are missing (nothing to do if they were already extracted after each download)
    for area_path, site_path, piece in zip(area_paths, site_paths, split_dates(dates)):
        if site_path.exists():
            continue
        if not area_path.exists():
            raise FileNotFoundError(
                f"Neither the piece {site_path.name} nor the file {area_path.name} of the area {area_name} is "
                f"there: retrieve it first (and note that the dates must cover whole months)."
            )
        extract_sites_from_file(area_path, area_name, piece, {site_folder_name(site): site}, data_dir,
                                max_workers=max_workers)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = "cdo -O mergetime " + " ".join(f'"{path}"' for path in site_paths) + f' "{output_path}"'
    print(f"Executing: cdo -O mergetime [{len(site_paths)} pieces of {site_folder_name(site)}] \"{output_path}\"")
    subprocess.run(cmd, shell=True, check=True)

def load_with_cdo(grib_path: str | Path, cdo_mean: bool = False) -> xr.Dataset:
    """
    Converts a GRIB file with CDO (-f nc4 copy, or fldmean) and returns it as a Dataset held in MEMORY.
    CDO writes to a private temporary folder that is removed in every case, even after an error: nothing is left
    on disk, no file handle stays open, and two runs can never meet on the same file (the cause of the
    "permission denied" / 0 B _CDO_processed_tmp.nc of earlier versions).
    """
    operator = "fldmean" if cdo_mean else "copy"
    with tempfile.TemporaryDirectory(prefix="era5_abl_cdo_") as tmp_dir:
        nc_path = Path(tmp_dir) / (Path(grib_path).stem + ".nc")
        cmd = ["cdo", "-f", "nc4", operator, str(grib_path), str(nc_path)]
        print("Executing:", " ".join(cmd), flush=True)
        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError as error:
            raise RuntimeError("CDO was not found: install it (e.g. `conda install -c conda-forge cdo`) "
                               "and make sure `cdo` is on the PATH.") from error
        with xr.open_dataset(nc_path, engine="netcdf4") as ds:
            return ds.load()


def load_model_levels(grib_ml_path: str | Path, cdo_mean: bool = False, fast: bool = True) -> xr.Dataset:
    """
    The model level file as a Dataset (t, q, u, v on 'lev', the hybrid coefficients, time).
    The files of single sites (the usual case) are read directly from the GRIB messages (gribpoint.py: ~10
    seconds instead of 20-30 minutes in CDO, identical values). Anything else, e.g. a file with several grid
    points that must be averaged (cdo_mean), goes through CDO.
    """
    if fast and not cdo_mean:
        try:
            ds = read_point_grib(grib_ml_path)
            print(f"Read {Path(grib_ml_path).name} directly ({ds.sizes['time']} hours), without CDO.", flush=True)
            return ds
        except NotAPointFile as error:
            print(f"{error}\n--> Using CDO instead (slower).", flush=True)
    return load_with_cdo(grib_ml_path, cdo_mean)


def check_time_axes(ds_ml: xr.Dataset, ds_srf: xr.Dataset, ml_path, srf_path) -> None:
    """
    Stops, with an explanation, if the level and surface data are not for exactly the same hours: the pressure
    and height are computed hour by hour from both, so they must line up.
    """
    problems = []
    def hours(ds):
        return ds["time" if "time" in ds.coords else "valid_time"].to_index()

    for label, ds, path in (("level", ds_ml, ml_path), ("surface", ds_srf, srf_path)):
        index = hours(ds)
        repeated = index[index.duplicated()].unique()
        if len(repeated):
            shown = ", ".join(f"{t:%Y-%m-%d %H:%M}" for t in repeated[:5])
            problems.append(f"the {label} file {Path(path).name} has {len(repeated)} hour(s) more than once "
                            f"({shown}{', ...' if len(repeated) > 5 else ''}): run scripts/fix_repeated_hours.py")
        elif not index.is_monotonic_increasing:
            problems.append(f"the hours of the {label} file {Path(path).name} are not in time order")
    if not problems:
        t_ml, t_srf = hours(ds_ml), hours(ds_srf)
        only_ml, only_srf = t_ml.difference(t_srf), t_srf.difference(t_ml)
        if len(only_ml) or len(only_srf):
            def span(index):
                return f"{index.min():%Y-%m-%d %H:%M} to {index.max():%Y-%m-%d %H:%M}" if len(index) else "none"
            problems.append(
                f"the level file ({len(t_ml)} hours, {span(t_ml)}) and the surface file ({len(t_srf)} hours, "
                f"{span(t_srf)}) are not for the same hours: {len(only_ml)} hour(s) only in the level file "
                f"({span(only_ml)}), {len(only_srf)} only in the surface file ({span(only_srf)}). "
                f"Check that both were downloaded for the same dates."
            )
    if problems:
        raise ValueError("Cannot combine the level and surface data:\n  - " + "\n  - ".join(problems))


def prepare_dataset(grib_ml_path: str, grib_srf_path: str, location: str = None, cdo_mean: bool = False,
                    fast_reader: bool = True) -> tuple[xr.Dataset, xr.Dataset]:
    """
    Reads the model level and surface GRIB files (the model level file directly if it is a single site, see
    load_model_levels; everything else through CDO), explicitly computes 3D pressure p(t, k) and hydrostatic
    height z_agl(t, k) from hybrid coefficients, derives theta_v, and reverses the vertical axis so k=0 is
    the surface.
    fast_reader=False forces CDO for the level file (slow; only to compare with the direct reader).
    """

    ds_ml = load_model_levels(grib_ml_path, cdo_mean=cdo_mean, fast=fast_reader)
    ds_srf = load_with_cdo(grib_srf_path, cdo_mean=cdo_mean)

    # The hours of both files must be the same, otherwise stop here with an explanation
    check_time_axes(ds_ml, ds_srf, grib_ml_path, grib_srf_path)

    # Standardize dimension names
    dim_map = {}
    if "valid_time" in ds_ml.sizes:
        dim_map["valid_time"] = "time"
    if "hybrid" in ds_ml.sizes:
        dim_map["hybrid"] = "model_level"
    elif "lev" in ds_ml.sizes:
        dim_map["lev"] = "model_level"

    if dim_map:
        ds_ml = ds_ml.rename(dim_map)
        ds_srf = ds_srf.rename({k: v for k, v in dim_map.items() if k in ds_srf.sizes})

    # Strip residual scalar lat/lon dimensions
    ds_ml = ds_ml.squeeze(drop=True)
    ds_srf = ds_srf.squeeze(drop=True)

    # Standardise surface dataset variable names
    ds_srf = standardize_surface_varnames(ds_srf)

    # Compute pressure [Pa] and height AGL [m] from hybrid coefficients
    p_da, z_agl_da = compute_ecmwf_pressure_and_height(ds_ml, ds_srf)

    # Assign pressure as a dataset variable and height AGL as a 2D coordinate
    ds_ml["pressure"] = p_da
    ds_ml = ds_ml.assign_coords(z=(("time", "model_level"), z_agl_da.values))

    # Drop hybrid coefficients (otherwise xarray's .where() in the
    # filters broadcasts them along time and model_level, enlarging filesize)
    ds_ml = ds_ml.drop_vars(["hyai", "hybi", "hyam", "hybm"], errors="ignore")

    # Compute virtual potential temperature
    ds_ml = compute_thetav(ds_ml)
    ds_srf = compute_thetav(ds_srf)

    # Compute wind speed 
    ds_ml["wind_speed"] = np.sqrt(ds_ml["u"]**2 + ds_ml["v"]**2)

    # Roughness lengths are stored as log-values for heat; convert back to physical units.
    ds_srf = ds_srf.rename({"fsr": "z0"})
    ds_srf["z0"].attrs.update({"long_name": "Roughness length for momentum", "units": "m"})
    ds_srf["z0h"] = np.exp(ds_srf["flsr"])
    ds_srf["z0h"].attrs.update({"long_name": "Roughness length for heat", "units": "m"})
    ds_srf = ds_srf.drop_vars(["flsr"])


    # Reindex vertical axis so k=0 is Surface and k=N-1 is Top of Atmosphere
    ds_ml = ds_ml.reindex(model_level=ds_ml.model_level[::-1])

    # Add location metadata
    if location:
        ds_ml.attrs["Location"] = location
        ds_srf.attrs["Location"] = location

    return ds_ml, ds_srf
