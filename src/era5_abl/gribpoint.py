"""
Fast reading of GRIB files in which every message holds ONE grid point (the per-site files made by
extract_sites_from_file / extract_site_data).

Why this exists
---------------
Such a file is almost all header: 1283 bytes of metadata around a single number, ~2 million messages for two
years of hourly data on 28 levels and 4 variables. CDO handles every message through the full ecCodes machinery
(~1 ms each), so `cdo -f nc4 copy` on one site takes 20-30 minutes, and prepare_dataset does it for the level
file of every site. Reading the few header fields we need and the single value directly takes ~15 seconds and
returns the same numbers, bit for bit (tests/test_gribpoint.py compares both on real files).

What it supports (everything else raises NotAPointFile, and prepare_dataset then falls back to CDO)
  * GRIB edition 2, grid template 3.0, one data point per message,
  * product template 4.0 (analysis) on hybrid levels, simple packing (template 5.0), no bitmap,
  * the parameters in PARAMS below (t, q, u, v, w, vo, d), all on the same set of levels.

It also refuses, with RepeatedHours, a file in which an hour is present twice: CDO would turn those into extra
time steps and the level and surface data would no longer line up (that is exactly what happened to the first
piece of January 2021).
"""

from __future__ import annotations

import mmap
import struct
from array import array
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


class NotAPointFile(Exception):
    """The file is not of the simple kind this reader handles: use CDO instead."""


class RepeatedHours(ValueError):
    """The same hour is present more than once in the file."""

    def __init__(self, repeated: list, path):
        self.repeated = repeated            # [(timestamp, how many times)]
        self.path = path
        shown = ", ".join(f"{t:%Y-%m-%d %H:%M} (x{n})" for t, n in repeated[:6])
        more = f" and {len(repeated) - 6} more" if len(repeated) > 6 else ""
        super().__init__(
            f"{Path(path).name}: {len(repeated)} hour(s) appear more than once: {shown}{more}. "
            f"CDO would read them as extra time steps (the level and surface data would then differ in length). "
            f"Remove the repeated messages with scripts/fix_repeated_hours.py."
        )


# (discipline, category, number) -> (name, standard_name, long_name, units), as CDO names them in its netCDF
PARAMS = {
    (0, 0, 0): ("t", "air_temperature", "Temperature", "K"),
    (0, 1, 0): ("q", "specific_humidity", "Specific humidity", "kg kg**-1"),
    (0, 2, 2): ("u", "eastward_wind", "U component of wind", "m s**-1"),
    (0, 2, 3): ("v", "northward_wind", "V component of wind", "m s**-1"),
    (0, 2, 8): ("w", "lagrangian_tendency_of_air_pressure", "Vertical velocity", "Pa s**-1"),
    (0, 2, 12): ("vo", "atmosphere_relative_vorticity", "Vorticity (relative)", "s**-1"),
    (0, 2, 13): ("d", "divergence_of_wind", "Divergence", "s**-1"),
}
HYBRID_LEVEL = 105          # GRIB2 code table 4.5: hybrid level

_SECTION = struct.Struct(">IB")
_U16, _U32 = struct.Struct(">H"), struct.Struct(">I")
_F32 = struct.Struct(">f")
_UNIT = {0: 60, 1: 3600, 2: 86400, 10: 3 * 3600, 11: 6 * 3600, 12: 12 * 3600, 13: 1}     # seconds per unit


def _sign_magnitude(value: int, bits: int) -> int:
    """GRIB stores signed scale factors as sign + magnitude, not two's complement."""
    sign = 1 << (bits - 1)
    return -(value & (sign - 1)) if value & sign else value


def _read_point_messages(buf, path):
    """Yields (valid time as seconds since 1970, param key, level, value, grid, pv bytes) for every message."""
    size = len(buf)
    pos = 0
    while pos < size:
        if buf[pos:pos + 4] != b"GRIB":
            raise NotAPointFile(f"{Path(path).name}: not a plain sequence of GRIB messages (byte {pos})")
        if buf[pos + 7] != 2:
            raise NotAPointFile(f"{Path(path).name}: GRIB edition {buf[pos + 7]}, only edition 2 is read directly")
        total = int.from_bytes(buf[pos + 8:pos + 16], "big")
        if pos + total > size:
            raise NotAPointFile(f"{Path(path).name}: the last message is cut short (the file is incomplete)")
        discipline = buf[pos + 6]
        p = pos + 16
        end = pos + total - 4
        reference = grid = key = level = pv = value = None
        nbits = scale_d = 0
        r_value = e_scale = 0
        data_start = data_end = 0
        while p < end:
            length, number = _SECTION.unpack_from(buf, p)
            if number == 1:
                reference = datetime(_U16.unpack_from(buf, p + 12)[0], buf[p + 14], buf[p + 15],
                                     buf[p + 16], buf[p + 17], buf[p + 18])
            elif number == 3:
                if _U32.unpack_from(buf, p + 6)[0] != 1:
                    raise NotAPointFile(f"{Path(path).name}: a message holds more than one grid point "
                                        f"(this reader is for the single-site files)")
                if _U16.unpack_from(buf, p + 12)[0] != 0:
                    raise NotAPointFile(f"{Path(path).name}: grid template other than lat/lon")
                grid = bytes(buf[p + 46:p + 54])        # latitude and longitude of the point
            elif number == 4:
                if _U16.unpack_from(buf, p + 7)[0] != 0:
                    raise NotAPointFile(f"{Path(path).name}: product template other than 4.0")
                if buf[p + 22] != HYBRID_LEVEL or buf[p + 23] != 0:
                    raise NotAPointFile(f"{Path(path).name}: not a hybrid (model) level field")
                param = (discipline, buf[p + 9], buf[p + 10])
                step = _U32.unpack_from(buf, p + 18)[0] * _UNIT.get(buf[p + 17], 3600)
                key = param
                level = _U32.unpack_from(buf, p + 24)[0]
                pv = bytes(buf[p + 34:p + length])
                reference_step = step
            elif number == 5:
                if _U16.unpack_from(buf, p + 9)[0] != 0:
                    raise NotAPointFile(f"{Path(path).name}: data packing other than simple")
                r_value = _F32.unpack_from(buf, p + 11)[0]
                e_scale = _sign_magnitude(_U16.unpack_from(buf, p + 15)[0], 16)
                scale_d = _sign_magnitude(_U16.unpack_from(buf, p + 17)[0], 16)
                nbits = buf[p + 19]
            elif number == 6:
                if buf[p + 5] != 255:
                    raise NotAPointFile(f"{Path(path).name}: the fields have a bitmap")
            elif number == 7:
                data_start, data_end = p + 5, p + length
            p += length
        if None in (reference, grid, key, level) or not data_end:
            raise NotAPointFile(f"{Path(path).name}: incomplete message at byte {pos}")

        if nbits == 0:
            if scale_d:
                raise NotAPointFile(f"{Path(path).name}: constant field with a decimal scale factor")
            value = r_value
        else:
            raw = int.from_bytes(buf[data_start:data_end], "big") >> (8 * (data_end - data_start) - nbits)
            value = (r_value + raw * 2.0 ** e_scale) * 10.0 ** (-scale_d)

        valid = reference + timedelta(seconds=reference_step)
        yield valid, key, level, value, grid, pv
        pos += total


def read_point_arrays(path: str | Path) -> dict:
    """
    The parsing half of read_point_grib (numpy only): {"times", "levels", "keys", "cube" (time, var, level),
    "lat", "lon", "a_half", "b_half"}. Same exceptions.
    """
    path = Path(path)
    if path.stat().st_size == 0:
        raise NotAPointFile(f"{path.name} is empty")

    seconds, var_idx, level_val, values = array("q"), array("b"), array("I"), array("f")
    var_keys: list = []
    first_grid = first_pv = None

    with open(path, "rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as buf:
        for valid, key, level, value, grid, pv in _read_point_messages(buf, path):
            if key not in PARAMS:
                raise NotAPointFile(f"{path.name}: parameter {key} is not one this reader knows")
            if first_grid is None:
                first_grid, first_pv = grid, pv
            elif grid != first_grid:
                raise NotAPointFile(f"{path.name}: the messages are not all for the same grid point")
            elif pv != first_pv:
                raise NotAPointFile(f"{path.name}: the hybrid coefficients change between messages")
            if key not in var_keys:
                var_keys.append(key)
            seconds.append(int((valid - datetime(1970, 1, 1)).total_seconds()))
            var_idx.append(var_keys.index(key))
            level_val.append(level)
            values.append(value)

    n = len(seconds)
    if n == 0:
        raise NotAPointFile(f"{path.name} has no messages")
    seconds = np.frombuffer(seconds, dtype=np.int64)
    var_idx = np.frombuffer(var_idx, dtype=np.int8)
    level_val = np.frombuffer(level_val, dtype=np.uint32)
    values = np.frombuffer(values, dtype=np.float32)

    times = np.unique(seconds)
    levels = np.unique(level_val)
    n_vars = len(var_keys)

    # Every (variable, level, hour) exactly once
    cell = (np.searchsorted(times, seconds) * n_vars + var_idx) * len(levels) + np.searchsorted(levels, level_val)
    counts = np.bincount(cell, minlength=len(times) * n_vars * len(levels))
    if counts.max() > 1:
        per_hour = np.bincount(np.searchsorted(times, seconds), minlength=len(times))
        expected = n_vars * len(levels)
        repeated = [(datetime(1970, 1, 1) + timedelta(seconds=int(times[i])), int(round(per_hour[i] / expected)))
                    for i in np.nonzero(per_hour > expected)[0]]
        raise RepeatedHours(repeated or [(datetime(1970, 1, 1) + timedelta(seconds=int(times[0])), 2)], path)
    if counts.min() == 0:
        raise NotAPointFile(f"{path.name}: some (hour, variable, level) combinations are missing: "
                            f"the file is incomplete or the variables do not share the same levels")

    cube = np.empty(len(times) * n_vars * len(levels), dtype=np.float32)
    cube[cell] = values
    cube = cube.reshape(len(times), n_vars, len(levels))

    lat = _sign_magnitude(int.from_bytes(first_grid[:4], "big"), 32) / 1e6
    lon = int.from_bytes(first_grid[4:8], "big") / 1e6          # 0..360, as CDO writes it

    n_coeff = len(first_pv) // 4
    coeff = np.frombuffer(first_pv, dtype=">f4").astype(np.float64)
    a_half, b_half = coeff[:n_coeff // 2], coeff[n_coeff // 2:]

    return {"times": times, "levels": levels, "keys": var_keys, "cube": cube, "lat": lat, "lon": lon,
            "a_half": a_half, "b_half": b_half}


def read_point_grib(path: str | Path):
    """
    Reads a single-site model level GRIB file into the xarray Dataset that `cdo -f nc4 copy` + xarray would give:
    t, q, u, v ... as (time, lev, lat, lon), the hybrid coefficients hyai/hybi/hyam/hybm, and the coordinates.
    Raises NotAPointFile if the file is not of the simple kind (use CDO then) and RepeatedHours if an hour is
    present twice.
    """
    import xarray as xr

    path = Path(path)
    a = read_point_arrays(path)
    times, levels, var_keys, cube = a["times"], a["levels"], a["keys"], a["cube"]
    lat, lon, a_half, b_half = a["lat"], a["lon"], a["a_half"], a["b_half"]
    time_index = (np.datetime64("1970-01-01T00:00:00", "ns") + times.astype("timedelta64[s]")).astype("datetime64[ns]")
    data_vars = {}
    for i, key in enumerate(var_keys):
        name, standard_name, long_name, units = PARAMS[key]
        data_vars[name] = (
            ("time", "lev", "lat", "lon"),
            cube[:, i, :, None, None],
            {"standard_name": standard_name, "long_name": long_name, "units": units,
             "param": f"{key[2]}.{key[1]}.{key[0]}"},
        )
    data_vars["hyai"] = (("nhyi",), a_half, {"long_name": "hybrid A coefficient at layer interfaces", "units": "Pa"})
    data_vars["hybi"] = (("nhyi",), b_half, {"long_name": "hybrid B coefficient at layer interfaces", "units": "1"})
    data_vars["hyam"] = (("nhym",), 0.5 * (a_half[:-1] + a_half[1:]),
                         {"long_name": "hybrid A coefficient at layer midpoints", "units": "Pa"})
    data_vars["hybm"] = (("nhym",), 0.5 * (b_half[:-1] + b_half[1:]),
                         {"long_name": "hybrid B coefficient at layer midpoints", "units": "1"})
    ds = xr.Dataset(
        data_vars,
        coords={
            "time": ("time", time_index, {"standard_name": "time", "axis": "T"}),
            "lev": ("lev", levels.astype(np.float64),
                    {"standard_name": "hybrid_sigma_pressure", "long_name": "hybrid level at layer midpoints",
                     "formula": "hyam hybm (mlev=hyam+hybm*aps)",
                     "formula_terms": "ap: hyam b: hybm ps: aps", "units": "level", "positive": "down"}),
            "lat": ("lat", [lat], {"standard_name": "latitude", "long_name": "latitude",
                                   "units": "degrees_north", "axis": "Y"}),
            "lon": ("lon", [lon], {"standard_name": "longitude", "long_name": "longitude",
                                   "units": "degrees_east", "axis": "X"}),
        },
        attrs={"Conventions": "CF-1.6", "institution": "European Centre for Medium-Range Weather Forecasts",
               "source": f"{path.name} (read directly from the GRIB messages, not through CDO)"},
    )
    return ds


# ----------------------------------------------------------------------------------------------------------
# Repeated messages
# ----------------------------------------------------------------------------------------------------------
# These work on any GRIB file (edition 1 or 2, one field per message), not only on single-site ones.

_UNITS_1 = {0: 60, 1: 3600, 2: 86400, 10: 3 * 3600, 11: 6 * 3600, 12: 12 * 3600, 13: 900, 14: 1800, 254: 1}


def _header(buf, pos: int):
    """(length, valid time, field key) of the message that starts at pos: the whole identity of a field."""
    edition = buf[pos + 7]
    if edition == 2:
        total = int.from_bytes(buf[pos + 8:pos + 16], "big")
        p, end, reference = pos + 16, pos + total, None
        while p + 5 <= end:
            length, number = _SECTION.unpack_from(buf, p)
            if length < 5:
                raise ValueError("corrupt section length")
            if number == 1:
                reference = datetime(_U16.unpack_from(buf, p + 12)[0], buf[p + 14], buf[p + 15],
                                     buf[p + 16], buf[p + 17], buf[p + 18])
            elif number == 4:
                step = _U32.unpack_from(buf, p + 18)[0] * _UNIT.get(buf[p + 17], 3600)
                key = (buf[pos + 6], buf[p + 9], buf[p + 10], buf[p + 22], _U32.unpack_from(buf, p + 24)[0])
                return total, reference + timedelta(seconds=step), key
            p += length
        raise ValueError("no product definition section")
    if edition == 1:
        total = int.from_bytes(buf[pos + 4:pos + 7], "big")
        pds = pos + 8
        year = (buf[pds + 24] - 1) * 100 + buf[pds + 12]
        reference = datetime(year, buf[pds + 13], buf[pds + 14], buf[pds + 15], buf[pds + 16])
        unit = _UNITS_1.get(buf[pds + 17], 3600)
        p1, p2, time_range = buf[pds + 18], buf[pds + 19], buf[pds + 20]
        offset = p2 if time_range in (2, 3, 4, 5) else p1
        key = (buf[pds + 3], buf[pds + 8], buf[pds + 9], _U16.unpack_from(buf, pds + 10)[0])
        return total, reference + timedelta(seconds=offset * unit), key
    raise ValueError(f"unknown GRIB edition {edition}")


def find_repeated_messages(path: str | Path) -> dict:
    """
    {"messages": n, "repeated": [(timestamp, field key, [byte offset of each copy])]} for the fields that occur
    more than once in the file, and "unordered": True if the hours do not follow each other in time order.
    """
    path = Path(path)
    seen: dict = {}
    order_ok, last = True, None
    messages = 0
    if path.stat().st_size == 0:
        return {"messages": 0, "repeated": [], "unordered": False}
    with open(path, "rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as buf:
        pos, size = 0, len(buf)
        while pos < size:
            if buf[pos:pos + 4] != b"GRIB":
                raise ValueError(f"{path.name}: not a plain sequence of GRIB messages (byte {pos})")
            total, valid, key = _header(buf, pos)
            if pos + total > size:
                raise ValueError(f"{path.name}: the last message is cut short (byte {pos}): file incomplete")
            seen.setdefault((valid, key), []).append(pos)
            if last is not None and valid < last:
                order_ok = False
            last = valid
            messages += 1
            pos += total
    repeated = [(v, k, offs) for (v, k), offs in seen.items() if len(offs) > 1]
    repeated.sort(key=lambda r: (r[0], r[1]))
    return {"messages": messages, "repeated": repeated, "unordered": not order_ok}


def drop_repeated_messages(src: str | Path, dst: str | Path) -> dict:
    """
    Copies src to dst keeping only the FIRST message of every (hour, field). Returns {"kept", "dropped"}.
    dst is written under a temporary name and renamed when complete, so an interruption never leaves a
    half-written file under the final name.
    """
    import os

    src, dst = Path(src), Path(dst)
    tmp = dst.with_name(dst.name + ".part")
    seen, kept, dropped = set(), 0, 0
    with open(src, "rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as buf, \
            open(tmp, "wb") as out:
        pos, size = 0, len(buf)
        while pos < size:
            if buf[pos:pos + 4] != b"GRIB":
                raise ValueError(f"{src.name}: not a plain sequence of GRIB messages (byte {pos})")
            total, valid, key = _header(buf, pos)
            if pos + total > size:
                raise ValueError(f"{src.name}: the last message is cut short (byte {pos}): file incomplete")
            if (valid, key) in seen:
                dropped += 1
            else:
                seen.add((valid, key))
                out.write(buf[pos:pos + total])
                kept += 1
            pos += total
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, dst)
    return {"kept": kept, "dropped": dropped}


def scan_hours(path: str | Path) -> dict:
    """
    What a GRIB file contains hour by hour, from the message headers only (seconds, not the 20 minutes cfgrib
    or CDO need on a two-year site file): 
      {"messages", "fields" (distinct fields in the file), "hours": {timestamp: number of messages},
       "present": {timestamp: number of DISTINCT fields}, "unordered": True if time goes backwards}
    An hour is complete if hours[t] == present[t] == fields; hours[t] > present[t] means a field is there twice.
    """
    path = Path(path)
    bit_of: dict = {}
    hours: dict = {}
    masks: dict = {}
    messages, last, unordered = 0, None, False
    if path.stat().st_size == 0:
        return {"messages": 0, "fields": 0, "hours": {}, "present": {}, "unordered": False}
    with open(path, "rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as buf:
        pos, size = 0, len(buf)
        while pos < size:
            if buf[pos:pos + 4] != b"GRIB":
                raise ValueError(f"{path.name}: not a plain sequence of GRIB messages (byte {pos})")
            total, valid, key = _header(buf, pos)
            if pos + total > size:
                raise ValueError(f"{path.name}: the last message is cut short (byte {pos}): file incomplete")
            bit = bit_of.setdefault(key, len(bit_of))
            hours[valid] = hours.get(valid, 0) + 1
            masks[valid] = masks.get(valid, 0) | (1 << bit)
            if last is not None and valid < last:
                unordered = True
            last = valid
            messages += 1
            pos += total
    return {"messages": messages, "fields": len(bit_of), "hours": hours,
            "present": {t: bin(m).count("1") for t, m in masks.items()}, "unordered": unordered}
