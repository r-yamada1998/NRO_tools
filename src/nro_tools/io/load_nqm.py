from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import xarray as xr
from astropy.io import fits
from astropy.table import Table

from nro_tools.types import Scans


def _bintable_to_xr(hdu: fits.BinTableHDU) -> xr.Dataset:
    """BinTableHDU -> xr.Dataset (handles multi-dimensional columns)"""
    t = Table(hdu.data)

    data_vars: Dict[str, Any] = {}
    coords: Dict[str, Any] = {"row": np.arange(len(t))}

    for name in t.colnames:
        col = t[name]
        arr = np.asarray(col)

        # masked -> NaN/None
        if hasattr(col, "mask"):
            if np.issubdtype(arr.dtype, np.number):
                arr = np.where(col.mask, np.nan, arr)
            else:
                arr = np.where(col.mask, None, arr)

        # bytes -> str (FITS often stores strings as bytes)
        if arr.dtype.kind == "S":
            arr = arr.astype(str)

        # dims: (row,) or (row, dim0, dim1, ...)
        if arr.ndim == 1:
            dims = ("row",)
        else:
            extra_dims = tuple(f"{name}_dim{i}" for i in range(arr.ndim - 1))
            for i, d in enumerate(extra_dims):
                coords.setdefault(d, np.arange(arr.shape[i + 1]))
            dims = ("row",) + extra_dims

        data_vars[name] = (dims, arr)

    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    ds.attrs["fits_header"] = dict(hdu.header)
    return ds


def _make_time_from_lavst(
    ds: xr.Dataset, *, lavst: str = "LAVST", out: str = "time"
) -> xr.Dataset:
    """
    Create time(row) from LAVST(row,6) = [Y,m,d,H,M,S].
    Returns a NEW Dataset view (non-deep copy).
    """
    if lavst not in ds:
        raise ValueError(f"'{lavst}' not found in dataset variables.")

    a = ds[lavst].values
    if a.ndim != 2 or a.shape[1] < 6:
        raise ValueError(f"'{lavst}' must have shape (row, >=6). Got {a.shape}.")

    df = pd.DataFrame(a[:, :6], columns=["Y", "m", "d", "H", "M", "S"])

    # seconds are assumed integer; if float seconds exist, adjust here.
    dt = pd.to_datetime(
        df.rename(
            columns={
                "Y": "year",
                "m": "month",
                "d": "day",
                "H": "hour",
                "M": "minute",
                "S": "second",
            }
        )
    ).to_numpy(dtype="datetime64[ns]")

    out_ds = ds.copy(deep=False).assign_coords({out: ("row", dt)})
    return out_ds


def _reshape_time_array(
    ds: xr.Dataset, *, time: str = "time", array: str = "ARRYT"
) -> xr.Dataset:
    """row -> (time, array) with dims (time, array, ...)"""
    if time not in ds.coords:
        raise ValueError(f"Coordinate '{time}' not found. Create it before reshaping.")
    if array not in ds:
        raise ValueError(f"Variable '{array}' not found.")

    out = ds.copy(deep=False)
    out[array] = out[array].astype(str).str.strip()

    out = out.set_index(row=[time, array]).unstack("row").rename({array: "array"})
    return out


def load_scans(
    path: str | Path,
    *,
    ext: int = 0,
    time_from: str = "LAVST",
    array_name: str = "ARRYT",
    time_name: str = "time",
) -> Scans:
    """
    Load an NQM FITS BinTable into `Scans`.

    What this does (at load time):
      1) BinTableHDU -> xr.Dataset with 'row' dim
      2) build timestamp coord `time_name` from `time_from` (default: LAVST)
      3) reshape row -> (time, array) using `array_name` (default: ARRYT)

    Returns:
      Scans (immutable container from nro_tools/types.py)
    """
    path = Path(path)

    with fits.open(path, ignore_missing_simple=True) as hdul:
        hdu = hdul[ext]
        if not isinstance(hdu, fits.BinTableHDU):
            raise TypeError(f"HDU[{ext}] is not a BinTableHDU (got {type(hdu)}).")

        ds = _bintable_to_xr(hdu)
        ds = _make_time_from_lavst(ds, lavst=time_from, out=time_name)
        ds = _reshape_time_array(ds, time=time_name, array=array_name)

        primary_header: Optional[Dict[str, Any]] = dict(hdul[0].header)

    meta = {
        "loader": "nro_tools.io.load_nqm.load_scans",
        "path": str(path),
        "ext": ext,
        "time_from": time_from,
        "time_name": time_name,
        "array_name": array_name,
        "reshaped": "row -> (time, array)",
    }

    return Scans(path=path, ds=ds, primary_header=primary_header, meta=meta)
