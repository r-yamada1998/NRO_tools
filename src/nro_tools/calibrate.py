from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import xarray as xr
import astropy.units as u
from astropy.constants import c

from nro_tools.types import Scans, CalibratedScans


def calibrate(
    scans: Scans,
    mode: Tuple[str, ...] = ("intensity",),
    *,
    # intensity calibration
    ldata: str = "LDATA",
    sfctr: str = "SFCTR",
    adoff: str = "ADOFF",
    fill_value: int = np.int32(-2147483648),
    spec_name: str = "spec_cal",
    # axes
    freq_name: str = "freq",
    v_name: str = "v_los",
) -> CalibratedScans:
    """
    Scans -> CalibratedScans (non-mutating)

    mode:
      - "intensity": spec = LDATA * SFCTR + ADOFF
      - "frequency": add freq axis from FQCAL/CHCAL
      - "velocity":  add v_los axis from freq & FRQ0

    Notes:
      - Assumes numch is constant and equals nchan (your assumption).
      - Uses astropy.units as requested.
    """
    if not mode:
        raise ValueError("mode must contain at least one calibration step.")

    ds = scans.ds.copy(deep=False)
    performed = []

    # Normalize spectral axis name
    if "chan" not in ds.dims:
        if "LDATA_dim0" in ds.dims:
            ds = ds.rename_dims({"LDATA_dim0": "chan"}).assign_coords(
                chan=np.arange(ds.dims["LDATA_dim0"])
            )
        else:
            raise ValueError("No 'chan' dim and no 'LDATA_dim0' dim found.")

    # ---- intensity ----
    if "intensity" in mode:
        raw = ds[ldata].where(ds[ldata] != fill_value).astype("float32")
        ds[spec_name] = (
            raw * ds[sfctr].astype("float32") + ds[adoff].astype("float32")
        ).astype("float32")
        ds[spec_name].attrs.update({"description": f"{ldata} * {sfctr} + {adoff}"})
        performed.append("intensity")

    # ---- frequency / velocity ----
    need_freq = ("frequency" in mode) or ("velocity" in mode)
    if need_freq:
        nchan = ds.dims["chan"]
        t = np.linspace(0.0, 1.0, nchan, dtype=np.float64)

        # dims are (time, array, ncal)
        f0 = ds["FQCAL"].isel(FQCAL_dim0=0).astype("float64")
        f1 = ds["FQCAL"].isel(FQCAL_dim0=1).astype("float64")
        frq0 = ds["FRQ0"].astype("float64")

        freq = f0 + (f1 - f0) * xr.DataArray(t, dims=("chan",))

        # reverse by CHCAL direction (time,array)
        ch0 = ds["CHCAL"].isel(CHCAL_dim0=0)
        ch1 = ds["CHCAL"].isel(CHCAL_dim0=1)
        need_rev = ch0 > ch1
        freq_rev = freq.isel(chan=slice(None, None, -1))
        freq = xr.where(need_rev, freq_rev, freq)

        if "frequency" in mode:
            ds[freq_name] = freq
            ds[freq_name].attrs.update(
                {"units": "Hz", "long_name": "Observed frequency"}
            )
            performed.append("frequency")

        if "velocity" in mode:
            frq0v = frq0.values[..., None]  # -> (time, array, 1)
            freqv = freq.values  # -> (time, array, chan)

            v = (c * ((freqv - frq0v) * u.Hz) / (frq0v * u.Hz)).to(u.km / u.s).value

            ds[v_name] = xr.DataArray(v, dims=freq.dims)
            ds[v_name].attrs.update(
                {"units": "km/s", "long_name": "Line-of-sight velocity"}
            )
            performed.append("velocity")

        # coords化（存在するものだけ）
        setc = []
        if "frequency" in mode:
            setc.append(freq_name)
        if "velocity" in mode:
            setc.append(v_name)
        if setc:
            ds = ds.set_coords(setc)

    if not performed:
        raise ValueError(f"No calibration performed. mode={mode}")

    meta: Dict[str, Any] = {
        "performed": tuple(performed),
        "mode_requested": tuple(mode),
        "fill_value": int(fill_value),
        "intensity": {
            "enabled": "intensity" in performed,
            "spec_name": spec_name if "intensity" in performed else None,
            "formula": f"{ldata} * {sfctr} + {adoff}",
            "ldata": ldata,
            "sfctr": sfctr,
            "adoff": adoff,
        },
        "frequency": {
            "enabled": "frequency" in performed,
            "freq_name": freq_name if "frequency" in performed else None,
        },
        "velocity": {
            "enabled": "velocity" in performed,
            "v_name": v_name if "velocity" in performed else None,
        },
    }

    # Dataset側にも履歴（持ち出し対策）
    ds = ds.copy(deep=False)
    ds.attrs["nro_tools"] = meta

    return CalibratedScans(
        source=scans.path,
        ds=ds,
        meta=meta,
        primary_header=scans.primary_header,
    )
