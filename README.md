# Package

Tools for the CSV observations of the NRO-45m telescope

## Usage

### load nqm data

```python
from nro_tools.io import load_scans
from nro_tools import calibrate
import xarray as xr

scans = load_scans(f'/workspaces/NRO_tools/data/tmp/rcnc7b1.260103023508.01.nqm')     # time×array に整形済み
cal = calibrate(scans, mode=("intensity", "frequency", "velocity"))

cal.ds["spec_cal"]   # (time, array, chan)
cal.ds["v_los"]      # (time, array, chan)
cal.meta["performed"]  # ('intensity','frequency','velocity')
```
### pointing analysis
```python
from nro_tools.visualize.cross_point import plot_cross_pointing, add_pointing_axes_outside

ranges = ((-200, -100), (100, 200))  # emission-free ranges (速度軸の例)

fig, axes, df, ds_on = plot_cross_pointing(
    cal.ds,
    array_name="A13",
    emission_free_ranges=ranges,
    snr_sigma=5.0,
    round_ndec=6,
    ylims_from_center=True,
    figsize=(11, 9),
)

# tight_layout は呼ばない
add_pointing_axes_outside(fig, axes, df, agg="median", pad=0.05, daz_gap_extra=0.08, del_gap_extra=0.08)
```
