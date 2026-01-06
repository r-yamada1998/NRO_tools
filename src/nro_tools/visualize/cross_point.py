# src/nro_tools/visualize/cross_point.py
from __future__ import annotations

from typing import Dict, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from astropy.modeling import fitting, models

from ..pointing import estimate_pointing_offset, fit_spectrum_gaussian, panel_name
from ..types import PointingResult, SpectrumFitResult, TwoRanges


def _as_stripped(da: xr.DataArray) -> xr.DataArray:
    return da.astype(str).str.strip()


def _choose_x(dsi: xr.Dataset) -> Tuple[np.ndarray, str]:
    if "v_los" in dsi:
        return np.asarray(dsi["v_los"]), "V_LOS (km/s)"
    if "freq" in dsi:
        return np.asarray(dsi["freq"]), "Frequency (Hz)"
    if "chan" in dsi:
        return np.asarray(dsi["chan"]), "Channel"
    # fallback: first coord-like axis
    raise KeyError("No suitable x-axis found: expected one of v_los/freq/chan.")


def _resolve_array_name(ds: xr.Dataset, prefer: Sequence[str] = ("A01", "A1")) -> str:
    arrays = [str(a) for a in ds["array"].values.tolist()]
    for p in prefer:
        if p in arrays:
            return p
    # otherwise return first
    if not arrays:
        raise ValueError("Dataset has no 'array' coordinate.")
    return arrays[0]


def _pick_cross_points(daz: np.ndarray, dele: np.ndarray, round_ndec: int = 6):
    """
    Return rounded daz/dele arrays and the 5 unique points:
      center, p_daz, m_daz, p_del, m_del as (DAZ,DEL) pairs.
    """
    daz_r = np.round(np.asarray(daz, dtype=float), round_ndec)
    del_r = np.round(np.asarray(dele, dtype=float), round_ndec)

    pairs = np.unique(np.column_stack([daz_r, del_r]), axis=0)
    if len(pairs) < 3:
        raise ValueError(f"(DAZ,DEL) unique points too few: {len(pairs)}")

    # center closest to origin
    r2 = pairs[:, 0] ** 2 + pairs[:, 1] ** 2
    center = pairs[np.argmin(r2)]
    daz_c, del_c = center

    # +/- DAZ: among points with DEL close to center DEL
    del_diffs = np.unique(np.abs(pairs[:, 1] - del_c))
    del_tol = del_diffs[1] if len(del_diffs) > 1 else 0.0
    cand_daz = pairs[np.abs(pairs[:, 1] - del_c) <= (del_tol + 1e-30)]
    p_daz = cand_daz[np.argmax(cand_daz[:, 0])]
    m_daz = cand_daz[np.argmin(cand_daz[:, 0])]

    # +/- DEL: among points with DAZ close to center DAZ
    daz_diffs = np.unique(np.abs(pairs[:, 0] - daz_c))
    daz_tol = daz_diffs[1] if len(daz_diffs) > 1 else 0.0
    cand_del = pairs[np.abs(pairs[:, 0] - daz_c) <= (daz_tol + 1e-30)]
    p_del = cand_del[np.argmax(cand_del[:, 1])]
    m_del = cand_del[np.argmin(cand_del[:, 1])]

    return daz_r, del_r, center, p_daz, m_daz, p_del, m_del


def plot_cross_pointing(
    ds: xr.Dataset,
    *,
    array_name: Optional[str] = None,
    emission_free_ranges: Optional[TwoRanges] = None,
    snr_sigma: float = 5.0,
    round_ndec: int = 6,
    ylims_from_center: bool = True,
    alpha_spec: float = 0.25,
    overlay_fit: bool = True,
    figsize: Tuple[float, float] = (10, 10),
) -> Tuple[plt.Figure, np.ndarray, pd.DataFrame, xr.Dataset]:
    """
    Plot 5-point cross spectra (ON only) for a chosen array,
    with optional gated Gaussian fitting.

    Cross layout:
      horizontal: left=-DAZ, center=0, right=+DAZ
      vertical  : top=+DEL, center=0, bottom=-DEL

    Returns:
      fig, axes(3x3), df_fit, ds_on
    """
    # ON only
    ds_on = ds.where(_as_stripped(ds["SCNTP"]) == "ON", drop=True)

    # choose array
    if array_name is None:
        array_name = _resolve_array_name(ds_on)
    ds_on = ds_on.sel(array=array_name)

    # choose 5 points by DAZ/DEL only
    daz_r, del_r, center, p_daz, m_daz, p_del, m_del = _pick_cross_points(
        ds_on["DAZ"].values, ds_on["DEL"].values, round_ndec=round_ndec
    )

    # map panels to (DAZ,DEL)
    layout = {
        (1, 1): center,
        (1, 0): m_daz,  # left=-DAZ
        (1, 2): p_daz,  # right=+DAZ
        (0, 1): p_del,  # top=+DEL
        (2, 1): m_del,  # bottom=-DEL
    }

    # y-lims from center panel (optional)
    ymin, ymax = None, None
    if ylims_from_center:
        dz0, de0 = center
        idx_c = np.where((daz_r == dz0) & (del_r == de0))[0]
        if idx_c.size:
            y_all = [np.asarray(ds_on.isel(time=it)["spec_cal"]) for it in idx_c]
            y_all = np.concatenate([yy for yy in y_all if yy is not None])
            if y_all.size:
                ymin, ymax = float(np.nanmin(y_all)), float(np.nanmax(y_all))

    fig, axes = plt.subplots(3, 3, figsize=figsize)
    for ax in axes.flat:
        ax.axis("off")

    # titles
    axes[1, 0].set_title("-DAZ")
    axes[1, 1].set_title("0")
    axes[1, 2].set_title("+DAZ")
    axes[0, 1].set_title("+DEL")
    axes[2, 1].set_title("-DEL")

    records: list[dict] = []

    for (r, c), (dz0, de0) in layout.items():
        ax = axes[r, c]
        ax.axis("on")

        idx = np.where((daz_r == dz0) & (del_r == de0))[0]
        if idx.size == 0:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            if ymin is not None:
                ax.set_ylim(ymin, ymax)
            continue

        for it in idx:
            dsi = ds_on.isel(time=it)
            x, xlabel = _choose_x(dsi)
            y = np.asarray(dsi["spec_cal"])

            ax.plot(x, y, alpha=alpha_spec)

            fit_res: Optional[SpectrumFitResult] = None
            if emission_free_ranges is not None:
                fit_res = fit_spectrum_gaussian(
                    x, y, emission_free_ranges=emission_free_ranges, snr_sigma=snr_sigma
                )

                if (
                    fit_res.status == "detection"
                    and overlay_fit
                    and (fit_res.fit is not None)
                    and (fit_res.fit_error is None)
                ):
                    # overlay model curve
                    xs = np.linspace(np.nanmin(x), np.nanmax(x), 600)
                    model_y = fit_res.fit.c + fit_res.fit.amp * np.exp(
                        -0.5 * ((xs - fit_res.fit.mu) / fit_res.fit.sigma) ** 2
                    )
                    ax.plot(xs, model_y, alpha=0.8)

                if fit_res.status != "detection":
                    ax.text(
                        0.02,
                        0.95,
                        "non-detection",
                        transform=ax.transAxes,
                        va="top",
                        fontsize=9,
                    )

                # show noise windows
                for lo, hi in emission_free_ranges:
                    lo2, hi2 = (lo, hi) if lo <= hi else (hi, lo)
                    ax.axvspan(lo2, hi2, alpha=0.08)

            rec = {
                "panel_row": r,
                "panel_col": c,
                "panel": panel_name(r, c),
                "time": np.datetime_as_string(dsi["time"].values),
                "DAZ": float(dz0),
                "DEL": float(de0),
            }

            if fit_res is None:
                rec["status"] = "no-fit"
            else:
                rec["status"] = fit_res.status
                rec.update(
                    {
                        "baseline": fit_res.noise.baseline,
                        "noise": fit_res.noise.noise,
                        "n_noise_pts": fit_res.noise.npts,
                        "peak": fit_res.noise.peak,
                        "threshold": fit_res.noise.threshold,
                        "snr_sigma": fit_res.noise.snr_sigma,
                    }
                )
                if fit_res.fit is not None:
                    rec.update(
                        {
                            "c": fit_res.fit.c,
                            "amp": fit_res.fit.amp,
                            "mu": fit_res.fit.mu,
                            "sigma": fit_res.fit.sigma,
                            "fwhm": fit_res.fit.fwhm,
                            "rss": fit_res.fit.rss,
                        }
                    )
                if fit_res.fit_error:
                    rec["fit_error"] = fit_res.fit_error

            records.append(rec)

        if ymin is not None:
            ax.set_ylim(ymin, ymax)

        if r == 2:
            ax.set_xlabel(xlabel)
        if c == 0:
            ax.set_ylabel("spec_cal")

    fig.suptitle(f"5-point cross (ON only)  array={array_name}", fontsize=14)
    # NOTE: do NOT call tight_layout here (breaks add_axes workflows)

    df_fit = pd.DataFrame(records)
    return fig, axes, df_fit, ds_on


def add_pointing_axes_outside(
    fig: plt.Figure,
    axes_3x3: np.ndarray,
    df_fit: pd.DataFrame,
    *,
    agg: Literal["median", "mean"] = "median",
    pad: float = 0.04,
    daz_height: float = 0.05,
    del_width: float = 0.05,
    del_gap_extra: float = 0.06,
    daz_gap_extra: float = 0.06,
) -> Dict[str, plt.Axes]:
    """
    Add two axes outside the cross:
      - bottom: x=DAZ, y=amplitude (+ Gaussian fit curve)
      - right : x=amplitude, y=DEL (+ Gaussian fit curve as x(DEL) vs DEL)

    IMPORTANT:
      - Do not call tight_layout after this.
      - Ensure your figure has enough margin; increase figsize if needed.
    """
    pr: PointingResult = estimate_pointing_offset(df_fit, agg=agg)

    # cross bbox from 5 axes
    used = [
        axes_3x3[1, 0],
        axes_3x3[1, 1],
        axes_3x3[1, 2],
        axes_3x3[0, 1],
        axes_3x3[2, 1],
    ]
    x0 = min(a.get_position().x0 for a in used)
    x1 = max(a.get_position().x1 for a in used)
    y0 = min(a.get_position().y0 for a in used)
    y1 = max(a.get_position().y1 for a in used)

    # --- DAZ axis (below) ---
    bottom = max(0.02, y0 - pad - daz_gap_extra - daz_height)
    ax_daz = fig.add_axes([x0, bottom, (x1 - x0), daz_height])

    # --- DEL axis (right) ---
    left = min(0.98 - del_width, x1 + pad + del_gap_extra)
    ax_del = fig.add_axes([left, y0, del_width, (y1 - y0)])

    # Plot DAZ points + fit
    if (
        pr.summary.amp
        and pr.daz_fit
        and all(k in pr.summary.amp for k in ("-DAZ", "0", "+DAZ"))
    ):
        x_daz = np.array(
            [pr.summary.daz["-DAZ"], pr.summary.daz["0"], pr.summary.daz["+DAZ"]],
            dtype=float,
        )
        y_amp = np.array(
            [pr.summary.amp["-DAZ"], pr.summary.amp["0"], pr.summary.amp["+DAZ"]],
            dtype=float,
        )
        ax_daz.scatter(x_daz, y_amp)

        # reconstruct model curve
        m = models.Gaussian1D(
            amplitude=pr.daz_fit.amplitude,
            mean=pr.daz_fit.mean,
            stddev=pr.daz_fit.stddev,
        )
        xs = np.linspace(np.nanmin(x_daz), np.nanmax(x_daz), 300)
        ax_daz.plot(xs, m(xs))

        ax_daz.text(
            0.02,
            0.95,
            f"DAZ fit: mu={pr.daz_fit.mean:.4g}, sig={pr.daz_fit.stddev:.4g}",
            transform=ax_daz.transAxes,
            va="top",
        )
    else:
        ax_daz.text(
            0.02,
            0.95,
            "DAZ fit: not enough detections",
            transform=ax_daz.transAxes,
            va="top",
        )

    ax_daz.set_xlabel("DAZ")
    ax_daz.set_ylabel("Amplitude")

    # Plot DEL points (x=amp, y=DEL) + fit
    if (
        pr.summary.amp
        and pr.del_fit
        and all(k in pr.summary.amp for k in ("+DEL", "0", "-DEL"))
    ):
        x_amp = np.array(
            [pr.summary.amp["+DEL"], pr.summary.amp["0"], pr.summary.amp["-DEL"]],
            dtype=float,
        )
        y_del = np.array(
            [pr.summary.dele["+DEL"], pr.summary.dele["0"], pr.summary.dele["-DEL"]],
            dtype=float,
        )
        ax_del.scatter(x_amp, y_del)

        m = models.Gaussian1D(
            amplitude=pr.del_fit.amplitude,
            mean=pr.del_fit.mean,
            stddev=pr.del_fit.stddev,
        )
        ys = np.linspace(np.nanmin(y_del), np.nanmax(y_del), 300)  # DEL
        ax_del.plot(m(ys), ys)  # x=amp, y=DEL

        ax_del.text(
            0.02,
            0.98,
            f"DEL fit: mu={pr.del_fit.mean:.4g}, sig={pr.del_fit.stddev:.4g}",
            transform=ax_del.transAxes,
            va="top",
        )
    else:
        ax_del.text(
            0.02,
            0.98,
            "DEL fit: not enough detections",
            transform=ax_del.transAxes,
            va="top",
        )

    ax_del.set_xlabel("Amplitude")
    ax_del.set_ylabel("DEL")

    return {"ax_daz": ax_daz, "ax_del": ax_del}
