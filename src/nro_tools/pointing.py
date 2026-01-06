# src/nro_tools/pointing.py
from __future__ import annotations

from typing import Dict, Iterable, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from astropy.modeling import fitting, models

from .types import (
    NoiseEstimate,
    PanelAmplitudeSummary,
    Pointing1DFitResult,
    PointingResult,
    SpectrumFitParams,
    SpectrumFitResult,
    TwoRanges,
)


# --------------------------
# Baseline/noise + detection
# --------------------------
def estimate_baseline_noise(
    x: np.ndarray,
    y: np.ndarray,
    emission_free_ranges: TwoRanges,
    *,
    ddof: int = 1,
) -> Tuple[float, float, int]:
    """
    Estimate baseline (mean) and noise (std) from two emission-free x-ranges.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)

    sel = np.zeros_like(m, dtype=bool)
    for lo, hi in emission_free_ranges:
        lo2, hi2 = (lo, hi) if lo <= hi else (hi, lo)
        sel |= (x >= lo2) & (x <= hi2)

    mm = m & sel
    yy = y[mm]
    n = int(yy.size)
    if n < 10:
        raise ValueError(f"Too few points in emission-free ranges: n={n}")

    baseline = float(np.mean(yy))
    noise = float(np.std(yy, ddof=ddof)) if n > 1 else float(np.std(yy))
    return baseline, noise, n


def is_detection(
    y: np.ndarray,
    baseline: float,
    noise: float,
    *,
    snr_sigma: float = 5.0,
) -> bool:
    """
    Detection gate: peak > baseline + snr_sigma * noise
    """
    peak = float(np.nanmax(np.asarray(y, dtype=float)))
    return bool(peak > baseline + snr_sigma * noise)


# --------------------------
# Gaussian fit (Const+Gauss)
# --------------------------
def _gaussian_init_from_peak(
    x: np.ndarray, y: np.ndarray, baseline: float
) -> Tuple[float, float, float]:
    """
    Initial guess based on peak position.
    returns (amp0, mu0, sigma0)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 10:
        raise ValueError("Too few finite points for fitting.")

    imax = int(np.nanargmax(y))
    mu0 = float(x[imax])
    amp0 = float(y[imax] - baseline)
    if not np.isfinite(amp0) or amp0 == 0:
        amp0 = float(np.nanmax(y) - np.nanmin(y))

    # sigma guess from half-max width
    half = baseline + 0.5 * amp0
    idx = np.where(y > half)[0]
    if idx.size >= 2:
        fwhm = float(abs(x[idx[-1]] - x[idx[0]]))
        sigma0 = fwhm / 2.355 if fwhm > 0 else (x.max() - x.min()) / 30.0
    else:
        sigma0 = (x.max() - x.min()) / 30.0

    if sigma0 <= 0 or not np.isfinite(sigma0):
        sigma0 = (x.max() - x.min()) / 30.0

    return amp0, mu0, float(sigma0)


def fit_spectrum_gaussian(
    x: np.ndarray,
    y: np.ndarray,
    *,
    emission_free_ranges: TwoRanges,
    snr_sigma: float = 5.0,
) -> SpectrumFitResult:
    """
    1) baseline/noise from emission-free ranges
    2) detection gate: peak > baseline + snr_sigma*noise
    3) if detected: fit Const + Gaussian
       else: non-detection
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    baseline, noise, npts = estimate_baseline_noise(x, y, emission_free_ranges)
    peak = float(np.nanmax(y))
    threshold = baseline + snr_sigma * noise

    noise_est = NoiseEstimate(
        baseline=baseline,
        noise=noise,
        npts=npts,
        peak=peak,
        threshold=threshold,
        snr_sigma=snr_sigma,
    )

    if not is_detection(y, baseline, noise, snr_sigma=snr_sigma):
        return SpectrumFitResult(status="non-detection", noise=noise_est)

    try:
        amp0, mu0, sig0 = _gaussian_init_from_peak(x, y, baseline)
        model0 = models.Const1D(amplitude=baseline) + models.Gaussian1D(
            amplitude=amp0, mean=mu0, stddev=sig0
        )
        fitter = fitting.LevMarLSQFitter()
        fit_model = fitter(model0, x, y)

        c_fit = float(fit_model[0].amplitude.value)
        g = fit_model[1]
        amp_fit = float(g.amplitude.value)
        mu_fit = float(g.mean.value)
        sig_fit = float(g.stddev.value)
        fwhm = float(2.355 * abs(sig_fit))

        yhat = fit_model(x)
        rss = float(np.nansum((y - yhat) ** 2))

        return SpectrumFitResult(
            status="detection",
            noise=noise_est,
            fit=SpectrumFitParams(
                c=c_fit, amp=amp_fit, mu=mu_fit, sigma=sig_fit, fwhm=fwhm, rss=rss
            ),
        )
    except Exception as e:
        return SpectrumFitResult(
            status="detection", noise=noise_est, fit=None, fit_error=str(e)
        )


# --------------------------
# Panel label helpers
# --------------------------
def panel_name(panel_row: int, panel_col: int) -> Optional[str]:
    """
    Cross layout fixed:
      row=1 col=0: -DAZ
      row=1 col=1: 0
      row=1 col=2: +DAZ
      row=0 col=1: +DEL
      row=2 col=1: -DEL
    """
    if (panel_row, panel_col) == (1, 0):
        return "-DAZ"
    if (panel_row, panel_col) == (1, 1):
        return "0"
    if (panel_row, panel_col) == (1, 2):
        return "+DAZ"
    if (panel_row, panel_col) == (0, 1):
        return "+DEL"
    if (panel_row, panel_col) == (2, 1):
        return "-DEL"
    return None


# --------------------------
# Pointing offset estimation
# --------------------------
def summarize_panel_amplitudes(
    df_fit: pd.DataFrame,
    *,
    agg: Literal["median", "mean"] = "median",
) -> PanelAmplitudeSummary:
    """
    Summarize detections per panel into representative amp and representative DAZ/DEL.
    Expects df_fit to contain columns:
      - status ("detection"/"non-detection")
      - panel_row, panel_col
      - amp (for detections)
      - DAZ, DEL
    """
    det = df_fit[df_fit["status"] == "detection"].copy()
    if det.empty:
        return PanelAmplitudeSummary(amp={}, daz={}, dele={}, agg=agg)

    det["panel"] = [
        panel_name(int(r), int(c)) for r, c in zip(det["panel_row"], det["panel_col"])
    ]
    det = det.dropna(subset=["panel"])

    g = det.groupby("panel", sort=False)

    if agg == "median":
        amp = g["amp"].median()
        daz = g["DAZ"].median()
        dele = g["DEL"].median()
    elif agg == "mean":
        amp = g["amp"].mean()
        daz = g["DAZ"].mean()
        dele = g["DEL"].mean()
    else:
        raise ValueError("agg must be 'median' or 'mean'")

    return PanelAmplitudeSummary(
        amp=amp.to_dict(),
        daz=daz.to_dict(),
        dele=dele.to_dict(),
        agg=agg,
    )


def _fit_gaussian_1d(
    x: np.ndarray, y: np.ndarray, *, axis: Literal["DAZ", "DEL"]
) -> Pointing1DFitResult:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.size < 3 or y.size < 3:
        raise ValueError("Need at least 3 points for 1D pointing fit.")

    fitter = fitting.LevMarLSQFitter()
    mu0 = float(x[np.argmax(y)])
    sig0 = float(np.ptp(x) / 2 if np.ptp(x) > 0 else 1.0)
    m0 = models.Gaussian1D(amplitude=float(np.nanmax(y)), mean=mu0, stddev=sig0)
    mfit = fitter(m0, x, y)

    return Pointing1DFitResult(
        axis=axis,
        amplitude=float(mfit.amplitude.value),
        mean=float(mfit.mean.value),
        stddev=float(abs(mfit.stddev.value)),
    )


def estimate_pointing_offset(
    df_fit: pd.DataFrame,
    *,
    agg: Literal["median", "mean"] = "median",
) -> PointingResult:
    """
    Fit amplitude vs DAZ and amplitude vs DEL from the 5-point cross.
    Returns PointingResult with up to two 1D fits.
    """
    summary = summarize_panel_amplitudes(df_fit, agg=agg)

    daz_fit = None
    del_fit = None

    # DAZ: (-DAZ, 0, +DAZ)
    if all(k in summary.amp for k in ("-DAZ", "0", "+DAZ")) and all(
        k in summary.daz for k in ("-DAZ", "0", "+DAZ")
    ):
        x_daz = np.array(
            [summary.daz["-DAZ"], summary.daz["0"], summary.daz["+DAZ"]], dtype=float
        )
        y_amp = np.array(
            [summary.amp["-DAZ"], summary.amp["0"], summary.amp["+DAZ"]], dtype=float
        )
        daz_fit = _fit_gaussian_1d(x_daz, y_amp, axis="DAZ")

    # DEL: (+DEL, 0, -DEL)  (order doesn't matter for fit)
    if all(k in summary.amp for k in ("+DEL", "0", "-DEL")) and all(
        k in summary.dele for k in ("+DEL", "0", "-DEL")
    ):
        x_del = np.array(
            [summary.dele["+DEL"], summary.dele["0"], summary.dele["-DEL"]], dtype=float
        )
        y_amp = np.array(
            [summary.amp["+DEL"], summary.amp["0"], summary.amp["-DEL"]], dtype=float
        )
        del_fit = _fit_gaussian_1d(x_del, y_amp, axis="DEL")

    return PointingResult(
        summary=summary,
        az=df_fit["az"].mean(),
        el=df_fit["el"].mean(),
        daz_fit=daz_fit,
        del_fit=del_fit,
    )
