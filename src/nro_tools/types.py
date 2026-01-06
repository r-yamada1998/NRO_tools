from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import xarray as xr


@dataclass(frozen=True, slots=True)
class Scans:
    """
    Raw scans container (already reshaped to time×array at load time).

    Notes
    -----
    - This class is a pure container (no mutation methods).
    - `ds` is expected to have dims like (time, array, chan) after loading.
    """

    path: Path
    ds: xr.Dataset
    primary_header: Optional[Dict[str, Any]] = None
    meta: Dict[str, Any] = None  # loader-side meta (e.g., ext, reshaping info)

    def __post_init__(self) -> None:
        # ensure meta is always a dict (while keeping frozen dataclass)
        if self.meta is None:
            object.__setattr__(self, "meta", {})


@dataclass(frozen=True, slots=True)
class CalibratedScans:
    """
    Calibrated scans container.

    Contract
    --------
    - This container MUST represent data where at least one calibration step
      has been performed (intensity/frequency/velocity/etc).
    - What was done is stored in `meta["performed"]` etc.
    """

    source: Path
    ds: xr.Dataset
    meta: Dict[str, Any]
    primary_header: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.meta, dict):
            raise TypeError("CalibratedScans.meta must be a dict.")
        performed = self.meta.get("performed", ())
        if not performed:
            raise ValueError(
                "CalibratedScans requires meta['performed'] to be non-empty."
            )


# ---- Common type aliases ----
Range2 = Tuple[float, float]
TwoRanges = Tuple[Range2, Range2]
DetectionStatus = Literal["detection", "non-detection"]


@dataclass(frozen=True)
class NoiseEstimate:
    baseline: float
    noise: float
    npts: int
    peak: float
    threshold: float
    snr_sigma: float


@dataclass(frozen=True)
class SpectrumFitParams:
    """Const + Gaussian fit params for one spectrum."""

    c: float
    amp: float
    mu: float
    sigma: float
    fwhm: float
    rss: float


@dataclass(frozen=True)
class SpectrumFitResult:
    status: DetectionStatus
    noise: NoiseEstimate
    fit: Optional[SpectrumFitParams] = None
    fit_error: Optional[str] = None


@dataclass(frozen=True)
class PanelAmplitudeSummary:
    """
    Representative amplitudes per cross panel.
    Keys: "-DAZ", "0", "+DAZ", "+DEL", "-DEL" (subset if missing)
    """

    amp: Dict[str, float] = field(default_factory=dict)
    daz: Dict[str, float] = field(default_factory=dict)
    dele: Dict[str, float] = field(default_factory=dict)
    agg: Literal["median", "mean"] = "median"


@dataclass(frozen=True)
class Pointing1DFitResult:
    axis: Literal["DAZ", "DEL"]
    amplitude: float
    mean: float
    stddev: float


@dataclass(frozen=True)
class PointingResult:
    """
    1D pointing fits inferred from amplitude vs DAZ and amplitude vs DEL.
    """

    summary: PanelAmplitudeSummary
    daz_fit: Optional[Pointing1DFitResult] = None
    del_fit: Optional[Pointing1DFitResult] = None
