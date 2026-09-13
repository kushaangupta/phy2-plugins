"""Shared amplitude-completeness metrics for the Phy plugins.

The amplitude-cutoff metric estimates the fraction of a unit's spikes that fell
below the sorter's detection threshold.  It assumes that the *untruncated*
amplitude distribution is approximately symmetric and stationary.  The result
is therefore a curation aid, not a direct measurement: drift, overlapping
units, burst-dependent waveform changes, and amplitude normalization can all
violate that assumption.

This implementation follows the histogram-tail method used by the Allen
Institute/SpikeInterface quality-metrics ecosystem.  Keeping it in a small,
Phy-independent module lets the interactive view and automated quality plugin
report exactly the same quantity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d


@dataclass(frozen=True)
class AmplitudeCutoffEstimate:
    """Amplitude-cutoff result plus the arrays needed to visualize it."""

    fraction_missing: float
    bin_centers: np.ndarray
    density: np.ndarray
    smoothed_density: np.ndarray
    peak_index: int | None
    tail_index: int | None
    n_spikes: int
    reason: str = ""

    @property
    def valid(self) -> bool:
        """Whether ``fraction_missing`` is a finite estimate."""

        return bool(np.isfinite(self.fraction_missing))


def estimate_amplitude_cutoff(
    amplitudes,
    *,
    num_bins: int = 100,
    smoothing_sigma: float = 3.0,
    min_spikes: int = 50,
) -> AmplitudeCutoffEstimate:
    """Estimate the fraction of spikes missing below the amplitude threshold.

    The high-amplitude tail is used as a proxy for the unobserved low-amplitude
    tail.  Values are capped at 0.5 because once more than half the spikes are
    missing this symmetry-based estimate is not identifiable.

    Non-finite amplitudes are discarded.  ``NaN`` is returned for small or
    degenerate samples instead of the misleading value zero.
    """

    values = np.asarray(amplitudes, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    n_spikes = int(values.size)

    def invalid(reason: str) -> AmplitudeCutoffEstimate:
        empty = np.empty(0, dtype=np.float64)
        return AmplitudeCutoffEstimate(
            np.nan, empty, empty, empty, None, None, n_spikes, reason
        )

    if n_spikes < int(min_spikes):
        return invalid(f"need at least {min_spikes} finite amplitudes")
    if num_bins < 10:
        raise ValueError("num_bins must be at least 10")
    if smoothing_sigma < 0:
        raise ValueError("smoothing_sigma must be non-negative")
    if float(np.ptp(values)) <= np.finfo(np.float64).eps:
        return invalid("amplitudes have no measurable spread")

    density, edges = np.histogram(values, bins=int(num_bins), density=True)
    centers = (edges[:-1] + edges[1:]) / 2.0
    smoothed = gaussian_filter1d(density.astype(np.float64), smoothing_sigma)

    if not np.all(np.isfinite(smoothed)) or not np.any(smoothed > 0):
        return invalid("could not construct an amplitude density")

    peak_index = int(np.argmax(smoothed))
    # Locate the point on the high-amplitude side whose density most closely
    # matches the low-amplitude boundary.  Its remaining probability mass is
    # the estimated tail lost below the detection threshold.
    tail_index = peak_index + int(
        np.argmin(np.abs(smoothed[peak_index:] - smoothed[0]))
    )
    bin_width = float(np.mean(np.diff(edges)))
    fraction_missing = float(np.sum(smoothed[tail_index:]) * bin_width)
    fraction_missing = float(np.clip(fraction_missing, 0.0, 0.5))

    return AmplitudeCutoffEstimate(
        fraction_missing,
        centers,
        density,
        smoothed,
        peak_index,
        tail_index,
        n_spikes,
    )


def compute_amplitude_cutoff(amplitudes, num_bins: int = 100) -> float:
    """Return only the missing-spike fraction (``NaN`` if not estimable)."""

    return estimate_amplitude_cutoff(amplitudes, num_bins=num_bins).fraction_missing
