"""RawDataFilterPluginMeanAndHighpass — Advanced Raw Signal Filtering & Reference Removal.

This plugin adds multi-stage raw data filters to phy's TraceView and WaveformView,
allowing electrophysiologists to remove common-mode noise across recording channels
and apply high-pass temporal filtering.

Usage:
------
1. Enable 'RawDataFilterPluginMeanAndHighpass' in your phy_config.py plugins list.
2. In the phy GUI, open TraceView or WaveformView.
3. Press `Alt+R` to cycle through the registered filter modes:
   - High-Pass Only (3rd-order Butterworth, 150 Hz cutoff)
   - High-Pass -> Mean Subtraction (CAR)
   - Median Subtraction (CMR) -> High-Pass
   - Mean Subtraction (CAR) -> High-Pass
   - High-Pass -> Median Subtraction (CMR)
   - Mean Subtraction (CAR) Only (LFP-friendly)
   - Median Subtraction (CMR) Only (LFP-friendly)

Filters Breakdown:
------------------
- **High-Pass (150 Hz Butterworth)**: Strips out low-frequency LFP signals, movement artifacts,
  and baseline drift to isolate action potential spikes.
- **Common Average Referencing (CAR / Mean)**: Computes the spatial mean across all recording
  channels at each sample point and subtracts it, cancelling global probe-wide noise.
- **Common Median Referencing (CMR / Median)**: Computes the spatial median across channels and
  subtracts it. CMR is robust against high-amplitude outlier spikes on individual channels.
- **Order of Operations**: Filtering before CAR/CMR vs. CAR/CMR before filtering allows you to
  evaluate non-linear median interaction with temporal high-pass filtering on your dataset.

"""

import logging
import numpy as np
from scipy.signal import butter, filtfilt

from phy import IPlugin

logger = logging.getLogger('phy')


class RawDataFilterPluginMeanAndHighpass(IPlugin):
    """Phy GUI plugin providing high-pass filtering, CAR (mean), and CMR (median) raw data filters."""

    def attach_to_controller(self, controller):
        sample_rate = getattr(controller.model, 'sample_rate', 30000.0)
        logger.info("Initializing RawDataFilterPluginMeanAndHighpass (sample_rate = %.1f Hz)", sample_rate)

        # Design a 3rd-order high-pass Butterworth filter at 150 Hz cutoff
        cutoff_hz = 150.0
        nyquist_hz = sample_rate / 2.0
        b, a = butter(3, cutoff_hz / nyquist_hz, 'high')

        @controller.raw_data_filter.add_filter
        def high_pass_only(arr, axis=0, mean_axis=1):
            """High-pass filter (150 Hz cutoff) without spatial reference subtraction."""
            return filtfilt(b, a, arr, axis=axis)

        @controller.raw_data_filter.add_filter
        def high_pass_then_mean(arr, axis=0, mean_axis=1):
            """High-pass filter followed by Common Average Referencing (CAR)."""
            arr = filtfilt(b, a, arr, axis=axis)
            arr = arr - np.mean(arr, axis=mean_axis, keepdims=True)
            return arr

        @controller.raw_data_filter.add_filter
        def median_then_high_pass(arr, axis=0, mean_axis=1):
            """Common Median Referencing (CMR) followed by high-pass filter."""
            arr = arr - np.median(arr, axis=mean_axis, keepdims=True)
            return filtfilt(b, a, arr, axis=axis)

        @controller.raw_data_filter.add_filter
        def mean_then_high_pass(arr, axis=0, mean_axis=1):
            """Common Average Referencing (CAR) followed by high-pass filter."""
            arr = arr - np.mean(arr, axis=mean_axis, keepdims=True)
            return filtfilt(b, a, arr, axis=axis)

        @controller.raw_data_filter.add_filter
        def high_pass_then_median(arr, axis=0, mean_axis=1):
            """High-pass filter followed by Common Median Referencing (CMR)."""
            arr = filtfilt(b, a, arr, axis=axis)
            arr = arr - np.median(arr, axis=mean_axis, keepdims=True)
            return arr

        @controller.raw_data_filter.add_filter
        def mean_only(arr, axis=0, mean_axis=1):
            """Common Average Referencing (CAR) only (preserves low frequencies / LFP)."""
            return arr - np.mean(arr, axis=mean_axis, keepdims=True)

        @controller.raw_data_filter.add_filter
        def median_only(arr, axis=0, mean_axis=1):
            """Common Median Referencing (CMR) only (preserves low frequencies / LFP)."""
            return arr - np.median(arr, axis=mean_axis, keepdims=True)

        logger.info("RawDataFilterPluginMeanAndHighpass attached successfully (7 filter pipelines registered).")
