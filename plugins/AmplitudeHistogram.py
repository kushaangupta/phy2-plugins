"""Interactive amplitude-completeness view for Phy 2.

Why this exists
---------------
Phy's AmplitudeView already shows amplitude over time and includes a compact
side histogram.  This plugin adds the part that view does not provide: a
quantitative estimate of spikes lost below the detection threshold, together
with the distribution used to calculate it.

The estimate is the standard symmetry-based ``amplitude_cutoff`` quality
metric.  It is useful for triage, but it assumes a stationary, approximately
symmetric amplitude distribution.  Inspect the ordinary AmplitudeView for
drift and multimodality before treating the number as evidence of unit quality.
It does not measure contamination from other units.

Usage
-----
Enable ``AmplitudeHistogramPlugin`` in ``phy_config.py``, then open
``View > New view > AmplitudeHistogramView``.  Selecting up to eight clusters
plots their complete template-amplitude distributions.  The shaded far tail is
the probability mass used as the estimate of the missing low-amplitude tail.

The original 2016 plugin by Luke Shaheen inspired this view.  This version uses
Phy 2's view API, removes the unstable discontinuous Gaussian fit, avoids
process-wide Matplotlib style changes, and shares its metric implementation
with ``CellMetrics``.
"""

from __future__ import annotations

import logging

import numpy as np
from matplotlib import colormaps

from amplitude_metrics import estimate_amplitude_cutoff
from phy import IPlugin
from phy.cluster.views import ManualClusteringView
from phy.plot.plot import PlotCanvasMpl

logger = logging.getLogger("phy")


class AmplitudeHistogramView(ManualClusteringView):
    """Plot template-amplitude distributions and missing-spike estimates."""

    plot_canvas_class = PlotCanvasMpl
    max_n_clusters = 8

    def __init__(self, amplitude_getter=None):
        super().__init__()
        self.amplitude_getter = amplitude_getter

    def on_select(self, cluster_ids=(), **kwargs):
        """Recompute the diagnostic for the selected clusters."""

        self.cluster_ids = tuple(cluster_ids[: self.max_n_clusters])
        ax = self.canvas.ax
        ax.clear()
        ax.set_facecolor("#111111")
        self.figure.set_facecolor("#111111")

        if not self.cluster_ids:
            ax.text(
                0.5,
                0.5,
                "Select one or more clusters",
                color="0.8",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            self.canvas.update()
            return

        try:
            bunches = self.amplitude_getter(self.cluster_ids)
        except Exception as exc:
            logger.exception("AmplitudeHistogram: could not load amplitudes")
            ax.text(
                0.5,
                0.5,
                f"Could not load amplitudes: {exc}",
                color="#ff7777",
                ha="center",
                va="center",
                wrap=True,
                transform=ax.transAxes,
            )
            self.canvas.update()
            return

        colors = colormaps["tab10"]
        for index, (cluster_id, bunch) in enumerate(zip(self.cluster_ids, bunches)):
            color = colors(index % 10)
            estimate = estimate_amplitude_cutoff(bunch.amplitudes)
            if not estimate.valid:
                ax.plot([], [], color=color, label=f"{cluster_id}: {estimate.reason}")
                continue

            x = estimate.bin_centers
            ax.fill_between(
                x,
                estimate.density,
                step="mid",
                color=color,
                alpha=0.18,
            )
            ax.plot(x, estimate.smoothed_density, color=color, linewidth=1.8)

            tail = slice(estimate.tail_index, None)
            ax.fill_between(
                x[tail],
                estimate.smoothed_density[tail],
                color=color,
                alpha=0.45,
            )
            ax.axvline(
                x[estimate.peak_index], color=color, linewidth=0.8, alpha=0.55
            )
            ax.plot(
                [],
                [],
                color=color,
                label=(
                    f"{cluster_id}: {estimate.fraction_missing:.1%} missing "
                    f"(n={estimate.n_spikes:,})"
                ),
            )
            logger.info(
                "AmplitudeHistogram: cluster %s estimated missing spikes %.1f%% (n=%d)",
                cluster_id,
                100.0 * estimate.fraction_missing,
                estimate.n_spikes,
            )

        if len(cluster_ids) > self.max_n_clusters:
            logger.warning(
                "AmplitudeHistogram displays the first %d selected clusters.",
                self.max_n_clusters,
            )

        ax.set_title("Amplitude completeness (template amplitudes)", color="white")
        ax.set_xlabel("Template amplitude", color="0.85")
        ax.set_ylabel("Probability density", color="0.85")
        ax.tick_params(colors="0.75")
        for spine in ax.spines.values():
            spine.set_color("0.45")
        ax.grid(color="white", alpha=0.08, linewidth=0.6)
        if ax.lines:
            legend = ax.legend(loc="best", framealpha=0.25, fontsize="small")
            if legend is not None:
                for text in legend.get_texts():
                    text.set_color("white")
        ax.text(
            0.01,
            0.01,
            "Shaded tail estimates spikes lost below threshold; check drift in AmplitudeView.",
            color="0.6",
            fontsize=8,
            transform=ax.transAxes,
        )
        self.figure.tight_layout()
        self.canvas.update()


class AmplitudeHistogramPlugin(IPlugin):
    """Register the amplitude-completeness view with Phy's View menu."""

    def attach_to_controller(self, controller):
        def get_amplitudes(cluster_ids):
            return controller._amplitude_getter(
                cluster_ids, name="template", load_all=True
            )

        def create_amplitude_histogram_view():
            return AmplitudeHistogramView(amplitude_getter=get_amplitudes)

        controller.view_creator["AmplitudeHistogramView"] = (
            create_amplitude_histogram_view
        )
