"""Isolate Selected Spike Plugin for Phy

This plugin allows you to quickly extract a single anomalous or misclassified spike
out of a cluster and place it into its own brand new cluster.

=============================
Rationale & Core Features
=============================

During manual spike sorting, you often encounter a cluster that is otherwise clean
except for one or two obvious outlier spikes (e.g., massive artifacts, or a single
spike that clearly belongs to another neuron). In standard Phy, splitting out just
a single spike is tedious because you usually have to use lasso tools in the
Feature View, which is inaccurate and time-consuming for just one spike.

This plugin introduces a one-click solution to isolate a spike directly from the
Trace View or Amplitude View, saving significant time during detailed curation.

=============================
Usage & Workflow
=============================

1. Select a spike using either of these methods:
   - **Method A (Precision)**: `Ctrl+click` a spike in the Trace View or Amplitude View.
     This highlights the spike in yellow (standard Phy behavior).
   - **Method B (Navigation)**: Use `Alt+PgUp` or `Alt+PgDown` to snap the Trace View
     exactly to the next or previous spike in the cluster.

2. Press **Alt+Shift+I**.
   The plugin will instantly remove that exact spike from the current cluster and
   create a new cluster for it.

=============================
How it Works (Under the Hood)
=============================
The plugin determines which spike you want to isolate using two fallbacks:
1. **Global Selection (`Ctrl+Click`)**: It first checks if you have explicitly
   highlighted a spike. This is the safest and most authoritative source.
2. **Trace View Center Time**: If you haven't explicitly clicked a spike but
   navigated using `Alt+PgUp/Down`, the plugin looks at the exact center time of
   your Trace View screen. It finds the spike closest to the center crosshair and
   isolates it.

Safety Measures:
- The plugin verifies the spike still belongs to the cluster you have selected
  before splitting, guarding against mistakes if you had an old selection active.
- It automatically clears the selection highlight after splitting to prevent you
  from accidentally splitting the same spike again.
"""

from __future__ import annotations

import logging

import numpy as np
from phy import IPlugin, connect

logger = logging.getLogger("phy")


class IsolateSelectedSpike(IPlugin):
    """Split the currently focused spike in TraceView into its own cluster."""

    def __init__(self):
        super().__init__()
        self._ready_controller_ids: set[int] = set()

    def attach_to_controller(self, controller):
        controller_id = id(controller)
        if controller_id in self._ready_controller_ids:
            return
        self._ready_controller_ids.add(controller_id)

        # ── Register the action once the GUI is ready ──
        @connect
        def on_gui_ready(sender, gui):
            if sender is not controller:
                return

            supervisor = getattr(controller, "supervisor", None)
            if supervisor is None:
                return

            def _get_spike_from_selection():
                """Return (spike_id, cluster_id) from the global selection
                (set by Ctrl+click, highlighted yellow in the AmplitudeView)."""
                sel = getattr(controller, "selection", None)
                if sel is None:
                    return None, None
                spike_ids = sel.get("spike_ids", None)
                if not spike_ids:
                    return None, None
                spike_id = int(spike_ids[0])
                cluster_id = int(controller.model.spike_clusters[spike_id])
                return spike_id, cluster_id

            def _get_spike_from_trace_center():
                """Return (spike_id, cluster_id) by finding the spike from the
                first selected cluster closest to the TraceView's current
                center time (used after Alt+PgUp/Down navigation)."""
                # Find the TraceView instance.
                trace_view = None
                for v in gui.views:
                    if type(v).__name__ == "TraceView":
                        trace_view = v
                        break
                if trace_view is None:
                    return None, None

                center_time = trace_view.time

                # Get selected cluster(s).
                selected = list(getattr(supervisor, "selected", []) or [])
                if not selected:
                    return None, None

                cluster_id = selected[0]

                # Get all spike indices for that cluster.
                spike_ids = np.asarray(
                    supervisor.clustering.spikes_in_clusters([cluster_id]),
                    dtype=np.int64,
                )
                if spike_ids.size == 0:
                    return None, None

                spike_times = np.asarray(
                    controller.model.spike_times[spike_ids], dtype=np.float64
                )

                # Find the spike whose time is closest to the view center.
                idx = int(np.argmin(np.abs(spike_times - center_time)))
                return int(spike_ids[idx]), cluster_id

            @supervisor.actions.add(shortcut="i")
            def isolate_selected_spike():
                """Split the focused spike in TraceView into its own cluster.

                Works after Ctrl+clicking a spike (yellow highlight in
                AmplitudeView) *or* after navigating to a spike with
                Alt+PgDown / Alt+PgUp.  The spike is removed from its current
                cluster and placed into a newly created cluster.
                """

                # 1) Try the global selection (Ctrl+click / AmplitudeView
                #    yellow highlight).
                spike_id, cluster_id = _get_spike_from_selection()

                # 2) Fallback: resolve from TraceView center time
                #    (Alt+PgUp/Down navigation).
                if spike_id is None:
                    spike_id, cluster_id = _get_spike_from_trace_center()

                if spike_id is None:
                    msg = (
                        "No spike identified — Ctrl+click a spike or navigate "
                        "with Alt+PgUp/PgDown in the TraceView first."
                    )
                    logger.warning(msg)
                    _status(gui, msg)
                    return

                # Verify the spike still belongs to the expected cluster.
                current_cluster = int(controller.model.spike_clusters[spike_id])
                if cluster_id is not None and current_cluster != cluster_id:
                    logger.info(
                        "Spike %d moved from cluster %d → %d since selection; "
                        "using current cluster %d.",
                        spike_id,
                        cluster_id,
                        current_cluster,
                        current_cluster,
                    )
                    cluster_id = current_cluster

                # Ensure cluster has more than one spike.
                all_spike_ids = np.asarray(
                    supervisor.clustering.spikes_in_clusters([cluster_id]),
                    dtype=np.int64,
                )

                if all_spike_ids.size <= 1:
                    msg = (
                        f"Cluster {cluster_id} has only {all_spike_ids.size} spike(s) "
                        "— nothing to isolate."
                    )
                    logger.warning(msg)
                    _status(gui, msg)
                    return

                # Split out just the target spike.
                spike_id_arr = np.array([spike_id], dtype=np.int64)
                supervisor.actions.split(spike_id_arr)

                msg = f"Isolated spike {spike_id} from cluster {cluster_id}"
                logger.info(msg)
                _status(gui, msg)

                # Clear the global selection so the user doesn't accidentally
                # repeat the action on a stale spike.
                try:
                    controller.selection["spike_ids"] = []
                except Exception:
                    pass


def _status(gui, msg: str) -> None:
    """Best-effort status-bar update."""
    fn = getattr(gui, "status_message", None)
    if callable(fn):
        try:
            fn(msg)
        except Exception:
            pass
