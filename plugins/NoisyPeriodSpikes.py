"""NoisyPeriodSpikes — Isolate spikes from noisy recording epochs by time window.

During electrophysiology recordings, specific time periods may contain
movement artifacts, electrical noise bursts, or other transient disturbances
that contaminate spike clusters. This plugin lets you surgically extract
spikes falling within a user-specified time window [t1, t2] and split them
into a separate cluster for review or removal.

Usage:
------
1. Enable 'NoisyPeriodSpikes' in your phy_config.py plugins list.

2. **Single-cluster isolation** (Alt+Q):
   - Select a cluster in the ClusterView.
   - Press Alt+Q and enter two values: ``t1 t2`` (in seconds).
   - All spikes from the selected cluster between t1 and t2 are split
     into a new cluster.

3. **Global isolation across all clusters** (Alt+Shift+Q):
   - Press Alt+Shift+Q and enter two values: ``t1 t2`` (in seconds).
   - Iterates over every cluster in the dataset and splits out spikes
     in the [t1, t2] window. Useful for removing an entire noisy epoch
     from the recording in one step.

Notes:
------
- Times are in seconds relative to recording start (same units as
  ``controller.model.spike_times``).
- Only clusters that actually contain spikes inside the window are modified.
- The operation is undoable via Ctrl+Z in phy.
"""

import logging

import numpy as np
from phy import IPlugin, connect

logger = logging.getLogger('phy')


class NoisyPeriodSpikes(IPlugin):
    """Split spikes within a specified time window into separate clusters."""

    def attach_to_controller(self, controller):

        @connect
        def on_gui_ready(sender, gui):

            @controller.supervisor.actions.add(
                shortcut='alt+q',
                prompt=True,
                n_args=2,
            )
            def noisy_period_spikes(t1, t2):
                """Isolate spikes from the selected cluster between t1 and t2 (seconds).

                Enter two time-points: ``t1 t2``
                """
                t1, t2 = float(t1), float(t2)
                if t1 >= t2:
                    logger.warning("NoisyPeriodSpikes: t1 (%.3f) must be less than t2 (%.3f)", t1, t2)
                    return

                cluster_id = controller.supervisor.selected[0]
                spike_ids = controller.model.get_cluster_spikes(cluster_id)
                spike_times = controller.model.spike_times[spike_ids]

                noisy_mask = (spike_times > t1) & (spike_times < t2)
                n_noisy = int(noisy_mask.sum())

                if n_noisy == 0:
                    logger.info("NoisyPeriodSpikes: no spikes found in [%.3f, %.3f] for cluster %d",
                                t1, t2, cluster_id)
                    return

                labels = np.where(noisy_mask, 1, 0)
                logger.info("NoisyPeriodSpikes: splitting %d / %d spikes from cluster %d "
                            "in window [%.3f, %.3f] s",
                            n_noisy, len(spike_ids), cluster_id, t1, t2)
                controller.supervisor.actions.split(spike_ids, labels)

        @connect
        def on_gui_ready(sender, gui):

            @controller.supervisor.actions.add(
                shortcut='alt+shift+q',
                prompt=True,
                n_args=2,
            )
            def noisy_period_spikes_from_all(t1, t2):
                """Isolate spikes from ALL clusters between t1 and t2 (seconds).

                Enter two time-points: ``t1 t2``
                """
                t1, t2 = float(t1), float(t2)
                if t1 >= t2:
                    logger.warning("NoisyPeriodSpikes: t1 (%.3f) must be less than t2 (%.3f)", t1, t2)
                    return

                clu_ids = np.unique(controller.model.spike_clusters)
                spike_times_all = controller.model.spike_times
                n_modified = 0

                for clu in clu_ids:
                    spike_ids_clu = controller.model.get_cluster_spikes(clu)
                    spike_times_clu = spike_times_all[spike_ids_clu]

                    noisy_mask = (spike_times_clu > t1) & (spike_times_clu < t2)
                    n_noisy = int(noisy_mask.sum())

                    if n_noisy > 0:
                        labels = np.where(noisy_mask, 1, 0)
                        controller.supervisor.actions.split(spike_ids_clu, labels)
                        n_modified += 1
                        logger.info("NoisyPeriodSpikes: cluster %d — split %d spikes", clu, n_noisy)

                logger.info("NoisyPeriodSpikes: finished. Modified %d / %d clusters in [%.3f, %.3f] s",
                            n_modified, len(clu_ids), t1, t2)
