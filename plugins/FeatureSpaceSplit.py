"""FeatureSpaceSplit — MeanShift and Gaussian Mixture Model splitting on channel PCs.

This plugin provides two feature-space clustering algorithms for splitting
a selected cluster based on user-specified channel principal components:

1. **MeanShift** (Alt+N): A non-parametric clustering algorithm that
   automatically determines the number of sub-clusters by finding modes
   (density peaks) in the feature space. Useful when you suspect multiple
   units but don't know how many.

2. **Gaussian Mixture** (Alt+Shift+F): Fits a Gaussian Mixture Model (GMM)
   with a user-specified number of components. More flexible than K-Means
   because it models elliptical (covariance-aware) cluster shapes and
   supports soft assignment.

Usage:
------
1. Enable 'FeatureSpaceSplit' in your phy_config.py plugins list.
2. Select a cluster in the ClusterView.

3. **MeanShift** — press Alt+N and enter:
   ``channel1  pc_component1  channel2  pc_component2``
   Example: ``139 1 133 1`` splits using PC1 of channels 139 and 133.

4. **Gaussian Mixture** — press Alt+Shift+F and enter:
   ``channel1  channel2  n_components``
   Example: ``139 133 3`` fits a 3-component GMM using all PCs of channels
   139 and 133.

Notes:
------
- Channel numbers refer to the *original* channel IDs (as shown in the
  ChannelMap), not 0-indexed positions.
- PC component indices are 1-based (1 = first PC, 2 = second PC, etc.).
- MeanShift ``bandwidth`` is set to 2 by default; tune if your feature
  scale differs significantly.
- The operations are undoable via Ctrl+Z in phy.
"""

import logging

import numpy as np
from sklearn.cluster import MeanShift
from sklearn.mixture import GaussianMixture

from phy import IPlugin, connect

logger = logging.getLogger('phy')


def _resolve_channel_index(controller, channel_id):
    """Map an original channel ID to its index in the channel mapping array.

    Returns the index (or indices) where ``controller.model.channel_mapping``
    equals *channel_id*.
    """
    return np.where(controller.model.channel_mapping == int(channel_id))[0]


class FeatureSpaceSplit(IPlugin):
    """Split clusters using MeanShift or Gaussian Mixture on selected channel PCs."""

    def attach_to_controller(self, controller):

        @connect
        def on_gui_ready(sender, gui):

            @controller.supervisor.actions.add(
                submenu='Clustering',
                shortcut='alt+n',
                prompt=True,
                n_args=4,
            )
            def mean_shift_split(chan1, comp1, chan2, comp2):
                """Split using MeanShift clustering on two channel PCs.

                Enter: ``channel1  pc1  channel2  pc2``
                """
                chan1, comp1, chan2, comp2 = int(chan1), int(comp1), int(chan2), int(comp2)

                cluster_ids = controller.supervisor.selected
                if not cluster_ids:
                    logger.warning("FeatureSpaceSplit: no cluster selected.")
                    return

                channel1_idx = _resolve_channel_index(controller, chan1)
                channel2_idx = _resolve_channel_index(controller, chan2)

                if len(channel1_idx) == 0 or len(channel2_idx) == 0:
                    logger.warning("FeatureSpaceSplit: channel %d or %d not found in channel mapping.",
                                   chan1, chan2)
                    return

                logger.info("MeanShift: cluster %d, channels %d (PC%d) & %d (PC%d)",
                            cluster_ids[0], chan1, comp1, chan2, comp2)

                m1 = controller._get_features(cluster_ids[0], channel1_idx, load_all=True)
                m2 = controller._get_features(cluster_ids[0], channel2_idx, load_all=True)

                a1 = m1.data[:, :, comp1 - 1]
                a2 = m2.data[:, :, comp2 - 1]
                features = np.concatenate((a1, a2), axis=1)

                labels = MeanShift(bandwidth=2).fit_predict(features)
                spike_ids = m1.spike_ids

                n_clusters_found = len(np.unique(labels))
                logger.info("MeanShift: found %d sub-clusters from %d spikes",
                            n_clusters_found, len(spike_ids))

                assert spike_ids.shape == labels.shape
                controller.supervisor.actions.split(spike_ids, labels)

        @connect
        def on_gui_ready(sender, gui):

            @controller.supervisor.actions.add(
                submenu='Clustering',
                shortcut='alt+shift+f',
                prompt=True,
                n_args=3,
            )
            def gaussian_mixture_split(chan1, chan2, n_components):
                """Split using Gaussian Mixture Model on two channels (all PCs).

                Enter: ``channel1  channel2  n_components``
                """
                chan1, chan2, n_components = int(chan1), int(chan2), int(n_components)

                cluster_ids = controller.supervisor.selected
                if not cluster_ids:
                    logger.warning("FeatureSpaceSplit: no cluster selected.")
                    return

                channel1_idx = _resolve_channel_index(controller, chan1)
                channel2_idx = _resolve_channel_index(controller, chan2)

                if len(channel1_idx) == 0 or len(channel2_idx) == 0:
                    logger.warning("FeatureSpaceSplit: channel %d or %d not found in channel mapping.",
                                   chan1, chan2)
                    return

                logger.info("GaussianMixture: cluster %d, channels %d & %d, n_components=%d",
                            cluster_ids[0], chan1, chan2, n_components)

                m1 = controller._get_features(cluster_ids[0], channel1_idx, load_all=True)
                m2 = controller._get_features(cluster_ids[0], channel2_idx, load_all=True)

                a1 = m1.data[:, :, :].squeeze()
                a2 = m2.data[:, :, :].squeeze()
                features = np.concatenate((a1, a2), axis=1)

                labels = GaussianMixture(n_components=n_components).fit_predict(features)
                spike_ids = m1.spike_ids

                logger.info("GaussianMixture: split %d spikes into %d components",
                            len(spike_ids), n_components)

                assert spike_ids.shape == labels.shape
                controller.supervisor.actions.split(spike_ids, labels)
