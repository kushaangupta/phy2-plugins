"""IsolationForestSplit — Unsupervised outlier detection using Isolation Forest.

This plugin applies sklearn's Isolation Forest algorithm to the flattened
PC-feature space of a selected cluster, automatically identifying and
splitting off outlier spikes. Unlike Mahalanobis distance (which assumes
a unimodal Gaussian distribution), Isolation Forest is non-parametric and
handles multimodal, skewed, or irregularly shaped feature distributions.

How It Works:
-------------
Isolation Forest isolates observations by randomly selecting a feature and
then randomly selecting a split value between the maximum and minimum values
of that feature. Outliers — points that are few and different — require fewer
random splits to be isolated, so they receive shorter average path lengths
in the ensemble of isolation trees. Points with short average path lengths
are flagged as anomalies.

Usage:
------
1. Enable 'IsolationForestSplit' in your phy_config.py plugins list.
2. Select a cluster in the ClusterView.
3. Press Shift+O to run Isolation Forest outlier detection.
4. Detected outlier spikes are split into a new cluster for review.

Notes:
------
- Uses all PC features from the selected cluster (flattened across channels).
- No user parameters required — the algorithm auto-detects the contamination
  proportion from the data.
- Complements the existing Mahalanobis-based outlier detection (Alt+X /
  Alt+Shift+X) by handling non-Gaussian cluster shapes.
- The operation is undoable via Ctrl+Z in phy.
"""

import logging

import numpy as np
from sklearn.ensemble import IsolationForest

from phy import IPlugin, connect

logger = logging.getLogger('phy')


class IsolationForestSplit(IPlugin):
    """Split outlier spikes from a cluster using Isolation Forest."""

    def attach_to_controller(self, controller):

        @connect
        def on_gui_ready(sender, gui):

            @controller.supervisor.actions.add(
                submenu='Outlier',
                shortcut='shift+o',
            )
            def isolation_forest_split():
                """Detect and split outlier spikes using Isolation Forest on PC features."""

                cluster_ids = controller.supervisor.selected
                if not cluster_ids:
                    logger.warning("IsolationForestSplit: no cluster selected.")
                    return

                cluster_data = controller._get_features(cluster_ids[0], load_all=True)
                features = cluster_data['data']
                spike_ids = cluster_data['spike_ids']

                if features.ndim == 3:
                    features = features.reshape(features.shape[0], -1)

                n_spikes = len(spike_ids)
                if n_spikes < 10:
                    logger.warning("IsolationForestSplit: cluster %d has only %d spikes — "
                                   "too few for reliable outlier detection.", cluster_ids[0], n_spikes)
                    return

                logger.info("IsolationForestSplit: running on cluster %d (%d spikes, %d features)",
                            cluster_ids[0], n_spikes, features.shape[1])

                predictions = IsolationForest(random_state=0).fit_predict(features)

                # IsolationForest returns -1 for outliers, 1 for inliers.
                # Phy expects non-negative integer labels.
                labels = np.where(predictions == -1, 2, 1)

                n_outliers = int((labels == 2).sum())
                if n_outliers == 0:
                    logger.info("IsolationForestSplit: no outliers detected in cluster %d.",
                                cluster_ids[0])
                    return

                logger.info("IsolationForestSplit: detected %d outliers (%.1f%%) in cluster %d",
                            n_outliers, 100.0 * n_outliers / n_spikes, cluster_ids[0])

                assert len(spike_ids) == len(labels)
                controller.supervisor.actions.split(spike_ids, labels)
