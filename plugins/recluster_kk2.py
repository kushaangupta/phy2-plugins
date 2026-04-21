"""
Phy2 plugin for reclustering using KlustaKwik2 (Python implementation).

This replaces the original recluster.py which shelled out to the C++ KlustaKwik
binary. Instead, it calls KlustaKwik2's KK class directly in-process.

Installation:
    1. Install klustakwik2:
            cd klustakwik2 && pip install --no-build-isolation .
    2. Copy this file (or symlink it) into your phy plugins directory,
       or add its path to your ~/.phy/phy_config.py:
           c.TemplateGUI.plugins = ['KK2ClusterPlugin']

Keyboard shortcuts (same as the original plugin):
    Alt+Ctrl+K  — Recluster selected clusters (local PCAs) with KlustaKwik2
    Alt+Ctrl+T  — Recluster selected clusters (global PCAs) with KlustaKwik2
    Alt+Ctrl+Q  — K-means clustering (unchanged logic, new shortcut)
    Alt+Ctrl+X  — Mahalanobis distance outlier removal (unchanged logic, new shortcut)
"""

from phy import IPlugin, connect

import logging
import numpy as np

from scipy.cluster.vq import whiten
from sklearn.cluster import KMeans

logger = logging.getLogger('phy')

# ---------------------------------------------------------------------------
# KlustaKwik2 helpers
# ---------------------------------------------------------------------------

def _dense_to_sparse_data(features):
    """Convert a dense feature matrix into KlustaKwik2's SparseData format.

    For phy's PC features the data is fully unmasked (no per-channel mask
    structure), so we take a fast vectorised path that avoids the per-spike
    Python loop used in the general sparse case.

    Parameters
    ----------
    features : ndarray, shape (n_spikes, n_features)
        Dense feature matrix (float).

    Returns
    -------
    klustakwik2.SparseData
        Ready to be passed to ``KK(data, ...)``.
    """
    from klustakwik2 import RawSparseData

    features = np.asarray(features, dtype=float)
    n_spikes, n_features = features.shape

    # Per-channel normalisation to [0, 1]
    vmin = features.min(axis=0)
    vmax = features.max(axis=0)
    vdiff = vmax - vmin
    vdiff[vdiff == 0] = 1.0
    normed = (features - vmin) / vdiff

    # Build the flat sparse arrays.  With all masks = 1 every spike has the
    # same unmasked index set [0, 1, ..., n_features-1], so the "sparse"
    # representation is just the row-major flattened normalised features.
    # .copy() ensures contiguous owned memory — KK2's Cython kernels keep
    # raw pointers into these buffers.
    all_features = normed.ravel().copy()
    all_fmasks   = np.ones(n_spikes * n_features)
    all_unmasked = np.tile(np.arange(n_features, dtype=int), n_spikes).copy()
    offsets      = np.arange(0, (n_spikes + 1) * n_features,
                             n_features, dtype=int)

    # noise_variance serves double duty in KK2:
    #   1. Noise model for masked features (irrelevant here, all masks=1)
    #   2. Prior regularisation on the covariance diagonal (m_step.py):
    #        block_diagonal += prior_point * noise_variance[unmasked]
    # Setting it to 0 removes ALL regularisation → Cholesky fails on real
    # data with correlated features.  We use the per-feature variance of
    # the normalised data, with a floor to handle rank-deficient data
    # (e.g. 18 features from 6 channels × 3 PCs with only ~4 independent dims).
    noise_mean     = normed.mean(axis=0)
    noise_variance = normed.var(axis=0)
    variance_floor = max(noise_variance.mean() * 0.01, 1e-5)
    noise_variance = np.maximum(noise_variance, variance_floor)

    raw = RawSparseData(noise_mean, noise_variance,
                        all_features, all_fmasks,
                        all_unmasked, offsets)
    return raw.to_sparse_data()


def _run_kk2(features, num_starting_clusters=20, **kk_params):
    """Run KlustaKwik2 on a dense feature matrix and return cluster labels.

    Parameters
    ----------
    features : ndarray, shape (n_spikes, n_features)
    num_starting_clusters : int
        Number of random initial clusters to seed the algorithm with.
    **kk_params
        Forwarded to ``KK(...)``.  Useful overrides include
        ``max_possible_clusters``, ``max_iterations``,
        ``use_noise_cluster``, ``use_mua_cluster``.

    Returns
    -------
    clusters : ndarray of int, shape (n_spikes,)
        Cluster labels (0-indexed).
    """
    from klustakwik2 import KK

    data = _dense_to_sparse_data(features)

    logger.info("KK2: %d spikes, %d features, %d unique masks",
                data.num_spikes, data.num_features, data.num_masks)

    # Dense (fully unmasked) data always has exactly 1 unique mask, so
    # mask_starts() can't generate diverse initial clusters (it needs
    # num_masks >= num_starting_clusters).  Instead we seed with random
    # cluster assignments and call cluster_from().
    #
    # points_for_cluster_mask=0 ensures every feature stays unmasked in
    # every cluster regardless of cluster size.
    kk_params.setdefault('points_for_cluster_mask', 0)

    kk = KK(data, **kk_params)

    n_spikes = data.num_spikes
    initial_clusters = np.random.randint(0, num_starting_clusters,
                                         size=n_spikes)
    kk.cluster_from(initial_clusters)

    clusters = kk.clusters
    logger.info("KK2 finished: %d clusters found", len(np.unique(clusters)))
    return clusters


# ---------------------------------------------------------------------------
# Phy plugin
# ---------------------------------------------------------------------------

class ClusterKK2(IPlugin):
    """Phy2 plugin providing KlustaKwik2-based reclustering actions."""

    def attach_to_controller(self, controller):
        @connect
        def on_gui_ready(sender, gui):

            # ---------------------------------------------------------------
            # Recluster with Local PCAs (features from the model)
            # ---------------------------------------------------------------
            @controller.supervisor.actions.add(shortcut='alt+ctrl+k')
            def Recluster_KK2_Local_PCAs():
                """Recluster selected clusters using KlustaKwik2 on local PC features."""
                cluster_ids = controller.supervisor.selected
                bunchs = controller._amplitude_getter(cluster_ids, name='template', load_all=True)
                spike_ids = bunchs[0].spike_ids
                logger.info("KK2 Local PCA: reclustering %d spikes from clusters %s",
                            len(spike_ids), cluster_ids)

                # Load per-spike PC features  (n_spikes, n_channels, n_pcs)
                data3 = controller.model._load_features().data[spike_ids]
                # Flatten to 2-D: (n_spikes, n_channels * n_pcs)
                fet = data3.reshape(data3.shape[0], -1)

                # Validate
                if np.any(np.isnan(fet)) or np.any(np.isinf(fet)):
                    logger.error("Invalid values (NaN/Inf) in feature data")
                    return
                if np.abs(fet).max() == 0:
                    logger.error("All feature values are zero")
                    return

                # KK2 parameters — mirrors the old KlustaKwik flags:
                #   -UseDistributional 0 -MaxPossibleClusters 20
                #   -MinClusters 2 -MaxClusters 20
                clusters = _run_kk2(
                    fet,
                    num_starting_clusters=20,
                    use_noise_cluster=False,
                    use_mua_cluster=False,
                    max_possible_clusters=20,
                    max_iterations=1000,
                )

                controller.supervisor.actions.split(spike_ids, clusters)
                logger.warn("KK2 Local PCA reclustering complete!")

            # ---------------------------------------------------------------
            # Recluster with Global PCAs
            # ---------------------------------------------------------------
            @controller.supervisor.actions.add(shortcut='alt+ctrl+t')
            def Recluster_KK2_Global_PCAs():
                """Recluster selected clusters using KlustaKwik2 on global PC features."""
                cluster_ids = controller.supervisor.selected
                bunchs = controller._amplitude_getter(cluster_ids, name='template', load_all=True)
                spike_ids = bunchs[0].spike_ids
                logger.info("KK2 Global PCA: reclustering %d spikes from clusters %s",
                            len(spike_ids), cluster_ids)

                data3 = controller.model._load_features().data[spike_ids]
                fet = data3.reshape(data3.shape[0], -1)

                if np.any(np.isnan(fet)) or np.any(np.isinf(fet)):
                    logger.error("Invalid values (NaN/Inf) in feature data")
                    return
                if np.abs(fet).max() == 0:
                    logger.error("All feature values are zero")
                    return

                clusters = _run_kk2(
                    fet,
                    num_starting_clusters=20,
                    use_noise_cluster=False,
                    use_mua_cluster=False,
                    max_possible_clusters=20,
                    max_iterations=1000,
                )

                controller.supervisor.actions.split(spike_ids, clusters)
                logger.warn("KK2 Global PCA reclustering complete!")
