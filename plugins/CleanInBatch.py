"""Batch Processing Plugin for Phy.

Shortcuts:
- Shift+X: Batch Mahalanobis splitting on high firing rate clusters (default threshold: 9)
- Shift+K: Batch K-means clustering on 'review' clusters (default: 2 clusters)
- Shift+I: Batch short ISI analysis on 'review' clusters
- Shift+Alt+K: Batch KlustaKwik reclustering on 'review' clusters
"""

from phy import IPlugin, connect
import numpy as np
import logging
from scipy.cluster.vq import kmeans2, whiten
import os
import platform
from subprocess import Popen

# Import optimized cell metrics
import CellMetrics
from CellMetrics import analyze_cluster_quality

logger = logging.getLogger('phy')

try:
    import pandas as pd
except ImportError:
    logger.warn("Package pandas not installed.")
    
try:
    from phy.utils.config import phy_config_dir
except ImportError:
    logger.warn("phy_config_dir not available.")


class CleanInBatch(IPlugin):
    def attach_to_controller(self, controller):
        """Attach the Mahalanobis splitter action to the controller."""
        @connect
        def on_gui_ready(sender, gui):
            @controller.supervisor.actions.add(shortcut='shift+x', prompt=True, prompt_default=lambda: "9")
            def Batch_Mahalanobis(threshold_str):
                """Split clusters based on Mahalanobis distance outliers.
                
                Provide threshold value (default: 9)
                Only processes clusters with firing rate > 2Hz.
                """
                logger.info("Starting Mahalanobis distance-based splitting...")
                
                # Parse threshold from input
                try:
                    mahalanobis_threshold = float(threshold_str)
                    logger.info(f"Using threshold: {mahalanobis_threshold}")
                except ValueError:
                    mahalanobis_threshold = 9.0
                    logger.info(f"Invalid input. Using default threshold: {mahalanobis_threshold}")
                
                # Fixed firing rate threshold
                firing_rate_thresh = 2.0
                
                try:
                    # Get all cluster IDs and filter by firing rate
                    all_cluster_ids = controller.supervisor.clustering.cluster_ids
                    high_fr_clusters = []
                    spike_times = controller.model.spike_times
                    
                    # Find high firing rate clusters
                    for cid in all_cluster_ids:
                        spike_ids = controller.supervisor.clustering.spikes_in_clusters([cid])
                        if len(spike_ids) > 1:
                            cluster_times = spike_times[spike_ids]
                            duration = cluster_times[-1] - cluster_times[0]
                            if duration > 0:
                                firing_rate = len(cluster_times) / duration
                                if firing_rate > firing_rate_thresh:
                                    high_fr_clusters.append(cid)
                    
                    # Process only high firing rate clusters
                    if not high_fr_clusters:
                        logger.warn(f"No clusters found with firing rate > {firing_rate_thresh} Hz")
                        return
                        
                    logger.info(f"Processing {len(high_fr_clusters)} clusters with firing rate > {firing_rate_thresh} Hz")
                    
                    # Track results
                    total_outliers_found = 0
                    clusters_processed = 0
                    
                    # Process each high firing rate cluster
                    for cid in high_fr_clusters:
                        # Get spikes for this cluster
                        spike_ids = controller.supervisor.clustering.spikes_in_clusters([cid])
                        
                        if len(spike_ids) < 10:
                            continue
                            
                        # Load features
                        data = controller.model._load_features().data[spike_ids]
                        reshaped_data = np.reshape(data, (data.shape[0], -1))
                        
                        if reshaped_data.shape[0] <= reshaped_data.shape[1]:
                            continue

                        # Calculate Mahalanobis distance (using original method)
                        def mahalanobis_dist_calc(X):
                            """Calculate Mahalanobis distance for each sample in X."""
                            mean_vec = np.mean(X, axis=0)
                            try:
                                cov_matrix = np.cov(X, rowvar=False)
                                inv_cov_matrix = np.linalg.inv(cov_matrix)
                            except np.linalg.LinAlgError:
                                logger.error(f"Singular covariance matrix for cluster {cid}. Skipping Mahalanobis calculation.")
                                return np.zeros(X.shape[0])
                            diff = X - mean_vec
                            md = np.sqrt(np.sum(diff @ inv_cov_matrix * diff, axis=1))
                            return md

                        MD = mahalanobis_dist_calc(reshaped_data)
                        outlier_indices = np.where(MD > mahalanobis_threshold)[0]
                        
                        # Split if outliers found
                        if len(outlier_indices) > 0:
                            labels = np.ones(len(spike_ids), dtype=np.int64)
                            labels[outlier_indices] = 2
                            controller.supervisor.actions.split(spike_ids, labels)
                            logger.info(f"Split {len(outlier_indices)} outliers in cluster {cid}")
                            total_outliers_found += len(outlier_indices)
                            clusters_processed += 1

                    logger.info(f"Completed: {total_outliers_found} outliers in {clusters_processed} high firing rate clusters")

                except Exception as e:
                    logger.error(f"Error: {str(e)}")

            @controller.supervisor.actions.add(shortcut='shift+k', prompt=True, prompt_default=lambda: "2")
            def Batch_KMeans(kmeanclusters_str):
                """Run K-means clustering on all 'review' clusters in batch.
                
                Provide number of clusters (default: 2)
                """
                logger.info("Starting batch K-means clustering on review clusters...")
                
                # Parse number of clusters from input
                try:
                    n_clusters = int(kmeanclusters_str)
                    logger.info(f"Using {n_clusters} clusters for K-means")
                except ValueError:
                    n_clusters = 2
                    logger.info(f"Invalid input. Using default: {n_clusters} clusters")
                
                try:
                    # Get all cluster IDs and filter by 'review' group
                    all_cluster_ids = controller.supervisor.clustering.cluster_ids
                    review_clusters = []
                    
                    for cid in all_cluster_ids:
                        group_label = controller.supervisor.cluster_meta.get('group', cid)
                        if group_label == 'review':
                            review_clusters.append(cid)
                    
                    if not review_clusters:
                        logger.warn("No 'review' clusters found")
                        return
                    
                    logger.info(f"Processing {len(review_clusters)} review clusters with K-means (k={n_clusters})")
                    
                    # Process each review cluster
                    clusters_processed = 0
                    
                    for cid in review_clusters:
                        # Get spikes for this cluster
                        spike_ids = controller.supervisor.clustering.spikes_in_clusters([cid])
                        
                        if len(spike_ids) < n_clusters:
                            logger.info(f"Cluster {cid} has too few spikes ({len(spike_ids)}), skipping")
                            continue
                        
                        # Load features
                        data = controller.model._load_features().data[spike_ids]
                        data2 = np.reshape(data, (data.shape[0], data.shape[1] * data.shape[2]))
                        
                        # Whiten and cluster
                        whitened = whiten(data2)
                        clusters_out, label = kmeans2(whitened, n_clusters)
                        
                        # Split the cluster
                        controller.supervisor.actions.split(spike_ids, label)
                        logger.info(f"K-means split cluster {cid} into {n_clusters} groups")
                        clusters_processed += 1
                    
                    logger.info(f"Completed: K-means clustering on {clusters_processed} review clusters")
                
                except Exception as e:
                    logger.error(f"Error in batch K-means: {str(e)}")

            @controller.supervisor.actions.add(shortcut='shift+i')
            def Batch_ViolatedISI():
                """Analyze and split short ISI violations on all 'review' clusters in batch.
                
                Uses spike times, amplitudes, and waveforms to detect suspicious spikes.
                Uses optimized CellMetrics module.
                """
                logger.info("Starting batch short ISI analysis on review clusters...")
                
                try:
                    # Get all cluster IDs and filter by 'review' group
                    all_cluster_ids = controller.supervisor.clustering.cluster_ids
                    review_clusters = []
                    
                    for cid in all_cluster_ids:
                        group_label = controller.supervisor.cluster_meta.get('group', cid)
                        if group_label == 'review':
                            review_clusters.append(cid)
                    
                    if not review_clusters:
                        logger.warn("No 'review' clusters found")
                        return
                    
                    logger.info(f"Processing {len(review_clusters)} review clusters for short ISI analysis")
                    
                    # Process each review cluster
                    clusters_processed = 0
                    total_suspicious = 0
                    
                    for cid in review_clusters:
                        # Get spikes for this cluster
                        spike_ids = controller.supervisor.clustering.spikes_in_clusters([cid])
                        
                        if len(spike_ids) < 10:
                            continue
                        
                        # Get spike times
                        spike_times = controller.model.spike_times[spike_ids]
                        
                        # Get amplitudes
                        bunchs = controller._amplitude_getter([cid], name='template', load_all=True)
                        spike_amps = bunchs[0].amplitudes
                        
                        # Get waveforms
                        data = controller.model._load_features().data[spike_ids]
                        waveforms = np.reshape(data, (data.shape[0], -1))
                        
                        # OPTIMIZED: Use CellMetrics for analysis
                        results = analyze_cluster_quality(spike_times, spike_amps, waveforms)
                        suspicious = results['suspicious_spikes']
                        n_suspicious = results['n_suspicious']
                        
                        # Split if found enough suspicious spikes
                        if n_suspicious >= 10 and n_suspicious <= len(spike_ids) * 0.5:
                            labels = np.ones(len(spike_ids), dtype=int)
                            labels[suspicious] = 2
                            controller.supervisor.actions.split(spike_ids, labels)
                            logger.info(f"Cluster {cid}: split {n_suspicious} suspicious spikes ({n_suspicious/len(spike_ids)*100:.1f}%)")
                            clusters_processed += 1
                            total_suspicious += n_suspicious
                    
                    logger.info(f"Completed: Analyzed ISI on {len(review_clusters)} clusters, split {clusters_processed} clusters ({total_suspicious} suspicious spikes)")
                
                except Exception as e:
                    logger.error(f"Error in batch short ISI: {str(e)}")

            @controller.supervisor.actions.add(shortcut='shift+alt+k')
            def Batch_KlustaKwik():
                """Run KlustaKwik reclustering on all 'review' clusters in batch.
                
                Uses KlustaKwik algorithm for automatic clustering.
                """
                logger.info("Starting batch KlustaKwik reclustering on review clusters...")
                
                def write_fet(fet, filepath):
                    """Write features to .fet file for KlustaKwik"""
                    with open(filepath, 'w') as fd:
                        fd.write('%i\n' % fet.shape[1])
                        for x in range(0, fet.shape[0]):
                            fet[x, :].tofile(fd, sep="\t", format="%i")
                            fd.write("\n")
                
                def read_clusters(filename_clu):
                    """Read cluster assignments from .clu file"""
                    clusters = load_text(filename_clu, np.int64)
                    return process_clusters(clusters)
                
                def process_clusters(clusters):
                    """Remove first line (number of clusters) from .clu file"""
                    return clusters[1:]
                
                def load_text(filepath, dtype, skiprows=0, delimiter=' '):
                    """Load text file into numpy array"""
                    if not filepath:
                        raise IOError("The filepath is empty.")
                    with open(filepath, 'r') as f:
                        for _ in range(skiprows):
                            f.readline()
                        x = pd.read_csv(f, header=None, sep=delimiter).values.astype(dtype).squeeze()
                    return x
                
                try:
                    # Get all cluster IDs and filter by 'review' group
                    all_cluster_ids = controller.supervisor.clustering.cluster_ids
                    review_clusters = []
                    
                    for cid in all_cluster_ids:
                        group_label = controller.supervisor.cluster_meta.get('group', cid)
                        if group_label == 'review':
                            review_clusters.append(cid)
                    
                    if not review_clusters:
                        logger.warn("No 'review' clusters found")
                        return
                    
                    logger.info(f"Processing {len(review_clusters)} review clusters with KlustaKwik")
                    
                    # Process each review cluster
                    clusters_processed = 0
                    
                    for cid in review_clusters:
                        # Get spikes for this cluster
                        spike_ids = controller.supervisor.clustering.spikes_in_clusters([cid])
                        
                        if len(spike_ids) < 20:
                            logger.info(f"Cluster {cid} has too few spikes ({len(spike_ids)}), skipping")
                            continue
                        
                        logger.info(f"Running KlustaKwik on cluster {cid} with {len(spike_ids)} spikes")
                        
                        # Load and prepare features
                        data3 = controller.model._load_features().data[spike_ids]
                        fet2 = np.reshape(data3, (data3.shape[0], data3.shape[1] * data3.shape[2]))
                        
                        # Convert to integer format for KlustaKwik
                        dtype = np.int64
                        factor = 2.**60 / np.abs(fet2).max()
                        fet2 = (fet2 * factor).astype(dtype)
                        
                        # Write features to temporary file
                        name = f'tempClustering_cluster{cid}'
                        shank = 3
                        mainfetfile = os.path.join(name + '.fet.' + str(shank))
                        write_fet(fet2, mainfetfile)
                        
                        # Set up KlustaKwik command
                        if platform.system() == 'Windows':
                            program = os.path.join(phy_config_dir(), 'klustakwik.exe')
                        else:
                            program = '~/klustakwik/KlustaKwik'
                        
                        cmd = [program, name, str(shank)]
                        cmd += ["-UseDistributional", '0', "-MaxPossibleClusters", '20', "-MinClusters", '20']
                        
                        # Run KlustaKwik
                        p = Popen(cmd)
                        p.wait()
                        
                        # Read back the clusters and split
                        spike_clusters = read_clusters(name + '.clu.' + str(shank))
                        controller.supervisor.actions.split(spike_ids, spike_clusters)
                        
                        # Clean up temporary files
                        try:
                            os.remove(mainfetfile)
                            os.remove(name + '.clu.' + str(shank))
                        except:
                            pass
                        
                        logger.info(f"KlustaKwik reclustering complete for cluster {cid}")
                        clusters_processed += 1
                    
                    logger.info(f"Completed: KlustaKwik reclustering on {clusters_processed} review clusters")
                
                except Exception as e:
                    logger.error(f"Error in batch KlustaKwik: {str(e)}")