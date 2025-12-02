"""AutoQuality Plugin for Phy 2.0b6 - Enhanced with CellMetrics.

This plugin automatically assigns cluster quality labels using 'group' metadata.
Uses relaxed thresholds for first-pass curation - labels clusters as:
- 'good': Clean clusters with minimal issues
- 'review': Borderline clusters that need manual curation (crimson color)
- 'mua': Multi-unit activity with moderate issues
- 'noise': Very few spikes or severe quality issues

Shortcut: Shift+L

Quality is assessed using three CellMetrics features:
1. ISI violations (refractory period violations)
2. Amplitude stability (spike amplitude variance)
3. Waveform consistency (waveform shape variance)

Thresholds are intentionally relaxed to allow false positives - the goal is to
give users clusters to curate, not to weed out everything automatically.
"""

from phy import IPlugin, connect
import numpy as np
import logging
import sys
from pathlib import Path

# Add plugins directory to path for CellMetrics import
plugins_dir = Path(__file__).parent
if str(plugins_dir) not in sys.path:
    sys.path.insert(0, str(plugins_dir))

# Import optimized cell metrics
import CellMetrics
from CellMetrics import analyze_cluster_quality, count_isi_violations, compute_amplitude_cutoff

logger = logging.getLogger('phy')


class AutoQuality(IPlugin):
    def attach_to_controller(self, controller):
        """Attach the AutoLabel action to the controller."""
        @connect
        def on_gui_ready(sender, gui):
            @controller.supervisor.actions.add(shortcut='shift+l')
            def Auto_Quality():
                """Automatically label clusters as 'good', 'mua', 'noise', or 'review' using group labels (first pass - relaxed thresholds)."""
                logger.info("Starting automatic cluster quality labeling with CellMetrics...")

                try:
                    # RELAXED Quality thresholds for first pass curation
                    # Goal: Label obviously good and obviously bad, put borderline cases in 'review' for user curation
                    min_spike_count = 200           # Very low threshold - anything less is clearly noise
                    good_spike_count = 800          # Lower threshold - more lenient for 'good' (was 1000)
                    max_isi_violations_good = 0.01  # 1% ISI violations for 'good' (was 0.5% - more relaxed)
                    max_isi_violations_mua = 0.05   # 5% ISI violations for 'mua' (was 2% - more relaxed)
                    max_amplitude_cutoff = 0.15     # 15% amplitude cutoff (was 10% - more relaxed)
                    review_spike_count = 500        # Lower threshold for review consideration (was 1000)
                    waveform_threshold = 0.20       # Waveform variance threshold (relaxed)
                    
                    # Retrieve all cluster IDs
                    cluster_ids = controller.supervisor.clustering.cluster_ids
                    unique_clusters = np.unique(cluster_ids)
                    logger.debug(f"Found {len(unique_clusters)} unique clusters.")

                    # Initialize stats
                    stats_summary = {'good': 0, 'mua': 0, 'noise': 0, 'review': 0}

                    # Iterate over each cluster
                    for cid in unique_clusters:
                        spike_ids = controller.supervisor.clustering.spikes_per_cluster[cid]
                        spike_count = len(spike_ids)

                        # Noise: Too few spikes
                        if spike_count < min_spike_count:
                            controller.supervisor.cluster_meta.set('group', cid, 'noise')
                            stats_summary['noise'] += 1
                            logger.info(f"Cluster {cid}: {spike_count} spikes -> 'noise' (too few spikes)")
                            continue

                        # Get spike times and amplitudes
                        spike_times = controller.model.spike_times[spike_ids]
                        
                        # Get spike amplitudes
                        try:
                            spike_amps = controller.model.amplitudes[spike_ids]
                        except:
                            # If amplitudes not available, use a dummy array
                            spike_amps = np.ones(spike_count, dtype=np.float64)
                        
                        # Get waveforms if available
                        try:
                            waveforms = controller.model.get_waveforms(spike_ids)
                            if waveforms is None or len(waveforms) == 0:
                                waveforms = None
                        except:
                            waveforms = None

                        # Analyze cluster quality using CellMetrics (ISI, Amplitude, Waveform)
                        metrics = analyze_cluster_quality(
                            spike_times=spike_times,
                            spike_amps=spike_amps,
                            waveforms=waveforms
                        )

                        # Extract all CellMetrics features
                        isi_violations = metrics['isi_violations']
                        total_isis = spike_count - 1 if spike_count > 1 else 1
                        isi_violation_ratio = isi_violations / total_isis
                        
                        amplitude_cutoff = metrics.get('amplitude_cutoff', 0.0)
                        n_suspicious = metrics['n_suspicious']
                        suspicious_ratio = n_suspicious / spike_count if spike_count > 0 else 0
                        suspicious_spikes = metrics.get('suspicious_spikes', [])
                        
                        # Count issues detected by each feature (more relaxed thresholds)
                        has_isi_issues = isi_violation_ratio > max_isi_violations_good
                        has_amplitude_issues = amplitude_cutoff > max_amplitude_cutoff
                        has_waveform_issues = suspicious_ratio > waveform_threshold
                        quality_issues = sum([has_isi_issues, has_amplitude_issues, has_waveform_issues])

                        # RELAXED Decision logic - first pass to create 'good', 'review', 'mua', 'noise' categories
                        # Goal: Don't weed out everything, give user clusters to curate
                        if spike_count >= good_spike_count:
                            # High spike count: Be lenient - most should be good or review
                            if quality_issues == 0:
                                # Perfect: no issues in any feature
                                controller.supervisor.cluster_meta.set('group', cid, 'good')
                                stats_summary['good'] += 1
                                logger.info(f"Cluster {cid}: {spike_count} spikes, ISI:{isi_violation_ratio*100:.3f}%, Amp:{amplitude_cutoff*100:.1f}%, Wave:{suspicious_ratio*100:.1f}% -> 'good'")
                            elif quality_issues == 1 or (quality_issues == 2 and not has_isi_issues):
                                # 1-2 minor issues (not ISI) - still good enough for first pass
                                controller.supervisor.cluster_meta.set('group', cid, 'good')
                                stats_summary['good'] += 1
                                logger.info(f"Cluster {cid}: {spike_count} spikes, {quality_issues} minor issue(s) -> 'good'")
                            elif isi_violation_ratio <= max_isi_violations_mua:
                                # Moderate issues but not terrible ISI - mark for review
                                controller.supervisor.cluster_meta.set('group', cid, 'review')
                                stats_summary['review'] += 1
                                logger.info(f"Cluster {cid}: {spike_count} spikes, {quality_issues} issues -> 'review'")
                            else:
                                # Severe ISI violations - likely MUA or noise
                                controller.supervisor.cluster_meta.set('group', cid, 'mua')
                                stats_summary['mua'] += 1
                                logger.info(f"Cluster {cid}: ISI:{isi_violation_ratio*100:.3f}%, severe violations -> 'mua'")
                        
                        elif spike_count >= review_spike_count:
                            # Mid-range spike count: Most should be review or good (lenient)
                            if quality_issues >= 3:
                                # All features problematic - likely MUA
                                controller.supervisor.cluster_meta.set('group', cid, 'mua')
                                stats_summary['mua'] += 1
                                logger.info(f"Cluster {cid}: {spike_count} spikes, all features bad -> 'mua'")
                            elif quality_issues >= 2 or (has_isi_issues and suspicious_ratio > 0.4):
                                # Multiple issues - review needed
                                controller.supervisor.cluster_meta.set('group', cid, 'review')
                                stats_summary['review'] += 1
                                logger.info(f"Cluster {cid}: {spike_count} spikes, ISI:{isi_violation_ratio*100:.3f}%, Wave:{suspicious_ratio*100:.1f}% -> 'review'")
                            else:
                                # Clean or minor issues - good for first pass
                                controller.supervisor.cluster_meta.set('group', cid, 'good')
                                stats_summary['good'] += 1
                                logger.info(f"Cluster {cid}: {spike_count} spikes, clean/minor issues -> 'good'")
                        
                        else:
                            # Low spike count (200-500): Be conservative - likely MUA or noise
                            if isi_violation_ratio > max_isi_violations_mua or quality_issues >= 2:
                                controller.supervisor.cluster_meta.set('group', cid, 'noise')
                                stats_summary['noise'] += 1
                                logger.info(f"Cluster {cid}: {spike_count} spikes, multiple issues -> 'noise'")
                            else:
                                controller.supervisor.cluster_meta.set('group', cid, 'mua')
                                stats_summary['mua'] += 1
                                logger.info(f"Cluster {cid}: {spike_count} spikes -> 'mua'")

                    # Log summary
                    logger.info(f"Auto-Quality Summary: {stats_summary['good']} good, {stats_summary['mua']} mua, "
                               f"{stats_summary['noise']} noise, {stats_summary['review']} review")
                    logger.info("Automatic cluster labeling completed (using 'group' labels).")

                except AttributeError as ae:
                    logger.error(f"Attribute error during automatic cluster labeling: {ae}")
                except Exception as e:
                    logger.error(f"Unexpected error during automatic cluster labeling: {e}")
