"""Deduplicate near-coincident spikes by keeping the best one.

This plugin adds an action that is useful after merging two clusters that
contain duplicate detections at nearly the same timestamp.

Behavior
--------
For each selected cluster, spikes are grouped into conflict windows where
consecutive spikes are separated by <= interval_ms. Within each conflict group,
the spike with the largest absolute amplitude is kept and all other spikes are
split out into a separate cluster.

Shortcut
--------
Alt+Shift+D : Deduplicate selected cluster(s) using prompt interval (ms).

Notes
-----
1) ``SplitShortISI`` (``SplitShortISI.py``)
   - Trigger: fixed short ISI threshold (1.5 ms).
   - Rule: mark spikes based on timing-only refractory violations.
   - Intended use: visualization/inspection of suspicious short-ISI patterns.
   - Caveat: does not choose a "best" spike by amplitude.

2) ``DeduplicateKeepBest`` (``DeduplicateKeepBest.py``)
   - Trigger: user-defined interval (default 0.5 ms).
   - Rule: within each conflict run, keep the spike with max ``|amplitude|`` and
     split out the others.
   - Intended use: deduplication-style cleanup after merges that create
     near-synchronous duplicate detections.

Practical guidance
------------------
- Use ``SplitShortISI`` when you want to *inspect* refractory violations.
- Use ``DeduplicateKeepBest`` when you want to *resolve* duplicates while
  preserving one representative spike per conflict interval.
"""

from __future__ import annotations

import logging

import numpy as np
from phy import IPlugin, connect

logger = logging.getLogger("phy")


class DeduplicateKeepBest(IPlugin):
    """Keep best spike in each short-interval conflict group."""

    def __init__(self):
        super().__init__()
        self._ready_controller_ids: set[int] = set()

    def attach_to_controller(self, controller):
        controller_id = id(controller)
        if controller_id in self._ready_controller_ids:
            return
        self._ready_controller_ids.add(controller_id)

        @connect
        def on_gui_ready(sender, gui):
            if sender is not controller:
                return

            supervisor = getattr(controller, "supervisor", None)
            if supervisor is None:
                return

            @supervisor.actions.add(
                shortcut="alt+shift+d", prompt=True, prompt_default=lambda: "0.5"
            )
            def deduplicate_keep_best(interval_ms_str):
                """Split near-coincident duplicates, keeping only the best-amplitude spike.

                Enter interval in milliseconds (default: 0.5).

                Distinction from ``SplitShortISI``:
                this action uses amplitude-ranked conflict resolution rather than
                timing-only short-ISI marking.
                """

                try:
                    interval_ms = float(interval_ms_str)
                except (TypeError, ValueError):
                    interval_ms = 0.5
                    logger.warning(
                        "Invalid interval '%s'. Falling back to %.3f ms.",
                        interval_ms_str,
                        interval_ms,
                    )

                if interval_ms <= 0:
                    logger.warning("Interval must be > 0 ms. Got %.6f ms.", interval_ms)
                    return

                interval_s = interval_ms / 1000.0
                selected_clusters = list(getattr(supervisor, "selected", []) or [])
                if not selected_clusters:
                    logger.warning("No cluster selected for deduplication.")
                    return

                total_removed = 0
                total_processed = 0

                for cluster_id in selected_clusters:
                    spike_ids = np.asarray(
                        supervisor.clustering.spikes_in_clusters([cluster_id]),
                        dtype=np.int64,
                    )
                    if spike_ids.size < 2:
                        continue

                    spike_times = np.asarray(
                        controller.model.spike_times[spike_ids], dtype=np.float64
                    )
                    amplitudes = self._get_spike_amplitudes(
                        controller, cluster_id, spike_ids
                    )

                    order = np.argsort(spike_times, kind="mergesort")
                    spike_ids_sorted = spike_ids[order]
                    spike_times_sorted = spike_times[order]
                    amplitudes_sorted = np.abs(amplitudes[order])

                    remove_mask_sorted = self._build_remove_mask(
                        spike_times_sorted,
                        amplitudes_sorted,
                        interval_s,
                    )
                    n_remove = int(remove_mask_sorted.sum())
                    if n_remove <= 0:
                        logger.info(
                            "Cluster %s: no duplicates found within %.3f ms.",
                            cluster_id,
                            interval_ms,
                        )
                        continue

                    labels = np.ones(spike_ids_sorted.shape[0], dtype=np.int64)
                    labels[remove_mask_sorted] = 2

                    supervisor.actions.split(spike_ids_sorted, labels)
                    total_removed += n_remove
                    total_processed += 1
                    logger.info(
                        "Cluster %s: split out %d duplicate spikes (interval %.3f ms).",
                        cluster_id,
                        n_remove,
                        interval_ms,
                    )

                if total_processed == 0:
                    msg = f"No duplicates found within {interval_ms:.3f} ms"
                else:
                    msg = f"Deduplicated {total_removed} spikes across {total_processed} cluster(s)"

                logger.info(msg)
                status_message = getattr(gui, "status_message", None)
                if callable(status_message):
                    try:
                        status_message(msg)
                    except Exception:
                        pass

    @staticmethod
    def _build_remove_mask(
        spike_times_sorted: np.ndarray,
        amplitudes_sorted: np.ndarray,
        interval_s: float,
    ) -> np.ndarray:
        """Mark spikes to remove while keeping max-|amplitude| in each conflict run."""
        n_spikes = spike_times_sorted.size
        remove_mask = np.zeros(n_spikes, dtype=bool)

        i = 0
        while i < n_spikes:
            j = i + 1
            while (
                j < n_spikes
                and (spike_times_sorted[j] - spike_times_sorted[j - 1]) <= interval_s
            ):
                j += 1

            if (j - i) > 1:
                segment_scores = amplitudes_sorted[i:j]
                keep_rel = int(np.argmax(segment_scores))
                remove_mask[i:j] = True
                remove_mask[i + keep_rel] = False

            i = j

        return remove_mask

    @staticmethod
    def _get_spike_amplitudes(
        controller, cluster_id: int, spike_ids: np.ndarray
    ) -> np.ndarray:
        """Best-effort spike amplitudes aligned to spike_ids; defaults to ones."""
        model = getattr(controller, "model", None)
        if model is not None:
            model_amps = getattr(model, "amplitudes", None)
            if model_amps is not None:
                try:
                    amps = np.asarray(model_amps[spike_ids], dtype=np.float64)
                    if amps.shape[0] == spike_ids.shape[0]:
                        return amps
                except Exception:
                    pass

        try:
            bunches = controller._amplitude_getter(
                [cluster_id], name="template", load_all=True
            )
            if bunches:
                bunch = bunches[0]
                b_ids = np.asarray(getattr(bunch, "spike_ids", []), dtype=np.int64)
                b_amps = np.asarray(getattr(bunch, "amplitudes", []), dtype=np.float64)
                if b_ids.size == b_amps.size and b_ids.size > 0:
                    if np.array_equal(b_ids, spike_ids):
                        return b_amps

                    order = np.argsort(b_ids, kind="mergesort")
                    ids_sorted = b_ids[order]
                    amps_sorted = b_amps[order]
                    pos = np.searchsorted(ids_sorted, spike_ids)
                    valid = (pos < ids_sorted.size) & (ids_sorted[pos] == spike_ids)
                    out = np.ones(spike_ids.shape[0], dtype=np.float64)
                    out[valid] = amps_sorted[pos[valid]]
                    return out
        except Exception:
            pass

        return np.ones(spike_ids.shape[0], dtype=np.float64)
