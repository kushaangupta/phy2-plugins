"""Raw Waveform Mode Plugin for Phy

This plugin modifies how Phy fetches and displays waveforms, ensuring that
visualizations are based on a representative sample of all spikes in a
cluster, rather than a restricted, potentially biased subset.

=============================
Rationale & Core Features
=============================

By default, to save memory and processing time (especially for compressed data),
Phy restricts spike waveform displays to a subset of time-chunks (`n_chunks_kept=20`).
This means you are only seeing a small percentage (e.g., ~33%) of the actual
spikes in a cluster.

The Problem:
When you split a cluster, BOTH resulting clusters still pull their displayed
waveforms from the *exact same* restricted time-chunks. This often results in
the new clusters appearing to have very few spikes, making it incredibly difficult
to evaluate the quality of the split or see the true distribution of the waveforms.

The Solution:
Since you are using flat binary data (`.dat`/`.bin`), random access to the data is
extremely fast, rendering Phy's default time-chunk restriction unnecessary.
This plugin overrides the default behavior to:
1. Expand the chunk restriction so that it covers the entire recording session.
2. Ensure the Waveform View and Amplitude View render spikes sampled uniformly
   from the *entire* cluster, giving you a true representation of the data.

=============================
Limitations & Performance
=============================

While this plugin makes ALL spikes in a cluster *available* for visualization,
Phy's views (like the Waveform View) will still only *display* a random subset
of these spikes (e.g., a few hundred) if the cluster is very large.

This is a necessary performance trade-off: attempting to draw hundreds of
thousands or millions of waveforms would make the GUI unusably slow.

The key benefit of this plugin is ensuring that the displayed subset is a
**representative, uniform sample** of the *entire* cluster, not a biased
sample from a small time window of the recording. This gives you a much more
accurate picture of the cluster's true waveform distribution.


=============================
Shortcuts & Additional Features
=============================
- **Alt+Shift+W**: Force-refresh the views after a split to redraw all spikes.
- **Alt+Shift+C**: Toggle common median reference (CMR) subtraction *per shank*.
  When enabled, waveforms are extracted across all channels on the same shank,
  the per-timepoint median is subtracted, and only the cluster's best channels
  are returned for display. Shank membership is determined by `channel_shanks.npy`
  or x-coordinates.

=============================
Notes
=============================
- Best used for flat binary recordings. Compressed `.cbin` may be slower due to
  constant decompression.
- The `_phy_spikes_subset.*` files on disk are never modified by this plugin.
"""

from __future__ import annotations

import logging
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from phy import IPlugin, connect

logger = logging.getLogger("phy")


class RawWaveformMode(IPlugin):
    """Make all time-chunks available for spike selection in views."""

    def __init__(self):
        super().__init__()
        self._ready_controller_ids: set[int] = set()

    def attach_to_controller(self, controller):
        controller_id = id(controller)
        if controller_id in self._ready_controller_ids:
            return
        self._ready_controller_ids.add(controller_id)

        model = getattr(controller, "model", None)
        if model is None:
            return

        # ── 1. Disable precomputed waveform subset if present ──
        if getattr(model, "spike_waveforms", None) is not None:
            model.spike_waveforms = None
            logger.info("RawWaveformMode: disabled precomputed waveform subset.")

        # ── 2. Expand chunks_kept to cover the entire recording ──
        selector = getattr(controller, "selector", None)
        if selector is not None:
            last_sample = int(model.spike_samples[-1]) + 1
            old_n_ranges = len(selector.chunks_kept) // 2
            selector.chunks_kept = np.array([0, last_sample])
            logger.info(
                "RawWaveformMode: expanded chunks_kept from %d ranges to "
                "cover full recording (0 – %d samples).",
                old_n_ranges,
                last_sample,
            )

        # ── 3. Clear stale joblib disk cache for waveforms ──
        _purge_waveform_disk_cache(controller)

        # ── 4. Clear stale memcache entries for mean waveforms ──
        _purge_memcache(controller, "_get_mean_waveforms")

        # ── 5. Build per-shank channel map for CMR ──
        #
        # Determine which raw-data channels belong to each shank.
        # Priority: channel_shanks.npy → x-coordinate of channel_positions.
        shank_channels = _build_shank_channel_map(model)

        _state = {"cmr_enabled": False}

        _orig_get_waveforms = model.get_waveforms

        def _get_waveforms_cmr(spike_ids, channel_ids=None):
            """Wrapper that optionally applies per-shank common median ref."""
            if not _state["cmr_enabled"]:
                return _orig_get_waveforms(spike_ids, channel_ids=channel_ids)

            if channel_ids is None:
                channel_ids = np.arange(model.n_channels)

            # Find the union of shank channels covering all requested channels.
            shank_ch_set = set()
            for ch in channel_ids:
                shank_ch_set.update(shank_channels.get(int(ch), [int(ch)]))
            shank_ch_all = np.array(sorted(shank_ch_set))

            # Extract waveforms for the full shank.
            data_shank = _orig_get_waveforms(spike_ids, channel_ids=shank_ch_all)
            if data_shank is None:
                return None

            # data_shank: (n_spikes, n_samples, n_shank_channels)
            # Subtract per-timepoint median across the shank channels.
            cmr = np.median(data_shank, axis=2, keepdims=True)
            data_shank = data_shank - cmr

            # Select only the originally requested channels.
            from phylib.io.array import _index_of

            idx = _index_of(channel_ids, shank_ch_all)
            return data_shank[:, :, idx]

        model.get_waveforms = _get_waveforms_cmr

        # ── 6. Register shortcuts + auto-clear on cluster actions ──
        @connect
        def on_gui_ready(sender, gui):
            if sender is not controller:
                return

            supervisor = getattr(controller, "supervisor", None)
            if supervisor is None:
                return

            # Auto-clear waveform caches after any cluster action.
            @connect(sender=supervisor)
            def on_cluster(sender, up):
                if up.history is not None:
                    _purge_waveform_disk_cache(controller)
                    _purge_memcache(controller, "_get_mean_waveforms")
                    logger.debug(
                        "RawWaveformMode: cleared waveform caches after "
                        "cluster action '%s'.",
                        up.description,
                    )

            @supervisor.actions.add(shortcut="alt+shift+w")
            def refresh_waveforms():
                """Force-refresh waveform and amplitude views for selected cluster(s).

                Clears all waveform/amplitude caches and replots the views.
                """
                selected = list(getattr(supervisor, "selected", []) or [])
                if not selected:
                    msg = "No cluster selected."
                    logger.warning(msg)
                    _status(gui, msg)
                    return

                _purge_waveform_disk_cache(controller)
                _purge_memcache(controller, "_get_mean_waveforms")

                try:
                    from phylib.utils import emit

                    emit("select", supervisor, selected)
                except Exception as exc:
                    logger.debug("Could not re-emit select: %s", exc)

                msg = "Refreshed waveforms/amplitudes for cluster(s) %s" % (
                    ", ".join(str(c) for c in selected),
                )
                logger.info(msg)
                _status(gui, msg)

            @supervisor.actions.add(shortcut="alt+shift+c")
            def toggle_cmr():
                """Toggle per-shank common median reference for waveforms.

                When enabled, for each spike the per-timepoint median across
                all channels on the same shank is subtracted, removing
                common-mode noise while preserving cross-shank independence.
                """
                _state["cmr_enabled"] = not _state["cmr_enabled"]
                state_str = "ON" if _state["cmr_enabled"] else "OFF"
                msg = "Common median reference (per-shank): %s" % state_str
                logger.info(msg)
                _status(gui, msg)

                # Purge caches so waveforms recompute with/without CMR.
                _purge_waveform_disk_cache(controller)
                _purge_memcache(controller, "_get_mean_waveforms")

                # Refresh views.
                selected = list(getattr(supervisor, "selected", []) or [])
                if selected:
                    try:
                        from phylib.utils import emit

                        emit("select", supervisor, selected)
                    except Exception as exc:
                        logger.debug("Could not re-emit select: %s", exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_shank_channel_map(model):
    """Build a dict mapping each channel_id → list of channels on its shank.

    Uses ``model.channel_shanks`` if available and non-trivial (more than one
    shank).  Otherwise falls back to grouping channels by their x-coordinate
    in ``model.channel_positions``.

    Returns a dict ``{channel_id: np.array([ch0, ch1, ...])}``.
    """
    n_channels = model.n_channels
    shanks = getattr(model, "channel_shanks", None)

    # Check if channel_shanks is informative (>1 unique value).
    if shanks is not None and len(np.unique(shanks)) > 1:
        logger.info("RawWaveformMode: using channel_shanks for CMR grouping.")
        groups = defaultdict(list)
        for ch in range(n_channels):
            groups[int(shanks[ch])].append(ch)
    else:
        # Fall back to x-coordinate of channel positions.
        positions = getattr(model, "channel_positions", None)
        if positions is None or len(positions) == 0:
            logger.warning(
                "RawWaveformMode: no position data — CMR will use all channels."
            )
            all_ch = np.arange(n_channels)
            return {ch: all_ch for ch in range(n_channels)}

        logger.info(
            "RawWaveformMode: using x-coordinate of channel_positions "
            "for CMR shank grouping."
        )
        groups = defaultdict(list)
        for ch in range(len(positions)):
            x = float(positions[ch, 0])
            groups[x].append(ch)

    # Convert to {channel_id: sorted_np_array_of_shank_channels}.
    channel_to_shank = {}
    for group_channels in groups.values():
        arr = np.array(sorted(group_channels))
        for ch in group_channels:
            channel_to_shank[ch] = arr

    n_shanks = len(groups)
    sizes = [len(v) for v in groups.values()]
    logger.info(
        "RawWaveformMode: %d shank(s) detected, sizes: %s.",
        n_shanks,
        sizes,
    )
    return channel_to_shank


def _purge_waveform_disk_cache(controller):
    """Remove the joblib disk cache for _get_waveforms_with_n_spikes."""
    context = getattr(controller, "context", None)
    if context is None:
        return
    cache_dir = Path(context.cache_dir)
    wf_cache = (
        cache_dir
        / "phy"
        / "apps"
        / "base"
        / "WaveformMixin"
        / "_get_waveforms_with_n_spikes"
    )
    if wf_cache.is_dir():
        try:
            shutil.rmtree(wf_cache)
            logger.info("RawWaveformMode: purged joblib disk cache at %s", wf_cache)
        except Exception as exc:
            logger.warning("RawWaveformMode: could not purge disk cache: %s", exc)


def _purge_memcache(controller, method_name):
    """Clear in-memory cache for a specific method."""
    context = getattr(controller, "context", None)
    if context is None:
        return
    for key in list(context._memcache.keys()):
        if method_name in key:
            context._memcache[key].clear()
            logger.debug("RawWaveformMode: cleared memcache for '%s'.", key)


def _status(gui, msg: str) -> None:
    """Best-effort status-bar update."""
    fn = getattr(gui, "status_message", None)
    if callable(fn):
        try:
            fn(msg)
        except Exception:
            pass
