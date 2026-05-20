"""Phy plugin: Dynamic True Waveform Similarity

Overview
--------
This plugin provides a dynamic NumPy-based similarity calculator as an
alternative to Kilosort's static template similarities. The goal is to compute
cosine-style similarity using the actual waveforms extracted from the binary
for the cluster(s) you are inspecting. Because this extracts spikes from disk
on demand, the similarity values reflect the current, possibly edited clusters
you see in the GUI rather than the frozen templates Kilosort produced.

Key behaviors (what this plugin does)
-------------------------------------
- By default the plugin computes a cosine-like similarity between clusters using
    the mean waveform extracted from up to 1000 spikes per cluster (stable
    random sampling when clusters are large).
- The default similarity function applies a channel-overlap penalty: after
    computing similarity across the channels common to both clusters, the score
    is multiplied by ``n_common / n_union`` where ``n_union`` is the total number
    of unique channels involved across both clusters. The intuition: when two
    clusters only overlap on a tiny fraction of their combined channel support,
    a high shared-channel cosine should be downweighted.
- An alternate, unpenalized variant is provided (no channel-overlap penalty).
    You can compute it via the dedicated shortcut (see Shortcuts below) or by
    calling the exposed function on the controller.

Why a channel-overlap penalty?
-------------------------------
Naive users can be surprised by high similarity values when two clusters only
share a single channel even though their overall spatial footprints differ.
Imagine cluster A has 8 best channels and cluster B has 8 best channels but
they only share 1 — the waveform on that single channel might match perfectly
and produce a cosine near 1.0 on the shared subset, but that match is
misleading because the other channels (where A is strong and B is weak) were
ignored. Penalizing by common / union channels reduces such false-high scores
so similarity reflects the overlap of the two clusters' full channel support.

Why common / union channels?
------------------------------------------------------------------
This implementation uses a symmetric Jaccard-style overlap factor:
``n_common / (n_target + n_other - n_common)``. That matches the idea of
penalizing by the fraction of common channels out of all channels involved in
the two clusters combined.

Shortcuts (default)
--------------------
- Alt+Shift+N : Compute NumPy similarity (DEFAULT — penalizes for partial
    channel overlap). This becomes the active similarity used by the similarity
    view while it's set on the controller.
- Ctrl+Alt+Shift+N : Compute NumPy similarity without any channel-overlap
    penalty (unpenalized variant). Use this when you want to compare raw
    cosine-like similarity restricted to the common channels only.
- Alt+Shift+R : Restore original Kilosort similarity and clear the plugin's
    in-memory waveform cache.

Exposed API on the controller
-----------------------------
- ``controller._numpy_similarity``
        The default, penalized similarity function (callable(cluster_id) -> list
        of (other_id, score) sorted descending).
- ``controller._numpy_similarity_unpenalized``
        The unpenalized similarity variant (same signature).
- ``controller._clear_custom_similarity()``
        Clears in-memory waveform templates and the LRU caches for both
        similarity variants.

Implementation notes (for power users / developers)
---------------------------------------------------
- Waveform extraction: up to ``n_spikes_to_extract = 1000`` spikes are
    sampled per cluster; if a cluster is larger we sample deterministically
    using a RandomState seeded by the cluster id so repeated runs are stable.
- Normalization: mean waveform is L2-normalized and flattened so the
    similarity is simply a dot product (fast cosine-like measure).
- Caching: cluster templates are stored in ``cluster_template_cache`` (dict)
    and both similarity functions are wrapped with ``functools.lru_cache``
    (size 1024). If ``ChannelContextPlugin`` is active, its revision token is
    included in the similarity cache key so added/removed context channels are
    reflected without restarting Phy. Clearing functions call ``cache_clear``
    and clear the dict.
- Error-handling: the plugin is defensive and tolerates missing model/supervisor
    attributes; failures to extract waveforms for a cluster quietly skip that
    pair so it won't crash the GUI.

Performance & practical tips
----------------------------
- Extracting waveforms from disk is the slowest part. For large datasets you
    may notice a delay the first time a cluster is processed; subsequent calls
    are much faster due to caching.
- If you need faster but noisier results, reduce ``n_spikes_to_extract``.
- If you prefer a symmetric overlap penalty (Jaccard) change the penalty to
    ``n_common / (n_target + n_other - n_common)`` in the penalized function.

How to change behavior
----------------------
- To make the unpenalized variant the default, set
    ``controller._numpy_similarity = controller._numpy_similarity_unpenalized``
    (done programmatically or manually in a small patch).
- To change shortcuts, edit the two ``@supervisor.actions.add(shortcut=...)``
    decorators near the bottom of this file.

Examples (naive-user friendly)
-----------------------------
- You pick cluster A in the GUI and press Alt+Shift+N: the similarity panel
    now shows other clusters ranked by how similar they are to A, but the
    ranking is reduced for clusters that only overlap A on a small number of
    channels.
- If you want to see the raw, unpenalized cosine-like similarity restricted
    to the shared channels (for debugging or exploratory analysis), press
    Ctrl+Alt+Shift+N instead.

Notes and history
-----------------
This file was extended to provide both a penalized default (safer for most
manual curation workflows) and an explicit unpenalized alternative after
discussion in the development chat. The short explanation and reasoning above
is intended to help new users choose the behavior that fits their needs.
"""

from __future__ import annotations

import logging

from phy import IPlugin, connect


logger = logging.getLogger("phy")


class SpikeEditSimilarityPlugin(IPlugin):
    """Auto-refresh similarity ranking after clustering edits."""

    def __init__(self):
        super().__init__()
        self._ready_controller_ids: set[int] = set()

    def attach_to_controller(self, controller):
        try:
            self._attach_to_controller_impl(controller)
        except Exception as e:
            import traceback

            logger.error(
                "Error in SpikeEditSimilarityPlugin: %s\n%s", e, traceback.format_exc()
            )

    def _attach_to_controller_impl(self, controller):
        controller_id = id(controller)
        if controller_id in self._ready_controller_ids:
            return
        self._ready_controller_ids.add(controller_id)

        recoverable_errors = (AttributeError, RuntimeError, TypeError, ValueError)

        def _setup_numpy_similarity(ctrl):
            """Override Phy's default similarity with a pure NumPy weighted template calculation."""
            import numpy as np
            import functools

            model = getattr(ctrl, "model", None)
            if model is None:
                return False

            try:
                templates_path = model.dir_path / "templates.npy"
                if not templates_path.exists():
                    logger.debug(
                        "No templates.npy found; skipping NumPy similarity override."
                    )
                    return False

                templates = np.load(templates_path, mmap_mode="r")
                n_templates = templates.shape[0]

                cluster_template_cache = {}

                def get_channel_context_revision():
                    try:
                        return int(getattr(ctrl, "_channel_context_revision", 0))
                    except Exception:
                        return 0

                def get_effective_channels(cluster_id):
                    channel_getter = getattr(
                        ctrl, "_channel_context_get_best_channels", None
                    )
                    if not callable(channel_getter):
                        channel_getter = ctrl.get_best_channels
                    return np.asarray(channel_getter(cluster_id), dtype=np.int64)

                def get_cluster_template_on_channels(cluster_id, channel_ids):
                    supervisor = getattr(ctrl, "supervisor", None)
                    if supervisor is None:
                        return None
                    clustering = getattr(supervisor, "clustering", None)
                    if clustering is None:
                        return None

                    cache_key = (cluster_id, tuple(channel_ids))
                    if cache_key in cluster_template_cache:
                        return cluster_template_cache[cache_key]

                    spikes = clustering.spikes_per_cluster.get(cluster_id, [])
                    if len(spikes) == 0:
                        return None

                    # Extract raw waveforms from the binary for true morphological similarity
                    n_spikes_to_extract = 1000
                    if len(spikes) > n_spikes_to_extract:
                        rng = np.random.RandomState(
                            cluster_id
                        )  # Stable random seed per cluster
                        spikes_to_extract = rng.choice(
                            spikes, size=n_spikes_to_extract, replace=False
                        )
                    else:
                        spikes_to_extract = spikes

                    # Sort for sequential disk access
                    spikes_to_extract = np.sort(spikes_to_extract)

                    try:
                        wf = model.get_waveforms(spikes_to_extract, channel_ids)

                        if wf is None or len(wf) == 0:
                            return None

                        # wf shape is typically (n_spikes, n_samples, n_channels)
                        w_template = np.mean(wf, axis=0)
                        w_template = w_template.flatten()

                        # L2 normalize for fast cosine similarity
                        norm = np.linalg.norm(w_template)
                        if norm > 0:
                            w_template = w_template / norm

                        cluster_template_cache[cache_key] = w_template
                        return w_template
                    except Exception as e:
                        logger.debug(
                            "Failed to extract waveforms for cluster %s: %s",
                            cluster_id,
                            e,
                        )
                        return None

                @functools.lru_cache(maxsize=1024)
                def _custom_similarity_penalized_cached(
                    cluster_id, channel_context_revision
                ):
                    supervisor = getattr(ctrl, "supervisor", None)
                    if supervisor is None:
                        return []
                    clustering = getattr(supervisor, "clustering", None)
                    if clustering is None:
                        return []

                    try:
                        target_ch = get_effective_channels(cluster_id)
                    except Exception:
                        return []

                    similarities = []
                    for other_id in clustering.cluster_ids:
                        if other_id == cluster_id:
                            continue

                        try:
                            other_ch = get_effective_channels(other_id)
                        except Exception:
                            continue

                        # Find common channels
                        common_ch = np.intersect1d(target_ch, other_ch)
                        if len(common_ch) == 0:
                            similarities.append((other_id, 0.0))
                            continue

                        target_t = get_cluster_template_on_channels(
                            cluster_id, common_ch
                        )
                        if target_t is None:
                            continue

                        other_t = get_cluster_template_on_channels(other_id, common_ch)
                        if other_t is None:
                            continue

                        sim = float(np.sum(target_t * other_t))

                        # Penalize partial channel overlap using common / union
                        # channels (Jaccard-style overlap).
                        n_target = len(target_ch)
                        n_common = len(common_ch)
                        n_other = len(other_ch)
                        n_union = n_target + n_other - n_common
                        if n_union > 0 and n_common < n_union:
                            sim *= n_common / n_union

                        similarities.append((other_id, sim))

                    similarities.sort(key=lambda x: x[1], reverse=True)
                    return similarities

                def custom_similarity_penalized(cluster_id):
                    return _custom_similarity_penalized_cached(
                        int(cluster_id), get_channel_context_revision()
                    )

                custom_similarity_penalized.cache_clear = (
                    _custom_similarity_penalized_cached.cache_clear
                )

                @functools.lru_cache(maxsize=1024)
                def _custom_similarity_unpenalized_cached(
                    cluster_id, channel_context_revision
                ):
                    supervisor = getattr(ctrl, "supervisor", None)
                    if supervisor is None:
                        return []
                    clustering = getattr(supervisor, "clustering", None)
                    if clustering is None:
                        return []

                    try:
                        target_ch = get_effective_channels(cluster_id)
                    except Exception:
                        return []

                    similarities = []
                    for other_id in clustering.cluster_ids:
                        if other_id == cluster_id:
                            continue

                        try:
                            other_ch = get_effective_channels(other_id)
                        except Exception:
                            continue

                        # Find common channels
                        common_ch = np.intersect1d(target_ch, other_ch)
                        if len(common_ch) == 0:
                            similarities.append((other_id, 0.0))
                            continue

                        target_t = get_cluster_template_on_channels(
                            cluster_id, common_ch
                        )
                        if target_t is None:
                            continue

                        other_t = get_cluster_template_on_channels(other_id, common_ch)
                        if other_t is None:
                            continue

                        sim = float(np.sum(target_t * other_t))

                        # No channel-overlap penalty applied here.
                        similarities.append((other_id, sim))

                    similarities.sort(key=lambda x: x[1], reverse=True)
                    return similarities

                def custom_similarity_unpenalized(cluster_id):
                    return _custom_similarity_unpenalized_cached(
                        int(cluster_id), get_channel_context_revision()
                    )

                custom_similarity_unpenalized.cache_clear = (
                    _custom_similarity_unpenalized_cached.cache_clear
                )

                def clear_numpy_cache():
                    cluster_template_cache.clear()
                    # clear both cached similarity variants if present
                    try:
                        custom_similarity_penalized.cache_clear()
                    except Exception:
                        pass
                    try:
                        custom_similarity_unpenalized.cache_clear()
                    except Exception:
                        pass

                # Default: penalized similarity
                ctrl._numpy_similarity = custom_similarity_penalized
                # Expose an unpenalized variant as well
                ctrl._numpy_similarity_unpenalized = custom_similarity_unpenalized
                ctrl._clear_custom_similarity = clear_numpy_cache
                logger.info("Successfully registered NumPy similarity action.")
                return True

            except Exception as exc:
                logger.warning("Could not setup NumPy similarity: %s", exc)
                return False

        # Register the NumPy similarity function on the controller without overwriting the default.
        _setup_numpy_similarity(controller)

        def _clear_cache(target, attr_names: tuple[str, ...]) -> int:
            cleared = 0
            for name in attr_names:
                obj = getattr(target, name, None)
                if obj is None:
                    continue

                # functools.lru_cache wrappers
                cache_clear = getattr(obj, "cache_clear", None)
                if callable(cache_clear):
                    try:
                        cache_clear()
                        cleared += 1
                    except recoverable_errors as exc:
                        logger.debug(
                            "Failed to clear %s.%s cache: %s", target, name, exc
                        )
                    continue

                # dict-like caches
                clear = getattr(obj, "clear", None)
                if callable(clear):
                    try:
                        clear()
                        cleared += 1
                    except recoverable_errors as exc:
                        logger.debug(
                            "Failed to clear %s.%s object: %s", target, name, exc
                        )
            return cleared

        def _pick_target_cluster(up=None):
            added = list(getattr(up, "added", []) or []) if up is not None else []
            if added:
                return int(added[-1])

            selected_clusters = list(
                getattr(controller.supervisor, "selected_clusters", []) or []
            )
            if selected_clusters:
                return int(selected_clusters[-1])

            selected = list(getattr(controller.supervisor, "selected", []) or [])
            if selected:
                return int(selected[-1])

            return None

        def _refresh_views(supervisor, target_cluster_id=None) -> None:
            similarity_view = getattr(supervisor, "similarity_view", None)
            if similarity_view is not None and target_cluster_id is not None:
                try:
                    similarity_view.reset([target_cluster_id])
                    selected_clusters = list(
                        getattr(supervisor, "selected_clusters", []) or []
                    )
                    set_offset = getattr(
                        similarity_view, "set_selected_index_offset", None
                    )
                    if callable(set_offset):
                        set_offset(len(selected_clusters))
                except recoverable_errors as exc:
                    logger.debug(
                        "Could not reset similarity view for cluster %s: %s",
                        target_cluster_id,
                        exc,
                    )

            # Best-effort UI refresh across Phy variants.
            for view_name in ("similarity_view", "cluster_view"):
                view = getattr(supervisor, view_name, None)
                if view is None:
                    continue
                refresh = getattr(view, "refresh", None)
                if callable(refresh):
                    try:
                        refresh()
                    except recoverable_errors as exc:
                        logger.debug("Could not refresh %s: %s", view_name, exc)

        def _recompute_similarity(gui=None, up=None) -> None:
            cache_count = 0

            # Clear our custom NumPy cache if it exists
            clear_custom = getattr(controller, "_clear_custom_similarity", None)
            if callable(clear_custom):
                try:
                    clear_custom()
                    cache_count += 2
                except recoverable_errors:
                    pass

            # Clear default caches
            cache_count += _clear_cache(
                controller,
                (
                    "similarity",
                    "_get_similar_clusters",
                    "_similarity",
                    "_similarity_cache",
                ),
            )
            model = getattr(controller, "model", None)
            if model is not None:
                cache_count += _clear_cache(
                    model,
                    (
                        "_load_similar_templates",
                        "_load_template_features",
                        "_template_similarity",
                    ),
                )

            target_cluster_id = _pick_target_cluster(up=up)
            _refresh_views(controller.supervisor, target_cluster_id=target_cluster_id)

            if gui is not None:
                status_message = getattr(gui, "status_message", None)
                if callable(status_message):
                    try:
                        status_message("Similarity refreshed")
                    except recoverable_errors:
                        pass

            logger.info(
                "Similarity refresh complete (cleared %d caches, target_cluster=%s).",
                cache_count,
                target_cluster_id,
            )

        @connect
        def on_cluster(sender, up):
            supervisor = getattr(controller, "supervisor", None)
            if supervisor is None or sender is not supervisor:
                return

            # Only react to true clustering edits (not metadata-only changes).
            if not (
                list(getattr(up, "added", []) or [])
                or list(getattr(up, "deleted", []) or [])
            ):
                return

            try:
                _recompute_similarity(up=up)
            except recoverable_errors as exc:
                logger.warning(
                    "Could not recompute similarity after cluster update: %s", exc
                )

        @connect
        def on_gui_ready(sender, gui):
            # Avoid adding action to unrelated GUI controllers.
            if sender is not controller:
                return

            supervisor = getattr(controller, "supervisor", None)
            if supervisor is None:
                return

            # Store the original Kilosort similarity so Alt+Shift+R can restore it
            _original_similarity = getattr(supervisor, "similarity", None)

            @supervisor.actions.add(shortcut="alt+shift+r")
            def recompute_similarity_now():
                """Restore original Kilosort similarity and refresh the view."""

                def msg(text):
                    logger.info("PLUGIN MSG: %s", text)
                    if gui is not None:
                        sm = getattr(gui, "status_message", None)
                        if callable(sm):
                            try:
                                sm(text)
                            except Exception:
                                pass

                try:
                    if _original_similarity is not None:
                        controller.similarity = _original_similarity
                        controller.supervisor.similarity = _original_similarity

                    # Also clear the numpy waveform cache
                    clear_func = getattr(controller, "_clear_custom_similarity", None)
                    if callable(clear_func):
                        clear_func()

                    _recompute_similarity(gui=gui)
                    msg("Restored original Kilosort similarity.")
                except recoverable_errors as exc:
                    logger.warning("Manual similarity refresh failed: %s", exc)

            @supervisor.actions.add(shortcut="alt+shift+n")
            def recompute_numpy_similarity_now():
                """Recompute accurate NumPy similarity for current selection."""

                def msg(text):
                    logger.info("PLUGIN MSG: %s", text)
                    if gui is not None:
                        sm = getattr(gui, "status_message", None)
                        if callable(sm):
                            try:
                                sm(text)
                            except Exception:
                                pass

                try:
                    msg("Computing true waveform similarity...")

                    target_cluster_id = _pick_target_cluster()
                    if target_cluster_id is None:
                        msg("No target cluster selected.")
                        return

                    numpy_sim_func = getattr(controller, "_numpy_similarity", None)
                    if not callable(numpy_sim_func):
                        msg("NumPy similarity function not properly initialized.")
                        return

                    # Swap the similarity function temporarily on both the controller and the supervisor
                    original_sim = getattr(controller, "similarity", None)
                    original_sup_sim = (
                        getattr(controller.supervisor, "similarity", None)
                        if controller.supervisor
                        else None
                    )

                    try:
                        controller.similarity = numpy_sim_func
                    except Exception as e:
                        logger.warning("Could not set controller.similarity: %s", e)

                    if controller.supervisor:
                        try:
                            controller.supervisor.similarity = numpy_sim_func
                        except Exception as e:
                            logger.warning("Could not set supervisor.similarity: %s", e)

                    similarity_view = getattr(
                        controller.supervisor, "similarity_view", None
                    )

                    if similarity_view is not None:
                        similarity_view.reset([target_cluster_id])
                        selected_clusters = list(
                            getattr(controller.supervisor, "selected_clusters", [])
                            or []
                        )
                        set_offset = getattr(
                            similarity_view, "set_selected_index_offset", None
                        )
                        if callable(set_offset):
                            set_offset(len(selected_clusters))

                    msg(
                        f"True waveform similarity computed for cluster {target_cluster_id}"
                    )
                except Exception as exc:
                    import traceback

                    msg(f"Fatal error in NumPy similarity: {exc}")
                    logger.error(
                        "NumPy similarity refresh fatal error:\n%s",
                        traceback.format_exc(),
                    )
            
            @supervisor.actions.add(shortcut="alt+shift+ctrl+n")
            def recompute_numpy_similarity_unpenalized_now():
                """Recompute NumPy similarity without channel-overlap penalty."""

                def msg(text):
                    logger.info("PLUGIN MSG: %s", text)
                    if gui is not None:
                        sm = getattr(gui, "status_message", None)
                        if callable(sm):
                            try:
                                sm(text)
                            except Exception:
                                pass

                try:
                    msg("Computing true waveform similarity (no channel penalty)...")

                    target_cluster_id = _pick_target_cluster()
                    if target_cluster_id is None:
                        msg("No target cluster selected.")
                        return

                    numpy_sim_func = getattr(controller, "_numpy_similarity_unpenalized", None)
                    if not callable(numpy_sim_func):
                        msg("Unpenalized NumPy similarity function not properly initialized.")
                        return

                    # Swap the similarity function temporarily on both the controller and the supervisor
                    try:
                        controller.similarity = numpy_sim_func
                    except Exception as e:
                        logger.warning("Could not set controller.similarity: %s", e)

                    if controller.supervisor:
                        try:
                            controller.supervisor.similarity = numpy_sim_func
                        except Exception as e:
                            logger.warning("Could not set supervisor.similarity: %s", e)

                    similarity_view = getattr(
                        controller.supervisor, "similarity_view", None
                    )

                    if similarity_view is not None:
                        similarity_view.reset([target_cluster_id])
                        selected_clusters = list(
                            getattr(controller.supervisor, "selected_clusters", [])
                            or []
                        )
                        set_offset = getattr(
                            similarity_view, "set_selected_index_offset", None
                        )
                        if callable(set_offset):
                            set_offset(len(selected_clusters))

                    msg(
                        f"True waveform similarity (no penalty) computed for cluster {target_cluster_id}"
                    )
                except Exception as exc:
                    import traceback

                    msg(f"Fatal error in NumPy similarity (no penalty): {exc}")
                    logger.error(
                        "NumPy similarity (no penalty) refresh fatal error:\n%s",
                        traceback.format_exc(),
                    )
