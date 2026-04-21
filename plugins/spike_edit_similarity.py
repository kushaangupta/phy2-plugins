"""Phy plugin: Dynamic True Waveform Similarity

This plugin provides an alternative to Kilosort's default similarity scores. 
Kilosort's default similarities are static: they are based strictly on the original 
Kilosort templates and do not update when you manually split a cluster. (For example, 
splitting a cluster will result in the two halves having a 1.0 similarity forever 
because they share the same Kilosort template).

This plugin allows you to toggle on a custom, dynamic NumPy similarity calculator 
that extracts raw spikes from the binary on-the-fly, calculates the true mean 
waveform of your newly split clusters, and computes accurate cosine similarities.
This allows you to quantitatively validate whether your manual splits actually 
separated distinct waveforms.

Note: Because this dynamically extracts spikes from the large binary file, it is an 
in-memory calculation that resets when Phy is closed. It does not overwrite the 
static `similar_templates.npy` file on disk.

Shortcuts
---------
Alt+Shift+N : Activate True Waveform Similarity (NumPy)
    Switches the similarity view to use the custom dynamic math. It extracts up 
    to 1000 raw waveforms for the currently selected cluster, calculates its true 
    mean shape, and computes cosine similarity against all other clusters. 
    Once activated, this custom math will persist and update as you click around 
    different clusters.

Alt+Shift+R : Restore Default Similarity (Kilosort)
    Restores the original static Kilosort similarity math and clears the custom 
    waveform cache. Use this to revert the GUI back to default behavior.
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
            logger.error("Error in SpikeEditSimilarityPlugin: %s\n%s", e, traceback.format_exc())

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
                    logger.debug("No templates.npy found; skipping NumPy similarity override.")
                    return False

                templates = np.load(templates_path, mmap_mode="r")
                n_templates = templates.shape[0]

                cluster_template_cache = {}

                def get_cluster_template(cluster_id):
                    supervisor = getattr(ctrl, "supervisor", None)
                    if supervisor is None:
                        return None
                    clustering = getattr(supervisor, "clustering", None)
                    if clustering is None:
                        return None

                    if cluster_id in cluster_template_cache:
                        return cluster_template_cache[cluster_id]

                    spikes = clustering.spikes_per_cluster.get(cluster_id, [])
                    if len(spikes) == 0:
                        return None

                    # Extract raw waveforms from the binary for true morphological similarity
                    n_spikes_to_extract = 1000
                    if len(spikes) > n_spikes_to_extract:
                        rng = np.random.RandomState(cluster_id)  # Stable random seed per cluster
                        spikes_to_extract = rng.choice(spikes, size=n_spikes_to_extract, replace=False)
                    else:
                        spikes_to_extract = spikes
                        
                    # Sort for sequential disk access
                    spikes_to_extract = np.sort(spikes_to_extract)

                    try:
                        # Try to get waveforms for all channels
                        channel_ids = np.arange(templates.shape[2])
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

                        cluster_template_cache[cluster_id] = w_template
                        return w_template
                    except Exception as e:
                        logger.debug("Failed to extract waveforms for cluster %s: %s", cluster_id, e)
                        return None

                @functools.lru_cache(maxsize=1024)
                def custom_similarity(cluster_id):
                    supervisor = getattr(ctrl, "supervisor", None)
                    if supervisor is None:
                        return []
                    clustering = getattr(supervisor, "clustering", None)
                    if clustering is None:
                        return []

                    target_t = get_cluster_template(cluster_id)
                    if target_t is None:
                        return []

                    similarities = []
                    for other_id in clustering.cluster_ids:
                        if other_id == cluster_id:
                            continue
                        other_t = get_cluster_template(other_id)
                        if other_t is None:
                            continue

                        sim = float(np.sum(target_t * other_t))
                        similarities.append((other_id, sim))

                    similarities.sort(key=lambda x: x[1], reverse=True)
                    return similarities

                def clear_numpy_cache():
                    cluster_template_cache.clear()
                    custom_similarity.cache_clear()

                ctrl._numpy_similarity = custom_similarity
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
                        logger.debug("Failed to clear %s.%s cache: %s", target, name, exc)
                    continue

                # dict-like caches
                clear = getattr(obj, "clear", None)
                if callable(clear):
                    try:
                        clear()
                        cleared += 1
                    except recoverable_errors as exc:
                        logger.debug("Failed to clear %s.%s object: %s", target, name, exc)
            return cleared

        def _pick_target_cluster(up=None):
            added = list(getattr(up, "added", []) or []) if up is not None else []
            if added:
                return int(added[-1])

            selected_clusters = list(getattr(controller.supervisor, "selected_clusters", []) or [])
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
                    selected_clusters = list(getattr(supervisor, "selected_clusters", []) or [])
                    set_offset = getattr(similarity_view, "set_selected_index_offset", None)
                    if callable(set_offset):
                        set_offset(len(selected_clusters))
                except recoverable_errors as exc:
                    logger.debug("Could not reset similarity view for cluster %s: %s", target_cluster_id, exc)

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
            if not (list(getattr(up, "added", []) or []) or list(getattr(up, "deleted", []) or [])):
                return

            try:
                _recompute_similarity(up=up)
            except recoverable_errors as exc:
                logger.warning("Could not recompute similarity after cluster update: %s", exc)

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
                    original_sup_sim = getattr(controller.supervisor, "similarity", None) if controller.supervisor else None
                    
                    try:
                        controller.similarity = numpy_sim_func
                    except Exception as e:
                        logger.warning("Could not set controller.similarity: %s", e)
                        
                    if controller.supervisor:
                        try:
                            controller.supervisor.similarity = numpy_sim_func
                        except Exception as e:
                            logger.warning("Could not set supervisor.similarity: %s", e)

                    similarity_view = getattr(controller.supervisor, "similarity_view", None)
                    
                    if similarity_view is not None:
                        similarity_view.reset([target_cluster_id])
                        selected_clusters = list(getattr(controller.supervisor, "selected_clusters", []) or [])
                        set_offset = getattr(similarity_view, "set_selected_index_offset", None)
                        if callable(set_offset):
                            set_offset(len(selected_clusters))
                    
                    msg(f"True waveform similarity computed for cluster {target_cluster_id}")
                except Exception as exc:
                    import traceback
                    msg(f"Fatal error in NumPy similarity: {exc}")
                    logger.error("NumPy similarity refresh fatal error:\n%s", traceback.format_exc())
