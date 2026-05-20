"""Swap the ClusterView and SimilarityView selections.

Shortcut
--------
Alt+Shift+S : Swap cluster ↔ similar selection.

Behaviour
---------
When you have cluster A selected in the ClusterView and cluster B selected in
the SimilarityView, pressing the shortcut will:

1. Select cluster B in the ClusterView (this refreshes the SimilarityView).
2. Select cluster A in the SimilarityView (if it appears in the new similarity
   ranking for cluster B).

This is useful when you realise the "similar" cluster is actually the one you
want to keep curating, and you want to compare it with the original cluster.
"""

from __future__ import annotations

import logging

from phy import IPlugin, connect

logger = logging.getLogger("phy")


class SwapClusterSimilarPlugin(IPlugin):
    """Swap the ClusterView and SimilarityView selections."""

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

            @supervisor.actions.add(shortcut="alt+shift+s")
            def swap_cluster_similar():
                """Swap ClusterView ↔ SimilarityView selections.

                The cluster currently selected in the SimilarityView becomes
                the primary selection in the ClusterView, and the former
                ClusterView selection is looked up in the new SimilarityView.
                """
                best = list(supervisor.selected_clusters or [])
                similar = list(supervisor.selected_similar or [])

                if not best or not similar:
                    msg = "Need a cluster selected in both views to swap."
                    logger.warning(msg)
                    _status(gui, msg)
                    return

                new_best = similar[0]
                old_best = best[0]

                logger.info(
                    "Swapping: ClusterView %d → %d, will look for %d in SimilarityView.",
                    old_best,
                    new_best,
                    old_best,
                )

                # Step 1: Select the similar cluster in ClusterView.
                # This triggers a similarity-view refresh for new_best.
                # We use a callback to chain step 2 after the view updates.
                def _after_cluster_select(cluster_ids=None):
                    # Step 2: Try to select old_best in the SimilarityView.
                    sim_view = getattr(supervisor, "similarity_view", None)
                    if sim_view is None:
                        return

                    # The similarity_view.select() method works like
                    # cluster_view.select() — it takes a list of ids.
                    try:
                        sim_view.select([old_best])
                    except Exception as exc:
                        logger.debug(
                            "Could not select cluster %d in SimilarityView: %s",
                            old_best,
                            exc,
                        )

                    msg = f"Swapped: ClusterView={new_best}, SimilarityView={old_best}"
                    logger.info(msg)
                    _status(gui, msg)

                supervisor.select([new_best], callback=_after_cluster_select)


def _status(gui, msg: str) -> None:
    """Best-effort status-bar update."""
    fn = getattr(gui, "status_message", None)
    if callable(fn):
        try:
            fn(msg)
        except Exception:
            pass
