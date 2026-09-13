"""Debounced automatic saving for Phy 2 curation sessions.

Phy intentionally keeps merges, splits, group assignments, custom labels, and
undo/redo results in memory until the user saves.  This plugin limits the work
lost to a crash by saving shortly after the last clustering change.

Unlike the original 2021 proof of concept, this implementation:

* covers every dirty clustering update, not only ``group`` metadata;
* debounces bursts of changes into one write (important for batch plugins such
  as ``AutoQuality``);
* uses a Qt timer on the GUI thread instead of an unsafe background thread;
* calls the Supervisor's supported save path, which persists
  ``spike_clusters.npy``, cluster metadata TSVs, and ``cluster_info.tsv``;
* leaves Phy's ordinary Ctrl+S action and close-with-unsaved-changes prompt
  intact.

Enable ``AutoSavePlugin`` in ``phy_config.py``.  The default quiet period is
five seconds.  Autosave overwrites Phy's normal curation output files; it is not
a versioned backup and does not replace a pre-curation copy of the dataset.

Original concept: Luke Shaheen, Laboratory of Brain, Hearing and Behavior
(September 2021).  Modernized for this plugin collection.
"""

from __future__ import annotations

import logging

from phy import IPlugin, connect
from phy.gui.qt import QTimer

logger = logging.getLogger("phy")


class _DebouncedAutoSaver:
    """Coalesce clustering events and save from the Qt GUI thread."""

    def __init__(self, supervisor, parent, delay_ms=5_000):
        if delay_ms < 0:
            raise ValueError("delay_ms must be non-negative")
        self.supervisor = supervisor
        self.delay_ms = int(delay_ms)
        self.pending_descriptions = set()
        self.timer = QTimer(parent)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.save_if_dirty)

    def schedule(self, update=None):
        """Restart the quiet-period timer after a real unsaved update."""

        if not self.supervisor.is_dirty():
            return
        description = getattr(update, "description", None)
        self.pending_descriptions.add(description or "clustering update")
        # Starting an active single-shot QTimer restarts its countdown.  A long
        # batch operation therefore produces one save after control returns to
        # the event loop, rather than one disk write per cluster.
        self.timer.start(self.delay_ms)

    def save_if_dirty(self):
        """Persist the current curation state if changes are still pending."""

        if not self.supervisor.is_dirty():
            self.pending_descriptions.clear()
            return

        descriptions = ", ".join(sorted(self.pending_descriptions))
        try:
            self.supervisor.save()
        except Exception:
            # Supervisor remains dirty, so Ctrl+S and Phy's close prompt still
            # provide recovery paths.  Do not retry indefinitely on a bad path.
            logger.exception("AutoSavePlugin: autosave failed")
            return

        self.pending_descriptions.clear()
        logger.info("Autosaved curation changes (%s).", descriptions)


class AutoSavePlugin(IPlugin):
    """Save all curation changes after a five-second quiet period."""

    delay_ms = 5_000

    def attach_to_controller(self, controller):
        @connect(sender=controller)
        def on_gui_ready(sender, gui):
            autosaver = _DebouncedAutoSaver(
                controller.supervisor,
                parent=gui,
                delay_ms=self.delay_ms,
            )
            # Retain the Python wrapper for the lifetime of the GUI.  The QTimer
            # itself is parented to the GUI and is destroyed with it.
            gui._phy2_autosaver = autosaver

            @connect(sender=controller.supervisor)
            def on_cluster(sender, up):
                autosaver.schedule(up)

            logger.info(
                "AutoSavePlugin enabled: saving %.1f seconds after the last change.",
                self.delay_ms / 1_000.0,
            )
