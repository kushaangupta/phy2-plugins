"""Phy plugin: Template Match Split (interactive)

What this plugin does
---------------------
This action helps split a target cluster by asking: "Which spikes in the target
look like a reference cluster template?"

Workflow for naive users
------------------------
1) Select two clusters in Phy:
    - First selected cluster = reference (the waveform shape you trust)
    - Second selected cluster = target (the one you want to split)
2) Press Alt+M.
3) The plugin computes cosine similarity of every target spike to the reference
    template and opens a histogram.
4) Move the threshold line (drag or type value).
    - Optional: click "Use Otsu" for an automatic starting threshold.
5) Use the waveform panel (right side) to *see* what you are selecting:
    - "Matched" mean waveform (spikes above threshold)
    - "Unmatched" mean waveform (spikes below threshold)
    - Reference mean waveform for comparison
6) Click "Apply Split" when satisfied.

Why the waveform panel helps
----------------------------
A threshold value alone is abstract. The waveform panel gives immediate visual feedback
about waveform morphology while you tune threshold:
- If matched and unmatched means separate clearly, split is likely meaningful.
- If traces are almost identical, threshold may be arbitrary/noisy.
- Otsu can provide a quick first pass, but waveform separation in the inset
    should still drive the final decision.
- The matched/unmatched traces include ±SEM shaded error bands when possible.

Efficiency and performance design
---------------------------------
- Similarities are computed in chunks to avoid loading all target waveforms at
  once (better memory behavior on large clusters).
- Histogram/counts are based on *all* target spikes.
- Waveform preview uses a deterministic subsample when target is huge,
  so interaction remains fast while still representative.
- Similarity arrays are float32 for reduced memory pressure.

Standalone behavior
-------------------
This plugin is self-contained and does not require `ChannelContextPlugin`.
It only uses standard Phy controller/supervisor APIs plus NumPy/Qt/Matplotlib.

Tunable performance knobs (in code)
-----------------------------------
- Reference template extraction cap: ``max_spikes=1000`` in
    ``_build_reference_template``.
- Similarity compute chunk size: ``chunk_size=2000`` in
    ``_compute_similarities``.
- Waveform preview sample cap: ``max_preview_spikes=4000`` in
    ``_compute_similarities``.

Important implementation detail
-------------------------------
Target spike IDs are sorted before extraction so similarity scores remain
correctly aligned with spike IDs passed to Phy's split action.

Shortcut
--------
Alt+M : Open Template Match Split window.
"""

from phy import IPlugin, connect
import logging
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from qtpy import QtWidgets, QtCore
import seaborn as sns

logger = logging.getLogger("phy")


class TemplateMatchSplit(IPlugin):
    def __init__(self):
        super(TemplateMatchSplit, self).__init__()
        self._shortcuts_created = False
        self.plot_window = None
        self._similarities = None
        self._spike_ids = None

    def attach_to_controller(self, controller):

        # ------------------------------------------------------------------
        # Waveform helpers
        # ------------------------------------------------------------------
        def _get_best_channels(cluster_id):
            """Return the best channel indices for a cluster using phy's native selection.

            Delegates to controller.get_best_channels() which returns the
            channel_ids from the cluster's template — the same channels shown
            in the waveform view and used everywhere else in phy.
            """
            try:
                ch = controller.get_best_channels(cluster_id)
                ch = np.asarray(ch)
                logger.info(
                    "Cluster %s: using %d channels (phy native): %s",
                    cluster_id,
                    len(ch),
                    ch,
                )
                return ch
            except Exception as e:
                logger.error("Could not get channels for cluster %s: %s", cluster_id, e)
                return np.array([0])

        def _extract_waveforms(spike_ids, channel_ids, max_spikes=None, seed=42):
            """Extract raw waveforms from the binary file.

            Returns
            -------
            waveforms : ndarray (n_spikes, n_samples, n_channels) or None
            """
            if len(spike_ids) == 0:
                return None

            ids = spike_ids
            if max_spikes is not None and len(ids) > max_spikes:
                rng = np.random.default_rng(seed)
                ids = rng.choice(ids, size=max_spikes, replace=False)
            ids = np.sort(ids)

            try:
                wf = controller.model.get_waveforms(ids, channel_ids)
                if wf is None or len(wf) == 0:
                    return None
                return np.asarray(wf)  # (n_spikes, n_samples, n_channels)
            except Exception as e:
                logger.error("Failed to extract waveforms: %s", e)
                return None

        def _build_reference_template(cluster_id, channel_ids):
            """Average up to 1000 spikes from *cluster_id* into a single template.

            Returns
            -------
            template_unit : ndarray (n_samples * n_channels,) L2-normalised, or None
            template_mean : ndarray (n_samples, n_channels) un-normalized mean or None
            """
            spike_ids = controller.supervisor.clustering.spikes_per_cluster.get(
                cluster_id, np.array([], dtype=int)
            )
            spike_ids = np.asarray(spike_ids)
            if len(spike_ids) == 0:
                logger.error("Reference cluster %s has no spikes", cluster_id)
                return None, None

            wf = _extract_waveforms(
                spike_ids, channel_ids, max_spikes=1000, seed=cluster_id
            )
            if wf is None:
                return None, None

            template_mean = np.mean(wf, axis=0)
            template = template_mean.flatten()
            norm = np.linalg.norm(template)
            if norm == 0:
                logger.error("Reference template is all zeros")
                return None, None
            return template / norm, template_mean

        def _compute_similarities(
            target_spike_ids,
            channel_ids,
            ref_template,
            ref_template_mean,
            max_preview_spikes=4000,
            chunk_size=2000,
            seed=0,
        ):
            """Compute cosine similarity for all target spikes in memory-safe chunks.

            Also prepares data for the inset waveform preview using a deterministic
            subsample if needed.

            Returns
            -------
            similarities : ndarray (n_spikes,) in [-1, +1]
            preview_sims : ndarray (n_preview,)
            preview_traces : ndarray (n_preview, n_samples)
            ref_trace : ndarray (n_samples,)
            display_channel : int
            """
            n_total = len(target_spike_ids)
            if n_total == 0:
                return None, None, None, None, None

            # Pick one informative channel for preview: max peak-to-peak in reference mean.
            if ref_template_mean is None or ref_template_mean.ndim != 2:
                logger.error("Invalid reference template mean shape for preview")
                return None, None, None, None, None
            ptp = np.ptp(ref_template_mean, axis=0)
            display_channel = int(np.argmax(ptp)) if len(ptp) > 0 else 0
            ref_trace = ref_template_mean[:, display_channel].astype(np.float32)

            sims = np.empty(n_total, dtype=np.float32)

            # Deterministic preview subsample to keep GUI responsive.
            if n_total > max_preview_spikes:
                rng = np.random.default_rng(seed)
                preview_idx = np.sort(
                    rng.choice(n_total, size=max_preview_spikes, replace=False)
                )
            else:
                preview_idx = np.arange(n_total)

            preview_mask = np.zeros(n_total, dtype=bool)
            preview_mask[preview_idx] = True
            preview_sims_chunks = []
            preview_traces_chunks = []

            for start in range(0, n_total, chunk_size):
                end = min(start + chunk_size, n_total)
                chunk_ids = target_spike_ids[start:end]
                wf = _extract_waveforms(chunk_ids, channel_ids, max_spikes=None)
                if wf is None:
                    logger.error("Failed waveform extraction for chunk %d:%d", start, end)
                    return None, None, None, None, None

                # Flatten each spike: (n_chunk, n_samples * n_channels)
                flat = wf.reshape(wf.shape[0], -1).astype(np.float32)

                # Cosine similarity: dot(a, b) / (|a| * |b|)
                norms = np.linalg.norm(flat, axis=1, keepdims=True)
                norms[norms == 0] = 1.0  # avoid division by zero
                flat_normed = flat / norms
                chunk_sims = flat_normed @ ref_template
                chunk_sims = np.nan_to_num(chunk_sims, nan=0.0).astype(np.float32)
                sims[start:end] = chunk_sims

                local_preview_mask = preview_mask[start:end]
                if np.any(local_preview_mask):
                    preview_sims_chunks.append(chunk_sims[local_preview_mask])
                    preview_traces_chunks.append(
                        wf[local_preview_mask, :, display_channel].astype(np.float32)
                    )

            if preview_sims_chunks:
                preview_sims = np.concatenate(preview_sims_chunks)
                preview_traces = np.concatenate(preview_traces_chunks, axis=0)
            else:
                preview_sims = np.array([], dtype=np.float32)
                preview_traces = np.zeros((0, len(ref_trace)), dtype=np.float32)

            return sims, preview_sims, preview_traces, ref_trace, display_channel

        # ------------------------------------------------------------------
        # GUI
        # ------------------------------------------------------------------
        def _show_histogram(
            similarities,
            spike_ids,
            ref_id,
            tgt_id,
            preview_sims,
            preview_traces,
            ref_trace,
            display_channel,
        ):
            """Open interactive histogram with draggable threshold."""

            def _compute_otsu_threshold(values, n_bins=256):
                """Return Otsu threshold for a 1D array, or None if unavailable."""
                arr = np.asarray(values, dtype=np.float64)
                if arr.size < 2:
                    return None

                vmin = float(np.min(arr))
                vmax = float(np.max(arr))
                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                    return None

                hist, bin_edges = np.histogram(arr, bins=n_bins, range=(vmin, vmax))
                hist = hist.astype(np.float64)
                total = hist.sum()
                if total <= 0:
                    return None

                prob = hist / total
                centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5

                omega = np.cumsum(prob)
                mu = np.cumsum(prob * centers)
                mu_t = mu[-1]

                denom = omega * (1.0 - omega)
                sigma_b2 = np.full_like(denom, -np.inf, dtype=np.float64)
                valid = denom > 0
                sigma_b2[valid] = ((mu_t * omega[valid] - mu[valid]) ** 2) / denom[valid]

                idx = int(np.argmax(sigma_b2))
                if not np.isfinite(sigma_b2[idx]):
                    return None
                return float(centers[idx])

            if self.plot_window is not None:
                try:
                    self.plot_window.close()
                except Exception:
                    pass

            # ---- Qt window setup ----
            win = QtWidgets.QMainWindow()
            win.setWindowTitle(f"Template Match Split — ref={ref_id}  target={tgt_id}")
            self.plot_window = win

            central = QtWidgets.QWidget()
            win.setCentralWidget(central)
            layout = QtWidgets.QVBoxLayout(central)

            # ---- Matplotlib figure ----
            fig = Figure(figsize=(15, 7))
            canvas = FigureCanvas(fig)
            gs = fig.add_gridspec(1, 2, width_ratios=[1.8, 1.0], wspace=0.3)
            ax = fig.add_subplot(gs[0, 0])
            ax_wave = fig.add_subplot(gs[0, 1])
            fig.subplots_adjust(left=0.07, right=0.98, bottom=0.12, top=0.92)

            n_bins = min(120, max(30, int(np.sqrt(len(similarities)))))
            sns.histplot(
                similarities,
                ax=ax,
                bins=n_bins,
                stat="density",
                color="#5B9BD5",
                edgecolor="#3A7CC0",
                alpha=0.75,
            )

            # KDE overlay
            try:
                sns.kdeplot(
                    similarities, ax=ax, color="#E85D3A", linewidth=1.8, label="KDE"
                )
            except Exception:
                pass

            ax.set_xlabel("Cosine Similarity to Reference", fontsize=11)
            ax.set_ylabel("Density", fontsize=11)
            ax.set_title(
                f"Similarity of cluster {tgt_id} spikes to cluster {ref_id} template",
                fontsize=12,
                fontweight="bold",
            )
            ax.tick_params(labelsize=9)

            # Initial threshold at the 50th percentile
            init_thresh = float(np.percentile(similarities, 50))
            otsu_thresh = _compute_otsu_threshold(similarities)

            # ---- Draggable vertical line ----
            thresh_line = ax.axvline(
                x=init_thresh,
                color="#2D2D2D",
                linewidth=2,
                linestyle="--",
                zorder=10,
                label="Threshold",
            )

            # Suggested threshold (Otsu)
            if otsu_thresh is not None:
                ax.axvline(
                    x=otsu_thresh,
                    color="#6D4C41",
                    linewidth=1.6,
                    linestyle="-.",
                    alpha=0.9,
                    label=f"Otsu: {otsu_thresh:.3f}",
                )

            # Count annotation (top-right of plot)
            n_total = len(similarities)
            count_text = ax.text(
                0.98,
                0.95,
                "",
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    facecolor="#FFFDE7",
                    edgecolor="#BDBDBD",
                    alpha=0.9,
                ),
            )

            def _update_count_text(threshold):
                n_match = int(np.sum(similarities >= threshold))
                pct = n_match / n_total * 100 if n_total > 0 else 0
                count_text.set_text(
                    f"≥ thresh: {n_match}/{n_total} ({pct:.1f}%)\n"
                    f"< thresh: {n_total - n_match}/{n_total} ({100 - pct:.1f}%)"
                )

            _update_count_text(init_thresh)

            # ---- Side waveform preview panel ----
            def _plot_mean_sem(ax_obj, traces, color, label):
                n = int(traces.shape[0])
                if n == 0:
                    return
                mean = np.mean(traces, axis=0)
                if n > 1:
                    sem = np.std(traces, axis=0, ddof=1) / np.sqrt(n)
                else:
                    sem = np.zeros_like(mean)

                x = np.arange(mean.shape[0])
                ax_obj.plot(x, mean, color=color, linewidth=2.0, label=f"{label} ({n})")
                ax_obj.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.22)

            def _update_wave_panel(threshold):
                ax_wave.clear()

                if (
                    preview_sims is None
                    or preview_traces is None
                    or len(preview_sims) == 0
                    or len(preview_traces) == 0
                ):
                    ax_wave.text(0.5, 0.5, "No preview data", ha="center", va="center")
                    ax_wave.set_title("Waveform panel")
                    return

                matched = preview_sims >= threshold
                unmatched = ~matched

                if np.any(unmatched):
                    _plot_mean_sem(
                        ax_wave,
                        preview_traces[unmatched],
                        color="#1E88E5",
                        label="Unmatched ±SEM",
                    )
                if np.any(matched):
                    _plot_mean_sem(
                        ax_wave,
                        preview_traces[matched],
                        color="#E53935",
                        label="Matched ±SEM",
                    )
                if ref_trace is not None and len(ref_trace) > 0:
                    ax_wave.plot(
                        np.arange(len(ref_trace)),
                        ref_trace,
                        color="#2E7D32",
                        linewidth=1.6,
                        linestyle="--",
                        label="Reference mean",
                    )

                ax_wave.set_title(
                    f"Waveform panel (ch={display_channel}, preview n={len(preview_sims)})",
                    fontsize=10,
                )
                ax_wave.set_xlabel("Sample", fontsize=9)
                ax_wave.set_ylabel("Amplitude", fontsize=9)
                ax_wave.tick_params(labelsize=8)
                ax_wave.legend(fontsize=8, framealpha=0.9, loc="best")

            _update_wave_panel(init_thresh)

            threshold_state = {"value": init_thresh}

            qtimer_cls = getattr(QtCore, "QTimer", None)
            if qtimer_cls is None:
                logger.error("Qt timer API unavailable; cannot open interactive plot.")
                return

            # Throttle expensive waveform redraw during drag.
            pending_threshold = {"value": init_thresh}
            inset_timer = qtimer_cls(win)
            inset_timer.setSingleShot(True)
            inset_timer.setInterval(40)

            def _schedule_wave_panel(threshold):
                pending_threshold["value"] = float(threshold)
                inset_timer.start()

            def _flush_wave_panel():
                _update_wave_panel(pending_threshold["value"])
                canvas.draw_idle()

            inset_timer.timeout.connect(_flush_wave_panel)

            # ---- Drag state ----
            drag_state = {"active": False}

            def on_press(event):
                if event.inaxes != ax or event.button != 1:
                    return
                # Start drag if click is near the threshold line
                xdata = threshold_state["value"]
                xlim = ax.get_xlim()
                tolerance = (xlim[1] - xlim[0]) * 0.02
                if abs(event.xdata - xdata) < tolerance:
                    drag_state["active"] = True

            def on_motion(event):
                if not drag_state["active"] or event.inaxes != ax:
                    return
                thresh_line.set_xdata([event.xdata, event.xdata])
                threshold_state["value"] = float(event.xdata)
                _update_count_text(event.xdata)
                threshold_input.setText(f"{event.xdata:.4f}")
                _schedule_wave_panel(event.xdata)
                canvas.draw_idle()

            def on_release(_event):
                drag_state["active"] = False

            canvas.mpl_connect("button_press_event", on_press)
            canvas.mpl_connect("motion_notify_event", on_motion)
            canvas.mpl_connect("button_release_event", on_release)

            # ---- Legend with preset thresholds ----
            presets = {}
            for pct_label, pct_val in [
                ("10th pct", 10),
                ("25th pct", 25),
                ("50th pct", 50),
                ("75th pct", 75),
                ("90th pct", 90),
            ]:
                v = float(np.percentile(similarities, pct_val))
                presets[pct_label] = v

            colors_preset = ["#E53935", "#FB8C00", "#43A047", "#1E88E5", "#8E24AA"]
            for (name, value), c in zip(presets.items(), colors_preset):
                n_above_p = int(np.sum(similarities >= value))
                ax.axvline(
                    x=value,
                    color=c,
                    linewidth=1,
                    linestyle=":",
                    alpha=0.5,
                    label=f"{name}: {value:.3f} ({n_above_p})",
                )

            ax.legend(
                bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8, framealpha=0.9
            )

            layout.addWidget(canvas)

            # ---- Controls row ----
            form = QtWidgets.QHBoxLayout()
            form.addWidget(QtWidgets.QLabel("Threshold:"))

            threshold_input = QtWidgets.QLineEdit()
            threshold_input.setFixedWidth(120)
            threshold_input.setText(f"{init_thresh:.4f}")
            form.addWidget(threshold_input)

            # Otsu auto-threshold button
            if otsu_thresh is not None:
                otsu_btn = QtWidgets.QPushButton("Use Otsu")
                otsu_btn.setMinimumWidth(90)
                otsu_btn.setToolTip(
                    "Auto-set threshold using Otsu's method on similarity histogram"
                )
                otsu_btn.clicked.connect(
                    lambda _checked=False, v=otsu_thresh: threshold_input.setText(
                        f"{v:.4f}"
                    )
                )
                form.addWidget(otsu_btn)

            # Sync text → line
            def _sync_line_from_text():
                try:
                    val = float(threshold_input.text())
                    thresh_line.set_xdata([val, val])
                    threshold_state["value"] = val
                    _update_count_text(val)
                    _schedule_wave_panel(val)
                    canvas.draw_idle()
                except ValueError:
                    pass

            threshold_input.textChanged.connect(_sync_line_from_text)

            # Preset buttons
            for (name, value), c in zip(presets.items(), colors_preset):
                btn = QtWidgets.QPushButton(name)
                btn.setMinimumWidth(70)
                btn.setStyleSheet(f"QPushButton {{ color: {c}; font-weight: bold; }}")
                btn.clicked.connect(
                    lambda checked, v=value: threshold_input.setText(f"{v:.4f}")
                )
                form.addWidget(btn)

            layout.addLayout(form)

            # ---- Action buttons ----
            btn_row = QtWidgets.QHBoxLayout()

            preview_btn = QtWidgets.QPushButton("Preview")
            preview_btn.setMinimumWidth(110)
            apply_btn = QtWidgets.QPushButton("Apply Split")
            apply_btn.setMinimumWidth(110)
            apply_btn.setStyleSheet(
                "QPushButton { background-color: #43A047; color: white; font-weight: bold; }"
            )

            def _on_preview():
                try:
                    t = float(threshold_input.text())
                except ValueError:
                    logger.error("Invalid threshold value")
                    return
                n_match = int(np.sum(similarities >= t))
                QtWidgets.QMessageBox.information(
                    win,
                    "Preview",
                    f"Threshold: {t:.4f}\n\n"
                    f"Spikes >= threshold (new cluster): {n_match} "
                    f"({n_match / n_total * 100:.1f}%)\n"
                    f"Spikes < threshold (stay):        {n_total - n_match} "
                    f"({(n_total - n_match) / n_total * 100:.1f}%)\n\n"
                    f"Min similarity: {np.min(similarities):.4f}\n"
                    f"Max similarity: {np.max(similarities):.4f}\n"
                    f"Mean similarity: {np.mean(similarities):.4f}",
                )

            def _on_apply():
                try:
                    t = float(threshold_input.text())
                except ValueError:
                    logger.error("Invalid threshold value")
                    return

                n_match = int(np.sum(similarities >= t))
                if n_match == 0 or n_match == n_total:
                    QtWidgets.QMessageBox.warning(
                        win,
                        "Nothing to split",
                        f"Threshold {t:.4f} puts all spikes in one group — nothing to split.",
                    )
                    return

                # Warn if the split is very asymmetric
                minority = min(n_match, n_total - n_match)
                if minority < 5:
                    reply = QtWidgets.QMessageBox.question(
                        win,
                        "Very small split",
                        f"One side of the split has only {minority} spikes. Continue?",
                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                    )
                    if reply == QtWidgets.QMessageBox.No:
                        return

                # Build labels: 1 = stays (< threshold), 2 = new cluster (>= threshold)
                labels = np.ones(n_total, dtype=int)
                labels[similarities >= t] = 2

                logger.info(
                    "Template Match Split: threshold=%.4f  matched=%d  unmatched=%d",
                    t,
                    n_match,
                    n_total - n_match,
                )
                controller.supervisor.actions.split(spike_ids, labels)
                win.close()

            preview_btn.clicked.connect(_on_preview)
            apply_btn.clicked.connect(_on_apply)

            # Enter key in threshold box → apply
            threshold_input.returnPressed.connect(apply_btn.click)

            btn_row.addStretch()
            btn_row.addWidget(preview_btn)
            btn_row.addWidget(apply_btn)
            btn_row.addStretch()
            layout.addLayout(btn_row)

            win.resize(1600, 900)

            # Focus the threshold input after the window is visible
            qtimer_cls.singleShot(
                100, lambda: (threshold_input.setFocus(), threshold_input.selectAll())
            )

            win.show()

        # ------------------------------------------------------------------
        # Action registration
        # ------------------------------------------------------------------
        @connect
        def on_gui_ready(_sender, _gui):
            if self._shortcuts_created:
                return
            self._shortcuts_created = True

            @controller.supervisor.actions.add(shortcut="alt+shift+m")
            def template_match_split():
                """
                Template Match Split (Alt+Shift+M)
                Select reference cluster first, then target cluster.
                """
                try:
                    cluster_ids = controller.supervisor.selected
                    if not cluster_ids or len(cluster_ids) < 2:
                        logger.warning(
                            "Select exactly 2 clusters: reference (1st) and target (2nd)"
                        )
                        return

                    ref_id = cluster_ids[0]
                    tgt_id = cluster_ids[1]
                    logger.info(
                        "Template Match Split: ref=%s  target=%s", ref_id, tgt_id
                    )

                    # Determine channels from reference cluster
                    channel_ids = _get_best_channels(ref_id)
                    logger.info("Using %d channels: %s", len(channel_ids), channel_ids)

                    # Build reference template
                    logger.info(
                        "Building reference template from cluster %s ...", ref_id
                    )
                    ref_template, ref_template_mean = _build_reference_template(
                        ref_id, channel_ids
                    )
                    if ref_template is None:
                        logger.error("Could not build reference template")
                        return

                    # Get target spike IDs
                    bunchs = controller._amplitude_getter(
                        [tgt_id], name="template", load_all=True
                    )
                    target_spike_ids = np.asarray(bunchs[0].spike_ids, dtype=np.int64)
                    # Keep IDs sorted so waveform extraction order and split labels align.
                    target_spike_ids = np.sort(target_spike_ids)
                    if len(target_spike_ids) == 0:
                        logger.error("Target cluster %s has no spikes", tgt_id)
                        return

                    logger.info(
                        "Computing similarities for %d target spikes ...",
                        len(target_spike_ids),
                    )
                    sims, preview_sims, preview_traces, ref_trace, display_channel = (
                        _compute_similarities(
                            target_spike_ids,
                            channel_ids,
                            ref_template,
                            ref_template_mean,
                            max_preview_spikes=4000,
                            chunk_size=2000,
                            seed=tgt_id,
                        )
                    )
                    if sims is None:
                        logger.error("Failed to compute similarities")
                        return
                    if preview_sims is None:
                        preview_sims = np.array([], dtype=np.float32)

                    # Store for external access / debugging
                    self._similarities = sims
                    self._spike_ids = target_spike_ids

                    logger.info(
                        "Similarity stats: min=%.4f  mean=%.4f  max=%.4f",
                        np.min(sims),
                        np.mean(sims),
                        np.max(sims),
                    )
                    logger.info(
                        "Preview trace set: %d spikes on channel %d",
                        len(preview_sims),
                        display_channel,
                    )

                    # Open GUI
                    _show_histogram(
                        sims,
                        target_spike_ids,
                        ref_id,
                        tgt_id,
                        preview_sims,
                        preview_traces,
                        ref_trace,
                        display_channel,
                    )

                except Exception as e:
                    logger.error("Error in template_match_split: %s", e)
                    import traceback

                    logger.error(traceback.format_exc())
