"""Phy plugin: interactive Mahalanobis outlier split.

This plugin computes Mahalanobis distance in flattened PC-feature space for the
selected cluster and opens an interactive threshold window. The histogram uses
all selected spikes, while the waveform panel uses a deterministic capped raw
waveform preview so dragging the threshold stays responsive on large clusters.

Workflow
--------
1) Select a cluster and press Alt+X.
2) Drag the threshold line, type a value, or choose one of the preset buttons.
3) Watch the waveform panel update: kept spikes are drawn separately from
   outlier spikes, with mean +/- SEM on the display channel.
4) Click Apply Threshold to split spikes with distance greater than the
   threshold into a new cluster.
"""

from phy import IPlugin, connect
import logging
import numpy as np
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler
import warnings
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from qtpy import QtWidgets, QtCore
import seaborn as sns

logger = logging.getLogger('phy')


class StableMahalanobisDetection(IPlugin):
    def __init__(self):
        super(StableMahalanobisDetection, self).__init__()
        self._shortcuts_created = False
        self.current_distances = None
        self.current_threshold = None
        self.plot_window = None
        self._spike_ids = None
        self._n_features = None
        self._feature_structure = None

    def attach_to_controller(self, controller):
        def get_feature_dimensions(features_arr):
            """Analyze the feature array structure to get actual dimensions"""
            try:
                # Get the feature dimensions from the model
                feature_shape = controller.model._load_features().data.shape
                if len(feature_shape) == 3:  # (n_spikes, n_channels, n_pcs)
                    n_channels = feature_shape[1]
                    n_pcs = feature_shape[2]
                    logger.info(f"Feature structure: {n_channels} channels with {n_pcs} PCs each")
                    return n_channels * n_pcs
                else:
                    logger.warn(f"Unexpected feature shape: {feature_shape}")
                    return features_arr.shape[1]
            except Exception as e:
                logger.error(f"Error getting feature dimensions: {str(e)}")
                return features_arr.shape[1]

        def prepare_features(spike_ids):
            """Prepare feature matrix from spike data with proper dimensionality"""
            try:
                # Load features with original structure
                features = controller.model._load_features().data[spike_ids]

                # Log feature shape information
                logger.info(f"Original feature shape: {features.shape}")

                # Reshape maintaining actual structure
                features_flat = np.reshape(features, (features.shape[0], -1))

                # Get actual feature dimensions
                self._n_features = get_feature_dimensions(features)
                logger.info(f"Using {self._n_features} feature dimensions for Mahalanobis distance")

                return features_flat

            except Exception as e:
                logger.error(f"Error preparing features: {str(e)}")
                return None

        def stable_mahalanobis(X):
            """Compute Mahalanobis distances with numerical stability safeguards"""
            if X is None or len(X) == 0:
                logger.error("Empty or invalid feature matrix")
                return None

            try:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                cov = np.cov(X_scaled, rowvar=False)
                n_features = cov.shape[0]
                cov += np.eye(n_features) * 1e-6

                try:
                    U, s, Vt = np.linalg.svd(cov)
                    s[s < 1e-8] = 1e-8
                    inv_cov = (U / s) @ Vt
                    mu = np.mean(X_scaled, axis=0)
                    diff = X_scaled - mu
                    distances = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))
                    return np.nan_to_num(distances, nan=np.inf)

                except np.linalg.LinAlgError as e:
                    logger.error(f"SVD failed: {str(e)}, falling back to diagonal covariance")
                    inv_cov = np.diag(1.0 / np.diag(cov))
                    mu = np.mean(X_scaled, axis=0)
                    diff = X_scaled - mu
                    return np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))

            except Exception as e:
                logger.error(f"Error in Mahalanobis distance calculation: {str(e)}")
                return None

        def calculate_robust_threshold(distances):
            """Calculate default threshold based on chi-square distribution"""
            if distances is None or len(distances) == 0 or self._n_features is None:
                return None
            # Use 99.99% chi-square threshold as default (very conservative)
            return np.sqrt(chi2.ppf(0.9999, self._n_features))

        def suggest_thresholds(distances):
            """Suggest thresholds with focus on empirical distribution"""
            if distances is None or len(distances) == 0:
                return {}

            try:
                distances = np.asarray(distances)
                distances = distances[np.isfinite(distances)]
                if len(distances) == 0:
                    return {}

                # Calculate empirical thresholds
                empirical_thresholds = {
                    'pct_99': np.percentile(distances, 99),
                    'pct_999': np.percentile(distances, 99.9),
                    'pct_9999': np.percentile(distances, 99.99)  # More conservative
                }

                # Add chi-square thresholds if dimensionality is available
                if self._n_features is not None:
                    p = self._n_features
                    chi2_thresh_999 = np.sqrt(chi2.ppf(0.999, p))  # More conservative (0.1% false positive rate)
                    chi2_thresh_9999 = np.sqrt(chi2.ppf(0.9999, p))  # Very conservative (0.01% false positive rate)
                    empirical_thresholds['Chi2_999'] = chi2_thresh_999
                    empirical_thresholds['Chi2_9999'] = chi2_thresh_9999

                return empirical_thresholds

            except Exception as e:
                logger.error(f"Error calculating threshold suggestions: {str(e)}")
                return {}

        def prepare_waveform_preview(
            spike_ids, distances, cluster_id, max_preview_spikes=4000
        ):
            """Extract a capped raw-waveform preview aligned to distance values."""
            if spike_ids is None or distances is None or len(spike_ids) == 0:
                return None, None, None

            spike_ids = np.asarray(spike_ids, dtype=np.int64)
            distances = np.asarray(distances, dtype=np.float32)
            if len(spike_ids) != len(distances):
                logger.warning(
                    "Mahalanobis preview skipped: %d spike ids but %d distances.",
                    len(spike_ids),
                    len(distances),
                )
                return None, None, None

            try:
                channel_ids = np.asarray(
                    controller.get_best_channels(cluster_id), dtype=np.int64
                )
                if len(channel_ids) == 0:
                    return None, None, None

                if len(spike_ids) > max_preview_spikes:
                    rng = np.random.default_rng(int(cluster_id))
                    preview_idx = np.sort(
                        rng.choice(
                            len(spike_ids), size=max_preview_spikes, replace=False
                        )
                    )
                else:
                    preview_idx = np.arange(len(spike_ids))

                preview_spike_ids = spike_ids[preview_idx]
                preview_distances = distances[preview_idx]

                order = np.argsort(preview_spike_ids)
                preview_spike_ids = preview_spike_ids[order]
                preview_distances = preview_distances[order]

                waveforms = controller.model.get_waveforms(
                    preview_spike_ids, channel_ids
                )
                if waveforms is None or len(waveforms) == 0:
                    return None, None, None

                waveforms = np.asarray(waveforms, dtype=np.float32)
                mean_waveform = np.mean(waveforms, axis=0)
                display_channel_index = int(np.argmax(np.ptp(mean_waveform, axis=0)))
                display_channel = int(channel_ids[display_channel_index])
                preview_traces = waveforms[:, :, display_channel_index]

                logger.info(
                    "Mahalanobis preview: %d/%d spikes on channel %d.",
                    len(preview_distances),
                    len(spike_ids),
                    display_channel,
                )
                return preview_distances, preview_traces, display_channel

            except Exception as e:
                logger.warning(f"Could not prepare Mahalanobis waveform preview: {e}")
                return None, None, None

        def plot_distribution(
            distances,
            threshold=None,
            preview_distances=None,
            preview_traces=None,
            display_channel=None,
        ):
            """Create an interactive threshold plot with a live waveform preview."""
            if distances is None or len(distances) == 0:
                logger.error("No valid distances to plot")
                return

            if self.plot_window is not None:
                try:
                    self.plot_window.close()
                except Exception:
                    pass

            self.plot_window = QtWidgets.QMainWindow()
            self.plot_window.setWindowTitle('Mahalanobis Distance Threshold Split')
            widget = QtWidgets.QWidget()
            self.plot_window.setCentralWidget(widget)
            layout = QtWidgets.QVBoxLayout(widget)

            fig = Figure(figsize=(17, 6))
            canvas = FigureCanvas(fig)
            gs = fig.add_gridspec(1, 2, width_ratios=[1.8, 1.0], wspace=0.3)
            ax = fig.add_subplot(gs[0, 0])
            ax_wave = fig.add_subplot(gs[0, 1])
            fig.subplots_adjust(left=0.07, right=0.98, bottom=0.12, top=0.92)

            distances = np.asarray(distances, dtype=np.float32)
            finite_distances = distances[np.isfinite(distances)]
            if len(finite_distances) == 0:
                logger.error("No finite distances to plot")
                return

            max_dist = float(np.max(finite_distances))
            q99_9 = float(np.percentile(finite_distances, 99.9))
            init_threshold = float(threshold) if threshold is not None else q99_9
            plot_max = min(max_dist, q99_9 * 1.2)
            plot_max = max(plot_max, init_threshold * 1.05, 1.0)

            n_bins = min(120, max(30, int(np.sqrt(len(finite_distances)))))
            sns.histplot(
                finite_distances,
                ax=ax,
                bins=n_bins,
                stat='density',
                color="#5B9BD5",
                edgecolor="#3A7CC0",
                alpha=0.75,
            )
            try:
                sns.kdeplot(
                    finite_distances, ax=ax, color="#E85D3A", linewidth=1.8, label="KDE"
                )
            except Exception:
                pass
            ax.set_xlim(0, plot_max)

            if self._n_features is not None:
                x = np.linspace(0, plot_max, 200)
                chi_density = 2 * x * chi2.pdf(x ** 2, self._n_features)
                ax.plot(x, chi_density, 'r--', alpha=0.3,
                        label=rf'$\chi^2$ ({self._n_features} df)')

            # Labels and formatting
            ax.set_xlabel('Mahalanobis Distance', fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.set_title(
                "Mahalanobis distance threshold",
                fontsize=12,
                fontweight="bold",
            )
            ax.tick_params(labelsize=9)

            suggestions = suggest_thresholds(distances)
            colors = ['#E53935', '#FB8C00', '#43A047', '#1E88E5', '#8E24AA']

            def threshold_label(name, legend=False):
                if name == 'Chi2_999':
                    return (r'$\chi^2$ 99.9%' if legend else 'Chi2 99.9%')
                if name == 'Chi2_9999':
                    return (r'$\chi^2$ 99.99%' if legend else 'Chi2 99.99%')
                return name

            for (name, value), color in zip(suggestions.items(), colors):
                if value <= plot_max:
                    n_spikes = np.sum(distances > value)
                    ax.axvline(x=value, color=color, linestyle='--',
                               label=f'{threshold_label(name, legend=True)}: {value:.1f}\n({n_spikes} spikes, {n_spikes / len(distances) * 100:.1f}%)')

            threshold_state = {"value": init_threshold}
            thresh_line = ax.axvline(
                x=init_threshold,
                color="#2D2D2D",
                linewidth=2,
                linestyle='--',
                zorder=10,
                label='Current threshold',
            )

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

            def update_count_text(value):
                n_outliers = int(np.sum(distances > value))
                pct = n_outliers / len(distances) * 100 if len(distances) else 0
                count_text.set_text(
                    f"> thresh: {n_outliers}/{len(distances)} ({pct:.1f}%)\n"
                    f"<= thresh: {len(distances) - n_outliers}/{len(distances)} "
                    f"({100 - pct:.1f}%)"
                )

            def plot_mean_sem(ax_obj, traces, color, label):
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

            def update_wave_panel(value):
                ax_wave.clear()
                if (
                    preview_distances is None
                    or preview_traces is None
                    or len(preview_distances) == 0
                    or len(preview_traces) == 0
                ):
                    ax_wave.text(
                        0.5,
                        0.5,
                        "No waveform preview",
                        ha="center",
                        va="center",
                    )
                    ax_wave.set_title("Waveform preview", fontsize=10)
                    return

                outliers = preview_distances > value
                kept = ~outliers

                if np.any(kept):
                    plot_mean_sem(
                        ax_wave,
                        preview_traces[kept],
                        color="#1E88E5",
                        label="Kept +/- SEM",
                    )
                if np.any(outliers):
                    plot_mean_sem(
                        ax_wave,
                        preview_traces[outliers],
                        color="#E53935",
                        label="Outliers +/- SEM",
                    )

                channel_label = (
                    f"ch={display_channel}" if display_channel is not None else "channel"
                )
                ax_wave.set_title(
                    f"Waveform preview ({channel_label}, n={len(preview_distances)})",
                    fontsize=10,
                )
                ax_wave.set_xlabel("Sample", fontsize=9)
                ax_wave.set_ylabel("Amplitude", fontsize=9)
                ax_wave.tick_params(labelsize=8)
                ax_wave.legend(fontsize=8, framealpha=0.9, loc="best")

            update_count_text(init_threshold)
            update_wave_panel(init_threshold)

            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)

            qtimer_cls = getattr(QtCore, "QTimer", None)
            if qtimer_cls is None:
                logger.error("Qt timer API unavailable; cannot open interactive plot.")
                return

            pending_threshold = {"value": init_threshold}
            preview_timer = qtimer_cls(self.plot_window)
            preview_timer.setSingleShot(True)
            preview_timer.setInterval(40)

            def schedule_wave_panel(value):
                pending_threshold["value"] = float(value)
                preview_timer.start()

            def flush_wave_panel():
                update_wave_panel(pending_threshold["value"])
                canvas.draw_idle()

            preview_timer.timeout.connect(flush_wave_panel)

            form_layout = QtWidgets.QHBoxLayout()
            form_layout.addWidget(QtWidgets.QLabel("Threshold:"))
            threshold_input = QtWidgets.QLineEdit()
            threshold_input.setFixedWidth(120)
            threshold_input.setPlaceholderText('Enter threshold value')
            threshold_input.setText(f"{init_threshold:.2f}")
            form_layout.addWidget(threshold_input)

            def on_return_pressed():
                apply_button.click()

            threshold_input.returnPressed.connect(on_return_pressed)

            def sync_line_from_text():
                try:
                    value = float(threshold_input.text())
                    if not np.isfinite(value):
                        return
                except ValueError:
                    return

                threshold_state["value"] = value
                thresh_line.set_xdata([value, value])
                if value > ax.get_xlim()[1]:
                    ax.set_xlim(0, value * 1.05)
                update_count_text(value)
                schedule_wave_panel(value)
                canvas.draw_idle()

            threshold_input.textChanged.connect(sync_line_from_text)

            for (name, value), color in zip(suggestions.items(), colors):
                preset_button = QtWidgets.QPushButton(threshold_label(name))
                preset_button.setMinimumWidth(100)
                preset_button.setStyleSheet(
                    f"QPushButton {{ color: {color}; font-weight: bold; }}"
                )
                preset_button.clicked.connect(
                    lambda checked, v=value: threshold_input.setText(f"{v:.2f}")
                )
                form_layout.addWidget(preset_button)

            drag_state = {"active": False}

            def on_press(event):
                if event.inaxes != ax or event.button != 1 or event.xdata is None:
                    return
                value = max(float(event.xdata), 0.0)
                drag_state["active"] = True
                threshold_state["value"] = value
                thresh_line.set_xdata([value, value])
                update_count_text(value)
                threshold_input.blockSignals(True)
                threshold_input.setText(f"{value:.2f}")
                threshold_input.blockSignals(False)
                schedule_wave_panel(value)
                canvas.draw_idle()

            def on_motion(event):
                if (
                    not drag_state["active"]
                    or event.inaxes != ax
                    or event.xdata is None
                ):
                    return
                value = max(float(event.xdata), 0.0)
                threshold_state["value"] = value
                thresh_line.set_xdata([value, value])
                update_count_text(value)
                threshold_input.blockSignals(True)
                threshold_input.setText(f"{value:.2f}")
                threshold_input.blockSignals(False)
                schedule_wave_panel(value)
                canvas.draw_idle()

            def on_release(_event):
                drag_state["active"] = False

            canvas.mpl_connect("button_press_event", on_press)
            canvas.mpl_connect("motion_notify_event", on_motion)
            canvas.mpl_connect("button_release_event", on_release)

            preview_button = QtWidgets.QPushButton('Preview Selection')
            apply_button = QtWidgets.QPushButton('Apply Threshold')
            preview_button.setMinimumWidth(120)
            apply_button.setMinimumWidth(120)
            apply_button.setStyleSheet(
                "QPushButton { background-color: #43A047; color: white; font-weight: bold; }"
            )

            def on_preview():
                try:
                    new_threshold = float(threshold_input.text())
                    n_outliers = int(np.sum(distances > new_threshold))
                    QtWidgets.QMessageBox.information(
                        self.plot_window, 'Preview',
                        f'This threshold would mark {n_outliers} spikes ({n_outliers / len(distances) * 100:.2f}%) as outliers.\n'
                        f'Maximum distance: {np.max(finite_distances):.1f}\n'
                        f'99.9th percentile: {np.percentile(finite_distances, 99.9):.1f}\n'
                        f'99th percentile: {np.percentile(finite_distances, 99):.1f}'
                    )
                except ValueError:
                    logger.error("Invalid threshold value")

            def on_apply():
                try:
                    new_threshold = float(threshold_input.text())
                    if not new_threshold > 0:
                        logger.error("Threshold must be positive")
                        return

                    self.current_threshold = new_threshold
                    n_outliers = int(np.sum(distances > new_threshold))

                    if n_outliers == 0 or n_outliers == len(distances):
                        QtWidgets.QMessageBox.warning(
                            self.plot_window, 'Nothing to split',
                            f'Threshold {new_threshold:.2f} puts all spikes in one group.'
                        )
                        return

                    if n_outliers > len(distances) * 0.1:
                        reply = QtWidgets.QMessageBox.question(
                            self.plot_window, 'Warning',
                            f'This threshold would mark {n_outliers} spikes ({n_outliers / len(distances) * 100:.1f}%) as outliers. Continue?',
                            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
                        )
                        if reply == QtWidgets.QMessageBox.No:
                            return

                    perform_outlier_detection(new_threshold, self.current_distances)
                    self.plot_window.close()

                except ValueError:
                    logger.error("Invalid threshold value")
                except Exception as e:
                    logger.error(f"Error applying threshold: {str(e)}")

            preview_button.clicked.connect(on_preview)
            apply_button.clicked.connect(on_apply)

            layout.addWidget(canvas)
            layout.addLayout(form_layout)
            layout.addSpacing(10)
            button_row = QtWidgets.QHBoxLayout()
            button_row.addStretch()
            button_row.addWidget(preview_button)
            button_row.addWidget(apply_button)
            button_row.addStretch()
            layout.addLayout(button_row)

            self.plot_window.resize(1900, 950)

            def select_text():
                threshold_input.setFocus()
                threshold_input.selectAll()

            qtimer_cls.singleShot(100, select_text)

            self.plot_window.show()

        def perform_outlier_detection(threshold, distances):
            """Perform outlier detection with given threshold"""
            if distances is None or self._spike_ids is None:
                logger.warn("No distances or spike IDs available")
                return

            try:
                outliers = distances > threshold
                n_outliers = int(np.sum(outliers))

                # Log results
                logger.info(f"Analysis with threshold {threshold}:")
                logger.info(f"- Detected {n_outliers} outliers ({n_outliers / len(distances) * 100:.1f}%)")
                logger.info(f"- Maximum distance: {np.max(distances):.1f}")
                logger.info(f"- 99.9th percentile: {np.percentile(distances, 99.9):.1f}")
                logger.info(f"- 99th percentile: {np.percentile(distances, 99):.1f}")
                logger.info(f"- Median distance: {np.median(distances):.1f}")

                # Sort and display top distances
                sorted_dist = np.sort(distances)[-10:]
                logger.info(f"Top 10 distances: {', '.join(f'{d:.1f}' for d in sorted_dist)}")

                # Prepare for split
                if 0 < n_outliers < len(distances):
                    labels = np.ones(len(distances), dtype=int)
                    labels[outliers] = 2
                    controller.supervisor.actions.split(self._spike_ids, labels)
                else:
                    logger.info("No valid split at current threshold")

            except Exception as e:
                logger.error(f"Error in outlier detection: {str(e)}")

        @connect
        def on_gui_ready(sender, gui):
            if self._shortcuts_created:
                return
            self._shortcuts_created = True

            @controller.supervisor.actions.add(shortcut='alt+x')
            def stable_mahalanobis_outliers():
                """
                Stable Mahalanobis Outlier Detection with visualization (Alt+X)
                """
                try:
                    # Get selected clusters and spikes
                    cluster_ids = controller.supervisor.selected
                    if not cluster_ids:
                        logger.warn("No clusters selected!")
                        return
                    cluster_id = int(cluster_ids[0])

                    bunchs = controller._amplitude_getter(cluster_ids, name='template', load_all=True)
                    self._spike_ids = np.asarray(bunchs[0].spike_ids, dtype=np.int64)

                    # Prepare features
                    features = prepare_features(self._spike_ids)
                    if features is None:
                        return

                    # Minimum spikes check
                    if features.shape[0] < features.shape[1] * 2:
                        logger.warn(f"Warning: Need at least {features.shape[1] * 2} spikes!")
                        return

                    # Compute distances
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        distances = stable_mahalanobis(features)

                    if distances is None:
                        return

                    # Store current distances
                    self.current_distances = distances
                    preview_distances, preview_traces, display_channel = (
                        prepare_waveform_preview(
                            self._spike_ids,
                            distances,
                            cluster_id,
                            max_preview_spikes=4000,
                        )
                    )

                    # Calculate initial threshold using chi-square distribution
                    initial_threshold = calculate_robust_threshold(distances)
                    if initial_threshold is None:
                        return

                    # Log distribution analysis
                    logger.info("\nDistribution Analysis:")
                    logger.info(f"Number of dimensions: {self._n_features}")
                    logger.info(f"Expected mean distance (sqrt(p)): {np.sqrt(self._n_features):.2f}")
                    logger.info(f"Observed mean distance: {np.mean(distances):.2f}")
                    logger.info(f"Observed median distance: {np.median(distances):.2f}")

                    # Check for substantial deviation from theoretical expectation
                    expected_mean = np.sqrt(self._n_features)
                    observed_mean = np.mean(distances)
                    if abs(observed_mean - expected_mean) / expected_mean > 0.5:
                        logger.warn(f"Substantial deviation from theoretical expectation:")
                        logger.warn(f"This might indicate non-normal features or other irregularities.")

                    # Show distribution plot
                    plot_distribution(
                        distances,
                        initial_threshold,
                        preview_distances=preview_distances,
                        preview_traces=preview_traces,
                        display_channel=display_channel,
                    )

                except Exception as e:
                    logger.error(f"Error in stable_mahalanobis_outliers: {str(e)}")
                    logger.error("Stack trace:", exc_info=True)
