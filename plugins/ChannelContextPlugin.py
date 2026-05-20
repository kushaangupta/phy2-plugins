"""Add or remove context channels from Phy's selected-cluster views.

Why this plugin exists
----------------------
Phy chooses a small list of "best channels" for each cluster. Those channels
are the ones normally shown in WaveformView, highlighted in TraceView, colored
in ProbeView, and used as defaults in FeatureView and AmplitudeView. This is
usually convenient, but sometimes you want to inspect a neighboring or related
channel that was not in that original list. This plugin lets you temporarily
add or remove channels from that best-channel list without rewriting Kilosort
or Phy output files.

What is changed, and what is not changed
----------------------------------------
The plugin monkey-patches Phy's in-memory controller methods so that selected
views *see* a modified best-channel list. It does not edit source data files
such as ``templates.npy``, ``pc_features.npy``, ``pc_feature_ind.npy``,
``amplitudes.npy``, ``spike_times.npy``, or raw binary traces. The only file
this plugin writes is ``channel_context_best_channels.json`` in the dataset
directory, and only when the "save channel context" action is run. That JSON
file stores your requested add/remove channel edits by cluster id, so the same
display context can be restored when Phy is reopened.

Important limitation: Kilosort PC features are sparse
-----------------------------------------------------
Kilosort/Phy PC features are not normally available on every channel for every
spike. They are stored sparsely: for each spike/template, ``pc_feature_ind.npy``
lists the channels where Kilosort actually saved PC features. If you ask Phy for
features on a channel that is outside that sparse list, Phy returns zeros. This
is not data loss; it means those PC features were never present in the saved
Kilosort feature matrix for that spike/channel.

To make FeatureView somewhat useful on manually added context channels, this
plugin fills missing features for plugin-added channels by computing simple
PCA-like features from the loaded waveforms for those spikes and channels. That
fallback is useful for visualization, but it is not identical to Kilosort's
original PC feature basis and should not be treated as a perfect replacement for
the original sparse features. Original Kilosort features are not overwritten.

AmplitudeView and the optional ``context_raw`` mode
--------------------------------------------------
AmplitudeView's ordinary "feature" mode uses PC features. On channels that were
not in the sparse Kilosort feature set, feature amplitudes can be zero for many
comparison cells. For that reason, the plugin adds a separate amplitude type
called ``context_raw``. It computes peak-to-peak raw waveform amplitude on the
currently selected channel, using the same channel for the selected cluster and
for grey comparison cells. This gives a more honest arbitrary-channel amplitude
comparison than sparse PC features can. The plugin does not force this mode; it
is available from AmplitudeView's amplitude-type cycling/menu when you need it.

Mean/template waveform refresh
------------------------------
Phy caches mean and template waveforms by cluster id. Channel-context edits
change the channel list without changing the cluster id, so this plugin has to
replace WaveformView's stored mean/template callbacks with uncached versions.
That is why the plugin explicitly refreshes WaveformView's waveform function
table after every channel-context edit.

Splits and merges
-----------------
When a cluster with channel-context edits is split, the new child clusters
inherit the parent's added/removed channels in the current Phy session. When
clusters are merged, the new cluster inherits the union of context edits from
the source clusters. Press the save action again after splitting or merging if
you want those newly inherited edits written to JSON for future Phy sessions.

Channel ids versus channel labels
---------------------------------
Phy often displays channel labels from ``channel_map.npy``. Those labels can be
different from Phy's internal zero-based channel indices. The prompt accepts the
displayed label by default. If you need to force an internal index, prefix it
with ``i`` or ``idx:``, for example ``i12`` or ``idx:12``.

Shortcuts
---------
``Shift+C``
    Prompt for a displayed channel label and toggle it for the selected cluster.
``Alt+Shift+A``
    Toggle channels immediately adjacent to the primary best channel.
``Ctrl+Shift+C``
    Clear all in-memory channel-context edits.
``Alt+Shift+S``
    Save current edits to ``channel_context_best_channels.json``.
"""

import logging
import json
from functools import partial

import numpy as np
from phylib.utils import Bunch, emit
from phylib.io.model import compute_features
from phy import IPlugin
from phy.utils import connect

logger = logging.getLogger(__name__)


class ChannelContextPlugin(IPlugin):
    def attach_to_controller(self, controller):
        self._added_by_cluster = {}
        self._removed_by_cluster = {}
        self._original_get_best_channels = controller.get_best_channels
        self._original_get_channel_amplitudes = getattr(
            controller, "get_channel_amplitudes", None
        )
        self._original_get_waveforms = controller._get_waveforms
        self._original_get_mean_waveforms = controller._get_mean_waveforms
        self._original_get_template_waveforms = getattr(
            controller, "_get_template_waveforms", None
        )
        self._original_get_features = getattr(controller, "_get_features", None)
        self._original_get_spike_features = getattr(
            controller, "_get_spike_features", None
        )
        self._original_get_spike_feature_amplitudes = getattr(
            controller, "get_spike_feature_amplitudes", None
        )
        self._original_amplitude_getter = getattr(controller, "_amplitude_getter", None)
        self._original_get_amplitude_functions = getattr(
            controller, "_get_amplitude_functions", None
        )
        self._context_path = controller.dir_path / "channel_context_best_channels.json"

        def _load_saved_context():
            if not self._context_path.exists():
                return
            try:
                with open(str(self._context_path), "r") as f:
                    payload = json.load(f)
            except Exception:
                logger.warning(
                    "ChannelContextPlugin: could not load %s.", self._context_path
                )
                return

            clusters = payload.get("clusters", {})
            for cluster_id, edit in clusters.items():
                try:
                    cluster_id = int(cluster_id)
                except (TypeError, ValueError):
                    continue
                added = []
                for ch in edit.get("added", []):
                    try:
                        ch = int(ch)
                    except (TypeError, ValueError):
                        continue
                    if _valid_channel(ch):
                        added.append(ch)
                removed = set()
                for ch in edit.get("removed", []):
                    try:
                        ch = int(ch)
                    except (TypeError, ValueError):
                        continue
                    if _valid_channel(ch):
                        removed.add(ch)
                if added:
                    self._added_by_cluster[cluster_id] = added
                if removed:
                    self._removed_by_cluster[cluster_id] = removed
            logger.info(
                "ChannelContextPlugin: loaded channel context from %s.",
                self._context_path,
            )

        def _save_context():
            clusters = {}
            for cluster_id in sorted(
                set(self._added_by_cluster) | set(self._removed_by_cluster)
            ):
                added = self._added_by_cluster.get(cluster_id, [])
                removed = sorted(self._removed_by_cluster.get(cluster_id, set()))
                if not added and not removed:
                    continue
                clusters[str(cluster_id)] = {
                    "added": [int(ch) for ch in added],
                    "removed": [int(ch) for ch in removed],
                    "added_labels": [_channel_label(ch) for ch in added],
                    "removed_labels": [_channel_label(ch) for ch in removed],
                }

            payload = {
                "version": 1,
                "note": (
                    "Saved by ChannelContextPlugin. Channel ids are Phy internal "
                    "indices; labels are included for readability."
                ),
                "clusters": clusters,
            }
            with open(str(self._context_path), "w") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
                f.write("\n")
            logger.info(
                "ChannelContextPlugin: saved channel context to %s.", self._context_path
            )

        def _valid_channel(channel_id):
            return 0 <= channel_id < controller.model.n_channels

        def _uses_mapped_channel_labels():
            return hasattr(controller.model, "channel_mapping") and getattr(
                controller.model,
                "show_mapped_channels",
                controller.default_show_mapped_channels,
            )

        def _channel_label(channel_id):
            labels = controller._get_channel_labels([channel_id])
            return labels[0] if labels else str(channel_id)

        def _parse_channel(channel_text):
            text = str(channel_text).strip().lower()
            force_internal = False
            for prefix in ("i:", "idx:", "index:"):
                if text.startswith(prefix):
                    text = text[len(prefix) :].strip()
                    force_internal = True
                    break
            if text.startswith("i") and text[1:].strip().isdigit():
                text = text[1:].strip()
                force_internal = True

            value = int(text)
            if force_internal or not _uses_mapped_channel_labels():
                return value

            matches = np.flatnonzero(controller.model.channel_mapping == value)
            if len(matches):
                return int(matches[0])
            return value

        def _base_channels(cluster_id):
            return [int(ch) for ch in self._original_get_best_channels(cluster_id)]

        def _added(cluster_id):
            return self._added_by_cluster.setdefault(int(cluster_id), [])

        def _removed(cluster_id):
            return self._removed_by_cluster.setdefault(int(cluster_id), set())

        def _has_context(cluster_id):
            cluster_id = int(cluster_id)
            return bool(
                self._added_by_cluster.get(cluster_id)
                or self._removed_by_cluster.get(cluster_id)
            )

        def _context_channels():
            out = set()
            for channels in self._added_by_cluster.values():
                out.update(channels)
            return out

        def _copy_context(old_cluster_id, new_cluster_id):
            old_cluster_id = int(old_cluster_id)
            new_cluster_id = int(new_cluster_id)
            if old_cluster_id == new_cluster_id:
                return False

            old_added = self._added_by_cluster.get(old_cluster_id, [])
            old_removed = self._removed_by_cluster.get(old_cluster_id, set())
            if not old_added and not old_removed:
                return False

            new_added = _added(new_cluster_id)
            changed = False
            for channel_id in old_added:
                if channel_id not in new_added:
                    new_added.append(channel_id)
                    changed = True

            new_removed = _removed(new_cluster_id)
            before = set(new_removed)
            new_removed.update(old_removed)
            changed = changed or before != new_removed

            if not new_added:
                self._added_by_cluster.pop(new_cluster_id, None)
            if not new_removed:
                self._removed_by_cluster.pop(new_cluster_id, None)
            return changed

        def _apply_cluster_update(up):
            changed = False
            for old_cluster_id, new_cluster_id in getattr(up, "descendants", []) or []:
                changed = _copy_context(old_cluster_id, new_cluster_id) or changed

            # Remove context attached to cluster ids Phy says no longer exist.
            for cluster_id in getattr(up, "deleted", []) or []:
                cluster_id = int(cluster_id)
                if cluster_id in self._added_by_cluster:
                    self._added_by_cluster.pop(cluster_id, None)
                    changed = True
                if cluster_id in self._removed_by_cluster:
                    self._removed_by_cluster.pop(cluster_id, None)
                    changed = True
            return changed

        def _effective_channels(cluster_id):
            base = _base_channels(cluster_id)
            removed = _removed(cluster_id)
            channels = [ch for ch in base if ch not in removed]
            for ch in _added(cluster_id):
                if _valid_channel(ch) and ch not in channels:
                    channels.append(ch)
            return channels

        def _set_channel_present(cluster_id, channel_id, present):
            base = _base_channels(cluster_id)
            added = _added(cluster_id)
            removed = _removed(cluster_id)

            if present:
                removed.discard(channel_id)
                if channel_id not in base and channel_id not in added:
                    added.append(channel_id)
            else:
                if channel_id in added:
                    added.remove(channel_id)
                if channel_id in base:
                    removed.add(channel_id)

            if not added:
                self._added_by_cluster.pop(int(cluster_id), None)
            if not removed:
                self._removed_by_cluster.pop(int(cluster_id), None)

        def _normalized_template_amplitudes(cluster_id, channel_ids):
            if not hasattr(controller, "get_template_for_cluster"):
                return None
            try:
                template_id = controller.get_template_for_cluster(cluster_id)
                template = controller.model.get_template(
                    template_id, channel_ids=np.asarray(channel_ids, dtype=np.int64)
                ).template
                amplitudes = template.max(axis=0) - template.min(axis=0)
            except Exception:
                return None
            max_amplitude = float(np.max(amplitudes)) if len(amplitudes) else 0.0
            if max_amplitude <= 0:
                return np.zeros(len(channel_ids), dtype=np.float32)
            return amplitudes / max_amplitude

        def _context_waveforms(cluster_id, n_spikes_waveforms, current_filter=None):
            pos = controller.model.channel_positions

            if getattr(controller.model, "spike_waveforms", None) is not None:
                subset_spikes = controller.model.spike_waveforms.spike_ids
                spike_ids = controller.selector(
                    n_spikes_waveforms, [cluster_id], subset_spikes=subset_spikes
                )
            else:
                spike_ids = controller.selector(
                    n_spikes_waveforms, [cluster_id], subset_chunks=True
                )

            channel_ids = np.asarray(_effective_channels(cluster_id), dtype=np.int64)
            data = controller.model.get_waveforms(spike_ids, channel_ids)
            if data is not None:
                data = data - np.median(data, axis=1)[:, np.newaxis, :]
                data = controller.raw_data_filter.apply(data, axis=1)

            return Bunch(
                data=data,
                channel_ids=channel_ids,
                channel_labels=controller._get_channel_labels(channel_ids),
                channel_positions=pos[channel_ids],
            )

        def get_best_channels(cluster_id):
            return _effective_channels(cluster_id)

        def get_channel_amplitudes(cluster_id):
            channel_ids = _effective_channels(cluster_id)
            template_amplitudes = _normalized_template_amplitudes(
                cluster_id, channel_ids
            )
            if template_amplitudes is not None:
                return channel_ids, list(template_amplitudes)

            if self._original_get_channel_amplitudes is not None:
                original_channels, amplitudes = self._original_get_channel_amplitudes(
                    cluster_id
                )
                amp_by_channel = {
                    int(ch): float(amp)
                    for ch, amp in zip(original_channels, amplitudes)
                }
            else:
                amp_by_channel = {ch: 1.0 for ch in _base_channels(cluster_id)}

            amplitudes = [amp_by_channel.get(int(ch), 0.0) for ch in channel_ids]
            return channel_ids, amplitudes

        def _feature_missing_mask(spike_ids, channel_ids, data):
            missing = np.isnan(data).any(axis=2)
            sparse_features = getattr(controller.model, "sparse_features", None)
            if sparse_features is None or sparse_features.cols is None:
                return missing
            cols = sparse_features.cols[controller.model.spike_templates[spike_ids]]
            for channel_index, channel_id in enumerate(channel_ids):
                missing[:, channel_index] |= ~np.any(cols == channel_id, axis=1)
            if sparse_features.rows is not None:
                missing |= ~np.isin(spike_ids, sparse_features.rows)[:, np.newaxis]
            return missing

        def _fill_missing_features_from_waveforms(spike_ids, channel_ids, data):
            if data is None or not len(spike_ids):
                return data
            missing = _feature_missing_mask(spike_ids, channel_ids, data)
            if not np.any(missing):
                return data

            context_channels = _context_channels()
            for channel_index, channel_id in enumerate(channel_ids):
                if int(channel_id) not in context_channels:
                    continue
                rows = np.flatnonzero(missing[:, channel_index])
                if not len(rows):
                    continue
                try:
                    waveforms = controller.model.get_waveforms(
                        spike_ids[rows], np.asarray([channel_id], dtype=np.int64)
                    )
                    if waveforms is None or not len(waveforms):
                        continue
                    data[rows, channel_index, :] = compute_features(waveforms)[:, 0, :]
                except Exception as e:
                    logger.debug(
                        "ChannelContextPlugin: feature fallback failed for channel %d: %s",
                        channel_id,
                        e,
                    )
            return data

        def get_spike_features(spike_ids, channel_ids):
            if self._original_get_spike_features is None:
                return Bunch()
            spike_ids = np.asarray(spike_ids)
            channel_ids = np.asarray(channel_ids, dtype=np.int64)
            if len(spike_ids) == 0:
                return Bunch()
            data = controller.model.get_features(spike_ids, channel_ids)
            if data is None:
                return Bunch()
            data = np.asarray(data).copy()
            data = _fill_missing_features_from_waveforms(spike_ids, channel_ids, data)
            data[np.isnan(data)] = 0
            return Bunch(
                data=data,
                spike_ids=spike_ids,
                channel_ids=channel_ids,
                channel_labels=controller._get_channel_labels(channel_ids),
            )

        def get_features(cluster_id=None, channel_ids=None, load_all=False):
            if self._original_get_features is None:
                return Bunch()
            spike_ids = controller._get_feature_view_spike_ids(
                cluster_id, load_all=load_all
            )
            if spike_ids is None or len(spike_ids) == 0:
                return Bunch()
            if cluster_id is not None and channel_ids is None:
                channel_ids = controller.get_best_channels(cluster_id)
            return get_spike_features(spike_ids, channel_ids)

        def get_spike_feature_amplitudes(
            spike_ids, channel_id=None, channel_ids=None, pc=None, **kwargs
        ):
            if self._original_get_spike_feature_amplitudes is None:
                return None
            if getattr(controller.model, "features", None) is None:
                return self._original_get_spike_feature_amplitudes(
                    spike_ids,
                    channel_id=channel_id,
                    channel_ids=channel_ids,
                    pc=pc,
                    **kwargs,
                )
            channel_id = channel_id if channel_id is not None else channel_ids[0]
            features = get_spike_features(spike_ids, [channel_id]).get("data", None)
            if features is None:
                return None
            return features[:, 0, pc or 0]

        def get_spike_context_raw_amplitudes(
            spike_ids, channel_id=None, channel_ids=None, **kwargs
        ):
            channel_id = channel_id if channel_id is not None else channel_ids[0]
            if channel_id is None or not _valid_channel(int(channel_id)):
                return None
            waveforms = controller.model.get_waveforms(
                spike_ids, np.asarray([int(channel_id)], dtype=np.int64)
            )
            if waveforms is None:
                return None
            waveforms = waveforms[..., 0]
            waveforms = controller.raw_data_filter.apply(waveforms, axis=1)
            return waveforms.max(axis=1) - waveforms.min(axis=1)

        def get_amplitude_functions():
            if self._original_get_amplitude_functions is None:
                out = {}
            else:
                out = self._original_get_amplitude_functions().copy()
            out["context_raw"] = get_spike_context_raw_amplitudes
            return out

        def amplitude_getter(cluster_ids, name=None, load_all=False):
            if self._original_amplitude_getter is None:
                return []

            out = []
            n = controller.n_spikes_amplitudes if not load_all else None
            first_cluster = next(
                cluster_id for cluster_id in cluster_ids if cluster_id is not None
            )
            channel_ids = controller.get_best_channels(first_cluster)
            default_channel_id = channel_ids[0]
            selected_channel_id = controller.selection.get("channel_id", None)
            if (
                name in ("feature", "raw", "context_raw")
                and selected_channel_id is not None
                and _valid_channel(int(selected_channel_id))
            ):
                channel_id = int(selected_channel_id)
            else:
                channel_id = default_channel_id

            # The grey comparison set must be chosen from the same channel
            # whose amplitudes are plotted, otherwise unrelated cells collapse
            # to zero when sparse PC features are missing on that channel.
            other_clusters = controller.get_clusters_on_channel(channel_id)
            other_clusters = [c for c in other_clusters if c not in cluster_ids]
            f = controller._get_amplitude_functions()[name]

            subset_chunks = subset_spikes = None
            if name in ("raw", "context_raw"):
                if controller.model.spike_waveforms is not None:
                    subset_spikes = controller.model.spike_waveforms.spike_ids
                else:
                    subset_chunks = True

            for cluster_id in cluster_ids:
                if cluster_id is not None:
                    spike_ids = controller.get_spike_ids(
                        cluster_id,
                        n=n,
                        subset_spikes=subset_spikes,
                        subset_chunks=subset_chunks,
                    )
                else:
                    spike_ids = controller.selector(
                        n,
                        other_clusters,
                        subset_spikes=subset_spikes,
                        subset_chunks=subset_chunks,
                    )
                spike_times = controller._get_spike_times_reordered(spike_ids)
                pc = controller.selection.get("feature_pc", None)
                amplitudes = f(
                    spike_ids,
                    channel_ids=channel_ids,
                    channel_id=channel_id,
                    pc=pc,
                    first_cluster=first_cluster,
                )
                if amplitudes is None:
                    continue
                assert amplitudes.shape == spike_ids.shape == spike_times.shape
                out.append(
                    Bunch(
                        amplitudes=amplitudes,
                        spike_ids=spike_ids,
                        spike_times=spike_times,
                    )
                )
            return out

        def get_waveforms(cluster_id):
            if _has_context(cluster_id):
                return _context_waveforms(
                    cluster_id,
                    controller.n_spikes_waveforms,
                    current_filter=controller.raw_data_filter.current,
                )
            return self._original_get_waveforms(cluster_id)

        def get_mean_waveforms(cluster_id, current_filter=None):
            if not _has_context(cluster_id):
                return self._original_get_mean_waveforms(
                    cluster_id, current_filter=current_filter
                )
            bunch = _context_waveforms(
                cluster_id, controller.n_spikes_waveforms, current_filter=current_filter
            )
            if bunch.data is not None:
                bunch.data = bunch.data.mean(axis=0)[np.newaxis, ...]
            bunch["alpha"] = 1.0
            return bunch

        def get_template_waveforms(cluster_id):
            if self._original_get_template_waveforms is None:
                return Bunch()
            if not _has_context(cluster_id):
                return self._original_get_template_waveforms(cluster_id)

            pos = controller.model.channel_positions
            count = controller.get_template_counts(cluster_id)
            template_ids = np.nonzero(count)[0]
            count = count[template_ids]
            channel_ids = np.asarray(_effective_channels(cluster_id), dtype=np.int64)
            masks = count / float(count.max())
            masks = np.tile(masks.reshape((-1, 1)), (1, len(channel_ids)))
            templates = [
                controller.model.get_template(template_id)
                for template_id in template_ids
            ]
            ns = controller.model.n_samples_waveforms
            data = np.zeros((len(template_ids), ns, controller.model.n_channels))
            for i, template in enumerate(templates):
                data[i][:, template.channel_ids] = template.template
            waveforms = data[..., channel_ids]
            return Bunch(
                data=waveforms,
                channel_ids=channel_ids,
                channel_labels=controller._get_channel_labels(channel_ids),
                channel_positions=pos[channel_ids],
                masks=masks,
                alpha=1.0,
            )

        controller.get_best_channels = get_best_channels
        if self._original_get_channel_amplitudes is not None:
            controller.get_channel_amplitudes = get_channel_amplitudes
        controller._get_waveforms = get_waveforms
        controller._get_mean_waveforms = get_mean_waveforms
        if self._original_get_template_waveforms is not None:
            controller._get_template_waveforms = get_template_waveforms
        if self._original_get_features is not None:
            controller._get_features = get_features
        if self._original_get_spike_features is not None:
            controller._get_spike_features = get_spike_features
        if self._original_get_spike_feature_amplitudes is not None:
            controller.get_spike_feature_amplitudes = get_spike_feature_amplitudes
        if self._original_amplitude_getter is not None:
            controller._amplitude_getter = amplitude_getter
        if self._original_get_amplitude_functions is not None:
            controller._get_amplitude_functions = get_amplitude_functions

        _load_saved_context()

        @connect
        def on_gui_ready(sender, gui):
            if sender is not controller:
                return

            supervisor = getattr(controller, "supervisor", None)
            if supervisor is None:
                return

            def _selected_clusters():
                return [int(cluster_id) for cluster_id in supervisor.selected]

            def _clear_view_caches():
                context = getattr(controller, "context", None)
                memcache = getattr(context, "_memcache", {})
                for name, cache in list(memcache.items()):
                    if (
                        "waveform" in name
                        or "template" in name
                        or "get_features" in name
                        or "get_spike_feature_amplitudes" in name
                    ):
                        cache.clear()
                for method_name in (
                    "_get_features",
                    "get_spike_feature_amplitudes",
                    "_get_waveforms_with_n_spikes",
                    "_get_mean_waveforms",
                    "_get_template_waveforms",
                ):
                    method = getattr(controller, method_name, None)
                    clear = getattr(method, "clear", None)
                    if clear is not None:
                        try:
                            clear(warn=False)
                        except TypeError:
                            clear()

            def _refresh_waveform_view_functions(view):
                current = view.waveforms_type
                view.waveforms["waveforms"] = get_waveforms
                if "mean_waveforms" in view.waveforms:
                    view.waveforms["mean_waveforms"] = get_mean_waveforms
                if (
                    self._original_get_template_waveforms is not None
                    and "templates" in view.waveforms
                ):
                    view.waveforms["templates"] = get_template_waveforms

                if hasattr(view, "waveforms_types"):
                    view.waveforms_types._choices = {}
                    view.waveforms_types._current = None
                    for name, value in view.waveforms.items():
                        view.waveforms_types.add(name, value)
                    view.waveforms_types.set(
                        current if current in view.waveforms else None
                    )

                view.data_bounds = None

            def _refresh_amplitude_view_functions(view):
                view.amplitudes = {
                    name: partial(amplitude_getter, name=name)
                    for name in sorted(controller._get_amplitude_functions())
                }
                if hasattr(view, "amplitudes_types"):
                    current = view.amplitudes_type
                    view.amplitudes_types._choices = {}
                    view.amplitudes_types._current = None
                    for name, value in view.amplitudes.items():
                        view.amplitudes_types.add(name, value)
                    view.amplitudes_types.set(
                        current if current in view.amplitudes else "context_raw"
                    )

            def _force_update():
                selected = _selected_clusters()
                if not selected:
                    return

                _clear_view_caches()
                try:
                    from phy.cluster.views import (
                        AmplitudeView,
                        FeatureView,
                        ProbeView,
                        TraceView,
                        WaveformView,
                    )

                    for view in gui.list_views(WaveformView):
                        _refresh_waveform_view_functions(view)
                        view.on_select_threaded(supervisor, selected, gui=gui)
                    for view in gui.list_views(FeatureView):
                        view.features = get_features
                        if not view.fixed_channels:
                            view.channel_ids = None
                        view.on_select_threaded(supervisor, selected, gui=gui)
                    for view in gui.list_views(AmplitudeView):
                        _refresh_amplitude_view_functions(view)
                        if view.auto_update and view.cluster_ids:
                            view.plot()
                    for view in gui.list_views(TraceView):
                        if view.auto_update:
                            view.set_interval()
                    for view in gui.list_views(ProbeView):
                        view.best_channels = get_best_channels
                        if view.auto_update:
                            view.on_select(selected)
                finally:
                    emit("select", supervisor, selected)

            @connect
            def on_view_attached(view, gui_):
                try:
                    from phy.cluster.views import AmplitudeView, ProbeView, WaveformView
                except ImportError:
                    return
                if isinstance(view, WaveformView):
                    _refresh_waveform_view_functions(view)
                    selected = _selected_clusters()
                    if selected and view.auto_update:
                        view.on_select_threaded(supervisor, selected, gui=gui_)
                elif isinstance(view, ProbeView):
                    view.best_channels = get_best_channels
                    selected = _selected_clusters()
                    if selected and view.auto_update:
                        view.on_select(selected)
                elif isinstance(view, AmplitudeView):
                    _refresh_amplitude_view_functions(view)

            @connect(sender=supervisor)
            def on_select(sender, cluster_ids, update_views=True):
                try:
                    from phy.cluster.views import ProbeView
                except ImportError:
                    return
                for view in gui.list_views(ProbeView):
                    view.best_channels = get_best_channels
                    if update_views and view.auto_update and cluster_ids:
                        view.on_select(cluster_ids)

            @connect(sender=supervisor)
            def on_cluster(sender, up):
                if _apply_cluster_update(up):
                    logger.info(
                        "ChannelContextPlugin: inherited channel context for clusters %s.",
                        getattr(up, "added", []),
                    )
                    _clear_view_caches()

            @supervisor.actions.add(shortcut="shift+c", prompt=True)
            def toggle_channel(channel_id):
                """Toggle a channel for the selected cluster context."""
                try:
                    channel_id = _parse_channel(channel_id)
                except (TypeError, ValueError):
                    logger.warning(
                        "ChannelContextPlugin: please enter a valid channel integer."
                    )
                    return

                if not _valid_channel(channel_id):
                    logger.warning(
                        "ChannelContextPlugin: invalid channel %d.", channel_id
                    )
                    return

                selected = _selected_clusters()
                if not selected:
                    logger.warning(
                        "ChannelContextPlugin: select a cluster before toggling channels."
                    )
                    return

                present_in_all = all(
                    channel_id in _effective_channels(cluster_id)
                    for cluster_id in selected
                )
                for cluster_id in selected:
                    _set_channel_present(cluster_id, channel_id, not present_in_all)

                action = "Removed" if present_in_all else "Added"
                label = _channel_label(channel_id)
                logger.info(
                    "ChannelContextPlugin: %s channel label %s (index %d) for clusters %s.",
                    action,
                    label,
                    channel_id,
                    selected,
                )
                _force_update()

            @supervisor.actions.add(shortcut="ctrl+shift+c")
            def reset_toggled_channels():
                """Reset all channel context edits."""
                self._added_by_cluster.clear()
                self._removed_by_cluster.clear()
                logger.info("ChannelContextPlugin: reset all channel context edits.")
                _force_update()

            @supervisor.actions.add(shortcut="alt+ctrl+s")
            def save_channel_context():
                """Save channel context edits for this dataset."""
                _save_context()

            @supervisor.actions.add(shortcut="alt+shift+a")
            def toggle_adjacent_channels():
                """Toggle channels adjacent to the primary best channel."""
                selected = _selected_clusters()
                if not selected:
                    return

                changed = False
                for cluster_id in selected:
                    base = _base_channels(cluster_id)
                    if not base:
                        continue
                    primary = base[0]
                    for channel_id in (primary - 1, primary + 1):
                        if not _valid_channel(channel_id) or channel_id in base:
                            continue
                        present = channel_id in _effective_channels(cluster_id)
                        _set_channel_present(cluster_id, channel_id, not present)
                        changed = True

                if changed:
                    logger.info(
                        "ChannelContextPlugin: toggled adjacent channels for clusters %s.",
                        selected,
                    )
                    _force_update()
