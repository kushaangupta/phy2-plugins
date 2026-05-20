"""Place field viewer plugin for Phy.

This plugin computes and displays 2D place field maps and spike-trajectory plots
for the currently selected clusters in Phy. It leverages behavioral position data
found in the parent directory of your Kilosort output to map neural activity to
the animal's physical location during the recording session.

=============================
Rationale & Core Features
=============================

1. neuro_py.tuning.SpatialMap Integration:
   Instead of manually binning spikes and position data, this plugin utilizes
   `neuro_py.tuning.SpatialMap`. This is highly beneficial because SpatialMap
   natively employs speed-thresholding to filter out non-locomotor periods
   (e.g., when the animal is resting or grooming). This ensures that the resulting
   place fields reflect true navigation-related firing rather than stationary bursting.

2. Advanced Visualization:
   - **Ratemaps with Contours**: It plots the spatial firing rate (ratemap) and
     overlays contour boundaries around the detected place fields (using
     `skimage.measure.find_contours`). This allows you to immediately see the
     mathematically defined boundaries of the place field.
   - **Spike-on-Trajectory**: It plots the animal's physical trajectory and
     overlays red dots exactly where the neuron fired. This visualization is
     constrained to run-epochs, visually confirming the accuracy of the ratemap.
   - **Multi-Epoch Viewing**: It displays the global place field (across all tasks)
     as well as individual epochs side-by-side for comparison.

3. Multi-Cluster Batch Processing:
   If you select multiple clusters (e.g., in the Cluster View or Similarity View)
   and trigger this plugin, it will independently compute and spawn a place field
   diagnostic window for EACH selected cluster. This significantly accelerates
   the workflow when screening populations of cells.

=============================
Usage
=============================
Shortcut: **Alt+Shift+P** — Compute & show the place field for the selected cluster(s).

=============================
Requirements
=============================
- `nelpy` and `neuro_py` packages must be installed and accessible. If they are in
  a separate conda environment, the plugin will attempt to auto-detect
  and load them, or you can specify `PLACEFIELD_SITE_PACKAGES` as an environment variable.
- The parent directory of your Kilosort folder must contain:
  - `*.animal.behavior.mat` (Positional tracking data)
  - `*.session.mat` (Session metadata containing task epoch definitions)
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
from phy import IPlugin, connect

logger = logging.getLogger("phy")

# ---------------------------------------------------------------------------
# Optionally extend sys.path so nelpy / neuro_py can be imported from
# another conda environment.  Set this in phy_config.py:
#   c.PlaceFieldPlugin.site_packages = "/path/to/envs/vr2p/lib/python3.12/site-packages"
# or as an environment variable:
#   export PLACEFIELD_SITE_PACKAGES=/path/...
# ---------------------------------------------------------------------------
_extra_sp = os.environ.get("PLACEFIELD_SITE_PACKAGES", None)
if _extra_sp is None:
    # Auto-detect: look for nelpy in the vr2p conda env as a fallback.
    import glob as _glob

    _candidates = (
        _glob.glob(
            os.path.expanduser(
                "~/programs/miniforge3/envs/vr2p/lib/python*/site-packages"
            )
        )
        + _glob.glob(
            os.path.expanduser("~/miniforge3/envs/vr2p/lib/python*/site-packages")
        )
        + _glob.glob(
            os.path.expanduser("~/anaconda3/envs/vr2p/lib/python*/site-packages")
        )
    )
    for _cand in _candidates:
        if os.path.isdir(os.path.join(_cand, "nelpy")):
            _extra_sp = _cand
            break
if _extra_sp and _extra_sp not in sys.path:
    sys.path.insert(0, _extra_sp)


class PlaceFieldPlugin(IPlugin):
    """Show 2D place-field maps for the selected cluster."""

    def attach_to_controller(self, controller):
        model = getattr(controller, "model", None)
        if model is None:
            return

        # Resolve basepath: parent of the Kilosort directory.
        dir_path = getattr(model, "dir_path", None)
        if dir_path is None:
            logger.warning("PlaceFieldPlugin: model.dir_path is not set.")
            return
        basepath = str(Path(dir_path).parent)
        sample_rate = float(model.sample_rate)

        logger.info(
            "PlaceFieldPlugin: basepath = %s, sample_rate = %.0f Hz",
            basepath,
            sample_rate,
        )

        # Pre-load behavior & epoch data once at startup.
        behavior_cache: dict = {}

        def _ensure_behavior_loaded():
            """Load position & epoch data (cached across calls)."""
            if "pos" in behavior_cache:
                return True  # already loaded

            try:
                import neuro_py as npy
                import nelpy as nel
            except ImportError as exc:
                logger.error(
                    "PlaceFieldPlugin: could not import nelpy/neuro_py — "
                    "set PLACEFIELD_SITE_PACKAGES env var.  Error: %s",
                    exc,
                )
                return False

            # ── Load epoch info ──
            try:
                epoch_df = npy.io.load_epoch(basepath)
            except Exception as exc:
                logger.error("PlaceFieldPlugin: load_epoch failed: %s", exc)
                return False

            if epoch_df is None or epoch_df.empty:
                logger.warning("PlaceFieldPlugin: no epoch data found.")
                return False

            # ── Task epochs (non-sleep) ──
            try:
                task_mask = ~epoch_df.name.str.contains("sleep", case=False, na=False)
                task_ep_indices = np.where(task_mask.values)[0]
            except Exception:
                task_ep_indices = np.arange(len(epoch_df))

            if len(task_ep_indices) == 0:
                logger.warning("PlaceFieldPlugin: no task epochs found.")
                return False

            task_epochs = epoch_df.iloc[task_ep_indices]
            task_support = nel.EpochArray(
                np.array([task_epochs.startTime.values, task_epochs.stopTime.values]).T
            )

            # ── Load position ──
            try:
                position_df = npy.io.load_animal_behavior(basepath)
            except Exception as exc:
                logger.error("PlaceFieldPlugin: load_animal_behavior failed: %s", exc)
                return False

            if position_df is None or position_df.empty:
                logger.warning("PlaceFieldPlugin: no behavior data in %s.", basepath)
                return False

            # Interpolate NaNs.
            for col in ("x", "y"):
                if col in position_df.columns:
                    position_df[col] = (
                        position_df[col].interpolate(method="linear").ffill().bfill()
                    )

            ts_col = "timestamps" if "timestamps" in position_df.columns else "time"
            pos = nel.AnalogSignalArray(
                data=position_df[["x", "y"]].values.T,
                timestamps=position_df[ts_col].values,
            )

            # Compute speed for SpatialMap (non-epoched data)
            try:
                speed = nel.utils.ddt_asa(pos, smooth=True, sigma=0.1, norm=True)
            except Exception as exc:
                logger.warning("PlaceFieldPlugin: could not compute speed: %s", exc)
                speed = None

            behavior_cache["pos"] = pos
            behavior_cache["speed"] = speed
            behavior_cache["task_support"] = task_support
            behavior_cache["task_epochs"] = task_epochs
            behavior_cache["task_ep_indices"] = task_ep_indices
            behavior_cache["epoch_df"] = epoch_df

            logger.info(
                "PlaceFieldPlugin: loaded %d task epoch(s), position has %d samples.",
                len(task_ep_indices),
                len(position_df),
            )
            return True

        @connect
        def on_gui_ready(sender, gui):
            if sender is not controller:
                return

            supervisor = getattr(controller, "supervisor", None)
            if supervisor is None:
                return

            @supervisor.actions.add(shortcut="alt+shift+p")
            def show_place_field():
                """Compute and display the 2D place field for the selected cluster."""
                selected = list(getattr(supervisor, "selected", []) or [])
                if not selected:
                    logger.warning("PlaceFieldPlugin: no cluster selected.")
                    return

                if not _ensure_behavior_loaded():
                    logger.error(
                        "PlaceFieldPlugin: cannot compute — "
                        "behavior data not available."
                    )
                    return

                import nelpy as nel
                import neuro_py as npy

                pos = behavior_cache["pos"]
                speed = behavior_cache["speed"]
                task_support = behavior_cache["task_support"]
                task_epochs = behavior_cache["task_epochs"]
                task_ep_indices = behavior_cache["task_ep_indices"]
                epoch_df = behavior_cache["epoch_df"]

                # ── Session epochs as EpochArray ──
                sess_epochs = nel.EpochArray(
                    np.array([epoch_df.startTime.values, epoch_df.stopTime.values]).T
                )

                for cluster_id in selected:
                    # ── Get spike times (in seconds) ──
                    spike_ids = controller.supervisor.clustering.spikes_in_clusters(
                        [cluster_id]
                    )
                    spike_samples = model.spike_samples[spike_ids]
                    spike_times = spike_samples / sample_rate

                    n_spikes = len(spike_times)
                    if n_spikes == 0:
                        logger.warning(
                            "PlaceFieldPlugin: cluster %d has no spikes.",
                            cluster_id,
                        )
                        continue

                    logger.info(
                        "PlaceFieldPlugin: computing place field for "
                        "cluster %d (%d spikes)…",
                        cluster_id,
                        n_spikes,
                    )

                    # ── Build SpikeTrainArray (full session) ──
                    st = nel.SpikeTrainArray(
                        timestamps=spike_times,
                        support=task_support,
                        unit_ids=[cluster_id],
                    )

                    if st.isempty or st.n_spikes[0] == 0:
                        logger.warning(
                            "PlaceFieldPlugin: cluster %d has no spikes in task "
                            "epochs.",
                            cluster_id,
                        )
                        continue

                    # ── Compute tuning curves: combined + per-epoch ──
                    maps = []  # list of (label, tc_map, pos_epoch, st_epoch)

                    # Combined across all task epochs.
                    try:
                        tc_all_sm = npy.tuning.SpatialMap(
                            pos=pos[task_support],
                            st=st[task_support],
                            s_binsize=3,
                            speed=speed,
                            speed_thres=4,
                            tuning_curve_sigma=3,
                            place_field_min_size=15,
                            place_field_max_size=1000,
                            place_field_sigma=3,
                        )
                        tc_all_sm.find_fields()
                        maps.append(
                            ("combined", tc_all_sm, pos[task_support], st[task_support])
                        )
                    except Exception as exc:
                        logger.warning(
                            "PlaceFieldPlugin: combined map failed for cluster %d: %s",
                            cluster_id,
                            exc,
                        )

                    # Per individual task epoch.
                    for ep_idx in task_ep_indices:
                        ep_epoch = sess_epochs[ep_idx]
                        ep_name = str(epoch_df.name.iloc[ep_idx])
                        pos_ep = pos[ep_epoch]
                        st_ep = st[ep_epoch]
                        if st_ep.isempty or st_ep.n_spikes[0] == 0:
                            continue
                        try:
                            tc_ep_sm = npy.tuning.SpatialMap(
                                pos=pos_ep,
                                st=st_ep,
                                s_binsize=3,
                                speed=speed,
                                speed_thres=4,
                                tuning_curve_sigma=3,
                                place_field_min_size=15,
                                place_field_max_size=1000,
                                place_field_sigma=3,
                            )
                            tc_ep_sm.find_fields()
                            maps.append((ep_name, tc_ep_sm, pos_ep, st_ep))
                        except Exception as exc:
                            logger.warning(
                                "PlaceFieldPlugin: epoch '%s' failed for cluster %d: %s",
                                ep_name,
                                cluster_id,
                                exc,
                            )
                            continue

                    if not maps:
                        logger.warning(
                            "PlaceFieldPlugin: no valid spatial maps computed for cluster %d.",
                            cluster_id,
                        )
                        continue

                    # ── Plot ──
                    _show_place_field_plot(
                        maps,
                        cluster_id,
                        n_spikes,
                    )


def _show_place_field_plot(maps, cluster_id, n_total):
    """Display ratemap + spike-on-trajectory for each epoch using SpatialMap.

    Parameters
    ----------
    maps : list of (label, SpatialMap, pos_epoch, st_epoch)
    cluster_id : int
    n_total : int
        Total number of spikes in the cluster.
    """
    import matplotlib

    matplotlib.use("Qt5Agg")
    import matplotlib.pyplot as plt
    import nelpy as nel
    import skimage.measure

    try:
        import seaborn as sns

        use_sns = True
    except ImportError:
        use_sns = False

    n_rows = len(maps)
    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(10, 4 * n_rows),
        squeeze=False,
    )

    ratemap_i = 0  # We only have one unit

    for row_idx, (label, map_obj, pos_ep, st_ep) in enumerate(maps):
        ax_heat = axes[row_idx, 0]
        ax_pos = axes[row_idx, 1]

        # ── Ratemap heatmap ──
        ratemap_ = map_obj.tc.ratemap[ratemap_i].copy()
        ratemap_[map_obj.tc.occupancy < 0.01] = np.nan

        if use_sns:
            sns.heatmap(ratemap_.T, ax=ax_heat, cmap="jet")
        else:
            im = ax_heat.imshow(
                ratemap_.T,
                origin="lower",
                aspect="auto",
                cmap="jet",
                interpolation="bilinear",
            )
            fig.colorbar(im, ax=ax_heat, shrink=0.8)

        # Plot place field boundaries
        try:
            field_ids = np.unique(map_obj.tc.field_mask[ratemap_i])
            if len(field_ids) > 1:
                # Loop through fields (excluding background 0)
                for field_i in range(len(field_ids) - 1):
                    bc = skimage.measure.find_contours(
                        (map_obj.tc.field_mask[ratemap_i] == field_i + 1).T,
                        0,
                        fully_connected="low",
                        positive_orientation="low",
                    )
                    for c_ in bc:
                        ax_heat.plot(c_[:, 1], c_[:, 0], linewidth=3)
        except Exception as exc:
            logger.debug("PlaceFieldPlugin: contour plot failed: %s", exc)

        ax_heat.invert_yaxis()
        ax_heat.set_aspect("equal")

        # Extract spatial information metrics
        try:
            field_w = map_obj.tc.field_width[ratemap_i]
            field_p = map_obj.tc.field_peak_rate[ratemap_i]
            si = map_obj.tc.spatial_information()[ratemap_i]
        except Exception:
            field_w, field_p, si = np.nan, np.nan, np.nan

        # Optionally, get shuffle values if shuffle_spatial_information() was run
        # but for performance we skip shuffle by default, unless pre-computed.
        try:
            si_z = map_obj.spatial_information_zscore[ratemap_i]
            si_p = map_obj.spatial_information_pvalues[ratemap_i]
            z_str = f"; p {si_p:.3f}; Z {si_z:.2f}"
        except Exception:
            z_str = ""

        ax_heat.set_title(
            f"field {field_w:.2f}; peak {field_p:.2f}\ninfo {si:.2f}{z_str}",
            fontsize=9,
        )

        # ── Trajectory + spikes ──
        # Plot trajectory only during running epochs
        try:
            pos_run = pos_ep[map_obj.run_epochs]
            nel.plotting.plot2d(pos_run, lw=2, c="0.8", ax=ax_pos)

            # Plot spikes that occurred during running epochs
            st_run = st_ep[map_obj.run_epochs]
            if not st_run.isempty and st_run.n_spikes[ratemap_i] > 0:
                _, pos_at_spikes = pos_run.asarray(at=st_run.data[ratemap_i])
                ax_pos.plot(
                    pos_at_spikes[0, :],
                    pos_at_spikes[1, :],
                    ".",
                    color="k",
                    markersize=3,
                )
        except Exception as exc:
            logger.debug("PlaceFieldPlugin: spike overlay failed: %s", exc)

        ax_pos.set_aspect("equal")
        ax_pos.set_title(
            f"Cluster {cluster_id} | {label}",
            fontsize=9,
        )

    fig.suptitle(
        f"Place field — Cluster {cluster_id}  ({n_total} total spikes)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.show()
    logger.info(
        "PlaceFieldPlugin: displayed %d map(s) for cluster %d.",
        n_rows,
        cluster_id,
    )
