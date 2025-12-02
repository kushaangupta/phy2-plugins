"""Show how to write a custom split action."""

from phy import IPlugin, connect
import numpy as np
import logging
import os
import scipy.io as sio

logger = logging.getLogger("phy")

def in_intervals(
    timestamps: np.ndarray, intervals: np.ndarray, return_interval=False, shift=False
) -> np.ndarray:
    """
    Find which timestamps fall within the given intervals.

    Parameters
    ----------
    timestamps : ndarray
        An array of timestamp values. Assumes sorted.
    intervals : ndarray
        An array of time intervals, represented as pairs of start and end times.
    return_interval : bool, optional (default=False)
        If True, return the index of the interval to which each timestamp belongs.
    shift : bool, optional (default=False)
        If True, return the shifted timestamps

    Returns
    -------
    in_interval : ndarray
        A logical index indicating which timestamps fall within the intervals.
    interval : ndarray, optional
        A ndarray indicating for each timestamps which interval it was within.
    shifted_timestamps : ndarray, optional
        The shifted timestamps

    Examples
    --------
    >>> timestamps = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    >>> intervals = np.array([[2, 4], [5, 7]])
    >>> in_intervals(timestamps, intervals)
    array([False,  True,  True,  True,  True,  True,  True, False])

    >>> in_intervals(timestamps, intervals, return_interval=True)
    (array([False,  True,  True,  True,  True,  True,  True, False]),
    array([nan,  0.,  0.,  0.,  1.,  1.,  1., nan]))

    >>> in_intervals(timestamps, intervals, shift=True)
    (array([False,  True,  True,  True,  True,  True,  True, False]),
    array([0, 1, 2, 2, 3, 4]))

    >>> in_intervals(timestamps, intervals, return_interval=True, shift=True)
    (array([False,  True,  True,  True,  True,  True,  True, False]),
    array([0, 0, 0, 1, 1, 1]),
    array([0, 1, 2, 2, 3, 4]))
    """
    in_interval = np.zeros(timestamps.shape, dtype=np.bool_)
    interval = np.full(timestamps.shape, np.nan)

    for i, (start, end) in enumerate(intervals):
        # Find the leftmost index of a timestamp that is >= start
        left = np.searchsorted(timestamps, start, side="left")
        if left == len(timestamps):
            # If start is greater than all timestamps, skip this interval
            continue
        # Find the rightmost index of a timestamp that is <= end
        right = np.searchsorted(timestamps, end, side="right")
        if right == left:
            # If there are no timestamps in the interval, skip it
            continue
        # Mark the timestamps in the interval
        in_interval[left:right] = True
        interval[left:right] = i

    if shift:
        # Restrict to the timestamps that fall within the intervals
        interval = interval[in_interval].astype(int)

        # Calculate shifts based on intervals
        shifts = np.insert(np.cumsum(intervals[1:, 0] - intervals[:-1, 1]), 0, 0)[
            interval
        ]

        # Apply shifts to timestamps
        shifted_timestamps = timestamps[in_interval] - shifts - intervals[0, 0]

    if return_interval and shift:
        return in_interval, interval, shifted_timestamps

    if return_interval:
        return in_interval, interval

    if shift:
        return in_interval, shifted_timestamps

    return in_interval

class SplitByEvent(IPlugin):
    def attach_to_controller(self, controller):
        @connect
        def on_gui_ready(sender, gui):
            # @gui.edit_actions.add(shortcut='alt+i')
            @controller.supervisor.actions.add(shortcut="alt+y")
            def VisualizeByEvent():
                """Split all spikes with close to event. THIS IS FOR VISUALIZATION ONLY, it will show you where potential noise
                spikes may be located. Re-merge the clusters again afterwards and cut the cluster with
                another method!"""

                logger.info("Detecting spikes within event start range.")

                # Selected clusters across the cluster view and similarity view.
                cluster_ids = controller.supervisor.selected

                # Get the amplitudes, using the same controller method as what the amplitude view
                # is using.
                # Note that we need load_all=True to load all spikes from the selected clusters,
                # instead of just the selection of them chosen for display.
                bunchs = controller._amplitude_getter(
                    cluster_ids, name="template", load_all=True
                )

                # We get the spike ids and the corresponding spike template amplitudes.
                # NOTE: in this example, we only consider the first selected cluster.
                spike_ids = bunchs[0].spike_ids
                spike_times = controller.model.spike_times[spike_ids]

                # find file with flag "events" in the name in current directory
                filename = [f for f in os.listdir() if "events" in f][0]

                data = sio.loadmat(filename, simplify_cells=True)
                events = data[list(data.keys())[-1]]["timestamps"][:, 0]

                # find spikes within 3 ms of event start
                spikes_within_range = in_intervals(spike_times, np.array([events-0.002,events+0.02]).T)

                labels = np.ones(len(spikes_within_range), "int64")
                labels[spikes_within_range] = 2

                assert spike_ids.shape == labels.shape

                # We split according to the labels.
                controller.supervisor.actions.split(spike_ids, labels)
                logger.info("Splitted spikes by event from main cluster")
