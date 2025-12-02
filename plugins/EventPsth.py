"""Show how to write a custom split action."""

from phy import IPlugin, connect
import numpy as np
import logging
import os
import scipy.io as sio

logger = logging.getLogger("phy")

# import os
# import numpy as np
import pandas as pd

# import scipy.io as sio
import matplotlib.pyplot as plt


def crossCorr(
    t1: np.ndarray,
    t2: np.ndarray,
    binsize: float,
    nbins: int,
) -> np.ndarray:
    """
    Performs the discrete cross-correlogram of two time series.
    The units should be in s for all arguments.
    Return the firing rate of the series t2 relative to the timings of t1.

    crossCorr functions from Guillaume Viejo of Peyrache Lab
    https://github.com/PeyracheLab/StarterPack/blob/master/python/main6_autocorr.py
    https://github.com/pynapple-org/pynapple/blob/main/pynapple/process/correlograms.py

    Parameters
    ----------
    t1 : array
        First time series.
    t2 : array
        Second time series.
    binsize : float
        Size of the bin in seconds.
    nbins : int
        Number of bins.

    Returns
    -------
    C : array
        Cross-correlogram of the two time series.

    """
    # Calculate the length of the input time series
    nt1 = len(t1)
    nt2 = len(t2)

    # Ensure that 'nbins' is an odd number
    if np.floor(nbins / 2) * 2 == nbins:
        nbins = nbins + 1

    # Calculate the half-width of the cross-correlogram window
    w = (nbins / 2) * binsize
    C = np.zeros(nbins)
    i2 = 1

    # Iterate through the first time series
    for i1 in range(nt1):
        lbound = t1[i1] - w

        # Find the index of the first element in 't2' that is within 'lbound'
        while i2 < nt2 and t2[i2] < lbound:
            i2 = i2 + 1

        # Find the index of the last element in 't2' that is within 'lbound'
        while i2 > 1 and t2[i2 - 1] > lbound:
            i2 = i2 - 1

        rbound = lbound
        l = i2

        # Calculate the cross-correlogram values for each bin
        for j in range(nbins):
            k = 0
            rbound = rbound + binsize

            # Count the number of elements in 't2' that fall within the bin
            while l < nt2 and t2[l] < rbound:
                l = l + 1
                k = k + 1

            C[j] += k

    # Normalize the cross-correlogram by dividing by the total observation time and bin size
    C = C / (nt1 * binsize)

    return C


def compute_psth(
    spikes: np.ndarray,
    event: np.ndarray,
    bin_width: float = 0.002,
    n_bins: int = 100,
    window: list = None,
):
    if window is not None:
        times = np.arange(window[0], window[1] + bin_width / 2, bin_width)
        n_bins = len(times) - 1
    else:
        times = np.linspace(
            -(n_bins * bin_width) / 2, (n_bins * bin_width) / 2, n_bins + 1
        )

    ccg = pd.DataFrame(index=times, columns=np.arange(len(spikes)))
    # Now we can iterate over spikes
    for i, s in enumerate(spikes):
        ccg[i] = crossCorr(event, s, bin_width, n_bins)
    return ccg


class EventPsth(IPlugin):
    def attach_to_controller(self, controller):
        @connect
        def on_gui_ready(sender, gui):
            # @gui.edit_actions.add(shortcut='alt+i')
            @controller.supervisor.actions.add(shortcut="alt+p")
            def event_psth():
                """Split all spikes with close to event. THIS IS FOR VISUALIZATION ONLY, it will show you where potential noise
                spikes may be located. Re-merge the clusters again afterwards and cut the cluster with
                another method!"""

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
                logger.info("event psth from %s", list(data.keys())[-1])
                events = data[list(data.keys())[-1]]["timestamps"]

                stim_psth = compute_psth(
                    [spike_times], events[:, 0], bin_width=0.005, window=[-1, 1]
                )
                stim_psth.plot()
                plt.axvline(0, color="k", linestyle="--")
                plt.show()
