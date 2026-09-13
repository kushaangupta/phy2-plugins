import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock

import matplotlib
import numpy as np

matplotlib.use("Agg")

PLUGIN_DIR = Path(__file__).parents[1] / "plugins"
sys.path.insert(0, str(PLUGIN_DIR))


class _FakeCanvas:
    def __init__(self, figure):
        self.ax = figure.subplots()
        self.update = Mock()


class _FakeManualClusteringView:
    def __init__(self):
        from matplotlib.figure import Figure

        self.figure = Figure()
        self.canvas = _FakeCanvas(self.figure)


class AmplitudeHistogramViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        phy = types.ModuleType("phy")
        phy.IPlugin = type("IPlugin", (), {})
        views = types.ModuleType("phy.cluster.views")
        views.ManualClusteringView = _FakeManualClusteringView
        plot = types.ModuleType("phy.plot.plot")
        plot.PlotCanvasMpl = object
        sys.modules.update(
            {
                "phy": phy,
                "phy.cluster": types.ModuleType("phy.cluster"),
                "phy.cluster.views": views,
                "phy.plot": types.ModuleType("phy.plot"),
                "phy.plot.plot": plot,
            }
        )
        cls.module = importlib.import_module("AmplitudeHistogram")

    def test_empty_selection_draws_prompt(self):
        view = self.module.AmplitudeHistogramView(amplitude_getter=Mock())
        view.on_select([])

        view.amplitude_getter.assert_not_called()
        view.canvas.update.assert_called_once()
        self.assertEqual(view.cluster_ids, ())

    def test_selection_draws_metric_and_uses_all_returned_amplitudes(self):
        rng = np.random.default_rng(4)
        bunch = types.SimpleNamespace(amplitudes=rng.normal(10, 1, 5_000))
        getter = Mock(return_value=[bunch])
        view = self.module.AmplitudeHistogramView(amplitude_getter=getter)

        view.on_select([17])

        getter.assert_called_once_with((17,))
        view.canvas.update.assert_called_once()
        labels = view.canvas.ax.get_legend_handles_labels()[1]
        self.assertTrue(any(label.startswith("17:") for label in labels))

    def test_plugin_registers_lazy_view_creator(self):
        controller = types.SimpleNamespace(view_creator={}, _amplitude_getter=Mock())
        plugin = self.module.AmplitudeHistogramPlugin()

        plugin.attach_to_controller(controller)
        view = controller.view_creator["AmplitudeHistogramView"]()
        view.amplitude_getter((3, 8))

        controller._amplitude_getter.assert_called_once_with(
            (3, 8), name="template", load_all=True
        )


if __name__ == "__main__":
    unittest.main()
