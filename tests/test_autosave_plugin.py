import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock

PLUGIN_DIR = Path(__file__).parents[1] / "plugins"
sys.path.insert(0, str(PLUGIN_DIR))


class _Signal:
    def __init__(self):
        self.callback = None

    def connect(self, callback):
        self.callback = callback


class _FakeTimer:
    instances = []

    def __init__(self, parent):
        self.parent = parent
        self.timeout = _Signal()
        self.single_shot = False
        self.starts = []
        self.__class__.instances.append(self)

    def setSingleShot(self, value):
        self.single_shot = value

    def start(self, delay_ms):
        self.starts.append(delay_ms)

    def fire(self):
        self.timeout.callback()


class AutoSaveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.connections = []

        def connect(function=None, *, sender=None):
            def decorate(callback):
                cls.connections.append((sender, callback))
                return callback

            return decorate(function) if function is not None else decorate

        phy = types.ModuleType("phy")
        phy.IPlugin = type("IPlugin", (), {})
        phy.connect = connect
        qt = types.ModuleType("phy.gui.qt")
        qt.QTimer = _FakeTimer
        sys.modules.update(
            {
                "phy": phy,
                "phy.gui": types.ModuleType("phy.gui"),
                "phy.gui.qt": qt,
            }
        )
        cls.module = importlib.import_module("AutoSavePlugin")

    def setUp(self):
        _FakeTimer.instances.clear()

    def test_repeated_changes_restart_one_single_shot_timer(self):
        supervisor = Mock()
        supervisor.is_dirty.return_value = True
        saver = self.module._DebouncedAutoSaver(supervisor, object(), delay_ms=5000)

        saver.schedule(types.SimpleNamespace(description="metadata_group"))
        saver.schedule(types.SimpleNamespace(description="assign"))

        self.assertTrue(saver.timer.single_shot)
        self.assertEqual(saver.timer.starts, [5000, 5000])
        saver.timer.fire()
        supervisor.save.assert_called_once_with()
        self.assertEqual(saver.pending_descriptions, set())

    def test_clean_event_does_not_schedule_or_save(self):
        supervisor = Mock()
        supervisor.is_dirty.return_value = False
        saver = self.module._DebouncedAutoSaver(supervisor, object())

        saver.schedule(types.SimpleNamespace(description="selection"))
        saver.timer.fire()

        self.assertEqual(saver.timer.starts, [])
        supervisor.save.assert_not_called()

    def test_failed_save_preserves_pending_context(self):
        supervisor = Mock()
        supervisor.is_dirty.return_value = True
        supervisor.save.side_effect = OSError("read-only dataset")
        saver = self.module._DebouncedAutoSaver(supervisor, object())
        saver.schedule(types.SimpleNamespace(description="merge"))

        with self.assertLogs("phy", level="ERROR"):
            saver.timer.fire()

        self.assertEqual(saver.pending_descriptions, {"merge"})

    def test_plugin_registers_controller_and_supervisor_hooks(self):
        self.connections.clear()
        controller = types.SimpleNamespace(supervisor=Mock())
        gui = types.SimpleNamespace()

        self.module.AutoSavePlugin().attach_to_controller(controller)
        gui_ready = next(
            callback
            for sender, callback in self.connections
            if sender is controller
        )
        gui_ready(controller, gui)
        cluster_callback = next(
            callback
            for sender, callback in self.connections
            if sender is controller.supervisor
        )
        controller.supervisor.is_dirty.return_value = True
        cluster_callback(
            controller.supervisor,
            types.SimpleNamespace(description="metadata_neurontype"),
        )

        self.assertIs(gui._phy2_autosaver.timer.parent, gui)
        self.assertEqual(gui._phy2_autosaver.timer.starts, [5000])


if __name__ == "__main__":
    unittest.main()
