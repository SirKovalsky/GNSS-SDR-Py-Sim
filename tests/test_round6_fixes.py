"""Round-6 fixes: TX-power value visibility, wheel guard for numeric inputs,
venv re-exec and live (while-transmitting) TX gain.

Offline, headless, no hardware, no network::

    .venv\\Scripts\\python.exe -m pytest tests -q
"""

from __future__ import annotations

import os
import subprocess
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

from gnss_sim import venv  # noqa: E402
from gnss_sim.config import SimConfig  # noqa: E402
from gnss_sim.iqfile import Sink  # noqa: E402
from gnss_sim.runner import SimulationRunner, _DuplexSink  # noqa: E402
from gnss_sim.uhd_tx import UhdTxSink  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RUN = os.path.join(_ROOT, "run.py")


def _app():
    from PyQt5 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _wheel_event():
    from PyQt5 import QtCore, QtGui
    return QtGui.QWheelEvent(
        QtCore.QPointF(1, 1), QtCore.QPointF(1, 1),
        QtCore.QPoint(0, 0), QtCore.QPoint(0, 120),
        QtCore.Qt.NoButton, QtCore.Qt.NoModifier,
        QtCore.Qt.NoScrollPhase, False)


# ======================================================================
# 1) «Мощность TX» — value visible in the row and updated while dragging
# ======================================================================
def test_tx_power_value_visible_in_row_and_updates() -> None:
    from PyQt5 import QtCore
    from gnss_sim.gui import MainWindow

    app = _app()
    win = MainWindow()
    try:
        win.resize(1920, 1080)
        win.show()
        app.processEvents()
        sl, lb = win.sl_tx_gain, win.lbl_tx_gain
        sl_x = sl.mapTo(win, QtCore.QPoint(0, 0)).x()
        lb_x = lb.mapTo(win, QtCore.QPoint(0, 0)).x()
        # label follows the slider in the same row, immediately next to it
        assert sl_x < lb_x
        assert (lb_x - (sl_x + sl.width())) < 80
        # ...and lies inside the visible settings viewport (not scrolled out)
        basic = win.left_tabs.widget(0)
        vp_x = basic.mapTo(win, QtCore.QPoint(0, 0)).x()
        assert vp_x <= lb_x <= vp_x + basic.viewport().width()
        # value tracks dragging
        win.sl_tx_gain.setValue(int(round(33.0 * 4)))
        assert "33" in lb.text()
    finally:
        win.close()
    del app


def test_tx_power_note_label_wraps() -> None:
    """The long note must wrap, else it forces ~2074 px wide settings tab."""
    from PyQt5 import QtWidgets
    from gnss_sim.gui import MainWindow

    app = _app()
    win = MainWindow()
    try:
        notes = [lbl for lbl in win.findChildren(QtWidgets.QLabel)
                 if lbl.text().startswith("«Старт»")]
        assert notes and all(lbl.wordWrap() for lbl in notes)
    finally:
        win.close()
    del app


# ======================================================================
# 2) Mouse wheel never changes a numeric input unless it is focused
# ======================================================================
def test_all_numeric_inputs_are_guarded_and_ignore_wheel() -> None:
    from PyQt5 import QtWidgets
    from gnss_sim.gui import MainWindow

    app = _app()
    win = MainWindow()
    try:
        spins = win.findChildren(QtWidgets.QAbstractSpinBox)
        assert len(spins) >= 20
        # the named inputs from the bug report are all covered
        for attr in ("sp_el", "sp_headroom", "sp_amp", "sp_dur", "sp_rx_gain",
                     "sp_ch", "sp_rx_ch", "sp_bw"):
            assert hasattr(getattr(win, attr), "_wheel_guard"), attr
        event = _wheel_event()
        for spin in spins:
            assert hasattr(spin, "_wheel_guard"), type(spin).__name__
            spin.clearFocus()
            before = spin.value() if hasattr(spin, "value") else None
            assert spin._wheel_guard.eventFilter(spin, event) is True
            if before is not None:
                assert spin.value() == before
    finally:
        win.close()
    del app


def test_guarded_spinbox_allows_wheel_when_focused() -> None:
    from PyQt5 import QtWidgets
    from gnss_sim.gui import MainWindow

    app = _app()
    win = MainWindow()
    try:
        win.show()
        win.activateWindow()
        spin = win.sp_el
        spin.setFocus()
        app.processEvents()
        if not spin.hasFocus():  # pragma: no cover - platform dependent
            pytest.skip("offscreen focus not available")
        assert spin._wheel_guard.eventFilter(spin, _wheel_event()) is False
        # a non focused spin box is swallowed
        other = win.sp_amp
        other.clearFocus()
        assert other._wheel_guard.eventFilter(other, _wheel_event()) is True
    finally:
        win.close()
    del app


# ======================================================================
# 3) run.py re-exec into the project-local venv
# ======================================================================
def _foreign_python() -> str | None:
    """An interpreter that is *not* the project venv (base install), if any."""
    base = sys.base_prefix
    if os.path.normcase(base) == os.path.normcase(sys.prefix):
        return None
    names = ("python.exe", "python",
             os.path.join("bin", "python3"), os.path.join("bin", "python"))
    venv_py = venv.find_venv_python(_ROOT)
    for name in names:
        cand = os.path.join(base, name)
        if (os.path.isfile(cand)
                and os.path.normcase(cand) != os.path.normcase(venv_py)):
            return cand
    return None


def _norm(path: str) -> str:
    # Not realpath: the WindowsApps ``python.exe`` shim redirects to the Store
    # install, but the child reports the shim path it was started with.
    return os.path.normcase(os.path.abspath(path))


def test_venv_helpers_cover_prefix_and_candidates() -> None:
    root = venv.project_root()
    assert venv.venv_dir(root).endswith(".venv")
    assert os.path.isfile(venv.find_venv_python(root))
    assert venv.needs_reexec(exe=venv.find_venv_python(root)) is False
    # Running the test suite under the venv: prefix detection says "there".
    if venv.same_interpreter():
        assert venv._prefix_is_venv(root) is True  # noqa: SLF001


def test_run_py_prints_venv_interpreter_under_venv() -> None:
    out = subprocess.check_output(
        [sys.executable, _RUN, "--print-interpreter"],
        cwd=_ROOT, text=True, timeout=180)
    assert _norm(out.strip()) == _norm(venv.find_venv_python(_ROOT))


def test_run_py_reexecs_from_foreign_interpreter() -> None:
    """The system interpreter re-execs into .venv (the old numpy crash)."""
    foreign = _foreign_python()
    if foreign is None:
        pytest.skip("no second interpreter available")
    proc = subprocess.run(
        [foreign, _RUN, "--print-interpreter"],
        cwd=_ROOT, text=True, capture_output=True, timeout=180)
    assert proc.returncode == 0, proc.stderr
    used = proc.stdout.strip().splitlines()[-1]
    assert _norm(used) == _norm(venv.find_venv_python(_ROOT))


def test_venv_reexec_module_argv_and_lazy_package_attr() -> None:
    import gnss_sim

    # lazy package attribute still resolves to the constants module
    assert gnss_sim.constants.R2D > 0
    py = venv.find_venv_python(_ROOT)
    assert venv.reexec_module_argv("gnss_sim", ["--help"], exe=py) is None
    assert venv.maybe_reexec_module("gnss_sim", exe=py) is None
    cmd = venv.reexec_module_argv("gnss_sim", ["--help"],
                                  exe="C:/other/python.exe")
    assert cmd is not None and cmd[:3] == [py, "-m", "gnss_sim"]
    assert cmd[3] == "--help"


def test_python_m_gnss_sim_works_under_venv() -> None:
    proc = subprocess.run([sys.executable, "-m", "gnss_sim", "--help"],
                          cwd=_ROOT, text=True, capture_output=True, timeout=180)
    assert proc.returncode == 0, proc.stderr
    assert "usage: gnss_sim" in proc.stdout


def test_python_m_gnss_sim_reexecs_from_foreign_interpreter() -> None:
    foreign = _foreign_python()
    if foreign is None:
        pytest.skip("no second interpreter available")
    proc = subprocess.run([foreign, "-m", "gnss_sim", "--help"],
                          cwd=_ROOT, text=True, capture_output=True, timeout=180)
    assert proc.returncode == 0, proc.stderr
    assert "usage: gnss_sim" in proc.stdout


def test_reexec_loop_guard_keeps_foreign_interpreter() -> None:
    foreign = _foreign_python()
    if foreign is None:
        pytest.skip("no second interpreter available")
    env = dict(os.environ)
    env[venv.REEXEC_ENV] = "1"
    proc = subprocess.run(
        [foreign, _RUN, "--print-interpreter"],
        cwd=_ROOT, text=True, capture_output=True, env=env, timeout=180)
    assert proc.returncode == 0, proc.stderr
    used = proc.stdout.strip().splitlines()[-1]
    # With the guard set the process stays on the interpreter it started with
    # (no re-exec loop into the venv).
    assert _norm(used) != _norm(venv.find_venv_python(_ROOT))


# ======================================================================
# 4) Dynamic TX power: live sink forwarding + clamping + safety
# ======================================================================
class _GainSink(Sink):
    """Fake TX sink that clamps to ``(lo, hi)`` and records requests."""

    def __init__(self, lo: float = 0.0, hi: float = 30.0) -> None:
        self.lo, self.hi = lo, hi
        self.gain = 10.0
        self.requests: list[float] = []

    def write(self, samples) -> None:  # pragma: no cover - not used
        pass

    def set_tx_gain(self, gain: float) -> float:
        self.requests.append(float(gain))
        self.gain = max(self.lo, min(self.hi, float(gain)))
        return self.gain


def test_runner_set_tx_gain_reaches_live_sink_and_clamps() -> None:
    cfg = SimConfig(use_usrp=True, fs=1.0e6)
    logs: list[str] = []
    runner = SimulationRunner(cfg, log=logs.append)
    sink = _GainSink(0.0, 30.0)
    runner.sink = sink

    assert runner.set_tx_gain(20.0) == 20.0
    assert sink.requests == [20.0]
    # Above the device range: applied/clamped value is returned and stored.
    assert runner.set_tx_gain(99.0) == 30.0
    assert cfg.tx_gain == 30.0
    assert any("на лету" in line for line in logs)


def test_runner_set_tx_gain_without_transmitting_is_safe() -> None:
    cfg = SimConfig(use_usrp=False)
    runner = SimulationRunner(cfg)
    assert runner.set_tx_gain(12.5) is None  # no sink, no crash
    assert cfg.tx_gain == 12.5


def test_duplex_sink_forwards_set_tx_gain(monkeypatch) -> None:
    import gnss_sim.uhd_duplex as duplex_mod

    class FakeDuplex:
        def __init__(self, **kwargs) -> None:
            self.tx_gain = 0.0
            self.set_seen: list[float] = []

        def set_tx_gain(self, gain: float) -> float:
            self.set_seen.append(float(gain))
            self.tx_gain = float(gain)
            return self.tx_gain

        def close(self) -> None:
            pass

    monkeypatch.setattr(duplex_mod, "UhdDuplex", FakeDuplex)
    cfg = SimConfig(use_usrp=True, monitor=True, tx_power_auto=False,
                    fs=1.0e6)
    sink = _DuplexSink(cfg, log=lambda _m: None,
                       spectrum=lambda _l, _s: None)
    assert sink.set_tx_gain(17.0) == 17.0
    assert sink._duplex.set_seen == [17.0]  # noqa: SLF001
    sink.close()


def test_uhd_tx_sink_set_tx_gain_clamps_and_reads_back() -> None:
    import threading

    sink = object.__new__(UhdTxSink)
    sink._lock = threading.RLock()
    sink._gain_range = (0.0, 30.0)
    sink.gain_warning = None
    sink.gain = 10.0
    sink.channel = 0
    sink._log = lambda _m: None

    class FakeUsrp:
        def __init__(self) -> None:
            self.g = 0.0

        def set_tx_gain(self, value, channel) -> None:
            self.g = float(value)

        def get_tx_gain(self, channel) -> float:
            return min(self.g, 30.0)

    sink.usrp = FakeUsrp()
    assert sink.set_tx_gain(20.0) == 20.0
    assert sink.set_tx_gain(99.0) == 30.0  # clamped to the device range
    assert sink.gain == 30.0
    sink.usrp = None  # closed device: only records the request
    assert sink.set_tx_gain(5.0) == 5.0


def test_gui_slider_applies_gain_to_active_runner() -> None:
    from gnss_sim.gui import MainWindow

    app = _app()
    win = MainWindow()
    try:
        calls: list[float] = []
        logs: list[str] = []

        class FakeRunner:
            def is_running(self) -> bool:
                return True

            def set_tx_gain(self, db: float) -> float:
                calls.append(float(db))
                return float(db)

        win.runner = FakeRunner()  # type: ignore[assignment]
        win._append_log = logs.append  # type: ignore[method-assign]
        win._tx_gain_log_ts = 0.0
        win.sl_tx_gain.setValue(int(round(17.0 * 4)))
        assert calls and calls[-1] == 17.0
        assert any("применено" in line for line in logs)
    finally:
        win.close()
    del app


def test_gui_slider_reports_clamped_value() -> None:
    from gnss_sim.gui import MainWindow

    app = _app()
    win = MainWindow()
    try:
        logs: list[str] = []

        class ClampingRunner:
            def is_running(self) -> bool:
                return True

            def set_tx_gain(self, db: float) -> float:
                return 30.0

        win.runner = ClampingRunner()  # type: ignore[assignment]
        win._append_log = logs.append  # type: ignore[method-assign]
        win._tx_gain_log_ts = 0.0
        win.sl_tx_gain.setValue(int(round(50.0 * 4)))
        assert any("запрошено" in line and "применено" in line for line in logs)
    finally:
        win.close()
    del app
