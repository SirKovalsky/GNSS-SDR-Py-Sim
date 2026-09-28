"""RAM-only playback/TX, unified IQ control, pause/play, venv re-exec.

Offline, headless, no hardware, no network::

    .venv\\Scripts\\python.exe -m pytest tests -q
"""

from __future__ import annotations

import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from gnss_sim import venv  # noqa: E402
from gnss_sim.config import SimConfig  # noqa: E402
from gnss_sim.iqfile import FileSink, NullSink, Sink  # noqa: E402
from gnss_sim.runner import SimulationRunner  # noqa: E402


def _app():
    from PyQt5 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class _FakeEngine:
    def __init__(self) -> None:
        self.calls = 0

    def generate_block(self, n: int) -> np.ndarray:
        self.calls += 1
        return np.full(int(n), 0.1 + 0.1j, dtype=np.complex128)


class _Recorder(Sink):
    def __init__(self) -> None:
        self.n = 0

    def write(self, samples) -> None:
        self.n += int(np.asarray(samples).size)


# ======================================================================
# 1) RAM-only TX: block the radio when the buffer exceeds free RAM
# ======================================================================
def _small_ram(monkeypatch, avail=4 * 10**9):
    import gnss_sim.runner as R
    monkeypatch.setattr(R.sysinfo, "available_ram", lambda: avail)
    monkeypatch.setattr(R.sysinfo, "total_ram", lambda: avail + 1)
    monkeypatch.setattr(R.sysinfo, "disk_free", lambda _p: 10**12)


def test_plan_blocks_nonloop_tx_when_duration_exceeds_ram(monkeypatch, tmp_path) -> None:
    _small_ram(monkeypatch)
    logs: list[str] = []
    cfg = SimConfig(use_usrp=True, fs=2.6e6, duration=1200.0, loop=False,
                    output=str(tmp_path / "out.cs16"))
    runner = SimulationRunner(cfg, log=logs.append)
    runner._plan()
    assert runner._tx_blocked is True
    sink = runner._make_sink()
    assert isinstance(sink, FileSink)      # file generation still allowed
    assert cfg.use_usrp is False           # radio blocked
    text = "\n".join(logs)
    assert "ЗАБЛОКИРОВАНА" in text and "только RAM" in text
    # Without an output file there is nothing to do: abort clearly.
    cfg2 = SimConfig(use_usrp=True, fs=2.6e6, duration=1200.0, loop=False)
    r2 = SimulationRunner(cfg2, log=lambda _m: None)
    r2._plan()
    with pytest.raises(ValueError) as exc:
        r2._make_sink()
    assert "заблокирована" in str(exc.value).lower()


def test_plan_blocks_explicit_oversize_loop_segment(monkeypatch) -> None:
    _small_ram(monkeypatch)
    cfg = SimConfig(use_usrp=True, fs=2.6e6, duration=0.0, loop=True,
                    loop_seconds=100000.0)
    runner = SimulationRunner(cfg, log=lambda _m: None)
    runner._plan()
    assert runner._tx_blocked is True


def test_plan_allows_tx_when_segment_fits(monkeypatch) -> None:
    _small_ram(monkeypatch, avail=60 * 10**9)
    cfg = SimConfig(use_usrp=True, fs=2.6e6, duration=120.0, loop=True)
    runner = SimulationRunner(cfg, log=lambda _m: None)
    runner._plan()
    assert runner._tx_blocked is False


# ======================================================================
# 2) Pause / resume of a live run
# ======================================================================
def test_runner_pause_blocks_then_resume_completes() -> None:
    cfg = SimConfig(fs=1.0e6, duration=0.02, loop=False, block_ms=5.0)
    runner = SimulationRunner(cfg, log=lambda _m: None)
    runner.engine = _FakeEngine()  # type: ignore[assignment]
    runner.sink = _Recorder()
    runner._segment_seconds = None
    runner.pause()
    thread = threading.Thread(target=runner._run_engine, args=(0.0,))
    thread.start()
    time.sleep(0.2)
    assert runner.is_paused() is True
    assert runner.sink.n == 0 and thread.is_alive()   # no samples while paused
    runner.resume()
    thread.join(timeout=3.0)
    assert runner.sink.n == 20_000
    assert runner.is_paused() is False


def test_runner_stop_while_paused_returns() -> None:
    cfg = SimConfig(fs=1.0e6, duration=0.0, loop=True, block_ms=5.0)
    runner = SimulationRunner(cfg, log=lambda _m: None)
    runner.engine = _FakeEngine()  # type: ignore[assignment]
    runner.sink = NullSink()
    runner._segment_seconds = 1.0
    runner.pause()
    thread = threading.Thread(target=runner.run)
    thread.start()
    time.sleep(0.15)
    runner.stop()          # must unblock the paused wait
    thread.join(timeout=3.0)
    assert not thread.is_alive()


# ======================================================================
# 3) Unified IQ control + internal output scale in the GUI
# ======================================================================
def test_gui_output_controls_unified() -> None:
    from gnss_sim.gui import MainWindow, _DEFAULT_OUTPUT_SCALE
    app = _app()
    win = MainWindow()
    try:
        # «Масштаб в int», «Длина сегмента» and «Бюджет RAM» are gone.
        for gone in ("sp_scale", "sp_loop", "sp_mem", "cb_iq_ram"):
            assert not hasattr(win, gone), gone
        # Internal default keeps file output working.
        assert win._collect().output_scale == _DEFAULT_OUTPUT_SCALE
        # ONE ready-IQ control: the chooser always means "load to RAM".
        assert win.cb_iq_in.text().startswith("Использовать готовый IQ")
        assert win.ed_iq_in.isEnabled() is False
        win.cb_iq_in.setChecked(True)
        assert win.ed_iq_in.isEnabled() is True
        win.ed_iq_in.setText("reuse.cs16")
        cfg = win._collect()
        assert cfg.iq_input == "reuse.cs16" and cfg.iq_in_ram is True
        assert cfg.loop_seconds == 0.0 and cfg.memory_budget_gb == 0.0
    finally:
        win.close()
    del app


def test_gui_has_pause_and_play_buttons() -> None:
    from gnss_sim.gui import MainWindow
    app = _app()
    win = MainWindow()
    try:
        assert win.btn_pause.text().endswith("Пауза")
        assert "Воспроизвести" in win.btn_replay.text()
        assert hasattr(win, "lbl_visible")   # dedicated satellite field
    finally:
        win.close()
    del app


def _write_iq(path: str, n: int = 512) -> None:
    sink = FileSink(path, fmt="cs16", fs=1.0e6, center_freq=1575.42e6,
                    scale=10000.0)
    sink.start()
    sink.write(np.full(n, 0.1 + 0.1j, dtype=np.complex128))
    sink.close()


def test_gui_start_autostarts_tx_when_b210(tmp_path, monkeypatch) -> None:
    from gnss_sim import gui
    app = _app()
    win = gui.MainWindow()
    path = str(tmp_path / "reuse.cs16")
    _write_iq(path)
    seen: dict = {}

    class FakeRunner:
        def __init__(self, cfg, **kw):
            seen["use_usrp"] = cfg.use_usrp
            seen["iq_input"] = cfg.iq_input

        def start(self):
            seen["started"] = True

        def is_running(self):
            return False

        def stop(self):
            pass

    monkeypatch.setattr(gui, "SimulationRunner", FakeRunner)
    try:
        # (D) No B210 -> file-only, with an explicit note in the journal.
        win.cb_iq_in.setChecked(True)
        win.ed_iq_in.setText(path)
        win._start()
        assert seen.get("started") is True
        assert seen["use_usrp"] is False
        assert seen["iq_input"] == path
        assert "B210 не обнаружен" in win.log.toPlainText()
        # (D) B210 present -> «Старт» auto-transmits (no checkbox exists).
        assert not hasattr(win, "cb_tx")
        seen.clear()
        win._b210_override = True
        win._start()
        assert seen.get("started") is True
        assert seen["use_usrp"] is True     # Start auto-starts transmission
        assert seen["iq_input"] == path
    finally:
        win.close()
    del app


def test_gui_playback_controls_grouped() -> None:
    """(A) loop checkbox + ready-IQ chooser share the playback group."""
    from PyQt5 import QtWidgets
    from gnss_sim.gui import MainWindow
    app = _app()
    win = MainWindow()
    try:
        group = win.btn_replay.parent()
        assert isinstance(group, QtWidgets.QGroupBox)
        for w in (win.btn_start, win.btn_pause, win.btn_stop, win.btn_replay,
                  win.cb_loop, win.cb_iq_in, win.ed_iq_in):
            assert group.isAncestorOf(w), w
        assert win.cb_loop.text().startswith("Зациклить сегмент")
    finally:
        win.close()
    del app


def test_gui_tx_power_slider_on_basic_tab_b210_range() -> None:
    """(C) TX power is a slider on «Базовые», 0..89.75 dB, default +10."""
    from gnss_sim.gui import MainWindow, _B210_TX_GAIN_MAX
    app = _app()
    win = MainWindow()
    try:
        basic = win.left_tabs.widget(0)
        assert basic.isAncestorOf(win.sl_tx_gain)
        assert not hasattr(win, "sp_txg")
        assert win._collect().tx_gain == 10.0
        assert "10" in win.lbl_tx_gain.text()
        win.sl_tx_gain.setValue(win.sl_tx_gain.maximum())
        assert win._tx_gain_db() == pytest.approx(_B210_TX_GAIN_MAX)
        assert win._collect().tx_gain <= _B210_TX_GAIN_MAX
        win.sl_tx_gain.setValue(win.sl_tx_gain.minimum())
        assert win._tx_gain_db() == 0.0
    finally:
        win.close()
    del app


def test_gui_b210_status_indicator() -> None:
    """(D) clear B210 status/indicator, no silent no-op."""
    from gnss_sim.gui import MainWindow
    app = _app()
    win = MainWindow()
    try:
        win._b210_override = False
        win._detect_b210()
        assert "не обнаружен" in win.lbl_b210.text()
        win._b210_override = True
        win._detect_b210()
        assert "обнаружен" in win.lbl_b210.text()
    finally:
        win.close()
    del app


def test_iq_source_ram_native_dtype_and_footprint(tmp_path, monkeypatch) -> None:
    """(B) RAM buffer is the native dtype (~file size), not 4x complex128."""
    import gnss_sim.sysinfo as sysinfo
    from gnss_sim.iqfile import FORMAT_BYTES
    from gnss_sim.iqfile import IqFileSource
    path = str(tmp_path / "big.cs16")
    n = 100_000
    _write_iq(path, n=n)
    file_bytes = n * FORMAT_BYTES["cs16"]
    # Free RAM is only 1.5x the file: a native load fits, a complex128 decode
    # (16 B/sample = 4x) would raise.
    monkeypatch.setattr(sysinfo, "available_ram", lambda: int(file_bytes * 1.5))
    src = IqFileSource(path, load_to_ram=True)
    src.start()
    try:
        assert src.in_ram is True
        assert src._ram.dtype == np.dtype("<i2")      # native int16
        assert src._ram.nbytes <= file_bytes * 1.1    # ~ file size
        block = src.read(1000)
        assert block.dtype == np.complex64 and block.size == 1000
    finally:
        src.close()


def test_run_source_progress_reflects_playback_position(tmp_path) -> None:
    """(B) the progress bar follows the playback position in the file."""
    from gnss_sim.config import SimConfig
    from gnss_sim.iqfile import IqFileSource, Sink
    path = str(tmp_path / "sig.cs16")
    _write_iq(path, n=1000)
    fracs: list[float] = []
    cfg = SimConfig(iq_input=path, fs=1000.0, duration=0.0, loop=True,
                    block_ms=200.0, use_usrp=True)
    runner = SimulationRunner(
        cfg, log=lambda _m: None,
        progress=lambda frac, _s, _w, _r: fracs.append(float(frac)))
    runner._source = IqFileSource(path, fmt="cs16", fs=1000.0)
    runner._source.start()
    runner._loop = True

    class _StopAfter(Sink):
        def __init__(self) -> None:
            self.n = 0

        def write(self, samples) -> None:
            self.n += int(np.asarray(samples).size)
            if self.n >= 2500:
                runner.stop()

    runner.sink = _StopAfter()
    produced = runner._run_source()
    assert produced >= 2500
    assert max(fracs) > 0.0
    assert any(0.0 < f < 1.0 for f in fracs)   # not stuck at 0.0


def test_run_source_honors_loop_checkbox(tmp_path) -> None:
    """(B) «Воспроизвести» loops only when the loop checkbox is set."""
    from gnss_sim.config import SimConfig
    from gnss_sim.iqfile import IqFileSource, Sink
    path = str(tmp_path / "sig.cs16")
    _write_iq(path, n=300)

    class _Rec(Sink):
        def __init__(self) -> None:
            self.n = 0

        def write(self, samples) -> None:
            self.n += int(np.asarray(samples).size)

    cfg = SimConfig(iq_input=path, fs=1000.0, duration=0.0, loop=False,
                    block_ms=100.0, use_usrp=True)
    runner = SimulationRunner(cfg, log=lambda _m: None)
    runner._source = IqFileSource(path, fmt="cs16", fs=1000.0)
    runner._source.start()
    runner._plan_source()
    assert runner._loop is False
    runner.sink = _Rec()
    assert runner._run_source() == 300       # played once, not looped
    assert runner.sink.n == 300


def test_gui_play_uses_last_generated_or_chooser(tmp_path, monkeypatch) -> None:
    from gnss_sim import gui
    app = _app()
    win = gui.MainWindow()
    path = str(tmp_path / "generated.cs16")
    _write_iq(path)
    seen: dict = {}

    class FakeRunner:
        def __init__(self, cfg, **kw):
            seen["iq_input"] = cfg.iq_input

        def start(self):
            seen["started"] = True

        def is_running(self):
            return False

        def stop(self):
            pass

    monkeypatch.setattr(gui, "SimulationRunner", FakeRunner)
    try:
        # No chooser value -> the LAST generated file is replayed.
        win.ed_iq_in.setText("")
        win.cb_iq_in.setChecked(False)
        win._last_iq_file = path
        win._play_iq()
        assert seen.get("started") is True and seen["iq_input"] == path
        assert win.cb_iq_in.isChecked() is True
        # A value typed in the chooser wins.
        other = str(tmp_path / "chooser.cs16")
        _write_iq(other)
        win.ed_iq_in.setText(other)
        win._play_iq()
        assert seen["iq_input"] == other
    finally:
        win.close()
    del app


def test_gui_pause_toggles_runner(monkeypatch) -> None:
    from gnss_sim import gui
    app = _app()
    win = gui.MainWindow()

    class FakeRunner:
        def __init__(self):
            self.paused = False

        def toggle_pause(self):
            self.paused = not self.paused
            return self.paused

    win.runner = FakeRunner()  # type: ignore[assignment]
    try:
        win._pause()
        assert win.runner.paused is True and "Продолжить" in win.btn_pause.text()
        win._pause()
        assert win.runner.paused is False and win.btn_pause.text().endswith("Пауза")
    finally:
        win.close()
    del app


def test_gui_stop_works_from_pause() -> None:
    """(E) «Стоп» after «Пауза» clears the pause and stops the run."""
    from gnss_sim import gui
    app = _app()
    win = gui.MainWindow()

    class FakeRunner:
        def __init__(self):
            self.paused = False
            self.stopped = False

        def toggle_pause(self):
            self.paused = not self.paused
            return self.paused

        def resume(self):
            self.paused = False

        def stop(self):
            self.stopped = True
            self.paused = False

    win.runner = FakeRunner()  # type: ignore[assignment]
    try:
        win._pause()
        assert win.runner.paused is True
        assert "Продолжить" in win.btn_pause.text()
        win._stop()
        assert win.runner.stopped is True
        assert win.runner.paused is False
        assert win.btn_pause.text().endswith("Пауза")
    finally:
        win.close()
    del app


# ======================================================================
# 4) venv re-exec helpers
# ======================================================================
def test_venv_paths_and_reexec_logic() -> None:
    root = venv.project_root()
    py = venv.venv_python(root)
    if os.name == "nt":
        assert py.endswith(os.path.join(".venv", "Scripts", "python.exe"))
    else:
        assert py.endswith(os.path.join(".venv", "bin", "python"))
    assert venv.same_interpreter(py, py) is True
    assert venv.same_interpreter("C:/other/python.exe", py) is False


def test_venv_needs_reexec_and_loop_guard(monkeypatch) -> None:
    if not venv.venv_available():
        pytest.skip("project-local .venv not present")
    py = venv.venv_python()
    # Already the venv interpreter: no re-exec.
    assert venv.needs_reexec(exe=py) is False
    assert venv.reexec_argv(["x"], exe=py) is None
    # A different interpreter: re-exec with the venv python.
    argv = venv.reexec_argv(["--flag"], exe="C:/other/python.exe")
    assert argv is not None and argv[0] == py
    # The loop guard disables it even for a foreign interpreter.
    assert venv.needs_reexec(exe="C:/other/python.exe",
                             env={venv.REEXEC_ENV: "1"}) is False
