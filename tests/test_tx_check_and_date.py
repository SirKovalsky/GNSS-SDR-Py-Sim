"""Offline tests for the high-fs TX guard, «контроль передачи» and read-only date.

No hardware, no network::

    E:\\MySoftware\\SDR_Scan\\.venv\\Scripts\\python.exe -m pytest tests -q
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from gnss_sim.config import SimConfig  # noqa: E402
from gnss_sim.runner import SimulationRunner, _DuplexSink  # noqa: E402


def _runner(fs: float, use_usrp: bool = True):
    cfg = SimConfig(fs=fs, use_usrp=use_usrp)
    logs: list[str] = []
    return SimulationRunner(cfg, log=logs.append), logs


# ======================================================================
# C3) High-fs TX guard
# ======================================================================
def test_high_fs_tx_warns_above_threshold() -> None:
    runner, logs = _runner(30.0e6)
    runner._warn_high_fs_tx()
    text = "\n".join(logs)
    assert "ВНИМАНИЕ" in text
    assert "underflow" in text and "LIBUSB_TRANSFER_NO_DEVICE" in text


def test_high_fs_tx_silent_at_low_fs_and_without_usrp() -> None:
    runner, logs = _runner(2.6e6)
    runner._warn_high_fs_tx()
    assert logs == []
    # The computed combined band (~25 Msps) is below the 30 Msps failure point.
    combined, clog = _runner(25.0e6)
    combined._warn_high_fs_tx()
    assert clog == []
    runner2, logs2 = _runner(30.0e6, use_usrp=False)
    runner2._warn_high_fs_tx()
    assert logs2 == []


# ======================================================================
# D) «Контроль передачи»: config / CLI / duplex sink logic
# ======================================================================
def test_tx_check_defaults_and_cli() -> None:
    from gnss_sim.cli import build_parser

    cfg = SimConfig()
    assert cfg.tx_check is False
    assert cfg.tx_check_margin_db == 6.0
    args = build_parser().parse_args(["--tx-check", "--tx-check-margin", "9"])
    assert args.tx_check is True and args.tx_check_margin == 9.0
    assert build_parser().parse_args([]).tx_check is False


def _sink(tx_check: bool, noise, rx, logs) -> _DuplexSink:
    sink = _DuplexSink.__new__(_DuplexSink)
    sink._cfg = SimConfig(tx_check=tx_check, tx_check_margin_db=6.0)
    sink._log = logs.append
    sink._noise_dbfs = noise
    sink._latest_rx = rx
    sink._tx_check = tx_check
    sink._tx_check_margin = 6.0
    return sink


def _tone(dbfs: float, n: int = 4096) -> np.ndarray:
    amp = 10.0 ** (dbfs / 20.0)
    return (amp * np.exp(2j * np.pi * 0.1 * np.arange(n))).astype(np.complex64)


def test_tx_check_detects_signal_above_noise() -> None:
    logs: list[str] = []
    _sink(True, -80.0, _tone(0.0), logs)._run_tx_check()
    assert any("Контроль передачи: сигнал обнаружен" in m for m in logs)
    assert not any("ВНИМАНИЕ" in m for m in logs)


def test_tx_check_warns_when_no_signal() -> None:
    logs: list[str] = []
    _sink(True, -80.0, _tone(-80.0), logs)._run_tx_check()
    assert any("ВНИМАНИЕ: передача не обнаружена" in m for m in logs)


def test_tx_check_skips_when_rx_unavailable() -> None:
    logs: list[str] = []
    zeros = np.zeros(4096, dtype=np.complex64)
    _sink(True, -80.0, zeros, logs)._run_tx_check()
    assert any("проверка пропущена" in m for m in logs)
    assert not any("ВНИМАНИЕ" in m for m in logs)


def test_tx_check_skips_when_noise_unknown() -> None:
    logs: list[str] = []
    _sink(True, None, _tone(0.0), logs)._run_tx_check()
    assert any("шумовой пол" in m and "пропущена" in m for m in logs)


def test_tx_check_disabled_is_noop() -> None:
    logs: list[str] = []
    _sink(False, -80.0, _tone(0.0), logs)._run_tx_check()
    assert logs == []


# ======================================================================
# E) «Стоп» is honoured within one block (pre-gen and streaming)
# ======================================================================
def test_generate_to_ram_stops_on_stop_event() -> None:
    runner, _logs = _runner(2.6e6)

    class _Engine:
        def generate_block(self, n: int) -> np.ndarray:
            runner.stop()  # user pressed «Стоп» mid pre-generation
            return np.zeros(int(n), dtype=np.complex128)

    runner.engine = _Engine()
    blocks, got = runner._generate_to_ram(10, 4)
    assert got == 4 and len(blocks) == 1


def test_generate_to_ram_emits_progress() -> None:
    """The GUI/CLI progress bar must advance during TX pre-generation."""
    cfg = SimConfig(fs=1.0e6)
    logs: list[str] = []
    seen: list[tuple] = []
    runner = SimulationRunner(
        cfg, log=logs.append,
        progress=lambda frac, sim_s, wall, rate: seen.append(
            (frac, sim_s, wall, rate)))

    class _Engine:
        def generate_block(self, n: int) -> np.ndarray:
            return np.zeros(int(n), dtype=np.complex128)

    runner.engine = _Engine()
    blocks, got = runner._generate_to_ram(8, 4)
    assert got == 8 and len(blocks) == 2
    # One progress update per generated block, ending at 100 %.
    assert len(seen) == 2
    assert seen[-1][0] == pytest.approx(1.0)
    assert seen[-1][1] == pytest.approx(8 / 1.0e6)
    # The human-readable log uses the «Предгенерация: X% (Y из Z с)» wording.
    assert any(m.startswith("Предгенерация:") for m in logs)


def test_write_ram_blocks_stops_between_blocks() -> None:
    runner, _logs = _runner(2.6e6)

    class _Sink:
        def __init__(self) -> None:
            self.n = 0

        def write(self, b) -> None:
            self.n += 1
            runner.stop()  # set after the very first block

    runner.sink = _Sink()
    blocks = [np.zeros(4, dtype=np.complex128) for _ in range(5)]
    produced = runner._write_ram_blocks(blocks, None, 0.0, loop=True)
    assert runner.sink.n == 1 and produced == 4


# ======================================================================
# E) GUI: read-only start date + tx-check checkbox
# ======================================================================
def _app():
    from PyQt5 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_gui_start_date_is_locked_only_time_editable() -> None:
    from PyQt5 import QtCore, QtGui, QtWidgets
    from gnss_sim.gui import MainWindow, _DateLockedDateTimeEdit

    app = _app()
    win = MainWindow()
    try:
        assert isinstance(win.ed_start, _DateLockedDateTimeEdit)
        assert "yyyy/MM/dd" in win.ed_start.displayFormat()
        # A click/arrow that lands on a date section is redirected to the hour.
        win.ed_start.setCurrentSection(QtWidgets.QDateTimeEdit.DaySection)
        win.ed_start.keyPressEvent(QtGui.QKeyEvent(
            QtCore.QEvent.KeyPress, QtCore.Qt.Key_Up, QtCore.Qt.NoModifier))
        assert (win.ed_start.currentSection()
                == QtWidgets.QDateTimeEdit.HourSection)
        # The read-only label always shows the resolved date.  Widen the widget
        # range so a cached coverage file cannot clamp the chosen value.
        win.ed_start.setDateTimeRange(
            QtCore.QDateTime(2000, 1, 1, 0, 0, 0),
            QtCore.QDateTime(2100, 1, 1, 0, 0, 0))
        win.chk_now.setChecked(False)
        win.ed_start.setDateTime(QtCore.QDateTime(2026, 9, 24, 13, 45, 30))
        win._refresh_start_date_label()
        assert "2026/09/24" in win.lbl_start_date.text()
    finally:
        win.close()
    del app


def test_gui_tx_check_checkbox_and_collect() -> None:
    from gnss_sim.gui import MainWindow

    app = _app()
    win = MainWindow()
    try:
        assert win.cb_tx_check.isChecked() is False
        win.cb_tx_check.setChecked(True)
        win.sp_ch.setValue(0)
        win.sp_rx_ch.setValue(0)
        # Same TX/RX channel must be flagged and block the start button.
        assert win.btn_start.isEnabled() is False
        assert win._collect().tx_check is True
        win.sp_rx_ch.setValue(1)
        cfg = win._collect()
        assert cfg.tx_check is True and cfg.tx_channel != cfg.rx_channel
    finally:
        win.close()
    del app


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
