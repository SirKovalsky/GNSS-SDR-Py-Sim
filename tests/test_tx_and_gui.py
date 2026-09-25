"""Tests for the TX RAM pre-generation, gain clamp and GUI fixes.

Headless, no hardware, no network::

    E:\\MySoftware\\SDR_Scan\\.venv\\Scripts\\python.exe -m pytest tests -q
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from gnss_sim.config import SimConfig  # noqa: E402
from gnss_sim.gui import _LOG_ERROR_COLOR, _LOG_WARN_COLOR  # noqa: E402
from gnss_sim.iqfile import Sink  # noqa: E402
from gnss_sim.runner import SimulationRunner  # noqa: E402
from gnss_sim.uhd_tx import (  # noqa: E402
    UhdTxSink,
    clamp_gain,
    gain_range_bounds,
)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NAV = os.path.join(_ROOT, "brdc2680.26n")


def _app():
    from PyQt5 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


# ======================================================================
# A2) TX gain clamp / gain range
# ======================================================================
def test_clamp_gain_below_range_warns_and_clamps() -> None:
    applied, warn = clamp_gain(-30.0, (0.0, 89.75))
    assert applied == 0.0
    assert warn and "вне диапазона" in warn
    assert "аттенюатор" in warn  # negative gain needs an external attenuator


def test_clamp_gain_inside_and_above_range() -> None:
    applied, warn = clamp_gain(30.0, (0.0, 89.75))
    assert applied == 30.0 and warn is None
    applied, warn = clamp_gain(100.0, (0.0, 89.75))
    assert applied == 89.75 and warn is not None


def test_clamp_gain_swapped_range_is_handled() -> None:
    applied, warn = clamp_gain(5.0, (89.75, 0.0))
    assert applied == 5.0 and warn is None


def test_gain_range_bounds_variants() -> None:
    class Rng:
        def start(self):  # noqa: D401
            return 0.0

        def stop(self):
            return 89.75

    assert gain_range_bounds(Rng()) == (0.0, 89.75)
    assert gain_range_bounds((0.0, 1.0)) == (0.0, 1.0)
    assert gain_range_bounds(None) == (0.0, 0.0)


def test_uhd_tx_sink_exposes_gain_range_property() -> None:
    assert isinstance(UhdTxSink.gain_range, property)


# ======================================================================
# A1) Runner TX: pre-generation into RAM / bounded producer-consumer
# ======================================================================
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


def _tx_runner(budget: int, duration: float = 2.0):
    cfg = SimConfig(use_usrp=True, fs=1.0e6, duration=duration, loop=False,
                    block_ms=100.0)
    logs: list[str] = []
    runner = SimulationRunner(cfg, log=logs.append)
    runner.engine = _FakeEngine()
    runner.sink = _Recorder()
    runner._ram_budget_bytes = int(budget)
    runner._segment_seconds = None
    return runner, logs


def test_tx_pregen_ram_path() -> None:
    runner, logs = _tx_runner(budget=10 ** 9)
    produced = runner._run_engine(0.0)
    assert produced == 2_000_000
    assert runner.sink.n == 2_000_000
    assert any("предгенерация" in m for m in logs)
    # Generated in larger blocks, but a single pre-pass (no interleaved send).
    assert runner.engine.calls == 20


def test_tx_bounded_producer_consumer_path() -> None:
    runner, logs = _tx_runner(budget=16 * 1000)  # only ~1000 samples fit
    produced = runner._run_engine(0.0)
    assert produced == 2_000_000
    assert runner.sink.n == 2_000_000
    assert any("очередь" in m for m in logs)


def test_tx_bounded_respects_stop() -> None:
    runner, _logs = _tx_runner(budget=16 * 1000, duration=0.0)
    runner.stop()
    produced = runner._run_engine(0.0)
    assert produced == 0


def test_file_only_no_segment_path_unchanged(tmp_path) -> None:
    from gnss_sim.iqfile import FileSink

    cfg = SimConfig(fs=1.0e6, duration=0.5, loop=False, block_ms=100.0)
    logs: list[str] = []
    runner = SimulationRunner(cfg, log=logs.append)
    runner.engine = _FakeEngine()
    runner._segment_seconds = None
    out = tmp_path / "sig.cs16"
    runner.sink = FileSink(str(out), fmt="cs16", fs=1.0e6)
    runner.sink.start()
    produced = runner._run_engine(0.0)
    runner.sink.close()
    assert produced == 500_000
    assert out.exists() and out.stat().st_size == 500_000 * 4


# ======================================================================
# A4) TX endless loop semantics (loop + USRP ignores duration)
# ======================================================================
class _StopAfter(Sink):
    def __init__(self, limit: int, runner) -> None:
        self.n = 0
        self.limit = limit
        self._runner = runner

    def write(self, samples) -> None:
        self.n += int(np.asarray(samples).size)
        if self.n >= self.limit:
            self._runner.stop()


def test_tx_loop_streams_endlessly_past_duration() -> None:
    # duration=0.5 s but loop+TX must keep streaming until «Стоп».
    cfg = SimConfig(use_usrp=True, fs=1.0e6, duration=0.5, loop=True,
                    block_ms=100.0)
    logs: list[str] = []
    runner = SimulationRunner(cfg, log=logs.append)
    runner.engine = _FakeEngine()  # type: ignore[assignment]
    runner._segment_seconds = 0.2
    runner._stop.clear()
    runner.sink = _StopAfter(3_000_000, runner)
    produced = runner._run_engine(0.0)
    assert produced >= 3_000_000  # far beyond duration * fs = 500_000
    assert runner.error is None


def test_plan_logs_endless_tx_loop_and_nav_warning() -> None:
    cfg = SimConfig(use_usrp=True, fs=1.0e6, duration=300.0, loop=True,
                    loop_seconds=300.0)
    logs: list[str] = []
    runner = SimulationRunner(cfg, log=logs.append)
    runner._plan()
    assert runner._segment_seconds == 300.0
    text = "\n".join(logs)
    assert "непрерывная передача до Стоп" in text
    assert "навигационные данные" in text and "--no-loop" in text


def test_report_underflows_logs_warning() -> None:
    class _Sink:
        underflows = 5

    logs: list[str] = []
    runner = SimulationRunner(SimConfig(), log=logs.append)
    runner.sink = _Sink()  # type: ignore[assignment]
    runner._report_underflows()
    assert any("ВНИМАНИЕ: 5 underflow" in m for m in logs)


def test_classify_tx_error_usb_drop_is_clear() -> None:
    from gnss_sim.uhd_tx import TxError, classify_tx_error

    for exc in (RuntimeError("LIBUSB_TRANSFER_NO_DEVICE"), OSError(13)):
        err = classify_tx_error(exc)
        assert isinstance(err, TxError)
        assert "Обрыв USB" in str(err)


# ======================================================================
# A3) CLI exit codes
# ======================================================================
def test_cli_main_success_returns_0(monkeypatch) -> None:
    from gnss_sim import cli

    class OkRunner:
        def __init__(self, cfg, **kwargs):
            self.error = None

        def prepare(self):
            pass

        def start(self):
            pass

        def is_running(self):
            return False

        def stop(self):
            pass

    monkeypatch.setattr(cli, "SimulationRunner", OkRunner)
    assert cli.main([]) == 0


def test_cli_main_prepare_error_returns_1(monkeypatch, capsys) -> None:
    from gnss_sim import cli

    class BadRunner:
        def __init__(self, cfg, **kwargs):
            self.error = None

        def prepare(self):
            raise RuntimeError("нет устройства")

        def start(self):  # pragma: no cover - must not be reached
            raise AssertionError("start() не должен вызываться")

        def is_running(self):  # pragma: no cover
            return False

        def stop(self):  # pragma: no cover
            pass

    monkeypatch.setattr(cli, "SimulationRunner", BadRunner)
    assert cli.main([]) == 1
    assert "нет устройства" in capsys.readouterr().err


# ======================================================================
# B5) Format combo -> output extension
# ======================================================================
def test_gui_format_updates_output_extension() -> None:
    from gnss_sim.gui import MainWindow
    app = _app()
    win = MainWindow()
    try:
        win.ed_out.setText("gnss_sim_output.cs16")
        win.cmb_fmt.setCurrentText("cs8")
        assert win.ed_out.text() == "gnss_sim_output.cs8"
        win.cmb_fmt.setCurrentText("cf32")
        assert win.ed_out.text() == "gnss_sim_output.cf32"
        # A non-IQ extension is left untouched.
        win.ed_out.setText("route.data")
        win.cmb_fmt.setCurrentText("cs4")
        assert win.ed_out.text() == "route.data"
    finally:
        win.close()
    del app


# ======================================================================
# B4) Start time bound to ephemeris coverage
# ======================================================================
def test_gui_start_coverage_binding() -> None:
    from PyQt5 import QtCore
    from gnss_sim.gui import MainWindow
    app = _app()
    win = MainWindow()
    try:
        win.chk_now.setChecked(False)
        win.ed_nav.setText(_NAV)
        win._update_start_range()
        assert "Покрытие эфемерид" in win.lbl_start_cover.text()
        lo = win.ed_start.minimumDateTime()
        hi = win.ed_start.maximumDateTime()
        assert lo < hi
        assert lo.date().year() >= 2026 and hi.date().year() <= 2026
        # An out-of-range value is clamped into the allowed range.
        win.ed_start.setDateTime(QtCore.QDateTime(2020, 1, 1, 0, 0, 0))
        assert win.ed_start.dateTime() >= lo
    finally:
        win.close()
    del app


# ======================================================================
# B3) Explicit multi-GNSS source choice
# ======================================================================
def test_gui_nav_source_choice() -> None:
    from gnss_sim.gui import MainWindow
    app = _app()
    win = MainWindow()
    try:
        labels = [win.cmb_nav_mode.itemText(i)
                  for i in range(win.cmb_nav_mode.count())]
        assert any("Мультисистемный (G/E/J/C)" in text for text in labels)
        note = win.lbl_nav_mode.text().lower()
        assert "прошедш" in note and "посистемн" in note
        win.cmb_nav_mode.setCurrentIndex(1)
        cfg = win._collect()
        assert cfg.nav_mode == "merged"
        assert cfg.download_source == "auto"
    finally:
        win.close()
    del app


# ======================================================================
# B2) CDDIS credentials modal
# ======================================================================
def test_gui_cddis_prompts_when_credentials_missing(monkeypatch) -> None:
    from gnss_sim import gui
    app = _app()
    win = gui.MainWindow()
    launched = {"n": 0}

    class FakeRunner:
        def __init__(self, cfg, **kwargs):
            pass

        def start(self):
            launched["n"] += 1

        def is_running(self):
            return False

        def stop(self):
            pass

    monkeypatch.setattr(gui, "SimulationRunner", FakeRunner)
    calls = {"n": 0}

    def reject():
        calls["n"] += 1
        return False

    monkeypatch.setattr(win, "_prompt_cddis_credentials", reject)
    try:
        win.cmb_source.setCurrentText("cddis")
        win.cb_auto.setChecked(True)
        win._start()
        assert calls["n"] == 1
        assert launched["n"] == 0  # aborted on cancel

        def accept():
            calls["n"] += 1
            win.ed_cddis_user.setText("user")
            win.ed_cddis_pass.setText("secret")
            return True

        monkeypatch.setattr(win, "_prompt_cddis_credentials", accept)
        win._start()
        assert calls["n"] == 2
        assert launched["n"] == 1  # retried and launched
    finally:
        win.close()
    del app


# ======================================================================
# B3) Multi-system start date follows the merged-RINEX availability
# ======================================================================
def test_latest_multignss_date_defaults_to_yesterday(tmp_path) -> None:
    from datetime import datetime, timedelta, timezone
    from gnss_sim.gui import MainWindow
    app = _app()
    win = MainWindow()
    try:
        now = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)
        got = win._latest_multignss_date(now=now, cache_dir=str(tmp_path))
        assert got == now.date() - timedelta(days=1)
    finally:
        win.close()
    del app


def test_latest_multignss_date_prefers_cached_merged(tmp_path) -> None:
    from datetime import datetime, timezone
    from gnss_sim.gui import MainWindow
    from gnss_sim.rinexfetch import doy_from_date
    app = _app()
    win = MainWindow()
    try:
        # Merged RINEX cached for 2026-09-22 (doy 265) ...
        doy = doy_from_date(2026, 9, 22)
        merged = tmp_path / f"BRDC00IGS_R_2026{doy:03d}0000_01D_MN.rnx"
        merged.write_text("x")
        # ... and only a per-system file for 2026-09-24 (doy 267).
        doy = doy_from_date(2026, 9, 24)
        (tmp_path / f"brdc{doy:03d}0.26n").write_text("x")
        now = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)
        got = win._latest_multignss_date(now=now, cache_dir=str(tmp_path))
        assert got.isoformat() == "2026-09-22"
    finally:
        win.close()
    del app


def test_gui_multignss_mode_moves_start_date(monkeypatch) -> None:
    from datetime import date
    from PyQt5 import QtCore
    from gnss_sim.gui import MainWindow
    app = _app()
    win = MainWindow()
    target = date(2026, 9, 24)
    monkeypatch.setattr(win, "_latest_multignss_date", lambda *a, **k: target)
    monkeypatch.setattr(win, "_coverage_by_sv", lambda *a, **k: None)
    try:
        win.ed_start.setDateTimeRange(QtCore.QDateTime(2000, 1, 1, 0, 0, 0),
                                      QtCore.QDateTime(2100, 1, 1, 0, 0, 0))
        win.chk_now.setChecked(False)
        win.ed_start.setDateTime(QtCore.QDateTime(2026, 9, 25, 13, 45, 30))
        win.cmb_nav_mode.setCurrentIndex(1)
        assert win.ed_start.date().toPyDate() == target
        assert win.ed_start.time().hour() == 13  # time-of-day preserved
        note = win.lbl_nav_mode.text()
        assert "прошедш" in note.lower() and "2026/09/24" in note
        # Switching back to GPS-only leaves the date untouched.
        win.cmb_nav_mode.setCurrentIndex(0)
        assert win.ed_start.date().toPyDate() == target
    finally:
        win.close()
    del app


def test_gui_correct_multignss_start_moves_today(monkeypatch) -> None:
    from datetime import datetime, timedelta, timezone
    from PyQt5 import QtCore
    from gnss_sim.gui import MainWindow
    from gnss_sim.config import SimConfig
    app = _app()
    win = MainWindow()
    today = datetime.now(timezone.utc).date()
    target = today - timedelta(days=2)
    monkeypatch.setattr(win, "_latest_multignss_date", lambda *a, **k: target)
    monkeypatch.setattr(win, "_coverage_by_sv", lambda *a, **k: None)
    try:
        win.ed_start.setDateTimeRange(QtCore.QDateTime(2000, 1, 1, 0, 0, 0),
                                      QtCore.QDateTime(2100, 1, 1, 0, 0, 0))
        win.cmb_nav_mode.setCurrentIndex(1)
        win.chk_now.setChecked(False)
        win.ed_start.setDateTime(QtCore.QDateTime(
            today.year, today.month, today.day, 8, 0, 0))
        cfg = SimConfig(nav_mode="merged", start_text=win._start_text())
        assert win._correct_multignss_start(cfg) is True
        assert win.ed_start.date().toPyDate() == target
        assert cfg.start_text.startswith(target.strftime("%Y/%m/%d"))
        assert target.strftime("%Y/%m/%d") in win.lbl_nav_mode.text()
        assert "переведено" in win.log.toPlainText().lower()
        # GPS-only mode is left alone.
        cfg2 = SimConfig(nav_mode="auto", start_text="now")
        assert win._correct_multignss_start(cfg2) is True
    finally:
        win.close()
    del app

# ======================================================================
# B1) Coloured log lines
# ======================================================================
def test_gui_log_colours() -> None:
    from gnss_sim.gui import MainWindow
    app = _app()
    win = MainWindow()
    try:
        win._append_log("ВНИМАНИЕ: тестовое предупреждение")
        win._append_log("ОШИБКА: тестовая ошибка")
        win._append_log("обычная строка")
        plain = win.log.toPlainText()
        assert "ВНИМАНИЕ" in plain and "ОШИБКА" in plain
        html = win.log.document().toHtml()
        assert _LOG_WARN_COLOR in html
        assert _LOG_ERROR_COLOR in html
    finally:
        win.close()
    del app


# ======================================================================
# B6/B7) B210 checkbox on the basic tab; RAM/format hint
# ======================================================================
def test_gui_b210_checkbox_on_basic_tab_and_hint() -> None:
    from gnss_sim.gui import MainWindow
    app = _app()
    win = MainWindow()
    try:
        basic = win.left_tabs.widget(0)
        assert basic.isAncestorOf(win.cb_tx)
        hint = win.lbl_format_hint.text()
        assert "cs8" in hint and "30" in hint
    finally:
        win.close()
    del app
