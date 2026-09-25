"""Regression tests for the round-2 user-reported issues.

Covers:
1) IQ-file generation progress (bar fills for the whole run, status line);
2) GUI log colour never leaking across a captured multi-line native chunk;
3) start time vs ephemeris coverage (auto-move when «now» is uncovered);
4) Basic/Advanced B1I control sync.

Run (headless, no hardware, no network)::

    E:\\MySoftware\\SDR_Scan\\.venv\\Scripts\\python.exe -m pytest tests -q
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from gnss_sim.config import SimConfig, parse_start_time  # noqa: E402
from gnss_sim.gpstime import date2gps, inc_gps_time  # noqa: E402
from gnss_sim.iqfile import Sink  # noqa: E402
from gnss_sim.rinex import Ephemeris  # noqa: E402
from gnss_sim.runner import SimulationRunner  # noqa: E402


def _app():
    from PyQt5 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class _FakeEngine:
    def __init__(self) -> None:
        self.blocks = 0

    def generate_block(self, n: int) -> np.ndarray:
        self.blocks += 1
        return np.zeros(int(n), dtype=np.complex128)


class _Recorder(Sink):
    def __init__(self) -> None:
        self.n = 0

    def write(self, samples) -> None:
        self.n += int(np.asarray(samples).size)


# ======================================================================
# 1) File-generation progress
# ======================================================================
def test_segment_file_generation_emits_progress_every_block() -> None:
    """The bar must advance during synthesis of a looped IQ-file segment."""
    cfg = SimConfig(fs=1.0e6, duration=2.0, loop=True, block_ms=100.0)
    logs: list[str] = []
    seen: list[tuple] = []
    runner = SimulationRunner(
        cfg, log=logs.append,
        progress=lambda f, s, w, r: seen.append((f, s, w, r)))
    runner.engine = _FakeEngine()
    runner.sink = _Recorder()
    runner._segment_seconds = 2.0
    produced = runner._run_engine(0.0)

    assert produced == 2_000_000
    assert runner.sink.n == 2_000_000
    # One update per produced block (no silent synthesis, no jump at the end).
    assert len(seen) == runner.engine.blocks == 20
    assert seen[0][0] == pytest.approx(0.05)
    assert seen[-1][0] == pytest.approx(1.0)
    assert seen[-1][1] == pytest.approx(2.0)
    # Monotonic, smooth filling.
    fracs = [s[0] for s in seen]
    assert fracs == sorted(fracs)
    # File run uses the «генерация» wording (not «передача»).
    assert any(m.startswith("Идёт генерация:") for m in logs)
    assert not any(m.startswith("Идёт передача") for m in logs)


def test_nonsegment_file_generation_logs_status_line() -> None:
    cfg = SimConfig(fs=1.0e6, duration=0.5, loop=False, block_ms=100.0)
    logs: list[str] = []
    runner = SimulationRunner(cfg, log=logs.append)
    runner.engine = _FakeEngine()
    runner.sink = _Recorder()
    runner._segment_seconds = None
    runner._run_engine(0.0)
    assert any(m.startswith("Идёт генерация:") for m in logs)
    assert any("100%" in m for m in logs if m.startswith("Идёт генерация:"))


def test_gui_progress_format_distinguishes_generation_and_tx() -> None:
    from gnss_sim.gui import MainWindow
    app = _app()
    win = MainWindow()
    try:
        win._run_cfg = SimConfig(use_usrp=False)
        win._on_progress(0.5, 1.0, 1.0, 1.0)
        assert win.progress.value() == 500
        assert "генерация" in win.progress.format()
        assert win.lbl_progress.text().startswith("Генерация")
        # A B210 run keeps the preparation bar until a tx phase replaces it.
        win._run_cfg = SimConfig(use_usrp=True)
        win._on_progress(0.5, 1.0, 1.0, 1.0)
        assert "подготовка" in win.progress.format()
        assert win.lbl_progress.text().startswith("Подготовка")
        win._on_phase("tx", 0.5, 0, 1.0, False)
        assert win.progress is win.progress_tx
        assert "передача" in win.progress.format()
        assert win.lbl_progress.text().startswith("Передача")
    finally:
        win.close()
    del app


# ======================================================================
# 2) GUI log colours
# ======================================================================
def _block_colors(win) -> list[tuple[str, set[str]]]:
    doc = win.log.document()
    out: list[tuple[str, set[str]]] = []
    for i in range(doc.blockCount()):
        blk = doc.findBlockByNumber(i)
        it = blk.begin()
        colors: set[str] = set()
        while not it.atEnd():
            frag = it.fragment()
            if frag.isValid():
                colors.add(frag.charFormat().foreground().color().name())
            it += 1
        out.append((blk.text(), colors))
    return out


def test_multiline_native_chunk_not_all_coloured() -> None:
    """A warning in a captured native chunk must not colour the whole chunk."""
    from gnss_sim.gui import MainWindow
    app = _app()
    win = MainWindow()
    try:
        win._append_log("plain first\n[WARNING] native warn\nplain after")
        win._append_log("обычная строка")
        res = _block_colors(win)
        assert [t for t, _ in res] == [
            "plain first", "[WARNING] native warn", "plain after",
            "обычная строка"]
        assert res[0][1] == {"#000000"}
        assert res[1][1] == {"#b26a00"}      # genuine warning coloured
        assert res[2][1] == {"#000000"}      # colour does not leak
        assert res[3][1] == {"#000000"}      # nor to later lines
    finally:
        win.close()
    del app


# ======================================================================
# 3) Start time vs ephemeris coverage
# ======================================================================
def _old_by_sv():
    eph = Ephemeris()
    eph.toe = date2gps(2022, 1, 1, 0, 0, 0)
    return {"G01": [eph]}


def test_update_start_range_disables_now_when_uncovered() -> None:
    from gnss_sim.gui import MainWindow
    app = _app()
    win = MainWindow()
    try:
        win.chk_now.setChecked(True)
        win._coverage_by_sv = lambda *a, **k: _old_by_sv()
        win._update_start_range()
        assert not win.chk_now.isChecked()
        assert not win.chk_now.isEnabled()
        # Start moved into the toe ±6 h window.
        assert (win.ed_start.dateTime() >= win.ed_start.minimumDateTime()
                and win.ed_start.dateTime() <= win.ed_start.maximumDateTime())
        assert "Покрытие эфемерид" in win.lbl_start_cover.text()
        assert win._start_text() != "now"
        start = parse_start_time(win._start_text())
        assert start.week * 604800 + start.sec == pytest.approx(
            date2gps(2022, 1, 1, 0, 0, 0).week * 604800
            + date2gps(2022, 1, 1, 0, 0, 0).sec, abs=1.0)
    finally:
        win.close()
    del app


def test_update_start_range_reenables_now_when_covered() -> None:
    from gnss_sim.gui import MainWindow
    app = _app()
    win = MainWindow()
    try:
        win.chk_now.setEnabled(False)
        # Two epochs around «now» give a real (non-degenerate) RINEX window
        # that covers the current time exactly, as a daily file would.
        now = parse_start_time("now")
        before, after = Ephemeris(), Ephemeris()
        before.toe = inc_gps_time(now, -3600.0)
        after.toe = inc_gps_time(now, +3600.0)
        win._coverage_by_sv = lambda *a, **k: {"G01": [before, after]}
        win._update_start_range()
        assert win.chk_now.isEnabled()
    finally:
        win.close()
    del app


# ======================================================================
# 4) Basic/Advanced B1I sync
# ======================================================================
def test_gui_b1i_basic_advanced_sync() -> None:
    from gnss_sim.gui import MainWindow
    app = _app()
    win = MainWindow()
    try:
        # Default: B1I UNCHECKED, so the Advanced B1I controls are disabled.
        assert not win.cb_bds.isChecked()
        assert not win.cb_combine.isEnabled()
        assert not win.cmb_b1i_data.isEnabled()
        assert not win.cb_auto_b1i.isEnabled()
        # Selecting B1I in Basic enables the Advanced opt-in.
        win.cb_bds.setChecked(True)
        assert win.cb_combine.isEnabled()
        assert win.cmb_b1i_data.isEnabled()
        assert win.cb_auto_b1i.isEnabled()
        # Vice versa: ticking the Advanced opt-in selects B1I in Basic.
        win.cb_bds.setChecked(False)
        assert not win.cb_combine.isChecked()
        win.cb_combine.setChecked(True)
        assert win.cb_bds.isChecked()
        assert win.cb_combine.isChecked()
        # B1I off again collapses the opt-in (never disagree).
        win.cb_bds.setChecked(False)
        assert not win.cb_combine.isChecked()
        # B1I back on re-enables the Advanced controls.
        win.cb_bds.setChecked(True)
        assert win.cmb_b1i_data.isEnabled()
        assert win.cb_auto_b1i.isEnabled()
    finally:
        win.close()
    del app
