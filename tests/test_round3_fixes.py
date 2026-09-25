"""Offline regression tests for the round-3 user-reported issues.

Covers:
1) progress UX: journal milestones only, GUI bar strictly increases, TX halves;
2) native UHD stderr forwarded to the GUI journal (plain vs coloured);
3) start-time field: time editable, date locked, coverage window in GPS time;
4) BeiDou B1I checkbox unchecked by default;
7) default generation duration 120 s.

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
from gnss_sim.runner import SimulationRunner  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MERGED = os.path.join(
    _ROOT, "rinex_cache", "BRDC00IGS_R_20262670000_01D_MN.rnx")


def _app():
    from PyQt5 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class _FakeEngine:
    def __init__(self) -> None:
        self.blocks = 0

    def generate_block(self, n: int) -> np.ndarray:
        self.blocks += 1
        return np.zeros(int(n), dtype=np.complex128)


class _Recorder:
    def __init__(self) -> None:
        self.n = 0

    def write(self, samples) -> None:
        self.n += int(np.asarray(samples).size)


# ======================================================================
# 7) default duration 120 s
# ======================================================================
def test_default_duration_is_120() -> None:
    assert SimConfig().duration == 120.0
    from gnss_sim.cli import build_parser
    assert build_parser().parse_args([]).duration == 120.0
    from gnss_sim.gui import MainWindow
    app = _app()
    win = MainWindow()
    try:
        assert win.sp_dur.value() == 120.0
        win._reset_nav_group()
        assert win.sp_dur.value() == 120.0
    finally:
        win.close()
    del app


# ======================================================================
# 4) B1I unchecked by default
# ======================================================================
def test_b1i_checkbox_unchecked_by_default() -> None:
    from gnss_sim.gui import MainWindow
    app = _app()
    win = MainWindow()
    try:
        assert win.cb_bds.isChecked() is False
        assert win._collect().enable_beidou is False
        win.cb_bds.setChecked(True)
        win._reset_signals_group()
        assert win.cb_bds.isChecked() is False  # reset restores the default
    finally:
        win.close()
    del app


# ======================================================================
# 1) progress UX
# ======================================================================
def test_progress_bar_strictly_increases_file_run() -> None:
    """The GUI bar advances (strictly) through a whole IQ-file generation."""
    from gnss_sim.gui import MainWindow
    app = _app()
    win = MainWindow()
    try:
        cfg = SimConfig(fs=1.0e6, duration=1.0, loop=False, block_ms=50.0)
        seen: list[float] = []

        def progress(frac, sim_s, wall, rate):
            win._on_progress(frac, sim_s, wall, rate)
            seen.append(float(win.progress.value()))

        runner = SimulationRunner(cfg, progress=progress)
        runner.engine = _FakeEngine()  # type: ignore[assignment]
        runner.sink = _Recorder()
        runner._segment_seconds = None
        runner._run_engine(0.0)
        assert len(seen) == 20
        assert all(b > a for a, b in zip(seen, seen[1:]))  # strictly increasing
        assert seen[-1] == 1000
    finally:
        win.close()
    del app


def test_tx_progress_pregen_then_streaming() -> None:
    """TX reserves the first half of the bar for pre-generation (issue 1b)."""
    cfg = SimConfig(fs=1.0e6, duration=1.0, loop=False, block_ms=50.0,
                    use_usrp=True)
    fracs: list[float] = []
    runner = SimulationRunner(
        cfg, progress=lambda f, s, w, r: fracs.append(round(f, 6)))
    runner.engine = _FakeEngine()  # type: ignore[assignment]
    runner.sink = _Recorder()
    runner._segment_seconds = None
    blocks, got = runner._generate_to_ram(10, 5, frac_scale=0.5)
    assert got == 10 and fracs == [0.25, 0.5]
    runner._write_ram_blocks(blocks, got, 0.0, loop=False, base=0.5, span=0.5)
    assert fracs[-1] == pytest.approx(1.0)
    assert all(b > a for a, b in zip(fracs, fracs[1:]))


def test_endless_tx_progress_never_reaches_100_before_stop() -> None:
    cfg = SimConfig(fs=1.0e6, use_usrp=True)
    fracs: list[float] = []

    class _StopAfter:
        def __init__(self) -> None:
            self.n = 0

        def write(self, samples) -> None:
            self.n += int(np.asarray(samples).size)
            if self.n >= 10:            # stop during the second pass
                runner.stop()

    runner = SimulationRunner(
        cfg, progress=lambda f, s, w, r: fracs.append(round(f, 6)))
    runner.sink = _StopAfter()
    blocks = [np.zeros(2, dtype=np.complex128) for _ in range(3)]
    runner._write_ram_blocks(blocks, None, 0.0, loop=True, base=0.5, span=0.5)
    assert len(fracs) >= 2
    assert all(0.5 <= f < 1.0 for f in fracs)  # 100 % only on real completion
    assert all(b >= a for a, b in zip(fracs, fracs[1:]))


def test_run_engine_tx_segment_progress_two_phases() -> None:
    """B210 segment run: bar fills during pre-gen AND streaming (issue 1b)."""
    cfg = SimConfig(fs=1.0e6, duration=0.0, loop=True, block_ms=1.0,
                    use_usrp=True)
    fracs: list[float] = []

    class _StopAfter:
        def __init__(self) -> None:
            self.n = 0

        def write(self, samples) -> None:
            self.n += int(np.asarray(samples).size)
            if self.n >= 5:
                runner.stop()

    runner = SimulationRunner(
        cfg, progress=lambda f, s, w, r: fracs.append(round(f, 6)))
    runner.engine = _FakeEngine()  # type: ignore[assignment]
    runner.sink = _StopAfter()
    runner._segment_seconds = 0.004          # 4000 samples, 4 blocks
    runner._run_engine(0.0)
    assert fracs[0] == pytest.approx(0.125)
    assert 0.5 in [round(f, 6) for f in fracs]     # pre-generation complete
    assert all(b >= a for a, b in zip(fracs, fracs[1:]))
    assert max(fracs) < 1.0                  # 100 % only on real completion


def test_journal_progress_is_milestone_throttled() -> None:
    """A 1 s file run must not log one progress line per generated block."""
    cfg = SimConfig(fs=1.0e6, duration=1.0, loop=False, block_ms=50.0)
    logs: list[str] = []
    runner = SimulationRunner(cfg, log=logs.append)
    runner.engine = _FakeEngine()  # type: ignore[assignment]
    runner.sink = _Recorder()
    runner._segment_seconds = None
    runner._run_engine(0.0)
    lines = [m for m in logs if m.startswith("Идёт генерация:")]
    assert 1 <= len(lines) <= 4              # 20 blocks would be 20 lines
    assert any("100%" in m for m in lines)


# ======================================================================
# 2) native UHD stderr -> GUI journal
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


def test_native_log_forwarded_plain_and_warning() -> None:
    from gnss_sim.gui import MainWindow
    app = _app()
    win = MainWindow()
    try:
        win._append_native_log(
            "[INFO] B200 Operating over USB 3\n"
            "an error word inside an info line\n"
            "[WARNING] tx late packet\n"
            "[ERROR] device gone")
        res = _block_colors(win)
        assert [t for t, _ in res] == [
            "[INFO] B200 Operating over USB 3",
            "an error word inside an info line",
            "[WARNING] tx late packet",
            "[ERROR] device gone"]
        assert res[0][1] == {"#000000"}
        assert res[1][1] == {"#000000"}      # substring-only -> plain
        assert res[2][1] == {_LOG_WARN_COLOR}
        assert res[3][1] == {_LOG_ERROR_COLOR}
        # The lines are actually present in the journal text.
        plain = win.log.toPlainText()
        assert "[INFO] B200 Operating over USB 3" in plain
        assert "[WARNING] tx late packet" in plain
    finally:
        win.close()
    del app


# ======================================================================
# 3) start time: editable time, locked date, GPS coverage window
# ======================================================================
@pytest.mark.skipif(not os.path.exists(_MERGED), reason="merged cache missing")
def test_b1i_toe_span_normalised_to_gps() -> None:
    from gnss_sim.rinex import ephemeris_toe_span, parse_nav_file
    by_sv, _iono = parse_nav_file(_MERGED)
    span = ephemeris_toe_span(by_sv)
    assert span is not None
    lo, hi = span
    assert lo.week >= 2400 and hi.week >= 2400   # not the BDT week 1081 (2000)
    assert lo.week <= hi.week


@pytest.mark.skipif(not os.path.exists(_MERGED), reason="merged cache missing")
def test_start_time_editable_date_locked() -> None:
    from PyQt5 import QtCore, QtWidgets
    from gnss_sim.gui import MainWindow
    app = _app()
    win = MainWindow()
    try:
        win.chk_now.setChecked(False)
        win.ed_nav.setText(_MERGED)
        win.ed_start.setDateTime(QtCore.QDateTime(2026, 9, 24, 18, 0, 0))
        win._update_start_range()
        cover = win.lbl_start_cover.text()
        assert "2026" in cover and "2000" not in cover

        # The coverage window is the RINEX field, not year 2000.
        assert win.ed_start.minimumDateTime().date().year() == 2026
        assert win.ed_start.maximumDateTime().date().year() == 2026

        # Stepping the time works and preserves the locked date.
        win.ed_start.setCurrentSection(QtWidgets.QDateTimeEdit.HourSection)
        win.ed_start.stepBy(1)
        assert win.ed_start.date().toPyDate().isoformat() == "2026-09-24"
        assert win.ed_start.time().hour() == 19

        # A date section is redirected to the time; the date never changes.
        win.ed_start.setCurrentSection(QtWidgets.QDateTimeEdit.DaySection)
        win.ed_start.setCurrentSection(
            win.ed_start.currentSection())
        win.ed_start.stepBy(1)
        assert win.ed_start.date().toPyDate().isoformat() == "2026-09-24"

        # The widget range and the label agree (no local->UTC shift).
        win.ed_start.setDateTime(QtCore.QDateTime(2020, 1, 1, 0, 0, 0))
        assert win.ed_start.dateTime() >= win.ed_start.minimumDateTime()
        assert win.ed_start.dateTime() <= win.ed_start.maximumDateTime()
        assert win.ed_start.minimumDateTime().toString("yyyy/MM/dd HH:mm") in cover
        assert win.ed_start.maximumDateTime().toString("yyyy/MM/dd HH:mm") in cover
    finally:
        win.close()
    del app
