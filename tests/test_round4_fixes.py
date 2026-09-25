"""Offline regression tests for the round-4 user-reported issues.

Covers:
A) ephemeris coverage / start date comes from the ACTUAL RINEX epochs
   (earliest…latest record epoch), not the old ``toe ± 6 h`` heuristic;
B) a single progress bar with per-phase labels (pre-generation / transmission
   are two segments of the same bar; cyclic TX wraps it and shows the pass);
C) captured native UHD stderr really reaches the GUI journal (plain/coloured)
   end-to-end, including the Windows ``STD_ERROR_HANDLE`` redirect.

Headless, no hardware, no network::

    E:\\MySoftware\\SDR_Scan\\.venv\\Scripts\\python.exe -m pytest tests -q
"""

from __future__ import annotations

import os
import subprocess
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from gnss_sim import nativelog  # noqa: E402
from gnss_sim.config import SimConfig, parse_start_time  # noqa: E402
from gnss_sim.gpstime import date2gps, gps2date, inc_gps_time  # noqa: E402
from gnss_sim.rinex import (  # noqa: E402
    Ephemeris,
    check_start_coverage,
    ephemeris_epoch_span,
)
from gnss_sim.runner import SimulationRunner  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DOY266 = os.path.join(
    _ROOT, "rinex_cache", "BRDC00IGS_R_20262660000_01D_MN.rnx")


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
# A) coverage from the actual RINEX epochs
# ======================================================================
def test_epoch_span_prefers_toc_over_toe() -> None:
    """The window is the record epoch (toc), even when toe is far away."""
    e1, e2 = Ephemeris(), Ephemeris()
    e1.toc = date2gps(2026, 9, 23, 0, 0, 0)
    e1.toe = date2gps(2026, 9, 25, 12, 0, 0)      # old ±6 h would include this
    e2.toc = date2gps(2026, 9, 23, 6, 0, 0)
    e2.toe = date2gps(2026, 9, 25, 18, 0, 0)
    lo, hi = ephemeris_epoch_span({"G01": [e1, e2]})
    assert gps2date(lo)[:5] == (2026, 9, 23, 0, 0)
    assert gps2date(hi)[:5] == (2026, 9, 23, 6, 0)


def test_epoch_span_falls_back_to_toe_for_synthetic_ephemeris() -> None:
    """Hand-built fixtures that only set ``toe`` keep working."""
    eph = Ephemeris()
    eph.toe = date2gps(2022, 1, 1, 0, 0, 0)
    lo, hi = ephemeris_epoch_span({"G01": [eph]})
    assert gps2date(lo)[:3] == (2022, 1, 1)
    assert gps2date(hi)[:3] == (2022, 1, 1)


def test_check_start_coverage_has_no_toe_margin() -> None:
    """A start 6 h after the last epoch must be rejected (old code allowed it)."""
    eph = Ephemeris()
    eph.toc = date2gps(2026, 9, 23, 0, 0, 0)
    eph.toe = date2gps(2026, 9, 23, 0, 0, 0)
    by_sv = {"G01": [eph]}
    assert check_start_coverage(date2gps(2026, 9, 23, 0, 0, 0), by_sv) is None
    msg = check_start_coverage(date2gps(2026, 9, 23, 6, 0, 0), by_sv)
    assert msg and "вне диапазона эфемерид" in msg


@pytest.mark.skipif(not os.path.exists(_DOY266), reason="cached merged RINEX")
def test_gui_window_matches_rinex_epochs_without_margin() -> None:
    from PyQt5 import QtCore
    from gnss_sim.gui import MainWindow

    app = _app()
    win = MainWindow()
    try:
        win.chk_now.setChecked(False)
        win.ed_nav.setText(_DOY266)
        win._update_start_range()
        lo = win.ed_start.minimumDateTime()
        hi = win.ed_start.maximumDateTime()
        from gnss_sim.rinex import parse_nav_file
        by_sv, _ = parse_nav_file(_DOY266)
        elo, ehi = ephemeris_epoch_span(by_sv)
        ey, em, ed, ehh, emi, _ = gps2date(elo)
        hy, hm, hd, hhh, hmi, _ = gps2date(ehi)
        assert (lo.date().year(), lo.date().month(), lo.date().day()) == (
            ey, em, ed)
        assert (lo.time().hour(), lo.time().minute()) == (ehh, emi)
        assert (hi.date().year(), hi.date().month(), hi.date().day()) == (
            hy, hm, hd)
        assert (hi.time().hour(), hi.time().minute()) == (hhh, hmi)
        assert "RINEX" in win.lbl_start_cover.text()
        # The stale «now» date (2026/09/25) is replaced by the file start.
        assert f"{ey:04d}/{em:02d}/{ed:02d}" in win.lbl_start_date.text()
        # The widget range has no extra ±6 h margin.
        assert lo == win.ed_start.minimumDateTime()
        assert hi == win.ed_start.maximumDateTime()
    finally:
        win.close()
    del app


def test_gui_stale_date_replaced_by_resolved_rinex() -> None:
    from PyQt5 import QtCore
    from gnss_sim.gui import MainWindow

    app = _app()
    win = MainWindow()
    try:
        lo = date2gps(2026, 9, 23, 0, 0, 0)
        hi = date2gps(2026, 9, 23, 12, 0, 0)
        e_lo, e_hi = Ephemeris(), Ephemeris()
        e_lo.toc = lo
        e_hi.toc = hi
        win._coverage_by_sv = lambda *a, **k: {"G01": [e_lo, e_hi]}
        win.chk_now.setChecked(False)
        # A stale start on a different (later) day …
        win.ed_start.setDateTimeRange(
            QtCore.QDateTime(2000, 1, 1, 0, 0, 0),
            QtCore.QDateTime(2100, 1, 1, 0, 0, 0))
        win.ed_start.setDateTime(QtCore.QDateTime(2026, 9, 25, 12, 0, 0))
        win._update_start_range()
        # … is moved to the RINEX start and the label reflects the file.
        assert win.ed_start.date().toPyDate().isoformat() == "2026-09-23"
        assert "2026/09/23" in win.lbl_start_date.text()
    finally:
        win.close()
    del app


# ======================================================================
# B) separate pre-generation / transmission progress
# ======================================================================
def test_phase_events_pregen_and_tx_are_separate() -> None:
    cfg = SimConfig(fs=1.0e6, duration=1.0, loop=False, use_usrp=True,
                    block_ms=50.0)
    phases: list[tuple[str, float, int, bool]] = []
    runner = SimulationRunner(
        cfg, phase=lambda k, f, l, s, c: phases.append((k, f, l, c)))
    runner.engine = _FakeEngine()  # type: ignore[assignment]
    runner.sink = _Recorder()

    blocks, got = runner._generate_to_ram(10, 5, frac_scale=0.5)
    pregen = [p for p in phases if p[0] == "pregen"]
    assert pregen and pregen[-1][1] == pytest.approx(1.0)
    assert all(p[3] is False for p in pregen)

    phases.clear()
    runner._write_ram_blocks(blocks, got, 0.0, loop=False, base=0.5, span=0.5)
    tx = [p for p in phases if p[0] == "tx"]
    assert tx and tx[-1][1] == pytest.approx(1.0)
    assert all(p[3] is False for p in tx)
    assert tx[0][1] == pytest.approx(0.5)


def test_phase_events_cyclic_tx_wraps_and_counts_loops() -> None:
    cfg = SimConfig(fs=1.0e6, duration=0.0, loop=True, use_usrp=True)
    phases: list[tuple[str, float, int, bool]] = []

    class _StopAfter:
        def __init__(self) -> None:
            self.n = 0

        def write(self, samples) -> None:
            self.n += int(np.asarray(samples).size)
            if self.n >= 10:
                runner.stop()

    runner = SimulationRunner(
        cfg, phase=lambda k, f, l, s, c: phases.append((k, f, l, c)))
    runner.sink = _StopAfter()
    blocks = [np.zeros(2, dtype=np.complex128) for _ in range(3)]
    runner._write_ram_blocks(blocks, None, 0.0, loop=True, base=0.5, span=0.5)
    tx = [p for p in phases if p[0] == "tx"]
    assert tx and all(p[3] is True for p in tx)
    assert any(b[1] < a[1] for a, b in zip(tx, tx[1:]))   # wraps each loop
    assert max(p[2] for p in tx) >= 1                     # loop count rises
    assert all(0.0 <= p[1] <= 1.0 for p in tx)


def test_gui_single_progress_bar_phase_labels() -> None:
    """The generation tab keeps exactly ONE bar for the whole operation."""
    from PyQt5 import QtWidgets
    from gnss_sim.gui import MainWindow

    app = _app()
    win = MainWindow()
    try:
        bars = win.findChildren(QtWidgets.QProgressBar)
        assert len(bars) == 1
        assert not hasattr(win, "progress_pre")
        assert not hasattr(win, "progress_tx")
        assert not hasattr(win, "lbl_progress_pre")
        assert not hasattr(win, "lbl_progress_tx")

        win._run_cfg = SimConfig(use_usrp=True)
        # Pre-generation: the bar carries the overall (pregen+tx) value while
        # the label names the phase.
        win._on_progress(0.25, 5.0, 5.0, 1.0)
        win._on_phase("pregen", 0.5, 0, 6.0, False)
        assert win.progress.value() == 250
        assert "Предгенерация" in win.lbl_progress.text()

        # Non-cyclic transmission: same bar, combined value; phase in label.
        win._on_progress(0.75, 15.0, 15.0, 1.0)
        win._on_phase("tx", 0.5, 0, 15.0, False)
        assert win.progress.value() == 750
        assert "Передача" in win.lbl_progress.text()

        # Cyclic transmission: the SAME bar wraps every pass and the label
        # shows elapsed time and pass number.
        win._on_phase("tx", 0.25, 2, 30.0, True)
        assert win.progress.value() == 250
        assert "циклическая передача" in win.progress.format().lower()
        assert "Циклическая передача" in win.lbl_progress.text()
        assert "проход 3" in win.lbl_progress.text()
        # A late overall update must not clobber the wrapping cyclic bar.
        win._on_progress(0.9, 35.0, 35.0, 1.0)
        assert win.progress.value() == 250
        # Finishing resets the cyclic flag and completes the bar.
        win._on_finished(None)
        assert win.progress.value() == 1000
    finally:
        win.close()
    del app


def test_gui_setting_nav_path_updates_date_immediately(tmp_path) -> None:
    """Set the RINEX path (as textChanged does) → window/date refresh at once.

    No explicit ``_update_start_range`` call: the ``ed_nav.textChanged`` wiring
    added for issue 3 must move the stale 2026/09/25 start to the file date.
    """
    from PyQt5 import QtCore
    from gnss_sim.gui import MainWindow

    app = _app()
    win = MainWindow()
    try:
        # Keep the initial toggle from resolving anything until the file is set.
        win._coverage_by_sv = lambda *a, **k: None
        win.chk_now.setChecked(False)
        win.ed_start.setDateTimeRange(
            QtCore.QDateTime(2000, 1, 1, 0, 0, 0),
            QtCore.QDateTime(2100, 1, 1, 0, 0, 0))
        win.ed_start.setDateTime(QtCore.QDateTime(2026, 9, 25, 12, 0, 0))
        win._refresh_start_date_label()
        assert "2026/09/25" in win.lbl_start_date.text()

        lo = date2gps(2026, 9, 23, 0, 0, 0)
        hi = date2gps(2026, 9, 23, 12, 0, 0)
        e_lo, e_hi = Ephemeris(), Ephemeris()
        e_lo.toc = lo
        e_hi.toc = hi
        win._coverage_by_sv = lambda *a, **k: {"G01": [e_lo, e_hi]}

        nav = tmp_path / "BRDC00IGS_R_20262660000_01D_MN.rnx"
        nav.write_text("", encoding="utf-8")
        # Only setText — the textChanged handler must do the refresh.
        win.ed_nav.setText(str(nav))

        assert win.ed_start.date().toPyDate().isoformat() == "2026-09-23"
        assert "2026/09/23" in win.lbl_start_date.text()
        assert "RINEX" in win.lbl_start_cover.text()
        assert win.ed_start.minimumDateTime().date().toPyDate().isoformat() == (
            "2026-09-23")
    finally:
        win.close()
    del app


# ======================================================================
# C) native UHD stderr -> GUI journal
# ======================================================================
def test_native_sink_reaches_gui_journal_plain_and_coloured() -> None:
    from gnss_sim.gui import MainWindow, _LOG_ERROR_COLOR, _LOG_WARN_COLOR

    app = _app()
    win = MainWindow()
    old_sink = None
    try:
        nativelog.set_native_stderr_sink(win.bridge.native.emit)
        nativelog._forward(
            "[INFO] B200 Operating over USB 3\n"
            "an error word inside an info line\n"
            "[WARNING] tx late packet\n")
        app.processEvents()
        plain = win.log.toPlainText()
        assert "[INFO] B200 Operating over USB 3" in plain
        assert "an error word inside an info line" in plain
        assert "[WARNING] tx late packet" in plain
        doc = win.log.document()
        by_text = {}
        for i in range(doc.blockCount()):
            blk = doc.findBlockByNumber(i)
            it = blk.begin()
            colors = set()
            while not it.atEnd():
                frag = it.fragment()
                if frag.isValid():
                    colors.add(frag.charFormat().foreground().color().name())
                it += 1
            by_text[blk.text()] = colors
        assert by_text["[INFO] B200 Operating over USB 3"] == {"#000000"}
        assert by_text["an error word inside an info line"] == {"#000000"}
        assert by_text["[WARNING] tx late packet"] == {_LOG_WARN_COLOR}
        assert _LOG_ERROR_COLOR not in {c for cs in by_text.values()
                                        for c in cs}
    finally:
        nativelog.set_native_stderr_sink(old_sink)
        win.close()
    del app


@pytest.mark.skipif(not sys.platform.startswith("win"),
                    reason="Windows STD_ERROR_HANDLE redirect")
def test_native_std_error_handle_reaches_sink(tmp_path) -> None:
    """UHD's own CRT writes to STD_ERROR_HANDLE; it must land in the sink.

    The old code pointed STD_ERROR_HANDLE at the pipe's write handle and then
    closed it, so native UHD lines were written to a dead handle and never
    reached the journal.
    """
    script = tmp_path / "native_handle.py"
    script.write_text(
        "import sys, ctypes, time\n"
        f"sys.path.insert(0, r'{_ROOT}')\n"
        "from gnss_sim.nativelog import (install_native_stderr_filter,\n"
        "    set_native_stderr_sink)\n"
        "def sink(t):\n"
        "    sys.stdout.write('SINK:' + t); sys.stdout.flush()\n"
        "set_native_stderr_sink(sink)\n"
        "install_native_stderr_filter(force=True)\n"
        "h = ctypes.windll.kernel32.GetStdHandle(-12)\n"
        "msg = b'[INFO] via STD_ERROR_HANDLE\\n'\n"
        "n = ctypes.c_ulong(0)\n"
        "ctypes.windll.kernel32.WriteFile(h, msg, len(msg),\n"
        "    ctypes.byref(n), None)\n"
        "time.sleep(0.5)\n",
        encoding="utf-8")
    proc = subprocess.run([sys.executable, str(script)], capture_output=True,
                          env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    assert proc.returncode == 0, proc.stderr
    assert b"SINK:[INFO] via STD_ERROR_HANDLE" in proc.stdout


#: Helper script: capture native stderr, build the real MainWindow, then write a
#: plain line through fd 2 and a warning through ``STD_ERROR_HANDLE``.  It
#: asserts inside the child that both reached the GUI journal (plain warning
#: uncoloured / genuine warning coloured) and prints ``PROBE_OK=True``.
_NATIVE_GUI_PROBE = r'''
import os, sys, time, ctypes
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, r"__ROOT__")
from gnss_sim import nativelog
from gnss_sim.gui import MainWindow, _LOG_WARN_COLOR
from PyQt5 import QtWidgets

nativelog.install_native_stderr_filter(force=True)
app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
win = MainWindow()

# UHD writes via the C runtime (fd 2) — an "error" word in an [INFO] line must
# stay plain.
os.write(2, b"[INFO] plain with error word\n")
# Some UHD/CRT combinations go straight to the Win32 standard-error handle.
h = ctypes.windll.kernel32.GetStdHandle(-12)
msg = b"[WARNING] native warning\n"
n = ctypes.c_ulong(0)
ctypes.windll.kernel32.WriteFile(h, msg, len(msg), ctypes.byref(n), None)

deadline = time.time() + 5.0
while time.time() < deadline:
    app.processEvents()
    time.sleep(0.02)

plain = win.log.toPlainText()
doc = win.log.document()
colors = {}
for i in range(doc.blockCount()):
    blk = doc.findBlockByNumber(i)
    it = blk.begin(); cs = set()
    while not it.atEnd():
        fr = it.fragment()
        if fr.isValid():
            cs.add(fr.charFormat().foreground().color().name())
        it += 1
    colors[blk.text()] = cs

ok = (("[INFO] plain with error word" in plain)
      and ("[WARNING] native warning" in plain)
      and colors.get("[INFO] plain with error word") == {"#000000"}
      and colors.get("[WARNING] native warning") == {_LOG_WARN_COLOR})
sys.stdout.write("PROBE_OK=" + str(ok) + "\n")
sys.stdout.flush()
win.close()
'''


@pytest.mark.skipif(not sys.platform.startswith("win"),
                    reason="Windows native stderr -> GUI journal")
def test_native_stderr_end_to_end_reaches_gui_journal(tmp_path) -> None:
    """UHD's native stderr must reach the real GUI journal (issue 2).

    A helper process installs the fd-2 / ``STD_ERROR_HANDLE`` capture, builds
    the actual ``MainWindow``, and writes a plain line through fd 2 plus a
    warning through ``GetStdHandle(-12)``.  The child asserts the journal shows
    both and colours only the genuine ``[WARNING]``.
    """
    script = tmp_path / "native_gui_probe.py"
    script.write_text(_NATIVE_GUI_PROBE.replace("__ROOT__", _ROOT),
                      encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k != "PYTEST_CURRENT_TEST"}
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run([sys.executable, str(script)], capture_output=True,
                          env=env, timeout=120)
    out = proc.stdout.decode("utf-8", "replace")
    err = proc.stderr.decode("utf-8", "replace")
    assert proc.returncode == 0, (out, err)
    assert "PROBE_OK=True" in out, (out, err)

