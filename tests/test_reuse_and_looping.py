"""Tests for IQ-file reuse, loop sizing, gzip RINEX, GUI fixes, stop.

Run (headless, no hardware, no network)::

    E:\\MySoftware\\SDR_Scan\\.venv\\Scripts\\python.exe -m pytest tests -q
"""

from __future__ import annotations

import gzip
import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from gnss_sim import rinex, sysinfo  # noqa: E402
from gnss_sim.config import SimConfig, parse_start_time  # noqa: E402
from gnss_sim.gpstime import date2gps  # noqa: E402
from gnss_sim.iqfile import (  # noqa: E402
    FileSink,
    IqFileSource,
    NullSink,
    Sink,
)
from gnss_sim.runner import SimulationRunner  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NAV = os.path.join(_ROOT, "brdc2680.26n")
_NAV_GZ = os.path.join(_ROOT, "brdc2680.26n.gz")


def _app():
    from PyQt5 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _write_iq(path: str, fmt: str = "cs16", fs: float = 1.0e6,
              n: int = 4096, scale: float = 10000.0) -> np.ndarray:
    rng = np.random.default_rng(1234)
    x = (0.1 * (rng.standard_normal(n) + 1j * rng.standard_normal(n)))
    sink = FileSink(path, fmt=fmt, fs=fs, center_freq=1575.42e6, scale=scale)
    sink.start()
    sink.write(x)
    sink.close()
    return x


# ======================================================================
# 1) IQ file reuse / IqFileSource
# ======================================================================
@pytest.mark.parametrize("fmt", ["cs16", "cf32", "cs8", "cs4"])
def test_iq_source_roundtrip(tmp_path, fmt) -> None:
    path = str(tmp_path / f"sig.{fmt}")
    x = _write_iq(path, fmt=fmt, n=2048)
    src = IqFileSource(path)
    src.start()
    try:
        assert src.fmt == fmt
        assert src.fs == 1.0e6
        assert src.center_freq == 1575.42e6
        assert src.total_samples == 2048
        y = src.read(2048)
    finally:
        src.close()
    assert y.size == 2048
    # Integer formats are quantised; cf32 is essentially exact.
    div = {"cf32": 1.0, "cs16": 1.0, "cs8": 256.0, "cs4": 4096.0}[fmt]
    tol = 1e-6 if fmt == "cf32" else 1.5 * div / 10000.0
    assert float(np.max(np.abs(y - x))) < tol


def test_iq_source_metadata_from_sidecar(tmp_path) -> None:
    path = str(tmp_path / "sig.cs16")
    _write_iq(path, fmt="cs16", fs=500000.0, n=512)
    # No arguments given -> everything from the JSON side-car.
    src = IqFileSource(path)
    src.start()
    try:
        assert src.fmt == "cs16"
        assert src.fs == 500000.0
        assert src.center_freq == 1575.42e6
        assert src.total_samples == 512
        assert src.metadata  # side-car loaded
        assert src.read(512).size == 512
    finally:
        src.close()


def test_iq_source_missing_json_uses_config_and_warns(tmp_path) -> None:
    path = str(tmp_path / "raw.bin")
    x = (0.2 * (np.arange(64) + 1j * np.arange(64))).astype(np.complex128)
    sink = FileSink(path, fmt="cs16", fs=1.0, center_freq=1.0, scale=10000.0)
    sink.start(); sink.write(x); sink.close()
    os.remove(path + ".json")  # pretend there is no side-car

    logs: list[str] = []
    src = IqFileSource(path, fmt="cs16", fs=1.0e6, center_freq=123.0,
                       scale=10000.0, log=logs.append)
    assert src.fmt == "cs16" and src.fs == 1.0e6
    assert src.center_freq == 123.0
    assert logs and "json" in logs[0].lower()


def test_config_and_cli_iq_input() -> None:
    assert SimConfig().iq_input == ""
    from gnss_sim.cli import build_parser
    args = build_parser().parse_args(["--iq-input", "a.cs16"])
    assert args.iq_input == "a.cs16"
    assert build_parser().parse_args([]).iq_input == ""


def test_runner_iq_input_skips_engine(tmp_path) -> None:
    path = str(tmp_path / "sig.cs16")
    _write_iq(path, n=128)
    logs: list[str] = []
    cfg = SimConfig(iq_input=path, fs=1.0e6, center_freq=1575.42e6,
                    duration=0.01, loop=False)
    runner = SimulationRunner(cfg, log=logs.append)
    runner.prepare()
    assert runner.engine is None
    assert runner._source is not None
    runner.run()
    assert runner.error is None
    assert any("Повторное использование IQ" in m for m in logs)


def test_runner_iq_input_playback_loops_from_disk(tmp_path) -> None:
    path = str(tmp_path / "sig.cs16")
    _write_iq(path, fs=1000.0, n=2000)

    class Recorder(Sink):
        def __init__(self) -> None:
            self.n = 0

        def write(self, samples) -> None:
            self.n += int(np.asarray(samples).size)

    cfg = SimConfig(iq_input=path, fs=1000.0, center_freq=1575.42e6,
                    duration=10.0, loop=True, block_ms=500.0, use_usrp=True)
    runner = SimulationRunner(cfg, log=lambda _m: None)
    runner._source = IqFileSource(path, fmt="cs16", fs=1000.0)
    runner._source.start()
    runner._loop = True
    runner.sink = Recorder()
    produced = runner._run_source()
    assert produced == 10000
    assert runner.sink.n == 10000


# ======================================================================
# 2) Loop-segment auto sizing
# ======================================================================
def test_auto_segment_exact_when_fits() -> None:
    # The reported bug: duration 360 s must give exactly 360 s (no fixed 90).
    seg, capped = sysinfo.auto_segment_seconds(360.0, 1000.0, 0.0)
    assert seg == 360.0 and capped is False
    seg, capped = sysinfo.auto_segment_seconds(60.0, 1000.0, 0.0)
    assert seg == 60.0 and capped is False


def test_auto_segment_capped_to_budget_multiple_of_90() -> None:
    seg, capped = sysinfo.auto_segment_seconds(360.0, 100.0, 0.0)
    assert capped is True and seg == 90.0
    seg, capped = sysinfo.auto_segment_seconds(360.0, 200.0, 0.0)
    assert capped is True and seg == 180.0


def test_auto_segment_explicit_and_indefinite() -> None:
    seg, capped = sysinfo.auto_segment_seconds(360.0, 1000.0, 30.0)
    assert seg == 30.0 and capped is False
    seg, capped = sysinfo.auto_segment_seconds(0.0, 1000.0, 0.0)
    assert seg == 90.0 and capped is False  # bounded default when indefinite
    seg, capped = sysinfo.auto_segment_seconds(0.0, 45.0, 0.0)
    assert seg == 45.0


# ======================================================================
# 3) Start-time validation
# ======================================================================
def test_start_time_coverage_validation() -> None:
    by_sv, _iono = rinex.parse_nav_file(_NAV)
    # brdc2680.26n covers its actual RINEX epochs (2026/09/25 00:00…06:14).
    inside = date2gps(2026, 9, 25, 3, 0, 0)
    assert rinex.check_start_coverage(inside, by_sv) is None
    outside = date2gps(2020, 1, 1, 0, 0, 0)
    msg = rinex.check_start_coverage(outside, by_sv)
    assert msg and "вне диапазона эфемерид" in msg
    assert "RINEX" in msg


def test_gui_start_widget_full_datetime_no_error(monkeypatch) -> None:
    from PyQt5 import QtCore
    from gnss_sim.gui import MainWindow
    app = _app()
    win = MainWindow()
    monkeypatch.setattr(win, "_coverage_by_sv", lambda *a, **k: None)
    win.chk_now.setChecked(True)  # emulate «no cached coverage» start state
    try:
        assert win._start_text() == "now"
        win.chk_now.setChecked(False)
        win.ed_start.setDateTime(QtCore.QDateTime(2026, 9, 25, 0, 0, 0))
        text = win._start_text()
        assert text == "2026/09/25 00:00:00"
        # «00:00:00» now parses as a full datetime (no «неверная дата/время»).
        parse_start_time(text)
    finally:
        win.close()
    del app


def test_gui_validate_start_rejects_out_of_range(monkeypatch) -> None:
    from PyQt5 import QtCore, QtWidgets
    from gnss_sim import gui
    from gnss_sim.gui import MainWindow
    app = _app()
    win = MainWindow()
    warnings = {"n": 0}
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "warning",
        staticmethod(lambda *a, **k: warnings.__setitem__(
            "n", warnings["n"] + 1)))
    try:
        win.chk_now.setChecked(False)
        # Widen the widget range so a genuinely out-of-coverage start reaches
        # _validate_start (the coverage binding would otherwise clamp it).
        win.ed_start.setDateTimeRange(
            QtCore.QDateTime(2000, 1, 1, 0, 0, 0),
            QtCore.QDateTime(2100, 1, 1, 0, 0, 0))
        win.ed_start.setDateTime(QtCore.QDateTime(2020, 1, 1, 0, 0, 0))
        cfg = win._collect()
        cfg.nav_file = _NAV
        assert win._validate_start(cfg) is False
        assert warnings["n"] == 1
        # Inside the actual RINEX epoch window (2026/09/25 00:00…06:14).
        win.ed_start.setDateTime(QtCore.QDateTime(2026, 9, 25, 3, 0, 0))
        cfg = win._collect()
        cfg.nav_file = _NAV
        assert win._validate_start(cfg) is True
    finally:
        win.close()
    del app


# ======================================================================
# 4) Combo wheel guard + default first item
# ======================================================================
def test_combo_wheel_ignored_unless_popup_open() -> None:
    from PyQt5 import QtCore, QtGui
    from gnss_sim.gui import MainWindow
    app = _app()
    win = MainWindow()
    try:
        event = QtGui.QWheelEvent(
            QtCore.QPointF(1, 1), QtCore.QPointF(1, 1), QtCore.QPoint(0, 0),
            QtCore.QPoint(0, 120), QtCore.Qt.NoButton,
            QtCore.Qt.NoModifier, QtCore.Qt.NoScrollPhase, False)
        # The frequency/centre combos were replaced by read-only labels, but the
        # remaining guarded combos (e.g. the synthesis backend) must still
        # swallow a stray wheel while their popup is closed.
        combo = win.cmb_backend
        before = combo.currentText()
        guard = combo._wheel_guard  # noqa: SLF001
        handled = guard.eventFilter(combo, event)
        assert handled is True
        assert combo.currentText() == before
        # fs/centre are no longer editable widgets at all.
        assert not hasattr(win, "cmb_fs") and not hasattr(win, "cmb_fc")
    finally:
        win.close()
    del app


# ======================================================================
# 5) Per-group reset must not clear credentials
# ======================================================================
def test_group_reset_does_not_touch_credentials() -> None:
    from gnss_sim.gui import MainWindow
    app = _app()
    win = MainWindow()
    try:
        win.ed_cddis_user.setText("user")
        win.ed_cddis_pass.setText("secret")
        win.ed_cddis_token.setText("tok")
        win.cmb_source.setCurrentText("cddis")

        win._reset_cddis_group()
        assert win.cmb_source.currentText() == "auto"
        assert win.ed_cddis_user.text() == "user"
        assert win.ed_cddis_pass.text() == "secret"
        assert win.ed_cddis_token.text() == "tok"

        win._reset_defaults()
        assert win.ed_cddis_user.text() == "user"
        assert win.ed_cddis_pass.text() == "secret"
        assert win.ed_cddis_token.text() == "tok"
    finally:
        win.close()
    del app


def test_output_group_reset_clears_iq_input() -> None:
    from gnss_sim.gui import MainWindow
    app = _app()
    win = MainWindow()
    try:
        win.cb_iq_in.setChecked(True)
        win.ed_iq_in.setText("reuse.cs16")
        assert win._collect().iq_input == "reuse.cs16"
        win._reset_output_group()
        assert win.cb_iq_in.isChecked() is False
        assert win._collect().iq_input == ""
    finally:
        win.close()
    del app


# ======================================================================
# 6) RINEX gzip + multi-GNSS reporting
# ======================================================================
def test_gzip_rinex_decompression_and_parse(tmp_path) -> None:
    with open(_NAV, "rb") as fh:
        data = fh.read()
    gz = tmp_path / "brdc2680.26n.gz"
    gz.write_bytes(gzip.compress(data))
    plain = tmp_path / "brdc2680.26n"
    assert not plain.exists()
    by_sv, _iono = rinex.parse_nav_file(str(gz))
    assert by_sv and rinex.system_counts(by_sv) == {"G": 106}
    # decompressed cache written next to the .gz
    assert plain.exists() and plain.stat().st_size == len(data)


def test_parse_nav_gz_extension_matches_plain(tmp_path) -> None:
    gz = tmp_path / "brdc2680.26n.gz"
    gz.write_bytes(open(_NAV_GZ, "rb").read())  # already gzip-compressed
    by_gz, _ = rinex.parse_nav_file(str(gz))
    by_plain, _ = rinex.parse_nav_file(_NAV)
    assert sorted(by_gz) == sorted(by_plain)
    assert rinex.count_systems(str(gz)) == {"G": 106}


def test_count_systems_multi_gnss() -> None:
    merged = os.path.join(_ROOT, "rinex_cache",
                          "BRDC00IGS_R_20262660000_01D_MN.rnx")
    if not os.path.exists(merged):
        pytest.skip("merged multi-GNSS cache not present")
    counts = rinex.count_systems(merged)
    assert counts.get("G", 0) > 0
    assert (counts.get("E", 0) + counts.get("C", 0) + counts.get("J", 0)) > 0


def test_unique_sv_counts_vs_record_counts() -> None:
    by_sv = {"G01": [1, 2, 3], "G02": [1], "E11": [1, 2],
             "J01": [1], "C01": [1]}
    # Unique satellites per system (what the runner reports).
    assert rinex.unique_sv_counts(by_sv) == {"G": 2, "E": 1, "J": 1, "C": 1}
    # Records per system (kept for other callers, no longer in the log line).
    assert rinex.system_counts(by_sv) == {"G": 4, "E": 2, "J": 1, "C": 1}


# ======================================================================
# 7) Stop responsiveness during RAM segment generation
# ======================================================================
def test_stop_reacts_within_one_block() -> None:
    class SlowEngine:
        def __init__(self) -> None:
            self.count = 0

        def generate_block(self, n: int) -> np.ndarray:
            time.sleep(0.03)
            self.count += 1
            return np.zeros(n, dtype=np.complex128)

    cfg = SimConfig(fs=1.0e6, duration=0.0, loop=True, block_ms=10.0)
    runner = SimulationRunner(cfg, log=lambda _m: None)
    runner.engine = SlowEngine()  # type: ignore[assignment]
    runner.sink = NullSink()
    runner.sink.start()
    runner._segment_seconds = 1000.0
    thread = threading.Thread(target=runner.run)
    thread.start()
    time.sleep(0.2)
    runner.stop()
    thread.join(timeout=3.0)
    assert not thread.is_alive()
    assert runner.error is None
    # With a 30 ms block, ~0.2 s yields a handful of blocks, not a whole
    # 1000 s segment.
    assert 1 <= runner.engine.count <= 15
