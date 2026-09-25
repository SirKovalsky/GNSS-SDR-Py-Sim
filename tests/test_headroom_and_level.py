"""Offline tests for the automatic amplitude + anti-clip headroom control and
the GUI TX level indicator.

No hardware and no network::

    E:\\MySoftware\\SDR_Scan\\.venv\\Scripts\\python.exe -m pytest tests -q
"""

from __future__ import annotations

import math
import os
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from gnss_sim.config import SimConfig  # noqa: E402
from gnss_sim.power import (  # noqa: E402
    AUTO_AMP_MAX,
    AUTO_AMP_MIN,
    HeadroomController,
    auto_amp_scale,
    combine_levels,
    level_stats,
)
from gnss_sim.runner import SimulationRunner  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MERGED = os.path.join(
    _ROOT, "rinex_cache", "BRDC00IGS_R_20262670000_01D_MN.rnx")


def _app():
    from PyQt5 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class _FakeEngine:
    """Constant composite block with a selectable peak."""

    def __init__(self, value: complex = 1.5 + 1.5j) -> None:
        self.value = complex(value)
        self.amp_scale = 0.15
        self.channels: list = []
        self.calls = 0

    def generate_block(self, n: int) -> np.ndarray:
        self.calls += 1
        return np.full(int(n), self.value, dtype=np.complex128)


class _PeakRecorder:
    """Sink that tracks the largest |x| it was asked to send."""

    def __init__(self) -> None:
        self.n = 0
        self.peak = 0.0
        self.clips = 0

    def write(self, samples) -> None:
        x = np.asarray(samples)
        if x.size:
            mag = np.abs(x)
            self.peak = max(self.peak, float(mag.max()))
            self.clips += int(np.count_nonzero(mag > 1.0))
        self.n += int(x.size)


# ======================================================================
# Level statistics
# ======================================================================
def test_level_stats_peak_rms_clips() -> None:
    x = np.array([1.0, 1j, -0.5, 2.0 + 0j, 0.0], dtype=np.complex128)
    st = level_stats(x)
    assert st.peak == pytest.approx(2.0)
    assert st.samples == 5
    assert st.clips == 1
    assert st.peak_dbfs == pytest.approx(20 * math.log10(2.0))
    assert level_stats(np.zeros(0)).peak == 0.0
    assert level_stats(np.zeros(0)).peak_dbfs == float("-inf")


def test_combine_levels_energy_weighted() -> None:
    a = np.full(100, 0.5)
    b = np.full(300, 1.5)
    st = combine_levels([a, b])
    assert st.peak == pytest.approx(1.5)
    assert st.samples == 400
    assert st.clips == 300
    expected_rms = math.sqrt((100 * 0.25 + 300 * 2.25) / 400)
    assert st.rms == pytest.approx(expected_rms)


# ======================================================================
# Automatic amplitude
# ======================================================================
def test_auto_amp_scale_bounds_and_shape() -> None:
    # No usable weights -> the maximum (keeps something usable).
    assert auto_amp_scale([]) == pytest.approx(AUTO_AMP_MAX)
    assert auto_amp_scale([0.0, 0.0]) == pytest.approx(AUTO_AMP_MAX)
    # A single channel must not blow up past the historic ceiling.
    assert auto_amp_scale([1.0]) == pytest.approx(AUTO_AMP_MAX)
    # Many channels -> a small, bounded scale (headroom without the controller).
    many = auto_amp_scale([1.0] * 45)
    assert AUTO_AMP_MIN <= many <= 0.10
    # Monotone: more channels can only lower the scale.
    assert auto_amp_scale([1.0] * 26) > many


def test_auto_amp_gps_only_stays_loud() -> None:
    # GPS-only L1 C/A ~ sum of 5.9 weight units: stays near the historic level.
    scale = auto_amp_scale([0.59] * 10)
    assert 0.10 <= scale <= AUTO_AMP_MAX


def test_auto_amp_respects_format_full_scale() -> None:
    """cs16 full scale is ~3.28, so a normal multi-GNSS sum keeps the nominal.

    This is the regression fix: with the old hard-coded 1.0 reference the same
    weights produced ≈0.05 (≈-10 dB) and the receiver lost the fix.
    """
    weights = [0.59] * 23          # ~13.6: GPS+QZSS+SBAS L1 scene
    assert auto_amp_scale(weights) < 0.08          # old 1.0 reference
    assert auto_amp_scale(weights, full_scale=3.2767) == pytest.approx(
        AUTO_AMP_MAX)
    # An over-range estimate is still allowed to fall (safety preserved).
    assert auto_amp_scale([1.0] * 45, full_scale=3.2767) < AUTO_AMP_MAX


def test_headroom_target_scales_with_format() -> None:
    ctl = HeadroomController(target=0.7, full_scale=3.2767,
                             log=lambda _m: None)
    assert ctl.target == pytest.approx(0.7 * 3.2767)
    hot = [np.full(100, 5.0 + 0j)]
    ctl.process_segment(hot)
    assert combine_levels(hot).peak == pytest.approx(ctl.target, rel=1e-6)


def test_config_and_cli_defaults() -> None:
    cfg = SimConfig()
    assert cfg.amp_scale is None       # historic nominal 0.15
    assert cfg.headroom is False       # anti-clip is opt-in
    assert cfg.headroom_target == pytest.approx(0.7)

    from gnss_sim.cli import build_parser
    p = build_parser()
    a = p.parse_args([])
    assert a.amp is None
    assert a.headroom is None          # off unless requested
    assert a.no_headroom is False
    a = p.parse_args(["--headroom"])
    assert a.headroom == pytest.approx(0.7)
    a = p.parse_args(["--headroom", "0.5", "--amp", "0.1"])
    assert a.headroom == pytest.approx(0.5)
    assert a.amp == pytest.approx(0.1)
    a = p.parse_args(["--no-headroom"])
    assert a.no_headroom is True


# ======================================================================
# Anti-clip controller
# ======================================================================
def test_headroom_segment_single_exact_scale() -> None:
    blocks = [np.full(1000, 1.0 + 0j), np.full(1000, -2.0 + 0j),
              np.full(10, 1.5j)]
    ctl = HeadroomController(target=0.7, log=lambda _m: None)
    out, scale = ctl.process_segment(blocks)
    assert out is blocks
    assert scale == pytest.approx(0.7 / 2.0)
    # Exactly one factor for the whole segment; peak at the target, no clips.
    st = combine_levels(blocks)
    assert st.peak == pytest.approx(0.7)
    assert st.clips == 0
    # Applying the scale twice would overshoot -> proves a single pass.
    assert np.abs(blocks[1]).max() == pytest.approx(0.7)


def test_headroom_segment_noop_below_target() -> None:
    blocks = [np.full(64, 0.3 + 0.3j)]
    ctl = HeadroomController(target=0.7, log=lambda _m: None)
    out, scale = ctl.process_segment(blocks)
    assert scale == 1.0
    assert out[0][0] == 0.3 + 0.3j


def test_headroom_disabled_is_noop() -> None:
    blocks = [np.full(64, 3.0 + 0j)]
    ctl = HeadroomController(target=0.7, enabled=False)
    _, scale = ctl.process_segment(blocks)
    assert scale == 1.0
    assert np.abs(blocks[0]).max() == pytest.approx(3.0)
    same = ctl.process_block(np.full(64, 3.0 + 0j))
    assert np.abs(same).max() == pytest.approx(3.0)


def test_headroom_stream_never_clips_and_only_lowers() -> None:
    ctl = HeadroomController(target=0.7, log=lambda _m: None)
    peaks = [2.0, 1.0, 4.0, 3.0]      # later block demands a lower scale
    prev_scale = ctl.scale
    for p in peaks:
        out = ctl.process_block(np.full(100, p + 0j))
        assert np.abs(out).max() <= 0.7 + 1e-9
        assert ctl.scale <= prev_scale
        prev_scale = ctl.scale


# ======================================================================
# Runner integration: streaming/bounded and file paths stay within target
# ======================================================================
def test_runner_file_output_applies_headroom() -> None:
    # cf32/float full scale is 1.0, so a >=1 peak must be trimmed to the target.
    cfg = SimConfig(fs=1.0e6, duration=0.05, loop=False, block_ms=50.0,
                    headroom=True, headroom_target=0.7,
                    output_format="cf32")
    runner = SimulationRunner(cfg, log=lambda _m: None)
    runner.engine = _FakeEngine(1.5 + 1.5j)  # peak 2.121
    runner.sink = _PeakRecorder()
    runner._segment_seconds = None
    produced = runner._run_engine(0.0)
    assert produced == 50_000
    assert runner.sink.peak <= 0.7 + 1e-6
    assert runner.sink.clips == 0


def test_runner_cs16_normal_scene_is_not_attenuated() -> None:
    """Regression: cs16 full scale is 32767/output_scale (≈3.28), not 1.0.

    A realistic multi-GNSS float peak (≈2.1) is well inside the int16 range,
    so the auto-amp + headroom chain must leave it alone.  Previously the
    hard-coded 1.0 reference scaled it to 0.7 (≈-10 dB) and the ZED-F9P stopped
    acquiring; ``--amp 0.15 --no-headroom`` restored it.
    """
    cfg = SimConfig(fs=1.0e6, duration=0.05, loop=False, block_ms=50.0,
                    headroom=True, headroom_target=0.7)  # cs16, scale 10000
    runner = SimulationRunner(cfg, log=lambda _m: None)
    fake = _FakeEngine(1.5 + 1.5j)  # peak 2.121 < 32767/10000
    runner.engine = fake
    runner.sink = _PeakRecorder()
    runner._segment_seconds = None
    produced = runner._run_engine(0.0)
    assert produced == 50_000
    assert runner.sink.peak == pytest.approx(2.1213, abs=1e-3)
    # The format flag is exposed to the GUI/journal and matches the writer.
    assert runner._full_scale() == pytest.approx(3.2767)


def test_runner_tx_bounded_applies_headroom() -> None:
    cfg = SimConfig(use_usrp=True, fs=1.0e6, duration=0.05, loop=False,
                    block_ms=50.0, headroom=True, headroom_target=0.7)
    runner = SimulationRunner(cfg, log=lambda _m: None)
    runner.engine = _FakeEngine(1.5 + 1.5j)
    runner.sink = _PeakRecorder()
    runner._ram_budget_bytes = 0          # force the bounded queue path
    runner._segment_seconds = None
    produced = runner._run_engine(0.0)
    assert produced == 50_000
    assert runner.sink.peak <= 0.7 + 1e-6
    assert runner.sink.clips == 0


def test_runner_tx_segment_applies_single_headroom_scale() -> None:
    cfg = SimConfig(use_usrp=True, fs=1.0e6, duration=0.0, loop=True,
                    block_ms=1.0, headroom=True, headroom_target=0.7)
    runner = SimulationRunner(cfg, log=lambda _m: None)

    class _StopAfter(_PeakRecorder):
        def write(self, samples) -> None:
            super().write(samples)
            if self.n >= 4000:
                runner.stop()

    runner.engine = _FakeEngine(-2.0 + 0j)   # peak 2.0
    runner.sink = _StopAfter()
    runner._segment_seconds = 0.001          # 1000 samples, 1 block per pass
    runner._run_engine(0.0)
    assert runner.sink.peak == pytest.approx(0.7, abs=1e-6)
    assert runner.sink.clips == 0


def test_runner_level_callback_throttled() -> None:
    calls: list[tuple] = []
    runner = SimulationRunner(
        SimConfig(fs=1.0e6), log=lambda _m: None,
        level=lambda label, p, r, c: calls.append((label, p, r, c)))
    runner._emit_spectrum("TX", np.full(128, 0.5 + 0j))
    runner._emit_spectrum("TX", np.full(128, 0.5 + 0j))  # throttled
    assert len(calls) == 1
    label, p_dbfs, r_dbfs, clips = calls[0]
    assert label == "TX" and p_dbfs == pytest.approx(-6.0206, abs=1e-3)
    assert r_dbfs == pytest.approx(-6.0206, abs=1e-3)
    assert clips == 0


# ======================================================================
# End-to-end: an all-signal composite ends up <= target with no clipping
# ======================================================================
@pytest.mark.skipif(not os.path.exists(_MERGED), reason="merged RINEX missing")
def test_all_signal_composite_peak_within_headroom_target() -> None:
    from gnss_sim.constants import R2D
    from gnss_sim.engine import SignalEngine
    from gnss_sim.gpstime import date2gps
    from gnss_sim.orbit import llh2xyz
    from gnss_sim.rinex import parse_nav_file

    by_sv, iono = parse_nav_file(_MERGED)
    assert by_sv
    xyz = llh2xyz(35.681298 / R2D, 139.766247 / R2D, 10.0)
    start = date2gps(2026, 9, 24, 0, 0, 0)
    cfg = SimConfig(amp_scale=None, headroom=True, headroom_target=0.7)
    runner = SimulationRunner(cfg, log=lambda _m: None)
    engine = SignalEngine(
        by_sv, iono, lambda _g: xyz, start, 2.6e6, center_freq=1575.42e6,
        el_mask=5.0 / R2D, amp_scale=0.15)
    runner._apply_auto_amp(engine)
    # For a cs16 file the full scale is 32767/output_scale (≈3.28), so the
    # automatic scale must stay at the historic nominal — reducing it (the old
    # 1.0 reference) cost ~10 dB and broke the F9P fix.  It only kicks in for
    # genuinely over-range scenes.
    assert engine.amp_scale == pytest.approx(AUTO_AMP_MAX)
    blocks = [engine.generate_block(65536) for _ in range(4)]
    before = combine_levels(blocks)
    ctl = runner._headroom_ctl()   # format-aware full scale
    ctl.process_segment(blocks)
    after = combine_levels(blocks)
    # A normal composite is inside the int16 limit and must not be trimmed.
    assert before.peak > 0.0
    assert before.peak <= runner._full_scale()
    assert after.peak == pytest.approx(before.peak, rel=1e-6)
    assert ctl.scale == pytest.approx(1.0)
    # A genuinely over-range block is still trimmed to the format target.
    hot = [np.full(4096, 4.0 + 0j)]
    ctl2 = runner._headroom_ctl()
    ctl2.target = 0.7 * runner._full_scale()
    ctl2.process_segment(hot)
    assert combine_levels(hot).peak == pytest.approx(ctl2.target, rel=1e-6)


# ======================================================================
# GUI: TX level label + amplitude/headroom controls
# ======================================================================
def test_gui_level_label_updates() -> None:
    from gnss_sim.gui import MainWindow

    app = _app()
    win = MainWindow()
    try:
        assert hasattr(win, "lbl_tx_level")
        win._on_level("TX", -3.1, -13.6, 0)
        txt = win.lbl_tx_level.text()
        assert "пик -3.1 dBFS" in txt and "RMS -13.6 dBFS" in txt
        win._on_level("TX", -3.1, -13.6, 5)
        assert "КЛИП 5" in win.lbl_tx_level.text()
        # Underflow is read back from the active sink.
        win.runner = types.SimpleNamespace(
            sink=types.SimpleNamespace(underflows=3))
        win._on_level("TX", -3.1, -13.6, 0)
        assert "underflow 3" in win.lbl_tx_level.text()
        # The Bridge signal really reaches the label.
        win.runner = None
        win.bridge.level.emit("RX", -10.0, -20.0, 0)
        app.processEvents()
        assert "RX: пик -10.0 dBFS" in win.lbl_tx_level.text()
        # ... and so does a real runner level callback (end-to-end wiring).
        runner = SimulationRunner(SimConfig(), log=lambda _m: None,
                                  level=win.bridge.level.emit)
        runner._emit_spectrum("TX", np.full(64, 0.5 + 0j))
        app.processEvents()
        assert "TX: пик -6.0 dBFS" in win.lbl_tx_level.text()
    finally:
        win.close()
    del app


def test_gui_auto_amp_and_headroom_controls() -> None:
    from gnss_sim.gui import MainWindow

    app = _app()
    win = MainWindow()
    try:
        assert win.cb_amp_auto.isChecked() is True
        assert win.sp_amp.isEnabled() is False
        assert win.cb_headroom.isChecked() is False
        assert win.sp_headroom.value() == pytest.approx(0.7)
        cfg = win._collect()
        assert cfg.amp_scale is None
        assert cfg.headroom is False
        assert cfg.headroom_target == pytest.approx(0.7)
        # Explicit override: uncheck «Авто» and set a single factor.
        win.cb_amp_auto.setChecked(False)
        win.sp_amp.setValue(0.1)
        assert win._collect().amp_scale == pytest.approx(0.1)
        win.cb_headroom.setChecked(True)
        assert win._collect().headroom is True
        # «по умолчанию» restores the automatic defaults.
        win._reset_signals_group()
        assert win.cb_amp_auto.isChecked() is True
        assert win.cb_headroom.isChecked() is False
        assert win._collect().amp_scale is None
    finally:
        win.close()
    del app
