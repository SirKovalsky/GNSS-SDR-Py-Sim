"""Time-continuous cyclic TX/loop and default-TX-gain regressions.

Covers the fix for the F9P ~400 s fix dropout: a looped pre-generated RAM
segment must not restart GPS time (TOW/HOW) on every pass.  The loop now keeps
synthesising from the same engine, so the GPS week/TOW, the 30 s legacy NAV
sub-frames, the 18 s CNAV-2 frames and the 30 s Galileo I/NAV blocks are
regenerated for the shifted time while code/carrier phase stays continuous.

A non-loop GPS L1 C/A run must stay byte-identical to the verified ``17db96d``
build (hard-coded SHA-256 below); the engine itself is untouched, only the
looping runner path changed.

Headless, no hardware, no network::

    E:\\MySoftware\\SDR_Scan\\.venv\\Scripts\\python.exe -m pytest tests -q
"""

from __future__ import annotations

import hashlib
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from gnss_sim.config import SimConfig, parse_start_time  # noqa: E402
from gnss_sim.constants import R2D  # noqa: E402
from gnss_sim.engine import SignalEngine  # noqa: E402
from gnss_sim.iqfile import NullSink, Sink  # noqa: E402
from gnss_sim.orbit import llh2xyz  # noqa: E402
from gnss_sim.rinex import parse_nav_file  # noqa: E402
from gnss_sim.runner import SimulationRunner  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NAV = os.path.join(_ROOT, "brdc2680.26n")
_START = "2026/09/25,03:00:00"
#: SHA-256 of a seeded, non-loop GPS L1 C/A cs16 file generated with the
#: verified ``17db96d`` code path (10 SVs, 20000 samples, 200 ksps, CPU
#: backend).  Any change to the non-loop synthesis path changes this digest.
_NONLOOP_L1CA_SHA256 = (
    "e1ca1f831bc64791ebbfa33b7bc3a3dd1e37b1c397aaaa9ac359391c3370ad43")


def _app():
    from PyQt5 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _gps_only_cfg(**kw) -> SimConfig:
    base = dict(
        nav_file=_NAV, start_text=_START, lat=60.0, lon=30.0, height=100.0,
        enable_ca=True, enable_l1c=False, enable_galileo=False,
        enable_qzss=False, enable_sbas=False, enable_beidou=False,
        backend="cpu")
    base.update(kw)
    return SimConfig(**base)


# ======================================================================
# 1) Default TX gain = +10 dB (B210 range 0..89.75 dB)
# ======================================================================
def test_default_tx_gain_is_plus10() -> None:
    assert SimConfig().tx_gain == 10.0
    from gnss_sim.cli import build_parser
    assert build_parser().parse_args([]).tx_gain == 10.0
    # An explicit override must still win (CLI and config).
    assert build_parser().parse_args(["--tx-gain", "42.5"]).tx_gain == 42.5
    assert SimConfig(tx_gain=0.0).tx_gain == 0.0


def test_gui_default_tx_gain_is_plus10_and_resets() -> None:
    from gnss_sim.gui import MainWindow
    app = _app()
    win = MainWindow()
    try:
        assert win.sp_txg.value() == 10.0
        win.sp_txg.setValue(33.0)
        assert win._collect().tx_gain == 33.0
        win._reset_uhd_group()
        assert win.sp_txg.value() == 10.0
    finally:
        win.close()
    del app


# ======================================================================
# 2) Non-loop GPS L1 C/A output is byte-identical to ``17db96d``
# ======================================================================
def _nonloop_l1ca_bytes(tmp_path) -> bytes:
    np.random.seed(20240926)
    cfg = _gps_only_cfg(duration=0.1, fs=200000.0, block_ms=100.0,
                        loop=False, output=str(tmp_path / "ref.cs16"),
                        output_format="cs16")
    runner = SimulationRunner(cfg, log=lambda _m: None)
    runner.prepare()
    runner.run()
    assert runner.error is None
    return (tmp_path / "ref.cs16").read_bytes()


def test_nonloop_l1ca_output_byte_identical(tmp_path) -> None:
    data = _nonloop_l1ca_bytes(tmp_path)
    assert len(data) == 20000 * 4          # 0.1 s @ 200 ksps, cs16
    assert hashlib.sha256(data).hexdigest() == _NONLOOP_L1CA_SHA256


class _Capture(Sink):
    def __init__(self) -> None:
        self.parts: list[np.ndarray] = []

    def write(self, samples) -> None:
        self.parts.append(np.asarray(samples).copy())

    def joined(self) -> np.ndarray:
        return np.concatenate(self.parts) if self.parts else np.zeros(0)


def _build_gps_engine() -> SignalEngine:
    by_sv, iono = parse_nav_file(_NAV)
    start = parse_start_time(_START)
    xyz = llh2xyz(60.0 / R2D, 30.0 / R2D, 100.0)
    np.random.seed(20240926)
    return SignalEngine(
        by_sv, iono, lambda _g: xyz, start, 200000.0,
        enable_ca=True, enable_l1c=False, enable_galileo=False,
        enable_qzss=False, enable_sbas=False, enable_beidou=False,
        el_mask=5.0 / R2D, amp_scale=0.15, backend="cpu")


def test_nonloop_runner_matches_engine_directly(tmp_path) -> None:
    """The non-loop runner path is a straight ``generate_block`` loop."""
    np.random.seed(20240926)
    cfg = _gps_only_cfg(duration=0.1, fs=200000.0, block_ms=50.0, loop=False)
    runner = SimulationRunner(cfg, log=lambda _m: None)
    runner.prepare()
    cap = _Capture()
    runner.sink = cap
    assert runner._run_engine(0.0) == 20000

    # Same block cadence as the runner (block = fs * block_ms / 1000 = 10000).
    eng = _build_gps_engine()
    direct = np.concatenate([eng.generate_block(10000) for _ in range(2)])
    assert direct.shape == (20000,)
    assert np.array_equal(cap.joined(), direct)   # bit-identical


# ======================================================================
# 3) Looped generation keeps TOW/HOW monotonic across several boundaries
# ======================================================================
def test_looped_tow_is_monotonic_across_boundaries() -> None:
    cfg = _gps_only_cfg(duration=100.0, fs=20000.0, block_ms=100.0,
                        loop=True, loop_seconds=30.0)
    runner = SimulationRunner(cfg, log=lambda _m: None)
    runner.prepare()
    assert runner._segment_seconds == 30.0
    start_sec = runner.engine.g.sec

    records: list[tuple[float, float, int]] = []
    orig = runner.engine.generate_block

    def rec(n: int) -> np.ndarray:
        b = orig(n)
        g = runner.engine.g  # type: ignore[union-attr]
        frame = max(ch.frame_start.sec for ch in runner.engine.channels
                    if ch.frame_start is not None)  # type: ignore[union-attr]
        records.append((g.sec, frame, n))
        return b

    runner.engine.generate_block = rec  # type: ignore[method-assign]
    runner.sink = NullSink()
    produced = runner._run_engine(0.0)

    # The full requested duration is generated (old replay wrote one segment).
    assert produced == int(100.0 * 20000.0)
    gsec = [r[0] for r in records]
    assert all(b > a for a, b in zip(gsec, gsec[1:]))       # strictly forward
    assert gsec[0] - start_sec == pytest.approx(0.1, abs=0.01)
    assert gsec[-1] - start_sec == pytest.approx(100.0, abs=0.01)
    # No backward TOW jump at any pass boundary: the 30 s frame epoch only ever
    # advances (>= 3 boundaries for a 100 s / 30 s loop).
    frames = [r[1] for r in records]
    assert all(b >= a for a, b in zip(frames, frames[1:]))
    assert len(set(frames)) >= 4
    assert max(frames) - min(frames) >= 90.0


# ======================================================================
# 4) Cyclic TX regenerates each pass (never replays the same RAM buffer)
# ======================================================================
class _CountingEngine:
    """Deterministic engine: every generated sample carries a global index."""

    def __init__(self) -> None:
        self.calls = 0
        self.pos = 0

    def generate_block(self, n: int) -> np.ndarray:
        self.calls += 1
        v = np.arange(self.pos, self.pos + int(n), dtype=np.float64)
        self.pos += int(n)
        return v.astype(np.complex128)


def test_tx_loop_regenerates_passes_no_replay() -> None:
    cfg = SimConfig(use_usrp=True, fs=1.0e6, duration=0.0, loop=True,
                    block_ms=1.0)
    blocks_seen: list[float] = []

    class _Stop(Sink):
        def __init__(self) -> None:
            self.n = 0

        def write(self, samples) -> None:
            a = np.asarray(samples)
            blocks_seen.append(float(a[0].real))
            self.n += int(a.size)
            if self.n >= 15000:                # stop within the third pass
                runner.stop()

    runner = SimulationRunner(cfg, log=lambda _m: None)
    engine = _CountingEngine()
    runner.engine = engine  # type: ignore[assignment]
    runner.sink = _Stop()
    runner._segment_seconds = 0.005            # 5000 samples = 5 blocks/pass
    runner._run_engine(0.0)

    # Every transmitted block is a freshly generated one: no buffer replay.
    assert len(blocks_seen) == len(set(blocks_seen))
    assert engine.calls >= 2                 # engine kept producing new chunks
    assert blocks_seen[-1] > blocks_seen[0]
    # Chunks are split into ~block-sized pieces for the USRP write path.
    assert max(blocks_seen) - min(blocks_seen) >= 15000
