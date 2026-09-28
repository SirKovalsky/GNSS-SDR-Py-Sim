"""Native C++ TX helper integration (no hardware required)."""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

import gnss_sim.native_tx as nt
from gnss_sim.cli import build_parser
from gnss_sim.config import SimConfig


def test_config_has_tx_native_field():
    assert SimConfig().to_dict()["tx_native"] is False
    assert SimConfig(tx_native=True).tx_native is True


def test_cli_flag_tx_native():
    parser = build_parser()
    assert parser.parse_args([]).tx_native is False
    assert parser.parse_args(["--tx-native"]).tx_native is True


def test_native_tx_path_honours_env(tmp_path, monkeypatch):
    exe = tmp_path / "gnss_sim_tx.exe"
    exe.write_bytes(b"MZ")
    monkeypatch.setenv("GNSS_SIM_NATIVE_TX", str(exe))
    assert nt.native_tx_path() == str(exe)
    assert nt.native_tx_available() is True


def test_native_tx_path_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("GNSS_SIM_NATIVE_TX", str(tmp_path / "nope.exe"))
    monkeypatch.setattr(nt, "_REPO_ROOT", str(tmp_path))
    assert nt.native_tx_path() is None
    assert nt.native_tx_available() is False


def test_build_cmd_loop_and_flags(tmp_path):
    cfg = SimConfig(
        fs=25e6, center_freq=1571.328e6, tx_gain=18.0, tx_channel=0,
        tx_antenna="TX/RX", uhd_args="type=b200,serial=000000372",
        clock_source="internal", tx_bandwidth=0.0)
    sink = nt.NativeTxSink(cfg)
    sink.exe = str(tmp_path / "gnss_sim_tx.exe")
    cmd = sink._build_cmd("scene.cs16", loop=True, seconds=60.0)
    assert cmd[0] == sink.exe
    assert "scene.cs16" in cmd
    assert "--loop" in cmd
    assert "--seconds" in cmd
    assert "1571328000.0" in cmd
    assert cfg.uhd_args in cmd
    assert "--format" in cmd and "cs16" in cmd
    # No loop / no fixed duration -> neither flag is emitted.
    cmd2 = sink._build_cmd("scene.cs16", loop=False, seconds=0.0)
    assert "--loop" not in cmd2
    assert "--seconds" not in cmd2


def test_native_sink_rejects_block_writes():
    sink = nt.NativeTxSink(SimConfig())
    with pytest.raises(RuntimeError):
        sink.write(np.zeros(8, dtype=np.complex64))


def test_native_sink_gain_is_remembered():
    sink = nt.NativeTxSink(SimConfig(tx_gain=18.0))
    assert sink.set_tx_gain(42.0) == 42.0
    assert sink.get_info()["native"] is True
    assert sink.get_info()["gain"] == 42.0


# ----------------------------------------------------------------------
# Runner integration (offline): temp cs16 generation + native playback
# ----------------------------------------------------------------------
class _FakeEngine:
    def generate_block(self, n: int) -> np.ndarray:
        return np.full(int(n), 0.1 + 0.1j, dtype=np.complex128)


class _FakeNativeSink:
    def __init__(self) -> None:
        self.exe = "fake_gnss_sim_tx.exe"
        self.path = None
        self.loop = None
        self.seconds = None
        self.size = 0

    def play(self, path, loop, seconds, stop_event=None, on_status=None):
        assert os.path.isfile(path)
        self.path = path
        self.loop = loop
        self.seconds = seconds
        self.size = os.path.getsize(path)
        if on_status is not None:
            on_status(float(seconds))
        return SimpleNamespace(underflows=0, short_sends=0, seq_errors=0,
                               sent_sim_s=float(seconds))

    @property
    def underflows(self):
        return 0


def _native_runner(**kw):
    from gnss_sim.runner import SimulationRunner

    cfg = SimConfig(use_usrp=True, fs=1.0e6, duration=1.0,
                    tx_native=True, output_scale=10000.0, **kw)
    logs: list[str] = []
    runner = SimulationRunner(cfg, log=logs.append)
    runner.engine = _FakeEngine()
    sink = _FakeNativeSink()
    runner.sink = sink
    runner._native_tx = True
    return runner, sink, logs


def test_runner_native_writes_temp_cs16_and_plays():
    runner, sink, logs = _native_runner(loop=False)
    runner._segment_seconds = None
    produced = runner._run_engine(0.0)
    assert produced == 1_000_000
    # 1e6 complex cs16 samples -> 4 MB written to the temp file.
    assert sink.size == 4_000_000
    assert sink.loop is False
    assert abs(sink.seconds - 1.0) < 1e-9
    # Temp file is deleted after playback.
    assert not os.path.exists(sink.path)
    assert any("native" in m.lower() for m in logs)


def test_runner_native_loop_passes_loop_flag():
    runner, sink, _logs = _native_runner(loop=True)
    runner._segment_seconds = 1.0
    runner._run_engine(0.0)
    assert sink.loop is True
    assert sink.seconds == 0.0
