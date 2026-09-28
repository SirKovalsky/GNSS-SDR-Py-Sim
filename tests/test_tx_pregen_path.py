"""Offline tests for the forced single pre-generated RAM TX path.

No hardware, no network::

    .venv\\Scripts\\python.exe -m pytest tests -q

The runner already auto-selects the single pre-generated RAM segment whenever a
finite TX ``duration`` fits the RAM budget.  ``--tx-pregen`` / ``cfg.tx_pregen``
makes that explicit and, when the requested duration does not fit, reduces it to
the largest segment that does instead of falling back to the live
synthesis/transmit path.  These tests pin that selection down without UHD.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from gnss_sim.config import SimConfig  # noqa: E402
from gnss_sim.iqfile import Sink  # noqa: E402
from gnss_sim.runner import SimulationRunner  # noqa: E402


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


def _tx_runner(budget: int, duration: float = 2.0, **kw):
    cfg = SimConfig(use_usrp=True, fs=1.0e6, duration=duration, loop=False,
                    block_ms=100.0, **kw)
    logs: list[str] = []
    runner = SimulationRunner(cfg, log=logs.append)
    runner.engine = _FakeEngine()
    runner.sink = _Recorder()
    runner._ram_budget_bytes = int(budget)
    runner._segment_seconds = None
    return runner, logs


def _small_ram(monkeypatch, avail: int = 4 * 10**9) -> None:
    import gnss_sim.runner as R

    monkeypatch.setattr(R.sysinfo, "available_ram", lambda: avail)
    monkeypatch.setattr(R.sysinfo, "total_ram", lambda: avail + 1)
    monkeypatch.setattr(R.sysinfo, "disk_free", lambda _p: 10**12)


# ======================================================================
# Config / CLI surface
# ======================================================================
def test_tx_pregen_config_and_cli() -> None:
    from gnss_sim.cli import build_parser

    assert SimConfig().tx_pregen is False
    args = build_parser().parse_args(["--tx-pregen"])
    assert args.tx_pregen is True
    assert build_parser().parse_args([]).tx_pregen is False


# ======================================================================
# Path selection in ``_run_tx_stream``
# ======================================================================
def test_forced_pregen_uses_single_ram_segment() -> None:
    runner, logs = _tx_runner(budget=10**9, tx_pregen=True)
    produced = runner._run_engine(0.0)
    assert produced == 2_000_000
    assert runner.sink.n == 2_000_000
    text = "\n".join(logs)
    assert "путь — единый предгенерённый RAM-сегмент" in text
    # One pre-pass: 20 x 0.1 s blocks, then streaming (no interleaved synth).
    assert runner.engine.calls == 20


def test_auto_pregen_when_duration_fits() -> None:
    runner, logs = _tx_runner(budget=10**9)  # no explicit flag
    produced = runner._run_engine(0.0)
    assert produced == 2_000_000
    assert "путь — единый предгенерённый RAM-сегмент" in "\n".join(logs)


def test_forced_pregen_caps_duration_to_ram_budget() -> None:
    # 100 000 B / 8 B = 12 500 complex64 samples = 0.0125 s at 1 Msps.
    runner, logs = _tx_runner(budget=100_000, duration=2.0, tx_pregen=True)
    produced = runner._run_engine(0.0)
    assert produced == 12_500
    assert runner.sink.n == 12_500
    text = "\n".join(logs)
    assert "сокращ" in text                       # logged the reduction
    # Still the single pre-gen path, not the live producer/consumer one.
    assert "единый предгенерённый RAM-сегмент" in text
    assert "ограниченной очередью" not in text


def test_forced_pregen_without_duration_falls_back_to_live() -> None:
    runner, logs = _tx_runner(budget=10**9, duration=0.0, tx_pregen=True)
    runner.stop()
    produced = runner._run_engine(0.0)
    assert produced == 0
    text = "\n".join(logs)
    assert "потоковый синтез с ограниченной очередью" in text
    assert "единый предгенерённый RAM-сегмент" not in text


def test_over_budget_without_flag_uses_live_path() -> None:
    runner, logs = _tx_runner(budget=16000, duration=2.0)  # ~2 000 samples fit
    produced = runner._run_engine(0.0)
    assert produced == 2_000_000
    assert "ограниченной очередью" in "\n".join(logs)


# ======================================================================
# ``_plan`` RAM policy honours the forced pre-gen (cap instead of block)
# ======================================================================
def test_plan_allows_forced_pregen_over_budget(monkeypatch) -> None:
    _small_ram(monkeypatch)
    cfg = SimConfig(use_usrp=True, fs=2.6e6, duration=1200.0, loop=False,
                    tx_pregen=True)
    logs: list[str] = []
    runner = SimulationRunner(cfg, log=logs.append)
    runner._plan()
    assert runner._tx_blocked is False
    assert any("сокращён" in m for m in logs)


def test_plan_still_blocks_over_budget_without_flag(monkeypatch) -> None:
    _small_ram(monkeypatch)
    cfg = SimConfig(use_usrp=True, fs=2.6e6, duration=1200.0, loop=False)
    runner = SimulationRunner(cfg, log=lambda _m: None)
    runner._plan()
    assert runner._tx_blocked is True


# ======================================================================
# The time-continuous live-loop path stays the default for cyclic TX
# ======================================================================
def test_live_loop_path_is_logged() -> None:
    cfg = SimConfig(use_usrp=True, fs=1.0e6, duration=1.0, loop=True)
    logs: list[str] = []
    runner = SimulationRunner(cfg, log=logs.append)
    runner.engine = _FakeEngine()
    runner.sink = _Recorder()
    runner._ram_budget_bytes = 10**9
    runner._segment_seconds = 1.0
    runner.stop()
    produced = runner._run_tx_segment_loop(1000, 100, 0.0)
    assert produced == 0
    assert any("живой цикл" in m for m in logs)
