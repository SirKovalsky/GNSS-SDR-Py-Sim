"""TX underflow timestamps and the periodic TX health journal.

The 760 s over-air run logged 5 UHD underflows with no timestamp, so their
correlation with three receiver PVT outages was impossible.  These tests lock
in the two diagnostic additions:

1. :class:`~gnss_sim.uhd_tx.UhdTxSink` journals every underflow with the
   simulated stream position (seconds since TX start) and the wall clock, and
   keeps a ``(sim_s, wall_s, count)`` history.
2. :class:`~gnss_sim.runner._TxHealthJournal` emits a throttled line with sim
   time, underflow delta, produced/queued blocks and synth/air realtime rates.

Headless, no hardware, no network::

    E:\\MySoftware\\SDR_Scan\\.venv\\Scripts\\python.exe -m pytest tests -q
"""

from __future__ import annotations

import os
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402

from gnss_sim.config import SimConfig  # noqa: E402
from gnss_sim.iqfile import NullSink  # noqa: E402
from gnss_sim.runner import SimulationRunner, _TxHealthJournal  # noqa: E402
from gnss_sim.uhd_tx import UhdTxSink  # noqa: E402


# ======================================================================
# 1) Underflow timestamps
# ======================================================================
def _bare_sink(fs: float = 1.0e6) -> UhdTxSink:
    """Sink with just the fields :meth:`_send_block` touches (no UHD)."""
    sink = object.__new__(UhdTxSink)
    sink._streamer = None
    sink._underflows = 0
    sink._underflow_events = []
    sink._sent = 0
    sink._max_samps = 4
    sink.sample_rate = fs
    sink._tx_start_wall = None
    sink._log = None
    return sink


def test_underflow_event_records_sim_and_wall_time() -> None:
    logs: list[str] = []

    class _ShortStreamer:
        def send(self, block, md, timeout):
            return max(1, int(block.shape[1]) - 1)  # always one short

    sink = _bare_sink(fs=1.0e6)
    sink._streamer = _ShortStreamer()
    sink._log = logs.append
    md = types.SimpleNamespace(start_of_burst=False, end_of_burst=False)
    sink._send_block(np.zeros(8, dtype=np.complex64), md, None)

    assert sink._underflows >= 1
    events = sink.underflow_events
    assert len(events) == sink._underflows
    # Each event is (sim_s, wall_s, running count); sim time grows from ~0.
    assert all(isinstance(e[0], float) and e[2] == i + 1
               for i, e in enumerate(events))
    assert events[-1][0] >= events[0][0]
    assert sink.underflow_events == events          # property returns a copy
    assert any("TX underflow #" in m and "sim" in m for m in logs)


def test_bare_sink_without_log_does_not_raise() -> None:
    """The unit test above uses object.__new__; missing attrs must be safe."""
    class _ShortStreamer:
        def send(self, block, md, timeout):
            return 0

    sink = _bare_sink()
    sink._streamer = _ShortStreamer()
    md = types.SimpleNamespace(start_of_burst=False, end_of_burst=False)
    sink._send_block(np.zeros(4, dtype=np.complex64), md, None)
    assert sink._underflows >= 1
    assert sink.underflow_events


# ======================================================================
# 2) TX health journal
# ======================================================================
def test_health_journal_line_fields_and_throttle() -> None:
    logs: list[str] = []
    health = _TxHealthJournal(logs.append, interval=3600.0)
    health.tick(12.5, 3, 7, 42, 9.0, 1.001, force=True)
    assert len(logs) == 1
    line = logs[0]
    for token in ("TX health", "sim 12.5", "underflow +3 (всего 3)",
                  "в очереди 7", "синтез 9.00x", "эфир 1.001x"):
        assert token in line, (token, line)
    # Immediately after, without force, the throttle must suppress the line.
    health.tick(13.0, 4, 6, 43, 9.0, 1.001)
    assert len(logs) == 1
    # The next forced line reports the delta since the previous one.
    health.tick(20.0, 5, 2, 50, 9.0, 1.000, force=True)
    assert len(logs) == 2
    assert "underflow +2 (всего 5)" in logs[1]
    assert "выдано 8 бл." in logs[1]


def test_health_journal_tolerates_missing_log() -> None:
    health = _TxHealthJournal(None, interval=0.0)
    health.tick(1.0, 0, 0, 0, 0.0, 0.0, force=True)  # must not raise


# ======================================================================
# 3) Runner integration: health line + underflow-adapted summary
# ======================================================================
class _StopSink(NullSink):
    def __init__(self, runner) -> None:
        super().__init__()
        self.n = 0
        self.underflows = 0
        self._runner = runner

    def write(self, samples) -> None:
        self.n += int(np.asarray(samples).size)
        if self.n >= 3000:
            self._runner.stop()


def test_segment_loop_logs_health_and_transition() -> None:
    cfg = SimConfig(use_usrp=True, fs=1000.0, duration=0.0, loop=True,
                    block_ms=10.0)
    logs: list[str] = []
    runner = SimulationRunner(cfg, log=logs.append)
    runner._ram_budget_bytes = 0

    class _Engine:
        def generate_block(self, n):
            return np.zeros(int(n), dtype=np.complex128)

    runner.engine = _Engine()  # type: ignore[assignment]
    runner.sink = _StopSink(runner)  # type: ignore[assignment]
    runner._segment_seconds = 0.001
    produced = runner._run_tx_segment_loop(1, 100, 0.0)
    assert produced >= 3000
    joined = "\n".join(logs)
    assert "RAM-сегмент исчерпан" in joined
    assert "TX health: sim" in joined


def test_report_underflows_lists_sim_times() -> None:
    class _Sink:
        underflows = 2
        underflow_events = [(12.5, 1000.0, 1), (44.0, 1100.0, 2)]

    logs: list[str] = []
    runner = SimulationRunner(SimConfig(), log=logs.append)
    runner.sink = _Sink()  # type: ignore[assignment]
    runner._report_underflows()
    assert any("ВНИМАНИЕ: 2 underflow" in m for m in logs)
    assert any("12.5" in m and "44.0" in m for m in logs)
