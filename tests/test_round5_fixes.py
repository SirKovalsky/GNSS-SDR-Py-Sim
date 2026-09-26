"""Round-5 fixes: TX underflow margin, IQ-replay warning, visible-vs-GSV.

Covers the user-approved work items:

1. TX priority/jitter: the synthesis producer gets an OS priority boost and the
   producer/consumer jitter buffer is enlarged (20 s auto, RAM-capped,
   ``--tx-jitter`` overrides).
2. A reused IQ file is replayed as recorded — its embedded TOW/HOW cannot be
   time-shifted; the runner now says so explicitly.
6. The simulator's visible-SV list is compared against the receiver's **GSV**
   list (never GSA), both as a pure function and in the GUI monitor path.

Headless, no hardware, no network::

    E:\\MySoftware\\SDR_Scan\\.venv\\Scripts\\python.exe -m pytest tests -q
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from gnss_sim.config import SimConfig  # noqa: E402
from gnss_sim.iqfile import FORMAT_BYTES, NullSink  # noqa: E402
from gnss_sim.rtprio import boost_thread_priority  # noqa: E402
from gnss_sim.runner import SimulationRunner, _tx_jitter_seconds  # noqa: E402
from gnss_sim.ublox import compare_visible, describe_visible  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NAV = os.path.join(_ROOT, "rinex_cache", "BRDC00IGS_R_20262670000_01D_MN.rnx")


# ======================================================================
# 1) TX priority + jitter buffer
# ======================================================================
def test_default_tx_jitter_is_large_and_configurable() -> None:
    assert _tx_jitter_seconds(SimConfig(fs=2.6e6), 0) == 20.0
    assert _tx_jitter_seconds(SimConfig(fs=2.6e6, tx_jitter_seconds=7.5), 0) == 7.5
    # Never below 2 s even for a tiny budget.
    assert _tx_jitter_seconds(SimConfig(fs=2.6e6), 1) >= 2.0
    # The bounded (non-loop) path uses a smaller default.
    assert _tx_jitter_seconds(SimConfig(fs=2.6e6), 0, default=4.0) == 4.0


def test_tx_jitter_is_capped_by_ram_budget() -> None:
    cfg = SimConfig(fs=25.0e6)
    # cap = 0.25 * budget / 8 / fs.  1 GB @25 Msps -> 1.25 s -> floor 2 s.
    assert _tx_jitter_seconds(cfg, int(1e9)) == 2.0
    # 4 GB @25 Msps -> 5.0 s.
    assert _tx_jitter_seconds(cfg, int(4e9)) == pytest.approx(5.0)
    # 64 GB @25 Msps -> cap 80 s > default 20 s, so the default wins.
    assert _tx_jitter_seconds(cfg, int(64e9)) == 20.0


def test_boost_thread_priority_never_raises() -> None:
    assert isinstance(boost_thread_priority(), bool)


def test_plan_single_segment_for_tx_when_duration_fits(monkeypatch) -> None:
    """GUI default (2.6 Msps, ample RAM) -> ONE segment, hence no TOW reset.

    This is the key to the ~400 s diagnosis: for the default narrow L1 session
    with enough RAM, ``_plan`` selects the full requested duration as the single
    segment, so the producer/consumer boundary (and any historic TOW reset) is
    never reached at all.
    """
    import gnss_sim.runner as R
    monkeypatch.setattr(R.sysinfo, "available_ram", lambda: 60 * 10**9)
    monkeypatch.setattr(R.sysinfo, "total_ram", lambda: 68 * 10**9)
    monkeypatch.setattr(R.sysinfo, "disk_free", lambda _p: 10**12)
    logs: list[str] = []
    cfg = SimConfig(use_usrp=True, fs=2.6e6, duration=1200.0, loop=True)
    runner = SimulationRunner(cfg, log=logs.append)
    runner._plan()
    assert runner._segment_seconds == 1200.0
    assert any("одним сегментом" in m for m in logs)


def test_plan_caps_tx_segment_to_multiple_of_90(monkeypatch) -> None:
    import gnss_sim.runner as R
    monkeypatch.setattr(R.sysinfo, "available_ram", lambda: 4 * 10**9)
    monkeypatch.setattr(R.sysinfo, "total_ram", lambda: 16 * 10**9)
    monkeypatch.setattr(R.sysinfo, "disk_free", lambda _p: 10**12)
    cfg = SimConfig(use_usrp=True, fs=2.6e6, duration=1200.0, loop=True)
    runner = SimulationRunner(cfg, log=lambda _m: None)
    runner._plan()
    # 0.6 * 4 GB / 8 B / 2.6 Msps ~= 115 s -> capped, rounded down to 90 s.
    assert runner._segment_seconds == 90.0



def test_loop_segment_path_uses_time_continuous_tx(monkeypatch) -> None:
    """TX+loop must dispatch to ``_run_tx_segment_loop`` (not a RAM replay)."""
    cfg = SimConfig(use_usrp=True, fs=1.0e6, duration=100.0, loop=True)
    runner = SimulationRunner(cfg, log=lambda _m: None)

    class _Engine:
        def generate_block(self, n):
            return np.zeros(int(n), dtype=np.complex128)

    runner.engine = _Engine()  # type: ignore[assignment]
    runner.sink = NullSink()
    runner._segment_seconds = 1.0
    seen = {}

    def stub(seg_total, block, t0):
        seen["seg"] = seg_total
        return 0

    monkeypatch.setattr(runner, "_run_tx_segment_loop", stub)
    runner._run_engine(0.0)
    assert seen["seg"] == 1_000_000


def test_tx_loop_jitter_buffer_holds_multiple_seconds() -> None:
    """The producer may run ahead by the whole (default) jitter buffer."""
    cfg = SimConfig(use_usrp=True, fs=1000.0, duration=0.0, loop=True,
                    block_ms=10.0)
    runner = SimulationRunner(cfg, log=lambda _m: None)
    runner._ram_budget_bytes = 0

    class _Engine:
        def __init__(self):
            self.received = None
        def generate_block(self, n):
            return np.zeros(int(n), dtype=np.complex128)

    class _Stop(NullSink):
        def __init__(self):
            super().__init__()
            self.n = 0
        def write(self, samples):
            self.n += int(np.asarray(samples).size)
            if self.n >= 3000:
                runner.stop()

    runner.engine = _Engine()  # type: ignore[assignment]
    runner.sink = _Stop()
    runner._segment_seconds = 0.001        # 1 sample-less tiny segment
    # Must not hang and must stream continuously.
    produced = runner._run_tx_segment_loop(1, 100, 0.0)
    assert produced >= 3000


# ======================================================================
# 2) IQ-replay time shift is impossible -> explicit warning
# ======================================================================
def test_iq_input_logs_time_shift_warning(tmp_path) -> None:
    path = tmp_path / "replay.cs16"
    path.write_bytes(np.zeros(64, dtype="<i2").tobytes())
    cfg = SimConfig(iq_input=str(path), fs=1.0e6, duration=0.0,
                    start_text="2026/09/25,12:00:00")
    logs: list[str] = []
    runner = SimulationRunner(cfg, log=logs.append)
    runner.prepare()
    joined = "\n".join(logs)
    assert "ВНИМАНИЕ" in joined
    assert "TOW/HOW" in joined or "не сдвигается" in joined
    assert FORMAT_BYTES["cs16"] == 4


# ======================================================================
# 6) visible-SV vs GSV comparison
# ======================================================================
def test_compare_visible_matches_and_diffs() -> None:
    sim = [{"system": "G", "prn": 5, "name": "G05"},
           {"system": "E", "prn": 8, "name": "E08"},
           {"system": "S", "prn": 120, "name": "S120"}]
    gsv = [{"system": "G", "prn": 5}, {"system": "E", "prn": 8},
           {"system": "G", "prn": 31}]
    cmp = compare_visible(sim, gsv)
    assert cmp["matched"] == [("E", 8), ("G", 5)]
    assert cmp["not_in_gsv"] == [("S", 120)]
    assert cmp["not_in_sim"] == [("G", 31)]
    assert cmp["sim_count"] == 3 and cmp["gsv_count"] == 3
    assert "нет в GSV" in describe_visible(cmp)
    assert "S120" in describe_visible(cmp)


def test_compare_visible_uses_name_when_system_missing() -> None:
    cmp = compare_visible([{"name": "S126", "prn": 126}], [])
    assert cmp["not_in_gsv"] == [("S", 126)]
    # Junk entries are ignored, not fatal.
    assert compare_visible([{"prn": None}], [{"system": "G", "prn": "x"}])[
        "matched"] == []


def test_gui_monitor_logs_visible_comparison() -> None:
    from PyQt5 import QtWidgets
    from gnss_sim.gui import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = MainWindow()
    try:
        class _Engine:
            channel_info = [
                {"system": "G", "prn": 5, "name": "G05"},
                {"system": "E", "prn": 8, "name": "E08"},
            ]
        class _Runner:
            engine = _Engine()
        win.runner = _Runner()  # type: ignore[assignment]
        win._vis_last = 0.0
        win._log_visible_comparison([{"system": "G", "prn": 5}])
        text = win.log.toPlainText()
        assert "нет в GSV" in text and "E08" in text
    finally:
        win.close()
    del app


@pytest.mark.skipif(not os.path.exists(_NAV), reason="cached merged RINEX")
def test_rinex_galileo_bgd_e5b_is_parsed(tmp_path) -> None:
    """RINEX Galileo orbit-6 field 4 (BGD E5b/E1) reaches ``bgd_e5b``.

    The cached file stores 0.0 there, so patch a record with a distinct value
    and check the parser keeps it (the old parser truncated it into ``iodc``).
    """
    from gnss_sim.rinex import parse_nav_file

    text = open(_NAV, "r", encoding="utf-8", errors="replace").read()
    lines = text.splitlines()
    hdr = next(i for i, ln in enumerate(lines)
               if ln[60:80].rstrip() == "END OF HEADER")
    # Locate the first Galileo record and patch its orbit-6 field 4.
    idx = next(i for i in range(hdr + 1, len(lines))
               if lines[i][:1] == "E" and lines[i][1:3].strip().isdigit())
    patched = f"{1.6e-9:19.12E}"
    ln = lines[idx + 6]
    lines[idx + 6] = (ln[:4 + 3 * 19] + patched + ln[4 + 4 * 19:])
    mod = tmp_path / "gal_bgd.rnx"
    mod.write_text("\n".join(lines) + "\n", encoding="utf-8")
    by_sv, _ = parse_nav_file(str(mod))
    ephs = [e for k, v in by_sv.items() if k.startswith("E") for e in v]
    assert ephs
    bgds = [e.bgd_e5b for e in ephs if e.prn == int(lines[idx][1:3])]
    assert bgds and abs(bgds[0] - 1.6e-9) <= 1e-12

