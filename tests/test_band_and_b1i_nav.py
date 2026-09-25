"""Band/session selection and BeiDou B1I D1 navigation message tests.

Offline (no hardware, no network)::

    E:\\MySoftware\\SDR_Scan\\.venv\\Scripts\\python.exe -m pytest tests -q
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from gnss_sim import beidou_nav, rinex  # noqa: E402
from gnss_sim.beidou_nav import (  # noqa: E402
    FRAME_BITS, NH_CODE, PREAMBLE, SUBFRAME_BITS, bch_decode, bch_encode,
    d1_frame_block, d1_subframe, deinterleave, interleave, parse_subframe,
)
from gnss_sim.config import BAND_PRESETS, SimConfig, band_preset  # noqa: E402
from gnss_sim.constants import R2D  # noqa: E402
from gnss_sim.engine import SignalEngine  # noqa: E402
from gnss_sim.gpstime import date2gps  # noqa: E402
from gnss_sim.orbit import llh2xyz  # noqa: E402
from gnss_sim.runner import SimulationRunner  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MERGED = os.path.join(_ROOT, "rinex_cache",
                       "BRDC00IGS_R_20262660000_01D_MN.rnx")


def _app():
    from PyQt5 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


# ======================================================================
# A) Band presets and runner band resolution
# ======================================================================
def test_band_presets_recommended_values() -> None:
    l1 = band_preset("l1")
    b1i = band_preset("b1i")
    wide = band_preset("wide")
    assert l1.center_freq == pytest.approx(1575.42e6)
    assert l1.fs == pytest.approx(2.6e6) and l1.beidou is False
    assert b1i.center_freq == pytest.approx(1561.098e6)
    assert b1i.fs in (4.092e6, 6.138e6) and b1i.beidou is True
    assert b1i.systems == ("C",)
    assert wide.center_freq == pytest.approx(1568.0e6)
    assert wide.fs == pytest.approx(30.0e6) and wide.tx_ok is False
    assert band_preset(None).key == "l1"
    assert band_preset("WIDE").key == "wide"
    assert set(BAND_PRESETS) == {"l1", "b1i", "wide"}
    assert SimConfig().band == "l1"
    assert SimConfig().b1i_data == "d1"


def test_runner_band_l1_drops_b1i_without_widening() -> None:
    cfg = SimConfig(fs=2.6e6, center_freq=1575.42e6, enable_beidou=True,
                    band="l1")
    logs: list[str] = []
    runner = SimulationRunner(cfg, log=logs.append)
    runner._apply_band()
    assert cfg.fs == 2.6e6 and cfg.center_freq == 1575.42e6
    assert cfg.enable_beidou is False
    assert any("переключает fs на 30" in m for m in logs)


def test_runner_band_b1i_sets_narrowband_beidou_only() -> None:
    cfg = SimConfig(fs=2.6e6, center_freq=1575.42e6, band="b1i")
    logs: list[str] = []
    SimulationRunner(cfg, log=logs.append)._apply_band()
    assert cfg.fs == pytest.approx(4.092e6)
    assert cfg.center_freq == pytest.approx(1561.098e6)
    assert cfg.enable_beidou is True
    assert not (cfg.enable_ca or cfg.enable_l1c or cfg.enable_galileo
                or cfg.enable_qzss or cfg.enable_sbas)
    assert any("Сессия b1i" in m for m in logs)


def test_runner_band_wide_warns_for_tx() -> None:
    cfg = SimConfig(fs=2.6e6, center_freq=1575.42e6, band="wide",
                    use_usrp=True)
    logs: list[str] = []
    SimulationRunner(cfg, log=logs.append)._apply_band()
    assert cfg.fs == pytest.approx(30.0e6)
    assert cfg.center_freq == pytest.approx(1568.0e6)
    assert cfg.enable_beidou is True
    assert any("ВНИМАНИЕ" in m and "wide" in m for m in logs)


def test_runner_band_unknown_falls_back_to_l1() -> None:
    cfg = SimConfig(fs=2.6e6, center_freq=1575.42e6, band="bogus")
    logs: list[str] = []
    SimulationRunner(cfg, log=logs.append)._apply_band()
    assert any("Неизвестный диапазон" in m for m in logs)


def test_cli_band_sets_radio_and_allows_override() -> None:
    from gnss_sim.cli import build_parser
    p = build_parser()
    args = p.parse_args([])
    assert args.band == "l1" and args.sample_rate is None
    assert args.center_freq is None
    args = p.parse_args(["--band", "b1i"])
    assert args.band == "b1i"
    args = p.parse_args(["--band", "wide", "-s", "10e6"])
    assert args.sample_rate == pytest.approx(10e6)
    args = p.parse_args(["--b1i-data", "placeholder"])
    assert args.b1i_data == "placeholder"


def test_gui_band_controls() -> None:
    from gnss_sim.gui import MainWindow
    app = _app()
    win = MainWindow()
    try:
        assert win.cmb_band.currentData() == "l1"
        # Default l1 session: B1I off, fs/centre are the L1 defaults.
        assert win.cb_bds.isChecked() is False
        assert win._collect().band == "l1"

        win.cmb_band.setCurrentIndex(1)        # b1i
        cfg = win._collect()
        assert cfg.band == "b1i" and cfg.fs == pytest.approx(4.092e6)
        assert cfg.center_freq == pytest.approx(1561.098e6)
        assert win.cb_bds.isChecked() is True
        assert win.cb_ca.isChecked() is False
        assert "B1I" in win.lbl_band.text()

        win.cmb_band.setCurrentIndex(2)        # wide
        cfg = win._collect()
        assert cfg.band == "wide" and cfg.fs == pytest.approx(30.0e6)
        assert win.cb_bds.isChecked() is True
        assert "wide" in win.lbl_band.text().lower() or "Все" in win.lbl_band.text()

        win.cmb_b1i_data.setCurrentText("placeholder")
        assert win._collect().b1i_data == "placeholder"
        win._reset_band_group()
        assert win.cmb_band.currentData() == "l1"
        assert win.cmb_b1i_data.currentText() == "d1"
        assert win.cb_auto_b1i.isChecked() is True
    finally:
        win.close()
    del app


# ======================================================================
# B) B1I D1 navigation message
# ======================================================================
def test_beidou_nav_self_test() -> None:
    beidou_nav.self_test()


def test_nh_and_preamble_constants() -> None:
    assert NH_CODE.tolist() == [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1,
                                0, 0, 1, 1, 1, 0]
    assert PREAMBLE == "11100010010"


def test_bch_encodes_and_corrects() -> None:
    info = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0]
    cw = bch_encode(info)
    assert len(cw) == 15 and cw[:11] == info
    for pos in range(15):
        bad = list(cw)
        bad[pos] ^= 1
        fixed, corrected = bch_decode(bad)
        assert corrected and fixed == cw


def test_interleaver_round_trip() -> None:
    c1 = [1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0]
    c2 = [0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1]
    word = interleave(c1, c2)
    assert len(word) == 30
    assert word[0] == c1[0] and word[1] == c2[0]
    assert deinterleave(word) == (c1, c2)


@pytest.mark.skipif(not os.path.exists(_MERGED), reason="merged cache missing")
def test_b1i_d1_round_trip_real_ephemeris() -> None:
    by_sv, _iono = rinex.parse_nav_file(_MERGED)
    keys = sorted(k for k in by_sv if k.startswith("C"))
    assert keys, "no BeiDou ephemerides in merged cache"
    eph = by_sv[keys[0]][0]
    frame = d1_frame_block(eph, 259200.0)
    assert frame.shape == (FRAME_BITS,)
    assert set(np.unique(frame).tolist()) == {0, 1}

    sf1 = parse_subframe(frame[:300])
    sf2 = parse_subframe(frame[300:600])
    sf3 = parse_subframe(frame[600:900])
    assert sf1["subframe_id"] == 1
    assert sf1["preamble"] == PREAMBLE
    assert sf1["sow"] == 259200
    assert sf1["wn"] == eph.toe.week
    assert sf1["aode"] == eph.iode and sf1["aodc"] == eph.iodc
    assert sf2["deltan"] == pytest.approx(eph.deltan, abs=1e-13)
    assert sf2["sqrta"] == pytest.approx(eph.sqrta, abs=2e-6)
    assert sf3["i0"] == pytest.approx(eph.inc0, abs=2e-9)
    assert sf3["omg0"] == pytest.approx(eph.omg0, abs=2e-9)
    toe = (int(sf2["toe_hi"]) << 15) | int(sf3["toe_lo"])
    assert toe == int(round(eph.toe.sec / 8.0))


@pytest.mark.skipif(not os.path.exists(_MERGED), reason="merged cache missing")
def test_engine_b1i_d1_and_placeholder() -> None:
    by_sv, iono = rinex.parse_nav_file(_MERGED)
    start = date2gps(2026, 9, 23, 12, 0, 0)
    xyz = llh2xyz(35.681298 / R2D, 139.766247 / R2D, 10.0)
    common = dict(
        enable_ca=False, enable_l1c=False, enable_galileo=False,
        enable_qzss=False, enable_sbas=False, enable_beidou=True,
        el_mask=5.0 / R2D, backend="cpu")

    eng = SignalEngine(by_sv, iono, lambda g: xyz, start, 4.092e6,
                       center_freq=1561.098e6, b1i_data="d1", **common)
    b1i = [c for c in eng.channels if c.kind == "beidou"]
    assert b1i and all(c.b1i_bits is not None
                       and c.b1i_bits.shape == (FRAME_BITS,) for c in b1i)
    # Real D1 data is a genuine bit stream, not constant.
    assert set(np.unique(b1i[0].b1i_bits).tolist()) == {0, 1}
    block = eng.generate_block(4092)
    assert block.shape == (4092,) and np.any(block)

    eng2 = SignalEngine(by_sv, iono, lambda g: xyz, start, 4.092e6,
                        center_freq=1561.098e6, b1i_data="placeholder",
                        **common)
    assert all(c.b1i_bits is None for c in eng2.channels
               if c.kind == "beidou")
    assert np.any(eng2.generate_block(4092))


@pytest.mark.skipif(not os.path.exists(_MERGED), reason="merged cache missing")
def test_engine_b1i_frame_rolls_every_30s() -> None:
    by_sv, iono = rinex.parse_nav_file(_MERGED)
    start = date2gps(2026, 9, 23, 12, 0, 0)
    xyz = llh2xyz(35.681298 / R2D, 139.766247 / R2D, 10.0)
    eng = SignalEngine(by_sv, iono, lambda g: xyz, start, 4.092e6,
                       center_freq=1561.098e6, b1i_data="d1",
                       enable_ca=False, enable_l1c=False, enable_galileo=False,
                       enable_qzss=False, enable_sbas=False, enable_beidou=True,
                       el_mask=5.0 / R2D, backend="cpu")
    ch = next(c for c in eng.channels if c.kind == "beidou")
    old_start = ch.b1i_frame_start
    old_bits = ch.b1i_bits.copy()
    eng.g.sec += 30.0
    eng._roll_frames()
    assert ch.b1i_frame_start.sec == old_start.sec + 30.0
    assert ch.b1i_bits.shape == old_bits.shape
