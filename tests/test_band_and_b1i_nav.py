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
from gnss_sim.config import (  # noqa: E402
    BAND_PRESETS, SimConfig, band_preset, compute_combined_band,
    derive_band_key, derive_band_plan,
)
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
    allp = band_preset("all")
    assert l1.center_freq == pytest.approx(1575.42e6)
    assert l1.fs == pytest.approx(2.6e6) and l1.beidou is False
    assert b1i.center_freq == pytest.approx(1561.098e6)
    assert b1i.fs in (4.092e6, 6.138e6) and b1i.beidou is True
    assert b1i.systems == ("C",)
    # Combined stream: centre/fs computed from every enabled system.
    assert allp.center_freq == pytest.approx(1571.328e6, abs=1e4)
    assert allp.fs == pytest.approx(25.0e6, abs=1.0)
    assert allp.tx_ok is True and allp.beidou is True
    assert allp.systems == ("G", "E", "J", "S", "C")
    assert band_preset(None).key == "l1"
    # `wide` is a backward-compatible alias of `all`.
    assert band_preset("wide").key == "all"
    assert band_preset("WIDE").key == "all"
    assert set(BAND_PRESETS) == {"l1", "b1i", "all"}
    assert SimConfig().band == "l1"
    assert SimConfig().b1i_data == "d1"
    assert SimConfig().fs_override is False
    assert SimConfig().center_override is False


def test_compute_combined_band_conservative_default() -> None:
    """No signal group widens automatically; only ``preserve_boc`` opts in."""
    # All five enabled, conservative default: the L1 group stays narrow
    # (±1.023 MHz) and NO ~25 Msps stream is produced silently.
    plan = compute_combined_band()
    assert plan.low == pytest.approx(1561.098e6 - 2.046e6)
    assert plan.high == pytest.approx(1575.42e6 + 1.023e6)
    assert plan.center_freq == pytest.approx((plan.low + plan.high) / 2.0)
    assert plan.fs == pytest.approx(20.0e6, abs=1.0)
    assert plan.fs < 25.0e6
    # Explicit opt-in preserves the BOC(6,1) side lobes -> ~1571.33 / 25 Msps.
    wide = compute_combined_band(preserve_boc=True)
    assert wide.low == pytest.approx(1561.098e6 - 2.046e6)
    assert wide.high == pytest.approx(1575.42e6 + 8.184e6)
    assert wide.center_freq == pytest.approx(1571.328e6, abs=1e4)
    assert wide.fs == pytest.approx(25.0e6, abs=1.0)
    assert wide.span == pytest.approx(24.552e6, abs=1.0)
    # A single B1I-only stream stays narrow.
    b1i = compute_combined_band(
        enable_ca=False, enable_l1c=False, enable_galileo=False,
        enable_qzss=False, enable_sbas=False, enable_beidou=True)
    assert b1i.center_freq == pytest.approx(1561.098e6)
    assert b1i.fs == pytest.approx(4.092e6, abs=1.0)
    # L1 only (no B1I) does not include the 1561 MHz band and stays at 2.6.
    l1 = compute_combined_band(enable_beidou=False)
    assert l1.low == pytest.approx(1575.42e6 - 1.023e6)
    assert l1.center_freq == pytest.approx(1575.42e6)
    assert l1.fs == pytest.approx(2.6e6, abs=1.0)
    # Respects B210 limits.
    for plan in (compute_combined_band(), wide, b1i, l1):
        assert 2.6e6 <= plan.fs <= 56.0e6


def test_derive_band_plan_conservative_banding() -> None:
    """derive_band_plan matches the runner's conservative band resolution."""
    # All systems, no opt-in -> narrow L1 (2.6 / 1575.42), B1I will be dropped.
    assert derive_band_key() == "l1"
    l1 = derive_band_plan()
    assert l1.fs == pytest.approx(2.6e6)
    assert l1.center_freq == pytest.approx(1575.42e6)
    # B1I only -> narrow 4.092 / 1561.098.
    assert derive_band_key(
        enable_ca=False, enable_l1c=False, enable_galileo=False,
        enable_qzss=False, enable_sbas=False, enable_beidou=True) == "b1i"
    b1i = derive_band_plan(
        enable_ca=False, enable_l1c=False, enable_galileo=False,
        enable_qzss=False, enable_sbas=False, enable_beidou=True)
    assert b1i.fs == pytest.approx(4.092e6, abs=1.0)
    assert b1i.center_freq == pytest.approx(1561.098e6)
    # Explicit opt-in -> combined ~25 / 1571.33.
    assert derive_band_key(combine=True) == "all"
    wide = derive_band_plan(combine=True)
    assert wide.fs == pytest.approx(25.0e6, abs=1.0)
    assert wide.center_freq == pytest.approx(1571.328e6, abs=1e4)


def test_compute_combined_band_narrow_ca_only() -> None:
    plan = compute_combined_band(
        enable_ca=True, enable_l1c=False, enable_galileo=False,
        enable_qzss=False, enable_sbas=False, enable_beidou=False)
    assert plan.center_freq == pytest.approx(1575.42e6)
    assert plan.fs == pytest.approx(2.6e6, abs=1.0)


def test_runner_band_l1_drops_b1i_without_widening() -> None:
    cfg = SimConfig(fs=2.6e6, center_freq=1575.42e6, enable_beidou=True,
                    band="l1")
    logs: list[str] = []
    runner = SimulationRunner(cfg, log=logs.append)
    runner._apply_band()
    assert cfg.fs == 2.6e6 and cfg.center_freq == 1575.42e6
    assert cfg.enable_beidou is False
    text = "\n".join(logs)
    assert "отключён" in text and "2.6 Мвыб/с" in text
    assert "--combine" in text and "Объединять" in text


def test_runner_combine_opt_in_uses_combined_stream() -> None:
    """``band=l1`` + ``combine`` is the explicit opt-in to L1+B1I (~25 Msps)."""
    cfg = SimConfig(fs=2.6e6, center_freq=1575.42e6, enable_beidou=True,
                    band="l1", combine=True, use_usrp=True)
    logs: list[str] = []
    SimulationRunner(cfg, log=logs.append)._apply_band()
    assert cfg.enable_beidou is True
    assert cfg.fs == pytest.approx(25.0e6, abs=1.0)
    assert cfg.center_freq == pytest.approx(1571.328e6, abs=1e4)
    text = "\n".join(logs)
    assert "Сессия all" in text
    assert "предгенерация" in text.lower()


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


def test_runner_band_all_computes_combined_stream() -> None:
    cfg = SimConfig(fs=2.6e6, center_freq=1575.42e6, band="all",
                    use_usrp=True)
    logs: list[str] = []
    SimulationRunner(cfg, log=logs.append)._apply_band()
    assert cfg.fs == pytest.approx(25.0e6, abs=1.0)
    assert cfg.center_freq == pytest.approx(1571.328e6, abs=1e4)
    assert cfg.enable_beidou is True
    text = "\n".join(logs)
    assert "Сессия all" in text and "единый поток" in text
    # Explicit combined stream warns about the slow/large pre-generation.
    assert "ВНИМАНИЕ" in text and "предгенерация" in text.lower()


def test_runner_band_all_alias_and_explicit_override() -> None:
    # `wide` is an alias of `all`.
    cfg = SimConfig(fs=2.6e6, center_freq=1575.42e6, band="wide")
    SimulationRunner(cfg, log=lambda m: None)._apply_band()
    assert cfg.fs == pytest.approx(25.0e6, abs=1.0)

    # Explicit -s/-f are honoured (not overwritten by the computed preset),
    # and the B1I auto-band must not fight the explicit narrow settings.
    cfg = SimConfig(fs=10.0e6, center_freq=1570.0e6, band="all",
                    fs_override=True, center_override=True)
    runner = SimulationRunner(cfg, log=lambda m: None)
    runner._apply_band()
    runner._apply_b1i_band()
    assert cfg.fs == pytest.approx(10.0e6)
    assert cfg.center_freq == pytest.approx(1570.0e6)


def test_runner_band_all_subset_only_b1i() -> None:
    cfg = SimConfig(fs=2.6e6, center_freq=1575.42e6, band="all",
                    enable_ca=False, enable_l1c=False, enable_galileo=False,
                    enable_qzss=False, enable_sbas=False)
    SimulationRunner(cfg, log=lambda m: None)._apply_band()
    assert cfg.fs == pytest.approx(4.092e6, abs=1.0)
    assert cfg.center_freq == pytest.approx(1561.098e6)


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
    args = p.parse_args(["--band", "all"])
    assert args.band == "all"
    args = p.parse_args(["--b1i-data", "placeholder"])
    assert args.b1i_data == "placeholder"
    # Explicit combined-stream opt-in is a separate flag (default off).
    assert p.parse_args([]).combine is False
    assert p.parse_args(["--combine"]).combine is True


def test_gui_unified_signal_selection() -> None:
    """«Сигналы» checkboxes are the only system choice; band/fs/nav derive."""
    from gnss_sim.gui import MainWindow
    app = _app()
    win = MainWindow()
    try:
        # Default: all six systems checked but no opt-in -> conservative narrow
        # L1 (2.6 / 1575.42); B1I is dropped (message only, no widening).
        assert win.cb_bds.isChecked() is True
        assert win.cb_combine.isChecked() is False
        cfg = win._collect()
        assert cfg.band == "l1" and cfg.combine is False
        assert cfg.fs == pytest.approx(2.6e6, abs=1.0)
        assert cfg.center_freq == pytest.approx(1575.42e6)
        assert cfg.nav_mode == "merged"
        assert "l1" in win.lbl_band.text()
        assert "2.6" in win.lbl_fs.text()

        # Explicit opt-in «Объединять L1+B1I» -> combined ~25 / 1571.33.
        win.cb_combine.setChecked(True)
        cfg = win._collect()
        assert cfg.band == "all" and cfg.combine is True
        assert cfg.fs == pytest.approx(25.0e6, abs=1.0)
        assert cfg.center_freq == pytest.approx(1571.328e6, abs=1e4)
        assert "all" in win.lbl_band.text() and "25" in win.lbl_fs.text()
        win.cb_combine.setChecked(False)
        assert win._collect().fs == pytest.approx(2.6e6, abs=1.0)

        # BeiDou only -> narrowband b1i, GPS-only systems off.
        for cb in (win.cb_ca, win.cb_l1c, win.cb_gal, win.cb_qzss, win.cb_sbas):
            cb.setChecked(False)
        cfg = win._collect()
        assert cfg.band == "b1i" and cfg.enable_beidou is True
        assert cfg.fs == pytest.approx(4.092e6)
        assert cfg.center_freq == pytest.approx(1561.098e6)
        assert cfg.enable_ca is False and cfg.enable_galileo is False
        assert "b1i" in win.lbl_band.text()

        # BeiDou off + GPS L1 C/A only -> the historic narrow l1 session.
        win.cb_bds.setChecked(False)
        win.cb_ca.setChecked(True)
        cfg = win._collect()
        assert cfg.band == "l1" and cfg.enable_beidou is False
        assert cfg.fs == pytest.approx(2.6e6, abs=1.0)
        assert cfg.center_freq == pytest.approx(1575.42e6)
        assert cfg.nav_mode == "auto"          # GPS-only RINEX source
        assert "l1" in win.lbl_band.text()

        # Enabling Galileo switches the RINEX source note to multi-system.
        win.cb_gal.setChecked(True)
        cfg = win._collect()
        assert cfg.nav_mode == "merged"
        assert "мультисистем" in win.lbl_nav_mode.text().lower()
        # Conservative banding: the CBOC(6,1) side lobes are NOT added
        # automatically, so fs stays at the narrow 2.6 Msps L1 rate.
        assert cfg.fs == pytest.approx(2.6e6, abs=1.0)
        assert cfg.center_freq == pytest.approx(1575.42e6)

        # BeiDou B1I data stays a separate (non-system) choice.
        win.cmb_b1i_data.setCurrentText("placeholder")
        assert win._collect().b1i_data == "placeholder"
        win._reset_band_group()
        assert win.cmb_b1i_data.currentText() == "d1"
        assert win.cb_auto_b1i.isChecked() is True

        # The redundant user-facing selectors are gone.
        assert not hasattr(win, "cmb_band")
        assert not hasattr(win, "cmb_nav_mode")
        assert not hasattr(win, "cmb_fs")
        assert not hasattr(win, "cmb_fc")
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
