"""Band/session selection and BeiDou B1I D1 navigation message tests.

Offline (no hardware, no network)::

    .venv\\Scripts\\python.exe -m pytest tests -q
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
from gnss_sim.constants import BDT_GPST_OFFSET_S, BDT_WEEK_OFFSET  # noqa: E402
from gnss_sim.constants import SPEED_OF_LIGHT  # noqa: E402
from gnss_sim.engine import (  # noqa: E402
    SignalEngine, b1i_frame_epoch, gps2bdt_sec, _Geo,
)
from gnss_sim.gpstime import GpsTime, date2gps, sub_gps_time  # noqa: E402
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
    # Explicit opt-in -> combined ~25 / 1571.33 (B1I must be enabled first).
    assert derive_band_key(combine=True, enable_beidou=True) == "all"
    assert derive_band_key(combine=True) == "l1"  # B1I off by default
    wide = derive_band_plan(combine=True, enable_beidou=True)
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
    cfg = SimConfig(fs=2.6e6, center_freq=1575.42e6, band="b1i",
                    enable_beidou=True)
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
                    enable_beidou=True, use_usrp=True)
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
    # `wide` is an alias of `all` (B1I must be explicitly enabled).
    cfg = SimConfig(fs=2.6e6, center_freq=1575.42e6, band="wide",
                    enable_beidou=True)
    SimulationRunner(cfg, log=lambda m: None)._apply_band()
    assert cfg.fs == pytest.approx(25.0e6, abs=1.0)

    # Explicit -s/-f are honoured (not overwritten by the computed preset),
    # and the B1I auto-band must not fight the explicit narrow settings.
    cfg = SimConfig(fs=10.0e6, center_freq=1570.0e6, band="all",
                    enable_beidou=True, fs_override=True, center_override=True)
    runner = SimulationRunner(cfg, log=lambda m: None)
    runner._apply_band()
    runner._apply_b1i_band()
    assert cfg.fs == pytest.approx(10.0e6)
    assert cfg.center_freq == pytest.approx(1570.0e6)


def test_runner_band_all_subset_only_b1i() -> None:
    cfg = SimConfig(fs=2.6e6, center_freq=1575.42e6, band="all",
                    enable_beidou=True,
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
    # B1I is an explicit CLI opt-in now (default off); --no-beidou is gone.
    assert p.parse_args([]).beidou is False
    assert p.parse_args(["--beidou"]).beidou is True
    assert not hasattr(p.parse_args([]), "no_beidou")
    # Explicit combined-stream opt-in is a separate flag (default off).
    assert p.parse_args([]).combine is False
    assert p.parse_args(["--combine"]).combine is True


def test_default_b1i_disabled_keeps_band_l1() -> None:
    """Default config: B1I off -> derived band stays l1, no B1I channels."""
    from gnss_sim.config import derive_band_plan
    assert SimConfig().enable_beidou is False
    assert SimConfig().b1i_data == "d1"
    assert derive_band_key() == "l1"
    assert derive_band_key(combine=True) == "l1"
    plan = derive_band_plan()
    assert plan.center_freq == pytest.approx(1575.42e6)
    assert plan.fs == pytest.approx(2.6e6, abs=1.0)
    # The narrow L1 plan contains no 1561.098 MHz B1I contribution.
    assert plan.low == pytest.approx(1575.42e6 - 1.023e6)


def test_runner_band_b1i_falls_back_when_b1i_off() -> None:
    """A b1i/all session without the explicit opt-in falls back to l1."""
    cfg = SimConfig(fs=4.092e6, center_freq=1561.098e6, band="b1i")
    logs: list[str] = []
    SimulationRunner(cfg, log=logs.append)._apply_band()
    assert cfg.enable_beidou is False
    assert cfg.band == "l1"
    assert cfg.fs == pytest.approx(2.6e6)
    assert cfg.center_freq == pytest.approx(1575.42e6)
    assert any("отключён" in m for m in logs)

    allp = SimConfig(fs=2.6e6, center_freq=1575.42e6, band="all")
    SimulationRunner(allp, log=lambda m: None)._apply_band()
    assert allp.enable_beidou is False
    assert allp.fs == pytest.approx(2.6e6)
    assert allp.center_freq == pytest.approx(1575.42e6)


def test_gui_unified_signal_selection() -> None:
    """«Сигналы» checkboxes are the only system choice; band/fs/nav derive.

    BeiDou B1I selection is temporarily disabled: the checkbox is unchecked and
    greyed out and the band stays on the historic L1 session.
    """
    from gnss_sim.gui import MainWindow
    app = _app()
    win = MainWindow()
    try:
        # Default: GPS/Galileo/QZSS/SBAS checked, BeiDou B1I disabled/unchecked
        # -> narrow L1 (2.6 / 1575.42).
        assert win.cb_bds.isChecked() is False
        assert win.cb_bds.isEnabled() is False
        assert "отключено" in win.cb_bds.text().lower()
        assert "временно" in win.cb_bds.toolTip().lower()
        assert win.cb_combine.isChecked() is False
        assert not win.cb_combine.isEnabled()
        assert not win.cmb_b1i_data.isEnabled()
        assert not win.cb_auto_b1i.isEnabled()
        cfg = win._collect()
        assert cfg.band == "l1" and cfg.combine is False
        assert cfg.enable_beidou is False
        assert cfg.fs == pytest.approx(2.6e6, abs=1.0)
        assert cfg.center_freq == pytest.approx(1575.42e6)
        assert cfg.nav_mode == "merged"
        assert "l1" in win.lbl_band.text()
        assert "2.6" in win.lbl_fs.text()

        # Programmatic toggles cannot select B1I any more: the greyed checkbox
        # must not affect the derived band or the auto-B1I controls.
        win.cb_bds.setChecked(True)
        win.cb_combine.setChecked(True)
        cfg = win._collect()
        assert cfg.band == "l1" and cfg.enable_beidou is False
        assert cfg.fs == pytest.approx(2.6e6, abs=1.0)
        assert win.cb_combine.isChecked() is False
        assert not win.cb_combine.isEnabled()

        # GPS L1 C/A only -> the historic narrow l1 session, GPS-only RINEX.
        for cb in (win.cb_l1c, win.cb_gal, win.cb_qzss, win.cb_sbas):
            cb.setChecked(False)
        cfg = win._collect()
        assert cfg.band == "l1" and cfg.enable_beidou is False
        assert cfg.fs == pytest.approx(2.6e6, abs=1.0)
        assert cfg.center_freq == pytest.approx(1575.42e6)
        assert cfg.nav_mode == "auto"          # GPS-only RINEX source

        # Enabling Galileo switches the RINEX source note to multi-system.
        win.cb_gal.setChecked(True)
        cfg = win._collect()
        assert cfg.nav_mode == "merged"
        assert "мультисистем" in win.lbl_nav_mode.text().lower()
        # Conservative banding: the CBOC(6,1) side lobes are NOT added
        # automatically, so fs stays at the narrow 2.6 Msps L1 rate.
        assert cfg.fs == pytest.approx(2.6e6, abs=1.0)
        assert cfg.center_freq == pytest.approx(1575.42e6)

        # BeiDou B1I data stays a separate (non-system) choice, still resettable.
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


def _make_b1i_engine() -> SignalEngine:
    by_sv, iono = rinex.parse_nav_file(_MERGED)
    start = date2gps(2026, 9, 23, 12, 0, 0)
    xyz = llh2xyz(35.681298 / R2D, 139.766247 / R2D, 10.0)
    return SignalEngine(
        by_sv, iono, lambda g: xyz, start, 4.092e6,
        center_freq=1561.098e6, b1i_data="d1",
        enable_ca=False, enable_l1c=False, enable_galileo=False,
        enable_qzss=False, enable_sbas=False, enable_beidou=True,
        el_mask=5.0 / R2D, backend="cpu")


@pytest.mark.skipif(not os.path.exists(_MERGED), reason="merged cache missing")
def test_engine_b1i_nh20_follows_code_wraps() -> None:
    """NH20 and D1 bit boundaries must sit exactly on the code-period wraps.

    Regression: the NH/bit index used to be derived from the transmit second
    alone (no ``rng/c`` and no code-phase term), so every NH transition was
    ``frac(rng/c)`` ms away from the code epochs.  With a ~120 ms pseudorange
    that is ~0.56 ms - half a code period - which chops the 20 ms secondary
    code correlation and leaves the receiver at C/N0 ~10 dB with no message.
    """
    from gnss_sim.beidou_nav import NH_CODE

    eng = _make_b1i_engine()
    ch = next(c for c in eng.channels if c.kind == "beidou")
    ch.ca = np.ones_like(ch.ca)                    # constant primary code
    ch.b1i_bits = np.zeros(1500, dtype=np.int8)    # constant +1 D1 data
    g = ch.b1i_frame_start
    rho = _Geo(2.1e7, 0.0, 2.1e7, (0.0, 1.0), 0.0)
    f_code = 2.046e6
    period = 2046.0 / f_code
    base_ms = (sub_gps_time(g, ch.b1i_frame_start)
               - rho.rng / SPEED_OF_LIGHT) * 1000.0
    # Sample exactly on 40 consecutive code epochs (one NH chip per period).
    t = (2046.0 - ch.b1_phase) / f_code + np.arange(40) * period + 0.5 * period
    term = eng._b1i_term(ch, g, t, f_code, np.ones_like(t), np, rho)
    # At the first sample cp == 2046, i.e. one code period has elapsed.
    k = int(base_ms) + 1 + np.arange(40)
    expected = 1.0 - 2.0 * NH_CODE[k % 20]
    assert np.allclose(term.real, expected)
    # A code period advances the NH chip by one (wrapping every 20 = 1 bit).
    assert np.all((np.diff(k) % 20) == 1)
    # The propagation delay is part of the index (old code omitted it).
    assert int(base_ms) != int(sub_gps_time(g, ch.b1i_frame_start) * 1000.0)
    assert abs(base_ms - sub_gps_time(g, ch.b1i_frame_start) * 1000.0) > 1.0


@pytest.mark.skipif(not os.path.exists(_MERGED), reason="merged cache missing")
def test_engine_b1i_data_bit_is_20_code_periods() -> None:
    """One D1 bit spans 20 ms = 20 code periods; NH resets per bit."""
    from gnss_sim.beidou_nav import NH_CODE

    eng = _make_b1i_engine()
    ch = next(c for c in eng.channels if c.kind == "beidou")
    ch.ca = np.ones_like(ch.ca)
    bits = ch.b1i_bits
    g = ch.b1i_frame_start
    rho = _Geo(2.1e7, 0.0, 2.1e7, (0.0, 1.0), 0.0)
    f_code = 2.046e6
    period = 2046.0 / f_code
    base_ms = (sub_gps_time(g, ch.b1i_frame_start)
               - rho.rng / SPEED_OF_LIGHT) * 1000.0
    t = (2046.0 - ch.b1_phase) / f_code + np.arange(45) * period + 0.5 * period  # >2 bits
    term = eng._b1i_term(ch, g, t, f_code, np.ones_like(t), np, rho)
    k = int(base_ms) + 1 + np.arange(45)
    expected = ((1.0 - 2.0 * bits[(k // 20) % bits.shape[0]])
                * (1.0 - 2.0 * NH_CODE[k % 20]))
    assert np.allclose(term.real, expected)
    # NH wraps to 0 exactly when the data bit index advances.
    bit_idx = k // 20
    nh_idx = k % 20
    assert np.all(np.diff(bit_idx) == (nh_idx[1:] == 0).astype(np.int64))


@pytest.mark.skipif(not os.path.exists(_MERGED), reason="merged cache missing")
def test_engine_b1i_broadcasts_bdt_time_fields() -> None:
    """D1 WN/SOW/toc/toe must be on the BDT scale, not the GPS scale."""
    eng = _make_b1i_engine()
    ch = next(c for c in eng.channels if c.kind == "beidou")
    start = ch.b1i_frame_start
    sow_bdt = int(gps2bdt_sec(start.sec))
    assert int(start.sec) - sow_bdt == pytest.approx(BDT_GPST_OFFSET_S)
    # The frame epoch is a BDT 6-s boundary: GPST == 14 (mod 30),
    # BDT SOW == 0 (mod 6).  (Previously it was the GPS-30 s boundary, so
    # the broadcast SOW was 4 (mod 6) and the receiver rejected the frame.)
    assert int(round(start.sec)) % 30 == 14
    assert sow_bdt % 6 == 0
    assert sow_bdt % 30 == 0

    sf1 = parse_subframe(ch.b1i_bits[:300])
    sf2 = parse_subframe(ch.b1i_bits[300:600])
    sf3 = parse_subframe(ch.b1i_bits[600:900])
    assert sf1["sow"] == sow_bdt
    assert sf1["sow"] % 6 == 0
    assert sf2["sow"] == sow_bdt + 6 and sf3["sow"] == sow_bdt + 12
    assert sf1["wn"] == int(start.week) - BDT_WEEK_OFFSET
    # The un-converted GPS week would be a different value (regression trap).
    assert sf1["wn"] != int(start.week)
    assert sf1["toc_raw"] == int(round(
        gps2bdt_sec(ch.eph.toc.sec) / 8.0))
    toe = (int(sf2["toe_hi"]) << 15) | int(sf3["toe_lo"])
    assert toe == int(round(gps2bdt_sec(ch.eph.toe.sec) / 8.0))


# ======================================================================
# B1I D1 frame epoch (BDT alignment) and reference-decoder cross-check
# ======================================================================
@pytest.mark.skipif(not os.path.exists(_MERGED), reason="merged cache missing")
def test_select_ephemeris_prefers_nearest_bdt_toe() -> None:
    """BDS ``toe`` is BDT; selection must normalise it before comparing.

    Regression: comparing the raw BDT week (GPS week - 1356) against GPS time
    made the ~1356-week offset dominate, so ``select_ephemeris`` always picked
    the *last* BDS record of the day (hours stale/future) and a real receiver
    rejected the ephemeris even though the D1 message decoded cleanly.
    """
    from gnss_sim.rinex import ephemeris_toe_gps, select_ephemeris
    by_sv, _iono = rinex.parse_nav_file(_MERGED)
    keys = sorted(k for k in by_sv if k.startswith("C"))
    assert keys
    start = date2gps(2026, 9, 23, 12, 0, 0)
    target = start.week * 604800.0 + start.sec

    def gps_toe(eph):
        t = ephemeris_toe_gps(eph)
        return t.week * 604800.0 + t.sec

    close = 0
    for key in keys:
        best = min(by_sv[key], key=lambda e: abs(target - gps_toe(e)))
        assert select_ephemeris(by_sv, key, start) is best
        if abs(target - gps_toe(best)) < 7200.0:
            close += 1
    # The daily cache has a fresh BDS record near noon (not only the 23:00 one).
    assert close > 0


def test_b1i_frame_epoch_is_a_bdt_boundary() -> None:
    """The D1 frame epoch must be GPST == 14 (mod 30) so SOW == 0 (mod 6)."""
    for sec in (14.0, 15.0, 43.9, 44.0, 3599.0, 43214.0, 600000.0):
        ep = b1i_frame_epoch(2300, sec)
        # latest boundary at or before ``sec``, strictly within one frame.
        assert sub_gps_time(GpsTime(2300, sec), ep) >= 0.0
        assert sub_gps_time(GpsTime(2300, sec), ep) < 30.0
        assert int(round(ep.sec)) % 30 == 14
        assert gps2bdt_sec(ep.sec) % 6.0 == pytest.approx(0.0)
        assert gps2bdt_sec(ep.sec) % 30.0 == pytest.approx(0.0)


def test_b1i_frame_epoch_week_boundary() -> None:
    # 00:00:05 GPST is 23:59:51 BDT of the previous BDT week.
    ep = b1i_frame_epoch(2300, 5.0)
    assert ep.week == 2299 and ep.sec == pytest.approx(604784.0)
    assert gps2bdt_sec(ep.sec) == pytest.approx(604770.0)
    # 00:00:14 GPST starts the new BDT week (BDT SOW 0).
    ep = b1i_frame_epoch(2300, 14.0)
    assert ep.week == 2300 and ep.sec == pytest.approx(14.0)
    assert gps2bdt_sec(ep.sec) == pytest.approx(0.0)


def _ref_bch_syndrome(codeword_pm1) -> int:
    """GNSS-SDRLIB ``decodebch_bi1`` syndrome register (0 == no error)."""
    reg = [1, 1, 1, 1]
    for sym in codeword_pm1:
        bit = reg[3]
        reg[3], reg[2], reg[1] = reg[2], reg[1], reg[0]
        reg[0] = sym * bit
        reg[1] *= bit
    val = 0
    for k in range(4):
        val = (val << 1) | (1 if reg[k] < 0 else 0)
    return val


def _ref_deinterleave(word) -> list[int]:
    """GNSS-SDRLIB ``interleave(in, 2, 15, out)``: ``out[r*15+c]=in[c*2+r]``."""
    out = [0] * 30
    for r in range(2):
        for c in range(15):
            out[r * 15 + c] = word[c * 2 + r]
    return out


def _ref_d1_logical(tx):
    """Independent D1 deinterleave + BCH decode of a 300-bit subframe.

    Returns ``(logical01, syndromes)``.  Every syndrome must be zero for a
    clean frame; a nonzero value would mean our BCH parity or interleaving
    differs from the BeiDou ICD.
    """
    bits = [1 - 2 * int(b) for b in tx]
    out = [0] * 300
    syn: list[int] = []
    for i in range(10):
        if i == 0:                    # word 1 is copied raw (not interleaved)
            out[0:30] = bits[0:30]
            continue
        cws = _ref_deinterleave(bits[30 * i:30 * i + 30])
        c1, c2 = cws[0:15], cws[15:30]
        syn += [_ref_bch_syndrome(c1), _ref_bch_syndrome(c2)]
        base = 30 * i
        for j in range(11):
            out[base + j] = c1[j]
            out[base + j + 11] = c2[j]
        for j in range(4):
            out[base + j + 22] = c1[11 + j]
            out[base + j + 26] = c2[11 + j]
    return [0 if v > 0 else 1 for v in out], syn


def _ref_field(bits, parts, signed: bool = False) -> int:
    width = sum(w for _p, w in parts)
    val = 0
    for pos, w in parts:
        for j in range(w):
            val = (val << 1) | (bits[pos + j] & 1)
    if signed and (val >> (width - 1)) & 1:
        val -= 1 << width
    return val


def test_b1i_d1_matches_gnss_sdrlib_reference() -> None:
    """Cross-check the D1 encoder against an independent reference decoder.

    The BCH syndrome, parallel/serial deinterleaver and field offsets are a
    direct port of GNSS-SDRLIB ``sdrnav_bds.c`` (Taro Suzuki).  RTKLIB
    ``decode_bds_d1`` and GNSS-SDR ``Beidou_DNAV.h`` agree on the same field
    offsets and scales, so this validates field order/scale/BCH/interleave.
    """
    eph = beidou_nav._synthetic_ephemeris()
    week = 1081
    sow = 259200.0
    for sfid in (1, 2, 3, 4, 5):
        tx = d1_subframe(eph, sfid, sow, week=week)
        logical, syn = _ref_d1_logical(tx)
        assert set(syn) == {0}, f"subframe {sfid}: BCH syndrome {syn}"
        assert _ref_field(logical, ((15, 3),)) == sfid
        assert _ref_field(logical, ((18, 8), (30, 12))) == int(sow)

    sf1, syn1 = _ref_d1_logical(d1_subframe(eph, 1, sow, week=week))
    assert set(syn1) == {0}
    assert _ref_field(sf1, ((42, 1),)) == 0                    # SatH1
    assert _ref_field(sf1, ((43, 5),)) == (eph.iodc & 0x1F)    # AODC
    assert _ref_field(sf1, ((48, 4),)) == (eph.ura & 0xF)      # URAI
    assert _ref_field(sf1, ((60, 13),)) == week                # WN (BDT)
    assert _ref_field(sf1, ((73, 9), (90, 8))) == int(round(eph.toc.sec / 8))
    # BDT field scales, MSB-first across the split parts, RTKLIB/GNSS-SDR LSBs.
    assert _ref_field(sf1, ((214, 11),), signed=True) == int(round(
        eph.af2 / 2.0 ** -66))
    assert _ref_field(sf1, ((225, 7), (240, 17)), signed=True) == int(round(
        eph.af0 / 2.0 ** -33))
    assert _ref_field(sf1, ((257, 5), (270, 17)), signed=True) == int(round(
        eph.af1 / 2.0 ** -50))

    sf2, syn2 = _ref_d1_logical(d1_subframe(eph, 2, sow, week=week))
    assert set(syn2) == {0}
    assert _ref_field(sf2, ((42, 10), (60, 6)), signed=True) == int(round(
        eph.deltan / (np.pi * 2.0 ** -43)))
    assert _ref_field(sf2, ((92, 20), (120, 12)), signed=True) == int(round(
        eph.m0 / (np.pi * 2.0 ** -31)))
    assert _ref_field(sf2, ((250, 12), (270, 20))) == int(round(
        eph.sqrta / 2.0 ** -19))

    sf3, syn3 = _ref_d1_logical(d1_subframe(eph, 3, sow, week=week))
    assert set(syn3) == {0}
    assert _ref_field(sf3, ((65, 17), (90, 15)), signed=True) == int(round(
        eph.inc0 / (np.pi * 2.0 ** -31)))
    assert _ref_field(sf3, ((211, 21), (240, 11)), signed=True) == int(round(
        eph.omg0 / (np.pi * 2.0 ** -31)))
    assert _ref_field(sf3, ((251, 11), (270, 21)), signed=True) == int(round(
        eph.aop / (np.pi * 2.0 ** -31)))
    toe_hi = _ref_field(sf2, ((290, 2),))
    toe_lo = _ref_field(sf3, ((42, 10), (60, 5)))
    assert (toe_hi << 15) | toe_lo == int(round(eph.toe.sec / 8.0))
