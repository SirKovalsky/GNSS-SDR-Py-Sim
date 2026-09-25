"""Тесты исправлений: CDDIS URL, политика RINEX, UHD, u-blox, GUI.

Запуск (без железа и без сети)::

    E:\\MySoftware\\SDR_Scan\\.venv\\Scripts\\python.exe -m pytest tests -q
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402

from gnss_sim import rinexfetch  # noqa: E402
from gnss_sim.config import SimConfig  # noqa: E402
from gnss_sim.gpstime import date2gps  # noqa: E402


def _cache_file(cache, year, month, day):
    from gnss_sim.rinexfetch import _daily_names, doy_from_date
    doy = doy_from_date(year, month, day)
    return os.path.join(cache, _daily_names(year, doy)[1])


# ----------------------------------------------------------------------
# A) CDDIS URL + порядок источников
# ----------------------------------------------------------------------
def test_cddis_urls_have_no_day_folder() -> None:
    urls = rinexfetch.cddis_daily_urls(2026, 268)
    base = "https://cddis.nasa.gov/archive/gnss/data/daily/2026/brdc/"
    assert urls == [
        base + "BRDC00IGS_R_20262680000_01D_MN.rnx.gz",
        base + "brdc2680.26n.gz",
        base + "brdc2680.26g.gz",
    ]
    assert all("/268/" not in u for u in urls)


def test_ensure_nav_prefers_cddis_when_credentials(tmp_path, monkeypatch):
    order = []

    def fake_cddis(*a, **kw):
        order.append("cddis")
        return str(tmp_path / "from_cddis.rnx")

    def fake_bkg(url, dest):
        order.append("bkg")
        raise AssertionError("BKG не должен вызываться первым")

    monkeypatch.setattr(rinexfetch, "download_cddis", fake_cddis)
    monkeypatch.setattr(rinexfetch, "_download", fake_bkg)

    start = date2gps(2026, 9, 21, 0, 0, 0)
    path = rinexfetch.ensure_nav_file(
        start, cache_dir=str(tmp_path), source="auto",
        cddis_user="u", cddis_password="p")
    assert path == str(tmp_path / "from_cddis.rnx")
    assert order and order[0] == "cddis"


def test_ensure_nav_bkg_first_without_credentials(tmp_path, monkeypatch):
    order = []

    def fake_cddis(*a, **kw):
        order.append("cddis")
        return "x"

    def fake_bkg(url, dest):
        order.append("bkg")
        with open(dest, "wb") as fh:
            fh.write(b"data")

    monkeypatch.setattr(rinexfetch, "download_cddis", fake_cddis)
    monkeypatch.setattr(rinexfetch, "_download", fake_bkg)

    start = date2gps(2026, 9, 21, 0, 0, 0)
    path = rinexfetch.ensure_nav_file(start, cache_dir=str(tmp_path),
                                      source="auto")
    assert os.path.exists(path)
    assert order and order[0] == "bkg"
    assert "cddis" not in order


# ----------------------------------------------------------------------
# B) Политика повторного использования RINEX
# ----------------------------------------------------------------------
def _no_download(monkeypatch, record=None):
    def fake(start, **kw):
        if record is not None:
            record.update(kw)
            record["called"] = True
        return "downloaded.rnx"
    monkeypatch.setattr(rinexfetch, "ensure_nav_file", fake)


def test_resolve_explicit_nav_never_downloads(tmp_path, monkeypatch):
    cfg = SimConfig(nav_file=str(tmp_path / "given.26n"))
    (tmp_path / "given.26n").write_text("x")
    rec = {}
    _no_download(monkeypatch, rec)
    path = rinexfetch.resolve_nav_file(date2gps(2026, 9, 21), cfg,
                                       cache_dir=str(tmp_path))
    assert path == str(tmp_path / "given.26n")
    assert not rec.get("called")


def test_resolve_reuses_past_cached_file(tmp_path, monkeypatch):
    p = _cache_file(str(tmp_path), 2020, 1, 1)
    with open(p, "wb") as fh:
        fh.write(b"data")
    rec = {}
    _no_download(monkeypatch, rec)
    path = rinexfetch.resolve_nav_file(date2gps(2020, 1, 1), SimConfig(),
                                       cache_dir=str(tmp_path))
    assert path == p and not rec.get("called")


def test_resolve_today_fresh_cache_reused(tmp_path, monkeypatch):
    now = datetime(2026, 9, 21, 12, 0, 0, tzinfo=timezone.utc)
    p = _cache_file(str(tmp_path), 2026, 9, 21)
    with open(p, "wb") as fh:
        fh.write(b"data")
    recent = now.timestamp() - 600.0
    os.utime(p, (recent, recent))
    rec = {}
    _no_download(monkeypatch, rec)
    path = rinexfetch.resolve_nav_file(date2gps(2026, 9, 21), SimConfig(),
                                       cache_dir=str(tmp_path), now=now)
    assert path == p and not rec.get("called")


def test_resolve_today_stale_ask_reuse(tmp_path, monkeypatch):
    now = datetime(2026, 9, 21, 12, 0, 0, tzinfo=timezone.utc)
    p = _cache_file(str(tmp_path), 2026, 9, 21)
    with open(p, "wb") as fh:
        fh.write(b"data")
    old = now.timestamp() - 7200.0
    os.utime(p, (old, old))
    rec = {}
    _no_download(monkeypatch, rec)
    asked = {}

    def ask(path, age):
        asked["age"] = age
        return False

    path = rinexfetch.resolve_nav_file(date2gps(2026, 9, 21), SimConfig(),
                                       cache_dir=str(tmp_path), now=now,
                                       ask=ask)
    assert path == p and not rec.get("called")
    assert asked["age"] >= 3600.0


def test_resolve_today_stale_ask_update(tmp_path, monkeypatch):
    now = datetime(2026, 9, 21, 12, 0, 0, tzinfo=timezone.utc)
    p = _cache_file(str(tmp_path), 2026, 9, 21)
    with open(p, "wb") as fh:
        fh.write(b"data")
    old = now.timestamp() - 7200.0
    os.utime(p, (old, old))
    rec = {}
    _no_download(monkeypatch, rec)
    path = rinexfetch.resolve_nav_file(date2gps(2026, 9, 21), SimConfig(),
                                       cache_dir=str(tmp_path), now=now,
                                       ask=lambda path, age: True)
    assert path == "downloaded.rnx"
    assert rec.get("called") and rec.get("force") is True


def test_resolve_no_cache_downloads(tmp_path, monkeypatch):
    rec = {}
    _no_download(monkeypatch, rec)
    path = rinexfetch.resolve_nav_file(date2gps(2026, 9, 21), SimConfig(),
                                       cache_dir=str(tmp_path))
    assert path == "downloaded.rnx" and rec.get("called")


def test_resolve_requires_auto_download_when_empty(tmp_path):
    try:
        rinexfetch.resolve_nav_file(
            date2gps(2026, 9, 21), SimConfig(auto_download=False),
            cache_dir=str(tmp_path))
    except ValueError as exc:
        assert "эфемерид" in str(exc).lower() or "авто" in str(exc).lower()
    else:  # pragma: no cover
        raise AssertionError("ожидался ValueError без файла и автоскачивания")


# ----------------------------------------------------------------------
# C) UHD metadata errors (без uhd в системе)
# ----------------------------------------------------------------------
def test_uhd_error_helpers_introspect_types() -> None:
    from gnss_sim.uhd_tx import (
        rx_error_code_enum,
        rx_has_error,
        tx_error_code_enum,
        tx_has_error,
    )

    class FakeUhd:
        class types:  # noqa: N801
            pass

    assert tx_error_code_enum(FakeUhd) is None
    assert rx_error_code_enum(FakeUhd) is None

    class MD:
        def __init__(self, code):
            self.error_code = code

    # Строковое и числовое сравнение работает без enum.
    assert tx_has_error(MD("underflow"), "underflow", None) is True
    assert tx_has_error(MD("none"), "underflow", None) is False
    assert tx_has_error(MD(2), "underflow", None) is True
    assert rx_has_error(MD("overflow"), "overflow", None) is True
    assert rx_has_error(MD(2), "overflow", None) is True
    assert rx_has_error(MD(1), "timeout", None) is True

    class EC:
        underflow = 2
        overflow = 8
        timeout = 1

    class Types:
        TXMetadataErrorCode = EC
        RXMetadataErrorCode = EC

    class Uhd:
        types = Types

    assert tx_error_code_enum(Uhd) is EC
    assert rx_error_code_enum(Uhd) is EC
    assert tx_has_error(MD(2), "underflow", EC) is True
    assert tx_has_error(MD(3), "underflow", EC) is False


def test_uhd_duplex_same_channel_rejected() -> None:
    from gnss_sim.uhd_duplex import UhdDuplex
    from gnss_sim.uhd_tx import TxError
    try:
        UhdDuplex(args="type=b200", tx_channel=0, rx_channel=0)
    except TxError as exc:
        assert "канал" in str(exc).lower() or "uhd" in str(exc).lower()
    else:  # pragma: no cover
        raise AssertionError("ожидался TxError для одинаковых каналов")


# ----------------------------------------------------------------------
# E) u-blox холодный старт
# ----------------------------------------------------------------------
def test_ubx_cfg_rst_frame_and_checksum() -> None:
    from gnss_sim.ublox import (
        ubx_cfg_rst_frame,
        ubx_checksum,
        ubx_frame,
        BBR_COLD_START,
        RESET_SOFTWARE,
    )
    frame = ubx_cfg_rst_frame()
    assert frame == bytes.fromhex("B5 62 06 04 04 00 FF FF 02 00 0E 61")
    assert frame[:2] == b"\xb5\x62"
    assert frame[2:4] == bytes([0x06, 0x04])
    assert int.from_bytes(frame[4:6], "little") == 4
    assert frame[6:10] == bytes([0xFF, 0xFF, 0x02, 0x00])
    assert ubx_checksum(bytes.fromhex("06 04 04 00 FF FF 02 00")) == (0x0E, 0x61)

    assert ubx_frame(0x06, 0x04, bytes.fromhex("FF FF 02 00")) == frame
    assert ubx_cfg_rst_frame(BBR_COLD_START, RESET_SOFTWARE) == frame

    other = ubx_cfg_rst_frame(0x0000, 0x00)
    ck = ubx_checksum(other[2:-2])
    assert bytes(ck) == other[-2:]


# ----------------------------------------------------------------------
# F) Конфликт имён cb_auto + GUI
# ----------------------------------------------------------------------
def _app():
    from PyQt5 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_config_defaults_are_valid() -> None:
    cfg = SimConfig()
    assert cfg.tx_channel == 0 and cfg.rx_channel == 1
    assert cfg.tx_bandwidth == 0.0


def test_cli_tx_bandwidth_default_zero() -> None:
    from gnss_sim.cli import build_parser
    assert build_parser().parse_args([]).tx_bandwidth == 0.0


def test_gui_layout_controls_and_cb_auto_distinct() -> None:
    from gnss_sim.gui import MainWindow
    app = _app()
    win = MainWindow()
    try:
        assert win.left_tabs.count() == 2
        assert win.tabs.count() == 3
        assert win.tabs.tabText(2) == "Спектр"
        # Коллизия имён устранена: автоскачивание != авторегулятор TX.
        assert win.cb_auto is not win.cb_tx_auto
        assert win.cb_tx_auto.text().startswith("Авторегулятор")
        # Валидные каналы дуплекса и полоса «авто».
        assert win.sp_ch.value() == 0 and win.sp_rx_ch.value() == 1
        assert win.sp_bw.value() == 0.0
        assert win.progress.maximum() == 1000
        assert hasattr(win, "btn_cold") and hasattr(win, "btn_gen")
        assert win.btn_gen.text() == "Сгенерировать IQ"

        # fs/centre are read-only derived labels (no manual combos any more);
        # the default selection is conservative L1 (2.6 Msps / 1575.42 MHz)
        # until «Объединять L1+B1I» is ticked.
        assert not hasattr(win, "cmb_fs") and not hasattr(win, "cmb_fc")
        assert "2.6" in win.lbl_fs.text()
        assert "1575.42" in win.lbl_fc.text()

        win.cb_auto.setChecked(False)
        win.cb_tx_auto.setChecked(True)
        cfg = win._collect()
        assert cfg.auto_download is False
        assert cfg.tx_power_auto is True
    finally:
        win.close()
    del app


def test_gui_channel_guard_disables_start() -> None:
    from gnss_sim.gui import MainWindow
    app = _app()
    win = MainWindow()
    try:
        win.cb_monitor.setChecked(True)
        win.sp_ch.setValue(0)
        win.sp_rx_ch.setValue(0)
        assert win.btn_start.isEnabled() is False
        assert not win.lbl_mon_warn.isHidden()
        # Смена TX автоматически переводит RX на другой канал.
        win.sp_ch.setValue(1)
        assert win.sp_rx_ch.value() == 0
        assert win.btn_start.isEnabled() is True
        assert win.lbl_mon_warn.isHidden()
    finally:
        win.close()
    del app


def test_gui_generate_iq_disables_usrp(monkeypatch) -> None:
    from gnss_sim import gui
    from gnss_sim.gui import MainWindow
    app = _app()
    win = MainWindow()
    captured = {}

    class FakeRunner:
        def __init__(self, cfg, **kwargs):
            captured["cfg"] = cfg
            captured["ask"] = kwargs.get("ask")

        def start(self):
            pass

        def is_running(self):
            return False

        def stop(self):
            pass

    monkeypatch.setattr(gui, "SimulationRunner", FakeRunner)
    try:
        win.cb_tx.setChecked(True)
        win.ed_out.setText("out.cs16")
        win._generate_iq()
        assert captured["cfg"].use_usrp is False
        assert captured["cfg"].output == "out.cs16"
        assert captured["ask"] is not None
    finally:
        win.close()
    del app


def test_gui_start_with_auto_download_no_warning(monkeypatch) -> None:
    from PyQt5 import QtWidgets
    from gnss_sim import gui
    from gnss_sim.gui import MainWindow
    app = _app()
    win = MainWindow()
    warnings = {"n": 0}
    launched = {"n": 0}

    class FakeRunner:
        def __init__(self, cfg, **kwargs):
            self.cfg = cfg

        def start(self):
            launched["n"] += 1

        def is_running(self):
            return False

        def stop(self):
            pass

    monkeypatch.setattr(gui, "SimulationRunner", FakeRunner)
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning",
                        staticmethod(lambda *a, **k: warnings.__setitem__(
                            "n", warnings["n"] + 1)))
    try:
        win.cb_auto.setChecked(True)
        win.ed_nav.setText("")
        win._start()
        assert warnings["n"] == 0
        assert launched["n"] == 1

        win.cb_auto.setChecked(False)
        win._start()
        assert warnings["n"] == 1
        assert launched["n"] == 1  # второй запуск не состоялся
    finally:
        win.close()
    del app


def test_gui_progress_reaches_100() -> None:
    from gnss_sim.gui import MainWindow
    app = _app()
    win = MainWindow()
    try:
        win._on_progress(0.5, 1.0, 1.0, 1.0)
        assert win.progress.value() == 500
        win._on_progress(1.0, 2.0, 2.0, 1.0)
        assert win.progress.value() == 1000
        win.progress.setValue(0)
        win._on_finished(None)
        assert win.progress.value() == 1000
    finally:
        win.close()
    del app


def test_spectrum_axis_mode() -> None:
    from gnss_sim.spectrum import SpectrumWidget
    app = _app()
    w = SpectrumWidget()
    try:
        x = np.exp(2j * np.pi * 0.1 * np.arange(8192)).astype(np.complex64)
        w.update_spectrum(x, 1e6, "TX", center_freq=1575.42e6)
        if w.plot is not None:
            assert w._x_absolute is True
        w.update_spectrum(x, 1e6, "TX", center_freq=0.0)
        if w.plot is not None:
            assert w._x_absolute is False
    finally:
        w.close()
    del app
