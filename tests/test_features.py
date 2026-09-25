"""Тесты новых функций: PSD, календарь/CDDIS, NMEA u-blox, GUI-контролы.

Запуск (без железа и без сети)::

    E:\\MySoftware\\SDR_Scan\\.venv\\Scripts\\python.exe -m pytest tests -q
"""

from __future__ import annotations

import math
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402

from gnss_sim.power import welch_psd  # noqa: E402
from gnss_sim.rinexfetch import (  # noqa: E402
    cddis_daily_urls,
    date_from_doy,
    doy_from_date,
    ensure_nav_file,
    download_cddis,
)
from gnss_sim.gpstime import date2gps  # noqa: E402
from gnss_sim.ublox import parse_nmea_line  # noqa: E402


# ----------------------------------------------------------------------
# A) Welch PSD
# ----------------------------------------------------------------------
def test_welch_psd_tone() -> None:
    fs = 1.0e6
    n = 32768
    nfft = 2048
    f0 = 200.0 * fs / nfft  # точно на бин -> без утечки
    t = np.arange(n) / fs
    x = np.exp(2j * np.pi * f0 * t).astype(np.complex64)

    freqs, psd = welch_psd(x, fs, nfft=nfft)
    assert freqs.size == psd.size == nfft
    assert abs(freqs[np.argmax(psd)] - f0) < fs / nfft

    df = fs / freqs.size
    integrated = 10.0 * math.log10(float(np.sum(10.0 ** (psd / 10.0))) * df)
    assert abs(integrated) < 0.5, integrated  # единичный тон -> ~0 dBFS


def test_welch_psd_empty_and_short() -> None:
    f, p = welch_psd(np.zeros(0, dtype=np.complex64), 1e6)
    assert f.size == 0 and p.size == 0
    # Блок короче nfft обрабатывается без ошибок.
    f2, p2 = welch_psd(np.ones(100, dtype=np.complex64), 1e6, nfft=2048)
    assert f2.size == 100 and p2.size == 100


# ----------------------------------------------------------------------
# B) Календарь и CDDIS
# ----------------------------------------------------------------------
def test_doy_date_roundtrip() -> None:
    assert doy_from_date(2026, 9, 21) == 264
    assert date_from_doy(2026, 264) == (2026, 9, 21)

    # 2024 — високосный.
    assert doy_from_date(2024, 3, 1) == 61
    assert date_from_doy(2024, 61) == (2024, 3, 1)

    for year, last in ((2024, 366), (2026, 365)):
        for doy in range(1, last + 1):
            y, m, d = date_from_doy(year, doy)
            assert y == year
            assert doy_from_date(y, m, d) == doy

    # 29 февраля есть только в високосном году.
    assert date_from_doy(2024, 60) == (2024, 2, 29)
    assert date_from_doy(2026, 60) == (2026, 3, 1)


def test_cddis_urls() -> None:
    urls = cddis_daily_urls(2026, 264)
    assert len(urls) == 3
    # CDDIS хранит дневные файлы прямо в .../{year}/brdc/ (без папки DOY).
    base = "https://cddis.nasa.gov/archive/gnss/data/daily/2026/brdc/"
    assert all(u.startswith(base) for u in urls)
    assert all("/264/brdc/" not in u for u in urls)
    assert urls[0] == base + "BRDC00IGS_R_20262640000_01D_MN.rnx.gz"
    assert urls[1] == base + "brdc2640.26n.gz"
    assert urls[2] == base + "brdc2640.26g.gz"

    # Смена года в 2-значном суффиксе (2024 -> .24n.gz).
    urls24 = cddis_daily_urls(2024, 61)
    assert urls24[1].endswith("brdc0610.24n.gz")


def test_cddis_requires_credentials() -> None:
    try:
        download_cddis(2026, 264)
    except ValueError as exc:
        assert "CDDIS" in str(exc) or "логин" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("ожидался ValueError без учётных данных")

    start = date2gps(2026, 9, 21, 0, 0, 0)
    try:
        ensure_nav_file(start, source="cddis")
    except RuntimeError as exc:
        assert "CDDIS" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("ожидался RuntimeError для source=cddis без логина")


# ----------------------------------------------------------------------
# C) NMEA / u-blox
# ----------------------------------------------------------------------
def test_parse_nmea_gga() -> None:
    line = ("$GNGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,"
            "M,46.9,M,,*47")
    d = parse_nmea_line(line)
    assert d["type"] == "GGA" and d["talker"] == "GN" and d["system"] == "G"
    assert abs(d["lat"] - 48.1173) < 1e-3
    assert abs(d["lon"] - 11.5166667) < 1e-3
    assert d["fix"] == 1 and d["num_sats"] == 8
    assert abs(d["hdop"] - 0.9) < 1e-9 and abs(d["height"] - 545.4) < 1e-9


def test_parse_nmea_gga_south_west() -> None:
    line = "$GPGGA,000000,3352.000,S,15112.000,W,0,00,,25.0,M,,M,,*00"
    d = parse_nmea_line(line)
    assert d["lat"] < 0 and d["lon"] < 0


def test_parse_nmea_rmc() -> None:
    line = ("$GNRMC,123519,A,4807.038,N,01131.000,E,022.4,084.4,"
            "230394,003.1,W*6A")
    d = parse_nmea_line(line)
    assert d["type"] == "RMC" and d["status"] == "A"
    assert d["datetime"] == "1994-03-23 12:35:19"
    assert abs(d["speed_knots"] - 22.4) < 1e-9


def test_parse_nmea_gsv_multignss() -> None:
    line = "$GPGSV,3,1,11,03,03,111,00,04,15,270,00,06,30,045,20,09,45,180,40*00"
    d = parse_nmea_line(line)
    assert d["type"] == "GSV" and d["system"] == "G"
    assert d["total"] == 3 and d["num"] == 1 and d["in_view"] == 11
    assert len(d["sats"]) == 4
    assert d["sats"][0]["prn"] == 3 and d["sats"][0]["snr"] == 0
    assert d["sats"][3]["prn"] == 9 and d["sats"][3]["elev"] == 45

    gal = parse_nmea_line("$GAGSV,1,1,02,11,20,300,35,12,40,120,42*00")
    assert gal["system"] == "E" and gal["sats"][0]["prn"] == 11
    qzs = parse_nmea_line("$GQGSV,1,1,01,194,50,200,44*00")
    assert qzs["system"] == "J" and qzs["sats"][0]["prn"] == 194


def test_parse_nmea_gsa_and_garbage() -> None:
    line = "$GPGSA,A,3,04,05,09,12,24,25,29,31,32,,,,1.8,0.9,1.5*00"
    d = parse_nmea_line(line)
    assert d["type"] == "GSA" and d["fix"] == 3
    assert d["svids"][:2] == [4, 5]
    assert abs(d["pdop"] - 1.8) < 1e-9 and abs(d["hdop"] - 0.9) < 1e-9

    assert parse_nmea_line("") == {}
    assert parse_nmea_line("not nmea") == {}
    assert parse_nmea_line("$") == {}


def test_ublox_module_imports_without_hardware() -> None:
    from gnss_sim.ublox import NmeaReader, list_ports
    assert isinstance(list_ports(), list)
    reader = NmeaReader("COM_DOES_NOT_EXIST", 38400)
    assert reader.is_running() is False
    reader.stop()  # должен быть безопасным без запуска


# ----------------------------------------------------------------------
# GUI (offscreen)
# ----------------------------------------------------------------------
def _app():
    from PyQt5 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_config_new_fields() -> None:
    from gnss_sim.config import SimConfig
    cfg = SimConfig()
    assert cfg.cddis_user == "" and cfg.cddis_password == ""
    assert cfg.cddis_token == "" and cfg.download_source == "auto"
    assert "download_source" in cfg.to_dict()


def test_cli_new_flags() -> None:
    from gnss_sim.cli import build_parser
    args = build_parser().parse_args([
        "--source", "cddis", "--cddis-user", "u",
        "--cddis-password", "p", "--cddis-token", "t",
    ])
    assert args.source == "cddis"
    assert args.cddis_user == "u" and args.cddis_password == "p"
    assert args.cddis_token == "t"
    assert build_parser().parse_args([]).source == "auto"


def test_gui_new_controls_headless() -> None:
    from PyQt5 import QtWidgets
    from gnss_sim.gui import MainWindow
    app = _app()
    win = MainWindow()
    try:
        assert win.tabs.count() == 3
        assert win.tabs.tabText(2) == "Спектр"
        assert win.cmb_source.currentText() == "auto"
        assert win.ed_cddis_pass.echoMode() == QtWidgets.QLineEdit.Password
        assert win.ed_date.text().count("/") == 2
        assert win.cmb_baud.currentText() == "38400"
        assert win.tbl_ublox.columnCount() == 5
        assert hasattr(win, "spectrum_widget")

        win.cmb_source.setCurrentText("cddis")
        win.ed_cddis_user.setText("user")
        win.ed_cddis_pass.setText("secret")
        win.ed_cddis_token.setText("tok")
        cfg = win._collect()
        assert cfg.download_source == "cddis"
        assert cfg.cddis_user == "user"
        assert cfg.cddis_password == "secret"
        assert cfg.cddis_token == "tok"
    finally:
        win.close()
    del app


def test_spectrum_widget_headless() -> None:
    from gnss_sim.spectrum import SpectrumWidget
    app = _app()
    w = SpectrumWidget()
    try:
        x = np.exp(2j * np.pi * 0.1 * np.arange(8192)).astype(np.complex64)
        w.update_spectrum(x, 1e6, "TX")
        w.update_spectrum(x, 1e6, "RX")
        if w.plot is not None:
            assert w.describe() == "spectrum TX=1 RX=1"
        w.clear()
        w.update_spectrum(np.zeros(0, dtype=np.complex64), 1e6, "TX")
    finally:
        w.close()
    del app
