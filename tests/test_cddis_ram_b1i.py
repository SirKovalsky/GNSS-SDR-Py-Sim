"""Тесты новых функций: CDDIS посистемные файлы, IQ в RAM, авто-полоса B1I.

Запуск (без железа и без сети)::

    E:\\MySoftware\\SDR_Scan\\.venv\\Scripts\\python.exe -m pytest tests -q
"""

from __future__ import annotations

import gzip
import os
import urllib.error
from datetime import datetime, timezone

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from gnss_sim import rinex, rinexfetch, sysinfo  # noqa: E402
from gnss_sim.config import SimConfig  # noqa: E402
from gnss_sim.engine import (  # noqa: E402
    B210_MAX_CENTER,
    B210_MAX_FS,
    B210_MIN_CENTER,
    CARR_FREQ_B1I,
    b1i_band_fits,
    b210_band_ok,
    suggest_b1i_band,
)
from gnss_sim.gpstime import date2gps  # noqa: E402
from gnss_sim.iqfile import FileSink, IqFileSource  # noqa: E402
from gnss_sim.runner import SimulationRunner  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NAV = os.path.join(_ROOT, "brdc2680.26n")
_MERGED = os.path.join(_ROOT, "rinex_cache",
                       "BRDC00IGS_R_20262660000_01D_MN.rnx")
_BASE = "https://cddis.nasa.gov/archive/gnss/data/daily"


def _app():
    from PyQt5 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


# ======================================================================
# 1) CDDIS: посистемные имена/URL
# ======================================================================
def test_cddis_system_names_and_urls() -> None:
    names = rinexfetch.cddis_system_names(2026, 268)
    assert names == [
        "brdc2680.26n", "brdc2680.26g", "brdc2680.26l",
        "brdc2680.26c", "brdc2680.26j",
    ]
    urls = rinexfetch.cddis_system_urls(2026, 268)
    assert urls == [f"{_BASE}/2026/brdc/{n}.gz" for n in names]
    # Двузначный год.
    assert rinexfetch.cddis_system_names(2024, 61)[0] == "brdc0610.24n"


def test_daily_names_keep_first_three_positions() -> None:
    names = rinexfetch._daily_names(2026, 268)
    assert names[0] == "BRDC00IGS_R_20262680000_01D_MN.rnx"
    assert names[1] == "brdc2680.26n" and names[2] == "brdc2680.26g"
    assert names[3:] == ["brdc2680.26l", "brdc2680.26c", "brdc2680.26j"]


def test_requested_is_today() -> None:
    now = datetime(2026, 9, 25, 12, 0, 0, tzinfo=timezone.utc)
    assert rinexfetch._requested_is_today(2026, 268, now=now) is True
    assert rinexfetch._requested_is_today(2026, 264, now=now) is False


# ======================================================================
# 2) CDDIS: merged-vs-per-system скачивание (сеть замокана)
# ======================================================================
class _Resp:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data

    def __enter__(self) -> "_Resp":
        return self

    def __exit__(self, *exc) -> bool:
        return False


class _FakeOpener:
    """Opener, отдающий .gz для доступных суффиксов и 404 для остальных."""

    def __init__(self, available=()) -> None:
        self.available = set(available)
        self.urls: list[str] = []

    def open(self, req, timeout: float = 0.0):
        url = req.full_url
        self.urls.append(url)
        name = url.rsplit("/", 1)[-1][:-3]  # без .gz
        suffix = name[-1]
        merged = name.startswith("BRDC00IGS_R_")
        if not merged and suffix not in self.available:
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
        return _Resp(gzip.compress(f"data:{name}".encode("utf-8")))


def _cddis_with(monkeypatch, available):
    opener = _FakeOpener(available)
    monkeypatch.setattr(rinexfetch, "_make_cddis_opener",
                        lambda *a, **k: opener)
    return opener


def test_download_cddis_today_uses_per_system(tmp_path, monkeypatch) -> None:
    now = datetime(2026, 9, 25, 12, 0, 0, tzinfo=timezone.utc)
    _cddis_with(monkeypatch, {"n", "l", "c"})
    logs: list[str] = []
    paths = rinexfetch.download_cddis(
        2026, 268, username="u", password="p", cache_dir=str(tmp_path),
        log=logs.append, now=now)
    assert isinstance(paths, list)
    assert [os.path.basename(p) for p in paths] == [
        "brdc2680.26n", "brdc2680.26l", "brdc2680.26c"]
    for p in paths:
        assert os.path.exists(p) and os.path.getsize(p) > 0
        with open(p, "rb") as fh:  # распаковано
            assert fh.read(2) != b"\x1f\x8b"
    joined = "\n".join(logs)
    assert "мультисистемный BRDC00IGS доступен только за прошедшие сутки" in joined
    assert "посистемные" in joined
    # 404 посчитаны один раз, а не по каждому URL.
    assert joined.count("ещё не опубликовано (404)") == 1
    assert "2 посистемных" in joined


def test_download_cddis_past_prefers_merged(tmp_path, monkeypatch) -> None:
    now = datetime(2026, 9, 25, 12, 0, 0, tzinfo=timezone.utc)
    opener = _cddis_with(monkeypatch, {"n"})
    logs: list[str] = []
    path = rinexfetch.download_cddis(
        2026, 264, username="u", password="p", cache_dir=str(tmp_path),
        log=logs.append, now=now)
    assert isinstance(path, str)
    assert path.endswith("BRDC00IGS_R_20262640000_01D_MN.rnx")
    assert os.path.exists(path)
    assert opener.urls and "BRDC00IGS_R_" in opener.urls[0]
    assert "текущая дата" not in "\n".join(logs)


def test_cached_nav_set_merged_vs_system(tmp_path) -> None:
    cache = str(tmp_path)
    assert rinexfetch.cached_nav_set_for_date(2026, 268, cache) is None

    for name in ("brdc2680.26n", "brdc2680.26l"):
        (tmp_path / name).write_bytes(b"x")
    got = rinexfetch.cached_nav_set_for_date(2026, 268, cache)
    assert isinstance(got, list) and len(got) == 2

    merged = tmp_path / "BRDC00IGS_R_20262680000_01D_MN.rnx"
    merged.write_bytes(b"y")
    assert rinexfetch.cached_nav_set_for_date(2026, 268, cache) == str(merged)


def test_resolve_nav_file_returns_per_system_list(tmp_path, monkeypatch) -> None:
    for name in ("brdc2680.26n", "brdc2680.26l", "brdc2680.26c"):
        (tmp_path / name).write_bytes(b"x")
    now = datetime(2026, 9, 25, 12, 0, 0, tzinfo=timezone.utc)
    path = rinexfetch.resolve_nav_file(
        date2gps(2026, 9, 25), SimConfig(), cache_dir=str(tmp_path), now=now)
    assert isinstance(path, list) and len(path) == 3


# ======================================================================
# 3) parse_nav_file: список путей и слияние
# ======================================================================
def test_parse_nav_file_accepts_list_and_merges() -> None:
    single, _ = rinex.parse_nav_file(_NAV)
    for container in (list, tuple):
        merged, iono = rinex.parse_nav_file(container([_NAV, _NAV]))
        assert sorted(merged) == sorted(single)
        for key in single:
            assert len(merged[key]) == 2 * len(single[key])
        assert isinstance(iono, rinex.IonoUtc)


def test_parse_nav_file_merges_multiple_systems() -> None:
    if not os.path.exists(_MERGED):
        pytest.skip("merged multi-GNSS cache not present")
    merged_only, _ = rinex.parse_nav_file(_MERGED)
    both, _ = rinex.parse_nav_file([_MERGED, _NAV])
    counts = rinex.system_counts(both)
    base = rinex.system_counts(merged_only)
    assert counts.get("E", 0) > 0 and counts.get("C", 0) > 0
    assert counts["G"] == base["G"] + rinex.system_counts(
        rinex.parse_nav_file(_NAV)[0])["G"]


# ======================================================================
# 4) IqFileSource: загрузка в RAM
# ======================================================================
def _write_iq(path: str, fmt: str = "cs16", n: int = 4096) -> np.ndarray:
    rng = np.random.default_rng(7)
    x = 0.1 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    sink = FileSink(path, fmt=fmt, fs=1.0e6, center_freq=1575.42e6,
                    scale=10000.0)
    sink.start()
    sink.write(x)
    sink.close()
    return x


@pytest.mark.parametrize("fmt", ["cs16", "cf32"])
def test_iq_source_ram_roundtrip(tmp_path, fmt) -> None:
    path = str(tmp_path / f"sig.{fmt}")
    _write_iq(path, fmt=fmt, n=2048)
    src = IqFileSource(path, load_to_ram=True)
    src.start()
    try:
        assert src.in_ram is True
        assert src.total_samples == 2048
        assert src.read(1024).size == 1024
        assert src.read(4096).size == 1024  # только до конца
        assert src.read(1).size == 0        # EOF
        src.seek_start()
        assert src.read(2048).size == 2048  # лууп с начала
    finally:
        src.close()


def test_iq_source_ram_too_small_raises(tmp_path, monkeypatch) -> None:
    path = str(tmp_path / "sig.cs16")
    _write_iq(path, n=4096)
    monkeypatch.setattr(sysinfo, "available_ram", lambda: 16)
    with pytest.raises(RuntimeError) as exc:
        IqFileSource(path, load_to_ram=True).start()
    assert "оперативной памяти" in str(exc.value)

    # force=True обходит проверку.
    src = IqFileSource(path, load_to_ram=True, force=True)
    src.start()
    try:
        assert src.in_ram is True
        assert src.read(4096).size == 4096
    finally:
        src.close()


def test_runner_iq_input_ram(tmp_path) -> None:
    path = str(tmp_path / "sig.cs16")
    _write_iq(path, n=128)
    logs: list[str] = []
    cfg = SimConfig(iq_input=path, iq_in_ram=True, fs=1.0e6,
                    center_freq=1575.42e6, duration=0.01, loop=False)
    runner = SimulationRunner(cfg, log=logs.append)
    runner.prepare()
    assert runner._source is not None and runner._source.in_ram is True
    runner.run()
    assert runner.error is None
    assert any("RAM" in m for m in logs)


def test_cli_iq_ram_and_auto_b1i_flags() -> None:
    from gnss_sim.cli import build_parser
    p = build_parser()
    assert p.parse_args([]).iq_ram is False
    assert p.parse_args(["--iq-ram"]).iq_ram is True
    assert p.parse_args([]).auto_b1i is True
    assert p.parse_args(["--no-auto-b1i"]).auto_b1i is False
    assert p.parse_args(["--auto-b1i"]).auto_b1i is True


# ======================================================================
# 5) Авто-подбор полосы под B1I
# ======================================================================
def test_b1i_band_fits_cases() -> None:
    assert abs(CARR_FREQ_B1I - 1561.098e6) < 1.0
    assert b1i_band_fits(2.6e6, 1575.42e6) is False   # узкая полоса около L1
    assert b1i_band_fits(30.0e6, 1568.0e6) is True
    assert b1i_band_fits(30.0e6, 1575.42e6) is False  # центр на L1
    assert b1i_band_fits(0.0, 1561.098e6) is False


def test_suggest_b1i_band() -> None:
    plan = suggest_b1i_band(2.6e6, 1575.42e6, {"G", "E", "J", "C"})
    assert plan is not None
    assert plan["center_freq"] == pytest.approx(1568.0e6)
    assert plan["fs"] == pytest.approx(30.0e6)
    assert plan["fits"] is True
    assert b210_band_ok(plan["fs"], plan["center_freq"]) is True
    assert plan["fs"] / 2.0 >= plan["offset_hz"] + 1.5e6

    # B1I выключен -> ничего не меняем.
    assert suggest_b1i_band(2.6e6, 1575.42e6, {"G", "E", "J"}) is None
    # Уже помещается -> ничего не меняем.
    assert suggest_b1i_band(30.0e6, 1568.0e6, {"C"}) is None
    # Недопустимое предложение вне лимитов B210 -> None.
    assert suggest_b1i_band(2.6e6, 1575.42e6, {"C"}, center=10.0e6) is None


def test_b210_limits() -> None:
    assert b210_band_ok(56.0e6, 1575.42e6) is True
    assert b210_band_ok(56.0e6 + 1, 1575.42e6) is False
    assert b210_band_ok(30.0e6, B210_MIN_CENTER) is True
    assert b210_band_ok(30.0e6, B210_MIN_CENTER - 1) is False
    assert b210_band_ok(30.0e6, B210_MAX_CENTER) is True
    assert b210_band_ok(30.0e6, B210_MAX_CENTER + 1) is False
    assert B210_MAX_FS == 56.0e6


def _runner(fs: float, center: float, enable_beidou: bool = True,
            auto_b1i: bool = True):
    cfg = SimConfig(fs=fs, center_freq=center,
                    enable_beidou=enable_beidou, auto_b1i=auto_b1i)
    logs: list[str] = []
    return cfg, SimulationRunner(cfg, log=logs.append), logs


def test_runner_auto_b1i_applies() -> None:
    cfg, runner, logs = _runner(2.6e6, 1575.42e6)
    runner._apply_b1i_band()
    assert cfg.fs == pytest.approx(30.0e6)
    assert cfg.center_freq == pytest.approx(1568.0e6)
    assert any("Авто-подбор под B1I" in m for m in logs)


def test_runner_auto_b1i_disabled_warns_but_keeps_values() -> None:
    cfg, runner, logs = _runner(2.6e6, 1575.42e6, auto_b1i=False)
    runner._apply_b1i_band()
    assert cfg.fs == 2.6e6 and cfg.center_freq == 1575.42e6
    assert any("авто-подбор отключён" in m for m in logs)


def test_runner_auto_b1i_noop_when_disabled_or_fits() -> None:
    cfg, runner, logs = _runner(2.6e6, 1575.42e6, enable_beidou=False)
    runner._apply_b1i_band()
    assert cfg.fs == 2.6e6 and cfg.center_freq == 1575.42e6
    assert logs == []

    cfg, runner, logs = _runner(30.0e6, 1568.0e6)
    runner._apply_b1i_band()
    assert cfg.fs == 30.0e6 and cfg.center_freq == 1568.0e6
    assert logs == []


def test_gui_iq_ram_and_auto_b1i_controls() -> None:
    from gnss_sim.gui import MainWindow
    app = _app()
    win = MainWindow()
    try:
        # Авто-подбор включён по умолчанию, галочка «в RAM» выключена.
        assert win.cb_auto_b1i.isChecked() is True
        assert win._collect().auto_b1i is True
        assert win.cb_iq_ram.isChecked() is False
        assert win.cb_iq_ram.isEnabled() is False
        win.cb_iq_in.setChecked(True)
        win.ed_iq_in.setText("reuse.cs16")
        win.cb_iq_ram.setChecked(True)
        cfg = win._collect()
        assert cfg.iq_in_ram is True and cfg.iq_input == "reuse.cs16"
        win._reset_output_group()
        assert win.cb_iq_ram.isChecked() is False
        win._reset_band_group()
        assert win.cb_auto_b1i.isChecked() is True
        # Понятная заметка про merged только за прошлые сутки.
        assert "только за" in win.lbl_cddis_days.text().lower()
    finally:
        win.close()
    del app
