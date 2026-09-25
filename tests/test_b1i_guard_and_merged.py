"""Focused tests for the B1I band guard and the merged start-date fallback.

Offline (no hardware, no network)::

    E:\\MySoftware\\SDR_Scan\\.venv\\Scripts\\python.exe -m pytest tests -q
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest

from gnss_sim import rinex, rinexfetch
from gnss_sim.config import SimConfig
from gnss_sim.constants import R2D
from gnss_sim.engine import visible_beidou_count
from gnss_sim.gpstime import date2gps, gps2date
from gnss_sim.orbit import llh2xyz
from gnss_sim.runner import SimulationRunner

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NAV = os.path.join(_ROOT, "brdc2680.26n")
_MERGED = os.path.join(_ROOT, "rinex_cache",
                       "BRDC00IGS_R_20262660000_01D_MN.rnx")


def _xyz_fn():
    xyz = llh2xyz(35.681298 / R2D, 139.766247 / R2D, 10.0)
    return lambda g: xyz


def _runner(fs: float = 2.6e6, center: float = 1575.42e6):
    cfg = SimConfig(fs=fs, center_freq=center, enable_beidou=True)
    logs: list[str] = []
    return cfg, SimulationRunner(cfg, log=logs.append), logs


# ======================================================================
# (1) B1I auto-band guard: only widen when B1I channels would exist
# ======================================================================
def test_apply_b1i_band_gps_only_keeps_band() -> None:
    by_sv, iono = rinex.parse_nav_file(_NAV)
    cfg, runner, logs = _runner()
    runner._start = date2gps(2026, 9, 24, 0, 0, 0)
    runner._apply_b1i_band(by_sv, iono, _xyz_fn())
    assert cfg.fs == 2.6e6 and cfg.center_freq == 1575.42e6
    assert any("не найден" in m for m in logs)
    assert not any("Авто-подбор" in m for m in logs)


def test_apply_b1i_band_present_but_not_visible_keeps_band(monkeypatch) -> None:
    monkeypatch.setattr("gnss_sim.runner.visible_beidou_count",
                        lambda *a, **k: 0)
    cfg, runner, logs = _runner()
    runner._start = date2gps(2026, 9, 23, 12, 0, 0)
    runner._apply_b1i_band({"C01": [object()]}, object(), _xyz_fn())
    assert cfg.fs == 2.6e6 and cfg.center_freq == 1575.42e6
    assert any("видимых" in m for m in logs)
    assert not any("Авто-подбор" in m for m in logs)


def test_apply_b1i_band_visible_switches(monkeypatch) -> None:
    monkeypatch.setattr("gnss_sim.runner.visible_beidou_count",
                        lambda *a, **k: 1)
    cfg, runner, logs = _runner()
    runner._start = date2gps(2026, 9, 23, 12, 0, 0)
    runner._apply_b1i_band({"C01": [object()]}, object(), _xyz_fn())
    assert cfg.fs == pytest.approx(30.0e6)
    assert cfg.center_freq == pytest.approx(1568.0e6)
    assert any("Авто-подбор" in m for m in logs)


@pytest.mark.skipif(not os.path.exists(_MERGED), reason="merged cache missing")
def test_apply_b1i_band_real_merged_switches() -> None:
    by_sv, iono = rinex.parse_nav_file(_MERGED)
    start = date2gps(2026, 9, 23, 12, 0, 0)
    assert visible_beidou_count(by_sv, iono, _xyz_fn(), start, 5.0 / R2D) > 0
    cfg, runner, logs = _runner()
    runner._start = start
    runner._apply_b1i_band(by_sv, iono, _xyz_fn())
    assert cfg.fs == pytest.approx(30.0e6)
    assert cfg.center_freq == pytest.approx(1568.0e6)
    assert any("Авто-подбор" in m for m in logs)


# ======================================================================
# (2) Latest merged day: reusable helpers (offline, cache only)
# ======================================================================
def _touch(cache, name: str) -> None:
    cache.joinpath(name).write_text("x")


def test_latest_merged_date_prefers_cache_then_yesterday(tmp_path) -> None:
    now = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)
    assert (rinexfetch.latest_merged_date(now=now, cache_dir=str(tmp_path))
            == now.date() - timedelta(days=1))

    doy = rinexfetch.doy_from_date(2026, 9, 20)
    _touch(tmp_path, f"BRDC00IGS_R_2026{doy:03d}0000_01D_MN.rnx")
    doy = rinexfetch.doy_from_date(2026, 9, 22)
    _touch(tmp_path, f"brdc{doy:03d}0.26n")  # per-system is ignored
    assert (rinexfetch.latest_merged_date(now=now, cache_dir=str(tmp_path))
            .isoformat() == "2026-09-20")


def test_latest_merged_start_moves_and_preserves_time(tmp_path) -> None:
    now = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)
    doy = rinexfetch.doy_from_date(2026, 9, 20)
    _touch(tmp_path, f"BRDC00IGS_R_2026{doy:03d}0000_01D_MN.rnx")

    start = date2gps(2026, 9, 25, 13, 45, 30)
    moved = rinexfetch.latest_merged_start(start, now=now,
                                           cache_dir=str(tmp_path))
    y, m, d, hh, mi, _ss = gps2date(moved)
    assert (y, m, d) == (2026, 9, 20)
    assert (hh, mi) == (13, 45)

    # Already on a merged day -> no move.
    on = date2gps(2026, 9, 20, 13, 45, 30)
    assert rinexfetch.latest_merged_start(on, now=now,
                                          cache_dir=str(tmp_path)) is None

    # Fallback (yesterday) equals the requested date -> no move.
    empty = tmp_path / "empty"
    empty.mkdir()
    yest = date2gps(2026, 9, 24, 1, 2, 3)
    assert rinexfetch.latest_merged_start(yest, now=now,
                                          cache_dir=str(empty)) is None


def test_runner_maybe_move_to_merged_start(monkeypatch) -> None:
    moved = date2gps(2026, 9, 22, 10, 0, 0)
    monkeypatch.setattr(rinexfetch, "latest_merged_start",
                        lambda *a, **k: moved)
    cfg = SimConfig(nav_mode="merged", start_text="2026/09/25,10:00:00")
    logs: list[str] = []
    runner = SimulationRunner(cfg, log=logs.append)
    runner._start = date2gps(2026, 9, 25, 10, 0, 0)
    runner._maybe_move_to_merged_start()
    y, m, d, *_ = gps2date(runner._start)
    assert (y, m, d) == (2026, 9, 22)
    assert any("переведено" in m for m in logs)


def test_runner_explicit_nav_file_is_respected(monkeypatch) -> None:
    called: list[int] = []
    monkeypatch.setattr(rinexfetch, "latest_merged_start",
                        lambda *a, **k: called.append(1) or date2gps(2026, 9, 22))
    cfg = SimConfig(nav_mode="merged", nav_file="explicit.rnx")
    runner = SimulationRunner(cfg, log=lambda m: None)
    runner._start = date2gps(2026, 9, 25, 10, 0, 0)
    runner._maybe_move_to_merged_start()
    assert called == []


def test_cli_nav_mode_argument() -> None:
    from gnss_sim.cli import build_parser
    parser = build_parser()
    assert parser.parse_args([]).nav_mode == "auto"
    assert parser.parse_args(["--nav-mode", "merged"]).nav_mode == "merged"
