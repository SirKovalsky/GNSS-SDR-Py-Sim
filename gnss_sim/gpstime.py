"""GPS time handling (week + seconds-of-week) and calendar conversion."""

from __future__ import annotations

from dataclasses import dataclass

from .constants import SECONDS_IN_WEEK

_DOY = (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334)


@dataclass
class GpsTime:
    week: int = 0
    sec: float = 0.0

    def copy(self) -> "GpsTime":
        return GpsTime(self.week, self.sec)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"GpsTime(week={self.week}, sec={self.sec:.3f})"


def date2gps(year: int, month: int, day: int,
             hour: int = 0, minute: int = 0, second: float = 0.0) -> GpsTime:
    """Calendar date -> GPS week / seconds-of-week (ports gps-sdr-sim)."""
    ye = year - 1980
    lpdays = ye // 4 + 1
    if ye % 4 == 0 and month <= 2:
        lpdays -= 1
    de = ye * 365 + _DOY[month - 1] + day + lpdays - 6
    wk = de // 7
    sec = (de % 7) * 86400.0 + hour * 3600.0 + minute * 60.0 + second
    return GpsTime(int(wk), float(sec))


def gps2date(g: GpsTime) -> tuple[int, int, int, int, int, float]:
    """GPS week / seconds -> (year, month, day, hour, minute, second)."""
    import math

    c = int(7 * g.week + math.floor(g.sec / 86400.0) + 2444245) + 1537
    d = int((c - 122.1) / 365.25)
    e = int(365 * d + d / 4)
    f = int((c - e) / 30.6001)
    day = c - e - int(30.6001 * f)
    month = f - 1 - 12 * (f // 14)
    year = d - 4715 - ((7 + month) // 10)
    hh = int(g.sec / 3600.0) % 24
    mm = int(g.sec / 60.0) % 60
    sec = g.sec - 60.0 * math.floor(g.sec / 60.0)
    return int(year), int(month), int(day), int(hh), int(mm), float(sec)


def sub_gps_time(t1: GpsTime, t0: GpsTime) -> float:
    return (t1.sec - t0.sec) + (t1.week - t0.week) * SECONDS_IN_WEEK


def inc_gps_time(t: GpsTime, dt: float) -> GpsTime:
    """Advance time by dt seconds, wrapping the week."""
    sec = round((t.sec + dt) * 1000.0) / 1000.0
    week = t.week
    if sec >= SECONDS_IN_WEEK:
        week += int(sec // SECONDS_IN_WEEK)
        sec -= SECONDS_IN_WEEK * int(sec // SECONDS_IN_WEEK)
    elif sec < 0.0:
        week -= int(-sec // SECONDS_IN_WEEK) + 1
        sec += SECONDS_IN_WEEK * (int(-sec // SECONDS_IN_WEEK) + 1)
    return GpsTime(week, sec)
