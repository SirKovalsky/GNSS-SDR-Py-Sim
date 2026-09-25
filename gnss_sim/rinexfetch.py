"""Automatic download of daily broadcast ephemerides (BKG / CDDIS).

Primary source: BKG (public, multi-GNSS merged RINEX 3, files under a per-day
folder).  Secondary / explicit source: CDDIS (Earthdata) — supports the merged
file and the classic GPS/GLONASS RINEX 2 daily files, authenticated with a
login or a bearer token.

CDDIS has **no per-day folders**: the daily files live directly in
``.../gnss/data/daily/{year}/brdc/``::

    BRDC00IGS_R_{year}{doy:03d}0000_01D_MN.rnx.gz   (merged multi-GNSS)
    brdc{doy:03d}0.{yy:02d}n.gz                     (GPS RINEX 2)
    brdc{doy:03d}0.{yy:02d}g.gz                     (GLONASS RINEX 2)
    brdc{doy:03d}0.{yy:02d}l.gz                     (Galileo RINEX 3)
    brdc{doy:03d}0.{yy:02d}c.gz                     (BeiDou RINEX 3)
    brdc{doy:03d}0.{yy:02d}j.gz                     (QZSS RINEX 3)

The merged multi-system file is only published for **past** days, so for the
current UTC day the per-system set ``brdc{DOY}0.{yy}{n,g,l,c,j}`` is used and
all available files are downloaded and parsed together.  BKG daily files are
published with a roughly one-day lag and CDDIS with a similar lag, so
unavailable days (HTTP 404) are skipped quietly and several previous days are
tried.  Downloaded files are cached locally and decompressed (``.gz`` -> plain
RINEX).
"""

from __future__ import annotations

import base64
import gzip
import os
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta, timezone

from .gpstime import (
    GpsTime,
    date2gps,
    gps2date,
    inc_gps_time,
    sub_gps_time,
)

DEFAULT_CACHE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rinex_cache")

#: Local prefix of the merged multi-GNSS RINEX (published for past days only).
MERGED_RINEX_PREFIX = "BRDC00IGS_R_"

_UA = {"User-Agent": "gnss-sim/0.2 (+https://example.invalid)"}
_CDDIS_HOSTS = ("https://cddis.nasa.gov/", "https://urs.earthdata.nasa.gov/")
_CDDIS_BASE = "https://cddis.nasa.gov/archive/gnss/data/daily"

#: CDDIS per-system daily RINEX suffixes (``brdc{DOY}0.{yy}{suffix}.gz``):
#: n=GPS, g=GLONASS, l=Galileo, c=BeiDou, j=QZSS.
CDDIS_SYSTEM_SUFFIXES = ("n", "g", "l", "c", "j")
#: Human-readable constellation labels for the suffixes above.
CDDIS_SYSTEM_LABELS = {"n": "GPS", "g": "GLONASS", "l": "Galileo",
                       "c": "BeiDou", "j": "QZSS"}


# ----------------------------------------------------------------------
# Calendar helpers
# ----------------------------------------------------------------------
def _is_leap(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def doy_from_date(year: int, month: int, day: int) -> int:
    """Calendar date -> day-of-year (1..366), leap years handled."""
    if not 1 <= month <= 12:
        raise ValueError(f"Некорректный месяц: {month}")
    return int(date(int(year), int(month), int(day)).timetuple().tm_yday)


def date_from_doy(year: int, doy: int) -> tuple[int, int, int]:
    """Day-of-year -> ``(year, month, day)`` (round-trip with the above)."""
    if not 1 <= int(doy) <= (366 if _is_leap(int(year)) else 365):
        raise ValueError(f"Некорректный doy={doy} для {year}")
    dt = date(int(year), 1, 1) + timedelta(days=int(doy) - 1)
    if dt.year != int(year):  # pragma: no cover - защита
        raise ValueError(f"doy={doy} вне года {year}")
    return dt.year, dt.month, dt.day


def _shift_date(year: int, month: int, day: int,
                delta_days: int) -> tuple[int, int, int]:
    dt = date(int(year), int(month), int(day)) + timedelta(days=int(delta_days))
    return dt.year, dt.month, dt.day


def _doy(year: int, month: int, day: int) -> int:  # backwards compatible name
    return doy_from_date(year, month, day)


# ----------------------------------------------------------------------
# URL / filename construction
# ----------------------------------------------------------------------
def _daily_names(year: int, doy: int) -> list[str]:
    """Local (decompressed) file names: merged then per-system RINEX 2/3.

    Index order is stable (merged first; then GPS, GLONASS, Galileo, BeiDou,
    QZSS) so callers that index positionally keep working.
    """
    yy = year % 100
    return [
        f"BRDC00IGS_R_{year}{doy:03d}0000_01D_MN.rnx",
        f"brdc{doy:03d}0.{yy:02d}n",
        f"brdc{doy:03d}0.{yy:02d}g",
        f"brdc{doy:03d}0.{yy:02d}l",
        f"brdc{doy:03d}0.{yy:02d}c",
        f"brdc{doy:03d}0.{yy:02d}j",
    ]


def cddis_system_names(year: int, doy: int) -> list[str]:
    """Per-system CDDIS local names ``brdc{DOY}0.{yy}{n,g,l,c,j}``."""
    yy = year % 100
    return [f"brdc{doy:03d}0.{yy:02d}{sfx}" for sfx in CDDIS_SYSTEM_SUFFIXES]


def cddis_system_urls(year: int, doy: int) -> list[str]:
    """CDDIS per-system daily URLs (used for the current UTC day)."""
    base = f"{_CDDIS_BASE}/{year}/brdc"
    return [f"{base}/{name}.gz" for name in cddis_system_names(year, doy)]


def cddis_daily_urls(year: int, doy: int) -> list[str]:
    """CDDIS daily URLs: merged multi-GNSS first, then GPS/GLONASS RINEX 2.

    CDDIS stores the daily broadcast files directly in ``.../{year}/brdc/``
    (there are **no** per-DOY sub-folders).  The merged multi-system file and
    the classic ``.n``/``.g`` files are tried first; the full per-system set
    (``.l``/``.c``/``.j``) is used separately for the current day via
    :func:`cddis_system_urls`.
    """
    base = f"{_CDDIS_BASE}/{year}/brdc"
    names = _daily_names(year, doy)[:3]
    return [f"{base}/{name}.gz" for name in names]


def _requested_is_today(year: int, doy: int, now=None) -> bool:
    """True when ``year``/``doy`` is the current UTC calendar day."""
    now_dt = now or datetime.now(timezone.utc)
    if now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=timezone.utc)
    now_dt = now_dt.astimezone(timezone.utc)
    return date(int(year), 1, 1) + timedelta(days=int(doy) - 1) == now_dt.date()


def _bkg_urls(year: int, doy: int) -> list[str]:
    name = f"BRDC00IGS_R_{year}{doy:03d}0000_01D_MN.rnx.gz"
    return [f"https://igs.bkg.bund.de/root_ftp/IGS/BRDC/{year}/{doy:03d}/{name}"]


def _urls(year: int, doy: int) -> list[str]:
    """BKG then CDDIS (legacy helper, kept for compatibility)."""
    return _bkg_urls(year, doy) + cddis_daily_urls(year, doy)


# ----------------------------------------------------------------------
# Download helpers
# ----------------------------------------------------------------------
def _logf(log, msg: str) -> None:
    if log is not None:
        log(msg)


def _short_error(exc: BaseException) -> str:
    """Compact error text (avoid dumping full HTTP tracebacks into the log)."""
    if isinstance(exc, urllib.error.HTTPError):
        return f"HTTP {exc.code}"
    return str(exc)


def _report_systems(path, log) -> None:
    """Log which constellations the selected RINEX file(s) actually provide.

    ``path`` may be a single path or a list/tuple of per-system files; the
    counts are aggregated across all of them.
    """
    paths = list(path) if isinstance(path, (list, tuple)) else [path]
    counts: dict[str, int] = {}
    try:
        from .rinex import count_systems
        for one in paths:
            sub = count_systems(str(one))
            for system, n in sub.items():
                counts[system] = counts.get(system, 0) + int(n)
    except Exception:  # noqa: BLE001 - отчёт не должен ломать выбор файла
        return
    if not counts:
        return
    summary = ", ".join(f"{s}:{counts[s]}" for s in "GEJCRS" if s in counts)
    _logf(log, f"Системы в файле: {summary}")
    if not (set(counts) & {"E", "J", "C"}):
        _logf(log, "ВНИМАНИЕ: файл только GPS (для Galileo/BeiDou нужен merged "
                   "BRDC00IGS_R_…)")


def cached_nav_for_date(year: int, doy: int,
                        cache_dir: str | None = None) -> str | None:
    """Return the first cached broadcast file for ``year``/``doy`` if present."""
    cache = cache_dir or DEFAULT_CACHE
    for name in _daily_names(year, doy):
        path = os.path.join(cache, name)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return path
    return None


def cached_nav_set_for_date(year: int, doy: int, cache_dir: str | None = None
                            ) -> str | list[str] | None:
    """Return the cached broadcast file set for ``year``/``doy``.

    The merged multi-system file (published for past days) is preferred and
    returned as a single path; otherwise every present per-system file is
    returned as a list (the current-day case).  ``None`` when nothing is
    cached.
    """
    cache = cache_dir or DEFAULT_CACHE
    names = _daily_names(year, doy)
    merged = os.path.join(cache, names[0])
    if os.path.exists(merged) and os.path.getsize(merged) > 0:
        return merged
    paths = []
    for name in names[1:]:
        path = os.path.join(cache, name)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            paths.append(path)
    if len(paths) == 1:
        return paths[0]
    if paths:
        return paths
    return None


# ----------------------------------------------------------------------
# Latest merged multi-GNSS day (offline; used by GUI and runner/CLI)
# ----------------------------------------------------------------------
def _is_merged_path(path) -> bool:
    return (isinstance(path, str)
            and os.path.basename(path).startswith(MERGED_RINEX_PREFIX))


def latest_merged_date(now=None, cache_dir: str | None = None,
                       max_back_days: int = 7):
    """Most recent UTC date (< today) with a cached merged multi-GNSS RINEX.

    Scans back up to ``max_back_days`` (``<=7``) for a cached merged file
    (``BRDC00IGS_R_…_MN.rnx``); when none is cached it falls back to UTC
    yesterday.  Never downloads — callers may stay offline.
    """
    now_dt = now or datetime.now(timezone.utc)
    if now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=timezone.utc)
    today = now_dt.astimezone(timezone.utc).date()
    for back in range(1, max(1, int(max_back_days)) + 1):
        day = today - timedelta(days=back)
        try:
            doy = doy_from_date(day.year, day.month, day.day)
            cached = cached_nav_set_for_date(day.year, doy, cache_dir)
        except Exception:  # noqa: BLE001
            continue
        if _is_merged_path(cached):
            return day
    return today - timedelta(days=1)


def has_merged_for_start(start: GpsTime, cache_dir: str | None = None) -> bool:
    """True when a merged multi-GNSS RINEX is cached for ``start``'s date."""
    try:
        y, m, d, _hh, _mm, _ss = gps2date(start)
        doy = doy_from_date(y, m, d)
        return _is_merged_path(cached_nav_set_for_date(y, doy, cache_dir))
    except Exception:  # noqa: BLE001
        return False


def merged_start_for_date(start: GpsTime, day, cache_dir: str | None = None,
                          margin_hours: float = 6.0) -> GpsTime:
    """Move ``start`` to ``day`` keeping the time-of-day.

    When the merged file for ``day`` is cached and parseable, the result is
    clamped into its ephemeris toe span (±``margin_hours``) so it stays inside
    :func:`gnss_sim.rinex.check_start_coverage`.
    """
    _y, _m, _d, hh, mm, ss = gps2date(start)
    moved = date2gps(day.year, day.month, day.day, hh, mm, ss)
    try:
        doy = doy_from_date(day.year, day.month, day.day)
        path = cached_nav_set_for_date(day.year, doy, cache_dir)
        if _is_merged_path(path):
            from .rinex import ephemeris_toe_span, parse_nav_file
            by_sv, _iono = parse_nav_file(str(path))
            span = ephemeris_toe_span(by_sv)
            if span is not None:
                lo, hi = span
                margin = max(0.0, float(margin_hours)) * 3600.0
                if sub_gps_time(moved, lo) < -margin:
                    moved = inc_gps_time(lo, -margin)
                elif sub_gps_time(moved, hi) > margin:
                    moved = inc_gps_time(hi, margin)
    except Exception:  # noqa: BLE001 - clamp is best effort
        pass
    return moved


def latest_merged_start(start: GpsTime, now=None,
                        cache_dir: str | None = None,
                        max_back_days: int = 7,
                        margin_hours: float = 6.0) -> GpsTime | None:
    """Return ``start`` moved to the latest merged-RINEX day, or ``None``.

    ``None`` means no move is needed: either the requested date already has a
    merged file, or the fallback day equals the requested date.  No network.
    """
    if has_merged_for_start(start, cache_dir):
        return None
    day = latest_merged_date(now=now, cache_dir=cache_dir,
                             max_back_days=max_back_days)
    y, m, d, _hh, _mm, _ss = gps2date(start)
    if (day.year, day.month, day.day) == (y, m, d):
        return None
    return merged_start_for_date(start, day, cache_dir=cache_dir,
                                 margin_hours=margin_hours)


def _download(url: str, dest: str, timeout: float = 60.0) -> None:
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    if url.endswith(".gz"):
        data = gzip.decompress(data)
    tmp = dest + ".part"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, dest)


def _make_cddis_opener(username: str = "", password: str = ""):
    """Opener with cookie jar and (optionally) HTTP Basic credentials."""
    handlers: list[urllib.request.BaseHandler] = [
        urllib.request.HTTPCookieProcessor()]
    if username and password:
        mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
        for uri in _CDDIS_HOSTS:
            mgr.add_password(None, uri, username, password)
        handlers.append(urllib.request.HTTPBasicAuthHandler(mgr))
    return urllib.request.build_opener(*handlers)


def _download_cddis_systems(year: int, doy: int, opener, headers,
                            cache: str, log=None) -> list[str]:
    """Download every available per-system CDDIS daily file.

    Returns the local paths that were downloaded (or already cached).  HTTP 404
    responses are counted and logged **once** instead of per URL.
    """
    names = cddis_system_names(year, doy)
    urls = cddis_system_urls(year, doy)
    paths: list[str] = []
    not_found = 0
    errors: list[str] = []
    for name, url in zip(names, urls):
        local = os.path.join(cache, name)
        if os.path.exists(local) and os.path.getsize(local) > 0:
            _logf(log, f"CDDIS из кэша: {local}")
            paths.append(local)
            continue
        try:
            req = urllib.request.Request(url, headers=headers)
            with opener.open(req, timeout=90.0) as resp:
                data = resp.read()
            if url.endswith(".gz"):
                data = gzip.decompress(data)
            if not data:
                raise RuntimeError("пустой ответ")
            tmp = local + ".part"
            with open(tmp, "wb") as fh:
                fh.write(data)
            os.replace(tmp, local)
            _logf(log, f"CDDIS: сохранено {local} ({len(data)} байт)")
            paths.append(local)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                not_found += 1
            else:
                errors.append(f"{name}: {_short_error(exc)}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {_short_error(exc)}")
    if not_found:
        _logf(log, f"CDDIS: {not_found} посистемных файл(ов) ещё не "
                   f"опубликовано (404)")
    if errors:
        _logf(log, "CDDIS: ошибки: " + "; ".join(errors))
    return paths


def download_cddis(year: int, doy: int, username: str = "", password: str = "",
                   token: str = "", cache_dir: str | None = None,
                   log=None, now=None) -> str | list[str]:
    """Download CDDIS daily broadcast file(s) and return the local path(s).

    Authenticates with Earthdata HTTP Basic (``username``/``password``) or a
    bearer ``token``.  CDDIS redirects to ``urs.earthdata.nasa.gov`` and back;
    cookies are kept by the opener.  ``.gz`` files are decompressed into
    ``cache_dir``.

    The merged multi-system file ``BRDC00IGS_R_…_MN.rnx`` is only published for
    **past** days, so it is preferred there and a single ``str`` is returned.
    For the **current UTC day** the per-system set
    ``brdc{DOY}0.{yy}{n,g,l,c,j}`` is used instead and a ``list`` of the
    successfully downloaded paths is returned.

    Raises ``ValueError`` when no credentials are given and ``RuntimeError``
    when no candidate file could be retrieved.
    """
    cache = cache_dir or DEFAULT_CACHE
    os.makedirs(cache, exist_ok=True)

    if not (token or (username and password)):
        raise ValueError(
            "Для скачивания с CDDIS нужны логин/пароль Earthdata или токен")

    merged = os.path.join(cache, _daily_names(year, doy)[0])
    if os.path.exists(merged) and os.path.getsize(merged) > 0:
        _logf(log, f"CDDIS из кэша: {merged}")
        return merged

    if token:
        _logf(log, "CDDIS: аутентификация по bearer-токену")
    else:
        _logf(log, f"CDDIS: аутентификация {username} (HTTP Basic)")
    opener = _make_cddis_opener(username, password)

    headers = dict(_UA)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    elif username:
        raw = f"{username}:{password}".encode("utf-8")
        headers["Authorization"] = "Basic " + base64.b64encode(raw).decode("ascii")

    yy = year % 100
    if _requested_is_today(year, doy, now=now):
        _logf(log, "CDDIS: мультисистемный BRDC00IGS доступен только за "
                   "прошедшие сутки; для текущей даты берём посистемные "
                   f"brdc{doy:03d}0.{yy:02d}{{n,g,l,c,j}}")
        paths = _download_cddis_systems(year, doy, opener, headers, cache, log)
        if paths:
            return paths
        raise RuntimeError(
            f"CDDIS: за текущую дату ({year} doy {doy:03d}) не удалось "
            f"скачать ни одного посистемного файла")

    # Past day: merged multi-GNSS first, per-system set as a fallback.
    urls = cddis_daily_urls(year, doy)
    names = _daily_names(year, doy)[:3]
    last_error: Exception | None = None
    for name, url in zip(names, urls):
        local = os.path.join(cache, name)
        if os.path.exists(local) and os.path.getsize(local) > 0:
            _logf(log, f"CDDIS из кэша: {local}")
            return local
        try:
            _logf(log, f"CDDIS: скачивание {url}")
            req = urllib.request.Request(url, headers=headers)
            with opener.open(req, timeout=90.0) as resp:
                data = resp.read()
            if url.endswith(".gz"):
                data = gzip.decompress(data)
            if not data:
                raise RuntimeError("пустой ответ")
            tmp = local + ".part"
            with open(tmp, "wb") as fh:
                fh.write(data)
            os.replace(tmp, local)
            _logf(log, f"CDDIS: сохранено {local} ({len(data)} байт)")
            return local
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code != 404:
                _logf(log, f"  не удалось: {_short_error(exc)}")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            _logf(log, f"  не удалось: {_short_error(exc)}")

    _logf(log, "CDDIS: merged-файл недоступен — пробуем посистемные "
               f"brdc{doy:03d}0.{yy:02d}{{n,g,l,c,j}}")
    paths = _download_cddis_systems(year, doy, opener, headers, cache, log)
    if paths:
        return paths
    raise RuntimeError(
        f"Не удалось скачать с CDDIS за {year} doy {doy:03d}: "
        f"{_short_error(last_error) if last_error else 'нет данных'}")


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------
def ensure_nav_file(start: GpsTime, cache_dir: str | None = None,
                    log=None, max_back_days: int = 7, source: str = "auto",
                    cddis_user: str = "", cddis_password: str = "",
                    cddis_token: str = "", force: bool = False,
                    now=None) -> str | list[str]:
    """Return a local path (or per-system set) to broadcast RINEX for ``start``.

    ``source``: ``auto`` (CDDIS first when credentials are present, then BKG),
    ``bkg`` (BKG only) or ``cddis`` (CDDIS only).  ``force`` re-downloads the
    requested day even when a cached file exists.  HTTP 404 responses (files
    not published yet) are counted, not logged per-URL, to avoid log spam.

    For a CDDIS **past** date the merged multi-system file is returned as a
    single ``str``; for the **current UTC day** the per-system set
    (``brdc{DOY}0.{yy}{n,g,l,c,j}``) is returned as a ``list`` (see
    :func:`parse_nav_file`, which accepts such a list).  Raises ``RuntimeError``
    when no file could be retrieved.
    """
    cache = cache_dir or DEFAULT_CACHE
    os.makedirs(cache, exist_ok=True)

    source = (source or "auto").lower()
    if source not in ("auto", "bkg", "cddis"):
        raise ValueError(f"Неизвестный источник эфемерид: {source!r}")

    has_creds = bool((cddis_user and cddis_password) or cddis_token)
    if source == "cddis" and not has_creds:
        raise RuntimeError(
            "Источник CDDIS требует логина/пароля Earthdata или токена")
    if source == "bkg":
        order = ["bkg"]
    elif source == "cddis":
        order = ["cddis"]
    else:  # auto: CDDIS first when credentials are available, else BKG
        order = ["cddis", "bkg"] if has_creds else ["bkg"]

    y, m, d, _hh, _mm, _ss = gps2date(start)
    base_doy = doy_from_date(y, m, d)

    last_error: Exception | None = None
    not_found = 0
    for back in range(max_back_days):
        yy, mm, dd = _shift_date(y, m, d, -back)
        doy = doy_from_date(yy, mm, dd)

        if not (force and back == 0):
            cached = cached_nav_set_for_date(yy, doy, cache)
            if cached is not None:
                if isinstance(cached, (list, tuple)):
                    _logf(log, "Эфемериды из кэша (посистемные): "
                               + ", ".join(os.path.basename(p)
                                           for p in cached))
                else:
                    _logf(log, f"Эфемериды из кэша: {cached}")
                return cached

        for tag in order:
            if tag == "bkg":
                merged = os.path.join(
                    cache, f"BRDC00IGS_R_{yy}{doy:03d}0000_01D_MN.rnx")
                for url in _bkg_urls(yy, doy):
                    try:
                        _logf(log, f"Скачивание RINEX (BKG): {url}")
                        _download(url, merged)
                        if os.path.getsize(merged) > 0:
                            _logf(log, f"Сохранено: {merged} "
                                       f"({os.path.getsize(merged)} байт)")
                            return merged
                    except urllib.error.HTTPError as exc:
                        last_error = exc
                        if exc.code == 404:
                            not_found += 1
                        else:
                            _logf(log, f"BKG: {_short_error(exc)} — {url}")
                    except Exception as exc:  # noqa: BLE001
                        last_error = exc
                        _logf(log, f"BKG: не удалось ({_short_error(exc)})")
            else:  # cddis
                try:
                    return download_cddis(
                        yy, doy, username=cddis_user, password=cddis_password,
                        token=cddis_token, cache_dir=cache, log=log, now=now)
                except urllib.error.HTTPError as exc:
                    last_error = exc
                    if exc.code == 404:
                        not_found += 1
                    else:
                        _logf(log, f"CDDIS: {_short_error(exc)} за "
                                   f"{yy} doy {doy:03d}")
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    _logf(log, f"CDDIS: не удалось за {yy} doy {doy:03d}: "
                               f"{_short_error(exc)}")

    if not_found:
        _logf(log, f"Источник эфемерид: {not_found} запрос(ов) вернули 404 — "
                   f"файлы публикуются с задержкой 1–2 суток (это нормально)")
    raise RuntimeError(
        f"Не удалось скачать RINEX для GPS week {start.week}, "
        f"doy {base_doy}: {last_error}")


# ----------------------------------------------------------------------
# Reuse / caching policy used by the runner
# ----------------------------------------------------------------------
def resolve_nav_file(start: GpsTime, cfg, ask=None, log=None,
                     cache_dir: str | None = None, now=None,
                     max_age_hours: float = 1.0,
                     max_back_days: int = 7) -> str | list[str]:
    """Return the RINEX path (or per-system set) to use for ``start``.

    Policy:

    * an explicitly configured ``cfg.nav_file`` that exists is always used
      (a ``;``-separated list is accepted for the CDDIS per-system set);
    * a cached file for a **past** requested date is reused as-is; for the
      current day the merged file is preferred, otherwise the cached
      per-system set is returned as a list;
    * for **today** a cached file younger than ``max_age_hours`` is reused;
      when older, ``ask(path, age_seconds)`` is called (GUI dialog / user
      decision).  ``ask`` returns ``True`` to re-download.  When ``ask`` is
      ``None`` (CLI) the existing file is reused and logged;
    * with no cached file the file is downloaded via :func:`ensure_nav_file`.

    ``cfg`` is a :class:`gnss_sim.config.SimConfig` (duck-typed to keep this
    module import-light).
    """
    cache = cache_dir or DEFAULT_CACHE
    os.makedirs(cache, exist_ok=True)

    def _use(path) -> str | list[str]:
        if isinstance(path, (list, tuple)):
            _logf(log, "CDDIS: используем посистемные файлы: "
                       + ", ".join(os.path.basename(p) for p in path))
        _report_systems(path, log)
        return path

    nav = str(getattr(cfg, "nav_file", "") or "").strip()
    if nav:
        parts = [p.strip() for p in nav.split(";") if p.strip()]
        if len(parts) > 1:
            if all(os.path.exists(p) for p in parts):
                _logf(log, f"Используем указанные файлы эфемерид: "
                           f"{len(parts)} шт.")
                return _use(parts)
            if not getattr(cfg, "auto_download", True):
                raise ValueError(f"Файлы эфемерид не найдены: {nav}")
        elif os.path.exists(nav):
            _logf(log, f"Используем указанный файл эфемерид: {nav}")
            return _use(nav)
        elif not getattr(cfg, "auto_download", True):
            raise ValueError(f"Файл эфемерид не найден: {nav}")

    if not getattr(cfg, "auto_download", True):
        raise ValueError(
            "Не задан файл эфемерид (RINEX navigation); автоскачивание "
            "выключено")

    source = str(getattr(cfg, "download_source", "auto") or "auto")
    creds = dict(
        source=source,
        cddis_user=getattr(cfg, "cddis_user", "") or "",
        cddis_password=getattr(cfg, "cddis_password", "") or "",
        cddis_token=getattr(cfg, "cddis_token", "") or "",
    )

    y, m, d, _hh, _mm, _ss = gps2date(start)
    doy = doy_from_date(y, m, d)

    cached = cached_nav_set_for_date(y, doy, cache)
    if cached is not None:
        now_dt = now or datetime.now(timezone.utc)
        try:
            stamps = [os.path.getmtime(p) for p in (
                cached if isinstance(cached, (list, tuple)) else [cached])]
            mtime = datetime.fromtimestamp(min(stamps), timezone.utc)
            age = max(0.0, (now_dt - mtime).total_seconds())
        except OSError:
            age = 0.0
        requested = date(y, m, d)
        today = now_dt.date()
        shown = (f"посистемные {len(cached)} шт." if isinstance(
            cached, (list, tuple)) else cached)
        if requested < today:
            _logf(log, f"Эфемериды за {y:04d}-{m:02d}-{d:02d} из кэша: {shown}")
            return _use(cached)
        if requested == today and age < max_age_hours * 3600.0:
            _logf(log, f"Свежий кэш за сегодня "
                       f"({age / 60.0:.0f} мин): {shown}")
            return _use(cached)
        update = False
        if ask is not None:
            try:
                update = bool(ask(cached, age))
            except Exception as exc:  # noqa: BLE001 - вопрос не должен ломать
                _logf(log, f"Не удалось спросить про обновление: {exc}")
                update = False
        else:
            _logf(log, f"Кэш старше {max_age_hours:.0f} ч — используем "
                       f"существующий файл (режим без вопросов): {shown}")
        if not update:
            return _use(cached)
        _logf(log, "Обновляем эфемериды по запросу…")
        return _use(ensure_nav_file(start, cache_dir=cache, log=log, force=True,
                                    max_back_days=max_back_days, now=now,
                                    **creds))

    return _use(ensure_nav_file(start, cache_dir=cache, log=log,
                                max_back_days=max_back_days, now=now, **creds))
