"""RINEX navigation file parser (multi-GNSS).

Supports RINEX version 2.x (GPS, two digit years, ``D`` exponents) and version
3.x (four digit years, sytem-prefixed SV ids such as ``G01``, ``E11``, ``J01``).

GPS (G), Galileo (E) and QZSS (J) broadcast ephemerides share the same Keplerian
record layout and are parsed into :class:`Ephemeris`.  GLONASS (R) and SBAS (S)
records have a different layout and are skipped here (SBAS geostationary
satellites are synthesised by the engine).

Ionospheric / UTC parameters are read when present.
"""

from __future__ import annotations

import gzip
import math
import os
from dataclasses import dataclass, field

from .constants import (
    BDT_GPST_OFFSET_S, BDT_WEEK_OFFSET,
    GM_EARTH,
    POW2_M5, POW2_M19, POW2_M29, POW2_M31, POW2_M33,
    POW2_M43, POW2_M55,
    SECONDS_IN_WEEK,
)
from .gpstime import GpsTime, date2gps, gps2date

#: Systems with an 8-line Keplerian navigation record.
KEPLERIAN_SYSTEMS = ("G", "E", "J", "C")
#: Record lengths (in lines) used to skip unsupported systems.
_RECORD_LINES = {"G": 8, "E": 8, "J": 8, "C": 8, "I": 8, "R": 4, "S": 4}

#: Accepted navigation-file extensions (informational / file dialogs).
NAV_EXTENSIONS = (".rnx", ".n", ".nav", ".gz") + tuple(
    f".{yy:02d}{s}" for yy in range(0, 100) for s in ("n", "g"))
#: Local cache used when decompression next to the source is not possible.
DEFAULT_CACHE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rinex_cache")


def is_gzip(path: str) -> bool:
    """True when ``path`` is gzip-compressed (by magic bytes or extension)."""
    try:
        with open(path, "rb") as fh:
            if fh.read(2) == b"\x1f\x8b":
                return True
    except OSError:
        return False
    return str(path).lower().endswith(".gz")


def decompress_if_needed(path: str, cache_dir: str | None = None) -> str:
    """Transparently decompress a gzip RINEX file and return the plain path.

    The decompressed file is cached next to the source (``foo.26n.gz`` ->
    ``foo.26n``) or, when that location is not writable, inside
    ``rinex_cache/``.  Non-gzip inputs are returned unchanged.
    """
    if not os.path.exists(path):
        return path
    if not (str(path).lower().endswith(".gz") or is_gzip(path)):
        return path

    if str(path).lower().endswith(".gz"):
        target = path[:-3]
    else:
        target = path + ".rnx"
    if os.path.exists(target) and os.path.getsize(target) > 0:
        try:
            if os.path.getmtime(target) >= os.path.getmtime(path):
                return target
        except OSError:
            return target

    try:
        with gzip.open(path, "rb") as src:
            data = src.read()
    except OSError:
        return path  # extension lied; let the normal parser try

    cache = cache_dir or DEFAULT_CACHE
    for candidate in (target, os.path.join(cache, os.path.basename(target))):
        try:
            folder = os.path.dirname(os.path.abspath(candidate))
            os.makedirs(folder, exist_ok=True)
            tmp = candidate + ".part"
            with open(tmp, "wb") as fh:
                fh.write(data)
            os.replace(tmp, candidate)
            return candidate
        except OSError:
            continue
    return path


def sv_key(system: str, prn: int) -> str:
    return f"{system}{prn:02d}"


@dataclass
class Ephemeris:
    system: str = "G"
    prn: int = 0
    toc: GpsTime = field(default_factory=GpsTime)
    toe: GpsTime = field(default_factory=GpsTime)
    iodc: int = 0
    iode: int = 0
    deltan: float = 0.0     # rad/s
    cuc: float = 0.0        # rad
    cus: float = 0.0        # rad
    cic: float = 0.0        # rad
    cis: float = 0.0        # rad
    crc: float = 0.0        # m
    crs: float = 0.0        # m
    ecc: float = 0.0
    sqrta: float = 0.0      # sqrt(m)
    m0: float = 0.0         # rad
    omg0: float = 0.0       # rad
    inc0: float = 0.0       # rad
    aop: float = 0.0        # rad
    omgdot: float = 0.0     # rad/s
    idot: float = 0.0       # rad/s
    af0: float = 0.0        # s
    af1: float = 0.0        # s/s
    af2: float = 0.0        # s/s^2
    tgd: float = 0.0        # s
    svhlth: int = 0
    codeL2: int = 0
    ura: int = 0
    # derived
    n: float = 0.0
    sq1e2: float = 0.0
    A: float = 0.0
    omgkdot: float = 0.0

    def finalize(self) -> "Ephemeris":
        """Compute derived orbital constants."""
        from .constants import OMEGA_EARTH

        self.A = self.sqrta * self.sqrta
        self.n = math.sqrt(GM_EARTH / (self.A ** 3)) + self.deltan
        self.sq1e2 = math.sqrt(1.0 - self.ecc * self.ecc)
        self.omgkdot = self.omgdot - OMEGA_EARTH
        return self

    @property
    def key(self) -> str:
        return sv_key(self.system, self.prn)


@dataclass
class IonoUtc:
    enable: bool = False
    vflg: bool = False
    alpha0: float = 0.0
    alpha1: float = 0.0
    alpha2: float = 0.0
    alpha3: float = 0.0
    beta0: float = 0.0
    beta1: float = 0.0
    beta2: float = 0.0
    beta3: float = 0.0
    A0: float = 0.0
    A1: float = 0.0
    dtls: int = 18
    tot: int = 0
    wnt: int = 0
    dtlsf: int = 0
    dn: int = 7
    wnlsf: int = 0


def _f(s: str) -> float:
    s = s.strip()
    if not s:
        return 0.0
    return float(s.replace("D", "E").replace("d", "e"))


def _i(s: str) -> int:
    s = s.strip()
    if not s:
        return 0
    try:
        return int(float(s.replace("D", "E").replace("d", "e")))
    except ValueError:
        return 0


def _fields(line: str, start: int, width: int, count: int) -> list[str]:
    return [line[start + k * width: start + (k + 1) * width] for k in range(count)]


def _parse_epoch_v2(line: str) -> GpsTime:
    yy = _i(line[3:5])
    year = 2000 + yy if yy < 80 else 1900 + yy
    return date2gps(
        year,
        _i(line[6:8]),
        _i(line[9:11]),
        _i(line[12:14]),
        _i(line[15:17]),
        _f(line[18:22]),
    )


def _parse_epoch_v3(line: str) -> GpsTime:
    return date2gps(
        _i(line[4:8]),
        _i(line[9:11]),
        _i(line[12:14]),
        _i(line[15:17]),
        _i(line[18:20]),
        _f(line[21:23]) if line[21:23].strip() else _f(line[21:24]),
    )


def _orbit(line: str, version: int) -> list[float]:
    if version >= 3:
        return [_f(x) for x in _fields(line, 4, 19, 4)]
    return [_f(x) for x in _fields(line, 3, 19, 4)]


def parse_nav_file(
        path: str | list[str] | tuple[str, ...]
) -> tuple[dict[str, list[Ephemeris]], IonoUtc]:
    """Parse a RINEX navigation file (or merge several of them).

    Returns ``(ephemerides_by_sv, iono_utc)`` where ``ephemerides_by_sv`` is
    keyed by strings such as ``"G01"``, ``"E11"`` or ``"J01"``.  Gzip files
    (by ``.gz`` extension or the ``1f 8b`` magic) are decompressed first.

    ``path`` may be a **list/tuple of paths** (e.g. the CDDIS per-system daily
    files ``brdc{DOY}0.{yy}{n,g,l,c,j}`` for the current day): the parsed
    ``by_sv`` dicts are merged and the first ionospheric/UTC record that is
    valid wins.  A single path behaves exactly as before.
    """
    if isinstance(path, (list, tuple)):
        merged: dict[str, list[Ephemeris]] = {}
        iono: IonoUtc | None = None
        for one in path:
            sub_sv, sub_iono = parse_nav_file(str(one))
            for key, ephs in sub_sv.items():
                merged.setdefault(key, []).extend(ephs)
            if iono is None or (sub_iono.vflg and not iono.vflg):
                iono = sub_iono
        for key in merged:
            merged[key].sort(key=lambda e: (e.toc.week, e.toc.sec))
        return merged, (iono if iono is not None else IonoUtc())

    path = decompress_if_needed(path)
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    version = 2
    if lines:
        try:
            version = int(float(lines[0][:9]))
        except ValueError:
            version = 2

    iono = IonoUtc()
    header_end = 0
    have_alpha = have_beta = have_utc = have_leap = False
    for idx, line in enumerate(lines):
        label = line[60:80].rstrip()
        if label == "END OF HEADER":
            header_end = idx + 1
            break
        try:
            if label == "ION ALPHA" and version < 3:
                v = [_f(x) for x in _fields(line, 2, 12, 4)]
                iono.alpha0, iono.alpha1, iono.alpha2, iono.alpha3 = v
                have_alpha = True
            elif label == "ION BETA" and version < 3:
                v = [_f(x) for x in _fields(line, 2, 12, 4)]
                iono.beta0, iono.beta1, iono.beta2, iono.beta3 = v
                have_beta = True
            elif label == "IONOSPHERIC CORR" and version >= 3:
                kind = line[:4].strip()
                v = [_f(x) for x in _fields(line, 5, 12, 4)]
                if kind == "GPSA":
                    iono.alpha0, iono.alpha1, iono.alpha2, iono.alpha3 = v
                    have_alpha = True
                elif kind == "GPSB":
                    iono.beta0, iono.beta1, iono.beta2, iono.beta3 = v
                    have_beta = True
            elif label == "DELTA-UTC: A0,A1,T,W" and version < 3:
                iono.A0 = _f(line[3:22])
                iono.A1 = _f(line[22:41])
                iono.tot = _i(line[41:50])
                iono.wnt = _i(line[50:59])
                have_utc = True
            elif label == "TIME SYSTEM CORR" and version >= 3:
                if line[:4].strip().startswith("GPS"):
                    iono.A0 = _f(line[5:22])
                    iono.A1 = _f(line[22:38])
                    have_utc = True
            elif label == "LEAP SECONDS":
                iono.dtls = _i(line[0:6])
                have_leap = True
        except (ValueError, IndexError):
            continue

    iono.enable = True
    iono.vflg = have_alpha and have_beta and have_utc and have_leap
    if not iono.vflg:
        iono.alpha0, iono.alpha1, iono.alpha2, iono.alpha3 = (
            0.1118e-7, 0.0, -0.596e-7, 0.0)
        iono.beta0, iono.beta1, iono.beta2, iono.beta3 = (
            0.1208e6, 0.0, 0.0, 0.0)
        iono.dtls = 18
        iono.wnlsf = 1929 % 256
        iono.dn = 7

    by_sv: dict[str, list[Ephemeris]] = {}
    n = len(lines)
    i = header_end
    while i < n:
        line = lines[i]
        if len(line.strip()) < 3:
            i += 1
            continue

        if version >= 3:
            sysch = line[0:1]
            if sysch not in _RECORD_LINES:
                i += 1
                continue
            if sysch not in KEPLERIAN_SYSTEMS:
                i += _RECORD_LINES[sysch]
                continue
            prn = _i(line[1:3])
            toc = _parse_epoch_v3(line)
        else:
            sysch = "G"
            prn = _i(line[0:2])
            toc = _parse_epoch_v2(line)

        block = lines[i:i + 8]
        if len(block) < 8:
            break
        try:
            if version >= 3:
                f0 = [_f(x) for x in _fields(block[0], 23, 19, 3)]
            else:
                f0 = [_f(x) for x in _fields(block[0], 22, 19, 3)]
            o1 = _orbit(block[1], version)
            o2 = _orbit(block[2], version)
            o3 = _orbit(block[3], version)
            o4 = _orbit(block[4], version)
            o5 = _orbit(block[5], version)
            o6 = _orbit(block[6], version)
        except (ValueError, IndexError):
            i += 8
            continue

        eph = Ephemeris()
        eph.system = sysch
        eph.prn = prn
        eph.toc = toc
        eph.af0, eph.af1, eph.af2 = f0
        eph.iode = int(o1[0])
        eph.crs = o1[1]
        eph.deltan = o1[2]
        eph.m0 = o1[3]
        eph.cuc = o2[0]
        eph.ecc = o2[1]
        eph.cus = o2[2]
        eph.sqrta = o2[3]
        toe_sec = o3[0]
        eph.cic = o3[1]
        eph.omg0 = o3[2]
        eph.cis = o3[3]
        eph.inc0 = o4[0]
        eph.crc = o4[1]
        eph.aop = o4[2]
        eph.omgdot = o4[3]
        eph.idot = o5[0]
        eph.codeL2 = int(o5[1])
        toe_week = int(o5[2])
        week = toe_week
        if week < 1000:
            week = toc.week + int(round((toe_sec - toc.sec) / SECONDS_IN_WEEK))
        eph.toe = GpsTime(week, toe_sec)
        eph.ura = int(o6[0])
        eph.svhlth = int(o6[1])
        eph.tgd = o6[2]
        eph.iodc = int(o6[3])
        eph.finalize()
        by_sv.setdefault(eph.key, []).append(eph)
        i += 8

    for key in by_sv:
        by_sv[key].sort(key=lambda e: (e.toc.week, e.toc.sec))
    return by_sv, iono


def select_ephemeris(by_sv: dict[str, list[Ephemeris]], key: str,
                     t: GpsTime, max_age: float = 7200.0) -> Ephemeris | None:
    """Return the ephemeris for ``key`` (e.g. ``"G01"``) closest to ``t``."""
    sets = by_sv.get(key)
    if not sets:
        return None
    target = t.week * SECONDS_IN_WEEK + t.sec
    best = None
    best_dt = None
    for eph in sets:
        toe = eph.toe.week * SECONDS_IN_WEEK + eph.toe.sec
        dt = abs(target - toe)
        if best_dt is None or dt < best_dt:
            best, best_dt = eph, dt
    if best_dt is not None and best_dt < max_age:
        return best
    return best


def _abs_gps(t: GpsTime) -> float:
    return t.week * SECONDS_IN_WEEK + t.sec


def system_counts(by_sv: dict[str, list[Ephemeris]]) -> dict[str, int]:
    """Count ephemeris *records* per constellation (``G``/``E``/``J``/``C``)."""
    counts: dict[str, int] = {}
    for key, sets in by_sv.items():
        system = key[0]
        counts[system] = counts.get(system, 0) + len(sets)
    return counts


def unique_sv_counts(by_sv: dict[str, list[Ephemeris]]) -> dict[str, int]:
    """Count *unique satellites* per constellation (``G``/``E``/``J``/``C``).

    ``by_sv`` is keyed by :func:`sv_key` (e.g. ``"G01"``) and its values are
    the ephemeris records for that SV, so the number of keys per system is the
    number of unique SVs.  This is the human-friendly figure to report, in
    contrast to :func:`system_counts`, which counts records (a merged RINEX
    holds many records per satellite).
    """
    counts: dict[str, int] = {}
    for key in by_sv:
        system = key[0]
        counts[system] = counts.get(system, 0) + 1
    return counts


def count_systems(path: str) -> dict[str, int]:
    """Lightly scan a RINEX file and count records per constellation.

    Unlike :func:`parse_nav_file` this does not decode the records, so it can
    be used to *report* what a file offers before/around a full parse.  Gzip
    inputs are decompressed transparently.  Returns ``{}`` on any failure.
    """
    try:
        path = decompress_if_needed(path)
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except OSError:
        return {}

    version = 2
    if lines:
        try:
            version = int(float(lines[0][:9]))
        except (ValueError, IndexError):
            version = 2
    header_end = 0
    for idx, line in enumerate(lines):
        if line[60:80].rstrip() == "END OF HEADER":
            header_end = idx + 1
            break

    counts: dict[str, int] = {}
    i = header_end
    n = len(lines)
    while i < n:
        line = lines[i]
        if len(line.strip()) < 3:
            i += 1
            continue
        if version >= 3:
            sysch = line[0:1]
            if sysch not in _RECORD_LINES:
                i += 1
                continue
            i += _RECORD_LINES[sysch]
        else:
            sysch = "G"
            i += 8
        counts[sysch] = counts.get(sysch, 0) + 1
    return counts


def ephemeris_toe_gps(eph: Ephemeris) -> GpsTime:
    """Return ``eph.toe`` expressed in GPS time.

    The RINEX BeiDou record stores ``toe`` in **BDT** (week = GPS week − 1356,
    BDT = GPS − 14 s); without normalisation the coverage window of a merged
    RINEX jumps to the year 2000 and the start-time field becomes unusable
    (user issue 3).
    """
    if eph.system == "C":
        sec = eph.toe.sec + BDT_GPST_OFFSET_S
        week = eph.toe.week + BDT_WEEK_OFFSET
        if sec >= SECONDS_IN_WEEK:
            sec -= SECONDS_IN_WEEK
            week += 1
        return GpsTime(week, sec)
    return eph.toe


def ephemeris_toe_span(
        by_sv: dict[str, list[Ephemeris]]) -> tuple[GpsTime, GpsTime] | None:
    """Return ``(min_toe, max_toe)`` in GPS time, or ``None``.

    BeiDou ``toe`` values are normalised from BDT to GPS (see
    :func:`ephemeris_toe_gps`) so the span is meaningful for a mixed RINEX.
    """
    toes = [ephemeris_toe_gps(e) for sets in by_sv.values() for e in sets]
    if not toes:
        return None
    lo = min(toes, key=_abs_gps)
    hi = max(toes, key=_abs_gps)
    return lo, hi


def check_start_coverage(
        start: GpsTime, by_sv: dict[str, list[Ephemeris]],
        margin_hours: float = 6.0) -> str | None:
    """Return a Russian error string when ``start`` is outside the toe span.

    ``None`` means the start time is (probably) covered by the ephemerides.
    """
    span = ephemeris_toe_span(by_sv)
    if span is None:
        return None
    lo, hi = span
    t = _abs_gps(start)
    margin = max(0.0, float(margin_hours)) * 3600.0
    if t < _abs_gps(lo) - margin or t > _abs_gps(hi) + margin:
        def _fmt(g: GpsTime) -> str:
            y, mo, d, hh, mi, _ss = gps2date(g)
            return f"{y:04d}/{mo:02d}/{d:02d} {hh:02d}:{mi:02d}"

        return (f"Время старта {_fmt(start)} вне диапазона эфемерид "
                f"{_fmt(lo)}…{_fmt(hi)} (±{margin_hours:.0f} ч). "
                f"Выберите дату внутри этого диапазона или другой "
                f"RINEX-файл (для Galileo/BeiDou нужен merged "
                f"BRDC00IGS_R_…).")
    return None
