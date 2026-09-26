"""Чтение NMEA/UBX-потока с приёмника u-blox ZED-F9P (COM-порт).

Модуль опирается только на ``pyserial`` и NumPy-независим: разбор строк
(``parse_nmea_line``) — чистая функция, пригодная для тестов без железа, а
:class:`NmeaReader` крутит фоновый daemon-поток и отдаёт состояние потокобезопасно.

Поддерживаются GGA (позиция/высота/fix/число спутников/HDOP), RMC (дата,
время, скорость), GSA и GSV (спутники), а также мульти-GNSS talker'ы
(``GNGGA``/``GNRMC``/``GPGSV``/``GAGSV``/…).

Дополнительно реализованы чистые (без железа) разборщики UBX:
``ubx_scan`` (поток → кадры), ``parse_mon_ver`` (0x0A/0x04),
``parse_cfg_gnss`` (0x06/0x3E) и ``parse_nav_sat`` (0x01/0x35), плюс
:class:`UbxStream` — независимый от :class:`NmeaReader` приём кадров.
"""

from __future__ import annotations

import threading

_TALKER_SYSTEM = {
    "GP": "G",  # GPS
    "GA": "E",  # Galileo
    "GB": "C",  # BeiDou
    "BD": "C",
    "GC": "C",
    "GI": "C",
    "GQ": "J",  # QZSS
    "GL": "R",  # GLONASS
    "GN": "G",  # multi-GNSS (систему отдельно не определить)
    "SB": "S",  # SBAS
    "GS": "S",
}


def _system_from_talker(talker: str) -> str:
    t = (talker or "").upper()
    if len(t) < 2:
        return "?"
    if t in _TALKER_SYSTEM:
        return _TALKER_SYSTEM[t]
    # Эвристика: последний символ talker'а часто кодирует систему.
    return {"P": "G", "A": "E", "B": "C", "D": "C", "Q": "J",
            "L": "R", "N": "G"}.get(t[-1], "?")


def _num(fields: list[str], idx: int, default=None):
    try:
        s = fields[idx]
    except IndexError:
        return default
    if s == "":
        return default
    try:
        return float(s)
    except ValueError:
        return default


def _int(fields: list[str], idx: int, default=0) -> int:
    v = _num(fields, idx, None)
    return int(v) if v is not None else default


def _nmea_angle(fields: list[str], vi: int, hi: int):
    v = _num(fields, vi, None)
    if v is None:
        return None
    hemi = fields[hi] if len(fields) > hi else "N"
    deg = int(v / 100.0)
    minutes = v - deg * 100.0
    res = deg + minutes / 60.0
    if hemi in ("S", "W"):
        res = -res
    return res


def _rmc_datetime(date_str: str, time_str: str) -> str:
    if not date_str or len(date_str) < 6:
        return ""
    try:
        dd = int(date_str[0:2])
        mm = int(date_str[2:4])
        yy = int(date_str[4:6])
        year = 2000 + yy if yy < 80 else 1900 + yy
        hh = mi = ss = 0
        if time_str:
            hms = time_str.split(".")[0]
            if len(hms) >= 6:
                hh, mi, ss = int(hms[0:2]), int(hms[2:4]), int(hms[4:6])
            elif len(hms) >= 4:  # pragma: no cover - нестандартный формат
                hh, mi = int(hms[0:2]), int(hms[2:4])
        return (f"{year:04d}-{mm:02d}-{dd:02d} "
                f"{hh:02d}:{mi:02d}:{ss:02d}")
    except (ValueError, IndexError):
        return ""


def parse_nmea_line(line: str) -> dict:
    """Разобрать одну NMEA-строку в словарь (чистая функция).

    Возвращает ``{}`` для мусора/неподдерживаемых предложений.  Для
    поддерживаемых: ``type`` (GGA/RMC/GSA/GSV), ``talker``, ``system`` и
    поля соответствующего предложения.
    """
    if not line:
        return {}
    s = line.strip()
    if not s.startswith("$"):
        return {}
    body = s[1:]
    if "*" in body:
        body = body.split("*", 1)[0]
    fields = body.split(",")
    if not fields:
        return {}
    tag = fields[0].upper()
    if len(tag) < 3:
        return {}
    talker, stype = tag[:2], tag[2:]
    system = _system_from_talker(talker)

    if stype == "GGA":
        return {
            "type": "GGA", "talker": talker, "system": system,
            "time": fields[1] if len(fields) > 1 else "",
            "lat": _nmea_angle(fields, 2, 3),
            "lon": _nmea_angle(fields, 4, 5),
            "fix": _int(fields, 6, 0),
            "num_sats": _int(fields, 7, 0),
            "hdop": _num(fields, 8, None),
            "height": _num(fields, 9, None),
        }

    if stype == "RMC":
        date_str = fields[9] if len(fields) > 9 else ""
        time_str = fields[1] if len(fields) > 1 else ""
        return {
            "type": "RMC", "talker": talker, "system": system,
            "time": time_str,
            "status": fields[2] if len(fields) > 2 else "",
            "lat": _nmea_angle(fields, 3, 4),
            "lon": _nmea_angle(fields, 5, 6),
            "speed_knots": _num(fields, 7, None),
            "course": _num(fields, 8, None),
            "date": date_str,
            "datetime": _rmc_datetime(date_str, time_str),
        }

    if stype == "GSA":
        svids = [int(x) for x in fields[3:15] if x and x.isdigit()]
        return {
            "type": "GSA", "talker": talker, "system": system,
            "mode": fields[1] if len(fields) > 1 else "",
            "fix": _int(fields, 2, 0),
            "svids": svids,
            "pdop": _num(fields, 15, None),
            "hdop": _num(fields, 16, None),
            "vdop": _num(fields, 17, None),
        }

    if stype == "GSV":
        sats = []
        for i in range(4, len(fields) - 3, 4):
            prn = fields[i]
            if not prn or not prn.isdigit():
                continue
            sats.append({
                "system": system, "prn": int(prn),
                "elev": _num(fields, i + 1, None),
                "az": _num(fields, i + 2, None),
                "snr": _num(fields, i + 3, None),
            })
        return {
            "type": "GSV", "talker": talker, "system": system,
            "total": _int(fields, 1, 0), "num": _int(fields, 2, 0),
            "in_view": _int(fields, 3, 0), "sats": sats,
        }

    return {"type": stype, "talker": talker, "system": system}


def list_ports() -> list[str]:
    """Список доступных последовательных портов (``['COM3', …]``)."""
    try:
        from serial.tools import list_ports as _lp
    except Exception:  # noqa: BLE001 - pyserial не установлен
        return []
    ports = []
    for p in _lp.comports():
        dev = getattr(p, "device", None)
        if dev:
            ports.append(str(dev))
    return ports


# ----------------------------------------------------------------------
# u-blox UBX protocol (cold start)
# ----------------------------------------------------------------------
UBX_SYNC1 = 0xB5
UBX_SYNC2 = 0x62

# Reset modes for UBX-CFG-RST.
RESET_HARDWARE = 0x00          # hardware reset (watchdog)
RESET_SOFTWARE = 0x02          # controlled software reset
RESET_GNSS_ONLY = 0x09         # controlled software reset (GNSS only)

# navBbrMask: clear all battery-backed RAM -> true cold start.
BBR_COLD_START = 0xFFFF


def ubx_checksum(data: bytes) -> tuple[int, int]:
    """2-byte Fletcher checksum (CK_A, CK_B) over ``data``."""
    ck_a = 0
    ck_b = 0
    for b in data:
        ck_a = (ck_a + b) & 0xFF
        ck_b = (ck_b + ck_a) & 0xFF
    return ck_a, ck_b


def ubx_frame(msg_class: int, msg_id: int, payload: bytes = b"") -> bytes:
    """Build a complete UBX frame with sync bytes and Fletcher checksum.

    The checksum covers class, id, length and payload (not the sync bytes).
    """
    payload = bytes(payload)
    body = (bytes([msg_class & 0xFF, msg_id & 0xFF])
            + len(payload).to_bytes(2, "little") + payload)
    ck_a, ck_b = ubx_checksum(body)
    return bytes([UBX_SYNC1, UBX_SYNC2]) + body + bytes([ck_a, ck_b])


def ubx_cfg_rst_frame(nav_bbr_mask: int = BBR_COLD_START,
                      reset_mode: int = RESET_SOFTWARE) -> bytes:
    """UBX-CFG-RST frame (class 0x06, id 0x04) — cold start by default.

    Payload is 4 bytes: ``navBbrMask`` (U2), ``resetMode`` (U1) and a
    reserved byte (U1, 0).  Default ``FF FF 02 00`` = clear all BBR +
    controlled software reset, i.e. ``B5 62 06 04 04 00 FF FF 02 00 0E 61``.
    """
    payload = (int(nav_bbr_mask).to_bytes(2, "little")
               + bytes([int(reset_mode) & 0xFF]) + b"\x00")
    return ubx_frame(0x06, 0x04, payload)


def send_ubx(port: str, frame: bytes, baudrate: int = 38400,
             timeout: float = 1.0) -> int:
    """Write ``frame`` to ``port`` and return the number of bytes written."""
    import serial  # ленивый импорт
    ser = serial.Serial(str(port), int(baudrate), timeout=float(timeout),
                        write_timeout=float(timeout))
    try:
        n = ser.write(bytes(frame))
        try:
            ser.flush()
        except Exception:  # noqa: BLE001
            pass
        return int(n)
    finally:
        try:
            ser.close()
        except Exception:  # noqa: BLE001
            pass


def cold_start(port: str, baudrate: int = 38400,
               nav_bbr_mask: int = BBR_COLD_START,
               reset_mode: int = RESET_SOFTWARE) -> bytes:
    """Send a UBX-CFG-RST cold-start command, returning the frame sent."""
    frame = ubx_cfg_rst_frame(nav_bbr_mask, reset_mode)
    send_ubx(port, frame, baudrate)
    return frame


# ----------------------------------------------------------------------
# u-blox UBX protocol (parsing helpers, pure functions)
# ----------------------------------------------------------------------
UBX_CLASS_NAV = 0x01
UBX_CLASS_CFG = 0x06
UBX_CLASS_MON = 0x0A

UBX_ID_NAV_SAT = 0x35
UBX_ID_CFG_GNSS = 0x3E
UBX_ID_MON_VER = 0x04

# u-blox gnssId -> RINEX-style system letter (same letters as NMEA talkers).
_GNSS_ID_SYSTEM = {
    0: "G",  # GPS
    1: "S",  # SBAS
    2: "E",  # Galileo
    3: "C",  # BeiDou
    4: "?",  # IMES
    5: "J",  # QZSS
    6: "R",  # GLONASS
    7: "?",  # NavIC / IRNSS
}

# UBX-CFG-GNSS sigCfMask bit (bits 16..23 of `flags`) -> signal name.
# Source: u-blox interface description / pyubx2 SIGCFMASK / PX4 ubx.h.
_CFG_GNSS_SIGNALS = {
    (0, 0x01): "GPS L1C/A",
    (0, 0x10): "GPS L2C",
    (0, 0x20): "GPS L5",
    (1, 0x01): "SBAS L1C/A",
    (2, 0x01): "Galileo E1",
    (2, 0x10): "Galileo E5a",
    (2, 0x20): "Galileo E5b",
    (3, 0x01): "BeiDou B1I",
    (3, 0x10): "BeiDou B2I",
    (3, 0x80): "BeiDou B2A",
    (4, 0x01): "IMES L1",
    (5, 0x01): "QZSS L1C/A",
    (5, 0x04): "QZSS L1S",
    (5, 0x10): "QZSS L2C",
    (5, 0x20): "QZSS L5",
    (6, 0x01): "GLONASS L1",
    (6, 0x10): "GLONASS L2",
}


def _cstr(raw: bytes) -> str:
    """Decode a NUL-terminated ASCII field (trailing NUL/space trimmed)."""
    return raw.split(b"\x00", 1)[0].decode("ascii", "ignore").strip()


def _i8(value: int) -> int:
    """Interpret an unsigned byte as a signed 8-bit integer."""
    return value - 0x100 if value & 0x80 else value


def _i16(raw: bytes) -> int:
    """Little-endian signed 16-bit integer."""
    return int.from_bytes(raw, "little", signed=True)


def ubx_scan(buf: bytes) -> tuple[list[tuple[int, int, bytes]], bytes]:
    """Split a raw byte stream into validated UBX frames.

    Scans ``buf`` for ``B5 62`` sync pairs, reads the little-endian U2
    length, and validates the 2-byte Fletcher checksum.  Returns
    ``(frames, remainder)`` where ``frames`` is a list of
    ``(msg_class, msg_id, payload)`` tuples and ``remainder`` holds bytes
    that could not be consumed yet (an incomplete trailing frame, or a
    dangling ``B5``).  False syncs (bad checksum) are resynchronised by
    advancing one byte so a valid frame later in the buffer is still found.

    Feed the remainder back in on the next read::

        frames, rest = ubx_scan(rest + ser.read(...))
    """
    data = bytes(buf)
    n = len(data)
    frames: list[tuple[int, int, bytes]] = []
    pos = 0
    while pos < n:
        if data[pos] != UBX_SYNC1:
            pos += 1
            continue
        if pos + 1 >= n:                      # dangling SYNC1
            break
        if data[pos + 1] != UBX_SYNC2:        # lone B5 inside garbage
            pos += 1
            continue
        if pos + 6 > n:                       # header not complete yet
            break
        length = data[pos + 4] | (data[pos + 5] << 8)
        frame_end = pos + 6 + length + 2
        if frame_end > n:                     # payload/checksum incomplete
            break
        body = data[pos + 2:pos + 6 + length]
        ck_a, ck_b = ubx_checksum(body)
        if ck_a == data[pos + 6 + length] and ck_b == data[pos + 7 + length]:
            frames.append((data[pos + 2], data[pos + 3], body[4:]))
            pos = frame_end
        else:                                 # false sync -> resynchronise
            pos += 1
    return frames, data[pos:]


def parse_mon_ver(payload: bytes) -> dict:
    """Parse UBX-MON-VER (0x0A/0x04).

    Layout: ``swVersion`` U1[30], ``hwVersion`` U1[10], then NUL-terminated
    30-byte extension fields (``ROM BASE …``, ``FWVER=…``, ``PROTVER=…``,
    ``MOD=…`` …).  Returns ``{'sw', 'hw', 'extensions': [...]}`` and, when a
    ``KEY=VALUE`` extension is present, the lower-cased key (``fwver``,
    ``protver``, ``mod`` …).
    """
    p = bytes(payload)
    sw = _cstr(p[0:30]) if len(p) >= 30 else _cstr(p)
    hw = _cstr(p[30:40]) if len(p) >= 40 else ""
    extensions: list[str] = []
    off = 40
    while off < len(p):
        chunk = p[off:off + 30]
        text = _cstr(chunk)
        if text:
            extensions.append(text)
        off += 30
    result = {"sw": sw, "hw": hw, "extensions": extensions}
    for ext in extensions:
        if "=" in ext:
            key, value = ext.split("=", 1)
            result[key.strip().lower()] = value.strip()
    return result


def parse_cfg_gnss(payload: bytes) -> list[dict]:
    """Parse UBX-CFG-GNSS (0x06/0x3E) into a list of config blocks.

    Header (4 bytes): ``msgVer`` U1, ``numTrkChHw`` U1, ``numTrkChUse`` U1,
    ``numConfigBlocks`` U1.  Each 8-byte block: ``gnssId`` U1,
    ``resTrkCh`` U1, ``maxTrkCh`` U1, reserved U1, ``flags`` X4.

    ``flags`` bit 0 = ``enable``; bits 16..23 = ``sigCfMask`` (signal mask,
    verified against the u-blox interface description, pyubx2 and PX4).
    """
    p = bytes(payload)
    if len(p) < 4:
        return []
    num_blocks = p[3]
    blocks: list[dict] = []
    off = 4
    for _ in range(num_blocks):
        if off + 8 > len(p):
            break
        gnss_id = p[off]
        flags = int.from_bytes(p[off + 4:off + 8], "little")
        sig_mask = (flags >> 16) & 0xFF
        blocks.append({
            "gnssId": gnss_id,
            "system": _GNSS_ID_SYSTEM.get(gnss_id, "?"),
            "resTrkCh": p[off + 1],
            "maxTrkCh": p[off + 2],
            "flags": flags,
            "enable": bool(flags & 0x01),
            "sigCfMask": sig_mask,
            "signals": _cfg_gnss_signals(gnss_id, sig_mask),
        })
        off += 8
    return blocks


def parse_cfg_gnss_msg(payload: bytes) -> dict:
    """Like :func:`parse_cfg_gnss` but also returns the message header."""
    p = bytes(payload)
    return {
        "msgVer": p[0] if len(p) > 0 else None,
        "numTrkChHw": p[1] if len(p) > 1 else None,
        "numTrkChUse": p[2] if len(p) > 2 else None,
        "numConfigBlocks": p[3] if len(p) > 3 else 0,
        "blocks": parse_cfg_gnss(p),
    }


def _cfg_gnss_signals(gnss_id: int, sig_mask: int) -> list[str]:
    names: list[str] = []
    for bit in range(8):
        if sig_mask & (1 << bit):
            name = _CFG_GNSS_SIGNALS.get((gnss_id, 1 << bit))
            names.append(name if name is not None else f"0x{1 << bit:02X}")
    return names


def parse_nav_sat(payload: bytes) -> list[dict]:
    """Parse UBX-NAV-SAT (0x01/0x35) version-1 per-satellite blocks.

    Header (8 bytes): ``iTOW`` U4, ``version`` U1, ``numSvs`` U1,
    reserved U1[2] (verified against the u-blox interface description and
    pyubx2).  Each satellite is a 12-byte block: ``gnssId`` U1, ``svId`` U1,
    ``cno`` U1, ``elev`` I1, ``azim`` I2, ``prRes`` I2, ``flags`` X4.

    ``flags`` (cross-checked against pyubx2/go-ubx): quality bits 0..2,
    ``svUsed`` bit 3, health bits 4..5, ``diffCorr`` bit 6, ``smoothed``
    bit 7, ``orbitSource`` bits 8..10, ``ephAvail`` bit 11 (0x800),
    ``almAvail`` bit 12, ``anoAvail`` bit 13, ``aopAvail`` bit 14.
    """
    p = bytes(payload)
    if len(p) < 8:
        return []
    itow = int.from_bytes(p[0:4], "little")
    version = p[4]
    num_svs = p[5]
    blocks: list[dict] = []
    off = 8
    for _ in range(num_svs):
        if off + 12 > len(p):
            break
        d = p[off:off + 12]
        gnss_id = d[0]
        flags = int.from_bytes(d[8:12], "little")
        blocks.append({
            "iTOW": itow,
            "version": version,
            "gnssId": gnss_id,
            "svId": d[1],
            "system": _GNSS_ID_SYSTEM.get(gnss_id, "?"),
            "cno": d[2],
            "elev": _i8(d[3]),
            "azim": _i16(d[4:6]),
            "prRes": _i16(d[6:8]),
            "prResM": _i16(d[6:8]) * 0.1,
            "flags": flags,
            "quality": flags & 0x07,
            "svUsed": bool(flags & 0x08),
            "health": (flags >> 4) & 0x03,
            "diffCorr": bool(flags & 0x40),
            "smoothed": bool(flags & 0x80),
            "orbitSource": (flags >> 8) & 0x07,
            "ephAvail": bool(flags & 0x800),
            "almAvail": bool(flags & 0x1000),
            "anoAvail": bool(flags & 0x2000),
            "aopAvail": bool(flags & 0x4000),
        })
        off += 12
    return blocks


def nav_sat_used(blocks: list[dict]) -> list[dict]:
    """Return the NAV-SAT blocks flagged ``svUsed`` (used in the solution)."""
    return [b for b in blocks if b.get("svUsed")]


def nav_sat_by_system(blocks: list[dict]) -> dict[str, list[dict]]:
    """Group NAV-SAT blocks by constellation letter (``G``/``E``/``C``/…)."""
    grouped: dict[str, list[dict]] = {}
    for block in blocks:
        grouped.setdefault(block.get("system", "?"), []).append(block)
    return grouped


def parse_ubx_message(msg_class: int, msg_id: int, payload: bytes):
    """Dispatch a single frame to the matching parser (``None`` if unknown)."""
    if msg_class == UBX_CLASS_MON and msg_id == UBX_ID_MON_VER:
        return parse_mon_ver(payload)
    if msg_class == UBX_CLASS_CFG and msg_id == UBX_ID_CFG_GNSS:
        return parse_cfg_gnss(payload)
    if msg_class == UBX_CLASS_NAV and msg_id == UBX_ID_NAV_SAT:
        return parse_nav_sat(payload)
    return None


class UbxStream:
    """Incremental UBX frame receiver, independent of :class:`NmeaReader`.

    ::

        stream = UbxStream()
        for cls, mid, payload in stream.feed(chunk):
            data = parse_ubx_message(cls, mid, payload)

    Incomplete trailing bytes are kept inside the stream and combined with
    the next ``feed`` call.
    """

    def __init__(self) -> None:
        self._buf = b""

    def feed(self, data: bytes) -> list[tuple[int, int, bytes]]:
        self._buf += bytes(data)
        frames, self._buf = ubx_scan(self._buf)
        return frames

    def remainder(self) -> bytes:
        return self._buf

    def reset(self) -> None:
        self._buf = b""


class NmeaReader:
    """Фоновое чтение NMEA с последовательного порта (daemon-поток)."""

    def __init__(self, port: str, baudrate: int = 38400) -> None:
        self.port = str(port)
        self.baudrate = int(baudrate)
        self._serial = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._pos = {
            "lat": None, "lon": None, "height": None,
            "fix": 0, "num_sats": 0, "hdop": None,
        }
        self._dt = ""
        self._sats: dict[tuple, dict] = {}
        self._last: dict = {}

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self.is_running():
            return
        import serial  # ленивый импорт: без pyserial модуль всё равно грузится
        self._serial = serial.Serial(self.port, self.baudrate, timeout=1.0)
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="nmea-reader",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=2.0)
        self._thread = None
        ser = self._serial
        self._serial = None
        if ser is not None:
            try:
                ser.close()
            except Exception:  # noqa: BLE001
                pass

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def send(self, data: bytes) -> int:
        """Write raw bytes to the open port (e.g. a UBX command)."""
        ser = self._serial
        if ser is None:
            raise RuntimeError("COM-порт не открыт")
        with self._lock:
            n = ser.write(bytes(data))
            try:
                ser.flush()
            except Exception:  # noqa: BLE001
                pass
        return int(n)

    # ------------------------------------------------------------------
    def _loop(self) -> None:  # pragma: no cover - требует COM-порт
        while not self._stop.is_set():
            ser = self._serial
            if ser is None:
                break
            try:
                raw = ser.readline()
            except Exception:  # noqa: BLE001 - порт отключён/ошибка
                break
            if not raw:
                continue
            try:
                line = raw.decode("ascii", "ignore")
            except Exception:  # noqa: BLE001
                continue
            data = parse_nmea_line(line)
            if data:
                self._apply(data)

    def _apply(self, data: dict) -> None:
        dtype = data.get("type")
        with self._lock:
            self._last = data
            if dtype == "GGA":
                self._pos.update(
                    lat=data.get("lat", self._pos["lat"]),
                    lon=data.get("lon", self._pos["lon"]),
                    height=data.get("height", self._pos["height"]),
                    fix=data.get("fix", self._pos["fix"]),
                    num_sats=data.get("num_sats", self._pos["num_sats"]),
                    hdop=data.get("hdop", self._pos["hdop"]),
                )
            elif dtype == "RMC":
                if data.get("datetime"):
                    self._dt = data["datetime"]
                for key in ("lat", "lon"):
                    if data.get(key) is not None:
                        self._pos[key] = data[key]
            elif dtype == "GSV":
                system = data.get("system", "?")
                if data.get("num", 0) in (0, 1):
                    self._sats = {k: v for k, v in self._sats.items()
                                  if k[0] != system}
                for sat in data.get("sats", []):
                    self._sats[(sat["system"], sat["prn"])] = dict(sat)
            elif dtype == "GSA":
                if data.get("hdop") is not None:
                    self._pos["hdop"] = data["hdop"]

    # ------------------------------------------------------------------
    def position(self) -> dict:
        with self._lock:
            return dict(self._pos)

    def datetime_utc(self) -> str:
        with self._lock:
            return self._dt

    def satellites(self) -> list[dict]:
        with self._lock:
            sats = [dict(v) for v in self._sats.values()]
        sats.sort(key=lambda s: (s.get("system", ""), s.get("prn", 0)))
        return sats

    def paragraph(self) -> str:
        p = self.position()
        return (
            f"широта {p['lat']}, долгота {p['lon']}, высота {p['height']} м; "
            f"fix {p['fix']}, спутников {p['num_sats']}, HDOP {p['hdop']}; "
            f"UTC {self.datetime_utc() or '—'}; всего в списке "
            f"{len(self.satellites())}")

    def last(self) -> dict:
        with self._lock:
            return dict(self._last)


# ----------------------------------------------------------------------
# Simulator-visible vs receiver-GSV comparison
# ----------------------------------------------------------------------
def _sv_key(sat: dict) -> tuple[str, int] | None:
    """Normalise a satellite entry to ``(system, prn)`` or ``None``.

    Accepts both the engine's ``channel_info`` dicts (``system``/``prn``) and
    :meth:`NmeaReader.satellites` entries.  A missing system is taken from the
    first character of ``name`` (e.g. ``"S120"`` -> ``("S", 120)``).
    """
    system = str(sat.get("system") or "").upper()
    name = str(sat.get("name") or "")
    if not system and name:
        system = name[0].upper()
    try:
        prn = int(sat.get("prn"))
    except (TypeError, ValueError):
        return None
    if not system:
        return None
    return (system, prn)


def compare_visible(sim_sats, gsv_sats) -> dict:
    """Compare the simulator's visible list against the receiver GSV list.

    ``sim_sats`` is ``SignalEngine.channel_info`` (or any iterable of dicts with
    ``system``/``prn``); ``gsv_sats`` is :meth:`NmeaReader.satellites` (built
    from ``GSV`` sentences — **not** ``GSA``, which lists *used* satellites and
    would hide tracked-but-unused ones).  Returns sets of ``(system, prn)``:

    * ``matched``   — visible to the simulator and reported in GSV;
    * ``not_in_gsv`` — transmitted but absent from the receiver's GSV
      (not acquired/tracked — the interesting failure signal);
    * ``not_in_sim`` — in GSV but not transmitted (real sky/other GNSS);
    * ``sim_count`` / ``gsv_count``.
    """
    sim = {k for k in (_sv_key(s) for s in sim_sats) if k is not None}
    gsv = {k for k in (_sv_key(s) for s in gsv_sats) if k is not None}
    return {
        "matched": sorted(sim & gsv),
        "not_in_gsv": sorted(sim - gsv),
        "not_in_sim": sorted(gsv - sim),
        "sim_count": len(sim),
        "gsv_count": len(gsv),
    }


def _fmt_sv(sv: tuple[str, int]) -> str:
    system, prn = sv
    return f"{system}{prn:02d}" if system != "S" else f"S{prn}"


def describe_visible(cmp: dict, limit: int = 8) -> str:
    """Human-readable one-line summary of :func:`compare_visible`."""
    def fmt(items):
        shown = ", ".join(_fmt_sv(s) for s in items[:limit])
        if len(items) > limit:
            shown += f" …(+{len(items) - limit})"
        return shown or "—"

    return (
        f"Видимые (симулятор vs GSV): {cmp['sim_count']} передано, "
        f"{cmp['gsv_count']} в GSV, совпало {len(cmp['matched'])}; "
        f"нет в GSV: {fmt(cmp['not_in_gsv'])}; "
        f"нет в симуляторе: {fmt(cmp['not_in_sim'])}")
