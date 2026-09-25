"""Чтение NMEA-потока с приёмника u-blox ZED-F9P (COM-порт).

Модуль опирается только на ``pyserial`` и NumPy-независим: разбор строк
(``parse_nmea_line``) — чистая функция, пригодная для тестов без железа, а
:class:`NmeaReader` крутит фоновый daemon-поток и отдаёт состояние потокобезопасно.

Поддерживаются GGA (позиция/высота/fix/число спутников/HDOP), RMC (дата,
время, скорость), GSA и GSV (спутники), а также мульти-GNSS talker'ы
(``GNGGA``/``GNRMC``/``GPGSV``/``GAGSV``/…).
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
