"""Simulation configuration shared by the CLI and the GUI."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from .gpstime import GpsTime, date2gps


@dataclass(frozen=True)
class BandPreset:
    """A named band/session with its recommended centre, ``fs`` and systems.

    ``l1`` is the historic default (GPS/Galileo/QZSS/SBAS around 1575.42 MHz
    at 2.6 Msps).  ``b1i`` is a BeiDou-only narrowband session around
    1561.098 MHz.  ``wide`` keeps every constellation together but needs
    ~30 Msps and is intended for IQ files (unstable for B210 TX over USB3).
    """

    key: str
    title: str
    center_freq: float
    fs: float
    beidou: bool
    systems: tuple[str, ...]          # enabled constellations
    tx_ok: bool
    note: str


#: Selectable bands/sessions (``SimConfig.band`` / ``--band``).
BAND_PRESETS: dict[str, BandPreset] = {
    "l1": BandPreset(
        key="l1",
        title="L1 · 1575.42 МГц · 2.6 Мвыб/с",
        center_freq=1575.42e6,
        fs=2.6e6,
        beidou=False,
        systems=("G", "E", "J", "S"),
        tx_ok=True,
        note="GPS L1 C/A + L1C, Galileo E1, QZSS L1, SBAS. Узкая полоса, "
             "стабильна для передачи на B210. BeiDou B1I (1561.098 МГц) сюда "
             "не попадает — для B1I выберите сессию b1i или wide.",
    ),
    "b1i": BandPreset(
        key="b1i",
        title="B1I · 1561.098 МГц · 4.092 Мвыб/с",
        center_freq=1561.098e6,
        fs=4.092e6,
        beidou=True,
        systems=("C",),
        tx_ok=True,
        note="Только BeiDou B1I (2046-чиповый код, NH20, D1 50 бит/с). "
             "Узкая полоса вокруг 1561.098 МГц — стабильная передача на B210. "
             "Запускайте как отдельную сессию, не вместе с L1.",
    ),
    "wide": BandPreset(
        key="wide",
        title="Wide · 1568 МГц · 30 Мвыб/с",
        center_freq=1568.0e6,
        fs=30.0e6,
        beidou=True,
        systems=("G", "E", "J", "S", "C"),
        tx_ok=False,
        note="Все системы (L1/E1 + B1I) в одной полосе. 30 Мвыб/с — тяжёлая "
             "по RAM, а передача на B210 по USB3 может уходить в underflow и "
             "обрыв (LIBUSB_TRANSFER_NO_DEVICE). Предназначено для IQ-файлов; "
             "для эфира используйте отдельные сессии l1 и b1i.",
    ),
}


def band_preset(band: str | None) -> BandPreset | None:
    """Return the :class:`BandPreset` for ``band`` (case-insensitive)."""
    if not band:
        return BAND_PRESETS["l1"]
    return BAND_PRESETS.get(str(band).strip().lower())


BANDS = tuple(BAND_PRESETS)


def parse_start_time(text: str) -> GpsTime:
    """Parse ``now`` or ``YYYY/MM/DD,hh:mm:ss`` (also accepts ``-`` and ``T``)."""
    text = (text or "").strip()
    if not text or text.lower() == "now":
        now = datetime.now(timezone.utc)
        return date2gps(now.year, now.month, now.day, now.hour,
                        now.minute, now.second + now.microsecond / 1e6)

    normalized = text.replace(",", " ").replace("T", " ")
    parts = normalized.split()
    if len(parts) != 2:
        raise ValueError(f"Неверная дата/время: {text!r}")
    date_part, time_part = parts
    date_sep = "-" if "-" in date_part else "/"
    y, m, d = (int(x) for x in date_part.split(date_sep))
    hms = time_part.split(":")
    hh = int(hms[0]) if len(hms) > 0 else 0
    mm = int(hms[1]) if len(hms) > 1 else 0
    ss = float(hms[2]) if len(hms) > 2 else 0.0
    return date2gps(y, m, d, hh, mm, ss)


@dataclass
class SimConfig:
    nav_file: str = ""
    lat: float = 35.681298
    lon: float = 139.766247
    height: float = 10.0
    motion_file: str = ""
    start_text: str = "now"
    duration: float = 60.0
    fs: float = 2.6e6
    center_freq: float = 1575.42e6
    #: Band/session: ``l1`` (default), ``b1i`` or ``wide`` (see BAND_PRESETS).
    band: str = "l1"

    enable_ca: bool = True
    enable_l1c: bool = True
    enable_galileo: bool = True
    enable_qzss: bool = True
    enable_sbas: bool = True
    enable_beidou: bool = True
    el_mask: float = 5.0
    amp_scale: float = 0.15
    iono_enable: bool = True
    l1c_data: str = "zeros"
    #: BeiDou B1I data: ``d1`` (real broadcast message) or ``placeholder``
    #: (all +1 bits, but still NH20-framed).
    b1i_data: str = "d1"

    auto_download: bool = True

    output: str = ""
    output_format: str = "cs16"
    output_scale: float = 10000.0
    #: Reuse an existing IQ file instead of generating one (skip RINEX/engine).
    iq_input: str = ""
    #: Load the whole ready IQ file into RAM and loop from memory.
    iq_in_ram: bool = False

    use_usrp: bool = False
    uhd_args: str = "type=b200"
    tx_channel: int = 0
    tx_gain: float = 0.0
    tx_antenna: str = "TX/RX"
    tx_bandwidth: float = 0.0  # 0 = let UHD choose
    clock_source: str = "internal"
    block_ms: float = 500.0
    backend: str = "auto"

    loop: bool = True
    loop_seconds: float = 0.0
    memory_budget_gb: float = 0.0

    # Мониторинг RX / регулятор мощности (B210, отдельный RX-канал)
    monitor: bool = False
    tx_power_target_dbfs: float = -30.0
    rx_channel: int = 1
    rx_antenna: str = "RX2"
    rx_gain: float = 30.0
    rx_center_freq: float = 0.0
    echo_analysis: bool = False
    tx_power_auto: bool = True
    #: Лёгкий «контроль передачи»: принять RX на другом канале во время TX и
    #: предупредить, если уровень не поднялся над шумовым полом.
    tx_check: bool = False
    #: Порог обнаружения передачи (дБ над шумовым полом) для ``tx_check``.
    tx_check_margin_db: float = 6.0

    # Источник эфемерид / вход в CDDIS (Earthdata)
    cddis_user: str = ""
    cddis_password: str = ""
    cddis_token: str = ""
    download_source: str = "auto"
    #: Ephemeris preference: ``auto`` (merged then per-system) or ``merged``
    #: (prefer the multi-GNSS merged file; available only for previous days).
    nav_mode: str = "auto"
    #: Automatically pick a centre/fs wide enough for BeiDou B1I + L1/E1.
    auto_b1i: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    def resolved_start(self) -> GpsTime:
        return parse_start_time(self.start_text)
