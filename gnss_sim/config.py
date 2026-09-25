"""Simulation configuration shared by the CLI and the GUI."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from .constants import CARR_FREQ_L1
from .gpstime import GpsTime, date2gps

# ----------------------------------------------------------------------
# Combined-band planning
# ----------------------------------------------------------------------
#: BeiDou B1I carrier (kept in sync with :mod:`gnss_sim.beidou`).
B1I_CARRIER_HZ = 1561.098e6
#: Half-width (Hz) each signal group needs around its carrier.
L1_NARROW_HALF_HZ = 1.023e6     # GPS/QZSS L1 C/A and SBAS L1
L1_BOC_HALF_HZ = 8.184e6        # GPS/QZSS L1C TMBOC(6,1) and Galileo E1 CBOC
B1I_HALF_HZ = 2.046e6           # BeiDou B1I BPSK(2)
#: B210 sample-rate limits for a generated stream.
BAND_MIN_FS = 2.6e6
BAND_MAX_FS = 56.0e6
#: B210-friendly sample rates the computed ``fs`` snaps to.
SANE_FS_HZ = (2.6e6, 4.092e6, 5.0e6, 10.0e6, 20.0e6, 24.576e6, 25.0e6,
              30.0e6, 40.0e6, 56.0e6)


@dataclass(frozen=True)
class BandPlan:
    """A computed single-stream band covering the enabled signal groups."""

    center_freq: float
    fs: float
    low: float
    high: float
    span: float
    raw_fs: float
    sources: tuple[str, ...]

    def describe(self) -> str:
        """Human-readable explanation of how centre/``fs`` were derived."""
        parts = " + ".join(self.sources) if self.sources else "—"
        return (f"{parts}: {self.low / 1e6:.3f}..{self.high / 1e6:.3f} МГц "
                f"(размах {self.span / 1e6:.3f}), центр "
                f"{self.center_freq / 1e6:.3f} МГц, fs {self.fs / 1e6:g} "
                f"Мвыб/с (до округления {self.raw_fs / 1e6:.3f})")


def _snap_fs(raw_fs: float, span: float) -> float:
    """Snap ``raw_fs`` to a B210-friendly rate that still covers ``span``."""
    fit = [c for c in SANE_FS_HZ if c >= span - 1.0]
    if fit:
        return min(fit, key=lambda c: abs(c - raw_fs))
    return math.ceil(raw_fs / 0.5e6) * 0.5e6


def compute_combined_band(
    *,
    enable_ca: bool = True,
    enable_l1c: bool = True,
    enable_galileo: bool = True,
    enable_qzss: bool = True,
    enable_sbas: bool = True,
    enable_beidou: bool = True,
) -> BandPlan:
    """Compute a combined centre/``fs`` for the *enabled* signal groups.

    Half-widths around each carrier (see the module constants): L1 C/A and
    SBAS ``±1.023 MHz``; L1C TMBOC and Galileo E1 CBOC ``±8.184 MHz``; BeiDou
    B1I ``±2.046 MHz``.  The result spans ``min..max`` of all enabled groups,
    is centred on the midpoint, widened by 5 %, snapped to a sane B210 rate
    and clamped to ``2.6..56 MHz``.
    """
    intervals: list[tuple[float, float]] = []
    sources: list[str] = []

    def add(name: str, carrier: float, half: float) -> None:
        intervals.append((carrier - half, carrier + half))
        sources.append(name)

    if enable_ca:
        add("L1 C/A", CARR_FREQ_L1, L1_NARROW_HALF_HZ)
    if enable_sbas:
        add("SBAS", CARR_FREQ_L1, L1_NARROW_HALF_HZ)
    if enable_qzss and enable_ca:
        add("QZSS C/A", CARR_FREQ_L1, L1_NARROW_HALF_HZ)
    if enable_l1c:
        add("L1C", CARR_FREQ_L1, L1_BOC_HALF_HZ)
    if enable_galileo:
        add("E1", CARR_FREQ_L1, L1_BOC_HALF_HZ)
    if enable_qzss and enable_l1c:
        add("QZSS L1C", CARR_FREQ_L1, L1_BOC_HALF_HZ)
    if enable_beidou:
        add("B1I", B1I_CARRIER_HZ, B1I_HALF_HZ)
    if not intervals:  # nothing enabled: keep the historic narrow L1 band
        add("L1 C/A", CARR_FREQ_L1, L1_NARROW_HALF_HZ)

    low = min(lo for lo, _ in intervals)
    high = max(hi for _, hi in intervals)
    span = high - low
    center = (low + high) / 2.0
    raw_fs = span * 1.05
    fs = _snap_fs(raw_fs, span)
    fs = max(BAND_MIN_FS, min(BAND_MAX_FS, fs))
    # Guarantee the band still covers every group after clamping/snapping.
    while fs < span and fs < BAND_MAX_FS:
        nxt = [c for c in SANE_FS_HZ if c > fs]
        fs = min(nxt) if nxt else BAND_MAX_FS
    return BandPlan(center, fs, low, high, span, raw_fs,
                    tuple(dict.fromkeys(sources)))


#: Canonical combined plan with every system enabled (used for the ``all``
#: preset title/values; the runner/GUI recompute it from the actual enables).
ALL_BAND_PLAN = compute_combined_band()


@dataclass(frozen=True)
class BandPreset:
    """A named band/session with its recommended centre, ``fs`` and systems.

    ``l1`` is the historic default (GPS/Galileo/QZSS/SBAS around 1575.42 MHz
    at 2.6 Msps).  ``b1i`` is a BeiDou-only narrowband session around
    1561.098 MHz.  ``all`` (alias ``wide``) is a single combined stream that
    keeps every constellation together; its centre/``fs`` are computed from the
    actually enabled systems (see :func:`compute_combined_band`).
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
             "не попадает — для B1I выберите сессию b1i или all.",
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
    "all": BandPreset(
        key="all",
        title=(f"All · {ALL_BAND_PLAN.center_freq / 1e6:.3f} МГц · "
               f"{ALL_BAND_PLAN.fs / 1e6:g} Мвыб/с"),
        center_freq=ALL_BAND_PLAN.center_freq,
        fs=ALL_BAND_PLAN.fs,
        beidou=True,
        systems=("G", "E", "J", "S", "C"),
        tx_ok=True,
        note="Единый поток L1 C/A + L1C + Galileo E1 + QZSS + SBAS + BeiDou "
             "B1I. Центр/fs вычисляются под включённые системы (общий TX LO "
             "B210 → всё одновременно, одна полоса).",
    ),
}


def band_preset(band: str | None) -> BandPreset | None:
    """Return the :class:`BandPreset` for ``band`` (case-insensitive).

    ``wide`` is accepted as a backward-compatible alias of ``all``.
    """
    if not band:
        return BAND_PRESETS["l1"]
    key = str(band).strip().lower()
    if key == "wide":
        key = "all"
    return BAND_PRESETS.get(key)


#: User-selectable session names (``wide`` stays as an alias of ``all``).
BANDS = ("l1", "b1i", "all", "wide")


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
    #: Band/session: ``l1`` (default), ``b1i`` or ``all``/``wide`` (see
    #: :data:`BAND_PRESETS`).
    band: str = "l1"
    #: True when ``fs``/``center_freq`` were explicitly set by the user; the
    #: computed ``all`` band must not overwrite them.
    fs_override: bool = False
    center_override: bool = False

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
