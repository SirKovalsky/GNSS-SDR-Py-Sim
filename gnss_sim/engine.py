"""Vectorised baseband IQ synthesis engine for L1 GNSS signals.

Signals generated (all inside the same L1/E1 band around 1575.42 MHz):
  * GPS L1 C/A  - BPSK(1) with the 50 bps legacy NAV message.
  * GPS L1C     - L1Cd BOC(1,1) + L1Cp TMBOC(6,1,4/33) with L1Co overlay.
  * Galileo E1  - E1-B/E1-C composite CBOC(6,1,1/11); real I/NAV at 250 bps.
  * QZSS L1 C/A - same BPSK(1) legacy NAV as GPS (PRN 193..206).
  * SBAS L1 C/A - BPSK(1) at synthetic geostationary positions (PRN 120..158).
  * BeiDou B1I  - BPSK(2) at 1561.098 MHz (2046-chip code, placeholder data).

Nav *data* for BeiDou B1I is a placeholder (codes, BPSK modulation and Doppler
are fully simulated); GPS/QZSS carry the real legacy NAV, SBAS carries real
250 bps DO-229 messages (see :mod:`gnss_sim.sbas`) and Galileo E1-B carries a
real 250 bps I/NAV sub-frame (see :mod:`gnss_sim.galileo_nav`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

try:  # CUDA acceleration is optional
    import cupy as _cp
except Exception:  # pragma: no cover - CUDA optional
    _cp = None


def cuda_available() -> bool:
    """True when CuPy imports and at least one CUDA device is visible."""
    if _cp is None:
        return False
    try:
        return int(_cp.cuda.runtime.getDeviceCount()) > 0
    except Exception:
        return False


from .beidou import (
    B1I_CODE_CHIPS, B1I_CODE_RATE, CARR_FREQ_B1I, generate_b1i,
)
from .beidou_nav import NH_CODE, d1_frame_block
from .ca_code import code_bipolar
from .constants import (
    CARR_FREQ_L1, CODE_FREQ_CA, L1C_CODE_LEN, LAMBDA_L1, R2D,
    SPEED_OF_LIGHT, WGS84_RADIUS,
)
from .galileo import CBOC_ALPHA, CBOC_BETA, e1_code, e1_secondary
from .galileo_nav import inav_bit_block
from .gpstime import GpsTime, sub_gps_time
from .l1c_codes import overlay_code, ranging_code, tmboc_is_boc61
from .navmsg import eph2sbf, generate_nav_msg
from .orbit import (
    compute_range, ecef2neu, ionospheric_delay, ltcmat, neu2azel, xyz2llh,
)
from .rinex import Ephemeris, IonoUtc, select_ephemeris
from .sbas import sbas_bit_block

# Antenna pattern (dB) versus boresight angle in 5 deg steps, ported from gpssim.
_ANT_PAT_DB = (
    0.00, 0.00, 0.22, 0.44, 0.67, 1.11, 1.56, 2.00, 2.44, 2.89,
    3.56, 4.22, 4.89, 5.56, 6.22, 6.89, 7.56, 8.22, 8.89, 9.78,
    10.67, 11.56, 12.44, 13.33, 14.44, 15.56, 16.67, 17.78, 18.89, 20.00,
    21.33, 22.67, 24.00, 25.56, 27.33, 29.33, 31.56,
)
_ANT_PAT = np.array([10.0 ** (-db / 20.0) for db in _ANT_PAT_DB])

TWO_PI = 2.0 * np.pi
_SBAS_GEO_RADIUS = 42164000.0
_SBAS_RANGE = 3.8e7
_GALILEO_SEC_BIT = 0.004          # 250 bps secondary / data symbol length

#: Default synthetic SBAS satellites: (PRN, longitude offset from user, deg)
SBAS_DEFAULT = ((120, -15.0), (123, 0.0), (126, 15.0))

# ----------------------------------------------------------------------
# BeiDou B1I band planning
# ----------------------------------------------------------------------
#: Guard band kept between a signal and the edge of the sampled band (Hz).
B1I_MARGIN = 1.5e6
#: B1I offset from the L1 centre: 1575.42 - 1561.098 = 14.322 MHz.
B1I_OFFSET = CARR_FREQ_L1 - CARR_FREQ_B1I
#: Suggested B210-friendly band that keeps B1I together with L1/E1.
B1I_SUGGEST_CENTER = 1568.0e6
B1I_SUGGEST_FS = 30.0e6
#: B210 tuning limits used to validate a suggested band.
B210_MAX_FS = 56.0e6
B210_MIN_CENTER = 70.0e6
B210_MAX_CENTER = 6.0e9


def b1i_band_fits(fs: float, center_freq: float,
                  margin: float = B1I_MARGIN) -> bool:
    """True when B1I (1561.098 MHz) fits inside the sampled band.

    The engine drops a signal when ``abs(carrier - center) > fs/2 - margin``;
    this mirrors that criterion.
    """
    try:
        fs = float(fs)
        center_freq = float(center_freq)
    except (TypeError, ValueError):
        return False
    if fs <= 0:
        return False
    return abs(CARR_FREQ_B1I - center_freq) <= fs / 2.0 - margin


def b210_band_ok(fs: float, center_freq: float) -> bool:
    """True when ``fs``/``center_freq`` are within the B210 tuning limits."""
    try:
        fs = float(fs)
        center_freq = float(center_freq)
    except (TypeError, ValueError):
        return False
    return (0.0 < fs <= B210_MAX_FS
            and B210_MIN_CENTER <= center_freq <= B210_MAX_CENTER)


def suggest_b1i_band(fs: float, center_freq: float, enabled_systems,
                     center: float = B1I_SUGGEST_CENTER,
                     suggest_fs: float = B1I_SUGGEST_FS,
                     margin: float = B1I_MARGIN) -> dict | None:
    """Suggest a wideband centre/``fs`` that keeps B1I together with L1/E1.

    ``enabled_systems`` is an iterable of constellation letters (``G``/``E``/
    ``J``/``C``).  Returns ``None`` when BeiDou (``C``) is disabled or the
    current ``fs``/``center_freq`` already fit B1I, so a default run stays
    unchanged.  Otherwise returns ``{"center_freq": float, "fs": float,
    "fits": bool}`` with B210-validated values.  The chosen ``fs`` is widened,
    if needed, so that ``fs/2 >= |B1I - center| + margin``.
    """
    systems = {str(s).upper() for s in (enabled_systems or ())}
    if "C" not in systems:
        return None
    if b1i_band_fits(fs, center_freq, margin):
        return None

    center_sug = float(center)
    fs_sug = max(float(suggest_fs),
                 2.0 * (abs(CARR_FREQ_B1I - center_sug) + margin))
    if not b210_band_ok(fs_sug, center_sug):
        return None
    return {"center_freq": center_sug, "fs": fs_sug,
            "fits": b1i_band_fits(fs_sug, center_sug, margin),
            "offset_hz": abs(CARR_FREQ_B1I - center_sug)}


def visible_beidou_count(by_sv: dict, iono, xyz_fn,
                         start: GpsTime, el_mask_rad: float = 0.0) -> int:
    """Number of BeiDou B1I channels the engine would allocate.

    Mirrors the geometry/elevation checks of :meth:`SignalEngine._allocate`
    but deliberately ignores the ``fs``/centre band test, so the auto-band
    logic can tell whether widening the band for B1I is worthwhile at all.
    Returns 0 when ``by_sv`` holds no usable/visible BeiDou ephemerides.
    """
    xyz = xyz_fn(start)
    g = GpsTime(start.week, start.sec)
    count = 0
    for key in by_sv:
        if not str(key).startswith("C"):
            continue
        eph = select_ephemeris(by_sv, key, g)
        if eph is None:
            continue
        rho = compute_range(eph, iono, g, xyz)
        if rho.azel[1] < el_mask_rad:
            continue
        try:
            generate_b1i(eph.prn)
        except ValueError:
            continue
        count += 1
    return count


@dataclass
class _Geo:
    rng: float
    rate: float
    d: float
    azel: tuple[float, float]
    iono_delay: float


@dataclass
class Channel:
    system: str                      # 'G', 'E', 'J', 'S', 'C'
    prn: int
    kind: str                        # 'gps', 'galileo', 'qzss', 'sbas', 'beidou'
    eph: Ephemeris | None = None
    sv_ecef: np.ndarray | None = None
    # GPS codes
    ca: np.ndarray | None = None
    cp: np.ndarray | None = None
    cd: np.ndarray | None = None
    overlay: np.ndarray | None = None
    tmboc: np.ndarray | None = None
    # Galileo codes
    e1b: np.ndarray | None = None
    e1c: np.ndarray | None = None
    e1sec: np.ndarray | None = None
    # Galileo I/NAV 250 bps data: repeating int8 bit block + 30 s sub-frame epoch
    galileo_frame_start: GpsTime | None = None
    galileo_bits: np.ndarray | None = None
    # persistent phase state
    ca_phase: float = 0.0
    l1c_phase: float = 0.0
    e1_phase: float = 0.0
    b1_phase: float = 0.0
    carr: float = 0.0
    # per-signal RF parameters (defaults reproduce the L1/E1 signals exactly)
    f_offset: float = 0.0
    carrier_freq: float = CARR_FREQ_L1
    code_freq: float = CODE_FREQ_CA
    code_len: int = 1023
    lam: float = LAMBDA_L1
    # geometry / amplitude
    amp: float = 0.0
    azel: tuple[float, float] = (0.0, 0.0)
    # nav message (GPS/QZSS legacy)
    sbf: np.ndarray | None = None
    dwrd: np.ndarray | None = None
    frame_start: GpsTime | None = None
    l1c_frame_start: GpsTime | None = None
    cnav2: np.ndarray | None = None
    # SBAS 250 bps data (DO-229): repeating int8 bit block + 6 s frame epoch
    sbas_frame_start: GpsTime | None = None
    sbas_bits: np.ndarray | None = None
    # BeiDou B1I D1 data (50 bps channel bits) + 30 s frame epoch
    b1i_frame_start: GpsTime | None = None
    b1i_bits: np.ndarray | None = None

    @property
    def name(self) -> str:
        return f"{self.system}{self.prn:02d}" if self.system != "S" else f"S{self.prn}"


def _ant_gain(elev_deg: float) -> float:
    idx = int((90.0 - elev_deg) / 5.0)
    idx = min(max(idx, 0), len(_ANT_PAT) - 1)
    return float(_ANT_PAT[idx])


def _synthetic_range(sv: np.ndarray, iono: IonoUtc, g: GpsTime,
                     xyz: np.ndarray) -> _Geo:
    los = sv - xyz
    rng = float(np.linalg.norm(los))
    lat, lon, h = xyz2llh(xyz)
    tmat = ltcmat(lat, lon)
    azel = neu2azel(ecef2neu(los, tmat))
    delay = ionospheric_delay(iono, g, [lat, lon, h], azel)
    return _Geo(rng + delay, 0.0, rng, (azel[0], azel[1]), delay)


class SignalEngine:
    """Generates a continuous stream of baseband samples for many satellites."""

    def __init__(
        self,
        by_sv: dict[str, list[Ephemeris]],
        iono: IonoUtc,
        xyz_fn: Callable[[GpsTime], np.ndarray],
        start: GpsTime,
        fs: float,
        center_freq: float = CARR_FREQ_L1,
        enable_ca: bool = True,
        enable_l1c: bool = True,
        enable_galileo: bool = True,
        enable_qzss: bool = True,
        enable_sbas: bool = True,
        enable_beidou: bool = True,
        sbas_defs=SBAS_DEFAULT,
        el_mask: float = 0.0,
        amp_scale: float = 0.15,
        iono_enable: bool = True,
        l1c_data: str = "zeros",
        b1i_data: str = "d1",
        backend: str = "auto",
    ) -> None:
        self.by_sv = by_sv
        self.iono = iono
        self.xyz_fn = xyz_fn
        self.start = start
        self.fs = float(fs)
        self.center_freq = float(center_freq)
        self.enable_ca = enable_ca
        self.enable_l1c = enable_l1c
        self.enable_galileo = enable_galileo
        self.enable_qzss = enable_qzss
        self.enable_sbas = enable_sbas
        self.enable_beidou = enable_beidou
        self.sbas_defs = tuple(sbas_defs)
        self.el_mask = el_mask
        self.amp_scale = amp_scale
        self.l1c_data = l1c_data
        self.b1i_data = ("d1" if str(b1i_data).strip().lower()
                         in ("d1", "real", "nav", "") else "placeholder")
        self._nh = np.asarray(NH_CODE, dtype=np.int8)
        if backend == "cpu":
            self.backend = "cpu"
        elif backend == "cuda" or (backend == "auto" and cuda_available()):
            self.backend = "cuda" if cuda_available() else "cpu"
        else:
            self.backend = "cpu"
        self._gpu_cache: dict[int, tuple] = {}
        self.b1i_dropped: list[str] = []   # BeiDou channels rejected by aliasing
        self.gpu_name: str = ""
        if self.backend == "cuda" and _cp is not None:
            try:
                props = _cp.cuda.runtime.getDeviceProperties(0)
                self.gpu_name = props["name"].decode() if isinstance(
                    props["name"], bytes) else str(props["name"])
            except Exception:
                self.gpu_name = "CUDA"
        if not iono_enable:
            self.iono = IonoUtc(enable=False, vflg=False)

        self.g = GpsTime(start.week, start.sec)
        self.channels: list[Channel] = self._allocate()

    # ------------------------------------------------------------------
    def _geometry(self, ch: Channel, g: GpsTime, xyz: np.ndarray) -> _Geo:
        if ch.kind == "sbas" and ch.sv_ecef is not None:
            return _synthetic_range(ch.sv_ecef, self.iono, g, xyz)
        assert ch.eph is not None
        rho = compute_range(ch.eph, self.iono, g, xyz)
        return _Geo(rho.range, rho.rate, rho.d, (rho.azel[0], rho.azel[1]),
                    rho.iono_delay)

    def _make_gps_channel(self, system: str, prn: int,
                          eph: Ephemeris, code_prn: int | None = None) -> Channel:
        cprn = prn if code_prn is None else code_prn
        ch = Channel(system=system, prn=prn, kind="gps", eph=eph)
        if self.enable_ca:
            try:
                ch.ca = code_bipolar(cprn)
            except ValueError:
                ch.ca = None
        if self.enable_l1c:
            try:
                ch.cp = ranging_code(cprn, "cp")
                ch.cd = ranging_code(cprn, "cd")
                ch.overlay = overlay_code(cprn)
                ch.tmboc = tmboc_is_boc61()
            except ValueError:
                ch.cp = ch.cd = ch.overlay = ch.tmboc = None
        ch.carr = float(np.random.uniform(0.0, TWO_PI))
        return ch

    def _allocate(self) -> list[Channel]:
        chans: list[Channel] = []
        xyz = self.xyz_fn(self.g)
        for key in sorted(self.by_sv):
            system = key[0]
            if system not in ("G", "E", "J", "C"):
                continue
            eph = select_ephemeris(self.by_sv, key, self.g)
            if eph is None:
                continue
            if system == "G" and not (self.enable_ca or self.enable_l1c):
                continue
            if system == "E" and not self.enable_galileo:
                continue
            if system == "J" and not self.enable_qzss:
                continue
            if system == "C" and not self.enable_beidou:
                continue
            rho = compute_range(eph, self.iono, self.g, xyz)
            if rho.azel[1] < self.el_mask:
                continue

            if system in ("G", "J"):
                # GPS L1C ranging/overlay codes are tabulated for PRN 1..32;
                # QZSS L1 C/A and L1C use their own code PRN (193..206).
                code_prn = (192 + eph.prn) if system == "J" else eph.prn
                ch = self._make_gps_channel(system, eph.prn, eph,
                                            code_prn=code_prn)
                if system == "J":
                    ch.kind = "qzss"
                g0 = GpsTime(self.g.week, float(int(self.g.sec // 30) * 30))
                ch.frame_start = g0
                ch.sbf = eph2sbf(eph, self.iono, transmit_week=self.g.week)
                ch.dwrd = np.zeros(60, dtype=np.int64)
                generate_nav_msg(g0, ch.sbf, ch.dwrd, 1)
            elif system == "C":  # BeiDou B1I
                f_off = CARR_FREQ_B1I - self.center_freq
                # Drop out-of-band B1I before drawing any RNG state so that the
                # other signals are bit-identical to a run without BeiDou.
                if abs(f_off) > self.fs / 2.0 - 1.5e6:
                    self.b1i_dropped.append(f"C{eph.prn:02d}")
                    continue
                ch = Channel(system="C", prn=eph.prn, kind="beidou", eph=eph)
                try:
                    ch.ca = generate_b1i(eph.prn)
                except ValueError:
                    continue
                ch.f_offset = f_off
                ch.carrier_freq = CARR_FREQ_B1I
                ch.code_freq = B1I_CODE_RATE
                ch.code_len = B1I_CODE_CHIPS
                ch.lam = SPEED_OF_LIGHT / CARR_FREQ_B1I
                ch.carr = float(np.random.uniform(0.0, TWO_PI))
            else:  # Galileo E1
                ch = Channel(system="E", prn=eph.prn, kind="galileo", eph=eph)
                ch.e1b = e1_code(eph.prn, "b")
                ch.e1c = e1_code(eph.prn, "c")
                ch.e1sec = e1_secondary()
                ch.carr = float(np.random.uniform(0.0, TWO_PI))

            # Keep only signals that fall inside the sampled band (+/- fs/2).
            # Zero-offset (L1/E1) channels are always in band; this safety net
            # only ever rejects off-centre signals such as BeiDou B1I.
            if ch.f_offset != 0.0 and abs(ch.f_offset) > self.fs / 2.0 - 1.5e6:
                if ch.kind == "beidou":
                    self.b1i_dropped.append(ch.name)
                continue

            self._init_phases(ch, rho, xyz)
            chans.append(ch)

        if self.enable_sbas:
            chans.extend(self._allocate_sbas(xyz))
        return chans

    def _init_phases(self, ch: Channel, rho: _Geo, xyz: np.ndarray) -> None:
        ch.azel = rho.azel
        ch.amp = (20200000.0 / rho.d) * _ant_gain(rho.azel[1] * R2D) * self.amp_scale
        rng = rho.rng if hasattr(rho, "rng") else rho.range
        ms = (sub_gps_time(self.g, self.g)
              + 6.0 - rng / SPEED_OF_LIGHT) * 1000.0
        if ch.kind == "beidou":
            ch.b1_phase = (ms % 1.0) * B1I_CODE_CHIPS
            # D1 frame epoch aligned to 30 s (frame = 5 subframes x 6 s).
            b0 = GpsTime(self.g.week, float(int(self.g.sec // 30) * 30))
            ch.b1i_frame_start = b0
            ch.b1i_bits = self._b1i_frame_bits(ch, b0)
        else:
            ch.ca_phase = (ms % 1.0) * 1023.0
        ch.l1c_phase = (ms % 10.0) * 1023.0
        ch.e1_phase = (ms % 4.0) * 1023.0
        if ch.kind == "gps" or (ch.kind == "qzss" and ch.cp is not None):
            l0 = GpsTime(self.g.week, float(int(self.g.sec // 18) * 18))
            ch.l1c_frame_start = l0
            ch.cnav2 = self._l1c_frame_bits(ch, l0)
        elif ch.kind == "sbas":
            # 6 s frame aligns the 3-message 0x53/0x9A/0xC6 preamble cycle
            # with the 6 s GPS subframe epoch (DO-229 A.1).
            s0 = GpsTime(self.g.week, float(int(self.g.sec // 6) * 6))
            ch.sbas_frame_start = s0
            ch.sbas_bits = self._sbas_block_bits(ch, s0)
        elif ch.kind == "galileo":
            # I/NAV nominal sub-frame = 30 s, T0 aligned with GST modulo 30 s.
            g0 = GpsTime(self.g.week, float(int(self.g.sec // 30) * 30))
            ch.galileo_frame_start = g0
            ch.galileo_bits = self._galileo_block_bits(ch, g0)

    def set_amp_scale(self, scale: float) -> None:
        """Re-scale every channel amplitude for the current geometry.

        Used by the runner to apply the automatic scene-wide amplitude derived
        from the allocated channels (``cfg.amp_scale is None``).  The per-block
        synthesis also recomputes ``ch.amp`` from ``self.amp_scale``, so this
        keeps the logged ``channel_info`` amplitudes consistent.
        """
        self.amp_scale = float(scale)
        xyz = self.xyz_fn(self.g)
        for ch in self.channels:
            rho = self._geometry(ch, self.g, xyz)
            ch.azel = rho.azel
            ch.amp = (20200000.0 / rho.d) * _ant_gain(
                rho.azel[1] * R2D) * self.amp_scale

    def _allocate_sbas(self, xyz: np.ndarray) -> list[Channel]:
        lat, lon, _h = xyz2llh(xyz)
        user_lon = lon * R2D
        out: list[Channel] = []
        for prn, dlon in self.sbas_defs:
            glon = np.radians(user_lon + dlon)
            sv = np.array([_SBAS_GEO_RADIUS * np.cos(glon),
                           _SBAS_GEO_RADIUS * np.sin(glon), 0.0])
            ch = Channel(system="S", prn=int(prn), kind="sbas", sv_ecef=sv)
            try:
                ch.ca = code_bipolar(int(prn))
            except ValueError:
                continue
            ch.carr = float(np.random.uniform(0.0, TWO_PI))
            rho = _synthetic_range(sv, self.iono, self.g, xyz)
            if rho.azel[1] < self.el_mask:
                continue
            self._init_phases(ch, rho, xyz)
            out.append(ch)
        return out

    def _l1c_frame_bits(self, ch: Channel, frame_start: GpsTime) -> np.ndarray:
        if self.l1c_data == "cnav2":
            from .cnav2 import cnav2_frame
            return cnav2_frame(ch.eph, frame_start)
        return np.zeros(1800, dtype=np.int8)

    def _sbas_block_bits(self, ch: Channel, frame_start: GpsTime) -> np.ndarray:
        """Real DO-229 250 bps block for a synthetic GEO (6 messages = 6 s).

        MT9 (GEO ephemeris) and MT17 (GEO almanac) are derived from the
        channel's fixed ECEF position with zero ECEF velocity, so a receiver
        decodes a self-consistent (static) GEO orbit.
        """
        t0_sod = frame_start.sec % 86400.0
        return sbas_bit_block(ch.prn, ch.sv_ecef, t0_sod=t0_sod)

    def _galileo_block_bits(self, ch: Channel, frame_start: GpsTime) -> np.ndarray:
        """Real Galileo E1-B I/NAV sub-frame (15 pages = 30 s = 7500 bits).

        Words 1..6 carry the broadcast ephemeris/clock, the ionosphere/BGD/GST
        model and the GST-UTC model; the remaining pages are the spare word.
        """
        return inav_bit_block(ch.eph, gst_week=frame_start.week,
                              gst_tow=frame_start.sec)

    def _b1i_frame_bits(self, ch: Channel, frame_start: GpsTime) -> np.ndarray | None:
        """Real BeiDou B1I D1 frame (1500 channel bits = 30 s at 50 bps).

        Returns ``None`` in ``placeholder`` mode (the engine then transmits a
        constant +1 data bit, still NH20-framed).
        """
        if self.b1i_data != "d1" or ch.eph is None:
            return None
        return d1_frame_block(ch.eph, frame_start.sec)

    # ------------------------------------------------------------------
    def _roll_frames(self) -> None:
        g = self.g
        for ch in self.channels:
            if ch.frame_start is not None:
                while g.sec - ch.frame_start.sec >= 30.0:
                    ch.frame_start = GpsTime(ch.frame_start.week,
                                             ch.frame_start.sec + 30.0)
                    if ch.frame_start.sec >= 604800.0:
                        ch.frame_start = GpsTime(
                            ch.frame_start.week + 1, ch.frame_start.sec - 604800.0)
                    ch.sbf = eph2sbf(ch.eph, self.iono,
                                     transmit_week=ch.frame_start.week)
                    generate_nav_msg(ch.frame_start, ch.sbf, ch.dwrd, 0)
            if ch.l1c_frame_start is not None:
                while g.sec - ch.l1c_frame_start.sec >= 18.0:
                    ch.l1c_frame_start = GpsTime(
                        ch.l1c_frame_start.week, ch.l1c_frame_start.sec + 18.0)
                    if ch.l1c_frame_start.sec >= 604800.0:
                        ch.l1c_frame_start = GpsTime(
                            ch.l1c_frame_start.week + 1,
                            ch.l1c_frame_start.sec - 604800.0)
                    ch.cnav2 = self._l1c_frame_bits(ch, ch.l1c_frame_start)
            if ch.sbas_frame_start is not None:
                while sub_gps_time(g, ch.sbas_frame_start) >= 6.0:
                    ch.sbas_frame_start = GpsTime(
                        ch.sbas_frame_start.week,
                        ch.sbas_frame_start.sec + 6.0)
                    if ch.sbas_frame_start.sec >= 604800.0:
                        ch.sbas_frame_start = GpsTime(
                            ch.sbas_frame_start.week + 1,
                            ch.sbas_frame_start.sec - 604800.0)
                    ch.sbas_bits = self._sbas_block_bits(ch, ch.sbas_frame_start)
            if ch.galileo_frame_start is not None:
                while sub_gps_time(g, ch.galileo_frame_start) >= 30.0:
                    ch.galileo_frame_start = GpsTime(
                        ch.galileo_frame_start.week,
                        ch.galileo_frame_start.sec + 30.0)
                    if ch.galileo_frame_start.sec >= 604800.0:
                        ch.galileo_frame_start = GpsTime(
                            ch.galileo_frame_start.week + 1,
                            ch.galileo_frame_start.sec - 604800.0)
                    ch.galileo_bits = self._galileo_block_bits(
                        ch, ch.galileo_frame_start)
            if ch.b1i_frame_start is not None:
                while sub_gps_time(g, ch.b1i_frame_start) >= 30.0:
                    ch.b1i_frame_start = GpsTime(
                        ch.b1i_frame_start.week,
                        ch.b1i_frame_start.sec + 30.0)
                    if ch.b1i_frame_start.sec >= 604800.0:
                        ch.b1i_frame_start = GpsTime(
                            ch.b1i_frame_start.week + 1,
                            ch.b1i_frame_start.sec - 604800.0)
                    ch.b1i_bits = self._b1i_frame_bits(ch, ch.b1i_frame_start)

    # ------------------------------------------------------------------
    def _device(self, arr, xp, cache: bool = True):
        """Return ``arr`` as a device array when running on CUDA (cached)."""
        if xp is np or arr is None:
            return arr
        if not cache:
            return _cp.asarray(arr)
        key = id(arr)
        hit = self._gpu_cache.get(key)
        if hit is not None and hit[0] is arr:
            return hit[1]
        gpu = _cp.asarray(arr)
        self._gpu_cache[key] = (arr, gpu)
        return gpu

    def _legacy_term(self, ch, g, rho, t, f_code, carrier, xp):
        ca = self._device(ch.ca, xp)
        dwrd = self._device(ch.dwrd, xp, cache=False)
        cp = ch.ca_phase + f_code * t
        chip = xp.floor(cp).astype(xp.int64) % 1023
        code = ca[chip].astype(xp.float64)             # +/-1
        ms = (sub_gps_time(g, ch.frame_start)
              + 6.0 - rho.rng / SPEED_OF_LIGHT) * 1000.0
        ims = int(ms)
        flat = ims + xp.floor(cp / 1023.0).astype(xp.int64)
        word = xp.clip(flat // 600, 0, 59)
        bitpos = (flat % 600) // 20
        data = (((dwrd[word] >> (29 - bitpos)) & 1) * 2 - 1).astype(xp.float64)
        return code * data * carrier

    def _l1c_term(self, ch, g, t, f_code, carrier, xp):
        cp_code = self._device(ch.cp, xp)
        cd_code = self._device(ch.cd, xp)
        overlay = self._device(ch.overlay, xp)
        tmboc = self._device(ch.tmboc, xp)
        cnav2 = self._device(ch.cnav2, xp, cache=False)
        cp = ch.l1c_phase + f_code * t
        chip = xp.floor(cp).astype(xp.int64) % L1C_CODE_LEN
        frac = cp - xp.floor(cp)
        m = xp.where(tmboc[chip], 6.0, 1.0)
        sub = xp.sign(xp.sin(TWO_PI * m * frac))
        sub[sub == 0.0] = 1.0
        boc11 = xp.sign(xp.sin(TWO_PI * frac))
        boc11[boc11 == 0.0] = 1.0
        sym = int((g.sec - ch.l1c_frame_start.sec) / 0.01) % 1800
        ov = 1 - 2 * overlay[sym]
        databit = 1 - 2 * cnav2[sym]
        pilot = (1 - 2 * cp_code[chip]) * ov
        data_ch = (1 - 2 * cd_code[chip]) * databit
        return (xp.sqrt(0.75) * pilot * sub
                + xp.sqrt(0.25) * data_ch * boc11) * carrier

    def _galileo_term(self, ch, g, t, f_code, carrier, xp):
        e1b = self._device(ch.e1b, xp)
        e1c = self._device(ch.e1c, xp)
        e1sec = self._device(ch.e1sec, xp)
        cp = ch.e1_phase + f_code * t
        chip = xp.floor(cp).astype(xp.int64) % 4092
        frac = cp - xp.floor(cp)
        boc11 = xp.sign(xp.sin(TWO_PI * frac))
        boc11[boc11 == 0.0] = 1.0
        boc61 = xp.sign(xp.sin(TWO_PI * 6.0 * frac))
        boc61[boc61 == 0.0] = 1.0
        comp_b = CBOC_ALPHA * boc11 + CBOC_BETA * boc61
        comp_c = CBOC_ALPHA * boc11 - CBOC_BETA * boc61
        # One I/NAV symbol spans 4 ms = one E1-B code period (250 bps).  The
        # index is taken relative to the 30 s sub-frame epoch so it stays small
        # and exact on the float32 CUDA path (the E1-C secondary code period is
        # 25 symbols, so k % 25 does not depend on the epoch).
        if ch.galileo_frame_start is not None:
            base = g.sec - ch.galileo_frame_start.sec
        else:
            base = g.sec
        k = xp.floor((base + t) / _GALILEO_SEC_BIT).astype(xp.int64)
        sec_idx = k % 25
        if ch.galileo_bits is not None:
            bits = self._device(ch.galileo_bits, xp, cache=False)
            sym = k % bits.shape[0]
            data = (1.0 - 2.0 * bits[sym]).astype(t.dtype)
        else:
            data = xp.ones_like(t)                    # fallback E1-B data
        sig = (e1b[chip] * data * comp_b
               + e1c[chip] * e1sec[sec_idx] * comp_c)
        return 0.7071067811865476 * sig * carrier

    def _sbas_term(self, ch, g, t, f_code, carrier, xp):
        ca = self._device(ch.ca, xp)
        cp = ch.ca_phase + f_code * t
        chip = xp.floor(cp).astype(xp.int64) % 1023
        code = ca[chip].astype(xp.float64)
        if ch.sbas_bits is not None and ch.sbas_frame_start is not None:
            block = self._device(ch.sbas_bits, xp, cache=False)
            # One DO-229 data bit spans 4 ms = 4 C/A code periods (250 bps).
            sym = int(sub_gps_time(g, ch.sbas_frame_start) / 0.004)
            bit = block[sym % block.shape[0]]
            data = (1.0 - 2.0 * bit).astype(xp.float64)
        else:
            data = xp.ones_like(t)
        return code * data * carrier

    def _b1i_term(self, ch, g, t, f_code, carrier, xp):
        """BPSK(2) B1I: primary code x NH20 (1 kbps) x D1 data (50 bps)."""
        ca = self._device(ch.ca, xp)
        nh = self._device(self._nh, xp)
        cp = ch.b1_phase + f_code * t
        chip = xp.floor(cp).astype(xp.int64) % B1I_CODE_CHIPS
        code = ca[chip].astype(xp.float64)
        if ch.b1i_frame_start is not None:
            base = g.sec - ch.b1i_frame_start.sec
        else:
            base = g.sec
        sec = base + t
        # NH20: one chip per 1 ms ranging-code period, 20 ms period.
        nh_idx = xp.floor(sec * 1000.0).astype(xp.int64) % NH_CODE.shape[0]
        nh_bit = (1.0 - 2.0 * nh[nh_idx]).astype(xp.float64)
        if ch.b1i_bits is not None:
            bits = self._device(ch.b1i_bits, xp, cache=False)
            # One D1 channel bit spans 20 ms (50 bps).
            sym = xp.floor(sec * 50.0).astype(xp.int64) % bits.shape[0]
            data = (1.0 - 2.0 * bits[sym]).astype(xp.float64)
        else:
            data = xp.ones_like(t)                    # placeholder +1 data
        return code * nh_bit * data * carrier

    def _synthesise(self, nsamp: int, xp, complex_dtype, real_dtype) -> np.ndarray:
        dt = 1.0 / self.fs
        t = xp.arange(nsamp, dtype=real_dtype) * dt
        acc = xp.zeros(nsamp, dtype=complex_dtype)
        xyz = self.xyz_fn(self.g)
        g = self.g
        self._roll_frames()

        for ch in self.channels:
            rho = self._geometry(ch, g, xyz)
            ch.azel = rho.azel
            ch.amp = (20200000.0 / rho.d) * _ant_gain(rho.azel[1] * R2D) * self.amp_scale
            # Generic carrier/code Doppler, identical to the old L1 formula for
            # channels with f_offset=0, carrier=L1, code=C/A.
            f_carr = ch.f_offset - rho.rate / ch.lam
            f_code = ch.code_freq + f_carr * (ch.code_freq / ch.carrier_freq)
            carrier = xp.exp(1j * (ch.carr + TWO_PI * f_carr * t))

            if ch.kind in ("gps", "qzss"):
                if ch.ca is not None:
                    acc += ch.amp * self._legacy_term(ch, g, rho, t, f_code,
                                                      carrier, xp)
                if ch.cp is not None:
                    acc += ch.amp * self._l1c_term(ch, g, t, f_code, carrier, xp)
            elif ch.kind == "galileo":
                acc += ch.amp * self._galileo_term(ch, g, t, f_code, carrier, xp)
            elif ch.kind == "sbas":
                acc += ch.amp * self._sbas_term(ch, g, t, f_code, carrier, xp)
            elif ch.kind == "beidou":
                acc += ch.amp * self._b1i_term(ch, g, t, f_code, carrier, xp)

            ch.carr = (ch.carr + TWO_PI * f_carr * nsamp * dt) % TWO_PI
            if ch.kind == "beidou":
                ch.b1_phase = (ch.b1_phase + f_code * nsamp * dt) % B1I_CODE_CHIPS
            else:
                ch.ca_phase = (ch.ca_phase + f_code * nsamp * dt) % 1023.0
            if ch.kind in ("gps", "qzss"):
                ch.l1c_phase = (ch.l1c_phase + f_code * nsamp * dt) % L1C_CODE_LEN
            elif ch.kind == "galileo":
                ch.e1_phase = (ch.e1_phase + f_code * nsamp * dt) % 4092.0

        self.g.sec += nsamp * dt
        while self.g.sec >= 604800.0:
            self.g.sec -= 604800.0
            self.g.week += 1
        return acc

    def generate_block(self, nsamp: int) -> np.ndarray:
        """Return ``nsamp`` baseband samples and advance the engine clock."""
        nsamp = int(nsamp)
        if self.backend == "cuda" and _cp is not None:
            acc = self._synthesise(nsamp, _cp, _cp.complex64, _cp.float32)
            return _cp.asnumpy(acc)
        return self._synthesise(nsamp, np, np.complex128, np.float64)

    # ------------------------------------------------------------------
    @property
    def channel_info(self) -> list[dict]:
        out = []
        for ch in self.channels:
            out.append({
                "system": ch.system,
                "prn": ch.prn,
                "name": ch.name,
                "kind": ch.kind,
                "elev": ch.azel[1],
                "azim": ch.azel[0],
                "amp": ch.amp,
            })
        return out
