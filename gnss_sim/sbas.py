"""RTCA DO-229 SBAS L1 C/A navigation message construction (250 bps).

Every SBAS L1 message is a 250-bit block transmitted at 250 bit/s (one message
per second, before the 1/2-rate convolutional FEC that halves the *channel*
rate to 500 sym/s at the user level)::

    bits   0..  7  preamble        (24-bit sequence 0x53, 0x9A, 0xC6 spread
                                    over three successive messages)
    bits   8.. 13  message type    (6-bit MT, 0..63)
    bits  14..225  data field      (212 bits, layout depends on MT)
    bits 226..249  CRC-24Q parity  (24 bits)

References
----------
* RTCA DO-229D, Appendix A "Signal characteristics and format":
  - A.1 general 250-bit message structure and preamble;
  - A.4.4.11 Message Type 9  (GEO Navigation / ephemeris);
  - A.4.4.12 Message Type 17 (GEO Satellite Almanac);
  - Appendix A parity: generator polynomial
    ``G(x) = x^24+x^23+x^18+x^17+x^14+x^11+x^10+x^7+x^6+x^5+x^4+x^3+x+1``
    i.e. mask ``0x1864CFB`` (the ubiquitous CRC-24Q).
* ICAO SARPS Annex 10 Vol. I, Appendix B (same field layouts).
* Field bit positions/scaling cross-checked against
  ``semuconsulting/pygnssutils`` ``rawnav_subframes_sba.py`` (BSD-3, which
  itself cites DO-229D) and RTKLIB ``rtklib_sbas.cc::decode_sbstype9``.

The CRC implementation in this module was validated bit-for-bit against three
captured WAAS/EGNOS messages from the IGS ``geo_sbas.txt`` sample record (see
:func:`self_test`); ``crc24(body)`` reproduces their broadcast parity exactly
and the full 250-bit codeword has zero remainder.

Scale factors for MT17 (from DO-229D / pygnssutils, the authoritative packing
used here)::

    data id     2 bits        health      8 bits
    PRN         8 bits        t0         11 bits, 64 s
    Xg, Yg     15 bits signed, 2600 m      Zg   9 bits signed, 26000 m
    Xgdot,Ygdot 3 bits signed,   10 m/s    Zgdot 4 bits signed,    60 m/s

Note: some secondary sources quote a different Zgdot scale (40.96 m/s per
3GPP TS 37.355, 60 m/s per DO-229D/pygnssutils); the DO-229D value (60 m/s) is
used here.  ``t0`` is packed unsigned (0..2047 * 64 s) because the DO-229D
range up to 86336 s exceeds a signed 11-bit field.
"""

from __future__ import annotations

from typing import Iterable, Iterator, Sequence

import numpy as np

#: 24-bit generator mask of the DO-229 CRC (CRC-24Q, applied MSB-first,
#: init 0, no reflection, no final XOR).
CRC24Q_POLY = 0x1864CFB

SBAS_MSG_BITS = 250
SBAS_DATA_BITS = 212
SBAS_RATE_BPS = 250
SYMBOL_SECONDS = 0.004          # 4 ms data bit -> 4 C/A code periods

#: The three 8-bit preambles, cycled once per message.  The start of every
#: other 24-bit preamble is synchronous with a 6-second GPS subframe epoch, so
#: a message starting an even multiple of 3 s (and 6 s) uses 0x53.
PREAMBLES: tuple[int, ...] = (0x53, 0x9A, 0xC6)

#: Realistic repeating broadcast cycle used by the engine.  Length must be a
#: multiple of 6 so the preamble rotation stays aligned to the 6-second GPS
#: subframe epoch; MT9 (ephemeris), MT17 (almanac) and null MT63 dominate.
DEFAULT_SCHEDULE: tuple[int, ...] = (9, 17, 63, 9, 17, 63)


# ----------------------------------------------------------------------
# CRC-24Q
# ----------------------------------------------------------------------
def crc24(bits: Iterable[int]) -> int:
    """Return the 24-bit DO-229 CRC of ``bits`` (MSB first).

    Implements polynomial division of ``M(x) * x^24`` by ``G(x)`` over GF(2)
    with seed 0; the returned value is the remainder, i.e. the parity placed in
    bits 226..249 of the message.  Feed the 226 body bits (preamble + type +
    data); feeding a full valid 250-bit codeword returns 0.
    """
    crc = 0
    for bit in bits:
        crc ^= (int(bit) & 1) << 23
        if crc & 0x800000:
            crc = ((crc << 1) ^ CRC24Q_POLY) & 0xFFFFFF
        else:
            crc = (crc << 1) & 0xFFFFFF
    return crc


# ----------------------------------------------------------------------
# bit helpers
# ----------------------------------------------------------------------
def _uint_bits(value: int, width: int) -> list[int]:
    return [(int(value) >> (width - 1 - i)) & 1 for i in range(width)]


def _int_bits(value: int, width: int) -> list[int]:
    return _uint_bits(int(value) & ((1 << width) - 1), width)


def _bits_to_uint(bits: Sequence[int]) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | (int(bit) & 1)
    return value


def _bits_to_int(bits: Sequence[int]) -> int:
    value = _bits_to_uint(bits)
    if bits and int(bits[0]) & 1:
        value -= 1 << len(bits)
    return value


def _scale(value: float, scale: float, width: int, signed: bool) -> int:
    """Quantise ``value`` with ``scale`` and clamp to a ``width``-bit field."""
    raw = int(round(float(value) / scale)) if scale else int(round(float(value)))
    if signed:
        lo, hi = -(1 << (width - 1)), (1 << (width - 1)) - 1
    else:
        lo, hi = 0, (1 << width) - 1
    return max(lo, min(hi, raw))


def _field(value: float, scale: float, width: int, signed: bool) -> list[int]:
    return _int_bits(_scale(value, scale, width, signed), width)


def _unscale(raw: int, scale: float) -> float:
    return float(raw) * scale if scale else float(raw)


def _pad(data_bits: Sequence[int], width: int = SBAS_DATA_BITS) -> list[int]:
    out = [int(b) & 1 for b in data_bits]
    if len(out) > width:
        raise ValueError(f"data field too long: {len(out)} > {width}")
    return out + [0] * (width - len(out))


# ----------------------------------------------------------------------
# message framing
# ----------------------------------------------------------------------
def make_message(msg_type: int, data_bits: Sequence[int],
                 preamble: int = PREAMBLES[0]) -> np.ndarray:
    """Build one 250-bit SBAS message as ``int8`` 0/1 bits.

    ``msg_type`` is 0..63, ``data_bits`` the 212-bit data field (shorter input
    is zero-padded on the right), ``preamble`` one of 0x53/0x9A/0xC6.
    """
    body = _uint_bits(int(preamble) & 0xFF, 8)
    body += _uint_bits(int(msg_type) & 0x3F, 6)
    body += _pad(data_bits)
    parity = crc24(body)
    return np.array(body + _uint_bits(parity, 24), dtype=np.int8)


def parse_message(bits: Sequence[int]) -> dict:
    """Decode an SBAS message into its header/data/parity fields.

    Returns ``{"preamble", "msg_type", "data", "parity", "crc_ok"}``; ``data``
    are the 212 raw bits and ``crc_ok`` is True when the CRC matches.
    """
    b = [int(x) & 1 for x in bits]
    if len(b) != SBAS_MSG_BITS:
        raise ValueError(f"expected {SBAS_MSG_BITS} bits, got {len(b)}")
    parity = _bits_to_uint(b[226:250])
    return {
        "preamble": _bits_to_uint(b[0:8]),
        "msg_type": _bits_to_uint(b[8:14]),
        "data": b[14:226],
        "parity": parity,
        "crc_ok": crc24(b[:226]) == parity,
    }


def message_bytes(bits: Sequence[int]) -> bytes:
    """Pack a 250-bit message into 32 bytes (6 trailing zero pad bits)."""
    b = [int(x) & 1 for x in bits]
    b = b + [0] * (256 - len(b))
    return bytes(_bits_to_uint(b[i:i + 8]) for i in range(0, 256, 8))


def bits_to_bipolar(bits: Sequence[int]) -> np.ndarray:
    """Map 0/1 bits to +1/-1 (``1 - 2*bit``)."""
    return (1 - 2 * np.asarray(bits, dtype=np.int16)).astype(np.int8)


# ----------------------------------------------------------------------
# MT 9 - GEO Navigation Message (ephemeris), DO-229 A.4.4.11
# ----------------------------------------------------------------------
_MT9_SCALES = {
    "xpos": (0.08, 30), "ypos": (0.08, 30), "zpos": (0.4, 25),
    "xdot": (0.000625, 17), "ydot": (0.000625, 17), "zdot": (0.004, 18),
    "xdot2": (0.0000125, 10), "ydot2": (0.0000125, 10),
    "zdot2": (0.0000625, 10),
    "agf0": (2.0 ** -31, 12), "agf1": (2.0 ** -40, 8),
}


def mt9_data(*, t0_sod: float = 0.0, ura: int = 0,
             position: Sequence[float] = (0.0, 0.0, 0.0),
             velocity: Sequence[float] = (0.0, 0.0, 0.0),
             acceleration: Sequence[float] = (0.0, 0.0, 0.0),
             af0: float = 0.0, af1: float = 0.0,
             iodn: int = 0) -> list[int]:
    """Pack the 212-bit MT9 data field (GEO position/velocity/clock)."""
    bits: list[int] = []
    bits += _uint_bits(iodn & 0xFF, 8)          # reserved / IODN
    bits += _field(t0_sod, 16.0, 13, False)     # 13-bit time of day, 16 s
    bits += _uint_bits(ura & 0xF, 4)
    for key, value in zip(("xpos", "ypos", "zpos"), position):
        scale, width = _MT9_SCALES[key]
        bits += _field(value, scale, width, True)
    for key, value in zip(("xdot", "ydot", "zdot"), velocity):
        scale, width = _MT9_SCALES[key]
        bits += _field(value, scale, width, True)
    for key, value in zip(("xdot2", "ydot2", "zdot2"), acceleration):
        scale, width = _MT9_SCALES[key]
        bits += _field(value, scale, width, True)
    bits += _field(af0, _MT9_SCALES["agf0"][0], 12, True)
    bits += _field(af1, _MT9_SCALES["agf1"][0], 8, True)
    return _pad(bits)


def mt9_decode(data_bits: Sequence[int]) -> dict:
    """Inverse of :func:`mt9_data` (raw, unscaled field values)."""
    b = _pad(data_bits)
    off = 0

    def take(width: int) -> list[int]:
        nonlocal off
        chunk = b[off:off + width]
        off += width
        return chunk

    out = {"iodn": _bits_to_uint(take(8)),
           "t0_sod": _unscale(_bits_to_uint(take(13)), 16.0),
           "ura": _bits_to_uint(take(4))}
    for key in ("xpos", "ypos", "zpos", "xdot", "ydot", "zdot",
                "xdot2", "ydot2", "zdot2"):
        scale, width = _MT9_SCALES[key]
        out[key] = _unscale(_bits_to_int(take(width)), scale)
    out["af0"] = _unscale(_bits_to_int(take(12)), _MT9_SCALES["agf0"][0])
    out["af1"] = _unscale(_bits_to_int(take(8)), _MT9_SCALES["agf1"][0])
    return out


def make_mt9(*, preamble: int = PREAMBLES[0], **kw) -> np.ndarray:
    """Build a full 250-bit MT9 message (see :func:`mt9_data`)."""
    return make_message(9, mt9_data(**kw), preamble=preamble)


# ----------------------------------------------------------------------
# MT 17 - GEO Satellite Almanac, DO-229 A.4.4.12
# ----------------------------------------------------------------------
def _almanac_entry(alm: dict) -> list[int]:
    pos = tuple(alm.get("position", (0.0, 0.0, 0.0)))
    vel = tuple(alm.get("velocity", (0.0, 0.0, 0.0)))
    bits: list[int] = []
    bits += _uint_bits(int(alm.get("data_id", 0)) & 0x3, 2)
    bits += _uint_bits(int(alm.get("prn", 0)) & 0xFF, 8)
    bits += _uint_bits(int(alm.get("health", 0)) & 0xFF, 8)
    bits += _field(pos[0], 2600.0, 15, True)
    bits += _field(pos[1], 2600.0, 15, True)
    bits += _field(pos[2], 26000.0, 9, True)
    bits += _field(vel[0], 10.0, 3, True)
    bits += _field(vel[1], 10.0, 3, True)
    bits += _field(vel[2], 60.0, 4, True)
    return bits


def mt17_data(almanacs: Sequence[dict], *, t0_sod: float = 0.0) -> list[int]:
    """Pack the 212-bit MT17 data field (up to three GEO almanacs + t0)."""
    bits: list[int] = []
    entries = list(almanacs)[:3]
    while len(entries) < 3:
        entries.append({})                 # unused slots have PRN == 0
    for alm in entries:
        bits += _almanac_entry(alm)
    bits += _field(t0_sod, 64.0, 11, False)
    return _pad(bits)


def mt17_decode(data_bits: Sequence[int]) -> dict:
    """Inverse of :func:`mt17_data`; ``almanacs`` holds the 3 raw entries."""
    b = _pad(data_bits)
    out: dict = {"almanacs": []}
    for i in range(3):
        off = i * 67
        entry = {
            "data_id": _bits_to_uint(b[off:off + 2]),
            "prn": _bits_to_uint(b[off + 2:off + 10]),
            "health": _bits_to_uint(b[off + 10:off + 18]),
            "x": _unscale(_bits_to_int(b[off + 18:off + 33]), 2600.0),
            "y": _unscale(_bits_to_int(b[off + 33:off + 48]), 2600.0),
            "z": _unscale(_bits_to_int(b[off + 48:off + 57]), 26000.0),
            "vx": _unscale(_bits_to_int(b[off + 57:off + 60]), 10.0),
            "vy": _unscale(_bits_to_int(b[off + 60:off + 63]), 10.0),
            "vz": _unscale(_bits_to_int(b[off + 63:off + 67]), 60.0),
        }
        out["almanacs"].append(entry)
    out["t0_sod"] = _unscale(_bits_to_uint(b[201:212]), 64.0)
    return out


def make_mt17(almanacs: Sequence[dict] = (), *, preamble: int = PREAMBLES[0],
              t0_sod: float = 0.0) -> np.ndarray:
    """Build a full 250-bit MT17 message (see :func:`mt17_data`)."""
    return make_message(17, mt17_data(almanacs, t0_sod=t0_sod),
                        preamble=preamble)


# ----------------------------------------------------------------------
# Fast corrections (MT 2..5), integrity (MT 6), long term (MT 24/25),
# ionosphere (MT 26) and the null message (MT 63)
# ----------------------------------------------------------------------
def mt2_data(*, iodf: int = 0, iodp: int = 0,
             prc: Sequence[float] | None = None,
             udrei: Sequence[int] | None = None) -> list[int]:
    """Pack a fast-correction data field (MT 2..5), 13 satellites.

    ``prc`` are pseudorange corrections in metres (12-bit signed, 0.125 m);
    ``udrei`` the 13 integrity indices (default 0 = "use").
    """
    prc = list(prc) if prc is not None else [0.0] * 13
    udrei = list(udrei) if udrei is not None else [0] * 13
    bits = _uint_bits(iodf & 0x3, 2) + _uint_bits(iodp & 0x3, 2)
    for value in (prc + [0.0] * 13)[:13]:
        bits += _field(value, 0.125, 12, True)
    for value in (udrei + [0] * 13)[:13]:
        bits += _uint_bits(int(value) & 0xF, 4)
    return _pad(bits)


def mt6_data(*, iodf: Sequence[int] = (0, 0, 0, 0),
             udrei: Sequence[int] | None = None) -> list[int]:
    """Pack the MT6 integrity data field (4 IODF + 51 UDREI)."""
    udrei = list(udrei) if udrei is not None else [0] * 51
    bits: list[int] = []
    for value in (list(iodf) + [0, 0, 0, 0])[:4]:
        bits += _uint_bits(int(value) & 0x3, 2)
    for value in (udrei + [0] * 51)[:51]:
        bits += _uint_bits(int(value) & 0xF, 4)
    return _pad(bits)


def mt24_data(*, iodp: int = 0, block_id: int = 0, iodf: int = 0,
              prc: Sequence[float] | None = None,
              udrei: Sequence[int] | None = None,
              velocity_code: int = 0) -> list[int]:
    """Pack the MT24 mixed fast / long-term data field (long term all zero)."""
    prc = list(prc) if prc is not None else [0.0] * 6
    udrei = list(udrei) if udrei is not None else [0] * 6
    bits: list[int] = []
    for value in (prc + [0.0] * 6)[:6]:
        bits += _field(value, 0.125, 12, True)
    for value in (udrei + [0] * 6)[:6]:
        bits += _uint_bits(int(value) & 0xF, 4)
    bits += _uint_bits(iodp & 0x3, 2)
    bits += _uint_bits(block_id & 0x3, 2)
    bits += _uint_bits(iodf & 0x3, 2)
    bits += [0, 0, 0, 0]               # 4 spare
    # 106-bit long-term half: zero corrections, velocity code as requested.
    half = [int(velocity_code) & 1] + [0] * 105
    return _pad(bits + half)


def mt25_data(*, velocity_code1: int = 0, velocity_code2: int = 0) -> list[int]:
    """Pack the MT25 long-term data field (two zero-correction halves)."""
    half1 = [int(velocity_code1) & 1] + [0] * 105
    half2 = [int(velocity_code2) & 1] + [0] * 105
    return _pad(half1 + half2)


def mt26_data(*, band: int = 0, block_id: int = 0, iodi: int = 0,
              delay: Sequence[float] | None = None,
              givei: Sequence[int] | None = None) -> list[int]:
    """Pack the MT26 ionospheric data field (15 IGPs + 7 spare bits)."""
    delay = list(delay) if delay is not None else [0.0] * 15
    givei = list(givei) if givei is not None else [0] * 15
    bits = _uint_bits(band & 0xF, 4) + _uint_bits(block_id & 0xF, 4)
    for d, g in zip((delay + [0.0] * 15)[:15], (givei + [0] * 15)[:15]):
        bits += _field(d, 0.125, 9, False)
        bits += _uint_bits(int(g) & 0xF, 4)
    bits += _uint_bits(iodi & 0x3, 2)
    bits += [0] * 7
    return _pad(bits)


def mt1_data(*, mask=None, iodp: int = 0) -> list[int]:
    """Pack the MT1 PRN-mask data field (210 mask bits + 2-bit IODP)."""
    bits = [int(bool(m)) & 1 for m in mask] if mask is not None else [0] * 210
    bits = (bits + [0] * 210)[:210]
    bits += _uint_bits(iodp & 0x3, 2)
    return _pad(bits)


# ----------------------------------------------------------------------
# Broadcast schedule / stream
# ----------------------------------------------------------------------
def _geo_almanac(prn: int, sv_ecef: Sequence[float]) -> dict:
    return {"data_id": 0, "prn": int(prn), "health": 0,
            "position": tuple(float(v) for v in sv_ecef),
            "velocity": (0.0, 0.0, 0.0)}


def _make_by_type(msg_type: int, preamble: int, prn: int,
                  sv_ecef: Sequence[float], t0_sod: float) -> np.ndarray:
    if msg_type == 9:
        return make_mt9(t0_sod=t0_sod, position=tuple(sv_ecef),
                        preamble=preamble)
    if msg_type == 17:
        return make_mt17([_geo_almanac(prn, sv_ecef)], t0_sod=t0_sod,
                         preamble=preamble)
    if msg_type == 25:
        data = mt25_data()
    elif msg_type == 24:
        data = mt24_data()
    elif msg_type == 26:
        data = mt26_data()
    elif msg_type == 6:
        data = mt6_data()
    elif msg_type in (2, 3, 4, 5):
        data = mt2_data()
    elif msg_type == 1:
        data = mt1_data()
    elif msg_type == 63:
        data = [0] * SBAS_DATA_BITS
    else:
        data = [0] * SBAS_DATA_BITS
    return make_message(msg_type, data, preamble=preamble)


def sbas_bit_block(prn: int, sv_ecef: Sequence[float], *,
                   t0_sod: float = 0.0,
                   schedule: Sequence[int] = DEFAULT_SCHEDULE) -> np.ndarray:
    """Return a repeating ``int8`` bit block for one synthetic GEO.

    The schedule must contain a whole number of 3-message preamble cycles so
    that concatenating blocks preserves the 0x53/0x9A/0xC6 rotation.  Default
    length 6 also makes the block start re-align with the 6-second GPS subframe
    epoch every repetition.  ``t0_sod`` is the SBAS/GPS time-of-day written
    into MT9/MT17.
    """
    sched = tuple(int(m) & 0x3F for m in schedule)
    if not sched or len(sched) % 3:
        raise ValueError("SBAS schedule length must be a positive multiple of 3")
    msgs = [_make_by_type(mt, PREAMBLES[k % 3], prn, sv_ecef, t0_sod)
            for k, mt in enumerate(sched)]
    return np.concatenate(msgs).astype(np.int8)


def sbas_message_stream(duration_s: float = 6.0, *,
                        prn: int = 0, sv_ecef: Sequence[float] = (0.0, 0.0, 0.0),
                        t0_sod: float = 0.0,
                        schedule: Sequence[int] = DEFAULT_SCHEDULE
                        ) -> Iterator[np.ndarray]:
    """Yield one 250-bit ``int8`` message per second (250 bps) for a duration."""
    sched = tuple(int(m) & 0x3F for m in schedule)
    if not sched or len(sched) % 3:
        raise ValueError("SBAS schedule length must be a positive multiple of 3")
    for k in range(int(duration_s)):
        yield _make_by_type(sched[k % len(sched)], PREAMBLES[k % 3],
                            prn, sv_ecef, t0_sod)


def sbas_stream_bits(duration_s: float = 6.0, **kw) -> np.ndarray:
    """Concatenate :func:`sbas_message_stream` into a single bit array."""
    return np.concatenate(list(sbas_message_stream(duration_s, **kw))).astype(np.int8)


# ----------------------------------------------------------------------
# self-test
# ----------------------------------------------------------------------
#: Captured WAAS/EGNOS messages from the IGS ``geo_sbas.txt`` sample record.
_REAL_MESSAGES = (
    bytes([0x53, 0x08, 0x00, 0x50, 0x00, 0x00, 0x00, 0x01, 0x80, 0x00,
           0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0xFF,
           0x40, 0x01, 0x7B, 0x97, 0xBA, 0xFB, 0xBB, 0x97, 0x8B, 0xFB,
           0x54, 0x40]),
    bytes([0x9A, 0x07, 0xFF, 0xBB, 0x7F, 0xF8, 0x00, 0x00, 0x00, 0x00,
           0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
           0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3C, 0x94,
           0x43, 0xC0]),
    bytes([0xC6, 0x0C, 0x00, 0x00, 0x00, 0x00, 0x03, 0xFB, 0x40, 0x00,
           0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x38, 0x00,
           0x00, 0x03, 0xBB, 0x97, 0xBB, 0xA7, 0xB9, 0xFB, 0x83, 0x06,
           0x37, 0x40]),
)


def _bytes_to_bits(data: bytes) -> list[int]:
    return [(byte >> (7 - i)) & 1 for byte in data for i in range(8)]


def self_test() -> None:
    """Run the module's built-in consistency checks."""
    # 1) CRC against real captured SBAS messages (IGS geo_sbas.txt).
    for raw in _REAL_MESSAGES:
        bits = _bytes_to_bits(raw)
        parsed = parse_message(bits[:250])
        if not parsed["crc_ok"]:
            raise AssertionError("real SBAS message failed CRC")
        if crc24(bits[:226]) != parsed["parity"]:
            raise AssertionError("CRC does not reproduce broadcast parity")
        if crc24(bits[:250]) != 0:
            raise AssertionError("valid codeword must have zero remainder")

    # 2) framing / preamble / length.
    msg = make_message(63, [0] * SBAS_DATA_BITS)
    if msg.shape != (SBAS_MSG_BITS,) or msg.dtype != np.int8:
        raise AssertionError("message must be 250 int8 bits")
    if int(msg[0:8].dot(1 << np.arange(7, -1, -1))) != PREAMBLES[0]:
        raise AssertionError("preamble mismatch")
    if not parse_message(msg)["crc_ok"]:
        raise AssertionError("make_message produced a bad CRC")

    # 3) free-length padding and bipolar mapping.
    if not np.array_equal(bits_to_bipolar([0, 1]), np.array([1, -1], dtype=np.int8)):
        raise AssertionError("bipolar mapping mismatch")

    # 4) MT9 round-trip.
    pos = (42000000.0, -12000000.0, 1000.0)
    vel = (0.0, 0.0, 0.0)
    d9 = mt9_data(t0_sod=43216.0, ura=2, position=pos, velocity=vel,
                  af0=1e-9, af1=0.0)
    dec9 = mt9_decode(d9)
    if abs(dec9["xpos"] - pos[0]) > 2600.0 or abs(dec9["zpos"] - pos[2]) > 26000.0:
        raise AssertionError("MT9 position round-trip out of tolerance")

    # 5) MT17 round-trip.
    d17 = mt17_data([_geo_almanac(122, pos)], t0_sod=43200.0)
    dec17 = mt17_decode(d17)
    if dec17["almanacs"][0]["prn"] != 122:
        raise AssertionError("MT17 PRN round-trip mismatch")
    if abs(dec17["almanacs"][0]["x"] - 42000000.0) > 2600.0:
        raise AssertionError("MT17 X round-trip out of tolerance")
    if abs(dec17["t0_sod"] - 43200.0) > 64.0:
        raise AssertionError("MT17 t0 round-trip out of tolerance")

    # 6) block geometry: 6 messages = 6 s = 1500 bits at 250 bps.
    block = sbas_bit_block(122, pos, t0_sod=43200.0)
    if block.size != 6 * SBAS_MSG_BITS:
        raise AssertionError("unexpected block length")
    for k in range(6):
        m = block[k * 250:(k + 1) * 250]
        if not parse_message(m)["crc_ok"]:
            raise AssertionError(f"block message {k} CRC failed")
        if int(m[0:8].dot(1 << np.arange(7, -1, -1))) != PREAMBLES[k % 3]:
            raise AssertionError(f"block message {k} preamble mismatch")


if __name__ == "__main__":  # pragma: no cover
    self_test()
    print("sbas self-test OK")
