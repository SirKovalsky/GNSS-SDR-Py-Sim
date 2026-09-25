"""Galileo E1-B I/NAV navigation message construction (250 symbol/s).

This module builds a standards-compliant Galileo I/NAV bit stream from the
Galileo ephemeris already parsed into :class:`gnss_sim.rinex.Ephemeris`, so a
real Galileo receiver can acquire the E1-B code, synchronise on the I/NAV
synchronisation pattern, Viterbi-decode each page part, check the CRC and
finally compute a position/clock solution.

Verified structure (all cross-checked against an authoritative source)
----------------------------------------------------------------------
Source used for every detail below: **Galileo OS SIS ICD, Issue 1.3
(December 2016)** [1]_, and the decoder implementation of **GNSS-SDR**
(``galileo_telemetry_decoder_gs.cc`` / ``galileo_inav_message.{h,cc}``) [2]_,
which is known to decode live Galileo signals.

* I/NAV page: nominal page = 2 s = two page parts ("even" then "odd"), one
  second each.  Channel rate = 250 symbol/s, so a page part = 250 symbols and
  a page = 500 symbols.
* Page part layout (ICD table 34): 10 *unencoded* synchronisation symbols
  (``0101100000``) + 240 FEC symbols.  The 240 FEC symbols encode 114
  information bits + 6 zero tail bits (120 bits at rate 1/2).
* Page part field layout (ICD table 35), E1-B, nominal page::

      even:  even/odd(1)=0  page-type(1)=0  data_i(1/2)=112  tail(6) = 120
      odd :  even/odd(1)=1  page-type(1)=0  data_j(2/2)=16
             reserved1(40) SAR(22) spare(2) CRCj(24) reserved2(8) tail(6) = 120

  The two page parts together carry a 128-bit "nominal word"
  (``data_i(1/2)`` = 112 bits, ``data_j(2/2)`` = 16 bits) whose first 6 bits
  are the *word type* (1..10, 0 for the spare word, ...).
* FEC (ICD 4.1.4.1 / table 23): rate 1/2, constraint length 7, generator
  polynomials ``G1 = 171o``, ``G2 = 133o``, output sequence G1 then G2, with
  the G2 branch inverted at the end (figure 13).
* Interleaver (ICD 4.1.4.2 / table 24): block interleaver, I/NAV = 30 columns
  x 8 rows.  Data is written column-wise and read row-wise.  In index form,
  with ``c`` the column and ``r`` the row::

      interleaved[r*30 + c] = encoded[c*8 + r]

* CRC-24Q (ICD 5.1.9.4): the 24-bit CRC is the remainder of
  ``m(X)*X**24`` divided by ``G(X) = (1+X)*P(X)`` with
  ``P(X) = X^23+X^17+X^13+X^12+X^11+X^9+X^8+X^7+X^5+X^3+1``; MSB-first,
  initial remainder 0, no reflection, no final XOR.  The generator mask is
  therefore ``0x1864CFB`` (the same CRC-24Q used by SBAS/RTCA DO-229).
  The protected message is the 196 bits
  ``even[0:114] + odd[0:82]`` (even/odd flags, page types, data i/j,
  reserved1, SAR and spare); reserved2 and the tail bits are *not* covered.

References
----------
.. [1] The European GNSS (Galileo) Open Service Signal-In-Space Interface
   Control Document, Issue 1.3, December 2016.
   https://www.gsc-europa.eu/sites/default/files/Galileo_OS_SIS_ICD_1.3.pdf
.. [2] GNSS-SDR, ``src/core/system_parameters/galileo_inav_message.*`` and
   ``src/algorithms/telemetry_decoder/gnuradio_blocks/galileo_telemetry_decoder_gs.cc``.
   https://github.com/gnss-sdr/gnss-sdr

Bit/scaling of the ephemeris words (ICD 4.3.5; identical numbering in
GNSS-SDR's ``Galileo_INAV.h``): the ``lsb`` values are the ICD scale factors.
``_put`` writes fields MSB first at the 1-based positions quoted in the ICD;
position 1 is the MSB of the 128-bit word.

Assumptions / notes
-------------------
* Global bit-to-symbol polarity is immaterial for a receiver: it resolves the
  180 deg carrier-phase ambiguity, so the engine maps ``bit -> 1 - 2*bit``
  (0 -> +1), consistent with the other signals in this project.
* The Galileo ionospheric coefficients (word 5) cannot be derived from the
  GPS-style Klobuchar parameters held by :class:`IonoUtc`; they default to
  zero ("no correction") and can be overridden by the caller.
* Pages 7..10 (almanac) and 16+ (RedCED/RS/ISM) are not required to compute a
  position; the remaining pages of the 30 s sub-frame are filled with word
  type 0 (the I/NAV spare word).
* The BGD-E1/E5b parameter is not preserved by the RINEX parser (the second
  BGD is stored in ``Ephemeris.iodc`` and truncated); the E1/E5a value is
  broadcast for both BGD fields.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

# ----------------------------------------------------------------------
# Format constants
# ----------------------------------------------------------------------
INAV_RATE_BPS = 250                # channel symbols per second (E1-B)
CODE_PERIOD_S = 0.004              # one symbol per 4 ms E1-B code period
PAGE_SECONDS = 2                   # nominal page duration
PAGE_PART_SECONDS = 1
SUBBFRAME_SECONDS = 30             # I/NAV sub-frame (15 pages)
SUBBFRAME_PAGES = 15

PAGE_PART_SYMBOLS = 250            # sync (10) + FEC (240)
PAGE_SYMBOLS = 2 * PAGE_PART_SYMBOLS
SYNC_PATTERN = "0101100000"
SYNC_BITS = np.array([int(c) for c in SYNC_PATTERN], dtype=np.int8)
SYNC_LEN = len(SYNC_PATTERN)

PAGE_PART_INFO_BITS = 114
PAGE_PART_TAIL_BITS = 6
PAGE_PART_BITS = PAGE_PART_INFO_BITS + PAGE_PART_TAIL_BITS     # 120 (pre-FEC)
DATA_JK_BITS = 128                 # 112 (data i) + 16 (data j)
CRC_BITS = 24
CRC_FRAME_BITS = 196               # CRC-protected bits of the vertical page

INTERLEAVER_ROWS = 8
INTERLEAVER_COLS = 30
FEC_K = 7                          # constraint length
FEC_G1 = 0o171                     # ICD table 23
FEC_G2 = 0o133
FEC_RATE = 2

CRC24Q_POLY = 0x1864CFB            # CRC-24Q (same as SBAS DO-229)

#: Nominal 30 s E1-B sub-frame used by the engine.  Ephemeris/clock/iono/GST
#: (words 1..6) are broadcast in the first 12 s; the rest are spare words.
DEFAULT_SEQUENCE: tuple[int, ...] = (1, 2, 3, 4, 5, 6) + (0,) * 9

#: ICD scale factors (per LSB) for the encoded ephemeris words.
LSB = {
    "pi_2_31": np.pi * 2.0 ** -31,
    "e": 2.0 ** -33,
    "sqrtA": 2.0 ** -19,
    "i_dot": np.pi * 2.0 ** -43,
    "omega_dot": np.pi * 2.0 ** -43,
    "delta_n": np.pi * 2.0 ** -43,
    "cuc_cus_cic_cis": 2.0 ** -29,
    "crc_crs": 2.0 ** -5,
    "t0e": 60.0,
    "t0c": 60.0,
    "af0": 2.0 ** -34,
    "af1": 2.0 ** -46,
    "af2": 2.0 ** -59,
    "bgd": 2.0 ** -32,
    "ai0": 2.0 ** -2,
    "ai1": 2.0 ** -8,
    "ai2": 2.0 ** -15,
    "a0_utc": 2.0 ** -30,
    "a1_utc": 2.0 ** -50,
    "t0t": 3600.0,
}


# ----------------------------------------------------------------------
# CRC-24Q
# ----------------------------------------------------------------------
def crc24q(bits: Iterable[int]) -> int:
    """Return the 24-bit Galileo/SBAS CRC (CRC-24Q) of ``bits`` (MSB first).

    Polynomial division of ``m(X) * X**24`` by ``0x1864CFB`` over GF(2),
    initial remainder 0.  Feeding a full valid codeword returns 0.
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
# small bit helpers
# ----------------------------------------------------------------------
def _bits(value: int, width: int) -> list[int]:
    return [(int(value) >> (width - 1 - i)) & 1 for i in range(width)]


def _scale(value: float, lsb: float, width: int, signed: bool) -> int:
    raw = int(round(float(value) / lsb)) if lsb else int(round(float(value)))
    if signed:
        lo, hi = -(1 << (width - 1)), (1 << (width - 1)) - 1
    else:
        lo, hi = 0, (1 << width) - 1
    return max(lo, min(hi, raw))


def _put(word: list[int], pos: int, width: int, value: float, lsb: float = 1.0,
         signed: bool = False) -> None:
    """Write ``value`` into a 128-bit word at ICD bit position ``pos``."""
    raw = _scale(value, lsb, width, signed)
    raw &= (1 << width) - 1
    for i in range(width):
        word[pos - 1 + i] = (raw >> (width - 1 - i)) & 1


def _put_raw(word: list[int], pos: int, width: int, value: int) -> None:
    raw = int(value) & ((1 << width) - 1)
    for i in range(width):
        word[pos - 1 + i] = (raw >> (width - 1 - i)) & 1


def _read_raw(bits: Sequence[int], pos: int, width: int, signed: bool = False) -> int:
    value = 0
    for i in range(width):
        value = (value << 1) | (int(bits[pos - 1 + i]) & 1)
    if signed and (value >> (width - 1)) & 1:
        value -= 1 << width
    return value


def _read_scaled(bits: Sequence[int], pos: int, width: int, lsb: float,
                 signed: bool = False) -> float:
    return _read_raw(bits, pos, width, signed) * lsb


# ----------------------------------------------------------------------
# convolutional FEC (rate 1/2, K=7, G1=171o, G2=133o, G2 inverted)
# ----------------------------------------------------------------------
def _parity(x: int) -> int:
    return bin(x).count("1") & 1


def conv_encode(bits: Sequence[int]) -> list[int]:
    """Rate-1/2 K=7 convolutional encode; output is ``[G1, G2]`` per input bit.

    Matches the ICD Annex D.2 worked example exactly (see :func:`self_test`).
    ``bits`` should include the 6 zero tail bits.
    """
    reg = 0
    out: list[int] = []
    for bit in bits:
        reg = ((int(bit) & 1) << 6) | (reg >> 1)
        reg &= 0x7F
        out.append(_parity(reg & FEC_G1))
        out.append(_parity(reg & FEC_G2) ^ 1)     # G2 branch inverted
    return out


def _conv_step(state: int, bit: int) -> tuple[int, int, int]:
    reg = ((int(bit) & 1) << 6) | state
    g1 = _parity(reg & FEC_G1)
    g2 = _parity(reg & FEC_G2) ^ 1
    nxt = ((int(bit) & 1) << 5) | (state >> 1)
    return g1, g2, nxt & 0x3F


def viterbi_decode(coded: Sequence[float]) -> np.ndarray:
    """Hard-decision Viterbi decoder for the Galileo I/NAV convolutional code.

    ``coded`` is the 240-symbol (de-interleaved) sequence; positive values are
    treated as bit 1, non-positive as bit 0.  Returns the 120 decoded bits.
    """
    stream = [1 if v > 0 else 0 for v in coded]
    if len(stream) % 2:
        raise ValueError("coded sequence length must be even")
    inf = 1 << 30
    cost = [inf] * 64
    cost[0] = 0
    trace: list[list[tuple[int, int] | None]] = []
    for t in range(len(stream) // 2):
        ncost = [inf] * 64
        nprev: list[tuple[int, int] | None] = [None] * 64
        r1, r2 = stream[2 * t], stream[2 * t + 1]
        for state in range(64):
            if cost[state] >= inf:
                continue
            for bit in (0, 1):
                g1, g2, nxt = _conv_step(state, bit)
                c = cost[state] + (g1 != r1) + (g2 != r2)
                if c < ncost[nxt]:
                    ncost[nxt] = c
                    nprev[nxt] = (state, bit)
        trace.append(nprev)
        cost = ncost
    state = min(range(64), key=lambda s: cost[s])
    out: list[int] = []
    for t in range(len(trace) - 1, -1, -1):
        prev = trace[t][state]
        assert prev is not None
        state, bit = prev
        out.append(bit)
    out.reverse()
    return np.array(out, dtype=np.int8)


def interleave(symbols: Sequence[int]) -> np.ndarray:
    """I/NAV block interleaver: ``out[r*30+c] = in[c*8+r]``."""
    arr = np.asarray(symbols, dtype=np.int8)
    if arr.size != INTERLEAVER_ROWS * INTERLEAVER_COLS:
        raise ValueError(f"expected {INTERLEAVER_ROWS * INTERLEAVER_COLS} symbols")
    return arr.reshape(INTERLEAVER_COLS, INTERLEAVER_ROWS).T.reshape(-1)


def deinterleave(symbols: Sequence[int]) -> np.ndarray:
    """Inverse of :func:`interleave`."""
    arr = np.asarray(symbols, dtype=np.int8)
    if arr.size != INTERLEAVER_ROWS * INTERLEAVER_COLS:
        raise ValueError(f"expected {INTERLEAVER_ROWS * INTERLEAVER_COLS} symbols")
    return arr.reshape(INTERLEAVER_ROWS, INTERLEAVER_COLS).T.reshape(-1)


# ----------------------------------------------------------------------
# page / page-part framing
# ----------------------------------------------------------------------
def encode_page_part(info_bits: Sequence[int]) -> np.ndarray:
    """FEC-encode and interleave a 120-bit page part, prefix the sync pattern."""
    if len(info_bits) != PAGE_PART_BITS:
        raise ValueError(f"page part must be {PAGE_PART_BITS} bits")
    coded = conv_encode(info_bits)                # 240 symbols
    inter = interleave(coded)                     # 240 symbols
    return np.concatenate([SYNC_BITS, inter]).astype(np.int8)


def decode_page_part(symbols: Sequence[int]) -> tuple[np.ndarray, bool]:
    """Decode one 250-symbol page part into 120 information bits.

    Returns ``(bits, sync_ok)``.  The returned bits still contain the 6 tail
    bits (which are zero for a valid frame).
    """
    sym = np.asarray(symbols)
    if sym.size != PAGE_PART_SYMBOLS:
        raise ValueError(f"page part must be {PAGE_PART_SYMBOLS} symbols")
    sync_ok = bool(np.array_equal(sym[:SYNC_LEN].astype(np.int8), SYNC_BITS))
    coded = deinterleave(sym[SYNC_LEN:])
    return viterbi_decode(coded), sync_ok


def make_page(word_bits: Sequence[int], *, reserved1: Sequence[int] | None = None,
              sar: Sequence[int] | None = None,
              spare: Sequence[int] | None = None,
              reserved2: Sequence[int] | None = None) -> np.ndarray:
    """Assemble a 500-symbol nominal I/NAV page from a 128-bit word.

    The CRC is computed over the 196 protected bits and placed in the odd
    part.  ``reserved1`` (40 bits), ``sar`` (22), ``spare`` (2) and
    ``reserved2`` (8) default to all zeros.
    """
    word = [int(b) & 1 for b in word_bits]
    if len(word) != DATA_JK_BITS:
        raise ValueError(f"word must be {DATA_JK_BITS} bits")

    def fixed(seq: Sequence[int] | None, width: int) -> list[int]:
        if seq is None:
            return [0] * width
        out = [int(b) & 1 for b in seq][:width]
        return out + [0] * (width - len(out))

    res1 = fixed(reserved1, 40)
    sar_b = fixed(sar, 22)
    sp = fixed(spare, 2)
    res2 = fixed(reserved2, 8)

    data_i = word[:112]
    data_j = word[112:128]

    even = [0, 0] + data_i + [0] * PAGE_PART_TAIL_BITS
    odd = [1, 0] + data_j + res1 + sar_b + sp
    # CRC over even[0:114] + odd[0:82]
    message = even[:PAGE_PART_INFO_BITS] + odd[:82]
    parity = crc24q(message)
    odd += _bits(parity, CRC_BITS)
    odd += res2 + [0] * PAGE_PART_TAIL_BITS

    assert len(even) == PAGE_PART_BITS, len(even)
    assert len(odd) == PAGE_PART_BITS, len(odd)
    return np.concatenate([encode_page_part(even),
                           encode_page_part(odd)]).astype(np.int8)


def parse_page(page: Sequence[int]) -> dict:
    """Decode a 500-symbol nominal I/NAV page.

    Returns a dict with ``word_type``, ``word`` (the 128 raw bits),
    ``data_i``, ``data_j``, ``crc``, ``crc_ok``, ``sync_ok``, ``even`` and
    ``odd`` (the 120 decoded bits of each part).
    """
    sym = np.asarray(page)
    if sym.size != PAGE_SYMBOLS:
        raise ValueError(f"page must be {PAGE_SYMBOLS} symbols")
    even, sync_e = decode_page_part(sym[:PAGE_PART_SYMBOLS])
    odd, sync_o = decode_page_part(sym[PAGE_PART_SYMBOLS:])
    vertical = list(even[:PAGE_PART_INFO_BITS].tolist()) + list(odd[:PAGE_PART_INFO_BITS].tolist())
    message = vertical[:CRC_FRAME_BITS]
    parity = 0
    for bit in vertical[CRC_FRAME_BITS:CRC_FRAME_BITS + CRC_BITS]:
        parity = (parity << 1) | int(bit)
    data_k = vertical[2:114]
    data_j = vertical[116:132]
    word = data_k + data_j
    word_type = 0
    for bit in word[:6]:
        word_type = (word_type << 1) | int(bit)
    return {
        "word_type": int(word_type),
        "word": word,
        "data_i": data_k,
        "data_j": data_j,
        "even": list(even),
        "odd": list(odd),
        "crc": int(parity),
        "crc_ok": bool(crc24q(message) == parity),
        "sync_ok": bool(sync_e and sync_o),
    }


# ----------------------------------------------------------------------
# word builders (fields per ICD 4.3.5)
# ----------------------------------------------------------------------
def word_spare(gst_week: int = 0, gst_tow: float = 0.0) -> np.ndarray:
    """Word type 0 - I/NAV spare word (time/week reference only)."""
    w = [0] * DATA_JK_BITS
    _put_raw(w, 1, 6, 0)                             # word type 0
    _put_raw(w, 7, 2, 0)                             # time_0
    _put_raw(w, 97, 12, int(gst_week) % 4096)        # WN_0
    _put_raw(w, 109, 20, int(gst_tow) % 604800)      # TOW_0
    return np.array(w, dtype=np.int8)


def word_1(eph) -> np.ndarray:
    """Word type 1 - ephemeris 1/4 (IODnav, t0e, M0, e, sqrtA)."""
    w = [0] * DATA_JK_BITS
    _put_raw(w, 1, 6, 1)
    _put_raw(w, 7, 10, _iodnav(eph))
    _put(w, 17, 14, float(eph.toe.sec) % 604800.0, LSB["t0e"], signed=False)
    _put(w, 31, 32, eph.m0, LSB["pi_2_31"], signed=True)
    _put(w, 63, 32, eph.ecc, LSB["e"], signed=False)
    _put(w, 95, 32, eph.sqrta, LSB["sqrtA"], signed=False)
    return np.array(w, dtype=np.int8)


def word_2(eph) -> np.ndarray:
    """Word type 2 - ephemeris 2/4 (Omega0, i0, omega, iDot)."""
    w = [0] * DATA_JK_BITS
    _put_raw(w, 1, 6, 2)
    _put_raw(w, 7, 10, _iodnav(eph))
    _put(w, 17, 32, eph.omg0, LSB["pi_2_31"], signed=True)
    _put(w, 49, 32, eph.inc0, LSB["pi_2_31"], signed=True)
    _put(w, 81, 32, eph.aop, LSB["pi_2_31"], signed=True)
    _put(w, 113, 14, eph.idot, LSB["i_dot"], signed=True)
    return np.array(w, dtype=np.int8)


def word_3(eph, sisa: int = 0) -> np.ndarray:
    """Word type 3 - ephemeris 3/4 + SISA."""
    w = [0] * DATA_JK_BITS
    _put_raw(w, 1, 6, 3)
    _put_raw(w, 7, 10, _iodnav(eph))
    _put(w, 17, 24, eph.omgdot, LSB["omega_dot"], signed=True)
    _put(w, 41, 16, eph.deltan, LSB["delta_n"], signed=True)
    _put(w, 57, 16, eph.cuc, LSB["cuc_cus_cic_cis"], signed=True)
    _put(w, 73, 16, eph.cus, LSB["cuc_cus_cic_cis"], signed=True)
    _put(w, 89, 16, eph.crc, LSB["crc_crs"], signed=True)
    _put(w, 105, 16, eph.crs, LSB["crc_crs"], signed=True)
    _put_raw(w, 121, 8, int(sisa) & 0xFF)
    return np.array(w, dtype=np.int8)


def word_4(eph, sv_id: int | None = None) -> np.ndarray:
    """Word type 4 - ephemeris 4/4 and clock correction parameters."""
    w = [0] * DATA_JK_BITS
    _put_raw(w, 1, 6, 4)
    _put_raw(w, 7, 10, _iodnav(eph))
    _put_raw(w, 17, 6, int(eph.prn if sv_id is None else sv_id) & 0x3F)
    _put(w, 23, 16, eph.cic, LSB["cuc_cus_cic_cis"], signed=True)
    _put(w, 39, 16, eph.cis, LSB["cuc_cus_cic_cis"], signed=True)
    _put(w, 55, 14, float(eph.toc.sec) % 604800.0, LSB["t0c"], signed=False)
    _put(w, 69, 31, eph.af0, LSB["af0"], signed=True)
    _put(w, 100, 21, eph.af1, LSB["af1"], signed=True)
    _put(w, 121, 6, eph.af2, LSB["af2"], signed=True)
    return np.array(w, dtype=np.int8)


def word_5(eph, gst_week: int, gst_tow: float, *,
           ai0: float = 0.0, ai1: float = 0.0, ai2: float = 0.0,
           bgd_e5a: float | None = None, bgd_e5b: float | None = None,
           health_e1b: int = 0, health_e5b: int = 0) -> np.ndarray:
    """Word type 5 - iono, BGD, signal health/data validity and GST."""
    w = [0] * DATA_JK_BITS
    _put_raw(w, 1, 6, 5)
    _put(w, 7, 11, ai0, LSB["ai0"], signed=False)
    _put(w, 18, 11, ai1, LSB["ai1"], signed=True)
    _put(w, 29, 14, ai2, LSB["ai2"], signed=True)
    # region disturbance flags 43..47: all zero (no disturbance)
    bgd = eph.tgd if bgd_e5a is None else bgd_e5a
    bgd_b = eph.tgd if bgd_e5b is None else bgd_e5b
    _put(w, 48, 10, bgd, LSB["bgd"], signed=True)
    _put(w, 58, 10, bgd_b, LSB["bgd"], signed=True)
    _put_raw(w, 68, 2, int(health_e5b) & 0x3)
    _put_raw(w, 70, 2, int(health_e1b) & 0x3)
    # data validity status 72, 73: zero (valid)
    _put_raw(w, 74, 12, int(gst_week) % 4096)
    _put_raw(w, 86, 20, int(gst_tow) % 604800)
    return np.array(w, dtype=np.int8)


def word_6(gst_tow: float, *, a0: float = 0.0, a1: float = 0.0,
           dtls: int = 18, t0t_hours: int = 0, wnot: int = 0,
           wn_lsf: int = 1929, dn: int = 7, dtlsf: int = 18) -> np.ndarray:
    """Word type 6 - GST-UTC conversion parameters."""
    w = [0] * DATA_JK_BITS
    _put_raw(w, 1, 6, 6)
    _put(w, 7, 32, a0, LSB["a0_utc"], signed=True)
    _put(w, 39, 24, a1, LSB["a1_utc"], signed=True)
    _put_raw(w, 63, 8, int(dtls) & 0xFF)
    _put_raw(w, 71, 8, int(t0t_hours) & 0xFF)
    _put_raw(w, 79, 8, int(wnot) & 0xFF)
    _put_raw(w, 87, 8, int(wn_lsf) % 256)
    _put_raw(w, 95, 3, int(dn) & 0x7)
    _put_raw(w, 98, 8, int(dtlsf) & 0xFF)
    _put_raw(w, 106, 20, int(gst_tow) % 604800)
    return np.array(w, dtype=np.int8)


def _iodnav(eph) -> int:
    """Galileo IODnav from the parsed ephemeris (``Ephemeris.iode``)."""
    return int(eph.iode) & 0x3FF


def _health(eph) -> int:
    return 0 if int(getattr(eph, "svhlth", 0)) == 0 else 1


# ----------------------------------------------------------------------
# sub-frame / bit block
# ----------------------------------------------------------------------
def inav_subframe(eph, *, gst_week: int, gst_tow: float,
                  sequence: Sequence[int] = DEFAULT_SEQUENCE,
                  iono: tuple[float, float, float] = (0.0, 0.0, 0.0),
                  utc: dict | None = None,
                  reserve: Sequence[int] | None = None) -> np.ndarray:
    """Return one 30 s (7500-symbol) I/NAV sub-frame for ``eph``.

    Words 1..6 carry the broadcast ephemeris/clock, the ionosphere/BGD/GST
    model and the UTC model.  Other page types in ``sequence`` become the
    spare word (type 0).  The result is a repeating ``int8`` 0/1 bit block at
    250 symbol/s for :class:`gnss_sim.engine.SignalEngine`.
    """
    builders = {
        0: lambda: word_spare(gst_week, gst_tow),
        1: lambda: word_1(eph),
        2: lambda: word_2(eph),
        3: lambda: word_3(eph),
        4: lambda: word_4(eph),
        5: lambda: word_5(eph, gst_week, gst_tow, ai0=iono[0], ai1=iono[1],
                          ai2=iono[2], health_e1b=_health(eph),
                          health_e5b=_health(eph)),
        6: lambda: word_6(gst_tow, **(utc or {})),
    }
    pages = []
    for word_type in sequence:
        if word_type not in builders:
            word = word_spare(gst_week, gst_tow)
        else:
            word = builders[word_type]()
        pages.append(make_page(word, spare=[0, 0]))
    return np.concatenate(pages).astype(np.int8)


def inav_bit_block(eph, *, gst_week: int, gst_tow: float, **kw) -> np.ndarray:
    """Alias of :func:`inav_subframe` (engine-friendly name)."""
    return inav_subframe(eph, gst_week=gst_week, gst_tow=gst_tow, **kw)


bit_block = inav_bit_block


# ----------------------------------------------------------------------
# self-test
# ----------------------------------------------------------------------
#: ICD Annex D.2 I/NAV FEC + interleaver numerical example.
_ICD_INPUT = (
    "11111111 11110000 11001100 10101010 00000000 00001111 00110011 01010101 "
    "11100011 11101100 11011111 10001010 00011100 00010011 01000000")
_ICD_ENCODED = (
    "10001100 00011010 10101010 01110011 00110001 01011010 01101111 01011001 "
    "01111000 10010101 01010101 10001100 11001110 10100101 10010000 10100110 "
    "10000100 00000010 00000010 00011011 10011001 11101011 01011100 00011000 "
    "10111011 11111101 11111101 11100100 01010011 00100010")
_ICD_INTERLEAVED = (
    "10100000 01011111 10001100 11110000 01011110 10100000 00011001 11100011 "
    "10101000 01010000 01001111 01010111 01111000 10000110 11111010 11100111 "
    "10011000 00011111 11100010 00001001 11110110 00001001 11000111 01100000 "
    "10010111 01001000 11000110 11011001 00000111 00111010")


def _string_bits(text: str) -> list[int]:
    return [int(c) for c in text.replace(" ", "")]


def self_test() -> None:
    """Run built-in consistency checks against the ICD numerical example."""
    # 1) FEC + interleaver reproduce the ICD Annex D.2 example bit-for-bit.
    inp = _string_bits(_ICD_INPUT)
    encoded = conv_encode(inp)
    if encoded != _string_bits(_ICD_ENCODED):
        raise AssertionError("FEC does not match ICD Annex D.2 encoded example")
    inter = interleave(encoded).tolist()
    if inter != _string_bits(_ICD_INTERLEAVED):
        raise AssertionError("interleaver does not match ICD Annex D.2 example")
    decoded = viterbi_decode(encoded).tolist()
    if decoded != inp:
        raise AssertionError("Viterbi decoder failed to invert the encoder")

    # 2) CRC-24Q agrees with the (independently validated) SBAS implementation.
    try:
        from .sbas import crc24 as sbas_crc24
    except Exception:  # pragma: no cover
        sbas_crc24 = None
    if sbas_crc24 is not None:
        rng = np.random.default_rng(1234)
        for _ in range(8):
            bits = rng.integers(0, 2, size=196).tolist()
            if crc24q(bits) != sbas_crc24(bits):
                raise AssertionError("CRC-24Q disagrees with SBAS CRC-24Q")

    # 3) CRC of a valid codeword has zero remainder and detects 1-bit errors.
    msg = [int(b) for b in np.random.default_rng(7).integers(0, 2, size=196)]
    parity = crc24q(msg)
    if crc24q(msg + _bits(parity, 24)) != 0:
        raise AssertionError("valid codeword must have zero CRC remainder")
    bad = list(msg)
    bad[17] ^= 1
    if crc24q(bad) == parity:
        raise AssertionError("CRC failed to detect a single-bit error")

    # 4) Full page round-trip: build a page from a word, decode it back.
    word = word_1(_synthetic_ephemeris())
    page = make_page(word)
    if page.shape != (PAGE_SYMBOLS,):
        raise AssertionError("page must be 500 symbols")
    parsed = parse_page(page)
    if not parsed["sync_ok"] or not parsed["crc_ok"]:
        raise AssertionError("page did not round-trip through sync/CRC")
    if parsed["word_type"] != 1:
        raise AssertionError("word type mismatch after round-trip")
    if parsed["word"] != [int(b) for b in word]:
        raise AssertionError("128-bit word mismatch after round-trip")

    # 5) Sub-frame geometry: 15 pages = 30 s = 7500 symbols at 250 bps.
    block = inav_subframe(_synthetic_ephemeris(), gst_week=2200,
                          gst_tow=432000.0)
    if block.shape != (SUBBFRAME_SECONDS * INAV_RATE_BPS,):
        raise AssertionError("unexpected sub-frame length")
    if set(np.unique(block).tolist()) != {0, 1}:
        raise AssertionError("I/NAV block is not a 0/1 bit block")


def _synthetic_ephemeris():
    """A representative Galileo ephemeris for the self-test."""
    from .gpstime import GpsTime
    from .rinex import Ephemeris

    eph = Ephemeris()
    eph.system = "E"
    eph.prn = 11
    eph.toc = GpsTime(2200, 432000.0)
    eph.toe = GpsTime(2200, 432000.0)
    eph.m0 = 0.5
    eph.deltan = 4.5e-9
    eph.ecc = 1.2e-4
    eph.sqrta = 5440.0
    eph.omg0 = -1.3
    eph.inc0 = 0.97
    eph.aop = 2.1
    eph.omgdot = -5.3e-9
    eph.idot = 1.1e-10
    eph.cuc = 4.0e-7
    eph.cus = 7.3e-6
    eph.cic = -1.2e-8
    eph.cis = -4.0e-8
    eph.crc = 180.0
    eph.crs = -40.0
    eph.af0 = -4.6e-4
    eph.af1 = 3.2e-11
    eph.af2 = 0.0
    eph.tgd = -4.7e-9
    eph.iode = 42
    eph.svhlth = 0
    eph.ura = 2
    return eph


if __name__ == "__main__":  # pragma: no cover
    self_test()
    print("galileo_nav self-test OK")
