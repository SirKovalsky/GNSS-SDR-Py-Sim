"""GPS L1C CNAV-2 navigation message construction (100 symbol/s).

This module builds one complete 18-second CNAV-2 frame (1800 symbols) from the
broadcast ephemeris already parsed into :class:`gnss_sim.rinex.Ephemeris`, so a
real receiver can acquire the L1C D channel, synchronise, LDPC-decode the
subframes, check the CRC and use the ephemeris/clock data.

Structure (IS-GPS-800C, 31 Jan 2013)
------------------------------------
* **Subframe 1** - 9-bit TOI (time of interval) BCH(51,8) encoded into 52
  symbols.  The 8 LSBs of TOI are encoded with the 8-stage LFSR generator
  polynomial 763 octal (``1 + X + X**4 + X**5 + X**6 + X**7 + X**8``); the TOI
  MSB is modulo-2 added to all 51 generator outputs and prepended as the first
  symbol (Section 3.2.3.2, Figure 3.2-4).
* **Subframe 2** - 576 data bits (message type 10: clock, ephemeris, ITOW) +
  24-bit CRC-24Q = 600 bits, LDPC(1200,600) encoded (Section 3.5.3, Figure
  3.5-1).
* **Subframe 3** - 250 bits (PRN, page number and page data) + 24-bit CRC-24Q =
  274 bits, LDPC(548,274) encoded (Section 3.5.4).  Page 3 (reduced almanac)
  and page 6 (text) are implemented; other pages are emitted as reserved.
* The 1748 LDPC symbols are block-interleaved on a 38 x 46 array (write
  row-wise, read column-wise, Section 3.2.3.5) and appended to the 52 BCH
  symbols to give the 1800-symbol frame.

CRC-24Q is the polynomial ``0x1864CFB`` (``(1+X)*P(X)``, ``P`` primitive),
seed 0, MSB first, no reflection - the same convention already implemented in
:mod:`gnss_sim.galileo_nav` and :mod:`gnss_sim.sbas`.

LDPC
----
Submatrices A, B, T, C, D, E of the parity-check matrix ``H = [[A B T],
[C D E]]`` are taken verbatim from the (row, column) coordinate tables of
IS-GPS-800C Section 6.2.4 (Tables 6.2-2 .. 6.2-13, stored in
:mod:`gnss_sim.cnav2_ldpc`).  For subframe 2 ``H`` is 600 x 1200 with
A(599x600), B(599x1), T(599x599), C(1x600), D(1x1), E(1x599); subframe 3 uses
273/274.  ``T`` is lower triangular with unit diagonal and ``phi =
E*T**-1*B + D = 1``, so the Section 3.2.3.4 encoding algorithm is applied
directly::

    p1 = phi**-1 * (E*T**-1*A + C) * s
    p2 = T**-1 * (A*s + B*p1)

The resulting systematic parity matrix was verified (bit for bit) against the
public SignalSim / Enhanced-Chimera generator matrices, and the BCH encoder was
verified against the 256-entry public TOI table for all inputs.

Assumptions / limitations
-------------------------
* Fields absent from the RINEX record (A-dot, delta-n-dot, URA NED1/NED2 and
  the inter-signal corrections) are broadcast as zero; the RINEX URA index is
  reused for the URA_ED and URA_NED0 fields.
* The reduced-almanac page (3) broadcasts the transmitting SV's own reduced
  almanac in packet 1 and filler in the remaining packets; the text page (6)
  broadcasts a fixed short ASCII string.
* Quality of the real-time upload is immaterial for a signal simulator; only
  the bit-for-bit wire format matters.
"""

from __future__ import annotations

import numpy as np

from .cnav2_ldpc import COORDS, DIMS
from .galileo_nav import crc24q
from .rinex import Ephemeris

# ----------------------------------------------------------------------
# Format constants
# ----------------------------------------------------------------------
FRAME_SYMBOLS = 1800               # one 18 s frame at 100 symbols/s
FRAME_SECONDS = 18
SYMBOL_RATE = 100

BCH_SYMBOLS = 52                   # subframe 1
BCH_DATA_BITS = 8                  # BCH(51,8) payload (TOI LSBs)
BCH_POLY = 0o763                   # generator polynomial (octal)

SF2_DATA_BITS = 576                # message type 10 payload
SF2_BITS = 600                     # 576 data + 24 CRC
SF3_DATA_BITS = 250                # page payload
SF3_BITS = 274                     # 250 data + 24 CRC
CRC_BITS = 24

INTERLEAVER_ROWS = 38
INTERLEAVER_COLS = 46
INTERLEAVED_SYMBOLS = INTERLEAVER_ROWS * INTERLEAVER_COLS   # 1748

AREF = 26559710.0                  # reference semi-major axis, metres
OMEGA_DOT_REF = -2.6e-9            # semi-circles/s (Delta-Omega reference)

#: IS-GPS-800C section 3.5.4 page numbers used by the engine.  Real SVs cycle
#: subframe 3 pages in a variable pattern; the simulator alternates the two
#: implemented pages deterministically.
SF3_PAGES = (3, 6)                 # reduced almanac, text

TEXT_PAGE_CHARS = 29
TEXT_PAGE_TEXT = "HELLO FROM GNSS SIM L1C CNAV "[:TEXT_PAGE_CHARS].ljust(
    TEXT_PAGE_CHARS)

_PI = np.pi
_POW = {k: 2.0 ** k for k in
        (-60, -57, -48, -44, -35, -34, -32, -30, -21, -9, -8, -6)}

# ----------------------------------------------------------------------
# CRC-24Q (shared with the Galileo/SBAS modules)
# ----------------------------------------------------------------------
crc24 = crc24q                       # engine-friendly alias


# ----------------------------------------------------------------------
# small bit helpers (1-based bit positions, MSB first)
# ----------------------------------------------------------------------
def _uint_bits(value: int, width: int) -> list[int]:
    return [(int(value) >> (width - 1 - i)) & 1 for i in range(width)]


def _bits_to_uint(bits) -> int:
    value = 0
    for b in bits:
        value = (value << 1) | (int(b) & 1)
    return value


def _bits_to_int(bits) -> int:
    value = _bits_to_uint(bits)
    if len(bits) and int(bits[0]) & 1:
        value -= 1 << len(bits)
    return value


def _scale(value: float, scale: float, width: int, signed: bool) -> int:
    raw = int(round(float(value) / scale)) if scale else int(round(float(value)))
    if signed:
        lo, hi = -(1 << (width - 1)), (1 << (width - 1)) - 1
    else:
        lo, hi = 0, (1 << width) - 1
    return max(lo, min(hi, raw)) & ((1 << width) - 1)


def _put(bits: list[int], pos: int, width: int, value, scale: float = 1.0,
         signed: bool = False) -> None:
    """Write ``value`` at 1-based ``pos``, MSB first (ICD convention)."""
    raw = _scale(value, scale, width, signed)
    for i in range(width):
        bits[pos - 1 + i] = (raw >> (width - 1 - i)) & 1


def _read(bits, pos: int, width: int, signed: bool = False) -> int:
    return _bits_to_int(bits[pos - 1:pos - 1 + width]) if signed else \
        _bits_to_uint(bits[pos - 1:pos - 1 + width])


def _read_scaled(bits, pos: int, width: int, scale: float,
                 signed: bool = False) -> float:
    return _read(bits, pos, width, signed) * scale


# ----------------------------------------------------------------------
# BCH(51,8) TOI encoding (IS-GPS-800C section 3.2.3.2)
# ----------------------------------------------------------------------
def bch51(lsb8: int) -> list[int]:
    """Encode the 8 LSBs of TOI into the 51 generator output symbols.

    The 8-stage Fibonacci LFSR matches Figure 3.2-4: stages 1..8 hold TOI bits
    1..8, each clock emits the stage-8 value and feeds back
    ``s1 ^ s4 ^ s5 ^ s6 ^ s7 ^ s8``.
    """
    v = int(lsb8) & 0xFF
    s = [(v >> i) & 1 for i in range(8)]          # stage 1 = LSB
    out = [0] * 51
    for k in range(51):
        out[k] = s[7]
        fb = s[0] ^ s[3] ^ s[4] ^ s[5] ^ s[6] ^ s[7]
        s = [fb] + s[:7]
    return out


def bch_encode(toi: int) -> np.ndarray:
    """Return the 52-symbol BCH-protected TOI word (MSB of TOI first)."""
    toi = int(toi) & 0x1FF
    msb = (toi >> 8) & 1
    code = bch51(toi & 0xFF)
    out = np.empty(BCH_SYMBOLS, dtype=np.int8)
    out[0] = msb
    for k, bit in enumerate(code):
        out[1 + k] = bit ^ msb
    return out


def bch_decode(symbols) -> int:
    """Decode a 52-symbol TOI word by maximum-likelihood correlation."""
    sym = np.asarray(symbols, dtype=np.int8)
    if sym.size != BCH_SYMBOLS:
        raise ValueError(f"TOI word must be {BCH_SYMBOLS} symbols")
    best = None
    best_score = None
    for msb in (0, 1):
        for lsb in range(256):
            word = bch_encode((msb << 8) | lsb)
            score = int(np.count_nonzero(word == sym))
            if best_score is None or score > best_score:
                best_score, best = score, (msb << 8) | lsb
    return int(best)


# ----------------------------------------------------------------------
# TOI / time helpers
# ----------------------------------------------------------------------
def page_index(frame_start) -> int:
    """18-second epoch index within the GPS week for ``frame_start``."""
    return int(frame_start.sec // FRAME_SECONDS)


def itow_count(frame_start) -> int:
    """Number of two-hour epochs since the start of the week (0..83)."""
    return (page_index(frame_start) // 400) & 0xFF


def toi_count(frame_start) -> int:
    """9-bit TOI for the frame starting at ``frame_start``.

    Per IS-GPS-800C section 3.5.2 the TOI identifies the epoch at the start of
    the *next* 18-second frame; TOI 1 follows the two-hour epoch boundary.
    """
    return (page_index(frame_start) % 400 + 1) % 400


# ----------------------------------------------------------------------
# LDPC(1200,600) / LDPC(548,274)  (IS-GPS-800C section 6.2.4)
# ----------------------------------------------------------------------
def _matrix(name: str) -> np.ndarray:
    rows, cols = DIMS[name]
    m = np.zeros((rows, cols), dtype=np.uint8)
    for r, c in COORDS[name]:
        m[r - 1, c - 1] ^= 1
    return m


def _solve_lower(t: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve ``t x = b`` for lower-triangular unit-diagonal ``t`` over GF(2)."""
    m = t.shape[0]
    single = b.ndim == 1
    if single:
        b = b[:, None]
    x = np.zeros_like(b)
    for i in range(m):
        acc = b[i].copy()
        for j in np.nonzero(t[i, :i])[0]:
            acc ^= x[j]
        x[i] = acc
    return x[:, 0] if single else x


_LDPC_CACHE: dict[int, tuple] = {}


def _ldpc_pieces(subframe: int):
    """Return ``(A, B, C, D, E, T, P)`` for ``subframe`` (2 or 3)."""
    hit = _LDPC_CACHE.get(subframe)
    if hit is not None:
        return hit
    if subframe == 2:
        keys = ("A2", "B2", "C2", "D2", "E2", "T2")
    elif subframe == 3:
        keys = ("A3", "B3", "C3", "D3", "E3", "T3")
    else:
        raise ValueError("subframe must be 2 or 3")
    a, b, c, d, e, t = (_matrix(k) for k in keys)
    # phi = E*T^-1*B + D is a scalar; T is unit lower triangular.
    ya = _solve_lower(t, a)                       # T^-1 A   (m1 x k)
    yb = _solve_lower(t, b[:, 0])                 # T^-1 B   (m1,)
    phi = int((e @ yb)[0] & 1) ^ int(d[0, 0])
    if phi != 1:
        raise NotImplementedError("unexpected non-invertible phi in LDPC H")
    p1 = ((e @ ya) % 2) ^ c                        # (1 x k)
    p1f = p1[0]
    p2 = (ya ^ (yb[:, None] * p1f[None, :])) % 2
    parity = np.vstack([p1, p2]).astype(np.uint8)  # (k x k) systematic parity
    _LDPC_CACHE[subframe] = (a, b, c, d, e, t, parity)
    return _LDPC_CACHE[subframe]


def ldpc_encode(data, subframe: int) -> np.ndarray:
    """Rate-1/2 LDPC encode a 600-bit (SF2) or 274-bit (SF3) data block."""
    d = np.asarray(data, dtype=np.uint8).reshape(-1)
    k = SF2_BITS if subframe == 2 else SF3_BITS
    if d.size != k:
        raise ValueError(f"subframe {subframe} data must be {k} bits")
    parity = _ldpc_pieces(subframe)[6]
    par = (parity @ d.astype(np.int64)) & 1
    return np.concatenate([d, par]).astype(np.int8)


def ldpc_parity_only(data, subframe: int) -> np.ndarray:
    """Return just the k parity symbols produced by :func:`ldpc_encode`."""
    n = SF2_BITS if subframe == 2 else SF3_BITS
    return ldpc_encode(data, subframe)[n:]


def ldpc_check(codeword, subframe: int) -> np.ndarray:
    """Return the syndrome ``H * codeword`` over GF(2) (all zeros if valid)."""
    if subframe == 2:
        a, b, c, d, e, t = _ldpc_pieces(2)[:6]
    else:
        a, b, c, d, e, t = _ldpc_pieces(3)[:6]
    h = np.vstack([np.hstack([a, b, t]), np.hstack([c, d, e])]).astype(np.uint8)
    x = np.asarray(codeword, dtype=np.uint8).reshape(-1)
    return (h @ x.astype(np.int64)) & 1


# ----------------------------------------------------------------------
# block interleaver (IS-GPS-800C section 3.2.3.5)
# ----------------------------------------------------------------------
def interleave(symbols) -> np.ndarray:
    """38 x 46 block interleaver: write row-wise, read column-wise.

    ``out[c*38 + r] = in[r*46 + c]``.
    """
    arr = np.asarray(symbols, dtype=np.int8)
    if arr.size != INTERLEAVED_SYMBOLS:
        raise ValueError(f"expected {INTERLEAVED_SYMBOLS} symbols")
    return arr.reshape(INTERLEAVER_ROWS, INTERLEAVER_COLS).T.reshape(-1)


def deinterleave(symbols) -> np.ndarray:
    """Inverse of :func:`interleave`."""
    arr = np.asarray(symbols, dtype=np.int8)
    if arr.size != INTERLEAVED_SYMBOLS:
        raise ValueError(f"expected {INTERLEAVED_SYMBOLS} symbols")
    return arr.reshape(INTERLEAVER_COLS, INTERLEAVER_ROWS).T.reshape(-1)


# ----------------------------------------------------------------------
# subframe 2 - message type 10 (clock / ephemeris / ITOW)
# ----------------------------------------------------------------------
def subframe2_bits(eph: Ephemeris, frame_start) -> np.ndarray:
    """Assemble the 600-bit subframe 2 (576 data bits + CRC-24Q)."""
    b = [0] * SF2_DATA_BITS
    week = int(frame_start.week) & 0x1FFF
    itow = itow_count(frame_start)
    top = int(round(float(eph.toe.sec) / 300.0)) & 0x7FF
    toe = top

    _put(b, 1, 13, week)                                   # WN
    _put(b, 14, 8, itow)                                   # ITOW
    _put(b, 22, 11, top)                                   # t_op
    _put(b, 33, 1, 1 if int(eph.svhlth) else 0)            # L1C health
    _put(b, 34, 5, int(eph.ura), signed=True)              # URA_ED index
    _put(b, 39, 11, toe)                                   # t_oe
    _put(b, 50, 26, eph.sqrta ** 2 - AREF, _POW[-9], True)  # Delta A
    _put(b, 76, 25, 0.0, _POW[-21], True)                  # A-dot
    _put(b, 101, 17, eph.deltan / _PI, _POW[-44], True)    # Delta n0
    _put(b, 118, 23, 0.0, _POW[-57], True)                 # Delta n0-dot
    _put(b, 141, 33, eph.m0 / _PI, _POW[-32], True)        # M0
    _put(b, 174, 33, eph.ecc, _POW[-34], False)            # e
    _put(b, 207, 33, eph.aop / _PI, _POW[-32], True)       # omega
    _put(b, 240, 33, eph.omg0 / _PI, _POW[-32], True)      # Omega0
    _put(b, 273, 33, eph.inc0 / _PI, _POW[-32], True)      # i0
    _put(b, 306, 17, eph.omgdot / _PI - OMEGA_DOT_REF, _POW[-44], True)
    _put(b, 323, 15, eph.idot / _PI, _POW[-44], True)      # i0-dot
    _put(b, 338, 16, eph.cis, _POW[-30], True)             # C_is
    _put(b, 354, 16, eph.cic, _POW[-30], True)             # C_ic
    _put(b, 370, 24, eph.crs, _POW[-8], True)              # C_rs
    _put(b, 394, 24, eph.crc, _POW[-8], True)              # C_rc
    _put(b, 418, 21, eph.cus, _POW[-30], True)             # C_us
    _put(b, 439, 21, eph.cuc, _POW[-30], True)             # C_uc
    _put(b, 460, 5, int(eph.ura), signed=True)             # URA_NED0 index
    _put(b, 465, 3, 0)                                     # URA_NED1 index
    _put(b, 468, 3, 0)                                     # URA_NED2 index
    _put(b, 471, 26, eph.af0, _POW[-35], True)             # a_f0
    _put(b, 497, 20, eph.af1, _POW[-48], True)             # a_f1
    _put(b, 517, 10, eph.af2, _POW[-60], True)             # a_f2
    _put(b, 527, 13, eph.tgd, _POW[-35], True)             # T_GD
    _put(b, 540, 13, 0.0, _POW[-35], True)                 # ISC_L1CP
    _put(b, 553, 13, 0.0, _POW[-35], True)                 # ISC_L1CD
    # bit 566 integrity status flag, bits 567..576 reserved - left zero
    return np.array(b + _uint_bits(crc24q(b), CRC_BITS), dtype=np.int8)


# ----------------------------------------------------------------------
# subframe 3 - page payloads
# ----------------------------------------------------------------------
def _fill_filler(bits: list[int], start: int) -> None:
    """ICD 3.5.4.3.5.1.1 filler: alternating 1,0 beginning with 1."""
    bit = 1
    for i in range(start, len(bits)):
        bits[i] = bit
        bit ^= 1


def reduced_almanac_packet(prn: int, a: float, omg0: float, phi0: float,
                           health: int = 0) -> list[int]:
    """One 33-bit reduced-almanac packet (IS-GPS-800C Figure 3.5-9/Table 3.5-6)."""
    p = [0] * 33
    _put(p, 1, 8, int(prn) & 0xFF)
    if int(prn) == 0:                                  # no further status words
        p[0:8] = [0] * 8
        _fill_filler(p, 8)
        return p
    _put(p, 9, 8, a - AREF, 512.0, True)               # delta-A (A_ref + dA)
    _put(p, 17, 7, omg0 / _PI, _POW[-6], True)         # Omega0
    _put(p, 24, 7, phi0 / _PI, _POW[-6], True)         # Phi0 = M0 + omega
    p[30] = int(health) & 1
    p[31] = (int(health) >> 1) & 1
    p[32] = (int(health) >> 2) & 1
    return p


def _put_block(bits: list[int], start: int, block) -> None:
    for i, bit in enumerate(block):
        bits[start - 1 + i] = int(bit) & 1


def _text_page_bits(text: str) -> list[int]:
    """232 bits = 29 seven/eight-bit characters (ICD 3.5.4, Figure 3.5-7)."""
    chars = (text + " " * TEXT_PAGE_CHARS)[:TEXT_PAGE_CHARS]
    out: list[int] = []
    for ch in chars:
        out.extend(_uint_bits(ord(ch) & 0xFF, 8))
    return out


def subframe3_bits(eph: Ephemeris, frame_start, page: int = 3) -> np.ndarray:
    """Assemble the 274-bit subframe 3 (250 data bits + CRC-24Q)."""
    b = [0] * SF3_DATA_BITS
    _put(b, 1, 8, int(eph.prn) & 0xFF)                 # PRN
    _put(b, 9, 6, int(page) & 0x3F)                    # page number

    if page == 3:                                      # reduced almanac
        _put(b, 15, 13, int(frame_start.week) & 0x1FFF)    # WN_la
        _put(b, 28, 8, int(round(float(eph.toe.sec) / 4096.0)) & 0xFF)  # t_oa
        a = float(eph.sqrta) ** 2
        packets = [reduced_almanac_packet(
            eph.prn, a, float(eph.omg0), float(eph.m0) + float(eph.aop),
            health=int(eph.svhlth) & 0x7)]
        packets += [reduced_almanac_packet(0, 0.0, 0.0, 0.0)] * 5
        starts = (36, 69, 102, 135, 168, 201)
        for start, packet in zip(starts, packets):
            _put_block(b, start, packet)
        # bits 234..250 reserved (left zero)
    elif page == 6:                                    # text message
        _put(b, 15, 4, 0)                              # text page number
        _put_block(b, 19, _text_page_bits(TEXT_PAGE_TEXT))
    else:                                              # reserved page
        _fill_filler(b, 15)

    return np.array(b + _uint_bits(crc24q(b), CRC_BITS), dtype=np.int8)


# ----------------------------------------------------------------------
# full frame
# ----------------------------------------------------------------------
def choose_sf3_page(frame_start) -> int:
    """Deterministic subframe 3 page schedule for the simulator."""
    return SF3_PAGES[page_index(frame_start) % len(SF3_PAGES)]


def cnav2_frame(eph: Ephemeris, frame_start, sf3_page: int | None = None
                ) -> np.ndarray:
    """Return one 1800-symbol CNAV-2 frame for the frame starting at
    ``frame_start`` (aligned to an 18 s GPS epoch).

    The first 52 symbols are the BCH-encoded TOI; the following 1748 are the
    interleaved LDPC symbols of subframes 2 and 3.
    """
    coded2 = ldpc_encode(subframe2_bits(eph, frame_start), 2)      # 1200
    page = choose_sf3_page(frame_start) if sf3_page is None else int(sf3_page)
    coded3 = ldpc_encode(subframe3_bits(eph, frame_start, page), 3)  # 548
    inter = interleave(np.concatenate([coded2, coded3]))            # 1748
    return np.concatenate([bch_encode(toi_count(frame_start)),
                           inter]).astype(np.int8)


def unmake_frame(frame: np.ndarray) -> dict:
    """De-interleave and split a received 1800-symbol frame (for tests)."""
    sym = np.asarray(frame, dtype=np.int8)
    if sym.size != FRAME_SYMBOLS:
        raise ValueError(f"frame must be {FRAME_SYMBOLS} symbols")
    inter = deinterleave(sym[BCH_SYMBOLS:])
    return {
        "toi_symbols": sym[:BCH_SYMBOLS],
        "toi": bch_decode(sym[:BCH_SYMBOLS]),
        "sf2": inter[:2 * SF2_BITS],
        "sf3": inter[2 * SF2_BITS:],
    }


# ----------------------------------------------------------------------
# self-test
# ----------------------------------------------------------------------
#: Known 52-symbol TOI words (from the public SignalSim/IS-GPS-800 TOI table).
_BCH_VECTORS = {
    0: "0000000000000000000000000000000000000000000000000000",
    1: "0000000011110011101010010000010110110100101111100011",
    2: "0000000100010100111110110000111011011101110000100101",
    100: "0011001000110111100000110011110100001101000100110110",
    255: "0111111110101110100110001111110010010011100101011110",
    399: "1011100000011011011000110101000010000000010100010110",
}

#: LDPC known-answer vectors (data, parity) from the public generator matrix,
#: expressed as hex bit strings (MSB first).
# <<LDPC_VECTORS_START>>
_LDPC_VECTORS = {
    2: [
        ("ab53cbd409254d15018c417d81a6409f4fcf816570100347ec333e449e05"
          "5d619e4ddef899f7cf0093e5b3c6ecddd1639dc5e6ad87c9501c98c98419"
          "c00d7ded639d58d8384a6ee660a41d",
         "f3413b0d82c516fe4d3e5e6c8c9d1b2f3d2fe162aa9eb2326292092e0fd6"
          "8aed34e69b69da20004b2068a9c8064a16e5e55a7487368886011da982b7"
          "38989ee4240dd3218fd9a89174d9e7"),
        ("fc931e5883f6ccdd8ad6d006aa8005db8e3eb65810d3b5ed5e3c57a428a4"
          "8ac9705ff70832d225f825f5f128676905d57acc1875af91386003e6240f"
          "fe340deade990891dcf394bb33156f",
         "7c002039218c2ca2db7aacb27307b9138d69a23589f93fc4d2372bf9f33b"
          "cf5a85d4f5b52416cf80468a57a554ce98185eb76dd9bf1e16b526769817"
          "9a39fc02b71948dc8eb50c0de039c0"),
    ],
    3: [
        ("33d69ee8e821f1d21d91f1a767b961728bd8fdccf9abe1ee7c07aaf002b2"
          "17c45a95f",
         "2a48c34219a04b1acee17968aa335b7f6a1504cc1f5e73f19d4386dbcd8a"
          "6b088ce80"),
        ("3764e4ba6afd87a2d2697ad47c0e1235121968dcbf79c14ac3672c451525"
          "804dfde82",
         "e0492eb3e8ca91592efb77a95a5b10d8a96b511ffa0d0ff38a40babd1c41"
          "b3dc2139"),
    ],
}
# <<LDPC_VECTORS_END>>


def _hex_to_bits(text: str, width: int) -> np.ndarray:
    value = int(text, 16)
    return np.array(_uint_bits(value, width), dtype=np.int8)


def _gps_ephemeris() -> Ephemeris:
    """A representative GPS (not Galileo) broadcast ephemeris for tests."""
    from .gpstime import GpsTime

    eph = Ephemeris()
    eph.system = "G"
    eph.prn = 5
    eph.toc = GpsTime(2190, 432000.0)
    eph.toe = GpsTime(2190, 432000.0)
    eph.m0 = 0.5
    eph.deltan = 4.5e-9
    eph.ecc = 0.01
    eph.sqrta = 5153.6                 # A ~ 26,559,593 m (near AREF)
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


def _synthetic_ephemeris() -> Ephemeris:
    """Backwards-compatible alias of :func:`_gps_ephemeris`."""
    return _gps_ephemeris()


def self_test() -> None:
    """Built-in consistency checks (also exercised by the unit tests)."""
    # 1) BCH known vectors.
    for toi, text in _BCH_VECTORS.items():
        got = "".join(str(int(b)) for b in bch_encode(toi))
        if got != text:
            raise AssertionError(f"BCH mismatch for TOI {toi}")
    if bch_decode(bch_encode(123)) != 123:
        raise AssertionError("BCH decode failed to recover TOI")

    # 2) CRC-24Q agrees with the SBAS implementation; valid codewords are 0.
    try:
        from .sbas import crc24 as sbas_crc24
        rng = np.random.default_rng(0)
        for _ in range(8):
            bits = rng.integers(0, 2, size=250).tolist()
            if crc24(bits) != sbas_crc24(bits):
                raise AssertionError("CRC-24Q disagrees with SBAS")
    except ImportError:  # pragma: no cover
        pass

    # 3) LDPC known-answer vectors and syndrome check.
    for sub, vectors in _LDPC_VECTORS.items():
        data_w = SF2_BITS if sub == 2 else SF3_BITS
        for data_hex, par_hex in vectors:
            data = _hex_to_bits(data_hex, data_w)
            par = _hex_to_bits(par_hex, data_w)
            expect = np.concatenate([data, par])
            if not np.array_equal(ldpc_encode(data, sub), expect):
                raise AssertionError(f"LDPC SF{sub} vector mismatch")
            if ldpc_check(expect, sub).any():
                raise AssertionError(f"LDPC SF{sub} syndrome non-zero")

    # 4) Interleaver round-trip.
    rng = np.random.default_rng(7)
    bits = rng.integers(0, 2, size=INTERLEAVED_SYMBOLS).astype(np.int8)
    if not np.array_equal(deinterleave(interleave(bits)), bits):
        raise AssertionError("interleaver round-trip failed")

    # 5) Full frame: length, binary, TOI consistency.
    from .gpstime import GpsTime
    eph = _synthetic_ephemeris()
    start = GpsTime(2190, 432000.0)
    frame = cnav2_frame(eph, start)
    if frame.shape != (FRAME_SYMBOLS,):
        raise AssertionError("frame must be 1800 symbols")
    if set(np.unique(frame).tolist()) - {0, 1}:
        raise AssertionError("frame is not a bit block")
    parts = unmake_frame(frame)
    if parts["toi"] != toi_count(start):
        raise AssertionError("TOI does not round-trip")
    if ldpc_check(parts["sf2"], 2).any():
        raise AssertionError("subframe 2 syndrome non-zero")
    if ldpc_check(parts["sf3"], 3).any():
        raise AssertionError("subframe 3 syndrome non-zero")
    if crc24q(parts["sf2"][:SF2_DATA_BITS].tolist()) != \
            _bits_to_uint(parts["sf2"][SF2_DATA_BITS:SF2_BITS]):
        raise AssertionError("subframe 2 CRC mismatch")
    if crc24q(parts["sf3"][:SF3_DATA_BITS].tolist()) != \
            _bits_to_uint(parts["sf3"][SF3_DATA_BITS:SF3_BITS]):
        raise AssertionError("subframe 3 CRC mismatch")


if __name__ == "__main__":  # pragma: no cover
    self_test()
    print("cnav2 self-test OK")
