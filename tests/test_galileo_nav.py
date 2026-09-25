"""Tests for the Galileo E1-B I/NAV navigation message (250 symbol/s).

Run without hardware or network::

    E:\\MySoftware\\SDR_Scan\\.venv\\Scripts\\python.exe -m pytest tests -q

Covered:
* convolutional FEC (rate 1/2, K=7, G1=171o / G2=133o, G2 inverted) and the
  30x8 block interleaver against the ICD Annex D.2 numerical example;
* CRC-24Q (0x1864CFB) against the independently validated SBAS CRC and on a
  valid codeword / single-bit error;
* full 500-symbol page round-trip (sync, FEC, CRC, 128-bit word);
* ephemeris word field round-trips and scaling;
* sub-frame geometry (15 pages = 30 s = 7500 symbols at 250 bps);
* engine integration (channel population, 4 ms symbol period) on CPU and,
  when a CUDA device is present, a CPU-vs-CUDA comparison.
"""

from __future__ import annotations

import numpy as np
import pytest

from gnss_sim import galileo_nav as gal
from gnss_sim import sbas
from gnss_sim.engine import SignalEngine, cuda_available
from gnss_sim.gpstime import GpsTime, date2gps
from gnss_sim.orbit import llh2xyz
from gnss_sim.rinex import IonoUtc, sv_key


# ----------------------------------------------------------------------
# FEC / interleaver
# ----------------------------------------------------------------------
def test_fec_matches_icd_annex_d2() -> None:
    inp = gal._string_bits(gal._ICD_INPUT)
    encoded = gal.conv_encode(inp)
    assert encoded == gal._string_bits(gal._ICD_ENCODED)
    inter = gal.interleave(encoded).tolist()
    assert inter == gal._string_bits(gal._ICD_INTERLEAVED)
    # The Viterbi decoder inverts the encoder.
    assert gal.viterbi_decode(encoded).tolist() == inp


def test_interleaver_deinterleaver_roundtrip() -> None:
    rng = np.random.default_rng(3)
    data = rng.integers(0, 2, size=240).astype(np.int8)
    assert np.array_equal(gal.deinterleave(gal.interleave(data)), data)


def test_conv_encode_length_and_tail() -> None:
    info = [0] * gal.PAGE_PART_BITS
    coded = gal.conv_encode(info)
    assert len(coded) == 2 * gal.PAGE_PART_BITS
    # A stream that is all zeros plus feedback-free encoder gives all zeros
    # except the inverted G2 branch on the first symbol.
    assert coded == [0, 1] * gal.PAGE_PART_BITS


# ----------------------------------------------------------------------
# CRC-24Q
# ----------------------------------------------------------------------
def test_crc24q_matches_sbas_and_codeword() -> None:
    rng = np.random.default_rng(1234)
    for _ in range(16):
        bits = rng.integers(0, 2, size=196).tolist()
        assert gal.crc24q(bits) == sbas.crc24(bits)

    msg = rng.integers(0, 2, size=196).tolist()
    parity = gal.crc24q(msg)
    assert gal.crc24q(msg + gal._bits(parity, 24)) == 0
    bad = list(msg)
    bad[13] ^= 1
    assert gal.crc24q(bad) != parity


# ----------------------------------------------------------------------
# page framing / word round-trips
# ----------------------------------------------------------------------
def test_page_length_sync_and_roundtrip() -> None:
    eph = gal._synthetic_ephemeris()
    word = gal.word_1(eph)
    page = gal.make_page(word)
    assert page.shape == (gal.PAGE_SYMBOLS,)
    assert page.dtype == np.int8
    assert set(np.unique(page).tolist()) == {0, 1}
    # The even part (first 250 symbols) starts with the I/NAV sync word.
    assert np.array_equal(page[:gal.SYNC_LEN], gal.SYNC_BITS)

    parsed = gal.parse_page(page)
    assert parsed["sync_ok"] and parsed["crc_ok"]
    assert parsed["word_type"] == 1
    assert parsed["word"] == [int(b) for b in word]


def test_page_large_error_breaks_crc() -> None:
    page = gal.make_page(gal.word_2(gal._synthetic_ephemeris()))
    assert gal.parse_page(page)["crc_ok"]
    bad = page.copy()
    # A ~25 % burst of symbol errors overwhelms the FEC; the CRC must reject
    # the resulting (wrong) codeword.
    bad[250 + 40:250 + 100] ^= 1
    assert gal.parse_page(bad)["crc_ok"] is False


def _read_fields(word: list[int]) -> dict:
    r = gal._read_scaled
    raw = gal._read_raw
    return {
        "word_type": raw(word, 1, 6),
        "iodnav": raw(word, 7, 10),
        "t0e": r(word, 17, 14, gal.LSB["t0e"]),
        "m0": r(word, 31, 32, gal.LSB["pi_2_31"], True),
        "ecc": r(word, 63, 32, gal.LSB["e"]),
        "sqrta": r(word, 95, 32, gal.LSB["sqrtA"]),
        "omg0": r(word, 17, 32, gal.LSB["pi_2_31"], True) if raw(word, 1, 6) == 2 else None,
        "inc0": r(word, 49, 32, gal.LSB["pi_2_31"], True) if raw(word, 1, 6) == 2 else None,
        "aop": r(word, 81, 32, gal.LSB["pi_2_31"], True) if raw(word, 1, 6) == 2 else None,
        "idot": r(word, 113, 14, gal.LSB["i_dot"], True) if raw(word, 1, 6) == 2 else None,
        "omgdot": r(word, 17, 24, gal.LSB["omega_dot"], True) if raw(word, 1, 6) == 3 else None,
        "deltan": r(word, 41, 16, gal.LSB["delta_n"], True) if raw(word, 1, 6) == 3 else None,
        "cuc": r(word, 57, 16, gal.LSB["cuc_cus_cic_cis"], True) if raw(word, 1, 6) == 3 else None,
        "cus": r(word, 73, 16, gal.LSB["cuc_cus_cic_cis"], True) if raw(word, 1, 6) == 3 else None,
        "crc": r(word, 89, 16, gal.LSB["crc_crs"], True) if raw(word, 1, 6) == 3 else None,
        "crs": r(word, 105, 16, gal.LSB["crc_crs"], True) if raw(word, 1, 6) == 3 else None,
        "cic": r(word, 23, 16, gal.LSB["cuc_cus_cic_cis"], True) if raw(word, 1, 6) == 4 else None,
        "cis": r(word, 39, 16, gal.LSB["cuc_cus_cic_cis"], True) if raw(word, 1, 6) == 4 else None,
        "t0c": r(word, 55, 14, gal.LSB["t0c"]) if raw(word, 1, 6) == 4 else None,
        "af0": r(word, 69, 31, gal.LSB["af0"], True) if raw(word, 1, 6) == 4 else None,
        "af1": r(word, 100, 21, gal.LSB["af1"], True) if raw(word, 1, 6) == 4 else None,
        "af2": r(word, 121, 6, gal.LSB["af2"], True) if raw(word, 1, 6) == 4 else None,
    }


def _page_word(word: np.ndarray) -> list[int]:
    return gal.parse_page(gal.make_page(word))["word"]


def test_ephemeris_word_roundtrips() -> None:
    eph = gal._synthetic_ephemeris()
    eph.toe = GpsTime(eph.toe.week, 432000.0)
    eph.toc = GpsTime(eph.toc.week, 432060.0)

    w1 = _read_fields(_page_word(gal.word_1(eph)))
    assert w1["word_type"] == 1
    assert w1["iodnav"] == 42
    assert abs(w1["m0"] - eph.m0) <= gal.LSB["pi_2_31"]
    assert abs(w1["ecc"] - eph.ecc) <= gal.LSB["e"]
    assert abs(w1["sqrta"] - eph.sqrta) <= gal.LSB["sqrtA"]
    assert abs(w1["t0e"] - 432000.0) <= 60.0

    w2 = _read_fields(_page_word(gal.word_2(eph)))
    assert abs(w2["omg0"] - eph.omg0) <= gal.LSB["pi_2_31"]
    assert abs(w2["inc0"] - eph.inc0) <= gal.LSB["pi_2_31"]
    assert abs(w2["aop"] - eph.aop) <= gal.LSB["pi_2_31"]
    assert abs(w2["idot"] - eph.idot) <= gal.LSB["i_dot"]

    w3 = _read_fields(_page_word(gal.word_3(eph)))
    assert w3["word_type"] == 3
    assert abs(w3["omgdot"] - eph.omgdot) <= gal.LSB["omega_dot"]
    assert abs(w3["deltan"] - eph.deltan) <= gal.LSB["delta_n"]
    assert abs(w3["cuc"] - eph.cuc) <= gal.LSB["cuc_cus_cic_cis"]
    assert abs(w3["cus"] - eph.cus) <= gal.LSB["cuc_cus_cic_cis"]
    assert abs(w3["crc"] - eph.crc) <= gal.LSB["crc_crs"]
    assert abs(w3["crs"] - eph.crs) <= gal.LSB["crc_crs"]

    w4 = _read_fields(_page_word(gal.word_4(eph)))
    assert abs(w4["cic"] - eph.cic) <= 2 * gal.LSB["cuc_cus_cic_cis"]
    assert abs(w4["cis"] - eph.cis) <= 2 * gal.LSB["cuc_cus_cic_cis"]
    assert abs(w4["t0c"] - 432060.0) <= 60.0
    assert abs(w4["af0"] - eph.af0) <= 2 * gal.LSB["af0"]
    assert abs(w4["af1"] - eph.af1) <= 2 * gal.LSB["af1"]


def test_word5_and_word6_roundtrip() -> None:
    eph = gal._synthetic_ephemeris()
    w5 = _page_word(gal.word_5(eph, 2200, 432000.0, ai0=2.5, ai1=0.0, ai2=0.0))
    parsed5 = gal.parse_page(gal.make_page(gal.word_5(
        eph, 2200, 432000.0, ai0=2.5)))
    assert parsed5["word_type"] == 5
    assert gal._read_raw(parsed5["word"], 74, 12) == 2200 % 4096
    assert gal._read_raw(parsed5["word"], 86, 20) == 432000
    assert abs(gal._read_scaled(parsed5["word"], 48, 10, gal.LSB["bgd"], True)
               - eph.tgd) <= 2 * gal.LSB["bgd"]
    assert len(w5) == 128

    w6 = gal.parse_page(gal.make_page(gal.word_6(432000.0, dtls=18)))["word"]
    assert gal._read_raw(w6, 1, 6) == 6
    assert gal._read_raw(w6, 63, 8) == 18
    assert gal._read_raw(w6, 106, 20) == 432000


# ----------------------------------------------------------------------
# sub-frame geometry / bit rate
# ----------------------------------------------------------------------
def test_subframe_geometry_and_page_order() -> None:
    eph = gal._synthetic_ephemeris()
    block = gal.inav_bit_block(eph, gst_week=2200, gst_tow=432000.0)
    assert block.shape == (30 * 250,)          # 30 s at 250 symbol/s
    assert set(np.unique(block).tolist()) == {0, 1}

    word_types = []
    for p in range(gal.SUBBFRAME_PAGES):
        page = block[p * gal.PAGE_SYMBOLS:(p + 1) * gal.PAGE_SYMBOLS]
        parsed = gal.parse_page(page)
        assert parsed["crc_ok"] and parsed["sync_ok"]
        word_types.append(parsed["word_type"])
    assert word_types == list(gal.DEFAULT_SEQUENCE)


def test_bit_period_is_4ms() -> None:
    """The I/NAV block is a 250 bps stream (one symbol per 4 ms)."""
    eph = gal._synthetic_ephemeris()
    block = gal.inav_bit_block(eph, gst_week=2200, gst_tow=0.0)
    assert block.size == gal.INAV_RATE_BPS * gal.SUBBFRAME_SECONDS
    # Reconstructing a page part uses exactly 250 symbols = 1 s.
    assert gal.PAGE_PART_SYMBOLS == gal.INAV_RATE_BPS * 1


# ----------------------------------------------------------------------
# engine integration
# ----------------------------------------------------------------------
def _make_engine(backend: str = "cpu") -> SignalEngine:
    eph = gal._synthetic_ephemeris()
    start = date2gps(2022, 1, 1, 0, 0, 0)
    eph.toe = GpsTime(start.week, 0.0)
    eph.toc = GpsTime(start.week, 0.0)
    eph.finalize()
    xyz = llh2xyz(np.radians(35.0), np.radians(139.0), 10.0)
    return SignalEngine(
        {sv_key("E", eph.prn): [eph]}, IonoUtc(enable=False, vflg=False),
        lambda g: xyz, start, 2.6e6, backend=backend,
        enable_ca=False, enable_l1c=False, enable_galileo=True,
        enable_qzss=False, enable_sbas=False, enable_beidou=False,
        el_mask=-2.0, iono_enable=False)


def test_engine_galileo_channel_is_populated() -> None:
    eng = _make_engine()
    chans = [c for c in eng.channels if c.kind == "galileo"]
    assert chans, "expected at least one Galileo channel"
    ch = chans[0]
    assert ch.galileo_bits is not None
    assert ch.galileo_bits.shape == (30 * 250,)
    assert set(np.unique(ch.galileo_bits).tolist()) == {0, 1}
    assert ch.galileo_frame_start is not None
    assert ch.galileo_frame_start.sec % 30.0 == 0.0


def test_engine_galileo_symbol_period_and_indexing() -> None:
    """I/NAV symbols span one 4 ms code period and follow the received code.

    Regression: the symbol index used to be derived from the *transmit* time
    (no ``rng/c``), so the data/secondary boundaries were shifted by the
    pseudorange and no longer coincided with the code-epoch wraps.  One symbol
    per code period, with the propagation delay included, is the ICD timing.
    """
    from gnss_sim.constants import SPEED_OF_LIGHT
    from gnss_sim.engine import _Geo
    from gnss_sim.gpstime import sub_gps_time

    eng = _make_engine()
    ch = next(c for c in eng.channels if c.kind == "galileo")
    bits = ch.galileo_bits
    # Constant E1-B code (ones) and silenced E1-C: the term is then the data
    # symbol times the CBOC subcarrier; sampling exactly on code epochs makes
    # the subcarrier constant too, so the ratio is the data ratio.
    ch.e1b = np.ones_like(ch.e1b)
    ch.e1c = np.zeros_like(ch.e1c)
    g = ch.galileo_frame_start
    rho = _Geo(2.0e7, 0.0, 2.0e7, (0.0, 1.0), 0.0)
    base_ms = (sub_gps_time(g, ch.galileo_frame_start) + 6.0
               - rho.rng / SPEED_OF_LIGHT) * 1000.0
    ch.e1_phase = (base_ms % 4.0) * 1023.0
    f_code = 1.023e6
    t = ((4092.0 - ch.e1_phase) / f_code
         + np.arange(40) * gal.CODE_PERIOD_S)
    term = eng._galileo_term(ch, g, t, f_code, np.ones_like(t), np, rho)
    assert not np.any(term == 0.0)
    ratio = (term / term[0]).real
    k = np.floor((base_ms + t * 1000.0) / 4.0).astype(np.int64)
    sym = k % bits.shape[0]
    expected = (1.0 - 2.0 * bits[sym]) / (1.0 - 2.0 * bits[sym[0]])
    assert np.allclose(ratio, expected)
    # One symbol per 4 ms: the index advances by exactly one each code period.
    assert np.all(np.diff(k) == 1)


def test_engine_galileo_composite_matches_icd_formula() -> None:
    """E1 = (E1B*CBOC+ - E1C*CBOC-)/sqrt(2) with symbol timing tied to code.

    Regression for two ICD deviations: the E1-C term used to be *added*
    (relative polarity of the data and pilot components was flipped) and the
    I/NAV symbol index ignored the propagation delay.
    """
    from gnss_sim.constants import SPEED_OF_LIGHT
    from gnss_sim.engine import CBOC_ALPHA, CBOC_BETA, _Geo
    from gnss_sim.gpstime import sub_gps_time

    eng = _make_engine()
    ch = next(c for c in eng.channels if c.kind == "galileo")
    ch.galileo_bits = np.zeros(7500, dtype=np.int8)      # constant +1 data
    g = ch.galileo_frame_start
    rho = _Geo(2.1e7, 0.0, 2.1e7, (0.0, 1.0), 0.0)
    f_code = 1.023e6
    fs = 8.184e6
    t = np.arange(int(fs * 0.008)) / fs
    term = eng._galileo_term(ch, g, t, f_code, np.ones_like(t), np, rho)

    cp = ch.e1_phase + f_code * t
    frac = cp - np.floor(cp)
    s1 = np.sign(np.sin(2.0 * np.pi * frac)); s1[s1 == 0.0] = 1.0
    s6 = np.sign(np.sin(2.0 * np.pi * 6.0 * frac)); s6[s6 == 0.0] = 1.0
    chip = np.floor(cp).astype(np.int64) % 4092
    base_ms = (sub_gps_time(g, ch.galileo_frame_start) + 6.0
               - rho.rng / SPEED_OF_LIGHT) * 1000.0
    k = (np.floor(base_ms / 4.0).astype(np.int64)
         + np.floor(cp / 4092.0).astype(np.int64))
    sec = ch.e1sec[k % 25]
    ref = (ch.e1b[chip] * (CBOC_ALPHA * s1 + CBOC_BETA * s6)
           - ch.e1c[chip] * sec * (CBOC_ALPHA * s1 - CBOC_BETA * s6)
           ) * 0.7071067811865476
    assert np.allclose(term, ref)


def test_engine_galileo_broadcasts_gst_week() -> None:
    """I/NAV word 5 must carry the GST week (GPS week - 1024), not the GPS week."""
    from gnss_sim.engine import GST_WEEK_OFFSET

    eng = _make_engine()
    ch = next(c for c in eng.channels if c.kind == "galileo")
    # page 4 of DEFAULT_SEQUENCE (1,2,3,4,5,..) is word type 5.
    page = ch.galileo_bits[4 * gal.PAGE_SYMBOLS:5 * gal.PAGE_SYMBOLS]
    parsed = gal.parse_page(page)
    assert parsed["word_type"] == 5
    wn = gal._read_raw(parsed["word"], 74, 12)
    expected = (int(ch.galileo_frame_start.week) - GST_WEEK_OFFSET) % 4096
    assert wn == expected
    # The GPS week would be a different value -> catches the regression.
    assert wn != int(ch.galileo_frame_start.week) % 4096


def test_engine_galileo_signal_nontrivial() -> None:
    eng = _make_engine()
    block = eng.generate_block(16384)
    assert block.shape == (16384,)
    assert np.all(np.isfinite(block))
    assert float(np.abs(block).max()) > 0.0
    assert float(np.abs(block).std()) > 0.0


def test_engine_galileo_rolls_subframe() -> None:
    eng = _make_engine()
    ch = next(c for c in eng.channels if c.kind == "galileo")
    first_epoch = ch.galileo_frame_start.sec
    # Push the engine clock past the 30 s boundary; the sub-frame must advance.
    eng.g.sec += 31.0
    eng._roll_frames()
    assert ch.galileo_frame_start.sec != first_epoch
    assert ch.galileo_frame_start.sec % 30.0 == 0.0
    assert ch.galileo_bits.shape == (30 * 250,)


@pytest.mark.skipif(not cuda_available(), reason="CUDA/CuPy not available")
def test_engine_cpu_vs_cuda() -> None:
    np.random.seed(0)
    cpu = _make_engine("cpu")
    np.random.seed(0)
    cuda = _make_engine("cuda")
    assert cuda.backend == "cuda"
    a = cpu.generate_block(65536)
    b = cuda.generate_block(65536)
    assert a.shape == b.shape
    # Only complex64 vs complex128 rounding differs; normalised correlation
    # must be very close to 1.
    num = float(np.abs(np.vdot(a, b)))
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    assert den > 0.0
    assert num / den > 0.99


if __name__ == "__main__":  # pragma: no cover
    gal.self_test()
    print("galileo_nav self-test OK")
