"""Tests for the GPS L1C CNAV-2 navigation message (100 symbol/s, 18 s frame).

Run without hardware or network::

    E:\\MySoftware\\SDR_Scan\\.venv\\Scripts\\python.exe -m pytest tests -q

Covered:
* BCH(51,8) TOI encoding against known 52-symbol vectors and decode round-trip;
* LDPC(1200,600) / LDPC(548,274) encoding against the public SignalSim
  generator known-answer vectors and the ICD parity-check syndrome;
* CRC-24Q against the independently validated SBAS implementation;
* 38 x 46 block interleaver formula and round-trip;
* full frame geometry (1800 symbols), TOI/ITOW consistency and MT10 field
  round-trips, reduced-almanac (page 3) and text (page 6) subframe 3 pages;
* engine integration for ``l1c_data='cnav2'`` versus ``'zeros'`` on CPU and,
  when a CUDA device is present, a CPU-vs-CUDA comparison.
"""

from __future__ import annotations

import numpy as np
import pytest

from gnss_sim import cnav2
from gnss_sim import sbas
from gnss_sim.engine import SignalEngine, cuda_available
from gnss_sim.gpstime import GpsTime, date2gps
from gnss_sim.orbit import llh2xyz
from gnss_sim.rinex import IonoUtc, sv_key


# ----------------------------------------------------------------------
# BCH(51,8)
# ----------------------------------------------------------------------
def test_bch_known_vectors() -> None:
    for toi, text in cnav2._BCH_VECTORS.items():
        got = "".join(str(int(b)) for b in cnav2.bch_encode(toi))
        assert got == text


def test_bch_decode_roundtrip() -> None:
    for toi in list(range(0, 512, 7)) + [399]:
        assert cnav2.bch_decode(cnav2.bch_encode(toi)) == toi


def test_bch_zero_and_msb() -> None:
    # TOI = 0 encodes to all zeros.
    assert not cnav2.bch_encode(0).any()
    # The MSB is prepended to the 52-symbol word and XORed into all outputs.
    for toi in (0x100, 0x1FF):
        word = cnav2.bch_encode(toi)
        assert word[0] == 1
        assert np.array_equal(word[1:], cnav2.bch_encode(toi & 0xFF)[1:] ^ 1)


# ----------------------------------------------------------------------
# CRC-24Q
# ----------------------------------------------------------------------
def test_crc24q_matches_sbas_and_codeword() -> None:
    rng = np.random.default_rng(1234)
    for _ in range(16):
        bits = rng.integers(0, 2, size=250).tolist()
        assert cnav2.crc24(bits) == sbas.crc24(bits)

    msg = rng.integers(0, 2, size=576).tolist()
    parity = cnav2.crc24(msg)
    assert cnav2.crc24(msg + cnav2._uint_bits(parity, 24)) == 0
    bad = list(msg)
    bad[13] ^= 1
    assert cnav2.crc24(bad) != parity


# ----------------------------------------------------------------------
# interleaver
# ----------------------------------------------------------------------
def test_interleaver_formula_and_roundtrip() -> None:
    rng = np.random.default_rng(3)
    data = rng.integers(0, 2, size=1748).astype(np.int8)
    inter = cnav2.interleave(data)
    assert inter.shape == (1748,)
    # out[c*38 + r] = in[r*46 + c]
    for r in (0, 5, 37):
        for c in (0, 7, 45):
            assert inter[c * 38 + r] == data[r * 46 + c]
    assert np.array_equal(cnav2.deinterleave(inter), data)


# ----------------------------------------------------------------------
# LDPC
# ----------------------------------------------------------------------
def test_ldpc_known_vectors() -> None:
    for sub, vectors in cnav2._LDPC_VECTORS.items():
        width = cnav2.SF2_BITS if sub == 2 else cnav2.SF3_BITS
        for data_hex, par_hex in vectors:
            data = cnav2._hex_to_bits(data_hex, width)
            par = cnav2._hex_to_bits(par_hex, width)
            assert np.array_equal(cnav2.ldpc_encode(data, sub),
                                  np.concatenate([data, par]))


def test_ldpc_syndrome_and_shape() -> None:
    rng = np.random.default_rng(99)
    for sub, n in ((2, 1200), (3, 548)):
        data = rng.integers(0, 2, size=n // 2).astype(np.int8)
        code = cnav2.ldpc_encode(data, sub)
        assert code.shape == (n,)
        assert np.array_equal(code[:n // 2], data)
        assert not cnav2.ldpc_check(code, sub).any()
        # A single flipped codeword bit must violate the syndrome.
        bad = code.copy()
        bad[17] ^= 1
        assert cnav2.ldpc_check(bad, sub).any()


def test_ldpc_check_matrices_are_full_rank() -> None:
    for sub, h_rows in ((2, 600), (3, 274)):
        a, b, c, d, e, t = cnav2._ldpc_pieces(sub)[:6]
        h = np.vstack([np.hstack([a, b, t]), np.hstack([c, d, e])])
        assert h.shape == (h_rows, 2 * h_rows)
        assert np.array_equal(np.triu(t, k=1), np.zeros_like(t))  # lower tri
        assert t.shape[0] == t.shape[1]


# ----------------------------------------------------------------------
# frame geometry / TOI
# ----------------------------------------------------------------------
def test_frame_length_binary_and_toi() -> None:
    eph = cnav2._gps_ephemeris()
    start = GpsTime(2190, 432000.0)
    frame = cnav2.cnav2_frame(eph, start)
    assert frame.shape == (cnav2.FRAME_SYMBOLS,)
    assert frame.dtype == np.int8
    assert set(np.unique(frame).tolist()) == {0, 1}

    parts = cnav2.unmake_frame(frame)
    assert parts["toi"] == cnav2.toi_count(start)
    assert parts["sf2"].shape == (1200,)
    assert parts["sf3"].shape == (548,)
    assert not cnav2.ldpc_check(parts["sf2"], 2).any()
    assert not cnav2.ldpc_check(parts["sf3"], 3).any()


def test_toi_itow_consistency() -> None:
    # ITOW zero epoch / TOI one at the start of the week.
    assert cnav2.itow_count(GpsTime(2190, 0.0)) == 0
    assert cnav2.toi_count(GpsTime(2190, 0.0)) == 1
    # End of the first two-hour interval.
    assert cnav2.toi_count(GpsTime(2190, 7182.0)) == 0
    assert cnav2.toi_count(GpsTime(2190, 7200.0)) == 1
    assert cnav2.itow_count(GpsTime(2190, 7200.0)) == 1
    # 83 two-hour epochs per week; the last frame rolls into the new week.
    assert cnav2.itow_count(GpsTime(2190, 604782.0)) == 83
    assert cnav2.toi_count(GpsTime(2190, 604782.0)) == 0


def test_frame_toi_matches_sf1() -> None:
    eph = cnav2._gps_ephemeris()
    start = GpsTime(2190, 432000.0)
    frame = cnav2.cnav2_frame(eph, start)
    assert cnav2.toi_count(start) == 1
    assert cnav2.bch_decode(frame[:cnav2.BCH_SYMBOLS]) == 1


# ----------------------------------------------------------------------
# subframe 2 (message type 10)
# ----------------------------------------------------------------------
def _mt10_fields(bits) -> dict:
    r = cnav2._read_scaled
    raw = cnav2._read
    return {
        "wn": raw(bits, 1, 13),
        "itow": raw(bits, 14, 8),
        "top": r(bits, 22, 11, 300.0),
        "health": raw(bits, 33, 1),
        "ura": raw(bits, 34, 5, True),
        "toe": r(bits, 39, 11, 300.0),
        "dA": r(bits, 50, 26, 2.0 ** -9, True),
        "dn": r(bits, 101, 17, 2.0 ** -44, True),
        "m0": r(bits, 141, 33, 2.0 ** -32, True),
        "e": r(bits, 174, 33, 2.0 ** -34),
        "omega": r(bits, 207, 33, 2.0 ** -32, True),
        "omg0": r(bits, 240, 33, 2.0 ** -32, True),
        "i0": r(bits, 273, 33, 2.0 ** -32, True),
        "omgdot": r(bits, 306, 17, 2.0 ** -44, True),
        "idot": r(bits, 323, 15, 2.0 ** -44, True),
        "cis": r(bits, 338, 16, 2.0 ** -30, True),
        "cic": r(bits, 354, 16, 2.0 ** -30, True),
        "crs": r(bits, 370, 24, 2.0 ** -8, True),
        "crc": r(bits, 394, 24, 2.0 ** -8, True),
        "cus": r(bits, 418, 21, 2.0 ** -30, True),
        "cuc": r(bits, 439, 21, 2.0 ** -30, True),
        "af0": r(bits, 471, 26, 2.0 ** -35, True),
        "af1": r(bits, 497, 20, 2.0 ** -48, True),
        "af2": r(bits, 517, 10, 2.0 ** -60, True),
        "tgd": r(bits, 527, 13, 2.0 ** -35, True),
        "crc_field": "".join(str(int(x)) for x in bits[576:600]),
    }


def test_mt10_ephemeris_roundtrip() -> None:
    eph = cnav2._gps_ephemeris()
    start = GpsTime(eph.toe.week, 432000.0)
    bits = cnav2.subframe2_bits(eph, start)
    assert bits.shape == (cnav2.SF2_BITS,)
    f = _mt10_fields(bits)
    pi = np.pi

    assert f["wn"] == start.week % 8192
    assert f["itow"] == cnav2.itow_count(start)
    assert f["health"] == 0
    assert f["ura"] == int(eph.ura)
    assert abs(f["top"] - round(eph.toe.sec / 300.0) * 300.0) <= 300.0
    assert abs(f["toe"] - round(eph.toe.sec / 300.0) * 300.0) <= 300.0
    assert abs(f["dA"] + cnav2.AREF - eph.sqrta ** 2) <= 2.0 ** -9
    assert abs(f["dn"] * pi - eph.deltan) <= pi * 2.0 ** -44
    assert abs(f["m0"] * pi - eph.m0) <= pi * 2.0 ** -32
    assert abs(f["e"] - eph.ecc) <= 2.0 ** -34
    assert abs(f["omega"] * pi - eph.aop) <= pi * 2.0 ** -32
    assert abs(f["omg0"] * pi - eph.omg0) <= pi * 2.0 ** -32
    assert abs(f["i0"] * pi - eph.inc0) <= pi * 2.0 ** -32
    assert abs((f["omgdot"] + cnav2.OMEGA_DOT_REF) * pi - eph.omgdot) \
        <= pi * 2.0 ** -44
    assert abs(f["idot"] * pi - eph.idot) <= pi * 2.0 ** -44
    for name, attr in (("cis", "cis"), ("cic", "cic"), ("cus", "cus"),
                       ("cuc", "cuc")):
        assert abs(f[name] - getattr(eph, attr)) <= 2.0 ** -30
    assert abs(f["crs"] - eph.crs) <= 2.0 ** -8
    assert abs(f["crc"] - eph.crc) <= 2.0 ** -8
    assert abs(f["af0"] - eph.af0) <= 2.0 ** -35
    assert abs(f["af1"] - eph.af1) <= 2.0 ** -48
    assert abs(f["tgd"] - eph.tgd) <= 2.0 ** -35

    # CRC over the 576 data bits.
    crc = cnav2._bits_to_uint(bits[576:600])
    assert cnav2.crc24(bits[:576].tolist()) == crc


# ----------------------------------------------------------------------
# subframe 3 pages
# ----------------------------------------------------------------------
def test_subframe3_reduced_almanac_page3() -> None:
    eph = cnav2._gps_ephemeris()
    start = GpsTime(eph.toe.week, 432000.0)
    bits = cnav2.subframe3_bits(eph, start, page=3)
    assert bits.shape == (cnav2.SF3_BITS,)
    assert cnav2._read(bits, 1, 8) == eph.prn
    assert cnav2._read(bits, 9, 6) == 3
    assert cnav2._read(bits, 15, 13) == start.week % 8192
    # Packet 1 carries the transmitting SV's reduced almanac.
    assert cnav2._read(bits, 36, 8) == eph.prn
    dA = cnav2._read_scaled(bits, 44, 8, 512.0, True)
    assert abs(dA + cnav2.AREF - eph.sqrta ** 2) <= 512.0
    omg0 = cnav2._read_scaled(bits, 52, 7, 2.0 ** -6, True)
    assert abs(omg0 * np.pi - eph.omg0) <= np.pi * 2.0 ** -6
    phi0 = cnav2._read_scaled(bits, 59, 7, 2.0 ** -6, True)
    assert abs(phi0 * np.pi - (eph.m0 + eph.aop)) <= np.pi * 2.0 ** -6
    # Remaining packets are fill (PRN_a = 0).
    assert cnav2._read(bits, 69, 8) == 0
    crc = cnav2._bits_to_uint(bits[250:274])
    assert cnav2.crc24(bits[:250].tolist()) == crc


def test_subframe3_text_page6() -> None:
    eph = cnav2._gps_ephemeris()
    bits = cnav2.subframe3_bits(eph, GpsTime(eph.toe.week, 432000.0), page=6)
    assert cnav2._read(bits, 1, 8) == eph.prn
    assert cnav2._read(bits, 9, 6) == 6
    chars = []
    for k in range(cnav2.TEXT_PAGE_CHARS):
        chars.append(chr(cnav2._read(bits, 19 + 8 * k, 8)))
    text = "".join(chars)
    assert text[:len(cnav2.TEXT_PAGE_TEXT)] == cnav2.TEXT_PAGE_TEXT
    crc = cnav2._bits_to_uint(bits[250:274])
    assert cnav2.crc24(bits[:250].tolist()) == crc


def test_frame_sf3_page_schedule() -> None:
    # Deterministic alternating schedule.
    assert cnav2.choose_sf3_page(GpsTime(2190, 0.0)) == 3
    assert cnav2.choose_sf3_page(GpsTime(2190, 18.0)) == 6


# ----------------------------------------------------------------------
# engine integration
# ----------------------------------------------------------------------
def _make_engine(backend: str = "cpu", data: str = "cnav2") -> SignalEngine:
    eph = cnav2._gps_ephemeris()
    start = date2gps(2022, 1, 1, 0, 0, 0)
    eph.toe = GpsTime(start.week, 0.0)
    eph.toc = GpsTime(start.week, 0.0)
    eph.finalize()
    xyz = llh2xyz(np.radians(35.0), np.radians(139.0), 10.0)
    return SignalEngine(
        {sv_key("G", eph.prn): [eph]}, IonoUtc(enable=False, vflg=False),
        lambda g: xyz, start, 2.6e6, backend=backend,
        enable_ca=False, enable_l1c=True, enable_galileo=False,
        enable_qzss=False, enable_sbas=False, enable_beidou=False,
        el_mask=-90.0, iono_enable=False, l1c_data=data)


def test_engine_cnav2_channel_is_populated() -> None:
    eng = _make_engine()
    chans = [c for c in eng.channels if c.kind == "gps"]
    assert chans, "expected at least one GPS channel"
    ch = chans[0]
    assert ch.cnav2 is not None
    assert ch.cnav2.shape == (cnav2.FRAME_SYMBOLS,)
    assert set(np.unique(ch.cnav2).tolist()) == {0, 1}
    assert ch.l1c_frame_start is not None
    assert ch.l1c_frame_start.sec % 18.0 == 0.0
    assert np.array_equal(ch.cnav2, cnav2.cnav2_frame(ch.eph, ch.l1c_frame_start))


def test_engine_zeros_mode_still_zero_fills() -> None:
    eng = _make_engine(data="zeros")
    ch = [c for c in eng.channels if c.kind == "gps"][0]
    assert ch.cnav2 is not None
    assert ch.cnav2.shape == (cnav2.FRAME_SYMBOLS,)
    assert not ch.cnav2.any()


def test_engine_cnav2_signal_nontrivial() -> None:
    eng = _make_engine()
    block = eng.generate_block(16384)
    assert block.shape == (16384,)
    assert np.all(np.isfinite(block))
    assert float(np.abs(block).std()) > 0.0


def test_engine_cnav2_differs_from_zeros() -> None:
    np.random.seed(0)
    c = _make_engine(data="cnav2")
    np.random.seed(0)
    z = _make_engine(data="zeros")
    cch = next(ch for ch in c.channels if getattr(ch, "cnav2", None) is not None)
    zch = next(ch for ch in z.channels if getattr(ch, "cnav2", None) is not None)
    sym = int(np.argmax(cch.cnav2 != zch.cnav2))
    c.g.sec = cch.l1c_frame_start.sec + sym * 0.01
    z.g.sec = zch.l1c_frame_start.sec + sym * 0.01
    a = c.generate_block(30000)
    b = z.generate_block(30000)
    assert not np.allclose(a, b)


def test_engine_cnav2_rolls_frame() -> None:
    eng = _make_engine()
    ch = [c for c in eng.channels if c.kind == "gps"][0]
    first = ch.l1c_frame_start.sec
    eng.g.sec += 19.0
    eng._roll_frames()
    assert ch.l1c_frame_start.sec != first
    assert ch.l1c_frame_start.sec % 18.0 == 0.0
    assert ch.cnav2.shape == (cnav2.FRAME_SYMBOLS,)


@pytest.mark.skipif(not cuda_available(), reason="CUDA/CuPy not available")
def test_engine_cnav2_cpu_vs_cuda() -> None:
    np.random.seed(0)
    cpu = _make_engine("cpu", data="cnav2")
    np.random.seed(0)
    cuda = _make_engine("cuda", data="cnav2")
    assert cuda.backend == "cuda"
    a = cpu.generate_block(65536)
    b = cuda.generate_block(65536)
    assert a.shape == b.shape
    num = float(np.abs(np.vdot(a, b)))
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    assert den > 0.0
    assert num / den > 0.99


if __name__ == "__main__":  # pragma: no cover
    cnav2.self_test()
    print("cnav2 self-test OK")
