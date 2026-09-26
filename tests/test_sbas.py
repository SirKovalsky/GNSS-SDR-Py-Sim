"""Тесты реальных SBAS L1 C/A сообщений (RTCA DO-229, 250 bps).

Запуск (без железа и без сети)::

    E:\\MySoftware\\SDR_Scan\\.venv\\Scripts\\python.exe -m pytest tests -q

Проверяются: CRC-24Q (на реальных захваченных сообщениях IGS geo_sbas.txt),
обрамление 250-битного сообщения, round-trip полей MT9/MT17, скорость потока
250 бит/с и интеграция с :class:`gnss_sim.engine.SignalEngine`.
"""

from __future__ import annotations

import numpy as np

from gnss_sim import sbas
from gnss_sim.engine import SignalEngine
from gnss_sim.gpstime import GpsTime, date2gps
from gnss_sim.orbit import llh2xyz
from gnss_sim.rinex import IonoUtc


# ----------------------------------------------------------------------
# CRC / обрамление
# ----------------------------------------------------------------------
def test_crc24_matches_real_captured_messages() -> None:
    """CRC совпадает с реальными SBAS-сообщениями из IGS geo_sbas.txt."""
    for raw in sbas._REAL_MESSAGES:
        bits = sbas._bytes_to_bits(raw)
        parsed = sbas.parse_message(bits[:250])
        assert parsed["crc_ok"] is True
        # Восстановленная чётность == переданной в битах 226..249.
        assert sbas.crc24(bits[:226]) == parsed["parity"]
        # Действительное кодовое слово делится на порождающий полином.
        assert sbas.crc24(bits[:250]) == 0


def test_crc24_detects_single_bit_error() -> None:
    msg = sbas.make_message(17, sbas.mt17_data([], t0_sod=0.0))
    assert sbas.parse_message(msg)["crc_ok"] is True
    for pos in (0, 13, 100, 226, 249):
        bad = msg.copy()
        bad[pos] ^= 1
        assert sbas.parse_message(bad)["crc_ok"] is False


def test_message_length_preamble_and_type() -> None:
    msg = sbas.make_message(63, [0] * sbas.SBAS_DATA_BITS)
    assert msg.shape == (sbas.SBAS_MSG_BITS,)
    assert msg.dtype == np.int8
    assert set(np.unique(msg).tolist()) <= {0, 1}

    parsed = sbas.parse_message(msg)
    assert parsed["msg_type"] == 63
    assert parsed["preamble"] == 0x53       # 0b10110011
    assert parsed["crc_ok"] is True

    # Короткое поле данных дополняется нулями справа, CRC остаётся верной.
    short = sbas.make_message(2, [1, 0, 1])
    assert sbas.parse_message(short)["crc_ok"] is True


def test_bits_to_bipolar_and_bytes() -> None:
    assert np.array_equal(sbas.bits_to_bipolar([0, 1, 0, 1]),
                          np.array([1, -1, 1, -1], dtype=np.int8))
    msg = sbas.make_message(9, sbas.mt9_data())
    packed = sbas.message_bytes(msg)
    assert len(packed) == 32
    assert packed[0] == 0x53
    # Последние 6 бит — нулевой паддинг.
    assert packed[31] & 0x3F == 0


# ----------------------------------------------------------------------
# MT9 / MT17 round-trip
# ----------------------------------------------------------------------
def test_mt17_field_roundtrip() -> None:
    pos = (42164000.0, -7000000.0, 1000.0)
    alm = sbas._geo_almanac(123, pos)
    data = sbas.mt17_data([alm], t0_sod=43200.0)
    assert len(data) == sbas.SBAS_DATA_BITS

    decoded = sbas.mt17_decode(data)
    entry = decoded["almanacs"][0]
    assert entry["prn"] == 123
    assert entry["health"] == 0
    assert abs(entry["x"] - pos[0]) <= 2600.0
    assert abs(entry["y"] - pos[1]) <= 2600.0
    assert abs(entry["z"] - pos[2]) <= 26000.0
    assert abs(decoded["t0_sod"] - 43200.0) <= 64.0
    # Неиспользуемые слоты помечены PRN = 0.
    assert decoded["almanacs"][1]["prn"] == 0
    assert decoded["almanacs"][2]["prn"] == 0

    # Полное сообщение декодируется и проходит CRC.
    msg = sbas.make_mt17([alm], t0_sod=43200.0, preamble=sbas.PREAMBLES[1])
    parsed = sbas.parse_message(msg)
    assert parsed["crc_ok"] is True
    assert parsed["msg_type"] == 17
    assert parsed["preamble"] == 0x9A


def test_mt9_field_roundtrip() -> None:
    pos = (42164000.0, -12000000.0, 0.0)
    data = sbas.mt9_data(t0_sod=43216.0, ura=2, position=pos,
                         velocity=(0.0, 0.0, 0.0), af0=1e-9)
    assert len(data) == sbas.SBAS_DATA_BITS

    decoded = sbas.mt9_decode(data)
    assert abs(decoded["t0_sod"] - 43216.0) <= 16.0
    assert decoded["ura"] == 2
    assert abs(decoded["xpos"] - pos[0]) <= 0.08 * 2
    assert abs(decoded["ypos"] - pos[1]) <= 0.08 * 2
    assert abs(decoded["zpos"] - pos[2]) <= 0.4 * 2
    assert abs(decoded["af0"] - 1e-9) <= 2.0 ** -31

    msg = sbas.make_mt9(t0_sod=43216.0, position=pos)
    assert sbas.parse_message(msg)["crc_ok"] is True


def test_mt9_matches_simulated_off_meridian_geo_position() -> None:
    """S120/S123/S126 MT9 must encode the exact simulated GEO ECEF position.

    The engine places each synthetic GEO at ``user_lon + dlon`` on the equator;
    an off-meridian GEO (S120/S126 at ±15°) is where a sign/scale/coordinate
    error in MT9 would show up.  Decoding the broadcast MT9 must reproduce the
    engine position to within the ICD quantisation (0.08 m for X/Y).
    """
    from gnss_sim.engine import _SBAS_GEO_RADIUS
    for user_lon_deg in (139.77, 30.0, -75.0):
        user_lon = np.radians(user_lon_deg)
        for prn, dlon in ((120, -15.0), (123, 0.0), (126, 15.0)):
            glon = user_lon + np.radians(dlon)
            sv = np.array([_SBAS_GEO_RADIUS * np.cos(glon),
                           _SBAS_GEO_RADIUS * np.sin(glon), 0.0])
            bits = sbas.sbas_stream_bits(6, prn=prn, sv_ecef=sv, t0_sod=0.0)
            mt9 = None
            for k in range(6):
                m = bits[k * sbas.SBAS_MSG_BITS:(k + 1) * sbas.SBAS_MSG_BITS]
                p = sbas.parse_message(m)
                assert p["crc_ok"]
                if p["msg_type"] == 9:
                    mt9 = sbas.mt9_decode(p["data"])
            assert mt9 is not None
            assert abs(mt9["xpos"] - sv[0]) <= 0.08
            assert abs(mt9["ypos"] - sv[1]) <= 0.08
            assert abs(mt9["zpos"] - sv[2]) <= 0.4
            # Also self-consistent across the MT9 propagation (zero velocity).
            assert mt9["xdot"] == 0.0 and mt9["ydot"] == 0.0



# ----------------------------------------------------------------------
# Поток сообщений 250 бит/с
# ----------------------------------------------------------------------
def test_stream_rate_is_250_bps() -> None:
    pos = (42164000.0, 0.0, 0.0)
    for duration in (3, 6, 9):
        messages = list(sbas.sbas_message_stream(
            duration, prn=122, sv_ecef=pos, t0_sod=1000.0))
        assert len(messages) == duration
        assert all(m.shape == (250,) for m in messages)
        bits = sbas.sbas_stream_bits(duration, prn=122, sv_ecef=pos,
                                     t0_sod=1000.0)
        # 250 бит/с за duration секунд.
        assert bits.size == sbas.SBAS_RATE_BPS * duration
        for k, msg in enumerate(messages):
            assert sbas.parse_message(msg)["crc_ok"] is True
            assert int(msg[0:8].dot(1 << np.arange(7, -1, -1))) == \
                sbas.PREAMBLES[k % 3]


def test_schedule_requires_multiple_of_three() -> None:
    try:
        sbas.sbas_bit_block(120, (42164000.0, 0.0, 0.0), schedule=(9, 17))
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("ожидался ValueError для расписания длиной 2")


# ----------------------------------------------------------------------
# Интеграция с движком
# ----------------------------------------------------------------------
def _make_engine() -> SignalEngine:
    start = date2gps(2022, 1, 1, 0, 0, 0)
    xyz = llh2xyz(np.radians(35.0), np.radians(139.0), 10.0)
    return SignalEngine(
        {}, IonoUtc(enable=False, vflg=False), lambda g: xyz, start, 2.6e6,
        backend="cpu", enable_ca=False, enable_l1c=False,
        enable_galileo=False, enable_qzss=False, enable_beidou=False,
        enable_sbas=True, iono_enable=False)


def test_engine_sbas_channel_is_populated() -> None:
    eng = _make_engine()
    sbas_chans = [c for c in eng.channels if c.kind == "sbas"]
    assert sbas_chans, "должен быть хотя бы один SBAS-канал"
    ch = sbas_chans[0]
    assert ch.sbas_bits is not None and ch.sbas_bits.ndim == 1
    assert ch.sbas_bits.size % sbas.SBAS_MSG_BITS == 0
    assert ch.sbas_bits.size == 6 * sbas.SBAS_MSG_BITS
    assert ch.sbas_frame_start is not None
    assert ch.sbas_frame_start.sec % 6.0 == 0.0
    assert set(np.unique(ch.sbas_bits).tolist()) == {0, 1}


def test_engine_sbas_term_has_expected_bit_transitions() -> None:
    eng = _make_engine()
    ch = next(c for c in eng.channels if c.kind == "sbas")
    t = np.arange(16, dtype=np.float64) * 1e-4
    carrier = np.ones_like(t)

    ca = ch.ca.astype(np.float64)
    chip = np.floor(ch.ca_phase + 0.0 * t).astype(np.int64) % 1023
    code = ca[chip]
    assert not np.any(code == 0.0)

    for sym in range(4):
        g = GpsTime(eng.g.week, float(sym) * 0.004)
        out = eng._sbas_term(ch, g, t, 0.0, carrier, np)
        data = out / code
        expected = 1.0 - 2.0 * float(ch.sbas_bits[sym])
        assert np.allclose(data, expected)
    # Соседние биты преамбулы MT9 (0x53 = 0101...) чередуются.
    bits = [1.0 - 2.0 * float(ch.sbas_bits[i]) for i in range(4)]
    assert bits[0] != bits[1] and bits[1] != bits[2]


def test_engine_sbas_signal_nontrivial() -> None:
    eng = _make_engine()
    block = eng.generate_block(8192)
    assert block.shape == (8192,)
    assert np.all(np.isfinite(block))
    assert float(np.abs(block).max()) > 0.0
    # Сигнал не константа (BPSK-модуляция реальным кодом/данными).
    assert float(np.abs(block).std()) > 0.0


if __name__ == "__main__":  # pragma: no cover
    sbas.self_test()
    print("sbas self-test OK")
