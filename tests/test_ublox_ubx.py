"""Tests for the u-blox UBX parsing helpers in ``gnss_sim/ublox.py``.

All frames are synthetic and carry correct Fletcher checksums, so no hardware
or COM port is needed.  Run with::

    E:\\MySoftware\\SDR_Scan\\.venv\\Scripts\\python.exe -m pytest tests -q
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from gnss_sim import ublox  # noqa: E402


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _field(text: str, size: int = 30) -> bytes:
    raw = text.encode("ascii")
    assert len(raw) <= size
    return raw + b"\x00" * (size - len(raw))


def _sv(gnss_id: int, sv_id: int, cno: int, elev: int, azim: int,
        pr_res: int, flags: int) -> bytes:
    """One 12-byte UBX-NAV-SAT version-1 satellite block."""
    return (
        bytes([gnss_id & 0xFF, sv_id & 0xFF, cno & 0xFF, elev & 0xFF])
        + (azim & 0xFFFF).to_bytes(2, "little")
        + (pr_res & 0xFFFF).to_bytes(2, "little")
        + (flags & 0xFFFFFFFF).to_bytes(4, "little")
    )


def _nav_sat_payload(*svs: bytes, itow: int = 123_456_000) -> bytes:
    num = len(svs)
    return (itow.to_bytes(4, "little") + bytes([1, num, 0, 0])
            + b"".join(svs))


# ----------------------------------------------------------------------
# ubx_scan
# ----------------------------------------------------------------------
def test_ubx_scan_roundtrip_and_garbage() -> None:
    f1 = ublox.ubx_frame(0x01, 0x35, b"\x01\x02\x03")
    f2 = ublox.ubx_frame(0x0A, 0x04, b"\xAA")
    buf = b"\x00\x99garbage" + f1 + f2
    frames, remainder = ublox.ubx_scan(buf)
    assert [(c, i) for c, i, _ in frames] == [(0x01, 0x35), (0x0A, 0x04)]
    assert frames[0][2] == f1[6:-2]
    assert frames[1][2] == b"\xAA"
    assert remainder == b""


def test_ubx_scan_bad_checksum_resync() -> None:
    bad = bytearray(ublox.ubx_frame(0x01, 0x35, b"\x11\x22"))
    bad[-1] ^= 0xFF  # corrupt CK_B
    good = ublox.ubx_frame(0x06, 0x3E, b"\x00\x01")
    frames, remainder = ublox.ubx_scan(bytes(bad) + good)
    assert len(frames) == 1
    assert frames[0][:2] == (0x06, 0x3E)
    assert remainder == b""


def test_ubx_scan_incomplete_remainder() -> None:
    frame = ublox.ubx_frame(0x01, 0x35, bytes(range(20)))
    frames, remainder = ublox.ubx_scan(frame[:9])
    assert frames == []
    assert remainder == frame[:9]
    frames2, remainder2 = ublox.ubx_scan(remainder + frame[9:])
    assert len(frames2) == 1
    assert frames2[0][2] == bytes(range(20))
    assert remainder2 == b""

    # a dangling sync byte must survive for the next read
    frames3, remainder3 = ublox.ubx_scan(b"\xb5")
    assert frames3 == []
    assert remainder3 == b"\xb5"

    # no sync at all: all bytes consumed as garbage
    frames4, remainder4 = ublox.ubx_scan(b"not-a-frame")
    assert frames4 == []
    assert remainder4 == b""


# ----------------------------------------------------------------------
# UBX-MON-VER
# ----------------------------------------------------------------------
def test_parse_mon_ver_extensions() -> None:
    payload = (
        _field("EXT CORE 4.00 (b41d34)", 30)
        + _field("00190000", 10)
        + _field("ROM BASE 0x118B2060")
        + _field("FWVER=HPG 1.32")
        + _field("PROTVER=27.32")
        + _field("MOD=ZED-F9P")
        + _field("GPS;GLO;GAL;BDS")
        + _field("SBAS;QZSS")
    )
    info = ublox.parse_mon_ver(payload)
    assert info["sw"].startswith("EXT CORE 4.00")
    assert info["hw"] == "00190000"
    assert "ROM BASE 0x118B2060" in info["extensions"]
    assert info["fwver"] == "HPG 1.32"
    assert info["protver"] == "27.32"
    assert info["mod"] == "ZED-F9P"

    # and through a real frame + dispatcher
    frames, _ = ublox.ubx_scan(ublox.ubx_frame(0x0A, 0x04, payload))
    assert len(frames) == 1
    parsed = ublox.parse_ubx_message(*frames[0])
    assert parsed["mod"] == "ZED-F9P"


# ----------------------------------------------------------------------
# UBX-CFG-GNSS
# ----------------------------------------------------------------------
def _gnss_block(gnss_id: int, res: int, mx: int, flags: int) -> bytes:
    return bytes([gnss_id, res, mx, 0]) + flags.to_bytes(4, "little")


def test_parse_cfg_gnss_galileo_enabled() -> None:
    payload = (
        bytes([0, 32, 32, 2])
        + _gnss_block(0, 8, 16, 0x00010001)   # GPS L1C/A + enable
        + _gnss_block(2, 8, 12, 0x00010001)   # Galileo E1 + enable
    )
    blocks = ublox.parse_cfg_gnss(payload)
    assert len(blocks) == 2
    gps, gal = blocks
    assert gps["system"] == "G" and gps["enable"] is True
    assert "GPS L1C/A" in gps["signals"]
    assert gal["gnssId"] == 2 and gal["system"] == "E"
    assert gal["enable"] is True
    assert gal["sigCfMask"] == 0x01
    assert gal["signals"] == ["Galileo E1"]

    msg = ublox.parse_cfg_gnss_msg(payload)
    assert msg["numConfigBlocks"] == 2
    assert msg["numTrkChHw"] == 32 and msg["numTrkChUse"] == 32
    assert msg["blocks"][1]["gnssId"] == 2


def test_parse_cfg_gnss_disabled_and_multibit_mask() -> None:
    payload = bytes([0, 32, 32, 2]) + _gnss_block(
        5, 3, 4, 0x00050000) + _gnss_block(2, 0, 0, 0x00000000)
    blocks = ublox.parse_cfg_gnss(payload)
    assert blocks[0]["enable"] is False           # enable bit0 clear
    assert blocks[0]["signals"] == ["QZSS L1C/A", "QZSS L1S"]
    assert blocks[1]["gnssId"] == 2
    assert blocks[1]["enable"] is False
    assert blocks[1]["signals"] == []


# ----------------------------------------------------------------------
# UBX-NAV-SAT
# ----------------------------------------------------------------------
def test_parse_nav_sat_galileo_block() -> None:
    flags = 0x808  # svUsed (bit 3) + ephAvail (bit 11)
    payload = _nav_sat_payload(_sv(2, 11, 45, 60, 180, 5, flags),
                               itow=900_000_000)
    blocks = ublox.parse_nav_sat(payload)
    assert len(blocks) == 1
    b = blocks[0]
    assert b["version"] == 1
    assert b["iTOW"] == 900_000_000
    assert b["gnssId"] == 2 and b["system"] == "E" and b["svId"] == 11
    assert b["cno"] == 45
    assert b["elev"] == 60
    assert b["azim"] == 180
    assert b["prRes"] == 5 and abs(b["prResM"] - 0.5) < 1e-9
    assert b["quality"] == 0
    assert b["svUsed"] is True
    assert b["health"] == 0
    assert b["ephAvail"] is True
    assert b["almAvail"] is False
    assert b["diffCorr"] is False and b["smoothed"] is False


def test_parse_nav_sat_two_svs_stride_and_signs() -> None:
    payload = _nav_sat_payload(
        _sv(2, 11, 44, -30, 270, -7, 0x808),
        _sv(0, 5, 40, 25, 45, 12, 0x00000008),
    )
    blocks = ublox.parse_nav_sat(payload)
    assert len(blocks) == 2
    a, b = blocks
    assert a["elev"] == -30 and a["prRes"] == -7
    assert abs(a["prResM"] + 0.7) < 1e-9
    assert b["gnssId"] == 0 and b["svId"] == 5
    assert b["svUsed"] is True and b["ephAvail"] is False


def test_nav_sat_flag_bit_positions() -> None:
    def flags_of(value: int) -> dict:
        return ublox.parse_nav_sat(
            _nav_sat_payload(_sv(2, 1, 40, 45, 90, 0, value)))[0]

    for q in range(8):
        assert flags_of(q)["quality"] == q
    assert flags_of(0x08)["svUsed"] is True
    assert flags_of(0x10)["health"] == 1
    assert flags_of(0x20)["health"] == 2
    assert flags_of(0x40)["diffCorr"] is True
    assert flags_of(0x80)["smoothed"] is True
    assert flags_of(0x100)["orbitSource"] == 1
    assert flags_of(0x700)["orbitSource"] == 7

    eph = flags_of(0x800)
    assert eph["ephAvail"] is True and eph["almAvail"] is False
    assert flags_of(0x1000)["ephAvail"] is False
    assert flags_of(0x1000)["almAvail"] is True
    assert flags_of(0x2000)["anoAvail"] is True
    assert flags_of(0x4000)["aopAvail"] is True

    # Cross-check the diagnostic's "bits 12..14": those are alm/ano/aop,
    # so ephAvail must stay False for a 0x7000 flags value.
    diag = flags_of(0x7000)
    assert diag["ephAvail"] is False
    assert diag["almAvail"] and diag["anoAvail"] and diag["aopAvail"]


def test_nav_sat_wrong_i4_layout_would_misparse() -> None:
    """Regression: reading prRes as I4 (NAV-SVINFO-like) shifts flags."""
    flags = 0x808
    sv = _sv(2, 11, 45, 60, 180, 5, flags)
    correct = ublox.parse_nav_sat(_nav_sat_payload(sv))[0]
    assert correct["svUsed"] is True and correct["ephAvail"] is True

    # Wrong layout: prRes as I4 (bytes 6..10) -> flags at bytes 10..14.
    wrong_flags = int.from_bytes(sv[10:14], "little")
    assert (wrong_flags & 0x08) == 0     # false svUsed = 0
    assert (wrong_flags & 0x800) == 0    # false ephAvail = 0


def test_nav_sat_helpers_and_dispatcher() -> None:
    payload = _nav_sat_payload(
        _sv(0, 3, 40, 20, 10, 0, 0x08),
        _sv(2, 11, 44, 60, 180, 3, 0x808),
        _sv(6, 7, 0, 5, 300, -2, 0x0000),
    )
    blocks = ublox.parse_nav_sat(payload)
    used = ublox.nav_sat_used(blocks)
    assert [b["svId"] for b in used] == [3, 11]
    grouped = ublox.nav_sat_by_system(blocks)
    assert set(grouped) == {"G", "E", "R"}
    assert grouped["E"][0]["svId"] == 11

    frame = ublox.ubx_frame(0x01, 0x35, payload)
    frames, _ = ublox.ubx_scan(frame)
    assert len(ublox.parse_ubx_message(*frames[0])) == 3
    assert ublox.parse_ubx_message(0x01, 0x99, payload) is None


def test_nav_sat_short_payload_is_safe() -> None:
    assert ublox.parse_nav_sat(b"") == []
    assert ublox.parse_nav_sat(b"\x01\x02\x03") == []
    # numSvs larger than the data must not crash or fabricate blocks
    payload = (0).to_bytes(4, "little") + bytes([1, 9, 0, 0]) + _sv(
        2, 1, 40, 45, 90, 0, 0x808)
    assert len(ublox.parse_nav_sat(payload)) == 1


# ----------------------------------------------------------------------
# UbxStream convenience entry
# ----------------------------------------------------------------------
def test_ubx_stream_incremental() -> None:
    payload = _nav_sat_payload(_sv(2, 11, 45, 60, 180, 5, 0x808))
    frame = ublox.ubx_frame(0x01, 0x35, payload)
    stream = ublox.UbxStream()
    assert stream.feed(frame[:7]) == []
    assert stream.remainder() == frame[:7]
    frames = stream.feed(frame[7:])
    assert len(frames) == 1
    assert frames[0][:2] == (0x01, 0x35)
    assert stream.remainder() == b""
    stream.reset()
    assert stream.remainder() == b""


# ----------------------------------------------------------------------
# cold start / NMEA untouched
# ----------------------------------------------------------------------
def test_cold_start_frame_unchanged() -> None:
    assert ublox.ubx_cfg_rst_frame() == bytes.fromhex(
        "B5 62 06 04 04 00 FF FF 02 00 0E 61")


def test_nmea_reader_still_available() -> None:
    assert ublox.parse_nmea_line(
        "$GNGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*47"
    )["type"] == "GGA"
    assert hasattr(ublox, "NmeaReader")
