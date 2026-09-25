"""GPS L1 C/A (legacy NAV) navigation message construction.

Ports ``eph2sbf`` / ``computeChecksum`` / ``generateNavMsg`` from gps-sdr-sim.
Data words are kept as Python ints masked to 30 bits (the two MSBs are the parity
bits of the previous word and are added during generation).
"""

from __future__ import annotations

import numpy as np

from .constants import (
    N_DWRD, N_DWRD_SBF, N_SBF,
    PI, POW2_M5, POW2_M19, POW2_M24, POW2_M27, POW2_M29, POW2_M30,
    POW2_M31, POW2_M33, POW2_M43, POW2_M50, POW2_M55,
)
from .gpstime import GpsTime
from .rinex import Ephemeris, IonoUtc

_MASK30 = 0x3FFFFFFF
_BMASK = (0x3B1F3480, 0x1D8F9A40, 0x2EC7CD00,
          0x1763E680, 0x2BB1F340, 0x0B7A89C0)


def count_bits(v: int) -> int:
    return bin(v & 0xFFFFFFFF).count("1")


def compute_checksum(source: int, nib: int) -> int:
    source &= 0xFFFFFFFF
    d = source & 0x3FFFFFC0
    d29 = (source >> 31) & 1
    d30 = (source >> 30) & 1

    if nib:
        if (d30 + count_bits(_BMASK[4] & d)) % 2:
            d ^= (1 << 6)
        if (d29 + count_bits(_BMASK[5] & d)) % 2:
            d ^= (1 << 7)

    D = d
    if d30:
        D ^= 0x3FFFFFC0

    D |= ((d29 + count_bits(_BMASK[0] & d)) % 2) << 5
    D |= ((d30 + count_bits(_BMASK[1] & d)) % 2) << 4
    D |= ((d29 + count_bits(_BMASK[2] & d)) % 2) << 3
    D |= ((d30 + count_bits(_BMASK[3] & d)) % 2) << 2
    D |= ((d30 + count_bits(_BMASK[4] & d)) % 2) << 1
    D |= ((d29 + count_bits(_BMASK[5] & d)) % 2)
    return D & _MASK30


def eph2sbf(eph: Ephemeris, iono: IonoUtc,
            transmit_week: int | None = None) -> np.ndarray:
    """Build the five 10-word subframes (30-bit words, MSBs zero).

    ``transmit_week`` is the week number placed in subframe 1 (mod 1024).  When
    ``None`` the ephemeris week is used.
    """
    wn = (eph.toe.week % 1024) if transmit_week is None else (transmit_week % 1024)
    toe = int(eph.toe.sec / 16.0)
    toc = int(eph.toc.sec / 16.0)
    iode = int(eph.iode)
    iodc = int(eph.iodc)
    deltan = int(eph.deltan / POW2_M43 / PI)
    cuc = int(eph.cuc / POW2_M29)
    cus = int(eph.cus / POW2_M29)
    cic = int(eph.cic / POW2_M29)
    cis = int(eph.cis / POW2_M29)
    crc = int(eph.crc / POW2_M5)
    crs = int(eph.crs / POW2_M5)
    ecc = int(eph.ecc / POW2_M33)
    sqrta = int(eph.sqrta / POW2_M19)
    m0 = int(eph.m0 / POW2_M31 / PI)
    omg0 = int(eph.omg0 / POW2_M31 / PI)
    inc0 = int(eph.inc0 / POW2_M31 / PI)
    aop = int(eph.aop / POW2_M31 / PI)
    omgdot = int(eph.omgdot / POW2_M43 / PI)
    idot = int(eph.idot / POW2_M43 / PI)
    af0 = int(eph.af0 / POW2_M31)
    af1 = int(eph.af1 / POW2_M43)
    af2 = int(eph.af2 / POW2_M55)
    tgd = int(eph.tgd / POW2_M31)
    svhlth = int(eph.svhlth)
    codeL2 = int(eph.codeL2)
    ura = 0
    dataId = 1
    sbf4_page25_svId = 63
    sbf5_page25_svId = 51
    sbf4_page18_svId = 56
    wna = eph.toe.week % 256
    toa = int(eph.toe.sec / 4096.0)

    alpha0 = round(iono.alpha0 / POW2_M30)
    alpha1 = round(iono.alpha1 / POW2_M27)
    alpha2 = round(iono.alpha2 / POW2_M24)
    alpha3 = round(iono.alpha3 / POW2_M24)
    beta0 = round(iono.beta0 / 2048.0)
    beta1 = round(iono.beta1 / 16384.0)
    beta2 = round(iono.beta2 / 65536.0)
    beta3 = round(iono.beta3 / 65536.0)
    A0 = round(iono.A0 / POW2_M30)
    A1 = round(iono.A1 / POW2_M50)
    dtls = int(iono.dtls)
    tot = int(iono.tot / 4096)
    wnt = int(iono.wnt % 256)
    wnlsf = 1929 % 256
    dn = 7
    dtlsf = 18

    sbf = np.zeros((5, N_DWRD_SBF), dtype=np.int64)

    # Subframe 1
    sbf[0][0] = 0x8B0000 << 6
    sbf[0][1] = 0x1 << 8
    sbf[0][2] = (((wn & 0x3FF) << 20) | ((codeL2 & 0x3) << 18)
                 | ((ura & 0xF) << 14) | ((svhlth & 0x3F) << 8)
                 | (((iodc >> 8) & 0x3) << 6))
    sbf[0][6] = (tgd & 0xFF) << 6
    sbf[0][7] = ((iodc & 0xFF) << 22) | ((toc & 0xFFFF) << 6)
    sbf[0][8] = ((af2 & 0xFF) << 22) | ((af1 & 0xFFFF) << 6)
    sbf[0][9] = (af0 & 0x3FFFFF) << 8

    # Subframe 2
    sbf[1][0] = 0x8B0000 << 6
    sbf[1][1] = 0x2 << 8
    sbf[1][2] = ((iode & 0xFF) << 22) | ((crs & 0xFFFF) << 6)
    sbf[1][3] = ((deltan & 0xFFFF) << 14) | (((m0 >> 24) & 0xFF) << 6)
    sbf[1][4] = (m0 & 0xFFFFFF) << 6
    sbf[1][5] = ((cuc & 0xFFFF) << 14) | (((ecc >> 24) & 0xFF) << 6)
    sbf[1][6] = (ecc & 0xFFFFFF) << 6
    sbf[1][7] = ((cus & 0xFFFF) << 14) | (((sqrta >> 24) & 0xFF) << 6)
    sbf[1][8] = (sqrta & 0xFFFFFF) << 6
    sbf[1][9] = (toe & 0xFFFF) << 14

    # Subframe 3
    sbf[2][0] = 0x8B0000 << 6
    sbf[2][1] = 0x3 << 8
    sbf[2][2] = ((cic & 0xFFFF) << 14) | (((omg0 >> 24) & 0xFF) << 6)
    sbf[2][3] = (omg0 & 0xFFFFFF) << 6
    sbf[2][4] = ((cis & 0xFFFF) << 14) | (((inc0 >> 24) & 0xFF) << 6)
    sbf[2][5] = (inc0 & 0xFFFFFF) << 6
    sbf[2][6] = ((crc & 0xFFFF) << 14) | (((aop >> 24) & 0xFF) << 6)
    sbf[2][7] = (aop & 0xFFFFFF) << 6
    sbf[2][8] = (omgdot & 0xFFFFFF) << 6
    sbf[2][9] = ((iode & 0xFF) << 22) | ((idot & 0x3FFF) << 8)

    if iono.vflg:
        sbf[3][0] = 0x8B0000 << 6
        sbf[3][1] = 0x4 << 8
        sbf[3][2] = ((dataId << 28) | (sbf4_page18_svId << 22)
                     | ((alpha0 & 0xFF) << 14) | ((alpha1 & 0xFF) << 6))
        sbf[3][3] = (((alpha2 & 0xFF) << 22) | ((alpha3 & 0xFF) << 14)
                     | ((beta0 & 0xFF) << 6))
        sbf[3][4] = (((beta1 & 0xFF) << 22) | ((beta2 & 0xFF) << 14)
                     | ((beta3 & 0xFF) << 6))
        sbf[3][5] = (A1 & 0xFFFFFF) << 6
        sbf[3][6] = ((A0 >> 8) & 0xFFFFFF) << 6
        sbf[3][7] = (((A0 & 0xFF) << 22) | ((tot & 0xFF) << 14)
                     | ((wnt & 0xFF) << 6))
        sbf[3][8] = (((dtls & 0xFF) << 22) | ((wnlsf & 0xFF) << 14)
                     | ((dn & 0xFF) << 6))
        sbf[3][9] = (dtlsf & 0xFF) << 22
    else:
        sbf[3][0] = 0x8B0000 << 6
        sbf[3][1] = 0x4 << 8
        sbf[3][2] = (dataId << 28) | (sbf4_page25_svId << 22)

    sbf[4][0] = 0x8B0000 << 6
    sbf[4][1] = 0x5 << 8
    sbf[4][2] = ((dataId << 28) | (sbf5_page25_svId << 22)
                 | ((toa & 0xFF) << 14) | ((wna & 0xFF) << 6))

    return sbf


def generate_nav_msg(g: GpsTime, sbf: np.ndarray, dwrd: np.ndarray,
                     init: int) -> tuple[GpsTime, np.ndarray]:
    """Fill ``dwrd`` (length 60) with parity-protected 30-bit words.

    ``dwrd[0:10]``  = subframe 5 of the previous frame (or generated here when
    ``init == 1``); ``dwrd[(isbf+1)*10 : ...]`` = subframes 1..5.
    Returns the aligned data-bit reference time ``g0``.
    """
    g0 = GpsTime(g.week, float((int(g.sec + 0.5) // 30) * 30))
    tow = int(g0.sec) // 6

    prevwrd = 0
    if init == 1:
        for iwrd in range(N_DWRD_SBF):
            sbfwrd = int(sbf[4][iwrd])
            if iwrd == 1:
                sbfwrd |= (tow & 0x1FFFF) << 13
            sbfwrd |= (prevwrd << 30) & 0xC0000000
            nib = 1 if (iwrd == 1 or iwrd == 9) else 0
            dwrd[iwrd] = compute_checksum(sbfwrd, nib)
            prevwrd = int(dwrd[iwrd])
    else:
        for iwrd in range(N_DWRD_SBF):
            dwrd[iwrd] = dwrd[N_DWRD_SBF * N_SBF + iwrd]
            prevwrd = int(dwrd[iwrd])

    for isbf in range(N_SBF):
        tow += 1
        for iwrd in range(N_DWRD_SBF):
            sbfwrd = int(sbf[isbf][iwrd])
            if isbf == 0 and iwrd == 2:
                sbfwrd |= (g0.week % 1024 & 0x3FF) << 20
            if iwrd == 1:
                sbfwrd |= (tow & 0x1FFFF) << 13
            sbfwrd |= (prevwrd << 30) & 0xC0000000
            nib = 1 if (iwrd == 1 or iwrd == 9) else 0
            dwrd[(isbf + 1) * N_DWRD_SBF + iwrd] = compute_checksum(sbfwrd, nib)
            prevwrd = int(dwrd[(isbf + 1) * N_DWRD_SBF + iwrd])

    return g0, dwrd
