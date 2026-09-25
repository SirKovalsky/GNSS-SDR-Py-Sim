"""GPS broadcast-ephemeris orbit / range geometry (ports gps-sdr-sim)."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .constants import (
    GM_EARTH, OMEGA_EARTH, PI, SPEED_OF_LIGHT,
    SECONDS_IN_HALF_WEEK, SECONDS_IN_WEEK,
    WGS84_E2, WGS84_ECCENTRICITY, WGS84_RADIUS,
)
from .gpstime import GpsTime, sub_gps_time
from .rinex import Ephemeris, IonoUtc


def xyz2llh(xyz):
    """ECEF -> (lat, lon, height) radians / metres."""
    x, y, z = xyz
    rho2 = x * x + y * y
    dz = WGS84_E2 * z
    for _ in range(10):
        zdz = z + dz
        nh = math.sqrt(rho2 + zdz * zdz)
        slat = zdz / nh
        n = WGS84_RADIUS / math.sqrt(1.0 - WGS84_E2 * slat * slat)
        dz_new = n * WGS84_E2 * slat
        if abs(dz - dz_new) < 1e-3:
            dz = dz_new
            break
        dz = dz_new
    zdz = z + dz
    nh = math.sqrt(rho2 + zdz * zdz)
    slat = zdz / nh
    n = WGS84_RADIUS / math.sqrt(1.0 - WGS84_E2 * slat * slat)
    lat = math.atan2(zdz, math.sqrt(rho2))
    lon = math.atan2(y, x)
    h = nh - n
    return [lat, lon, h]


def llh2xyz(lat, lon, h):
    slat = math.sin(lat)
    clat = math.cos(lat)
    slon = math.sin(lon)
    clon = math.cos(lon)
    n = WGS84_RADIUS / math.sqrt(1.0 - (WGS84_ECCENTRICITY * slat) ** 2)
    nph = n + h
    return [nph * clat * clon, nph * clat * slon,
            ((1.0 - WGS84_E2) * n + h) * slat]


def ltcmat(lat: float, lon: float):
    """Local tangent (N, E, U) rotation matrix rows."""
    slat, clat = math.sin(lat), math.cos(lat)
    slon, clon = math.sin(lon), math.cos(lon)
    return [
        [-slat * clon, -slat * slon, clat],
        [-slon, clon, 0.0],
        [clat * clon, clat * slon, slat],
    ]


def ecef2neu(xyz, tmat):
    return [sum(tmat[r][c] * xyz[c] for c in range(3)) for r in range(3)]


def neu2azel(neu):
    n, e, u = neu
    az = math.atan2(e, n)
    if az < 0.0:
        az += 2.0 * PI
    el = math.atan2(u, math.hypot(n, e))
    return [az, el]


def satpos(eph: Ephemeris, g: GpsTime):
    """Return (pos, vel, clk) for the satellite at GPS time ``g``."""
    tk = g.sec - eph.toe.sec
    if tk > SECONDS_IN_HALF_WEEK:
        tk -= SECONDS_IN_WEEK
    elif tk < -SECONDS_IN_HALF_WEEK:
        tk += SECONDS_IN_WEEK

    mk = eph.m0 + eph.n * tk
    ek = mk
    ekold = ek + 1.0
    OneMinusecosE = 0.0
    for _ in range(100):
        if abs(ek - ekold) <= 1e-14:
            break
        ekold = ek
        OneMinusecosE = 1.0 - eph.ecc * math.cos(ekold)
        ek = ek + (mk - ekold + eph.ecc * math.sin(ekold)) / OneMinusecosE

    sek, cek = math.sin(ek), math.cos(ek)
    ekdot = eph.n / OneMinusecosE
    relativistic = -4.442807633e-10 * eph.ecc * eph.sqrta * sek
    pk = math.atan2(eph.sq1e2 * sek, cek - eph.ecc) + eph.aop
    pkdot = eph.sq1e2 * ekdot / OneMinusecosE
    s2pk, c2pk = math.sin(2.0 * pk), math.cos(2.0 * pk)

    uk = pk + eph.cus * s2pk + eph.cuc * c2pk
    suk, cuk = math.sin(uk), math.cos(uk)
    ukdot = pkdot * (1.0 + 2.0 * (eph.cus * c2pk - eph.cuc * s2pk))

    rk = eph.A * OneMinusecosE + eph.crc * c2pk + eph.crs * s2pk
    rkdot = eph.A * eph.ecc * sek * ekdot + 2.0 * pkdot * (eph.crs * c2pk - eph.crc * s2pk)

    ik = eph.inc0 + eph.idot * tk + eph.cic * c2pk + eph.cis * s2pk
    sik, cik = math.sin(ik), math.cos(ik)
    ikdot = eph.idot + 2.0 * pkdot * (eph.cis * c2pk - eph.cic * s2pk)

    xpk, ypk = rk * cuk, rk * suk
    xpkdot = rkdot * cuk - ypk * ukdot
    ypkdot = rkdot * suk + xpk * ukdot

    ok = eph.omg0 + tk * eph.omgkdot - OMEGA_EARTH * eph.toe.sec
    sok, cok = math.sin(ok), math.cos(ok)

    pos = [xpk * cok - ypk * cik * sok,
           xpk * sok + ypk * cik * cok,
           ypk * sik]

    tmp = ypkdot * cik - ypk * sik * ikdot
    vel = [-eph.omgkdot * pos[1] + xpkdot * cok - tmp * sok,
           eph.omgkdot * pos[0] + xpkdot * sok + tmp * cok,
           ypk * cik * ikdot + ypkdot * sik]

    tk = g.sec - eph.toc.sec
    if tk > SECONDS_IN_HALF_WEEK:
        tk -= SECONDS_IN_WEEK
    elif tk < -SECONDS_IN_HALF_WEEK:
        tk += SECONDS_IN_WEEK
    clk = [eph.af0 + tk * (eph.af1 + tk * eph.af2) + relativistic - eph.tgd,
           eph.af1 + 2.0 * tk * eph.af2]
    return pos, vel, clk


def ionospheric_delay(iono: IonoUtc, g: GpsTime, llh, azel) -> float:
    if not iono.enable:
        return 0.0
    E = azel[1] / PI
    phi_u = llh[0] / PI
    lam_u = llh[1] / PI
    F = 1.0 + 16.0 * (0.53 - E) ** 3

    if not iono.vflg:
        return F * 5.0e-9 * SPEED_OF_LIGHT

    psi = 0.0137 / (E + 0.11) - 0.022
    phi_i = phi_u + psi * math.cos(azel[0])
    phi_i = max(-0.416, min(0.416, phi_i))
    lam_i = lam_u + psi * math.sin(azel[0]) / math.cos(phi_i * PI)
    phi_m = phi_i + 0.064 * math.cos((lam_i - 1.617) * PI)

    amp = (iono.alpha0 + iono.alpha1 * phi_m + iono.alpha2 * phi_m ** 2
           + iono.alpha3 * phi_m ** 3)
    amp = max(0.0, amp)
    per = (iono.beta0 + iono.beta1 * phi_m + iono.beta2 * phi_m ** 2
           + iono.beta3 * phi_m ** 3)
    per = max(72000.0, per)

    t = (43200.0 * lam_i + g.sec) % 86400.0
    x = 2.0 * PI * (t - 50400.0) / per
    if abs(x) < 1.57:
        delay = 5.0e-9 + amp * (1.0 - x * x / 2.0 + x ** 4 / 24.0)
    else:
        delay = 5.0e-9
    return F * delay * SPEED_OF_LIGHT


@dataclass
class Range:
    g: GpsTime
    range: float = 0.0      # pseudorange incl. iono (m)
    rate: float = 0.0       # pseudorange rate (m/s)
    d: float = 0.0          # geometric range (m)
    azel: list = None       # [az, el] rad
    iono_delay: float = 0.0


def compute_range(eph: Ephemeris, iono: IonoUtc, g: GpsTime, xyz) -> Range:
    pos, vel, clk = satpos(eph, g)
    los = [pos[i] - xyz[i] for i in range(3)]
    tau = math.sqrt(sum(v * v for v in los)) / SPEED_OF_LIGHT

    pos[0] -= vel[0] * tau
    pos[1] -= vel[1] * tau
    pos[2] -= vel[2] * tau
    xrot = pos[0] + pos[1] * OMEGA_EARTH * tau
    yrot = pos[1] - pos[0] * OMEGA_EARTH * tau
    pos[0], pos[1] = xrot, yrot

    los = [pos[i] - xyz[i] for i in range(3)]
    rng = math.sqrt(sum(v * v for v in los))
    rate = sum(vel[i] * los[i] for i in range(3)) / rng

    llh = xyz2llh(xyz)
    tmat = ltcmat(llh[0], llh[1])
    neu = ecef2neu(los, tmat)
    azel = neu2azel(neu)
    iono_delay = ionospheric_delay(iono, g, llh, azel)

    return Range(g, rng - SPEED_OF_LIGHT * clk[0] + iono_delay, rate,
                 rng, azel, iono_delay)
