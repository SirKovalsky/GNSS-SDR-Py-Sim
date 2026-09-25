"""Physical and GPS constants (mirrors gps-sdr-sim / IS-GPS-200)."""

from __future__ import annotations

import numpy as np

# ---- Mathematical ---------------------------------------------------------
PI = 3.1415926535898
R2D = 57.2957795131
TWO_PI = 2.0 * np.pi

# ---- GPS / WGS-84 ---------------------------------------------------------
GM_EARTH = 3.986005e14          # m^3/s^2
OMEGA_EARTH = 7.2921151467e-5   # rad/s
WGS84_RADIUS = 6378137.0        # m
WGS84_ECCENTRICITY = 0.0818191908426
WGS84_E2 = WGS84_ECCENTRICITY ** 2

SPEED_OF_LIGHT = 2.99792458e8   # m/s

# ---- GPS L1 ---------------------------------------------------------------
CARR_FREQ_L1 = 1575.42e6        # Hz (C/A and L1C share the carrier)
CODE_FREQ_CA = 1.023e6          # Hz - C/A chip rate
CODE_FREQ_L1C = 1.023e6         # Hz - L1C chip rate
LAMBDA_L1 = 0.190293672798365   # m - C/A wavelength (c / 1575.42e6)
CARR_TO_CODE = 1.0 / 1540.0

# L1C modulation
L1C_CODE_LEN = 10230            # chips per 10 ms
L1C_CODE_PERIOD = 0.010         # s
L1C_SUBFRAME_SYMBOLS = 100      # symbols/s on L1Cd
L1C_FRAME_SYMBOLS = 1800        # symbols per 18 s frame
L1C_FRAME_SECONDS = 18.0
L1C_PILOT_POWER = 0.75          # pilot fraction of L1C power
L1C_DATA_POWER = 0.25

# C/A
CA_SEQ_LEN = 1023

# ---- Time -----------------------------------------------------------------
SECONDS_IN_WEEK = 604800.0
SECONDS_IN_HALF_WEEK = 302400.0
SECONDS_IN_DAY = 86400.0

# ---- BeiDou time (BDT) ----------------------------------------------------
#: BDT = GPST - 14 s and BDT week = GPS week - 1356 (BDT epoch 2006-01-01).
BDT_GPST_OFFSET_S = 14.0
BDT_WEEK_OFFSET = 1356

# ---- Ephemeris / nav message ---------------------------------------------
MAX_SAT = 32
MAX_CHAN = 16
N_SBF = 5
N_DWRD_SBF = 10
N_DWRD = (N_SBF + 1) * N_DWRD_SBF
EPHEM_ARRAY_SIZE = 15

# Powers of two used when packing broadcast ephemeris
POW2_M5 = 0.03125
POW2_M19 = 1.9073486328125e-6
POW2_M29 = 1.862645149230957e-9
POW2_M31 = 4.656612873077393e-10
POW2_M33 = 1.164153218269348e-10
POW2_M43 = 1.136868377216160e-13
POW2_M55 = 2.775557561562891e-17
POW2_M30 = 9.313225746154785e-10
POW2_M27 = 7.450580596923828e-9
POW2_M24 = 5.960464477539063e-8
POW2_M50 = 8.881784197001252e-16

# ---- IQ output formats ----------------------------------------------------
SC01 = 1
SC08 = 8
SC16 = 16
