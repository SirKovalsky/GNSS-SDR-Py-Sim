"""GPS L1 C/A (Gold) ranging codes."""

from __future__ import annotations

import numpy as np

CA_LEN = 1023

_G2_DELAY = np.array([
    5, 6, 7, 8, 17, 18, 139, 140, 141, 251, 252, 254, 255, 256, 257, 258,
    469, 470, 471, 472, 473, 474, 509, 512, 513, 514, 515, 516, 859, 860,
    861, 862,
], dtype=np.int64)

# SBAS PRN 120..158 and QZSS PRN 193..206 share the same G1/G2 Gold generator
# as GPS, only the G2 delay differs (gps.gov L1 C/A PRN assignments 2026 /
# IS-QZSS Table 3.2.2-2).
_EXTRA_G2_DELAY = {
    120: 145, 121: 175, 122: 52, 123: 21, 124: 237, 125: 235, 126: 886,
    127: 657, 128: 634, 129: 762, 130: 355, 131: 1012, 132: 176, 133: 603,
    134: 130, 135: 359, 136: 595, 137: 68, 138: 386, 139: 797, 140: 456,
    141: 499, 142: 883, 143: 307, 144: 127, 145: 211, 146: 121, 147: 118,
    148: 163, 149: 628, 150: 853, 151: 484, 152: 289, 153: 811, 154: 202,
    155: 1021, 156: 463, 157: 568, 158: 904,
    193: 339, 194: 208, 195: 711, 196: 189, 197: 263, 198: 537, 199: 663,
    200: 942, 201: 173, 202: 900, 203: 30, 204: 500, 205: 935, 206: 556,
}

# Published "First 10 chips" (octal, first chip = MSB) for self-testing.
_FIRST10_OCTAL = {
    120: 0o0671, 121: 0o0536, 122: 0o1510, 123: 0o1545, 124: 0o0160,
    125: 0o0701, 126: 0o0013, 127: 0o1060, 128: 0o0245, 129: 0o0527,
    130: 0o1436, 131: 0o1226, 132: 0o1257, 133: 0o0046, 134: 0o1071,
    135: 0o0561, 136: 0o1037, 137: 0o0770, 138: 0o1327,
    193: 0o0727, 194: 0o0170, 195: 0o0030, 196: 0o0472, 197: 0o1237,
    198: 0o0414, 199: 0o1050, 200: 0o1630, 201: 0o0571, 202: 0o0732,
    203: 0o1301, 204: 0o1173, 205: 0o0020, 206: 0o0447,
}


def _g2_delay(prn: int) -> int:
    if 1 <= prn <= 32:
        return int(_G2_DELAY[prn - 1])
    if prn in _EXTRA_G2_DELAY:
        return int(_EXTRA_G2_DELAY[prn])
    raise ValueError(f"C/A PRN {prn} not tabulated")


def generate_ca(prn: int) -> np.ndarray:
    """Return the 1023-chip C/A code (0/1).

    Supports GPS PRN 1..32, SBAS PRN 120..158 and QZSS PRN 193..206.
    """
    delay = _g2_delay(prn)

    r1 = np.full(10, -1, dtype=np.int64)
    r2 = np.full(10, -1, dtype=np.int64)
    g1 = np.empty(CA_LEN, dtype=np.int64)
    g2 = np.empty(CA_LEN, dtype=np.int64)

    for i in range(CA_LEN):
        g1[i] = r1[9]
        g2[i] = r2[9]
        c1 = r1[2] * r1[9]
        c2 = r2[1] * r2[2] * r2[5] * r2[7] * r2[8] * r2[9]
        r1[1:] = r1[:-1]
        r2[1:] = r2[:-1]
        r1[0] = c1
        r2[0] = c2

    idx = (CA_LEN - delay + np.arange(CA_LEN)) % CA_LEN
    return ((1 - g1 * g2[idx]) // 2).astype(np.int8)


def code_bipolar(prn: int) -> np.ndarray:
    """C/A code as +1/-1 chips."""
    return (generate_ca(prn).astype(np.int8) * 2 - 1).astype(np.int8)


def self_test() -> None:
    """Check the GPS, SBAS and QZSS C/A codes against published chip words."""
    gps = generate_ca(1)
    head = 0
    for i in range(20):
        head = (head << 1) | int(gps[i])
    if head != int("11001000001110010100", 2):
        raise AssertionError("GPS PRN1 first chips mismatch")
    for prn, oct_val in _FIRST10_OCTAL.items():
        code = generate_ca(prn)
        val = 0
        for i in range(10):
            val = (val << 1) | int(code[i])
        if val != oct_val:
            raise AssertionError(
                f"C/A code mismatch PRN {prn}: {val:04o} != {oct_val:04o}")

