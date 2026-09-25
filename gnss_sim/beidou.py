"""BeiDou B1I ranging codes (BDS-SIS-ICD-2.0).

B1I: carrier 1561.098 MHz, BPSK(2) at 2.046 Mcps, 2046-chip truncated Gold code
(two 11-stage LFSRs G1/G2).  Data (D1 50 bps / D2 500 bps, BCH+interleave+NH) is
not generated here yet - the engine transmits a placeholder.  PRN 1..37 are
tabulated; :func:`self_test` checks the published first-32-chip words.
"""

from __future__ import annotations

import numpy as np

B1I_CODE_CHIPS = 2046
B1I_CODE_RATE = 2_046_000.0
CARR_FREQ_B1I = 1561.098e6

# PRN -> (G2 tap 1, G2 tap 2), 1-based, from BDS-SIS-ICD-2.0 Table 5-1.
_B1I_TAPS = {
    1: (1, 3), 2: (1, 4), 3: (1, 5), 4: (1, 6), 5: (1, 8), 6: (1, 9),
    7: (1, 10), 8: (1, 11), 9: (2, 7), 10: (3, 4), 11: (3, 5), 12: (3, 6),
    13: (3, 8), 14: (3, 9), 15: (3, 10), 16: (3, 11), 17: (4, 5), 18: (4, 6),
    19: (4, 8), 20: (4, 9), 21: (4, 10), 22: (4, 11), 23: (5, 6), 24: (5, 8),
    25: (5, 9), 26: (5, 10), 27: (5, 11), 28: (6, 8), 29: (6, 9), 30: (6, 10),
    31: (6, 11), 32: (8, 9), 33: (8, 10), 34: (8, 11), 35: (9, 10),
    36: (9, 11), 37: (10, 11),
}

# Published first 32 chips of PRN 1..8 (reference catalogue, 0/1 words).
_FIRST32 = {
    1: "01100101101101101100110111010100",
    2: "10010010011000100011100000011010",
    3: "01101001100010000100001011111101",
    4: "10010100011111010111111110001110",
    5: "10010101111110101010111001101011",
    6: "01101010010001000000100111000101",
    7: "10010101100110110101101000010010",
    8: "01101010011101001111001111111001",
}


def generate_b1i(prn: int) -> np.ndarray:
    """Return one 2046-chip B1I code as +1/-1 int8 (period repeats)."""
    if prn not in _B1I_TAPS:
        raise ValueError(f"B1I PRN {prn} not tabulated (1..37)")
    t1, t2 = (t - 1 for t in _B1I_TAPS[prn])
    g1 = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    g2 = list(g1)
    code = np.empty(B1I_CODE_CHIPS, dtype=np.int8)
    for i in range(B1I_CODE_CHIPS):
        chip = g1[10] ^ g2[t1] ^ g2[t2]
        code[i] = 1 if chip == 0 else -1
        fb1 = g1[0] ^ g1[6] ^ g1[7] ^ g1[8] ^ g1[9] ^ g1[10]
        fb2 = (g2[0] ^ g2[1] ^ g2[2] ^ g2[3] ^ g2[4]
               ^ g2[7] ^ g2[8] ^ g2[10])
        for j in range(10, 0, -1):
            g1[j] = g1[j - 1]
            g2[j] = g2[j - 1]
        g1[0] = fb1
        g2[0] = fb2
    return code


def self_test() -> None:
    for prn, bits in _FIRST32.items():
        code = generate_b1i(prn)
        if len(code) != B1I_CODE_CHIPS:
            raise AssertionError("B1I code length")
        got = "".join("0" if c > 0 else "1" for c in code[:32])
        if got != bits:
            raise AssertionError(
                f"B1I PRN {prn} prefix mismatch: {got} != {bits}")
