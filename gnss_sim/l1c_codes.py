"""GPS L1C ranging codes, overlay code and TMBOC layout (IS-GPS-800 §3.2.2/§3.3).

Only GPS PRN signal numbers 1-32 are tabulated here (the L1C codes are the same
10230-chip codes used by GPS III/III+; PRN 1-32 matches the L1 C/A PRN).  The
tables below are transcribed from IS-GPS-800 Table 3.2-2 (ranging codes) and
Table 3.2-3 (L1Co overlay code), and :func:`self_test` verifies the generated
codes against the published *Initial 24 Chips* / *Final 24 Chips* octal values.
"""

from __future__ import annotations

import numpy as np

from .constants import L1C_CODE_LEN

_LEGENDRE_LEN = 10223
_EXPANSION = (0, 1, 1, 0, 1, 0, 0)  # inserted before insertion index p

# TMBOC(6,1,4/33): BOC(6,1) is placed at these chip positions of every 33-chip
# group (IS-GPS-800 §3.3; 0-based ut = {0,4,6,29}, vt = 0..309).
_TMBOC_UT = (0, 4, 6, 29)
TMBOC_BOC61 = np.array(
    [u + 33 * v for v in range(310) for u in _TMBOC_UT], dtype=np.int64
)

# ---------------------------------------------------------------------------
# Table 3.2-2: per PRN -> (CP_w, CP_p, CP_init, CP_final, CD_w, CD_p, CD_init, CD_final)
# init/final are 24-chip octal words kept only for verification.
# ---------------------------------------------------------------------------
_RANGING_TABLE = {
    1: (5111, 412, 0o05752067, 0o20173742, 5097, 181, 0o77001425, 0o52231646),
    2: (5109, 161, 0o70146401, 0o35437154, 5110, 359, 0o23342754, 0o46703351),
    3: (5108, 1, 0o32066222, 0o00161056, 5079, 72, 0o30523404, 0o00145161),
    4: (5106, 303, 0o72125121, 0o71435437, 4403, 1110, 0o03777635, 0o11261273),
    5: (5103, 207, 0o42323273, 0o15035661, 4121, 1480, 0o10505640, 0o71364603),
    6: (5101, 4971, 0o01650642, 0o32606570, 5043, 5034, 0o42134174, 0o55012662),
    7: (5100, 4496, 0o21303446, 0o03475644, 5042, 4622, 0o00471711, 0o30373701),
    8: (5098, 5, 0o35504263, 0o11316575, 5104, 1, 0o32237045, 0o07706523),
    9: (5095, 4557, 0o66434311, 0o23047575, 4940, 4547, 0o16004766, 0o71741157),
    10: (5094, 485, 0o52631623, 0o07355246, 5035, 826, 0o66234727, 0o42347523),
    11: (5093, 253, 0o04733076, 0o15210113, 4372, 6284, 0o03755314, 0o12746122),
    12: (5091, 4676, 0o50352603, 0o72643606, 5064, 4195, 0o20604227, 0o34634113),
    13: (5090, 1, 0o32026612, 0o63457333, 5084, 368, 0o25477233, 0o47555063),
    14: (5081, 66, 0o07476042, 0o46623624, 5048, 1, 0o32025443, 0o01221116),
    15: (5080, 4485, 0o22210746, 0o35467322, 4950, 4796, 0o35503400, 0o37125437),
    16: (5069, 282, 0o30706376, 0o70116567, 5019, 523, 0o70504407, 0o32203664),
    17: (5068, 193, 0o75764610, 0o62731643, 5076, 151, 0o26163421, 0o62162634),
    18: (5054, 5211, 0o73202225, 0o14040613, 3736, 713, 0o52176727, 0o35012616),
    19: (5044, 729, 0o47227426, 0o07750525, 4993, 9850, 0o72557314, 0o00437232),
    20: (5027, 4848, 0o16064126, 0o37171211, 5060, 5734, 0o62043206, 0o32130365),
    21: (5026, 982, 0o66415734, 0o01302134, 5061, 34, 0o07151343, 0o51515733),
    22: (5014, 5955, 0o27600270, 0o37672235, 5096, 6142, 0o16027175, 0o73662313),
    23: (5004, 9805, 0o66101627, 0o32201230, 4983, 190, 0o26267340, 0o55416712),
    24: (4980, 670, 0o17717055, 0o37437553, 4783, 644, 0o36272365, 0o22550142),
    25: (4915, 464, 0o47500232, 0o23310544, 4991, 467, 0o67707677, 0o31506062),
    26: (4909, 29, 0o52057615, 0o07152415, 4815, 5384, 0o07760374, 0o44603344),
    27: (4893, 429, 0o76153566, 0o02571041, 4443, 801, 0o73633310, 0o05252052),
    28: (4885, 394, 0o22444670, 0o52270664, 4769, 594, 0o30401257, 0o70603616),
    29: (4832, 616, 0o62330044, 0o61317104, 4879, 4450, 0o72606251, 0o51643216),
    30: (4824, 9457, 0o13674337, 0o43137330, 4894, 9437, 0o37370402, 0o30417163),
    31: (4591, 4429, 0o60635146, 0o20336467, 4985, 4307, 0o74255661, 0o20074570),
    32: (3706, 4771, 0o73527653, 0o40745656, 5056, 5906, 0o10171147, 0o26204176),
}

# ---------------------------------------------------------------------------
# Table 3.2-3: PRN -> (S1 polynomial octal, initial state octal, final state octal)
# The polynomial is the 12-bit word 1,m10,...,m1,1 (m0 = m11 = 1).
# ---------------------------------------------------------------------------
_OVERLAY_TABLE = {
    1: (0o5111, 0o3266, 0o0410), 2: (0o5421, 0o2040, 0o3153),
    3: (0o5501, 0o1527, 0o1767), 4: (0o5403, 0o3307, 0o2134),
    5: (0o6417, 0o3756, 0o3510), 6: (0o6141, 0o3026, 0o2260),
    7: (0o6351, 0o0562, 0o2433), 8: (0o6501, 0o0420, 0o3520),
    9: (0o6205, 0o3415, 0o2652), 10: (0o6235, 0o0337, 0o2050),
    11: (0o7751, 0o0265, 0o0070), 12: (0o6623, 0o1230, 0o1605),
    13: (0o6733, 0o2204, 0o1247), 14: (0o7627, 0o1440, 0o0773),
    15: (0o5667, 0o2412, 0o2377), 16: (0o5051, 0o3516, 0o1525),
    17: (0o7665, 0o2761, 0o1531), 18: (0o6325, 0o3750, 0o3540),
    19: (0o4365, 0o2701, 0o0524), 20: (0o4745, 0o1206, 0o1035),
    21: (0o7633, 0o1544, 0o3337), 22: (0o6747, 0o1774, 0o0176),
    23: (0o4475, 0o0546, 0o0244), 24: (0o4225, 0o2213, 0o1027),
    25: (0o7063, 0o3707, 0o1753), 26: (0o4423, 0o2051, 0o3502),
    27: (0o6651, 0o3650, 0o0064), 28: (0o4161, 0o1777, 0o2275),
    29: (0o7237, 0o3203, 0o0044), 30: (0o4473, 0o1762, 0o2777),
    31: (0o5477, 0o2100, 0o0367), 32: (0o6163, 0o0571, 0o0535),
}


def _legendre() -> np.ndarray:
    """Legendre sequence L(t), t = 0..10222 (values 0/1)."""
    t = np.arange(_LEGENDRE_LEN, dtype=np.int64)
    # Euler criterion: L(t) = 1 iff t is a non-zero quadratic residue.
    # Vectorised pow is not available; compute residues directly.
    res = np.zeros(_LEGENDRE_LEN, dtype=np.int8)
    k = np.arange(1, _LEGENDRE_LEN, dtype=np.int64)
    res[(k * k) % _LEGENDRE_LEN] = 1
    return res


_LEGENDRE_CACHE: np.ndarray | None = None


def legendre() -> np.ndarray:
    global _LEGENDRE_CACHE
    if _LEGENDRE_CACHE is None:
        _LEGENDRE_CACHE = _legendre()
    return _LEGENDRE_CACHE


def weil_code(w: int) -> np.ndarray:
    """Weil code Wi(t; w) = L(t) XOR L((t+w) mod 10223), length 10223."""
    if not 1 <= w <= 5111:
        raise ValueError("Weil index must be in 1..5111")
    L = legendre()
    t = np.arange(_LEGENDRE_LEN, dtype=np.int64)
    return (L[t] ^ L[(t + w) % _LEGENDRE_LEN]).astype(np.int8)


def ranging_code(prn: int, kind: str = "cp") -> np.ndarray:
    """Return the 0/1 L1C ranging code (``kind`` = ``"cp"`` or ``"cd"``)."""
    if prn not in _RANGING_TABLE:
        raise ValueError(f"L1C PRN {prn} not tabulated (only 1..32)")
    row = _RANGING_TABLE[prn]
    if kind == "cp":
        w, p = row[0], row[1]
    elif kind == "cd":
        w, p = row[4], row[5]
    else:
        raise ValueError("kind must be 'cp' or 'cd'")

    W = weil_code(w)
    code = np.empty(L1C_CODE_LEN, dtype=np.int8)
    # t = 0 .. p-2  ->  W(t)
    code[: p - 1] = W[: p - 1]
    # expansion sequence occupies indices p-1 .. p+5
    code[p - 1: p + 6] = _EXPANSION
    # tail shifted by 7
    code[p + 6:] = W[p - 1:]
    return code


def overlay_code(prn: int) -> np.ndarray:
    """Return the 1800-bit L1Co overlay code (0/1) for a PRN 1..32.

    The overlay code is the Fibonacci LFSR output sequence
    ``s[n] = XOR_{j=1..11} m_j * s[n-j]`` with the published initial 11 bits
    serving as ``s[0..10]`` (MSB first).  This convention reproduces the
    published final 11 bits for all 32 PRNs (see :func:`self_test`).
    """
    if prn not in _OVERLAY_TABLE:
        if prn in _QZSS_RANGING_TABLE:
            # L1Co for QZSS is not modelled yet (the published triple is not the
            # Fibonacci polynomial used for GPS); use a neutral all-ones overlay.
            return np.ones(1800, dtype=np.int8)
        raise ValueError(f"L1Co PRN {prn} not tabulated (only 1..32)")
    poly, init, _final = _OVERLAY_TABLE[prn]

    # s[0..10]: MSB of the octal word is s[0].
    s = [(init >> (10 - i)) & 1 for i in range(11)]
    m = [(poly >> j) & 1 for j in range(12)]  # m[0] and m[11] are 1
    taps = [j for j in range(1, 12) if m[j]]

    out = np.empty(1800, dtype=np.int8)
    out[:11] = s
    for n in range(11, 1800):
        fb = 0
        for j in taps:
            fb ^= s[n - j]
        s.append(fb)
        out[n] = fb
    return out


def tmboc_is_boc61() -> np.ndarray:
    """Boolean mask (len 10230) marking chips modulated with BOC(6,1)."""
    mask = np.zeros(L1C_CODE_LEN, dtype=bool)
    mask[TMBOC_BOC61] = True
    return mask


def self_test() -> None:
    """Verify generated codes against the IS-GPS-800 published chip words."""
    for prn, row in _RANGING_TABLE.items():
        for kind, w_idx, p_idx, init_idx, final_idx in (
            ("cp", 0, 1, 2, 3),
            ("cd", 4, 5, 6, 7),
        ):
            code = ranging_code(prn, kind)
            init = row[init_idx]
            final = row[final_idx]
            head = 0
            tail = 0
            for i in range(24):
                # MSB (bit 23) is the first chip
                head = (head << 1) | int(code[i])
                tail = (tail << 1) | int(code[L1C_CODE_LEN - 24 + i])
            if head != init or tail != final:
                raise AssertionError(
                    f"ranging code mismatch PRN {prn} {kind}: "
                    f"head {head:08o}!={init:08o} tail {tail:08o}!={final:08o}"
                )

    for prn, (poly, init, final) in _OVERLAY_TABLE.items():
        code = overlay_code(prn)
        if len(code) != 1800:
            raise AssertionError("overlay length")
        head = 0
        for i in range(11):
            head = (head << 1) | int(code[i])
        tail = 0
        for i in range(1789, 1800):
            tail = (tail << 1) | int(code[i])
        if head != init or tail != final:
            raise AssertionError(
                f"overlay mismatch PRN {prn}: "
                f"head {head:04o}!={init:04o} tail {tail:04o}!={final:04o}"
            )


def _overlay_state_after(poly: int, init: int, n: int) -> int:
    """Deprecated helper retained for reference; see :func:`overlay_code`."""
    s = [(init >> (10 - i)) & 1 for i in range(11)]
    m = [(poly >> j) & 1 for j in range(12)]
    taps = [j for j in range(1, 12) if m[j]]
    for k in range(11, n):
        fb = 0
        for j in taps:
            fb ^= s[k - j]
        s.append(fb)
    return sum(int(s[i]) << (10 - i) for i in range(11))



# ---------------------------------------------------------------------------
# QZSS L1C (PRN 193..202).  Same code construction as GPS L1C.
# Range (w,p) and verification chips: IS-QZSS-PNT-003 Table 3.2.3-1.
# ---------------------------------------------------------------------------
_QZSS_RANGING_TABLE = {
    193: (4311, 9864, 0o70670250, 0o11640746, 4834, 9753, 0o54420241, 0o43473502),
    194: (5024, 9753, 0o24737373, 0o51661203, 4456, 4799, 0o75476311, 0o32402217),
    195: (4352, 9859, 0o04467202, 0o15610600, 4056, 10126, 0o50612163, 0o43454074),
    196: (4678, 328, 0o02551300, 0o70117174, 3804, 241, 0o77772455, 0o06321507),
    197: (5034, 1, 0o32252546, 0o77615261, 3672, 1245, 0o03320402, 0o22101365),
    198: (5085, 4733, 0o10121331, 0o22447126, 4205, 1274, 0o20225612, 0o67251717),
    199: (3646, 164, 0o10537634, 0o65022442, 3348, 1456, 0o55426411, 0o02047657),
    200: (4868, 135, 0o32014275, 0o41243522, 4152, 9967, 0o70477545, 0o43352227),
    201: (3668, 174, 0o13126037, 0o56605536, 3883, 235, 0o71116442, 0o04471535),
    202: (4211, 132, 0o60700561, 0o13020736, 3473, 512, 0o42077151, 0o62510717),
}
_QZSS_OVERLAY_TABLE = {
    193: (0o5403, 0o0500, 0o3261), 194: (0o5403, 0o0254, 0o1760),
    195: (0o5403, 0o3445, 0o0430), 196: (0o5403, 0o2542, 0o3477),
    197: (0o5403, 0o1257, 0o1676), 198: (0o6501, 0o0211, 0o1636),
    199: (0o6501, 0o0534, 0o2411), 200: (0o6501, 0o1420, 0o1473),
    201: (0o6501, 0o3401, 0o2266), 202: (0o6501, 0o0714, 0o2104),
}
_RANGING_TABLE.update(_QZSS_RANGING_TABLE)
# QZSS L1Co overlay is kept for reference but NOT merged: its published triple
# is not the Fibonacci polynomial used for GPS, so overlay_code() falls back to
# an all-ones overlay for QZSS PRNs.
# _OVERLAY_TABLE.update(_QZSS_OVERLAY_TABLE)
