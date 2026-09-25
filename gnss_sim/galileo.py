"""Galileo E1-B / E1-C spreading codes and CBOC(6,1,1/11) modulation.

The primary codes are the published 4092-chip memory codes (1.023 Mcps, 4 ms
period).  E1-B carries the I/NAV data (250 bps), E1-C is the pilot channel with
a 25-chip secondary code (250 bps, 100 ms period).  Both components use
CBOC(6,1,1/11): a weighted sum of sine-phased BOC(1,1) and BOC(6,1) subcarriers.

The hex tables live in :mod:`.galileo_e1_tables` (transcribed from the Galileo
OS SIS ICD via the public `bijux-gnss-signal` crate, PRN 1..50).
"""

from __future__ import annotations

import numpy as np

from .galileo_e1_tables import (
    E1B_HEX, E1C_HEX, E1C_SECONDARY,
    GALILEO_E1_PRIMARY_CHIPS, GALILEO_E1_SECONDARY_CHIPS,
)

# CBOC(6,1,1/11) weighting coefficients (sine-phased BOC).
CBOC_ALPHA = float(np.sqrt(10.0 / 11.0))   # 0.9534625892...
CBOC_BETA = float(np.sqrt(1.0 / 11.0))     # 0.3015113446...


def _hex_to_bipolar(text: str, nchips: int) -> np.ndarray:
    """Expand an MSB-first hex string into a ±1 chip array."""
    bits = np.empty(len(text) * 4, dtype=np.int8)
    for i, ch in enumerate(text):
        v = int(ch, 16)
        base = i * 4
        bits[base + 0] = (v >> 3) & 1
        bits[base + 1] = (v >> 2) & 1
        bits[base + 2] = (v >> 1) & 1
        bits[base + 3] = v & 1
    return (1 - 2 * bits[:nchips]).astype(np.int8)


def e1_code(prn: int, component: str = "b") -> np.ndarray:
    """Return the ±1 E1-B (``component='b'``) or E1-C (``'c'``) code."""
    if component == "b":
        table = E1B_HEX
    elif component == "c":
        table = E1C_HEX
    else:
        raise ValueError("component must be 'b' or 'c'")
    if prn not in table:
        raise ValueError(f"Galileo PRN {prn} not tabulated (only 1..50)")
    return _hex_to_bipolar(table[prn], GALILEO_E1_PRIMARY_CHIPS)


def e1_secondary() -> np.ndarray:
    """Return the 25-chip ±1 E1-C secondary code (same for every PRN)."""
    bits = np.array([int(c) for c in E1C_SECONDARY], dtype=np.int8)
    return (1 - 2 * bits[:GALILEO_E1_SECONDARY_CHIPS]).astype(np.int8)


def cboc(frac: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (boc11, boc61) sine-phased subcarrier values for chip fractions."""
    boc11 = np.sign(np.sin(2.0 * np.pi * frac))
    boc61 = np.sign(np.sin(2.0 * np.pi * 6.0 * frac))
    boc11[boc11 == 0] = 1.0
    boc61[boc61 == 0] = 1.0
    return boc11, boc61


def self_test() -> None:
    for prn in range(1, 51):
        b = e1_code(prn, "b")
        c = e1_code(prn, "c")
        if b.shape != (GALILEO_E1_PRIMARY_CHIPS,) or c.shape != (GALILEO_E1_PRIMARY_CHIPS,):
            raise AssertionError(f"E1 code length PRN {prn}")
        if not (np.any(b == 1) and np.any(b == -1)):
            raise AssertionError(f"E1-B code constant PRN {prn}")
        if not (np.any(c == 1) and np.any(c == -1)):
            raise AssertionError(f"E1-C code constant PRN {prn}")
    sec = e1_secondary()
    if sec.shape != (GALILEO_E1_SECONDARY_CHIPS,):
        raise AssertionError("E1-C secondary length")
    if len(set(sec.tolist())) != 2:
        raise AssertionError("E1-C secondary not bipolar")
