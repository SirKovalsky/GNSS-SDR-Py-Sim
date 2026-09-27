"""Nav-frame roll / ephemeris continuity (offline, no synthesis).

Background: a 760 s over-air run (band ``l1``, RINEX DOY267, start
2026/09/24 12:00:00) showed three non-boundary receiver PVT outages at
272.9/440.5/628.5 s while GSV still reported 9-12 GPS / 3-5 Galileo in view.
This test locks in that the *generator* is not the cause: over 800 s every
system's 30/18/6 s frame rolls strictly forward by its nominal period, the
GPS legacy NAV subframe-1 TOW stays aligned to the frame epoch, and the
per-channel ephemeris (selected once at allocation) never changes.

``_roll_frames`` is the exact method ``_synthesise`` calls once per block, so
driving it directly samples the 800 s scene without the I/Q cost.

Headless, no hardware, no network::

    .venv\\Scripts\\python.exe -m pytest tests -q
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from gnss_sim.config import parse_start_time  # noqa: E402
from gnss_sim.constants import R2D  # noqa: E402
from gnss_sim.engine import SignalEngine  # noqa: E402
from gnss_sim.orbit import llh2xyz  # noqa: E402
from gnss_sim.rinex import parse_nav_file  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NAV = os.path.join(_ROOT, "rinex_cache", "BRDC00IGS_R_20262670000_01D_MN.rnx")
_START = "2026/09/24,12:00:00"

pytestmark = pytest.mark.skipif(
    not os.path.exists(_NAV), reason="cached merged RINEX DOY267")


def _epochs(ch) -> tuple:
    return (
        None if ch.frame_start is None else round(ch.frame_start.sec, 3),
        None if ch.l1c_frame_start is None else round(ch.l1c_frame_start.sec, 3),
        None if ch.sbas_frame_start is None else round(ch.sbas_frame_start.sec, 3),
        None if ch.galileo_frame_start is None else round(ch.galileo_frame_start.sec, 3),
        None if ch.b1i_frame_start is None else round(ch.b1i_frame_start.sec, 3),
    )


def _engine() -> SignalEngine:
    by_sv, iono = parse_nav_file(_NAV)
    start = parse_start_time(_START)
    xyz = llh2xyz(60.0 / R2D, 30.0 / R2D, 100.0)
    np.random.seed(20240926)
    return SignalEngine(
        by_sv, iono, lambda _g: xyz, start, 2.6e6,
        enable_ca=True, enable_l1c=True, enable_galileo=True,
        enable_qzss=True, enable_sbas=True, enable_beidou=False,
        el_mask=5.0 / R2D, amp_scale=0.15, backend="cpu")


def test_roll_frames_800s_all_systems_forward_and_nominal() -> None:
    eng = _engine()
    assert eng.channels, "no channels allocated for the DOY267 l1 scene"
    # Ephemeris is chosen once at allocation; _roll_frames must not change it.
    keys = {ch.name: (ch.eph.toe.sec, ch.eph.iode) if ch.eph is not None
            else None for ch in eng.channels}
    prev = {ch.name: _epochs(ch) for ch in eng.channels}
    prev_g = eng.g.sec
    rolls = 0
    for _ in range(1600):                       # 800 s at 0.5 s blocks
        eng.g.sec += 0.5
        assert eng.g.sec > prev_g
        prev_g = eng.g.sec
        eng._roll_frames()
        for ch in eng.channels:
            cur = _epochs(ch)
            for b, a in zip(prev[ch.name], cur):
                if b is None or a is None or b == a:
                    continue
                d = a - b
                assert any(abs(d - s) < 1e-6 for s in (30.0, 18.0, 6.0)), (
                    eng.g.sec, ch.name, b, a, d)
                rolls += 1
            prev[ch.name] = cur
            if ch.eph is not None:
                assert (ch.eph.toe.sec, ch.eph.iode) == keys[ch.name]
    assert rolls >= 100, f"too few rolls observed ({rolls})"


def test_legacy_nav_sf1_tow_alignment_across_rolls() -> None:
    eng = _engine()
    gps = next(ch for ch in eng.channels if ch.kind == "gps")
    checked = 0
    for _ in range(200):                        # 100 s
        eng.g.sec += 0.5
        before = gps.frame_start.sec
        eng._roll_frames()
        if gps.frame_start.sec == before:
            continue
        # Subframe 1 word 1: 3-bit SF id (bit 8) and 17-bit TOW (bit 13).
        w = int(gps.dwrd[11])
        assert ((w >> 8) & 0x7) == 1
        assert (w >> 13) & 0x1FFFF == int(gps.frame_start.sec) // 6 + 1
        checked += 1
    assert checked >= 3


def test_decoded_gps_nav_survives_a_real_roll() -> None:
    """A short real synthesis across a 30 s boundary stays bit-stable.

    The GPS leg is deterministic (same seed, CPU backend); generating 0.4 s
    before + 0.4 s after the same boundary twice must give identical bytes.
    A frame-alignment bug would make the second call differ.
    """
    eng = _engine()
    gps = next(ch for ch in eng.channels if ch.kind == "gps")
    # Move close to the next frame roll (frame_start + 30 s).
    target = gps.frame_start.sec + 30.0
    while eng.g.sec < target - 0.5:
        eng.g.sec += 0.4
        eng._roll_frames()
    out1 = np.concatenate([eng.generate_block(1000) for _ in range(2)])

    eng2 = _engine()
    gps2 = next(ch for ch in eng2.channels if ch.kind == "gps")
    target2 = gps2.frame_start.sec + 30.0
    while eng2.g.sec < target2 - 0.5:
        eng2.g.sec += 0.4
        eng2._roll_frames()
    out2 = np.concatenate([eng2.generate_block(1000) for _ in range(2)])

    assert np.array_equal(out1, out2)
