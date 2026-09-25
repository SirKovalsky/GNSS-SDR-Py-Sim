"""User-motion file loading (10 Hz ECEF or lat/lon/height CSV)."""

from __future__ import annotations

import csv

import numpy as np

from .constants import R2D
from .orbit import llh2xyz


def load_user_motion(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a 10 Hz user-motion file.

    The file is a CSV whose rows are either ``x,y,z`` (ECEF metres) or
    ``lat,lon,h`` (degrees, metres).  Rows starting with ``#`` are ignored.

    Returns ``(times, xyz)`` where ``times = arange(N) * 0.1`` seconds and
    ``xyz`` has shape ``(N, 3)``.
    """
    rows: list[list[float]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p for p in line.replace(";", ",").split(",") if p != ""]
            if len(parts) < 3:
                continue
            try:
                rows.append([float(p) for p in parts[:3]])
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"Не удалось прочитать файл движения: {path}")
    arr = np.asarray(rows, dtype=np.float64)
    xyz = np.empty_like(arr)
    # Heuristic: ECEF coordinates are ~1e6 m; lat/lon are small degrees.
    if np.max(np.abs(arr)) > 1.0e5:
        xyz = arr
    else:
        for i in range(arr.shape[0]):
            xyz[i] = llh2xyz(arr[i, 0] / R2D, arr[i, 1] / R2D, arr[i, 2])
    times = np.arange(arr.shape[0], dtype=np.float64) * 0.1
    return times, xyz


def interpolation_fn(times: np.ndarray, xyz: np.ndarray):
    """Return ``xyz_fn(elapsed_seconds) -> ECEF`` with linear interpolation."""
    t0 = times[0]
    t1 = times[-1]

    def fn(elapsed: float) -> np.ndarray:
        e = min(max(elapsed, t0), t1)
        return np.array([np.interp(e, times, xyz[:, k]) for k in range(3)],
                        dtype=np.float64)

    return fn
