"""Track model and a heavy mining dump-truck motion simulator.

The user draws a haul-road polyline on the map (see :mod:`gnss_sim.mapview`);
this module resamples it and produces a 10 Hz receiver trajectory that can be
saved as a ``Lat,Lon,Hgt`` CSV and fed to :mod:`gnss_sim.motion`.

The truck model is deliberately simple but physically plausible for a mining
haul truck: very low acceleration when loaded, limited lateral acceleration in
curves, service-brake deceleration, speed limits for the laden and empty legs,
and loading/dumping dwell times.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass

import numpy as np

_R_EARTH = 6371000.0


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * _R_EARTH * math.asin(min(1.0, math.sqrt(a)))


def resample_track(points: list[tuple[float, float]],
                   spacing: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample a (lat, lon) polyline to roughly ``spacing`` metre steps."""
    if len(points) < 2:
        raise ValueError("Трек должен содержать минимум две точки")
    lat = np.array([p[0] for p in points], dtype=float)
    lon = np.array([p[1] for p in points], dtype=float)
    seg = np.array([haversine(lat[i], lon[i], lat[i + 1], lon[i + 1])
                    for i in range(len(points) - 1)], dtype=float)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if total <= 0:
        raise ValueError("Длина трека равна нулю")
    n = max(2, int(total / spacing) + 1)
    si = np.linspace(0.0, total, n)
    lati = np.interp(si, s, lat)
    loni = np.interp(si, s, lon)
    return lati, loni, si


def _curve_speed_limit(lat: np.ndarray, lon: np.ndarray,
                       lat_acc: float, ds: float,
                       v_min: float = 1.4) -> np.ndarray:
    """Speed limit from local curvature, v = sqrt(a_lat * R), clamped to >= v_min."""
    n = len(lat)
    vmax = 1.0e9
    if n < 3:
        return np.full(n, vmax)
    x = np.radians(lon) * _R_EARTH * np.cos(np.radians(np.mean(lat)))
    y = np.radians(lat) * _R_EARTH
    v = np.full(n, vmax)
    r_min = (v_min * v_min) / max(lat_acc, 1e-6)
    for i in range(1, n - 1):
        ax, ay = x[i - 1], y[i - 1]
        bx, by = x[i], y[i]
        cx, cy = x[i + 1], y[i + 1]
        a = math.hypot(bx - ax, by - ay)
        b = math.hypot(cx - bx, cy - by)
        c = math.hypot(cx - ax, cy - ay)
        area = abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax)) / 2.0
        if area < 1e-9 or a < 1e-9 or b < 1e-9 or c < 1e-9:
            R = _R_EARTH
        else:
            R = (a * b * c) / (4.0 * area)
        R = max(R, r_min)
        v[i] = max(v_min, math.sqrt(lat_acc * R))
    # braking lookahead: a point must be approachable from its neighbours
    ds = max(ds, 0.1)
    for _ in range(2):
        for i in range(n - 2, -1, -1):
            v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2.0 * 1.5 * ds))
        for i in range(1, n):
            v[i] = min(v[i], math.sqrt(v[i - 1] ** 2 + 2.0 * 1.5 * ds))
    return v



@dataclass
class TruckParams:
    v_laden_kmh: float = 40.0      # laden speed limit
    v_empty_kmh: float = 60.0      # empty speed limit
    accel: float = 0.5             # m/s^2
    brake: float = 1.5             # m/s^2
    lat_acc: float = 0.8           # m/s^2 lateral (anti-rollover)
    load_s: float = 180.0          # dwell at loading point
    dump_s: float = 60.0           # dwell at dump point
    cycles: int = 1
    grade_pct: float = 0.0         # constant haul-road grade
    altitude0: float = 10.0        # start altitude, m
    dt: float = 0.1                # output sample interval
    start_loaded: bool = True


def simulate_truck(lat: np.ndarray, lon: np.ndarray, s: np.ndarray,
                   p: TruckParams) -> dict:
    """Run the haul-cycle simulation; returns arrays ``t, lat, lon, h, v, phase``."""
    n = len(s)
    ds = float(np.median(np.diff(s))) if n > 1 else 1.0
    v_curve = _curve_speed_limit(lat, lon, p.lat_acc, ds)
    v_lad = p.v_laden_kmh / 3.6
    v_emp = p.v_empty_kmh / 3.6
    dt = p.dt

    t_list: list[float] = []
    lat_list: list[float] = []
    lon_list: list[float] = []
    h_list: list[float] = []
    v_list: list[float] = []
    ph_list: list[str] = []

    t = 0.0
    total = float(s[-1])

    def append(pos: float, v: float, phase: str, loaded: bool) -> None:
        pos = min(max(pos, 0.0), total)
        la = float(np.interp(pos, s, lat))
        lo = float(np.interp(pos, s, lon))
        if p.grade_pct:
            h = p.altitude0 + (p.grade_pct / 100.0) * pos
        else:
            h = p.altitude0
        t_list.append(t)
        lat_list.append(la)
        lon_list.append(lo)
        h_list.append(h)
        v_list.append(v)
        ph_list.append(phase)

    for cycle in range(p.cycles):
        for loaded, direction in ((p.start_loaded, +1), (not p.start_loaded, -1)):
            phase = "loaded" if loaded else "empty"
            vlim = v_lad if loaded else v_emp
            pos = 0.0 if direction > 0 else total
            v = 0.0
            dummy = 0.0
            while True:
                idx = min(n - 1, int(np.searchsorted(s, pos)))
                # target speed: min of truck limit, curve limit, and safe stop
                tgt = min(vlim, float(v_curve[idx]))
                dist_end = (total - pos) if direction > 0 else pos
                if dist_end < (v * v) / (2.0 * p.brake) + 0.5:
                    tgt = 0.0
                if v < tgt:
                    v = min(tgt, v + p.accel * dt)
                else:
                    v = max(tgt, v - p.brake * dt)
                append(pos, v, phase, loaded)
                t += dt
                pos += direction * v * dt
                if direction > 0 and pos >= total - 0.5:
                    break
                if direction < 0 and pos <= 0.5:
                    break
                dummy += 1
                if dummy > 5_000_000:
                    break
            # dwell at the end of the leg
            dwell = p.dump_s if (direction > 0 and loaded) else p.load_s
            end_pos = total if direction > 0 else 0.0
            v = 0.0
            steps = int(round(dwell / dt))
            for _ in range(steps):
                append(end_pos, 0.0, "dump" if (direction > 0 and loaded) else "load",
                       loaded)
                t += dt

    out = {
        "t": np.array(t_list),
        "lat": np.array(lat_list),
        "lon": np.array(lon_list),
        "h": np.array(h_list),
        "v": np.array(v_list),
        "phase": ph_list,
    }
    return out


def write_motion_csv(path: str, result: dict) -> None:
    """Write a 10 Hz ``Lat,Lon,Hgt`` CSV (compatible with :mod:`gnss_sim.motion`).

    The first three columns are latitude, longitude and height; rows are evenly
    spaced at ``TruckParams.dt`` (10 Hz by default), which is what
    :func:`gnss_sim.motion.load_user_motion` expects.
    """
    with open(path, "w", newline="", encoding="utf-8") as fh:
        fh.write("# lat_deg;lon_deg;height_m  (10 Hz motion file; "
                 "движение самосвала)\n")
        wr = csv.writer(fh, delimiter=";")
        for la, lo, h in zip(result["lat"], result["lon"], result["h"]):
            wr.writerow([f"{la:.9f}", f"{lo:.9f}", f"{h:.3f}"])


def motion_rows(result: dict):
    """Yield ``(t, lat, lon, h, speed_kmh, phase)`` for preview tables."""
    for t, la, lo, h, v, ph in zip(result["t"], result["lat"], result["lon"],
                                   result["h"], result["v"], result["phase"]):
        yield t, la, lo, h, v * 3.6, ph

