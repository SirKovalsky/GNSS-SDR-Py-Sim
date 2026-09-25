"""System memory / storage analysis for RAM-disk and looping decisions.

No third-party dependencies: Windows uses ``GlobalMemoryStatusEx`` via ctypes,
Linux reads ``/proc/meminfo``.  Used by the runner to report how many seconds of
IQ fit in RAM and to keep loops inside a safe memory budget.
"""

from __future__ import annotations

import os
import shutil
import sys


def _windows_memory() -> tuple[int, int]:
    import ctypes

    class _MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    stat = _MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    return int(stat.ullTotalPhys), int(stat.ullAvailPhys)


def _linux_memory() -> tuple[int, int]:
    total = avail = 0
    try:
        with open("/proc/meminfo", "r", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1]) * 1024
                elif line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) * 1024
    except OSError:
        pass
    return total, avail


def total_ram() -> int:
    """Physical RAM in bytes (0 if unknown)."""
    if sys.platform.startswith("win"):
        return _windows_memory()[0]
    return _linux_memory()[0]


def available_ram() -> int:
    """Currently available (free) RAM in bytes (0 if unknown)."""
    if sys.platform.startswith("win"):
        return _windows_memory()[1]
    return _linux_memory()[1]


def disk_free(path: str) -> int:
    """Free space in bytes on the volume holding ``path`` (0 if unknown)."""
    probe = path
    while probe and not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    try:
        return int(shutil.disk_usage(probe or ".").free)
    except OSError:
        return 0


def human(nbytes: float) -> str:
    step = 1000.0
    for unit in ("Б", "КБ", "МБ", "ГБ", "ТБ"):
        if abs(nbytes) < step:
            return f"{nbytes:.1f} {unit}"
        nbytes /= step
    return f"{nbytes:.1f} ПБ"


def default_budget(available: int | None = None, fraction: float = 0.6) -> int:
    """Safe share of available RAM for an IQ buffer (60 % by default)."""
    avail = available_ram() if available is None else available
    if avail <= 0:
        return 0
    return int(avail * fraction)


def auto_segment_seconds(requested_seconds: float,
                         max_fit_seconds: float,
                         loop_seconds: float = 0.0,
                         frame_lcm: float = 90.0) -> tuple[float, bool]:
    """Decide the loop segment length.

    * an explicit ``loop_seconds`` is honoured (capped by ``max_fit_seconds``);
    * otherwise the requested duration is used **exactly** when it fits the
      RAM budget (one segment, even with looping enabled);
    * only when it does not fit is the segment capped to the budget and
      rounded down to a multiple of ``frame_lcm`` (LCM of the NAV 30 s and
      CNAV-2 18 s frames = 90 s).  ``requested_seconds <= 0`` (indefinite)
      keeps the historical bounded default of 90 s.

    Returns ``(segment_seconds, capped)`` where ``capped`` is ``True`` when the
    segment is shorter than the requested duration because of the budget.
    """
    requested = float(requested_seconds or 0.0)
    max_fit = float(max_fit_seconds or 0.0)
    loop = float(loop_seconds or 0.0)

    if loop > 0.0:
        if max_fit > 0.0 and loop > max_fit:
            return max_fit, True
        return loop, False
    if max_fit <= 0.0:
        return requested, False
    if requested > 0.0 and requested <= max_fit:
        return requested, False

    base = max_fit if requested > 0.0 else min(frame_lcm, max_fit)
    if frame_lcm > 0.0 and base >= frame_lcm:
        base = float(int(base // frame_lcm) * frame_lcm)
    return base, requested > 0.0


def plan(buffer_bytes_per_second: float,
         requested_seconds: float,
         budget_bytes: int,
         free_disk_bytes: int = 0,
         to_file: bool = True) -> dict:
    """Compute a feasible segment length and a human-readable report.

    Returns a dict with ``segment_seconds`` (may be capped), ``fits_ram``,
    ``est_bytes``, ``est_disk_ok`` and ``messages`` (list of str).
    """
    messages: list[str] = []
    if buffer_bytes_per_second <= 0:
        return {"segment_seconds": requested_seconds, "fits_ram": False,
                "est_bytes": 0, "est_disk_ok": True, "messages": messages}

    est_bytes = max(0.0, requested_seconds) * buffer_bytes_per_second
    fits_ram = budget_bytes <= 0 or est_bytes <= budget_bytes

    segment = requested_seconds
    if budget_bytes > 0 and requested_seconds > 0 and est_bytes > budget_bytes:
        segment = budget_bytes / buffer_bytes_per_second
        messages.append(
            f"Запрошено {requested_seconds:.0f} с, но в бюджет RAM помещается "
            f"≈{segment:.1f} с — длительность сегмента ограничена.")

    est_disk_ok = True
    if to_file and free_disk_bytes > 0:
        need = segment * buffer_bytes_per_second
        if need > free_disk_bytes:
            est_disk_ok = False
            messages.append(
                f"На целевом диске только {human(free_disk_bytes)}, "
                f"а сегмент займёт {human(need)}.")
    return {"segment_seconds": segment, "fits_ram": fits_ram,
            "est_bytes": est_bytes, "est_disk_ok": est_disk_ok,
            "messages": messages}
