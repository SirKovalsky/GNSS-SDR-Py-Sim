"""Best-effort OS thread-priority boost for real-time TX streaming.

The B210 TX path has a bounded jitter queue between the synthesis producer and
the UHD consumer.  If the producer (or the UHD sender) is descheduled for
longer than the queue holds, the consumer starves and the USRP reports an
underflow — a short RF gap that can cost a receiver its fix.  Raising the two
threads' priority reduces those stalls.

Everything here is best-effort and never raises: the module must be importable
and safe on every platform without hardware.
"""

from __future__ import annotations

import ctypes
import sys

#: Windows ``SetThreadPriority`` levels (winbase.h).
THREAD_PRIORITY_ABOVE_NORMAL = 1
THREAD_PRIORITY_HIGHEST = 2


def boost_thread_priority(level: int = THREAD_PRIORITY_HIGHEST) -> bool:
    """Raise the calling thread's priority; return ``True`` on success.

    * Windows: ``SetThreadPriority(GetCurrentThread(), level)``.
    * Linux: ``nice(-10)`` when the process is allowed to (usually needs
      privilege); a failure is silently ignored.

    The function is deliberately total: any failure returns ``False`` instead
    of raising, so a run without permission still works.
    """
    if sys.platform.startswith("win"):
        try:
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetCurrentThread()
            return bool(kernel32.SetThreadPriority(handle, int(level)))
        except Exception:  # noqa: BLE001 - priority is an optimisation only
            return False
    try:
        import os
        if hasattr(os, "nice"):
            os.nice(-10)
        return True
    except Exception:  # noqa: BLE001
        return False
