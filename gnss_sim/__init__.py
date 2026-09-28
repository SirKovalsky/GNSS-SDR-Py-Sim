"""gnss_sim - GPS L1 C/A + L1C baseband signal simulator.

Pure-Python (NumPy) generator of GPS L1 baseband IQ samples.  Supports the
legacy GPS L1 C/A signal and the modernised GPS L1C signal (Legendre/Weil
ranging codes, TMBOC(6,1,4/33) pilot, BOC(1,1) data, CNAV-2 / BCH / LDPC).

The generated baseband can be written to an IQ file (SC16 / SC08 / SC01) or
streamed to a USRP B210 with UHD.
"""

from __future__ import annotations

__version__ = "0.1.0"


def __getattr__(name: str):
    """Lazily expose ``gnss_sim.constants`` without importing NumPy eagerly.

    Importing the package must stay dependency-free so ``python -m gnss_sim``
    can re-exec into the project virtualenv before any third-party import (the
    system interpreter has no NumPy).  ``gnss_sim.constants`` (and any future
    submodule) is still reachable as an attribute on first use.
    """
    if name == "constants":
        from . import constants
        return constants
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
