"""gnss_sim - GPS L1 C/A + L1C baseband signal simulator.

Pure-Python (NumPy) generator of GPS L1 baseband IQ samples.  Supports the
legacy GPS L1 C/A signal and the modernised GPS L1C signal (Legendre/Weil
ranging codes, TMBOC(6,1,4/33) pilot, BOC(1,1) data, CNAV-2 / BCH / LDPC).

The generated baseband can be written to an IQ file (SC16 / SC08 / SC01) or
streamed to a USRP B210 with UHD.
"""

from __future__ import annotations

__version__ = "0.1.0"

from . import constants  # noqa: F401
