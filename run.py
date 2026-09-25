"""Launch the GNSS Sim GUI.

UHD is imported *before* PyQt5 on Windows: the native UHD libraries must load
before Qt, otherwise creating a USRP object can crash with 0xC0000005 (the same
precaution used by the SDR_Scan application).
"""

import os
import sys

# UHD paints ``[INFO]``/``[WARNING]`` on stderr (orange in PowerShell); keep
# only real errors unless the user overrides UHD_LOG_LEVEL.  Must be set before
# the first ``import uhd``.
os.environ.setdefault("UHD_LOG_LEVEL", "error")

try:  # pragma: no cover - platform dependent
    import uhd  # noqa: F401
except Exception:
    pass

from gnss_sim.gui import main

if __name__ == "__main__":
    raise SystemExit(main())
