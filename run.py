"""Launch the GNSS Sim GUI.

UHD is imported *before* PyQt5 on Windows: the native UHD libraries must load
before Qt, otherwise creating a USRP object can crash with 0xC0000005 (the same
precaution used by the SDR_Scan application).
"""

import os
import sys

# Keep UHD at its quietest level and make sure its native stderr (including the
# bare ``U``/``O`` under/overflow markers that bypass UHD_LOG_LEVEL) never paints
# the console orange.  Both must happen before the first ``import uhd``.
from gnss_sim.nativelog import install_native_stderr_filter, quiet_uhd

quiet_uhd()
install_native_stderr_filter()

try:  # pragma: no cover - platform dependent
    import uhd  # noqa: F401
except Exception:
    pass

from gnss_sim.gui import main

if __name__ == "__main__":
    raise SystemExit(main())
