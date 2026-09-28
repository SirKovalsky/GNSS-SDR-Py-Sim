"""Launch the GNSS Sim GUI.

The project-local virtualenv is used automatically: if ``run.py`` was started
with another interpreter it re-executes itself with
``.venv\\Scripts\\python.exe`` (Windows) or ``.venv/bin/python`` (Linux) before
importing anything heavy.  ``GNSS_SIM_VENV_REEXEC`` guards against a loop.

The re-exec helper is loaded *directly from its file* rather than
``from gnss_sim.venv import ...``: importing the ``gnss_sim`` package runs
``__init__`` -> ``constants`` -> ``numpy``, and the system interpreter that
starts ``python run.py`` has no NumPy, so the old import crashed with
``ModuleNotFoundError`` before the re-exec could happen.  This launcher is now
dependency-free.

UHD is imported *before* PyQt5 on Windows: the native UHD libraries must load
before Qt, otherwise creating a USRP object can crash with 0xC0000005.
"""

import importlib.util
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _load_venv_module():
    """Load ``gnss_sim/venv.py`` without importing the ``gnss_sim`` package."""
    path = os.path.join(_ROOT, "gnss_sim", "venv.py")
    spec = importlib.util.spec_from_file_location("gnss_sim_venv", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_venv = _load_venv_module()

# Re-exec under the project-local venv when needed (no-op when already there).
# The loop guard is passed to the child through GNSS_SIM_VENV_REEXEC.
if _venv is not None:
    _reexec_rc = _venv.maybe_reexec(sys.argv[1:], root=_ROOT)
    if _reexec_rc is not None:
        raise SystemExit(_reexec_rc)

# Diagnostics / headless verification: print the interpreter that actually runs
# the GUI (after any re-exec) and exit before touching Qt/UHD.
if "--print-interpreter" in sys.argv or \
        str(os.environ.get("GNSS_SIM_PRINT_INTERPRETER", "")) == "1":
    print(sys.executable)
    raise SystemExit(0)

# Keep UHD's native stderr (including the bare ``U``/``O`` under/overflow markers
# that bypass UHD_LOG_LEVEL) out of the console and route its ``[INFO]`` lines
# into the GUI journal.  Both must happen before the first ``import uhd``.
from gnss_sim.nativelog import (install_native_stderr_filter, quiet_uhd_gui)

quiet_uhd_gui()
install_native_stderr_filter()

try:  # pragma: no cover - platform dependent
    import uhd  # noqa: F401
except Exception:
    pass

from gnss_sim.gui import main

if __name__ == "__main__":
    raise SystemExit(main())
