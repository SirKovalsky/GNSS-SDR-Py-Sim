"""``python -m gnss_sim`` entry point.

Re-executes under the project-local virtualenv when started with a foreign
interpreter (e.g. the system Python), exactly like ``run.py``.  Importing the
``gnss_sim`` package is dependency-free (see ``__init__``), so this happens
before NumPy is needed.
"""

import sys

from . import venv

# Re-exec ``python -m gnss_sim`` under the venv when needed (loop-guarded).
_reexec_rc = venv.maybe_reexec_module("gnss_sim", sys.argv[1:])
if _reexec_rc is not None:
    raise SystemExit(_reexec_rc)

from .cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
