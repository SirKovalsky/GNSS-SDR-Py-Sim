"""Expose pip-installed NVIDIA CUDA libraries to CuPy on Windows.

``cupy-cuda12x`` wheels do not bundle the CUDA runtime: CuPy normally finds
it through ``CUDA_PATH`` (a full CUDA Toolkit install) or a ``nvcc`` on
``PATH``.  The ``nvidia-*-cu12`` wheels ship the same DLLs under
``site-packages/nvidia/<component>/bin`` but never register those directories
with the Windows loader, so CuPy fails with e.g.::

    CuPy failed to load nvrtc64_120_0.dll

:func:`configure_cuda_dlls` locates those wheel directories and adds them to
the DLL search path, letting CuPy JIT-compile kernels without a system-wide
CUDA Toolkit.  It is a no-op off Windows and when no ``nvidia`` wheel is
installed.
"""

from __future__ import annotations

import os
import sys

_configured = False


def _nvidia_bin_dirs() -> list[str]:
    """Return ``bin`` dirs of every installed ``nvidia-*`` wheel component."""
    try:
        import nvidia
    except Exception:
        return []

    bin_dirs: list[str] = []
    for root in getattr(nvidia, "__path__", []):
        try:
            components = os.listdir(root)
        except OSError:
            continue
        for comp in components:
            candidate = os.path.join(root, comp, "bin")
            if os.path.isdir(candidate):
                bin_dirs.append(candidate)
    return bin_dirs


def configure_cuda_dlls() -> None:
    """Make pip-installed NVIDIA CUDA DLLs loadable (Windows only)."""
    global _configured
    if _configured or not sys.platform.startswith("win"):
        return
    _configured = True

    bin_dirs = _nvidia_bin_dirs()
    if not bin_dirs:
        return

    for path in bin_dirs:
        try:
            os.add_dll_directory(path)
        except OSError:
            pass
        os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")

    if not os.environ.get("CUDA_PATH"):
        for path in bin_dirs:
            names = os.listdir(path)
            if any(n.startswith("cudart64_") and n.endswith(".dll")
                   for n in names):
                os.environ["CUDA_PATH"] = os.path.dirname(path)
                break
