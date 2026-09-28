"""Project-local virtualenv handling for ``run.py``.

The GUI must run under the project-local interpreter (NumPy / PyQt5 / UHD live
there).  ``run.py`` calls :func:`maybe_reexec` at import time: when it was
started with a *different* interpreter it re-executes itself with
``.venv\\Scripts\\python.exe`` (Windows) or ``.venv/bin/python`` (Linux).
A loop guard is passed to the child through the ``GNSS_SIM_VENV_REEXEC``
environment variable, so the second launch imports the GUI instead of
re-executing again.

The earlier version was never reached when the *system* interpreter started
``run.py`` on Windows: ``run.py`` did ``from gnss_sim.venv import maybe_reexec``,
and importing the ``gnss_sim`` package runs ``__init__`` -> ``constants`` ->
``import numpy``.  The system interpreter has no NumPy, so the process died
with ``ModuleNotFoundError`` *before* the re-exec could run — only
``.venv\\Scripts\\python.exe run.py`` worked.  ``run.py`` now loads this module
directly from its file (see ``_load_venv_module``), which needs no third-party
packages.  Detection also checks ``sys.prefix`` (authoritative inside a venv)
on top of ``sys.executable``, and the child is launched with
:func:`subprocess.call` (not ``os.execv``: on Windows ``os.execv`` does not
reliably propagate the child exit code / console).  Paths are passed as a list
so spaces are quoted correctly.
"""

from __future__ import annotations

import os
import subprocess
import sys

#: Set to ``"1"`` in the re-executed child to prevent an infinite loop.
REEXEC_ENV = "GNSS_SIM_VENV_REEXEC"


def project_root() -> str:
    """Absolute path of the project root (the parent of this package)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def venv_dir(root: str | None = None) -> str:
    """Absolute path of the project-local virtualenv directory."""
    return os.path.join(root or project_root(), ".venv")


def venv_python(root: str | None = None) -> str:
    """The canonical project-local interpreter path for this OS."""
    root = root or project_root()
    if sys.platform.startswith("win"):
        return os.path.join(root, ".venv", "Scripts", "python.exe")
    return os.path.join(root, ".venv", "bin", "python")


def venv_candidates(root: str | None = None) -> list[str]:
    """Interpreter paths to try, in order, for this OS."""
    root = root or project_root()
    if sys.platform.startswith("win"):
        return [os.path.join(root, ".venv", "Scripts", "python.exe")]
    base = os.path.join(root, ".venv", "bin")
    return [os.path.join(base, "python"), os.path.join(base, "python3")]


def find_venv_python(root: str | None = None) -> str:
    """First existing venv interpreter, else :func:`venv_python`."""
    for candidate in venv_candidates(root):
        if os.path.isfile(candidate):
            return candidate
    return venv_python(root)


def venv_available(root: str | None = None) -> bool:
    """True when the project-local interpreter actually exists."""
    return os.path.isfile(find_venv_python(root))


def _normcase(path: str) -> str:
    try:
        return os.path.normcase(os.path.realpath(path))
    except OSError:  # pragma: no cover - realpath is best effort
        return os.path.normcase(os.path.abspath(path))


def same_interpreter(exe: str | None = None, venv: str | None = None) -> bool:
    """True when ``exe`` (default ``sys.executable``) is the venv python."""
    exe = exe or sys.executable
    venv = venv or venv_python()
    return _normcase(exe) == _normcase(venv)


def _prefix_is_venv(root: str | None = None) -> bool:
    """True when ``sys.prefix`` points at the project-local venv.

    ``sys.executable`` can be a shim/launcher path while ``sys.prefix`` still
    names the active environment; comparing the prefix makes detection robust
    (e.g. ``python -m venv`` wrappers on Windows).
    """
    try:
        return _normcase(sys.prefix) == _normcase(venv_dir(root))
    except OSError:  # pragma: no cover
        return False


def needs_reexec(root: str | None = None, exe: str | None = None,
                env: dict | None = None) -> bool:
    """True when ``run.py`` should re-exec under the project-local venv."""
    env = os.environ if env is None else env
    if str(env.get(REEXEC_ENV, "")) == "1":
        return False  # already re-executed: never loop
    venv = find_venv_python(root)
    if not os.path.isfile(venv):
        return False  # no local venv: run with the current interpreter
    if exe is None:
        # The interpreter that is running us: trust sys.executable, then the
        # venv prefix (authoritative) before deciding to re-exec.
        if _normcase(sys.executable) == _normcase(venv):
            return False
        if _prefix_is_venv(root):
            return False
        return True
    return _normcase(exe) != _normcase(venv)


def reexec_argv(argv: list[str] | None = None, root: str | None = None,
                exe: str | None = None) -> list[str] | None:
    """Command that re-runs ``run.py`` under the venv, or ``None``."""
    if not needs_reexec(root, exe):
        return None
    root = root or project_root()
    script = os.path.join(root, "run.py")
    return [find_venv_python(root), script] + list(argv or [])


def _spawn(cmd: list[str], root: str | None = None) -> int:
    """Run ``cmd`` with the loop guard; return its exit code."""
    env = dict(os.environ)
    env[REEXEC_ENV] = "1"
    # subprocess (list form) quotes the interpreter/script paths correctly and
    # returns the child's exit code; ``os.execv`` on Windows does neither
    # reliably.  ``cwd`` keeps relative CLI arguments meaningful.
    return subprocess.call(cmd, env=env, cwd=root or project_root())


def maybe_reexec(argv: list[str] | None = None, root: str | None = None,
                 exe: str | None = None) -> int | None:
    """Re-exec ``run.py`` under the venv; return the child exit code/``None``.

    ``None`` means no re-exec happened (already the venv interpreter or no
    local venv), and the caller should continue importing the GUI.
    """
    cmd = reexec_argv(argv, root, exe)
    if cmd is None:
        return None
    return _spawn(cmd, root)


def reexec_module_argv(module: str, argv: list[str] | None = None,
                       root: str | None = None,
                       exe: str | None = None) -> list[str] | None:
    """Command that re-runs ``python -m <module>`` under the venv, or ``None``."""
    if not needs_reexec(root, exe):
        return None
    root = root or project_root()
    return [find_venv_python(root), "-m", module] + list(argv or [])


def maybe_reexec_module(module: str, argv: list[str] | None = None,
                        root: str | None = None,
                        exe: str | None = None) -> int | None:
    """Re-exec ``python -m <module>`` under the venv; return child code/``None``."""
    cmd = reexec_module_argv(module, argv, root, exe)
    if cmd is None:
        return None
    return _spawn(cmd, root)
