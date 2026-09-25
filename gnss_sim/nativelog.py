"""Keep UHD's native stderr out of the console, without hiding real errors.

Background
----------
UHD writes ``[INFO]``/``[WARNING]``/``[ERROR]`` lines to **stderr**.  On Windows
PowerShell renders any native stderr output as orange/red ``NativeCommandError``
even when the process exits 0.

Worse, the B200 async handler calls UHD's ``standard_async_msg_prints()``
(see ``b200_impl::handle_async_task`` in ``b200_io_impl.cpp``) for every TX
async event.  That function writes a **bare ``U``/``O`` character straight to
``std::cerr``** (underflow/overflow/late markers) and flushes — it does *not*
go through the UHD logging system, so ``UHD_LOG_LEVEL`` cannot silence it.  Those
are the ``UU`` / ``OOOO`` glyphs the user saw.

Strategy
--------
1. :func:`quiet_uhd` sets the quietest valid UHD log level (``fatal``) before
   ``uhd`` is imported.
2. :func:`install_native_stderr_filter` redirects the process file descriptor 2
   into a pipe read by a daemon thread.  The thread strips the lone under/overflow
   markers with :func:`clean_native_text` and forwards everything else to the
   simulator's log / stdout, so genuine native errors stay visible while the
   console stays clean.

The Python UHD bindings (4.10) do not expose the native logging API, so the
environment variable is the only available knob; this is documented in the
README.
"""

from __future__ import annotations

import os
import re
import sys
import threading
from typing import Callable

#: Quietest valid UHD log level (only ``fatal`` messages survive).
DEFAULT_UHD_LOG_LEVEL = "fatal"

#: Underflow/overflow/late markers printed by UHD's ``standard_async_msg_prints``
#: as standalone characters.  A run of them is removed unless it is glued to a
#: normal word (so real text such as ``UHD``/``USB`` is preserved).
_MARKER_RE = re.compile(r"(?<![0-9A-Za-z])[UOL]+(?![0-9A-Za-z])")

_installed = False
_sink: Callable[[str], None] | None = None
_saved_stderr_fd: int | None = None
_reader_thread: threading.Thread | None = None


def quiet_uhd(level: str | None = None) -> str:
    """Set ``UHD_LOG_LEVEL`` to ``fatal`` (or ``level``) before ``import uhd``.

    Must run before UHD is imported.  A caller-provided level (or an explicit
    ``UHD_LOG_LEVEL`` already in the environment) wins.  Returns the level used.
    """
    chosen = level or os.environ.get("UHD_LOG_LEVEL") or DEFAULT_UHD_LOG_LEVEL
    os.environ["UHD_LOG_LEVEL"] = chosen
    return chosen


def clean_native_text(text: str) -> str:
    """Drop the bare UHD under/overflow/late markers from native output.

    ``"UU[INFO] ok" -> "[INFO] ok"`` while ``"UHD USB"`` is left intact.
    """
    if not text:
        return text
    return _MARKER_RE.sub("", text)


def set_native_stderr_sink(sink: Callable[[str], None] | None) -> None:
    """Route forwarded native text to ``sink`` (e.g. the GUI log) or stdout."""
    global _sink
    _sink = sink


def _forward(text: str) -> None:
    if not text:
        return
    if _sink is not None:
        try:
            _sink(text)
            return
        except Exception:  # noqa: BLE001 - logging must never abort the run
            pass
    try:
        sys.stdout.write(text)
        sys.stdout.flush()
    except Exception:  # noqa: BLE001
        pass


def _reader_loop(read_fd: int) -> None:
    try:
        while True:
            data = os.read(read_fd, 4096)
            if not data:
                break
            _forward(clean_native_text(data.decode("utf-8", "replace")))
    except Exception:  # noqa: BLE001 - thread dies with the process
        pass


def _redirect_windows_stderr_handle(fd: int) -> None:
    """Point ``STD_ERROR_HANDLE`` at ``fd`` so UHD's own CRT picks it up.

    ``os.dup2`` only updates Python's CRT tables; the UHD DLL links its own
    C runtime, so ``std::cerr`` keeps writing to the *original* stderr handle
    unless we also replace the process standard-error handle before the DLL is
    loaded.  Best effort, Windows only.
    """
    if not sys.platform.startswith("win"):
        return
    try:
        import ctypes
        import msvcrt
        handle = msvcrt.get_osfhandle(fd)
        ctypes.windll.kernel32.SetStdHandle(-12, ctypes.c_void_p(handle))
    except Exception:  # noqa: BLE001 - never fail because of this
        pass


def install_native_stderr_filter(force: bool = False) -> bool:
    """Capture native fd 2 and forward it (minus markers) to the log/stdout.

    Idempotent.  Skipped under pytest unless ``force=True`` so the test suite's
    own stream capture is not disturbed.  Returns ``True`` when it installed.
    """
    global _installed, _saved_stderr_fd, _reader_thread
    if _installed:
        return False
    # Only skip when a pytest test item is actually running (``cupy`` imports
    # pytest at startup, so ``"pytest" in sys.modules`` is not a valid signal).
    if not force and os.environ.get("PYTEST_CURRENT_TEST"):
        return False
    if not hasattr(os, "pipe"):  # pragma: no cover - exotic platform
        return False
    try:
        read_fd, write_fd = os.pipe()
        _saved_stderr_fd = os.dup(2)
        os.dup2(write_fd, 2)
        # Point STD_ERROR_HANDLE at fd 2 *after* the duplication and before the
        # original write handle is closed: SetStdHandle stores a raw handle and
        # does not duplicate it, so pointing it at ``write_fd`` and then
        # ``os.close(write_fd)`` left UHD's own CRT writing to a closed handle
        # (native lines never reached the journal).  fd 2 stays valid.
        _redirect_windows_stderr_handle(2)
        os.close(write_fd)
    except Exception:  # noqa: BLE001 - keep working with a raw console
        return False
    _installed = True
    _reader_thread = threading.Thread(
        target=_reader_loop, args=(read_fd,), name="gnss-native-stderr",
        daemon=True)
    _reader_thread.start()
    return True


__all__ = [
    "DEFAULT_UHD_LOG_LEVEL",
    "clean_native_text",
    "install_native_stderr_filter",
    "quiet_uhd",
    "set_native_stderr_sink",
]
