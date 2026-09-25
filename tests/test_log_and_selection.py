"""Offline tests for the UHD log/stderr handling and unified system selection.

No hardware, no network::

    E:\\MySoftware\\SDR_Scan\\.venv\\Scripts\\python.exe -m pytest tests -q
"""

from __future__ import annotations

import os
import subprocess
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from gnss_sim.config import derive_band_key, derive_nav_mode  # noqa: E402
from gnss_sim.nativelog import (  # noqa: E402
    DEFAULT_UHD_LOG_LEVEL,
    clean_native_text,
    quiet_uhd,
)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _app():
    from PyQt5 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


# ======================================================================
# 1) UHD log level / native stderr handling
# ======================================================================
def test_quiet_uhd_defaults_to_fatal(monkeypatch) -> None:
    monkeypatch.delenv("UHD_LOG_LEVEL", raising=False)
    assert quiet_uhd() == DEFAULT_UHD_LOG_LEVEL == "fatal"
    assert os.environ["UHD_LOG_LEVEL"] == "fatal"
    # An explicit level/override wins.
    assert quiet_uhd("error") == "error"
    assert os.environ["UHD_LOG_LEVEL"] == "error"


def test_clean_native_text_strips_lone_markers() -> None:
    # The B200 async handler emits bare U/O/L characters without a newline.
    assert clean_native_text("UU[INFO] ok\n") == "[INFO] ok\n"
    assert clean_native_text("O") == ""
    assert clean_native_text("UOL") == ""
    # Real words that merely contain U/O/L are preserved.
    assert clean_native_text("UHD USB OK") == "UHD USB OK"
    assert clean_native_text("B200 Operating over USB 3.\n") == (
        "B200 Operating over USB 3.\n")


def test_native_stderr_filter_suppresses_markers(tmp_path) -> None:
    """The fd-2 capture drops U/O markers but forwards real native text."""
    snippet = (
        "import os, sys, time\n"
        "sys.path.insert(0, r'{root}')\n"
        "from gnss_sim.nativelog import install_native_stderr_filter\n"
        "install_native_stderr_filter(force=True)\n"
        "os.write(2, b'UU')\n"
        "os.write(2, b'[INFO] native ok\\n')\n"
        "sys.stdout.flush()\n"
        "time.sleep(0.4)\n"
    ).format(root=_ROOT)
    script = tmp_path / "native_probe.py"
    script.write_text(snippet, encoding="utf-8")
    proc = subprocess.run([sys.executable, str(script)], capture_output=True,
                          env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    assert proc.returncode == 0
    assert proc.stderr == b""
    assert b"[INFO] native ok" in proc.stdout
    assert b"U" not in proc.stdout  # the bare marker was filtered


# ======================================================================
# 2) Unified selection helpers
# ======================================================================
def test_derive_band_key_from_systems() -> None:
    # Default GUI state (every system, no opt-in) -> conservative narrow L1;
    # B1I is dropped by the runner instead of silently widening the stream.
    assert derive_band_key() == "l1"
    assert derive_band_key(combine=True) == "all"
    # BeiDou only -> narrowband b1i.
    assert derive_band_key(
        enable_ca=False, enable_l1c=False, enable_galileo=False,
        enable_qzss=False, enable_sbas=False, enable_beidou=True) == "b1i"
    # No BeiDou -> L1 session (L1C/E1 etc. only influence the computed fs).
    assert derive_band_key(
        enable_beidou=False, enable_ca=True) == "l1"
    assert derive_band_key(
        enable_beidou=False, enable_ca=False, enable_l1c=False,
        enable_galileo=True) == "l1"
    # combine only matters when both groups are present.
    assert derive_band_key(
        enable_beidou=False, combine=True) == "l1"


def test_derive_nav_mode_from_systems() -> None:
    assert derive_nav_mode(enable_galileo=True, enable_qzss=True,
                           enable_beidou=True) == "merged"
    assert derive_nav_mode(enable_galileo=False, enable_qzss=False,
                           enable_beidou=True) == "merged"
    assert derive_nav_mode(enable_galileo=False, enable_qzss=False,
                           enable_beidou=False) == "auto"


def test_gui_signals_are_single_source_of_truth() -> None:
    from gnss_sim.gui import MainWindow
    app = _app()
    win = MainWindow()
    try:
        # No redundant band/session and ephemeris-mode selectors.
        for attr in ("cmb_band", "cmb_nav_mode", "cmb_fs", "cmb_fc"):
            assert not hasattr(win, attr)
        # fs/centre/band are read-only labels driven by the checkboxes.
        assert win.lbl_band.text() and win.lbl_fs.text() and win.lbl_fc.text()
        win.cb_bds.setChecked(False)
        win.cb_l1c.setChecked(False)
        win.cb_gal.setChecked(False)
        win.cb_qzss.setChecked(False)
        win.cb_sbas.setChecked(False)
        cfg = win._collect()
        assert cfg.band == "l1"
        assert cfg.fs == 2_600_000.0
        assert cfg.center_freq == 1_575_420_000.0
        assert cfg.fs_override is False and cfg.center_override is False
    finally:
        win.close()
    del app
