"""Native UHD TX helper integration (``native/gnss_sim_tx.exe``).

The native player (:file:`native/tx_player.cpp`) streams a ``cs16`` file to a
USRP from a tight C++ loop, without the per-block Python/GIL overhead that
caused TX underflows at ~25 Msps with :class:`gnss_sim.uhd_tx.UhdTxSink`.

This module locates the helper, spawns it with the active scenario's radio
parameters, routes its stderr into the simulator journal and reports the
underflow/short-send counters it emits.  When the executable is missing the
runner falls back to the built-in Python UHD path with a clear journal note.
"""

from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from typing import Callable

#: Repository root (one level above this package).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE_NAME = "gnss_sim_tx.exe"

_UNDERFLOW_RE = re.compile(r"underflow\s+(\d+)", re.IGNORECASE)
_SHORT_RE = re.compile(r"short\s+(\d+)", re.IGNORECASE)
_SEQ_RE = re.compile(r"seq_err\s+(\d+)", re.IGNORECASE)
_SIM_RE = re.compile(r"sim\s+([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
_DONE_RE = re.compile(r"sent\s+\d+\s+samp\s+\(([0-9]+(?:\.[0-9]+)?)\s*s\)",
                      re.IGNORECASE)


def native_tx_path() -> str | None:
    """Return the path to ``gnss_sim_tx.exe`` or ``None`` when absent.

    ``GNSS_SIM_NATIVE_TX`` overrides the location (useful for tests/CI); the
    default is ``<repo>/native/gnss_sim_tx.exe``.
    """
    candidates: list[str] = []
    env = os.environ.get("GNSS_SIM_NATIVE_TX", "").strip()
    if env:
        candidates.append(env)
    candidates.append(os.path.join(_REPO_ROOT, "native", EXE_NAME))
    for cand in candidates:
        if cand and os.path.isfile(cand):
            return cand
    return None


def native_tx_available() -> bool:
    """True when the native TX helper can be found."""
    return native_tx_path() is not None


def uhd_bin_dir() -> str:
    """Directory holding ``uhd.dll`` (from ``UHD_PKG_PATH``) or ``""``."""
    pkg = os.environ.get("UHD_PKG_PATH", "").strip()
    for cand in ([os.path.join(pkg, "bin")] if pkg else []) + [
            r"C:\Program Files\UHD\bin"]:
        if os.path.isdir(cand):
            return cand
    return ""


class NativeTxResult:
    """Counters parsed from the native player's stderr."""

    def __init__(self) -> None:
        self.underflows = 0
        self.short_sends = 0
        self.seq_errors = 0
        self.sent_sim_s = 0.0
        self.returncode: int | None = None


class NativeTxSink:
    """Runner sink that plays pre-generated samples through the native helper.

    It is not a streaming sink: the runner writes the whole segment to a temp
    ``cs16`` file and then calls :meth:`play`.  ``write`` is deliberately
    unsupported so a mis-routed path fails loudly instead of silently dropping
    samples.
    """

    def __init__(self, cfg, log: Callable[[str], None] | None = None) -> None:
        self.cfg = cfg
        self.exe = native_tx_path()
        self._log = log
        self._proc: subprocess.Popen | None = None
        self._underflows = 0
        self._short_sends = 0
        self._seq_errors = 0
        self._events: list[tuple[float, float, int]] = []
        self._count = 0
        self._gain = float(getattr(cfg, "tx_gain", 18.0))
        #: B210 TX gain range (0..89.75 dB); clamped in the native helper too.
        self.gain_range = (0.0, 89.75)
        self.gain_warning: str | None = None
        self._start_time: float | None = None
        self.max_num_samps = 2040
        # Kept for interface parity with UhdTxSink.
        self.sample_rate = float(getattr(cfg, "fs", 0.0) or 0.0)
        self.center_freq = float(getattr(cfg, "center_freq", 0.0) or 0.0)
        self.tx_bandwidth = float(getattr(cfg, "tx_bandwidth", 0.0) or 0.0)

    # -- Sink interface -------------------------------------------------
    def start(self) -> None:
        self._start_time = time.time()

    def write(self, samples) -> None:  # pragma: no cover - not used natively
        raise RuntimeError(
            "NativeTxSink не принимает блоки: сегмент сначала пишется в файл, "
            "затем играется native/gnss_sim_tx.exe")

    def close(self) -> None:
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=3.0)
            except Exception:  # noqa: BLE001 - best effort
                try:
                    proc.kill()
                except Exception:  # noqa: BLE001
                    pass

    @property
    def count(self) -> int:
        return self._count

    @property
    def underflows(self) -> int:
        return self._underflows

    @property
    def underflow_events(self) -> list[tuple[float, float, int]]:
        return list(self._events)

    def set_tx_gain(self, gain: float) -> float:
        """Remember a new gain; live changes are not possible natively."""
        self._gain = float(gain)
        if self._proc is not None and self._proc.poll() is None:
            msg = ("Native TX: изменение усиления на лету недоступно "
                   f"(запомнено {self._gain:g} дБ для следующего запуска)")
            if self.gain_warning != msg:
                self.gain_warning = msg
                self._warn(msg)
        return self._gain

    def describe(self) -> str:
        return (f"native {self.exe} -> B210 ch{getattr(self.cfg, 'tx_channel', 0)} "
                f"{self.center_freq / 1e6:.3f} МГц, "
                f"{self.sample_rate / 1e6:g} Мвыб/с, gain {self._gain:g} дБ "
                f"(без Python/GIL в петле отправки)")

    def get_info(self) -> dict:
        return {
            "native": True,
            "exe": self.exe,
            "channel": int(getattr(self.cfg, "tx_channel", 0)),
            "sample_rate": self.sample_rate,
            "center_freq": self.center_freq,
            "gain": self._gain,
            "gain_range": list(self.gain_range),
        }

    # -- native playback ------------------------------------------------
    def _warn(self, msg: str) -> None:
        if self._log is not None:
            try:
                self._log(msg)
            except Exception:  # noqa: BLE001
                pass

    def _build_cmd(self, path: str, loop: bool, seconds: float) -> list[str]:
        cfg = self.cfg
        assert self.exe is not None
        cmd = [
            self.exe,
            "--file", path,
            "--rate", repr(float(cfg.fs)),
            "--freq", repr(float(cfg.center_freq)),
            "--gain", repr(self._gain),
            "--antenna", str(getattr(cfg, "tx_antenna", "TX/RX") or "TX/RX"),
            "--args", str(getattr(cfg, "uhd_args", "type=b200") or "type=b200"),
            "--channel", str(int(getattr(cfg, "tx_channel", 0))),
            "--clock-source", str(getattr(cfg, "clock_source", "internal")
                                  or "internal"),
            "--format", "cs16",
            "--frame-size", "16384",
            "--num-frames", "32",
        ]
        bw = float(getattr(cfg, "tx_bandwidth", 0.0) or 0.0)
        if bw > 0:
            cmd += ["--bandwidth", repr(bw)]
        if loop:
            cmd.append("--loop")
        if seconds > 0:
            cmd += ["--seconds", repr(float(seconds))]
        return cmd

    def play(self, path: str, loop: bool, seconds: float,
             stop_event: threading.Event | None = None,
             on_status: Callable[[float], None] | None = None,
             on_underflow: Callable[[int], None] | None = None) -> NativeTxResult:
        """Run the native helper on ``path`` until it exits or is stopped."""
        if not self.exe:
            raise RuntimeError("native/gnss_sim_tx.exe не найден")
        result = NativeTxResult()
        cmd = self._build_cmd(path, loop, seconds)
        self._warn("TX native: " + " ".join(cmd))
        env = dict(os.environ)
        bindir = uhd_bin_dir()
        if bindir:
            env["PATH"] = bindir + os.pathsep + env.get("PATH", "")
        env.setdefault("UHD_LOG_LEVEL", "info")
        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._start_time = time.time()
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env, creationflags=creationflags)
        self._proc = proc

        def reader() -> None:
            assert proc.stdout is not None
            for raw in proc.stdout:
                line = raw.rstrip()
                if not line:
                    continue
                self._warn(line)
                m = _UNDERFLOW_RE.search(line)
                if m:
                    val = int(m.group(1))
                    if val > result.underflows:
                        result.underflows = val
                        self._underflows = val
                        if on_underflow is not None:
                            try:
                                on_underflow(val)
                            except Exception:  # noqa: BLE001
                                pass
                m = _SHORT_RE.search(line)
                if m:
                    result.short_sends = max(result.short_sends, int(m.group(1)))
                    self._short_sends = result.short_sends
                m = _SEQ_RE.search(line)
                if m:
                    result.seq_errors = max(result.seq_errors, int(m.group(1)))
                    self._seq_errors = result.seq_errors
                m = _SIM_RE.search(line)
                if m:
                    result.sent_sim_s = float(m.group(1))
                    self._count = int(result.sent_sim_s * self.sample_rate)
                    if on_status is not None:
                        try:
                            on_status(result.sent_sim_s)
                        except Exception:  # noqa: BLE001
                            pass
                m = _DONE_RE.search(line)
                if m:
                    result.sent_sim_s = max(result.sent_sim_s, float(m.group(1)))
                    self._count = int(result.sent_sim_s * self.sample_rate)

        thread = threading.Thread(target=reader, name="gnss-sim-native-tx",
                                  daemon=True)
        thread.start()
        try:
            while proc.poll() is None:
                if stop_event is not None and stop_event.is_set():
                    self.close()
                    break
                time.sleep(0.1)
            if stop_event is not None and stop_event.is_set() and proc.poll() is None:
                self.close()
        finally:
            thread.join(timeout=3.0)
            result.returncode = proc.poll()
            self._proc = None
        self._underflows = max(self._underflows, result.underflows)
        if result.returncode not in (0, None):
            self._warn(f"TX native: процесс завершился с кодом "
                       f"{result.returncode}")
        return result


__all__ = [
    "EXE_NAME",
    "NativeTxResult",
    "NativeTxSink",
    "native_tx_available",
    "native_tx_path",
    "uhd_bin_dir",
]
