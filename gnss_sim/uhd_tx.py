"""UHD (USRP B210) TX streamer for GNSS baseband playback.

The module imports :mod:`uhd` lazily so the rest of the simulator works (and
shows a readable error) when UHD is not installed.  This mirrors the receive
wrapper used by the SDR_Scan application but drives the TX chain.
"""

from __future__ import annotations

import queue
import sys
import threading
from typing import Any

import numpy as np

_IS_WIN = sys.platform.startswith("win")
_UHD_HINT = (
    "Не удалось импортировать модуль 'uhd'. Установите UHD:\n"
    "  Windows: UHD-инсталлятор (задаёт UHD_PKG_PATH, ставит uhd.dll) + "
    "pip install uhd==<версия>\n"
    "  Linux:   sudo apt install libuhd-dev uhd-host python3-uhd"
)


class TxError(RuntimeError):
    """Ошибка передающего тракта."""


#: Substrings that mark a USB/device loss (``EnvironmentError``/``OSError`` is
#: also treated as such) — the B210 reports ``LIBUSB_TRANSFER_NO_DEVICE`` when
#: the link drops mid-stream.
_USB_DROP_HINTS = (
    "libusb_transfer_no_device", "no_device", "no device", "usb_error",
    "usb error", "broken pipe", "device not found", "device is not available",
)


def classify_tx_error(exc: BaseException) -> TxError:
    """Turn a low-level send failure into a clear Russian :class:`TxError`.

    USB/libusb drops (including ``LIBUSB_TRANSFER_NO_DEVICE``) are reported as
    a link loss so the run stops cleanly instead of raising a raw traceback.
    """
    if isinstance(exc, TxError):
        return exc
    text = str(exc).lower()
    if isinstance(exc, (EnvironmentError, OSError)) or any(
            hint in text for hint in _USB_DROP_HINTS):
        return TxError(
            "Обрыв USB/устройства во время передачи "
            f"({exc}) — передача остановлена. Проверьте кабель USB3 и питание "
            "B210, уменьшите fs; при необходимости переподключите устройство.")
    return TxError(f"Ошибка передачи UHD: {exc}")


# ----------------------------------------------------------------------
# UHD metadata error handling.
#
# The Python bindings differ between UHD releases: some expose
# ``uhd.types.TXMetadataErrorCode`` / ``RXMetadataErrorCode``, some do not.
# Never reference a missing attribute — introspect ``dir(uhd.types)`` and
# fall back to comparing ``md.error_code`` by name or by its numeric value.
# ----------------------------------------------------------------------
_TX_CODE_INDEX = {
    "none": 0, "timeout": 1, "underflow": 2, "late": 3, "burst_error": 4,
    "burst_seq_error": 5, "alignment_error": 6, "bad_packet": 7, "overflow": 8,
}
_RX_CODE_INDEX = {
    "none": 0, "timeout": 1, "overflow": 2, "late": 3, "burst_error": 4,
    "burst_seq_error": 5, "alignment_error": 6, "bad_packet": 7,
}


def _enum_from_types(uhd, *names):
    types = getattr(uhd, "types", None)
    if types is None:
        return None
    for name in names:
        if hasattr(types, name):
            return getattr(types, name)
    return None


def tx_error_code_enum(uhd):
    """Return ``uhd.types.TXMetadataErrorCode`` or ``None`` if absent."""
    return _enum_from_types(uhd, "TXMetadataErrorCode", "TxMetadataErrorCode")


def rx_error_code_enum(uhd):
    """Return ``uhd.types.RXMetadataErrorCode`` or ``None`` if absent."""
    return _enum_from_types(uhd, "RXMetadataErrorCode", "RxMetadataErrorCode")


def _error_matches(md, kind: str, enum, index: dict) -> bool:
    code = getattr(md, "error_code", None)
    if code is None:
        return False
    if enum is not None:
        try:
            if code == getattr(enum, kind):
                return True
        except Exception:  # noqa: BLE001 - enum без нужного имени
            pass
    if kind in str(code).lower():
        return True
    idx = index.get(kind)
    if idx is not None:
        try:
            if int(code) == idx:
                return True
        except (TypeError, ValueError):
            pass
    return False


def tx_has_error(md, kind: str, enum=None) -> bool:
    """True when TX metadata ``md`` carries error ``kind`` (underflow/…)."""
    return _error_matches(md, kind, enum, _TX_CODE_INDEX)


def rx_has_error(md, kind: str, enum=None) -> bool:
    """True when RX metadata ``md`` carries error ``kind`` (overflow/…)."""
    return _error_matches(md, kind, enum, _RX_CODE_INDEX)


def gain_range_bounds(rng: Any) -> tuple[float, float]:
    """Return ``(min, max)`` from a UHD gain range or a plain pair.

    UHD exposes the TX gain range as ``uhd.types.gain_range_t`` /
    ``MetaRange``; depending on the release the bounds are methods
    (``start()``/``stop()``), attributes (``min``/``max``) or items.  A plain
    ``(lo, hi)`` tuple is accepted too, which keeps :func:`clamp_gain`
    testable without hardware.
    """
    if rng is None:
        return 0.0, 0.0
    if isinstance(rng, (tuple, list)) and len(rng) >= 2:
        return float(rng[0]), float(rng[1])
    for lo_name, hi_name in (("start", "stop"), ("min", "max")):
        try:
            lo = getattr(rng, lo_name)
            hi = getattr(rng, hi_name)
            return (float(lo()), float(hi())) if callable(lo) else (float(lo), float(hi))
        except Exception:  # noqa: BLE001 - перебираем варианты
            continue
    try:
        return float(rng[0]), float(rng[1])
    except Exception:  # noqa: BLE001
        return 0.0, 0.0


def clamp_gain(gain: float, gain_range: tuple[float, float]
               ) -> tuple[float, str | None]:
    """Clamp ``gain`` into ``gain_range`` and describe the change.

    Returns ``(applied_gain, warning)`` where ``warning`` is a clear Russian
    message when the request was outside the hardware range and ``None`` when
    it was already valid.  B210 TX gain is roughly ``0..89.75 dB``, so a
    negative gain (a common way to ask for less power) must be clamped to the
    minimum; real power reduction needs an external attenuator.
    """
    try:
        lo, hi = float(gain_range[0]), float(gain_range[1])
    except (TypeError, ValueError, IndexError):
        return float(gain), None
    if hi < lo:  # защита от перепутанного порядка
        lo, hi = hi, lo
    value = float(gain)
    if value < lo:
        return lo, (
            f"ВНИМАНИЕ: запрошенное TX усиление {value:g} дБ вне диапазона "
            f"устройства {lo:g}..{hi:g} дБ — применено {lo:g} дБ "
            f"(снижение мощности возможно только внешним аттенюатором)")
    if value > hi:
        return hi, (
            f"ВНИМАНИЕ: запрошенное TX усиление {value:g} дБ вне диапазона "
            f"устройства {lo:g}..{hi:g} дБ — применено {hi:g} дБ")
    return value, None


def uhd_available() -> bool:
    try:
        import uhd  # noqa: F401
    except Exception:
        return False
    return True


def uhd_version() -> str:
    try:
        import uhd
    except Exception:
        return "не установлен"
    try:
        return str(uhd.__version__)
    except Exception:
        return "неизвестно"


def list_devices(args: str = "") -> list[dict]:
    try:
        import uhd
    except Exception as exc:  # pragma: no cover
        raise TxError(_UHD_HINT) from exc
    try:
        return [dict(d) for d in uhd.find(args)]
    except Exception as exc:  # pragma: no cover
        raise TxError(f"Поиск устройств не удался: {exc}") from exc


class UhdTxSink:
    """Streams ``complex64`` blocks to a USRP TX channel."""

    def __init__(
        self,
        args: str = "type=b200",
        channel: int = 0,
        sample_rate: float = 2.6e6,
        center_freq: float = 1575.42e6,
        gain: float = 0.0,
        antenna: str = "TX/RX",
        bandwidth: float = 2.5e6,
        clock_source: str = "internal",
        log=None,
    ) -> None:
        try:
            import uhd
        except Exception as exc:  # pragma: no cover
            raise TxError(_UHD_HINT) from exc

        self._uhd = uhd
        self.args = args
        self.channel = int(channel)
        self._log = log
        self.gain_warning: str | None = None
        self._gain_range = (0.0, 0.0)
        self._lock = threading.RLock()
        self._streamer: Any = None
        self._buff: np.ndarray | None = None
        self._max_samps = 0
        self._count = 0
        self._sent = 0
        self._underflows = 0
        self._started = False
        self._queue: "queue.Queue[np.ndarray | None] | None" = None
        self._sender: threading.Thread | None = None
        self._sender_error: TxError | None = None

        with self._lock:
            self.usrp = uhd.usrp.MultiUSRP(args)
            if clock_source:
                self.usrp.set_clock_source(clock_source)
            self.usrp.set_tx_rate(float(sample_rate), self.channel)
            self.usrp.set_tx_freq(uhd.types.TuneRequest(float(center_freq)),
                                  self.channel)
            # Query and honour the device TX gain range: a negative request
            # (e.g. ``--tx-gain -30``) is invalid for the B210 (0..~89.75 dB)
            # and must be clamped, not silently accepted.
            try:
                bounds = gain_range_bounds(
                    self.usrp.get_tx_gain_range(self.channel))
            except Exception:  # noqa: BLE001 - старые/усечённые биндинги
                bounds = (0.0, 0.0)
            if bounds[1] > bounds[0]:
                self._gain_range = bounds
                applied_gain, self.gain_warning = clamp_gain(
                    gain, self._gain_range)
            else:
                # Range unavailable: do not clamp, just honour the request.
                applied_gain, self.gain_warning = float(gain), None
            if self.gain_warning:
                self._warn(self.gain_warning)
            self.usrp.set_tx_gain(applied_gain, self.channel)
            # Only force an analog bandwidth when the user asked for one;
            # 0/None lets UHD pick a sensible default.
            if bandwidth and float(bandwidth) > 0:
                try:
                    self.usrp.set_tx_bandwidth(float(bandwidth), self.channel)
                except Exception:
                    pass
            if antenna:
                try:
                    self.usrp.set_tx_antenna(str(antenna), self.channel)
                except Exception:
                    pass
            self._make_streamer()

        # Requested values are stored in cfg; here we read back the ACTUAL
        # values configured in the hardware/driver.
        self.requested_sample_rate = float(sample_rate)
        self.requested_tx_bandwidth = float(bandwidth or 0.0)
        self.requested_gain = float(gain)
        self.sample_rate = float(self.usrp.get_tx_rate(self.channel))
        self.center_freq = float(self.usrp.get_tx_freq(self.channel))
        self.gain = float(self.usrp.get_tx_gain(self.channel))
        try:
            self.tx_bandwidth = float(self.usrp.get_tx_bandwidth(self.channel))
        except Exception:
            self.tx_bandwidth = float(bandwidth or 0.0)

    # ------------------------------------------------------------------
    def _warn(self, msg: str) -> None:
        if self._log is not None:
            try:
                self._log(msg)
                return
            except Exception:  # noqa: BLE001
                pass
        import warnings
        warnings.warn(msg, stacklevel=2)

    @property
    def gain_range(self) -> tuple[float, float]:
        """Device TX gain range ``(min, max)`` in dB."""
        return self._gain_range

    def _make_streamer(self) -> None:
        uhd = self._uhd
        st_args = uhd.usrp.StreamArgs("fc32", "sc16")
        st_args.channels = [self.channel]
        self._streamer = self.usrp.get_tx_stream(st_args)
        self._max_samps = int(self._streamer.get_max_num_samps())
        self._buff = np.zeros((1, self._max_samps), dtype=np.complex64)

    @property
    def max_num_samps(self) -> int:
        return self._max_samps

    def start(self) -> None:
        """Start the dedicated sender thread (idempotent).

        A separate thread drains a small bounded queue and calls UHD's
        ``send`` in a tight loop.  Coupled with the pre-generated RAM segment
        this keeps the device fed (a few blocks of slack) instead of leaving
        Python-heavy gaps between sends, which is what caused the 0.94x
        realtime rate and the TX underflows.
        """
        with self._lock:
            if self._started:
                return
            self._started = True
            self._sender_error = None
            self._queue = queue.Queue(maxsize=8)
            self._sender = threading.Thread(target=self._sender_loop,
                                            name="gnss-sim-uhd-tx", daemon=True)
            self._sender.start()

    def _raise_sender_error(self) -> TxError:
        exc = self._sender_error
        if exc is not None:
            return exc
        return TxError("Передача остановлена")

    def write(self, samples: np.ndarray) -> None:
        """Queue ``samples`` for the sender thread (blocks on back-pressure)."""
        if not self._started:
            self.start()
        s = np.ascontiguousarray(samples, dtype=np.complex64).ravel()
        if s.size == 0:
            return
        q = self._queue
        assert q is not None
        while True:
            if self._sender_error is not None:
                raise self._raise_sender_error()
            try:
                q.put(s, timeout=0.1)
                break
            except queue.Full:
                continue
        self._count += int(s.size)

    def _sender_loop(self) -> None:
        """Drain the queue and push chunks to the USRP (one metadata object)."""
        q = self._queue
        assert q is not None
        ec = tx_error_code_enum(self._uhd)
        md = self._uhd.types.TXMetadata()  # reused: avoids per-chunk allocation
        while True:
            try:
                item = q.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                break
            try:
                self._send_block(item, md, ec)
            except BaseException as exc:  # noqa: BLE001 - stop cleanly
                self._sender_error = classify_tx_error(exc)
                self._warn(str(self._sender_error))
                try:
                    while True:  # unblock any writer waiting on a full queue
                        q.get_nowait()
                except queue.Empty:
                    pass
                return

    def _send_block(self, s: np.ndarray, md, ec) -> None:
        n = int(s.size)
        off = 0
        while off < n:
            chunk = min(self._max_samps or n, n - off)
            block = s[off:off + chunk].reshape(1, chunk)
            # ``start_of_burst`` is sent exactly once, on the very first
            # chunk of the whole stream (the one-time startup behaviour).
            md.start_of_burst = (off == 0 and self._sent == 0)
            md.end_of_burst = False
            sent = int(self._streamer.send(block, md, 1.0))
            if tx_has_error(md, "underflow", ec):
                self._underflows += 1
            off += sent
            self._sent += sent
            if sent == 0:
                break

    def close(self) -> None:
        # Stop the sender first, then flush a single end_of_burst.
        q = self._queue
        if q is not None and self._sender is not None:
            try:
                q.put(None, timeout=1.0)
            except queue.Full:
                pass
            self._sender.join(timeout=5.0)
            self._sender = None
        self._queue = None
        if self._streamer is not None:
            try:
                md = self._uhd.types.TXMetadata()
                md.end_of_burst = True
                self._streamer.send(self._buff[:, :0], md, 1.0)
            except Exception:
                pass
        self._streamer = None
        self._buff = None
        self.usrp = None
        self._started = False

    @property
    def count(self) -> int:
        return self._count

    @property
    def underflows(self) -> int:
        return self._underflows

    def describe(self) -> str:
        lo, hi = self._gain_range
        rng = (f", диапазон {lo:g}..{hi:g} дБ"
               if hi > lo else "")
        return (f"B210 TX ch{self.channel} {self.center_freq/1e6:.3f} МГц, "
                f"{self.sample_rate/1e6:g} Мвыб/с (запрошено "
                f"{self.requested_sample_rate/1e6:g}), "
                f"полоса {self.tx_bandwidth/1e6:g} МГц, gain {self.gain:g} дБ"
                + rng)

    def get_info(self) -> dict:
        info: dict[str, Any] = {
            "args": self.args,
            "uhd_version": uhd_version(),
            "channel": self.channel,
            "sample_rate": self.sample_rate,
            "requested_sample_rate": self.requested_sample_rate,
            "tx_bandwidth": self.tx_bandwidth,
            "requested_tx_bandwidth": self.requested_tx_bandwidth,
            "center_freq": self.center_freq,
            "gain": self.gain,
            "requested_gain": self.requested_gain,
            "gain_range": list(self._gain_range),
            "gain_warning": self.gain_warning,
        }
        with self._lock:
            if self.usrp is None:
                return info
            for key, fn in (
                ("mboard", lambda: self.usrp.get_mboard_name()),
                ("antenna", lambda: self.usrp.get_tx_antenna(self.channel)),
            ):
                try:
                    info[key] = fn()
                except Exception:
                    pass
        return info
