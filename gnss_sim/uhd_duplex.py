"""Одновременный TX/RX на USRP B210 (дуплекс) для контроля передачи.

Модуль импортирует :mod:`uhd` лениво (как :mod:`gnss_sim.uhd_tx`), поэтому
работает и на машине без UHD/железа — ошибка появляется только при создании
объекта.

Физика B210 (обязательно к пониманию)
-------------------------------------
B210 = один AD9361: **раздельные LO для RX и TX**, значит приём и передача
могут идти одновременно и на разных частотах (FDD).  НО:

* у канала 0 порт ``TX/RX`` один (TDD-переключатель), поэтому «слушать себя»
  на том же канале/порту нельзя;
* чтобы контролировать свой TX, передавайте на ``tx_channel`` (порт
  ``TX/RX``), а принимайте на **другом** канале (порт ``RX2``), соединённом
  кабелем/ответвителем или антенной в камере;
* один RX LO общий для обоих RX-каналов, один TX LO — для обоих TX-каналов.

Безопасность: между TX и RX ставьте аттенюатор 50–60 дБ и DC block, иначе
вход приёмника будет перегружен (или повреждён).
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from .power import estimate_noise_floor, rms_dbfs
from .uhd_tx import (
    _UHD_HINT,
    TxError,
    clamp_gain,
    gain_range_bounds,
    rx_error_code_enum,
    rx_has_error,
    tx_error_code_enum,
    tx_has_error,
    uhd_version,
)

_MAX_CONSECUTIVE_TIMEOUTS = 20


class UhdDuplex:
    """Одновременные потоки TX и RX на одном ``MultiUSRP`` (разные каналы)."""

    def __init__(
        self,
        args: str = "type=b200",
        tx_channel: int = 0,
        rx_channel: int = 1,
        sample_rate: float = 2.6e6,
        center_freq: float = 1575.42e6,
        tx_gain: float = 0.0,
        rx_gain: float = 30.0,
        tx_antenna: str = "TX/RX",
        rx_antenna: str = "RX2",
        bandwidth: float = 2.5e6,
        clock_source: str = "internal",
        rx_center_freq: float | None = None,
        recv_timeout: float = 1.0,
        log=None,
    ) -> None:
        try:
            import uhd
        except Exception as exc:  # pragma: no cover - зависит от среды
            raise TxError(_UHD_HINT) from exc

        if int(tx_channel) == int(rx_channel):
            raise TxError(
                "TX и RX каналы должны быть разными (у канала 0 порт TX/RX "
                "один): передавайте на tx_channel, слушайте на другом канале "
                "через порт RX2")

        self._uhd = uhd
        self.args = args
        self.tx_channel = int(tx_channel)
        self.rx_channel = int(rx_channel)
        self.recv_timeout = float(recv_timeout)
        self._log = log
        self.gain_warning: str | None = None
        self._gain_range = (0.0, 0.0)

        self._lock = threading.RLock()
        self._tx_lock = threading.Lock()
        self._rx_lock = threading.Lock()
        self._stop = threading.Event()
        self._tx_streamer: Any = None
        self._rx_streamer: Any = None
        self._tx_buff: np.ndarray | None = None
        self._rx_buff: np.ndarray | None = None
        self._tx_max = 0
        self._rx_max = 0
        self._count = 0
        self._underflows = 0
        self._overflows = 0
        self._rx_errors = 0
        self._started = False

        with self._lock:
            self.usrp = uhd.usrp.MultiUSRP(args)
            if clock_source:
                try:
                    self.usrp.set_clock_source(clock_source)
                except Exception:
                    pass

            self.usrp.set_tx_rate(float(sample_rate), self.tx_channel)
            self.usrp.set_tx_freq(uhd.types.TuneRequest(float(center_freq)),
                                  self.tx_channel)
            try:
                bounds = gain_range_bounds(
                    self.usrp.get_tx_gain_range(self.tx_channel))
            except Exception:  # noqa: BLE001
                bounds = (0.0, 0.0)
            if bounds[1] > bounds[0]:
                self._gain_range = bounds
                applied_tx_gain, self.gain_warning = clamp_gain(
                    tx_gain, self._gain_range)
            else:
                applied_tx_gain, self.gain_warning = float(tx_gain), None
            if self.gain_warning:
                self._warn(self.gain_warning)
            self.usrp.set_tx_gain(applied_tx_gain, self.tx_channel)
            if bandwidth and float(bandwidth) > 0:
                try:
                    self.usrp.set_tx_bandwidth(float(bandwidth),
                                               self.tx_channel)
                except Exception:
                    pass
            if tx_antenna:
                try:
                    self.usrp.set_tx_antenna(str(tx_antenna), self.tx_channel)
                except Exception:
                    pass

            rx_freq = (float(rx_center_freq) if rx_center_freq
                       else float(center_freq))
            self.usrp.set_rx_rate(float(sample_rate), self.rx_channel)
            self.usrp.set_rx_freq(uhd.types.TuneRequest(rx_freq),
                                  self.rx_channel)
            self.usrp.set_rx_gain(float(rx_gain), self.rx_channel)
            if bandwidth and float(bandwidth) > 0:
                try:
                    self.usrp.set_rx_bandwidth(float(bandwidth),
                                               self.rx_channel)
                except Exception:
                    pass
            if rx_antenna:
                try:
                    self.usrp.set_rx_antenna(str(rx_antenna), self.rx_channel)
                except Exception:
                    pass

            self._make_streamers()

        # Requested values live in cfg; read back the ACTUAL values here.
        self.requested_sample_rate = float(sample_rate)
        self.requested_tx_bandwidth = float(bandwidth or 0.0)
        self.requested_tx_gain = float(tx_gain)
        self.sample_rate = float(self.usrp.get_tx_rate(self.tx_channel))
        self.center_freq = float(self.usrp.get_tx_freq(self.tx_channel))
        self.tx_gain = float(self.usrp.get_tx_gain(self.tx_channel))
        self.rx_gain = float(self.usrp.get_rx_gain(self.rx_channel))
        self.rx_center_freq = float(self.usrp.get_rx_freq(self.rx_channel))
        try:
            self.tx_bandwidth = float(
                self.usrp.get_tx_bandwidth(self.tx_channel))
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
        """Device TX gain range ``(min, max)`` in dB (shared by both TX ch)."""
        return self._gain_range

    def _make_streamers(self) -> None:
        uhd = self._uhd
        st_tx = uhd.usrp.StreamArgs("fc32", "sc16")
        st_tx.channels = [self.tx_channel]
        self._tx_streamer = self.usrp.get_tx_stream(st_tx)
        self._tx_max = int(self._tx_streamer.get_max_num_samps())
        self._tx_buff = np.zeros((1, self._tx_max), dtype=np.complex64)

        st_rx = uhd.usrp.StreamArgs("fc32", "sc16")
        st_rx.channels = [self.rx_channel]
        self._rx_streamer = self.usrp.get_rx_stream(st_rx)
        self._rx_max = int(self._rx_streamer.get_max_num_samps())
        self._rx_buff = np.zeros((1, self._rx_max), dtype=np.complex64)

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._started:
            return
        try:
            cmd = self._uhd.types.StreamCMD(self._uhd.types.StreamMode.start_cont)
            cmd.stream_now = True
            self._rx_streamer.issue_stream_cmd(cmd)
        except Exception:
            pass  # поток RX стартует первым recv
        self._started = True

    # ------------------------------------------------------------------
    def _send_all(self, samples: np.ndarray) -> int:
        uhd = self._uhd
        ec = tx_error_code_enum(uhd)
        s = samples
        n = s.size
        off = 0
        with self._tx_lock:
            while off < n and not self._stop.is_set():
                chunk = min(self._tx_max, n - off)
                self._tx_buff[0, :chunk] = s[off:off + chunk]
                md = uhd.types.TXMetadata()
                md.start_of_burst = (off == 0 and self._count == 0)
                md.end_of_burst = False
                try:
                    sent = self._tx_streamer.send(self._tx_buff[:, :chunk],
                                                  md, 1.0)
                except Exception as exc:  # pragma: no cover - железо
                    raise TxError(f"Ошибка передачи UHD: {exc}") from exc
                if tx_has_error(md, "underflow", ec):
                    self._underflows += 1
                off += int(sent)
                if sent == 0:
                    break
        return off

    def _recv_n(self, n: int) -> np.ndarray:
        """Принять ~n отсчётов; не падать на первом таймауте."""
        uhd = self._uhd
        ec = rx_error_code_enum(uhd)
        chunks: list[np.ndarray] = []
        got = 0
        timeouts = 0
        with self._rx_lock:
            while got < n and not self._stop.is_set():
                chunk = min(self._rx_max, n - got)
                md = uhd.types.RXMetadata()
                try:
                    num = int(self._rx_streamer.recv(self._rx_buff[:, :chunk],
                                                     md, self.recv_timeout))
                except Exception:
                    timeouts += 1
                    if timeouts >= _MAX_CONSECUTIVE_TIMEOUTS:
                        break
                    continue
                if rx_has_error(md, "overflow", ec):
                    self._overflows += 1
                if num > 0:
                    chunks.append(self._rx_buff[0, :num].copy())
                    got += num
                    timeouts = 0
                elif rx_has_error(md, "timeout", ec):
                    timeouts += 1
                    if timeouts >= _MAX_CONSECUTIVE_TIMEOUTS:
                        break
                else:
                    self._rx_errors += 1
                    timeouts += 1
                    if timeouts >= _MAX_CONSECUTIVE_TIMEOUTS:
                        break
        if not chunks:
            return np.zeros(0, dtype=np.complex64)
        return np.concatenate(chunks).astype(np.complex64, copy=False)

    # ------------------------------------------------------------------
    def send(self, tx_samples: np.ndarray) -> np.ndarray:
        """Передать ``tx_samples`` и одновременно принять столько же.

        Возвращает комплексный RX-захват длины ``len(tx_samples)`` (точное
        временное совмещение делается в :func:`gnss_sim.power.delay_profile`).
        """
        if not self._started:
            self.start()
        s = np.asarray(tx_samples, dtype=np.complex64).ravel()
        n = s.size
        if n == 0:
            return np.zeros(0, dtype=np.complex64)

        errors: list[BaseException] = []

        def _worker() -> None:
            try:
                self._send_all(s)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        t = threading.Thread(target=_worker, name="uhd-tx", daemon=True)
        t.start()
        rx = self._recv_n(n)
        t.join()

        self._count += n
        if errors:
            raise TxError(f"Ошибка дуплексной передачи: {errors[0]}")

        if rx.size >= n:
            rx = rx[:n]
        else:
            rx = np.concatenate([rx, np.zeros(n - rx.size, dtype=np.complex64)])
        return rx.astype(np.complex64, copy=False)

    # ------------------------------------------------------------------
    def measure_noise_floor(self, nsamp: int, tx_off: bool = True) -> float:
        """Оценить шумовой пол (дБ) без передачи (``tx_off=True``).

        Если ``tx_off=False``, замер всё равно только принимает, но
        вызывающий код должен сам остановить TX.
        """
        del tx_off  # приёмник пассивен по построению; TX не запускаем
        if not self._started:
            self.start()
        n = max(self._rx_max, int(nsamp))
        x = self._recv_n(n)
        if x.size == 0:
            return float("nan")
        blocks = [x[i:i + self._rx_max]
                  for i in range(0, x.size, self._rx_max)]
        return estimate_noise_floor(blocks)

    def tune_rx(self, freq: float) -> float:
        """Перестроить RX LO (общий для обоих RX-каналов)."""
        with self._lock:
            self.usrp.set_rx_freq(self._uhd.types.TuneRequest(float(freq)),
                                  self.rx_channel)
            self.rx_center_freq = float(self.usrp.get_rx_freq(self.rx_channel))
        return self.rx_center_freq

    def set_tx_gain(self, gain: float) -> float:
        """Применить усиление TX (используется регулятором мощности)."""
        if self._gain_range[1] > self._gain_range[0]:
            value, warn = clamp_gain(gain, self._gain_range)
        else:
            value, warn = float(gain), None
        if warn:
            # Не засоряем журнал на каждом шаге регулятора: сообщаем один раз.
            if self.gain_warning is None:
                self.gain_warning = warn
                self._warn(warn)
        with self._lock:
            self.usrp.set_tx_gain(float(value), self.tx_channel)
            self.tx_gain = float(self.usrp.get_tx_gain(self.tx_channel))
        return self.tx_gain

    def get_rx_power_dbfs(self, samples: np.ndarray) -> float:
        """Уровень RX-блока в dBFS (обёртка над :func:`rms_dbfs`)."""
        return rms_dbfs(samples)

    # ------------------------------------------------------------------
    def close(self) -> None:
        self._stop.set()
        if self._tx_streamer is not None:
            try:
                md = self._uhd.types.TXMetadata()
                md.end_of_burst = True
                self._tx_streamer.send(self._tx_buff[:, :0], md, 1.0)
            except Exception:
                pass
        if self._rx_streamer is not None and self._started:
            try:
                cmd = self._uhd.types.StreamCMD(
                    self._uhd.types.StreamMode.stop_cont)
                self._rx_streamer.issue_stream_cmd(cmd)
            except Exception:
                pass
        with self._lock:
            self._tx_streamer = None
            self._rx_streamer = None
            self._tx_buff = None
            self._rx_buff = None
            self.usrp = None

    # ------------------------------------------------------------------
    @property
    def count(self) -> int:
        return self._count

    @property
    def underflows(self) -> int:
        return self._underflows

    @property
    def overflows(self) -> int:
        return self._overflows

    @property
    def rx_errors(self) -> int:
        return self._rx_errors

    def describe(self) -> str:
        lo, hi = self._gain_range
        rng = (f", диапазон TX {lo:g}..{hi:g} дБ" if hi > lo else "")
        return (f"B210 дуплекс TX ch{self.tx_channel} "
                f"{self.center_freq / 1e6:.3f} МГц / RX ch{self.rx_channel} "
                f"{self.rx_center_freq / 1e6:.3f} МГц, "
                f"{self.sample_rate / 1e6:g} Мвыб/с (запрошено "
                f"{self.requested_sample_rate / 1e6:g}), "
                f"полоса TX {self.tx_bandwidth / 1e6:g} МГц, "
                f"TX {self.tx_gain:g} дБ, RX {self.rx_gain:g} дБ" + rng)

    def get_info(self) -> dict:
        info: dict[str, Any] = {
            "args": self.args,
            "uhd_version": uhd_version(),
            "tx_channel": self.tx_channel,
            "rx_channel": self.rx_channel,
            "sample_rate": self.sample_rate,
            "requested_sample_rate": self.requested_sample_rate,
            "tx_bandwidth": self.tx_bandwidth,
            "requested_tx_bandwidth": self.requested_tx_bandwidth,
            "center_freq": self.center_freq,
            "rx_center_freq": self.rx_center_freq,
            "tx_gain": self.tx_gain,
            "requested_tx_gain": self.requested_tx_gain,
            "gain_range": list(self._gain_range),
            "gain_warning": self.gain_warning,
            "rx_gain": self.rx_gain,
            "count": self._count,
            "underflows": self._underflows,
            "overflows": self._overflows,
            "rx_errors": self._rx_errors,
        }
        with self._lock:
            if self.usrp is None:
                return info
            for key, fn in (
                ("mboard", lambda: self.usrp.get_mboard_name()),
                ("tx_antenna",
                 lambda: self.usrp.get_tx_antenna(self.tx_channel)),
                ("rx_antenna",
                 lambda: self.usrp.get_rx_antenna(self.rx_channel)),
            ):
                try:
                    info[key] = fn()
                except Exception:
                    pass
        return info
