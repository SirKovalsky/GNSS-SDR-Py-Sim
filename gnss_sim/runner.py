"""Runs a simulation: parse ephemerides, generate IQ, feed a sink."""

from __future__ import annotations

import math
import os
import threading
import time
from typing import Callable

import numpy as np

from .config import SimConfig, band_preset, compute_combined_band
from .constants import R2D
from .engine import (
    SignalEngine,
    b1i_band_fits,
    suggest_b1i_band,
    visible_beidou_count,
)
from .gpstime import GpsTime
from .iqfile import FileSink, IqFileSource, NullSink, Sink
from .motion import interpolation_fn, load_user_motion
from .orbit import llh2xyz
from .power import (
    AUTO_AMP_MAX,
    DEFAULT_HEADROOM_TARGET,
    HeadroomController,
    PowerRegulator,
    delay_profile,
    level_stats,
    rms_dbfs,
)
from .rinex import (
    check_start_coverage,
    parse_nav_file,
    unique_sv_counts,
)
from . import sysinfo

StatusFn = Callable[[str], None]
ProgressFn = Callable[[float, float, float, float], None]
#: Phase progress callback: ``(kind, frac, loops, sim_s, cyclic)``.
#: ``kind`` is ``"pregen"`` (RAM synthesis, own 0…1 value) or ``"tx"``
#: (actual transmission, 0…1 within the current segment/loop).  ``loops`` is
#: the completed loop count for cyclic transmission and ``cyclic`` marks it.
PhaseFn = Callable[[str, float, int, float, bool], None]
SpectrumFn = Callable[[str, np.ndarray, float], None]
#: TX/RX level callback: ``(label, peak_dbfs, rms_dbfs, clips)``.
LevelFn = Callable[[str, float, float, int], None]
AskFn = Callable[[str, float], bool]

_MAX_SPEC_SAMPLES = 65536
_SPEC_INTERVAL = 0.5
#: Minimum wall-time between two GUI level updates (never per sample).
_LEVEL_INTERVAL = 0.5
#: Conservative synthesis-rate estimate (samples/s) used to warn about long TX
#: pre-generation; real CPU/GPU synthesis is usually faster.
_EST_SYNTH_RATE_SPS = 2.0e6
#: Warn when the estimated TX pre-generation exceeds this many seconds.
_PREGEN_WARN_SECONDS = 30.0
#: Warn about USB3 drops (underflow / LIBUSB_TRANSFER_NO_DEVICE) above this
#: sample rate when transmitting to a B210.  30 Msps was the observed failure
#: point; the computed combined band (~25 Msps) is expected to be stable.
_HIGH_FS_TX_WARN_HZ = 30.0e6
#: Bytes per sample of the pre-generated TX RAM segment (complex64).
_TX_RAM_BYTES = 8
#: Cap on the TX write block (seconds): smaller writes feed the B210 more
#: smoothly on Windows/USB3 and measurably reduce TX underflows.
_TX_BLOCK_SECONDS = 0.1
#: Default «контроль передачи» threshold (dB above the RX noise floor).
_TX_CHECK_MARGIN_DB = 6.0
#: Progress journal milestones (%).  Logging every block/write spammed the
#: journal (user issue 1); only these milestones (or a long wall-time gap) are
#: written.
_PROGRESS_MILESTONES = (25, 50, 75, 100)
#: Maximum wall-time between two progress journal lines.
_PROGRESS_LOG_INTERVAL_S = 30.0
#: Time constant of the endless-TX streaming progress asymptote (``0.5 -> 1``).
_TX_STREAM_TAU_S = 20.0


class _DuplexSink(Sink):
    """Приёмопередающий sink: TX на B210 + периодический контроль RX.

    Строится только при ``cfg.use_usrp and cfg.monitor``.  Использует
    :class:`gnss_sim.uhd_duplex.UhdDuplex` (разные каналы для TX и RX),
    раз в ~0.5 с измеряет уровень RX, при ``tx_power_auto`` подстраивает
    усиление TX :class:`gnss_sim.power.PowerRegulator`, при
    ``echo_analysis`` строит :func:`gnss_sim.power.delay_profile`.
    """

    def __init__(self, cfg: SimConfig, log: StatusFn | None = None,
                 spectrum: Callable[[str, np.ndarray], None] | None = None
                 ) -> None:
        from .uhd_duplex import UhdDuplex

        self._cfg = cfg
        self._log = log
        # ``spectrum`` is the runner's two-argument ``_emit_spectrum(label,
        # samples)`` helper (it adds ``cfg.fs`` and throttling itself).
        self._spectrum = spectrum
        self._duplex = UhdDuplex(
            args=cfg.uhd_args, tx_channel=cfg.tx_channel,
            rx_channel=cfg.rx_channel, sample_rate=cfg.fs,
            center_freq=cfg.center_freq, tx_gain=cfg.tx_gain,
            rx_gain=cfg.rx_gain, tx_antenna=cfg.tx_antenna,
            rx_antenna=cfg.rx_antenna, bandwidth=cfg.tx_bandwidth,
            clock_source=cfg.clock_source,
            rx_center_freq=(cfg.rx_center_freq or None),
            log=log)
        self._count = 0
        self._interval = 0.5
        self._last_monitor = 0.0
        self._noise_dbfs: float | None = None
        self._latest_tx: np.ndarray | None = None
        self._latest_rx: np.ndarray | None = None
        self._regulator = (PowerRegulator(
            target_dbfs=cfg.tx_power_target_dbfs,
            initial_gain=self._duplex.tx_gain)
            if cfg.tx_power_auto else None)
        # «Контроль передачи»: проверить RX один раз после ~0.3 с TX.
        self._tx_check = bool(getattr(cfg, "tx_check", False))
        self._tx_check_done = False
        self._tx_check_after = max(1, int(cfg.fs * 0.3)) if cfg.fs else 1
        self._tx_check_margin = float(
            getattr(cfg, "tx_check_margin_db", _TX_CHECK_MARGIN_DB))

    # ------------------------------------------------------------------
    def _logf(self, msg: str) -> None:
        if self._log is not None:
            self._log(msg)

    def start(self) -> None:
        self._duplex.start()
        if self._cfg.monitor or self._tx_check:
            try:
                sniff = max(1, int(self._cfg.fs * 0.05))
                self._noise_dbfs = self._duplex.measure_noise_floor(sniff)
                self._logf(f"Шумовой пол RX (TX выкл): "
                           f"{self._noise_dbfs:.1f} dBFS")
            except Exception as exc:  # noqa: BLE001
                self._logf(f"Не удалось измерить шумовой пол: {exc}")

    def write(self, samples: np.ndarray) -> None:
        tx = np.asarray(samples)
        rx = self._duplex.send(tx)
        self._count += tx.size
        needs_rx = self._cfg.monitor or self._tx_check
        if needs_rx:
            self._latest_tx = tx
            self._latest_rx = rx
            if self._cfg.monitor and self._spectrum is not None:
                self._spectrum("RX", rx)
            now = time.time()
            if self._cfg.monitor and now - self._last_monitor >= self._interval:
                self._last_monitor = now
                self._monitor()
        if (self._tx_check and not self._tx_check_done
                and self._count >= self._tx_check_after):
            self._tx_check_done = True
            self._run_tx_check()

    def _run_tx_check(self) -> None:
        """Один раз проверить, что передача видна на RX выше шума.

        Если RX-захвата нет (нули/пусто) или шумовой пол неизвестен, проверка
        пропускается с сообщением — это не ошибка передачи.
        """
        if not self._tx_check:
            return
        rx = self._latest_rx
        if rx is None or rx.size == 0 or not np.any(rx):
            self._logf("Контроль передачи: RX недоступен — проверка пропущена")
            return
        noise = self._noise_dbfs
        if noise is None or not np.isfinite(noise):
            self._logf("Контроль передачи: шумовой пол RX неизвестен — "
                       "проверка пропущена")
            return
        p_dbfs = rms_dbfs(rx)
        lift = p_dbfs - noise
        if lift < self._tx_check_margin:
            self._logf(
                "ВНИМАНИЕ: передача не обнаружена (уровень RX не выше шума): "
                f"RX {p_dbfs:.1f} dBFS, шум {noise:.1f} dBFS, "
                f"превышение {lift:.1f} дБ < порога "
                f"{self._tx_check_margin:.0f} дБ")
        else:
            self._logf(
                f"Контроль передачи: сигнал обнаружен — RX {p_dbfs:.1f} dBFS "
                f"выше шума на {lift:.1f} дБ (порог "
                f"{self._tx_check_margin:.0f} дБ)")

    # ------------------------------------------------------------------
    def _monitor(self) -> None:
        rx = self._latest_rx
        if rx is None or rx.size == 0:
            return
        p_dbfs = rms_dbfs(rx)
        noise = self._noise_dbfs
        if noise is not None and np.isfinite(noise):
            snr: float | None = p_dbfs - noise
            noise_txt = f"{noise:.1f}"
        else:
            snr = None
            noise_txt = "n/a"
        snr_txt = f"{snr:.1f}" if snr is not None else "n/a"
        self._logf(f"RX: {p_dbfs:.1f} dBFS, шум {noise_txt} dBFS, "
                   f"SNR {snr_txt} дБ")

        if self._regulator is not None:
            old = self._duplex.tx_gain
            new_gain = self._regulator.update(p_dbfs, noise)
            if abs(new_gain - old) >= 0.1:
                applied = self._duplex.set_tx_gain(new_gain)
                self._logf(f"TX усиление: {old:.1f} -> {applied:.1f} дБ")

        if self._cfg.echo_analysis and self._latest_tx is not None:
            self._log_echo()

    def _log_echo(self) -> None:
        tx = self._latest_tx
        rx = self._latest_rx
        if tx is None or rx is None or tx.size == 0 or rx.size == 0:
            return
        n = min(tx.size, rx.size, max(1024, int(self._cfg.fs * 0.25)))
        try:
            delays, levels = delay_profile(tx[:n], rx[:n], self._cfg.fs)
        except Exception as exc:  # noqa: BLE001
            self._logf(f"Ошибка анализа эха: {exc}")
            return
        if delays.size == 0:
            return
        idx = np.argsort(levels)[::-1][:3]
        parts = [f"{delays[i] * 1e9:.0f} нс / {levels[i]:.1f} дБ"
                 for i in idx]
        self._logf("Эхо, топ-3: " + ", ".join(parts))

    def close(self) -> None:
        self._duplex.close()

    @property
    def count(self) -> int:
        return self._count

    @property
    def underflows(self) -> int:
        return self._duplex.underflows

    @property
    def overflows(self) -> int:
        return self._duplex.overflows

    def describe(self) -> str:
        return self._duplex.describe()

    def get_info(self) -> dict:
        return self._duplex.get_info()


class SimulationRunner:
    def __init__(
        self,
        cfg: SimConfig,
        log: StatusFn | None = None,
        progress: ProgressFn | None = None,
        channels: Callable[[list], None] | None = None,
        finished: Callable[[str | None], None] | None = None,
        spectrum: SpectrumFn | None = None,
        ask: AskFn | None = None,
        phase: PhaseFn | None = None,
        level: LevelFn | None = None,
    ) -> None:
        self.cfg = cfg
        self._log = log
        self._progress = progress
        self._phase = phase
        self._channels = channels
        self._finished = finished
        self._spectrum = spectrum
        self._ask = ask
        self._level = level
        self._spec_last: dict[str, float] = {}
        self._level_last: dict[str, float] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.engine: SignalEngine | None = None
        self.sink: Sink | None = None
        self.error: str | None = None
        self._start: GpsTime | None = None
        self._loop = False
        self._loop_kind = "RAM"
        self._segment_seconds: float | None = None
        self._ram_budget_bytes = 0
        self._max_seg_seconds = 0.0
        self._source: IqFileSource | None = None
        self._headroom: HeadroomController | None = None

    # ------------------------------------------------------------------
    def _logf(self, msg: str) -> None:
        if self._log is not None:
            self._log(msg)

    def _milestone_logger(self, label: str):
        """Return a ``(pct, detail) -> None`` journal logger (issue 1a).

        The journal is written only when ``pct`` crosses the next milestone
        (:data:`_PROGRESS_MILESTONES`) or when more than
        :data:`_PROGRESS_LOG_INTERVAL_S` elapsed since the previous line — never
        once per block.
        """
        state = {"idx": 0, "last": time.time()}

        def emit(pct: int, detail: str) -> None:
            now = time.time()
            due = (state["idx"] < len(_PROGRESS_MILESTONES)
                   and pct >= _PROGRESS_MILESTONES[state["idx"]])
            if due or now - state["last"] >= _PROGRESS_LOG_INTERVAL_S:
                while (state["idx"] < len(_PROGRESS_MILESTONES)
                       and _PROGRESS_MILESTONES[state["idx"]] <= pct):
                    state["idx"] += 1
                state["last"] = now
                self._logf(f"{label}: {pct}% ({detail})")

        return emit

    def _emit_level(self, label: str, samples: np.ndarray) -> None:
        """Отдать уровень блока (peak/RMS/клип) в GUI, не чаще 0.5 с."""
        fn = self._level
        if fn is None:
            return
        now = time.time()
        if now - self._level_last.get(label, 0.0) < _LEVEL_INTERVAL:
            return
        self._level_last[label] = now
        try:
            st = level_stats(samples)
            fn(label, st.peak_dbfs, st.rms_dbfs, st.clips)
        except Exception:  # noqa: BLE001 - индикатор не должен ломать TX
            pass

    def _emit_spectrum(self, label: str, samples: np.ndarray) -> None:
        """Отдать ~0.5-секундный блок (не более 65536 отсчётов) в график.

        Вызывается из потока генерации; ошибки графика не должны прерывать
        синтез, поэтому любые исключения глотаются.  Уровень (peak/RMS/клип)
        обновляется тем же вызовом, даже когда график отключён.
        """
        if self._spectrum is None and self._level is None:
            return
        x = np.asarray(samples)
        if x.size == 0:
            return
        self._emit_level(label, x)
        if self._spectrum is None:
            return
        now = time.time()
        if now - self._spec_last.get(label, 0.0) < _SPEC_INTERVAL:
            return
        self._spec_last[label] = now
        if x.size > _MAX_SPEC_SAMPLES:
            x = x[:_MAX_SPEC_SAMPLES]
        try:
            self._spectrum(label, x, self.cfg.fs)
        except Exception:  # noqa: BLE001
            pass

    def _full_scale(self) -> float:
        """Clip limit of the delivery format in baseband units.

        Live USRP TX streams ``fc32`` (full scale 1.0).  Integer IQ files are
        written as ``round(x * output_scale)`` clipped to the integer range, so
        the baseband clip limit is ``int_max / output_scale`` (``cs16`` with the
        default ``output_scale=10000`` gives ≈3.28, not 1.0).  Using 1.0 here
        was the regression: it made the anti-clip path attenuate a normal
        scene by ~10 dB and the F9P stopped acquiring.
        """
        cfg = self.cfg
        fmt = "cf32" if bool(getattr(cfg, "use_usrp", False)) else str(
            getattr(cfg, "output_format", "cs16") or "cs16")
        scale = float(getattr(cfg, "output_scale", 10000.0) or 1.0)
        if fmt == "cs16":
            return 32767.0 / scale
        if fmt == "cs8":
            return (127.0 * 256.0) / scale
        if fmt == "cs4":
            return (7.0 * 4096.0) / scale
        return 1.0  # cf32 / float

    def _headroom_ctl(self) -> HeadroomController:
        """Lazily build the anti-clip controller for this run."""
        if self._headroom is None:
            cfg = self.cfg
            self._headroom = HeadroomController(
                target=float(getattr(cfg, "headroom_target",
                                     DEFAULT_HEADROOM_TARGET)),
                enabled=bool(getattr(cfg, "headroom", True)),
                log=self._logf,
                full_scale=self._full_scale())
        return self._headroom

    def _apply_auto_amp(self, engine: SignalEngine) -> None:
        """Use the historic nominal amplitude when ``cfg.amp_scale`` is ``None``.

        The earlier "automatic" formula reduced the per-channel amplitude by the
        summed weights (≈-10 dB on a full L1 scene).  That attenuation was
        redundant — the anti-clip controller (when enabled) already measures and
        trims the real composite peak — and it took the ZED-F9P below its
        acquisition threshold on a marginal link.  The nominal 0.15 restores the
        level that produced fixes; an explicit ``cfg.amp_scale`` still wins.
        """
        cfg = self.cfg
        if cfg.amp_scale is not None:
            return
        try:
            channels = list(engine.channels)
            provisional = float(engine.amp_scale) or AUTO_AMP_MAX
            weights = [c.amp / provisional for c in channels]
        except Exception:  # noqa: BLE001 - fake engines in tests
            return
        # Auto = historic nominal; the headroom controller (opt-in) enforces
        # the composite peak, so attenuating here would double-count.
        engine.set_amp_scale(AUTO_AMP_MAX)
        self._logf(
            f"Амплитуда: авто — номинал {AUTO_AMP_MAX:.4f} "
            f"(по {len(channels)} каналам, сумма весов {sum(weights):.2f}, "
            f"полная шкала {self._full_scale():.2f}); явно задать: "
            f"--amp / поле «Амплитуда»")

    def _warn_high_fs_tx(self) -> None:
        """Warn about USB3 drops when transmitting a very high-``fs`` stream.

        The observed B210 failure at 30 Msps is an underflow that ends in
        ``LIBUSB_TRANSFER_NO_DEVICE``; the computed combined ``all`` band
        (~25 Msps) stays below that and is verified stable over the air.
        """
        cfg = self.cfg
        if not cfg.use_usrp:
            return
        fs = float(cfg.fs or 0.0)
        if fs < _HIGH_FS_TX_WARN_HZ:
            return
        self._logf(
            f"ВНИМАНИЕ: высокая fs {fs / 1e6:.1f} Мвыб/с для передачи на B210 "
            "по USB3 — канал может не успевать: возможны underflow и обрыв "
            "потока (LIBUSB_TRANSFER_NO_DEVICE). Это наблюдалось при "
            "30 Мвыб/с. Для объединённого потока используйте --band all "
            "(~25 Мвыб/с) или уменьшите fs.")

    def _report_underflows(self) -> None:
        """Report the sink's TX underflow count (0 = silent)."""
        sink = self.sink
        if sink is None:
            return
        try:
            uf = int(getattr(sink, "underflows", 0) or 0)
        except Exception:  # noqa: BLE001
            return
        if uf > 0:
            self._logf(
                f"ВНИМАНИЕ: {uf} underflow при передаче — в потоке возможны "
                "короткие разрывы. Уменьшите fs (-s), block-ms (~100 мс) или "
                "проверьте USB3/питание B210. Обрыва LIBUSB_TRANSFER_NO_DEVICE "
                "не было.")
        elif getattr(self.cfg, "use_usrp", False):
            # Always state the count so a clean TX run is unambiguous.
            self._logf("TX underflow за сеанс: 0")

    def _make_xyz_fn(self):
        cfg = self.cfg
        if cfg.motion_file:
            times, xyz = load_user_motion(cfg.motion_file)
            fn = interpolation_fn(times, xyz)
            self._logf(f"Динамика: {cfg.motion_file} ({len(times)} точек, "
                       f"{times[-1]:.1f} с)")
            t0 = self._start.week * 604800.0 + self._start.sec
            return lambda g: fn(g.week * 604800.0 + g.sec - t0)

        static = llh2xyz(cfg.lat / R2D, cfg.lon / R2D, cfg.height)
        self._logf(f"Статичная позиция: {cfg.lat:.6f}, {cfg.lon:.6f}, "
                   f"{cfg.height:.1f} м")
        return lambda g: static

    def _make_sink(self, ignore_file: bool = False) -> Sink:
        cfg = self.cfg
        if getattr(cfg, "tx_check", False) and not cfg.use_usrp:
            self._logf("Контроль передачи: TX на B210 выключен — "
                       "проверка пропущена")
        if cfg.use_usrp:
            self._warn_high_fs_tx()
            needs_rx = bool(cfg.monitor or getattr(cfg, "tx_check", False))
            if needs_rx and int(cfg.tx_channel) == int(cfg.rx_channel):
                raise ValueError(
                    "Для дуплексного контроля TX и RX каналы должны быть "
                    f"разными (сейчас TX={cfg.tx_channel}, RX={cfg.rx_channel}); "
                    "выберите RX-канал, отличный от TX (например, TX=0, RX=1)")
            if needs_rx:
                sink = _DuplexSink(cfg, self._logf, self._emit_spectrum)
                self._logf("UHD DUPLEX: " + sink.describe())
                return sink
            from .uhd_tx import UhdTxSink
            sink = UhdTxSink(
                args=cfg.uhd_args, channel=cfg.tx_channel, sample_rate=cfg.fs,
                center_freq=cfg.center_freq, gain=cfg.tx_gain,
                antenna=cfg.tx_antenna, bandwidth=cfg.tx_bandwidth,
                clock_source=cfg.clock_source, log=self._logf)
            self._logf("UHD TX: " + sink.describe())
            return sink
        if cfg.output and not ignore_file:
            bps = {"cs16": 4, "cf32": 8, "cs8": 2, "cs4": 2}.get(
                cfg.output_format, 4)
            if cfg.duration and cfg.duration > 0 and not self._loop:
                gb = cfg.duration * cfg.fs * bps / 1e9
                self._logf(f"Ожидаемый размер файла: {gb:.1f} ГБ "
                           f"({cfg.duration:.0f} с x {cfg.fs / 1e6:.2f} Мвыб/с x "
                           f"{bps} Б/отсчёт)")
            sink = FileSink(cfg.output, fmt=cfg.output_format, fs=cfg.fs,
                            center_freq=cfg.center_freq, scale=cfg.output_scale,
                            metadata=cfg.to_dict())
            self._logf("Файл IQ: " + sink.describe())
            return sink
        return NullSink()

    # ------------------------------------------------------------------
    def _plan(self) -> None:
        """Analyse RAM/disk and decide the loop segment (RAM-disk friendly)."""
        cfg = self.cfg
        bps = {"cs16": 4, "cf32": 8, "cs8": 2, "cs4": 2}.get(cfg.output_format, 4)
        bytes_per_s = cfg.fs * bps
        avail = sysinfo.available_ram()
        total = sysinfo.total_ram()
        budget = (int(cfg.memory_budget_gb * 1e9) if cfg.memory_budget_gb > 0
                  else sysinfo.default_budget(avail))
        max_seg = budget / bytes_per_s if bytes_per_s else 0.0
        # Budget kept for the pre-generation of a TX segment (engine blocks
        # are complex128 -> 16 bytes/sample in RAM).
        self._ram_budget_bytes = int(budget)
        self._max_seg_seconds = max_seg
        self._logf(f"RAM: доступно {sysinfo.human(avail)} из "
                   f"{sysinfo.human(total)}; бюджет {sysinfo.human(int(budget))} "
                   f"(до {max_seg:.0f} с сигнала @ {bytes_per_s / 1e6:.1f} МБ/с)")
        if cfg.use_usrp and budget > 0 and cfg.fs:
            tx_cap = budget / _TX_RAM_BYTES / cfg.fs
            self._logf(f"TX-сегмент в RAM (complex64, 8 Б/отсчёт): до "
                       f"{tx_cap:.0f} с")

        req = cfg.duration if cfg.duration and cfg.duration > 0 else 0.0
        self._loop = bool(cfg.loop)
        if req <= 0:
            self._loop = True  # indefinite: keep looping until stopped
        if self._loop:
            seg, capped = sysinfo.auto_segment_seconds(
                req, max_seg, cfg.loop_seconds)
            if capped:
                if req > 0:
                    self._logf(
                        f"ВНИМАНИЕ: запрошенные {req:.1f} с не помещаются в "
                        f"бюджет RAM (до {max_seg:.1f} с) — сегмент урезан "
                        f"до {seg:.1f} с")
                else:
                    self._logf(
                        f"Сегмент урезан до {seg:.1f} с (бюджет RAM)")
            self._segment_seconds = seg
            self._loop_kind = "RAM"
            self._logf(f"Зацикливание ВКЛ: сегмент {seg:.1f} с "
                       f"(~{sysinfo.human(int(seg * bytes_per_s))} в RAM); "
                       f"источник зацикливания: RAM (буфер в памяти)")
            if cfg.use_usrp:
                # TX + loop is endless: ``duration`` below is deliberately not
                # used as a stop condition for the generated RAM segment.
                self._logf("Зацикливание: непрерывная передача до Стоп")
                self._logf(
                    "ВНИМАНИЕ: навигационные данные сегмента (TOW/эфемериды) "
                    "повторяются каждый сегмент — приёмник может не удержать "
                    "фикс между циклами. Для тестов приёмника используйте "
                    "--no-loop или длинный сегмент.")
        else:
            self._segment_seconds = None
            self._logf("Зацикливание ВЫКЛ: полная запись")

        if cfg.output:
            free = sysinfo.disk_free(os.path.dirname(cfg.output) or ".")
            self._logf(f"Диск для '{cfg.output}': свободно {sysinfo.human(free)}")
            out_s = self._segment_seconds if self._segment_seconds else req
            if out_s and out_s > 0:
                est = out_s * bytes_per_s
                self._logf(f"Размер файла: ~{sysinfo.human(int(est))} "
                           f"({out_s:.0f} с)")
                if free and est > free * 0.95:
                    self._logf("ВНИМАНИЕ: свободного места на диске может "
                               "не хватить")

    def _apply_band(self) -> None:
        """Resolve ``cfg.band`` into centre/``fs`` and the enabled systems.

        ``l1`` (default) keeps the historic narrow L1 band and, when BeiDou is
        enabled, drops B1I with a clear message instead of silently widening
        to 30 Msps (which destabilises B210 TX over USB3).  ``b1i`` is a
        BeiDou-only narrowband session.  ``all`` (alias ``wide``) is an
        explicit choice for one combined stream carrying every constellation
        simultaneously; its centre/``fs`` are computed from the *enabled*
        systems (a B210 has one shared TX LO, so all bands must be one stream).
        """
        cfg = self.cfg
        key = str(getattr(cfg, "band", "l1") or "l1").strip().lower()
        preset = band_preset(key)
        if preset is None:
            self._logf(f"Неизвестный диапазон '{key}' — используется l1")
            preset = band_preset("l1")
            key = "l1"

        if key == "l1" and not (cfg.enable_beidou
                                and getattr(cfg, "combine", False)):
            if cfg.enable_beidou:
                cfg.enable_beidou = False
                self._logf(
                    "Сессия l1: BeiDou B1I (1561.098 МГц) не помещается в "
                    "узкую полосу L1 и отключён (fs остаётся 2.6 Мвыб/с). "
                    "Чтобы объединить L1+B1I в одну широкую полосу "
                    "(~25 Мвыб/с), включите «Объединять L1+B1I» в GUI или "
                    "флаг --combine (либо --band all). Для только B1I "
                    "выберите сессию «b1i».")
            return
        if key == "l1":
            # Explicit opt-in (cfg.combine) with BeiDou enabled: fall through to
            # the combined stream below.
            key = "all"

        if key == "b1i":
            changed: list[str] = []
            if abs(cfg.fs - preset.fs) > 1.0:
                changed.append(f"fs {cfg.fs / 1e6:.3f} -> "
                               f"{preset.fs / 1e6:.3f} Мвыб/с")
                cfg.fs = preset.fs
            if abs(cfg.center_freq - preset.center_freq) > 1.0:
                changed.append(f"центр {cfg.center_freq / 1e6:.3f} -> "
                               f"{preset.center_freq / 1e6:.3f} МГц")
                cfg.center_freq = preset.center_freq
            cfg.enable_beidou = True
            cfg.enable_ca = cfg.enable_l1c = False
            cfg.enable_galileo = cfg.enable_qzss = cfg.enable_sbas = False
            self._logf("Сессия b1i: только BeiDou B1I, "
                       f"центр {cfg.center_freq / 1e6:.3f} МГц, "
                       f"fs {cfg.fs / 1e6:.3f} Мвыб/с"
                       + ((" (" + "; ".join(changed) + ")") if changed else ""))
            return

        # key in ("all", "wide") or an explicit l1+combine opt-in: one computed
        # combined stream, preserving the BOC(6,1) side lobes (~25 Msps).
        cfg.enable_beidou = True
        plan = compute_combined_band(
            enable_ca=cfg.enable_ca, enable_l1c=cfg.enable_l1c,
            enable_galileo=cfg.enable_galileo, enable_qzss=cfg.enable_qzss,
            enable_sbas=cfg.enable_sbas, enable_beidou=True,
            preserve_boc=True)
        changed = []
        if getattr(cfg, "fs_override", False):
            changed.append(f"fs {cfg.fs / 1e6:.3f} Мвыб/с (задано явно)")
        elif abs(cfg.fs - plan.fs) > 1.0:
            changed.append(f"fs {cfg.fs / 1e6:.3f} -> "
                           f"{plan.fs / 1e6:.3f} Мвыб/с")
            cfg.fs = plan.fs
        if getattr(cfg, "center_override", False):
            changed.append(f"центр {cfg.center_freq / 1e6:.3f} МГц (задано явно)")
        elif abs(cfg.center_freq - plan.center_freq) > 1.0:
            changed.append(f"центр {cfg.center_freq / 1e6:.3f} -> "
                           f"{plan.center_freq / 1e6:.3f} МГц")
            cfg.center_freq = plan.center_freq

        self._logf("Сессия all: единый поток GPS L1 C/A + L1C + Galileo E1 + "
                   "QZSS L1 + SBAS + BeiDou B1I")
        self._logf("Полоса all (расчёт): " + plan.describe())
        if changed:
            self._logf("  итог: " + "; ".join(changed))
        has_l1 = bool(cfg.enable_ca or cfg.enable_l1c or cfg.enable_galileo
                      or cfg.enable_qzss or cfg.enable_sbas)
        if has_l1 and cfg.enable_beidou:
            self._logf(
                "ВНИМАНИЕ: объединённый поток L1+B1I широкий "
                f"(~{plan.fs / 1e6:g} Мвыб/с, сохранены боковые лепестки "
                "BOC(6,1)): предгенерация TX-сегмента в RAM может занять "
                "минуты и несколько ГБ. Уменьшите длительность или "
                "используйте узкую сессию l1/b1i.")
        if cfg.use_usrp:
            self._logf(
                f"TX: единый поток {cfg.fs / 1e6:.2f} Мвыб/с, центр "
                f"{cfg.center_freq / 1e6:.3f} МГц на общем TX LO B210")

    def _apply_b1i_band(self, by_sv=None, iono=None, xyz_fn=None) -> None:
        """Auto-pick a centre/fs that keeps BeiDou B1I with L1/E1.

        Only acts when BeiDou is enabled, at least one B1I channel would
        actually be allocated (ephemerides present and visible) and the current
        ``fs``/``center_freq`` cannot fit B1I.  With ``cfg.auto_b1i`` on the
        suggestion is applied before the engine is built; otherwise manual
        values are kept and a warning is logged.  When B1I is disabled (or
        already fits) nothing changes, so such runs stay bit-identical.

        ``by_sv``/``iono``/``xyz_fn`` describe the ephemerides about to be
        used; when omitted (legacy direct calls) the presence guard is skipped.
        """
        cfg = self.cfg
        if not cfg.enable_beidou or b1i_band_fits(cfg.fs, cfg.center_freq):
            return
        # In the explicit `all`/`wide` session an explicit -s/-f wins: do not
        # silently widen the band the user asked for (B1I will be dropped and
        # reported by the engine instead).
        key = str(getattr(cfg, "band", "") or "").strip().lower()
        if key in ("all", "wide") and (getattr(cfg, "fs_override", False)
                                       or getattr(cfg, "center_override", False)):
            return
        if by_sv is not None:
            has_beidou = any(str(key).startswith("C") for key in by_sv)
            if not has_beidou:
                self._logf(
                    "BeiDou B1I не найден в эфемеридах — полоса не изменена "
                    "(центр/fs оставлены; для B1I нужен multi-GNSS RINEX)")
                return
            visible = 0
            if iono is not None and xyz_fn is not None and self._start is not None:
                visible = visible_beidou_count(
                    by_sv, iono, xyz_fn, self._start, cfg.el_mask / R2D)
            if visible <= 0:
                self._logf(
                    "BeiDou B1I не найден (нет видимых спутников выше маски) "
                    "— полоса не изменена")
                return
        systems = set()
        if cfg.enable_ca or cfg.enable_l1c:
            systems.add("G")
        if cfg.enable_galileo:
            systems.add("E")
        if cfg.enable_qzss:
            systems.add("J")
        systems.add("C")
        plan = suggest_b1i_band(cfg.fs, cfg.center_freq, systems)
        if plan is None:
            self._logf("ВНИМАНИЕ: BeiDou B1I не помещается в выбранную полосу, "
                       "а подходящие центр/fs не найдены — каналы B1I будут "
                       "пропущены")
            return
        if not getattr(cfg, "auto_b1i", True):
            self._logf(
                f"ВНИМАНИЕ: BeiDou B1I (1561.098 МГц) не помещается при "
                f"центре {cfg.center_freq / 1e6:.3f} МГц и fs "
                f"{cfg.fs / 1e6:.3f} Мвыб/с; авто-подбор отключён — каналы "
                f"B1I будут пропущены (нужен центр "
                f"{plan['center_freq'] / 1e6:.1f} МГц и fs "
                f"{plan['fs'] / 1e6:.1f} Мвыб/с)")
            return
        old_fs, old_cf = cfg.fs, cfg.center_freq
        cfg.fs = float(plan["fs"])
        cfg.center_freq = float(plan["center_freq"])
        self._logf(
            f"Авто-подбор под B1I: центр {old_cf / 1e6:.3f} -> "
            f"{cfg.center_freq / 1e6:.3f} МГц, fs {old_fs / 1e6:.3f} -> "
            f"{cfg.fs / 1e6:.3f} Мвыб/с (B1I + L1/E1 сохранены)")

    def _maybe_move_to_merged_start(self) -> None:
        """Move the start to the latest merged-RINEX day when required.

        Only acts when ``nav_mode`` asks for multi-system ephemerides and no
        explicit ``nav_file`` is configured: if the chosen date has no merged
        file, the start moves to the latest cached merged day (time-of-day
        preserved, clamped into the file's toe coverage).  No network here.
        An explicit ``cfg.nav_file`` is always respected.
        """
        cfg = self.cfg
        if getattr(cfg, "nav_mode", "auto") != "merged":
            return
        if str(getattr(cfg, "nav_file", "") or "").strip():
            return
        if self._start is None:
            return
        from .gpstime import gps2date
        from .rinexfetch import latest_merged_start
        moved = latest_merged_start(self._start)
        if moved is None:
            return
        oy, om, od = gps2date(self._start)[:3]
        ny, nm, nd, nh, nmi, nss = gps2date(moved)
        self._logf(
            "Мультисистемный merged RINEX (G/E/J/C) публикуется только за "
            "прошедшие сутки; время старта переведено с "
            f"{oy:04d}/{om:02d}/{od:02d} на {ny:04d}/{nm:02d}/{nd:02d} "
            f"{nh:02d}:{nmi:02d}:{int(nss):02d}")
        self._start = moved
        cfg.start_text = (f"{ny:04d}/{nm:02d}/{nd:02d},{nh:02d}:"
                          f"{nmi:02d}:{int(nss):02d}")

    def prepare(self) -> None:
        cfg = self.cfg
        self._start = cfg.resolved_start()

        if cfg.iq_input:
            self._prepare_source()
            return

        self._maybe_move_to_merged_start()

        from .rinexfetch import resolve_nav_file
        nav_file = resolve_nav_file(self._start, cfg, ask=self._ask,
                                    log=self._logf)
        cfg.nav_file = nav_file
        if isinstance(nav_file, (list, tuple)):
            missing = [p for p in nav_file if not os.path.exists(p)]
            if missing:
                raise ValueError(f"Файлы эфемерид не найдены: {missing}")
            self._logf("Чтение эфемерид (" + str(len(nav_file)) + " шт.): "
                       + ", ".join(os.path.basename(p) for p in nav_file))
        else:
            if not os.path.exists(nav_file):
                raise ValueError(f"Файл эфемерид не найден: {nav_file}")
            self._logf(f"Чтение эфемерид: {nav_file}")
        by_sv, iono = parse_nav_file(nav_file)
        if not by_sv:
            raise ValueError("В файле нет пригодных эфемерид (G/E/J/C)")
        counts = unique_sv_counts(by_sv)
        summary = ", ".join(f"{s}:{counts[s]}" for s in "GEJC" if s in counts)
        n_sv = len(by_sv)
        self._logf(f"Эфемериды: {n_sv} спутников ({summary})")
        if not (set(counts) & {"E", "J", "C"}):
            self._logf(
                "ВНИМАНИЕ: файл только GPS (для Galileo/BeiDou нужен merged "
                "BRDC00IGS_R_…)")
            if getattr(cfg, "nav_mode", "auto") == "merged":
                self._logf(
                    "ВНИМАНИЕ: выбран режим «Мультисистемный (G/E/J/C)», но "
                    "merged RINEX за эту дату недоступен (публикуется только "
                    "за прошедшие сутки) — доступен лишь GPS; выберите "
                    "прошлую дату или посистемные файлы.")
        cover = check_start_coverage(self._start, by_sv)
        if cover is not None:
            self._logf("ВНИМАНИЕ: " + cover)
        self._logf(f"Старт: GPS week {self._start.week}, "
                   f"{self._start.sec:.0f} с; "
                   f"{'ионосфера вкл' if cfg.iono_enable else 'ионосфера выкл'}")

        xyz_fn = self._make_xyz_fn()

        self._apply_band()
        self._apply_b1i_band(by_sv, iono, xyz_fn)

        provisional_amp = (0.15 if cfg.amp_scale is None
                           else float(cfg.amp_scale))
        self.engine = SignalEngine(
            by_sv, iono, xyz_fn, self._start, cfg.fs,
            center_freq=cfg.center_freq,
            enable_ca=cfg.enable_ca, enable_l1c=cfg.enable_l1c,
            enable_galileo=cfg.enable_galileo, enable_qzss=cfg.enable_qzss,
            enable_sbas=cfg.enable_sbas, enable_beidou=cfg.enable_beidou,
            el_mask=cfg.el_mask / R2D,
            amp_scale=provisional_amp, iono_enable=cfg.iono_enable,
            l1c_data=cfg.l1c_data, b1i_data=cfg.b1i_data,
            backend=cfg.backend)
        self._apply_auto_amp(self.engine)
        self._logf(f"Бэкенд синтеза: {self.engine.backend}"
                   + (f" ({self.engine.gpu_name})" if self.engine.gpu_name else ""))
        if cfg.enable_beidou:
            kept_b1i = sum(1 for c in self.engine.channels
                           if c.kind == "beidou")
            self._logf(
                f"Полоса: центр {cfg.center_freq / 1e6:.3f} МГц, "
                f"fs {cfg.fs / 1e6:.3f} Мвыб/с; каналов B1I сохранено "
                f"{kept_b1i}")
        dropped = self.engine.b1i_dropped
        if dropped:
            self._logf(f"BeiDou B1I вне полосы (±{cfg.fs / 2e6:.2f} МГц от "
                       f"центра): пропущено {len(dropped)} "
                       f"({', '.join(dropped[:8])}"
                       + (" ..." if len(dropped) > 8 else "") + ")")

        chans = self.engine.channel_info
        if not chans:
            raise ValueError("Нет видимых спутников выше маски по углу места")
        self._logf(f"Каналов: {len(chans)}")
        for c in chans:
            self._logf(f"  {c['name']:>4} {c['kind']:<8} "
                       f"элевация {c['elev'] * R2D:5.1f}° "
                       f"азимут {c['azim'] * R2D:5.1f}° "
                       f"амплитуда {c['amp']:.4f}")
        if self._channels is not None:
            self._channels(chans)
        self._plan()
        self.sink = self._make_sink()
        self.sink.start()

    # ------------------------------------------------------------------
    def _prepare_source(self) -> None:
        """Route to an existing IQ file (feature «готовый IQ-файл»)."""
        cfg = self.cfg
        path = cfg.iq_input
        if not os.path.exists(path):
            raise ValueError(f"IQ-файл не найден: {path}")
        self._source = IqFileSource(
            path, fmt=cfg.output_format, fs=cfg.fs,
            center_freq=cfg.center_freq, scale=cfg.output_scale,
            load_to_ram=bool(getattr(cfg, "iq_in_ram", False)),
            log=self._logf)
        self._source.start()
        self._logf("Повторное использование IQ: " + self._source.describe())
        self._logf("Источник воспроизведения: "
                   + ("RAM (файл загружен в память)"
                      if self._source.in_ram else "диск (потоковое чтение)"))
        # Honour the file's real radio parameters for playback.
        if self._source.fs > 0 and abs(self._source.fs - cfg.fs) > 1.0:
            self._logf(f"fs берётся из файла: {self._source.fs / 1e6:.3f} "
                       f"Мвыб/с (в настройках {cfg.fs / 1e6:.3f})")
            cfg.fs = self._source.fs
        if (self._source.center_freq > 0
                and abs(self._source.center_freq - cfg.center_freq) > 1.0):
            self._logf(f"центр берётся из файла: "
                       f"{self._source.center_freq / 1e6:.3f} МГц "
                       f"(в настройках {cfg.center_freq / 1e6:.3f})")
            cfg.center_freq = self._source.center_freq
        self._plan_source()
        self.sink = self._make_sink(ignore_file=True)
        self.sink.start()

    def _plan_source(self) -> None:
        cfg = self.cfg
        src = self._source
        assert src is not None
        self._loop = bool(cfg.loop)
        if cfg.duration and cfg.duration > 0:
            self._loop = bool(cfg.loop)
        else:
            self._loop = True  # indefinite playback
        kind = "B210" if cfg.use_usrp else "нет (только проверка)"
        source_kind = ("RAM (файл загружен в память)" if src.in_ram
                       else f"диск ({src.path})")
        self._logf(
            f"Источник зацикливания: {source_kind}; "
            f"передача: {kind}; повтор: {'ВКЛ' if self._loop else 'ВЫКЛ'}"
            + (f"; длительность {cfg.duration:.1f} с"
               if cfg.duration and cfg.duration > 0 else "; до «Стоп»"))

    def _run_source(self) -> int:
        """Play/validate an existing IQ file; returns produced samples."""
        cfg = self.cfg
        src = self._source
        assert src is not None
        block = max(1, int(cfg.fs * cfg.block_ms / 1000.0))
        produced = 0
        t0 = time.time()

        if not cfg.use_usrp:
            self._logf("Передача на B210 выключена: файл только проверяется "
                       "и описывается (плей в никуда)")
            first = src.read(block)
            if first.size == 0:
                raise ValueError(f"IQ-файл пуст или нечитаем: {src.path}")
            self._logf(f"Проверка IQ: прочитано {first.size} отсчётов, "
                       f"формат {src.fmt}, fs {src.fs:.0f}, "
                       f"центр {src.center_freq:.0f}")
            src.seek_start()
            return 0

        duration = cfg.duration if cfg.duration and cfg.duration > 0 else None
        target = int(duration * cfg.fs) if duration else None
        assert self.sink is not None
        src.seek_start()
        while not self._stop.is_set():
            b = src.read(block)
            if b.size == 0:
                if self._loop:
                    src.seek_start()
                    self._logf("Лууп (RAM): возврат в начало"
                               if src.in_ram else
                               "Лууп (диск): возврат в начало файла")
                    continue
                break
            if target is not None:
                remain = target - produced
                if remain <= 0:
                    break
                if b.size > remain:
                    b = b[:remain]
            self._emit_spectrum("TX", b)
            self.sink.write(b)
            produced += b.size
            if self._progress is not None:
                wall = max(1e-9, time.time() - t0)
                sim_s = produced / cfg.fs
                frac = (produced / target) if target else 0.0
                self._progress(frac, sim_s, wall, sim_s / wall)
            if target is not None and produced >= target:
                break
        return produced

    # ------------------------------------------------------------------
    def _run_engine(self, t0: float) -> int:
        """Generate IQ from the engine (RAM segment loop or straight write)."""
        cfg = self.cfg
        assert self.engine is not None and self.sink is not None
        block = max(1, int(cfg.fs * cfg.block_ms / 1000.0))
        if cfg.use_usrp:
            # Smaller TX writes keep the USRP fed smoothly (fewer underflows).
            block = min(block, max(1, int(cfg.fs * _TX_BLOCK_SECONDS)))
        duration = cfg.duration if cfg.duration and cfg.duration > 0 else None
        produced = 0

        if self._segment_seconds:
            seg_total = int(self._segment_seconds * cfg.fs)
            buf: list = []
            got = 0
            # «генерация» is a pure IQ-file run (no radio); a B210 run only
            # pre-synthesises the segment first, so it says «предгенерация».
            label = ("Идёт предгенерация" if cfg.use_usrp
                     else "Идёт генерация")
            log_progress = self._milestone_logger(label)
            # A TX segment reserves the first half of the bar for synthesis and
            # the second half for the (endless) streaming, so the bar keeps
            # advancing for the whole run (issue 1b).
            gen_span = 0.5 if cfg.use_usrp else 1.0
            # Check the stop event for every generated block so «Стоп» reacts
            # within roughly one block_ms, not after the whole segment.
            while got < seg_total:
                if self._stop.is_set():
                    break
                n = min(block, seg_total - got)
                b = self.engine.generate_block(n)
                got += n
                stopped = self._stop.is_set()
                if cfg.use_usrp:
                    # TX needs the whole segment in RAM for the endless loop.
                    # The anti-clip scale is applied to the whole segment once,
                    # after synthesis (one exact factor, no TX level jump), so
                    # the spectrum/level is emitted only once streaming starts.
                    buf.append(b)
                else:
                    # Pure IQ-file run: write every produced block straight
                    # away, so the progress bar tracks generation (and the bar
                    # does not stay empty for a long segment, nor jump).
                    b = self._headroom_ctl().process_block(b)
                    self.sink.write(b)
                    self._emit_spectrum("TX", b)
                    produced += len(b)
                if self._progress is not None and seg_total:
                    wall = max(1e-9, time.time() - t0)
                    sim_s = got / cfg.fs
                    self._progress(gen_span * got / seg_total, sim_s, wall,
                                   sim_s / wall if wall else 0.0)
                if self._phase is not None and seg_total:
                    # Pre-generation has its OWN 0…1 value (separate from the
                    # transmission bar the user sees once TX starts).
                    if cfg.use_usrp:
                        self._phase("pregen", got / seg_total, 0,
                                    got / cfg.fs, False)
                if seg_total:
                    log_progress(int(100 * got / seg_total),
                                 f"{got / cfg.fs:.1f} из "
                                 f"{seg_total / cfg.fs:.1f} с")
                if stopped:
                    break
            where = "в память" if cfg.use_usrp else "в файл"
            self._logf(f"Сегмент {got / cfg.fs:.1f} с сгенерирован {where}")

            if cfg.use_usrp:
                # One exact anti-clip scale for the whole pre-generated segment.
                self._headroom_ctl().process_segment(buf)
                # TX + loop (the segment branch is only entered when looping):
                # stream the RAM segment continuously until «Стоп».  ``duration``
                # is deliberately ignored as a stop condition here; it still
                # bounds file output and non-loop TX.
                stream_t0 = time.time()
                segment_len = seg_total or 1
                within = 0
                loops = 0
                while not self._stop.is_set():
                    for b in buf:
                        if self._stop.is_set():
                            break
                        self.sink.write(b)
                        self._emit_spectrum("TX", b)
                        produced += len(b)
                        within += len(b)
                        if self._phase is not None:
                            # Cyclic TX: the transmission bar wraps every loop
                            # (0…1 per segment) so it keeps updating instead of
                            # freezing at 100 %; the label shows elapsed/loops.
                            self._phase("tx", (within % segment_len) / segment_len,
                                        loops, produced / cfg.fs, True)
                    if self._progress is not None:
                        wall = max(1e-9, time.time() - t0)
                        sim_s = produced / cfg.fs
                        t = max(0.0, time.time() - stream_t0)
                        frac = gen_span + (1.0 - gen_span) * (
                            1.0 - math.exp(-t / _TX_STREAM_TAU_S))
                        self._progress(frac, sim_s, wall, sim_s / wall)
                    loops += 1
                    within = 0
        else:
            total = int(duration * cfg.fs) if duration else None
            if cfg.use_usrp:
                # TX must not serialise synthesis and send: pre-generate the
                # whole segment into RAM and stream from it, or (when it does
                # not fit) run synthesis and send in parallel via a bounded
                # queue.  Otherwise UHD underflows between blocks.
                return self._run_tx_stream(total, block, t0)
            log_progress = self._milestone_logger("Идёт генерация")
            while not self._stop.is_set():
                n = block
                if total is not None:
                    n = min(block, total - produced)
                    if n <= 0:
                        break
                samples = self.engine.generate_block(n)
                samples = self._headroom_ctl().process_block(samples)
                self.sink.write(samples)
                self._emit_spectrum("TX", samples)
                produced += n
                if self._progress is not None:
                    wall = max(1e-9, time.time() - t0)
                    sim_s = produced / cfg.fs
                    frac = (produced / total) if total else 0.0
                    self._progress(frac, sim_s, wall, sim_s / wall)
                if total is not None:
                    log_progress(int(100 * produced / total),
                                 f"{produced / cfg.fs:.1f} из "
                                 f"{total / cfg.fs:.1f} с")
        return produced

    # ------------------------------------------------------------------
    # TX streaming (pre-generated RAM segment or bounded producer/consumer)
    # ------------------------------------------------------------------
    def _generate_to_ram(self, target: int, block: int,
                         frac_scale: float = 1.0) -> tuple[list, int]:
        """Synthesise exactly ``target`` samples into a list of RAM blocks.

        Emits throttled progress so the user sees that work is happening while
        nothing is transmitted yet (the TX LED only lights after this phase),
        and honours ``self._stop`` between blocks so «Стоп» aborts promptly.
        ``frac_scale`` maps this phase onto a slice of the progress bar (the TX
        path reserves the first half for pre-generation and uses the second
        half for the streaming phase).
        """
        assert self.engine is not None
        cfg = self.cfg
        fs = cfg.fs if cfg.fs else 1.0
        blocks: list[np.ndarray] = []
        got = 0
        t0 = time.time()
        log_progress = self._milestone_logger("Предгенерация")
        self._logf(
            "TX: передача начнётся только после предгенерации сегмента в RAM; "
            "светодиод TX загорится по её завершении")
        while got < target and not self._stop.is_set():
            n = min(block, target - got)
            # Store the RAM segment as complex64 (half of complex128) so a
            # 25 Msps combined segment stays comfortably in memory.
            b = np.asarray(self.engine.generate_block(n), dtype=np.complex64)
            blocks.append(b)
            got += n
            # Advance the GUI/CLI progress bar while nothing is transmitted
            # yet (the sink has not received a single sample in this phase).
            if self._progress is not None:
                wall = max(1e-9, time.time() - t0)
                sim_s = got / fs
                frac = frac_scale * ((got / target) if target else 0.0)
                self._progress(frac, sim_s, wall, sim_s / wall)
            if self._phase is not None:
                self._phase("pregen", (got / target) if target else 1.0,
                            0, got / fs, False)
            pct = int(100 * got / target) if target else 100
            log_progress(pct, f"{got / fs:.1f} с из {target / fs:.1f} с")
        elapsed = max(1e-9, time.time() - t0)
        rate = got / elapsed
        self._logf(f"Синтез: {got / fs:.2f} с сигнала за {elapsed:.1f} с "
                   f"({rate / 1e6:.2f} Мвыб/с, {rate / fs:.2f}x realtime)")
        if got > 0 and rate < fs:
            self._logf(
                f"ВНИМАНИЕ: синтез медленнее реального времени "
                f"({rate / fs:.2f}x) — прямая передача без предгенерации "
                f"уходила бы в underflow; здесь сегмент предгенерён в RAM.")
        return blocks, got

    def _write_ram_blocks(self, blocks: list, target: int | None, float_t0: float,
                          loop: bool, base: float = 0.0,
                          span: float = 1.0) -> int:
        """Stream pre-generated RAM blocks to the sink (optionally looping).

        ``base``/``span`` place the streaming phase on the overall progress bar
        (the TX path uses ``base=0.5`` after pre-generation).  A finite
        ``target`` fills ``base..base+span`` monotonically; an endless stream
        (``target is None``) follows an asymptote that only reaches 100 % when
        the user actually stops the run.
        """
        cfg = self.cfg
        produced = 0
        stream_t0 = time.time()
        segment_len = sum(int(np.asarray(b).size) for b in blocks) or 1
        within = 0
        loops = 0
        while not self._stop.is_set():
            for b in blocks:
                # Honour «Стоп» within one block, not after the whole segment.
                if self._stop.is_set():
                    break
                self.sink.write(b)
                self._emit_spectrum("TX", b)
                produced += len(b)
                within += len(b)
                if self._phase is not None:
                    if target:
                        pfrac = min(1.0, produced / target)
                        cyclic = False
                    else:
                        pfrac = (within % segment_len) / segment_len
                        cyclic = bool(loop)
                    self._phase("tx", pfrac, loops, produced / cfg.fs, cyclic)
                if target is not None and produced >= target:
                    break
            if self._progress is not None:
                wall = max(1e-9, time.time() - float_t0)
                sim_s = produced / cfg.fs
                if target:
                    frac = base + span * min(1.0, produced / target)
                elif loop:
                    t = max(0.0, time.time() - stream_t0)
                    frac = base + span * (1.0 - math.exp(-t / _TX_STREAM_TAU_S))
                else:
                    frac = base
                self._progress(frac, sim_s, wall, sim_s / wall)
            if self._stop.is_set():
                break
            if target is not None and produced >= target:
                break
            if not loop:
                break
            loops += 1
            within = 0
        return produced

    def _warn_large_tx_segment(self, target: int) -> None:
        """Warn when a TX segment is so large that pre-generation is slow."""
        cfg = self.cfg
        fs = cfg.fs if cfg.fs else 1.0
        gen_s = target / _EST_SYNTH_RATE_SPS
        if gen_s <= _PREGEN_WARN_SECONDS:
            return
        gb = target * _TX_RAM_BYTES / 1e9
        self._logf(
            f"ВНИМАНИЕ: TX-сегмент {target / fs:.0f} с @ "
            f"{fs / 1e6:.2f} Мвыб/с (~{gb:.1f} ГБ RAM) — предгенерация "
            f"займёт ориентировочно ≥ {gen_s:.0f} с. Чтобы ускорить: "
            f"уменьшите fs, отключите B1I (--no-beidou) или используйте "
            f"cs8/узкополосный режим.")

    def _run_tx_stream(self, target: int | None, block: int, t0: float) -> int:
        """Non-looping TX: pre-generate to RAM or bounded producer/consumer."""
        cfg = self.cfg
        assert self.engine is not None
        budget = int(getattr(self, "_ram_budget_bytes", 0) or 0)
        # RAM segment is stored as complex64 (_TX_RAM_BYTES bytes/sample).
        max_samples = budget // _TX_RAM_BYTES if budget > 0 else 0
        if target is not None and max_samples > 0 and target <= max_samples:
            est_s = target / _EST_SYNTH_RATE_SPS
            self._logf(
                f"TX: предгенерация {target / cfg.fs:.2f} с в RAM "
                f"(~{sysinfo.human(target * _TX_RAM_BYTES)}), затем передача "
                f"из памяти; оценка предгенерации ~{est_s:.0f} с")
            self._warn_large_tx_segment(target)
            # Pre-generation fills the first half of the bar; transmission the
            # second half (issue 1b), so the bar advances during both phases.
            blocks, got = self._generate_to_ram(target, block, frac_scale=0.5)
            if got < target:
                self._logf(f"TX: предгенерация прервана ({got}/{target})")
            else:
                self._logf("TX: сегмент в RAM готов — потоковая передача "
                           "на B210 без синтеза между блоками")
            # One exact anti-clip scale for the whole pre-generated segment.
            self._headroom_ctl().process_segment(blocks)
            return self._write_ram_blocks(blocks, got, t0, loop=False,
                                          base=0.5, span=0.5)
        if target is not None:
            fit_s = (budget / _TX_RAM_BYTES / cfg.fs) if budget > 0 else 0.0
            self._logf(
                f"TX: {target / cfg.fs:.1f} с не помещается в бюджет RAM "
                f"(до {fit_s:.1f} с) — синтез и передача параллельно "
                f"(ограниченная очередь)")
            self._warn_large_tx_segment(target)
        else:
            self._logf("TX: длительность не задана — синтез и передача "
                       "параллельно (ограниченная очередь)")
        return self._run_tx_bounded(target, block, t0)

    def _run_tx_bounded(self, target: int | None, block: int,
                        t0: float) -> int:
        """Bounded producer/consumer: generation thread + send in main."""
        import queue

        cfg = self.cfg
        assert self.engine is not None
        q: "queue.Queue" = queue.Queue(maxsize=4)
        stop = self._stop
        errors: list[BaseException] = []

        def producer() -> None:
            try:
                got = 0
                while not stop.is_set():
                    if target is not None and got >= target:
                        break
                    n = block if target is None else min(block, target - got)
                    b = self.engine.generate_block(n)
                    while not stop.is_set():
                        try:
                            q.put((b, n), timeout=0.1)
                            break
                        except queue.Full:
                            continue
                    got += n
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
            finally:
                try:
                    q.put(None, timeout=1.0)
                except queue.Full:
                    pass

        thread = threading.Thread(target=producer, name="gnss-sim-gen",
                                  daemon=True)
        thread.start()
        produced = 0
        stream_t0 = time.time()
        while True:
            try:
                item = q.get(timeout=0.2)
            except queue.Empty:
                if stop.is_set() or not thread.is_alive():
                    break
                continue
            if item is None:
                break
            b, n = item
            b = self._headroom_ctl().process_block(b)
            self.sink.write(b)
            self._emit_spectrum("TX", b)
            produced += n
            if self._progress is not None:
                wall = max(1e-9, time.time() - t0)
                sim_s = produced / cfg.fs
                if target:
                    frac = min(1.0, produced / target)
                else:
                    t = max(0.0, time.time() - stream_t0)
                    frac = 1.0 - math.exp(-t / _TX_STREAM_TAU_S)
                self._progress(frac, sim_s, wall, sim_s / wall)
            if self._phase is not None:
                if target:
                    self._phase("tx", min(1.0, produced / target), 0,
                                produced / cfg.fs, False)
                else:
                    t = max(0.0, time.time() - stream_t0)
                    self._phase("tx", 1.0 - math.exp(-t / _TX_STREAM_TAU_S),
                                0, produced / cfg.fs, True)
            if stop.is_set():
                break
        thread.join(timeout=10.0)
        if errors:
            raise errors[0]
        return produced

    # ------------------------------------------------------------------
    def run(self) -> None:
        cfg = self.cfg
        try:
            if self.engine is None and self._source is None:
                self.prepare()
            assert self.sink is not None
            t0 = time.time()
            if self._source is not None:
                produced = self._run_source()
            else:
                produced = self._run_engine(t0)

            self.sink.close()
            self._report_underflows()
            wall = max(1e-9, time.time() - t0)
            self._logf(f"Готово: {produced} отсчётов ({produced / cfg.fs:.2f} с) "
                       f"за {wall:.2f} с ({produced / cfg.fs / wall:.2f}x realtime)")
            # Final progress: 100% even when the run had no finite target
            # (frac would be None/0.0).  The GUI also handles 100% in
            # _on_finished, but emitting it here is explicit and testable.
            if self._progress is not None:
                sim_s = produced / cfg.fs if cfg.fs else 0.0
                self._progress(1.0, sim_s, wall, sim_s / wall if wall else 0.0)
            self.error = None
        except Exception as exc:  # noqa: BLE001
            self.error = str(exc)
            self._logf(f"ОШИБКА: {exc}")
            try:
                if self.sink is not None:
                    self.sink.close()
                self._report_underflows()
            except Exception:
                pass
        finally:
            if self._finished is not None:
                self._finished(self.error)

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self.run, name="gnss-sim",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
