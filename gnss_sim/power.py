"""Мощность сигнала, регулятор усиления TX и профиль задержек канала.

Модуль чистый NumPy — без UHD и без Qt, поэтому его можно использовать в
тестах и на машине без подключённого B210.

Шкала dBFS (согласована с SDR_Scan)
-----------------------------------
Принята шкала ``dBFS = 10*log10(mean(|x|^2))`` (то же, что
``20*log10(rms)``, где ``rms = sqrt(mean(|x|^2))``).  Комплексный тон
единичной амплитуды (``|x| = 1``) даёт **0 dBFS** — это полная шкала
комплексного тракта ``fc32``/``sc16``.  Реальный косинус амплитуды 1.0
имеет ``rms = 1/sqrt(2)`` и даёт ``-3.01 dBFS``.  Уровень 0 dB — это не
«мощность 1 Вт», а верх шкалы АЦП/ЦАП.

Физика B210 (кратко)
--------------------
B210 построен на одном AD9361: отдельные LO для RX и TX, поэтому приём и
передача могут идти **одновременно и на разных частотах** (FDD).  Но у
канала 0 порт ``TX/RX`` один (TDD-переключатель), поэтому нельзя слушать
собственный TX на том же порту.  Для контроля передачи: TX на
``tx_channel`` (порт ``TX/RX``), RX на **другом** канале (порт ``RX2``),
соединённом кабелем/ответвителем.  Один RX LO общий для обоих RX-каналов,
один TX LO — для обоих TX-каналов.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np

_EPS = 1e-24

# ----------------------------------------------------------------------
# Automatic baseband headroom (anti-clip) control
# ----------------------------------------------------------------------
#: Default safe composite peak the transmitted block/segment is scaled to.
DEFAULT_HEADROOM_TARGET = 0.7
#: Default estimated composite peak (before headroom) the automatic per-scene
#: amplitude targets; kept just under full scale so a typical scene does not
#: clip even before :class:`HeadroomController` runs.
DEFAULT_AUTO_AMP_PEAK = 1.0
#: Clamp range of the automatic amplitude: keeps a GPS-only scene usable and
#: stops a many-channel scene from going too quiet.
AUTO_AMP_MIN = 0.02
AUTO_AMP_MAX = 0.15
#: Empirical margin for ``peak / (amp * sum(per-channel amplitudes))`` measured
#: on real multi-GNSS scenes (≈0.74…1.34); the value is chosen so the estimated
#: pre-headroom peak stays below full scale.
AUTO_AMP_PEAK_RATIO = 1.5
#: Full-scale amplitude of the *delivery format* in baseband units.  A unit
#: float is 0 dBFS only for ``cf32``/live USRP; ``cs16`` files use
#: ``output_scale`` (default 10000), so an ``|x|`` of 1.0 is NOT full scale —
#: the real clip limit is ``32767 / output_scale`` (≈3.28).  Ignoring this was
#: the regression that made auto-amp + headroom attenuate a normal scene by
#: ~10 dB (no F9P fix).  The runner passes the concrete value per run.
DEFAULT_FULL_SCALE = 1.0


@dataclass(frozen=True)
class LevelStats:
    """Peak/RMS/clip statistics of a complex baseband block (same amplitude
    reference as :func:`rms_dbfs`: a unit-amplitude complex tone is 0 dBFS)."""

    peak: float = 0.0
    rms: float = 0.0
    clips: int = 0
    samples: int = 0

    @property
    def peak_dbfs(self) -> float:
        return 20.0 * math.log10(self.peak) if self.peak > 0.0 else float("-inf")

    @property
    def rms_dbfs(self) -> float:
        return 20.0 * math.log10(self.rms) if self.rms > 0.0 else float("-inf")


def _magnitudes(x) -> np.ndarray:
    return np.abs(np.asarray(x)).ravel().astype(np.float64, copy=False)


def level_stats(samples) -> LevelStats:
    """Peak, RMS and clipped-sample count (``|x| > 1``) of ``samples``."""
    mag = _magnitudes(samples)
    if mag.size == 0:
        return LevelStats()
    peak = float(mag.max())
    rms = float(np.sqrt(np.mean(mag * mag)))
    clips = int(np.count_nonzero(mag > 1.0))
    return LevelStats(peak, rms, clips, int(mag.size))


def combine_levels(blocks: Sequence) -> LevelStats:
    """Level statistics over several blocks (energy-weighted RMS)."""
    peak = 0.0
    clips = 0
    n = 0
    energy = 0.0
    for b in blocks:
        mag = _magnitudes(b)
        if mag.size == 0:
            continue
        peak = max(peak, float(mag.max()))
        clips += int(np.count_nonzero(mag > 1.0))
        n += int(mag.size)
        energy += float(np.dot(mag, mag))
    rms = math.sqrt(energy / n) if n else 0.0
    return LevelStats(peak, rms, clips, n)


def auto_amp_scale(weights: Iterable[float],
                   target: float = DEFAULT_AUTO_AMP_PEAK,
                   full_scale: float = DEFAULT_FULL_SCALE) -> float:
    """Single scene-wide amplitude derived from the per-channel weights.

    ``weights`` are the geometry/antenna weighted per-channel amplitudes (the
    composite peak grows roughly linearly with their sum).  ``full_scale`` is
    the clip limit of the delivery format in baseband units (1.0 for cf32/live
    USRP, ``32767/output_scale`` for cs16); the target peak is expressed
    relative to it.  The result is clamped to ``AUTO_AMP_MIN..AUTO_AMP_MAX`` so
    a GPS-only scene stays loud and a many-channel scene does not become too
    quiet.
    """
    total = float(sum(w for w in weights if w and w > 0.0))
    if total <= 0.0:
        return AUTO_AMP_MAX
    limit = float(full_scale) if full_scale and full_scale > 0.0 else 1.0
    value = float(target) * limit / (AUTO_AMP_PEAK_RATIO * total)
    return float(min(AUTO_AMP_MAX, max(AUTO_AMP_MIN, value)))


class HeadroomController:
    """Anti-clip scaler for the composite baseband.

    ``process_segment`` is used when the whole block/segment is pre-generated
    into RAM: the exact global peak is measured and **one** scale is applied to
    every sample (no level discontinuity on the air).  ``process_block`` is the
    streaming fallback: the scale starts from the first block and only ever
    decreases when a later block would clip, so no sample exceeds the target.

    ``target`` is a fraction of ``full_scale`` — the clip limit of the delivery
    format (1.0 for cf32/live USRP, ``32767/output_scale`` for cs16).  For a
    cs16 file the effective target is therefore ~2.3, not 0.7, and a normal
    multi-GNSS scene (float peak ≈2.1) is left untouched — which is what the
    F9P needs to acquire.
    """

    def __init__(self, target: float = DEFAULT_HEADROOM_TARGET,
                 enabled: bool = True, log: Callable[[str], None] | None = None,
                 name: str = "TX",
                 full_scale: float = DEFAULT_FULL_SCALE) -> None:
        limit = float(full_scale) if full_scale and full_scale > 0.0 else 1.0
        self.target = float(target) * limit
        self.enabled = bool(enabled) and self.target > 0.0
        self.name = str(name)
        self._log = log
        self.scale = 1.0
        self._logged = False

    def _logf(self, msg: str) -> None:
        if self._log is not None:
            try:
                self._log(msg)
            except Exception:  # noqa: BLE001 - logging must not break a run
                pass

    @staticmethod
    def _scale_block(x, scale: float):
        if scale == 1.0:
            return x
        a = np.asarray(x)
        out = a * scale
        if out.dtype != a.dtype:
            out = out.astype(a.dtype, copy=False)
        return out

    @staticmethod
    def _fmt(st: LevelStats) -> str:
        return (f"пик {st.peak:.3f} (пик {st.peak_dbfs:.1f} dBFS), "
                f"RMS {st.rms:.3f} ({st.rms_dbfs:.1f} dBFS), "
                f"клип {st.clips}")

    def _report(self, before: LevelStats, after: LevelStats, scale: float,
                stream: bool) -> None:
        where = "поток" if stream else "сегмент"
        self._logf(
            f"{self.name} headroom ({where}): масштаб {scale:.4f} "
            f"(цель пика {self.target:.3f}); до: {self._fmt(before)}; "
            f"после: {self._fmt(after)}")

    def process_segment(self, blocks: list) -> tuple[list, float]:
        """Scale a whole pre-generated segment by one exact factor."""
        if not self.enabled or not blocks:
            return blocks, 1.0
        before = combine_levels(blocks)
        scale = (1.0 if before.peak <= self.target
                 else self.target / before.peak)
        if scale != 1.0:
            for i in range(len(blocks)):
                blocks[i] = self._scale_block(blocks[i], scale)
        after = before if scale == 1.0 else combine_levels(blocks)
        self.scale = scale
        self._report(before, after, scale, stream=False)
        return blocks, scale

    def process_block(self, block):
        """Scale one streaming block (scale never increases above 1.0)."""
        if not self.enabled:
            return block
        st = level_stats(block)
        if st.samples == 0:
            return block
        if st.peak > 0.0:
            need = self.target / st.peak
            if need < self.scale:
                self.scale = need
        out = self._scale_block(block, self.scale)
        if not self._logged:
            self._logged = True
            after = level_stats(out)
            self._report(st, after, self.scale, stream=True)
        return out


def rms_dbfs(x: np.ndarray) -> float:
    """Уровень комплексного сигнала в dBFS: ``10*log10(mean(|x|^2))``.

    Комплексный тон амплитуды 1.0 -> 0 dBFS (см. модульный docstring).
    Пустой массив -> ``-inf``.
    """
    a = np.asarray(x)
    if a.size == 0:
        return float("-inf")
    a = a.astype(np.complex128, copy=False)
    p = float(np.mean(a.real * a.real + a.imag * a.imag))
    return 10.0 * float(np.log10(p + _EPS))


class PowerRegulator:
    """Ограниченный пропорциональный регулятор усиления TX по уровню RX.

    Алгоритм (устойчивый, без неограниченных колебаний):

    1. Сглаживание измерения EMA: ``m <- ema*m + (1-ema)*measured``.
    2. Ошибка ``e = target - m`` (или ``target_snr - SNR``).
    3. Шаг ``d = clip(e, -step_db, step_db)``; при ``|e| <= deadband_db``
       шаг нулевой (мёртвая зона гасит шум измерения).
    4. Новое усиление ``gain = clip(gain + d, gain_min, gain_max)``.

    Так как шаг ограничен, а усиление зажато в ``[gain_min, gain_max]``,
    регулятор не может «убежать» или раскачаться.  При линейной зависимости
    «усиление [дБ] -> измеренный уровень [дБ]» сходимость геометрическая с
    множителем ``(1 - ema)``.

    Если задан ``target_dbfs``, регулируется абсолютный уровень; если
    ``target_snr_db`` — отношение сигнал/шум, и в :meth:`update` нужно
    передавать ``noise_dbfs``.
    """

    def __init__(
        self,
        target_dbfs: float | None = None,
        target_snr_db: float | None = None,
        gain_min: float = -30.0,
        gain_max: float = 30.0,
        step_db: float = 1.0,
        ema: float = 0.3,
        deadband_db: float = 0.5,
        initial_gain: float = 0.0,
    ) -> None:
        if target_dbfs is None and target_snr_db is None:
            raise ValueError(
                "Нужен target_dbfs или target_snr_db (нечего регулировать)")
        if gain_min > gain_max:
            raise ValueError(f"gain_min ({gain_min}) > gain_max ({gain_max})")
        if step_db <= 0.0:
            raise ValueError("step_db должен быть > 0")
        if not 0.0 < ema <= 1.0:
            raise ValueError("ema должен быть в (0, 1]")

        self.target_dbfs = None if target_dbfs is None else float(target_dbfs)
        self.target_snr_db = (None if target_snr_db is None
                              else float(target_snr_db))
        self.gain_min = float(gain_min)
        self.gain_max = float(gain_max)
        self.step_db = float(step_db)
        self.ema = float(ema)
        self.deadband_db = float(deadband_db)

        self.gain = float(np.clip(initial_gain, self.gain_min, self.gain_max))
        self.updates = 0
        self._meas: float | None = None
        self.last_measured_dbfs: float | None = None
        self.last_noise_dbfs: float | None = None
        self.last_snr_db: float | None = None
        self.last_error_db: float | None = None

    # ------------------------------------------------------------------
    def update(self, measured_dbfs: float, noise_dbfs: float | None = None) -> float:
        """Скорректировать усиление и вернуть новое значение (дБ)."""
        if measured_dbfs is None or not np.isfinite(measured_dbfs):
            return self.gain

        measured = float(measured_dbfs)
        self.last_measured_dbfs = measured
        self.last_noise_dbfs = (None if noise_dbfs is None
                                else float(noise_dbfs))
        self._meas = (measured if self._meas is None
                      else self.ema * measured + (1.0 - self.ema) * self._meas)

        snr: float | None = None
        if self.last_noise_dbfs is not None and np.isfinite(self.last_noise_dbfs):
            snr = self._meas - self.last_noise_dbfs
        self.last_snr_db = snr

        if self.target_snr_db is not None:
            if snr is None:
                return self.gain  # нет опоры — не двигаем усиление
            error = self.target_snr_db - snr
        else:
            assert self.target_dbfs is not None
            error = self.target_dbfs - self._meas

        self.last_error_db = float(error)
        if abs(error) <= self.deadband_db:
            delta = 0.0
        else:
            delta = float(np.clip(error, -self.step_db, self.step_db))

        self.gain = float(np.clip(self.gain + delta, self.gain_min,
                                  self.gain_max))
        self.updates += 1
        return self.gain

    # ------------------------------------------------------------------
    def converged(self) -> bool:
        return (self.last_error_db is not None
                and abs(self.last_error_db) <= self.deadband_db)

    def state(self) -> dict:
        return {
            "gain": self.gain,
            "gain_min": self.gain_min,
            "gain_max": self.gain_max,
            "step_db": self.step_db,
            "ema": self.ema,
            "deadband_db": self.deadband_db,
            "target_dbfs": self.target_dbfs,
            "target_snr_db": self.target_snr_db,
            "measured_dbfs": self._meas,
            "noise_dbfs": self.last_noise_dbfs,
            "snr_db": self.last_snr_db,
            "error_db": self.last_error_db,
            "updates": self.updates,
            "converged": self.converged(),
        }


def _next_fast_len(n: int) -> int:
    try:
        return int(np.fft.next_fast_len(n))
    except Exception:  # pragma: no cover - старые numpy
        return int(n)


def delay_profile(
    tx: np.ndarray,
    rx: np.ndarray,
    fs: float,
    max_delay: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Профиль задержек канала (эхо/многолучёвость) по TX-опоре и RX-захвату.

    Считается линейная (не круговая) взаимная корреляция
    ``corr[k] = sum_n rx[n+k] * conj(tx[n])`` через БПФ с zero-padding.
    Возвращает ``(delays_s, levels_db)``:

    * ``delays_s`` — неотрицательные задержки в секундах (k / fs);
    * ``levels_db`` — уровень относительно самого сильного луча (пик 0 дБ),
      т.е. удобно видеть эхо: «-6 дБ на 350 нс».

    ``max_delay`` (с) ограничивает окно поиска.  Пустой вход -> пустые
    массивы.
    """
    a = np.asarray(tx, dtype=np.complex128).ravel()
    b = np.asarray(rx, dtype=np.complex128).ravel()
    if a.size == 0 or b.size == 0:
        return np.zeros(0), np.zeros(0)

    max_lag = b.size - 1
    if max_delay is not None and max_delay > 0:
        max_lag = min(max_lag, int(round(float(max_delay) * float(fs))))

    n = _next_fast_len(a.size + b.size - 1)
    corr = np.fft.ifft(np.fft.fft(b, n) * np.conj(np.fft.fft(a, n)))
    corr = corr[:max_lag + 1]

    e_tx = float(np.sum(a.real * a.real + a.imag * a.imag))
    e_rx = float(np.sum(b.real * b.real + b.imag * b.imag))
    norm = np.sqrt(e_tx * e_rx)
    mag = np.abs(corr) / (norm if norm > _EPS else 1.0)

    peak = float(mag.max()) if mag.size else 0.0
    if peak > _EPS:
        levels = 20.0 * np.log10(mag / peak + _EPS)
    else:
        levels = np.full(mag.shape, -np.inf)
    delays = np.arange(mag.size, dtype=np.float64) / float(fs)
    return delays, levels


def estimate_noise_floor(blocks: Iterable[np.ndarray] | np.ndarray) -> float:
    """Медиана мощности блоков (дБ).  Устойчива к всплескам/эхо.

    Принимает как отдельный массив, так и итерируемое блоков.
    """
    if isinstance(blocks, np.ndarray):
        items: list[np.ndarray] = [blocks]
    else:
        items = list(blocks)
    powers = []
    for b in items:
        arr = np.asarray(b)
        if arr.size == 0:
            continue
        arr = arr.astype(np.complex128, copy=False)
        powers.append(float(np.mean(arr.real * arr.real + arr.imag * arr.imag)))
    if not powers:
        return float("-inf")
    med = float(np.median(powers))
    return 10.0 * float(np.log10(med + _EPS))


def welch_psd(
    x: np.ndarray,
    fs: float,
    nfft: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """Оценка спектральной плотности мощности методом Уэлча (чистый NumPy).

    Окно Ханна, перекрытие 50 %.  Возвращает ``(freqs_hz, psd_dbfs)``:

    * ``freqs_hz`` — частоты, центрированные на 0 (``-fs/2 .. fs/2``);
    * ``psd_dbfs`` — ``10*log10(PSD)`` в согласованной с модулем шкале dBFS:
      интеграл PSD по частоте для комплексного тона амплитуды 1.0 равен
      0 dBFS (нормировка на энергию окна ``sum(w^2)`` и ``fs``).

    SciPy не требуется.  Блок короче ``nfft`` обрабатывается целиком
    (``nfft`` уменьшается до длины входа).  Пустой вход -> пустые массивы.
    """
    a = np.asarray(x).ravel()
    if a.size == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    if not np.isfinite(fs) or fs <= 0.0:
        raise ValueError("fs должен быть > 0")

    a = a.astype(np.complex128, copy=False)
    n = int(min(max(2, int(nfft)), a.size))
    win = np.hanning(n)
    u = float(np.sum(win * win))
    if u <= 0.0:  # pragma: no cover - защита от вырождения
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    step = max(1, n // 2)
    n_seg = 1 + (a.size - n) // step
    if n_seg < 1:
        n_seg = 1
    acc = np.zeros(n, dtype=np.float64)
    for i in range(n_seg):
        seg = a[i * step:i * step + n]
        spec = np.fft.fft(seg * win)
        acc += (spec.real * spec.real + spec.imag * spec.imag)
    acc *= 1.0 / (float(fs) * u * float(n_seg))

    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / float(fs)))
    psd = np.fft.fftshift(acc)
    psd_dbfs = 10.0 * np.log10(psd + _EPS)
    return freqs.astype(np.float64), psd_dbfs.astype(np.float64)
