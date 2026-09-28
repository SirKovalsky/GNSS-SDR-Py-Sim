"""IQ output sinks (file formats and the common streaming interface).

The engine emits ``complex128`` blocks.  A *sink* consumes those blocks and is
responsible for sample formatting.  Supported file formats:

``cf32``  interleaved ``float32`` I/Q (GNU Radio ``complex64``)
``cs16``  interleaved ``int16`` I/Q (UHD ``sc16``; gps-sdr-sim ``-b 16``)
``cs8``   interleaved ``int8`` I/Q (HackRF style; gps-sdr-sim ``-b 8``)
``cs4``   interleaved ``int8`` I/Q scaled to 4-bit level
"""

from __future__ import annotations

import json
import os
import time
from typing import BinaryIO

import numpy as np


def _clip_round(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(np.rint(x), lo, hi)


class Sink:
    """Base class: consumes complex base-band blocks."""

    def start(self) -> None:  # pragma: no cover - trivial
        pass

    def write(self, samples: np.ndarray) -> None:  # pragma: no cover
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - trivial
        pass

    @property
    def count(self) -> int:  # pragma: no cover - trivial
        return 0

    def describe(self) -> str:  # pragma: no cover - trivial
        return "sink"


class FileSink(Sink):
    """Writes samples to an IQ file plus a side-car JSON metadata file."""

    def __init__(
        self,
        path: str,
        fmt: str = "cs16",
        fs: float | None = None,
        center_freq: float | None = None,
        scale: float = 10000.0,
        metadata: dict | None = None,
    ) -> None:
        self.path = path
        self.fmt = fmt.lower()
        if self.fmt not in ("cf32", "cs16", "cs8", "cs4"):
            raise ValueError(f"unsupported format: {fmt}")
        self.fs = fs
        self.center_freq = center_freq
        self.scale = float(scale)
        self.metadata = dict(metadata or {})
        self._fh: BinaryIO | None = None
        self._count = 0
        self._start_time: float | None = None

    def start(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self._fh = open(self.path, "wb")
        self._start_time = time.time()
        self._count = 0

    def write(self, samples: np.ndarray) -> None:
        if self._fh is None:
            self.start()
        assert self._fh is not None
        s = np.asarray(samples)
        if self.fmt == "cf32":
            data = np.empty(s.size * 2, dtype="<f4")
            data[0::2] = s.real
            data[1::2] = s.imag
            self._fh.write(data.tobytes())
        elif self.fmt == "cs16":
            data = np.empty(s.size * 2, dtype="<i2")
            data[0::2] = _clip_round(s.real * self.scale, -32768, 32767)
            data[1::2] = _clip_round(s.imag * self.scale, -32768, 32767)
            self._fh.write(data.tobytes())
        elif self.fmt == "cs8":
            data = np.empty(s.size * 2, dtype="<i1")
            data[0::2] = _clip_round(s.real * self.scale / 256.0, -128, 127)
            data[1::2] = _clip_round(s.imag * self.scale / 256.0, -128, 127)
            self._fh.write(data.tobytes())
        else:  # cs4
            data = np.empty(s.size * 2, dtype="<i1")
            data[0::2] = _clip_round(s.real * self.scale / 4096.0, -8, 7)
            data[1::2] = _clip_round(s.imag * self.scale / 4096.0, -8, 7)
            self._fh.write(data.tobytes())
        self._count += s.size

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        meta = dict(self.metadata)
        meta.update({
            "format": self.fmt,
            "sample_rate": self.fs,
            "center_freq": self.center_freq,
            "samples": self._count,
            "duration_s": (self._count / self.fs) if self.fs else None,
            "wall_seconds": (time.time() - self._start_time) if self._start_time else None,
        })
        try:
            with open(self.path + ".json", "w", encoding="utf-8") as fh:
                json.dump(meta, fh, indent=2, ensure_ascii=False)
        except OSError:
            pass

    @property
    def count(self) -> int:
        return self._count

    def describe(self) -> str:
        return f"file {self.path} ({self.fmt})"


class NullSink(Sink):
    """Discards samples but keeps a counter (useful for benchmarks)."""

    def __init__(self) -> None:
        self._count = 0

    def write(self, samples: np.ndarray) -> None:
        self._count += samples.size

    @property
    def count(self) -> int:
        return self._count

    def describe(self) -> str:
        return "null"


# ----------------------------------------------------------------------
# Reusing an existing IQ file
# ----------------------------------------------------------------------
#: Bytes per complex sample for every supported file format.
FORMAT_BYTES = {"cf32": 8, "cs16": 4, "cs8": 2, "cs4": 2}
#: Native NumPy dtype of the *interleaved* I/Q stream of each file format.
#: A RAM buffer keeps these samples in their native width (int16 = 4 B,
#: int8 = 2 B, float32 = 8 B per complex sample) instead of decoding the whole
#: file to ``complex128`` (16 B/sample), which used to need ~4x the file size.
FORMAT_DTYPE = {"cf32": "<f4", "cs16": "<i2", "cs8": "<i1", "cs4": "<i1"}
#: Fallback divisor applied to integer formats when the side-car is missing.
_FORMAT_SCALE_DIV = {"cs8": 256.0, "cs4": 4096.0}


def _format_div(fmt: str, scale: float) -> float:
    """Normalisation divisor that turns native samples into baseband units.

    ``cf32`` is already normalised; ``cs16`` is divided by ``scale``; ``cs8``
    and ``cs4`` by ``scale / (256 or 4096)`` (same rule as
    :func:`_decode_samples`).
    """
    fmt = fmt.lower()
    if fmt == "cf32":
        return 1.0
    if fmt == "cs16":
        return float(scale or 1.0)
    return float(scale or 1.0) / _FORMAT_SCALE_DIV.get(fmt, 1.0)


def load_iq_metadata(path: str) -> dict:
    """Return the side-car JSON for ``path`` (``<file>.json``) or ``{}``.

    The writer (:class:`FileSink`) stores ``<file>.json`` next to the data.
    """
    candidates = [path + ".json"]
    stem, _ext = os.path.splitext(path)
    if stem:
        candidates.append(stem + ".json")
    for cand in candidates:
        if os.path.isfile(cand):
            try:
                with open(cand, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, dict):
                    return data
            except (OSError, ValueError):
                continue
    return {}


def _decode_samples(raw: bytes, fmt: str, scale: float) -> np.ndarray:
    """Decode raw file bytes into a ``complex128`` array (normalised)."""
    fmt = fmt.lower()
    if fmt == "cf32":
        data = np.frombuffer(raw, dtype="<f4")
        n = data.size // 2
        return (data[0:2 * n:2].astype(np.float64)
                + 1j * data[1:2 * n:2].astype(np.float64))
    if fmt == "cs16":
        data = np.frombuffer(raw, dtype="<i2")
        n = data.size // 2
        return (data[0:2 * n:2].astype(np.float64)
                + 1j * data[1:2 * n:2].astype(np.float64)) / scale
    if fmt in ("cs8", "cs4"):
        data = np.frombuffer(raw, dtype="<i1")
        n = data.size // 2
        div = scale / _FORMAT_SCALE_DIV[fmt]
        return (data[0:2 * n:2].astype(np.float64)
                + 1j * data[1:2 * n:2].astype(np.float64)) / div
    raise ValueError(f"unsupported format: {fmt}")


class IqFileSource:
    """Reads an existing IQ file as complex blocks for playback/reuse.

    The format, sample rate and centre frequency are taken from the side-car
    ``<file>.json`` when present; otherwise the caller-supplied config values
    are used (and a warning is logged).  Integer formats are divided by the
    recorded ``output_scale`` (config fallback) so the returned samples are
    normalised complex values suitable for :class:`~gnss_sim.uhd_tx.UhdTxSink`.

    With ``load_to_ram=True`` the whole file is loaded into memory once and
    looped from RAM.  The buffer keeps the file's *native* interleaved dtype, so
    a 6 GB ``cs16`` file needs ~6 GB (not the ~24 GB a full ``complex128``
    decode used to require); :meth:`read` converts one block at a time to
    ``complex64``.  Available RAM is checked via :mod:`gnss_sim.sysinfo` and a
    clear Russian error is raised when the file would not fit, unless
    ``force=True`` or the available RAM is unknown (``0``).
    """

    def __init__(
        self,
        path: str,
        fmt: str = "cs16",
        fs: float = 0.0,
        center_freq: float = 0.0,
        scale: float = 10000.0,
        metadata: dict | None = None,
        log=None,
        load_to_ram: bool = False,
        force: bool = False,
    ) -> None:
        self.path = path
        self._log = log
        self._load_to_ram = bool(load_to_ram)
        self._force = bool(force)
        self._ram: np.ndarray | None = None
        self._ram_nsamples = 0
        self._ram_div = 1.0
        self._ram_pos = 0
        have_meta = metadata is not None
        meta = dict(metadata) if have_meta else load_iq_metadata(path)
        have_meta = bool(meta)
        self.metadata = meta

        self.fmt = str(meta.get("format") or fmt or "cs16").lower()
        if self.fmt not in FORMAT_BYTES:
            self.fmt = str(fmt or "cs16").lower()
        if self.fmt not in FORMAT_BYTES:
            self.fmt = "cs16"

        fs_meta = meta.get("sample_rate") or meta.get("fs")
        self.fs = float(fs_meta) if fs_meta else float(fs or 0.0)
        cf_meta = meta.get("center_freq")
        self.center_freq = (float(cf_meta) if cf_meta
                            else float(center_freq or 0.0))
        scale_meta = meta.get("output_scale")
        self.scale = float(scale_meta) if scale_meta else float(scale or 1.0)
        if not self.scale:
            self.scale = 1.0

        self._fh: BinaryIO | None = None
        self._size = 0
        self.total_samples = 0
        self._warned = not have_meta
        if not have_meta:
            self._warn(
                "Рядом с IQ-файлом нет .json — используются значения из "
                f"настроек (формат {self.fmt}, fs {self.fs:.0f}, "
                f"центр {self.center_freq:.0f})")

    # ------------------------------------------------------------------
    def _warn(self, msg: str) -> None:
        if self._log is not None:
            self._log(msg)

    def start(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        self._size = os.path.getsize(self.path)
        self.total_samples = self._size // FORMAT_BYTES[self.fmt]
        meta_samples = self.metadata.get("samples")
        if isinstance(meta_samples, (int, float)) and meta_samples > 0:
            self.total_samples = min(self.total_samples, int(meta_samples))
        if self._load_to_ram:
            if self._ram is None:
                self._load_ram()
            else:
                self._ram_pos = 0
        else:
            self._fh = open(self.path, "rb")

    def _load_ram(self) -> None:
        """Load the whole file into RAM in its *native* dtype (with RAM check).

        The buffer is exactly one file-sized allocation (no per-sample
        ``complex128`` blow-up).  A previously loaded buffer is released before
        the new one is allocated so a re-load does not double the footprint.
        """
        from . import sysinfo

        # Release any previous buffer before allocating the new one.
        self._ram = None
        self._ram_nsamples = 0
        n_samp = int(self.total_samples)
        bps = FORMAT_BYTES.get(self.fmt, 4)
        need = max(0, n_samp) * bps  # native size == file size
        avail = sysinfo.available_ram()
        if not self._force and avail > 0 and need > avail:
            raise RuntimeError(
                "Недостаточно оперативной памяти для загрузки IQ-файла "
                f"целиком: нужно ~{sysinfo.human(need)}, доступно "
                f"{sysinfo.human(avail)}. Воспроизведение выполняется только "
                "из RAM — уменьшите файл или освободите память.")
        self._ram = np.fromfile(
            self.path, dtype=FORMAT_DTYPE[self.fmt], count=max(0, n_samp) * 2)
        self._ram_nsamples = int(self._ram.size) // 2
        self._ram_div = _format_div(self.fmt, self.scale)
        self._ram_pos = 0
        self._warn(
            f"IQ-файл загружен в RAM: {self._ram_nsamples} отсч. "
            f"({self.fmt}, ~{sysinfo.human(int(self._ram.nbytes))}; "
            "нативный формат, без complex128)")

    def seek_start(self) -> None:
        if self._ram is not None:
            self._ram_pos = 0
            return
        if self._fh is None:
            self.start()
        assert self._fh is not None
        self._fh.seek(0)

    def read(self, n: int) -> np.ndarray:
        """Return up to ``n`` normalised complex samples (empty at EOF)."""
        n = max(0, int(n))
        if self._ram is not None:
            return self._read_ram(n)
        if self._fh is None:
            self.start()
        assert self._fh is not None
        bps = FORMAT_BYTES[self.fmt]
        raw = self._fh.read(n * bps)
        if not raw:
            return np.zeros(0, dtype=np.complex128)
        return _decode_samples(raw, self.fmt, self.scale)

    def _read_ram(self, n: int) -> np.ndarray:
        """Convert the next ``n`` samples of the native RAM buffer to complex64.

        Only one block-sized ``complex64`` array is allocated per call; the
        file-sized native buffer is never decoded wholesale.
        """
        total = self._ram_nsamples
        pos = self._ram_pos
        if n <= 0 or pos >= total:
            return np.zeros(0, dtype=np.complex64)
        end = min(total, pos + n)
        inter = self._ram[2 * pos:2 * end]
        out = np.empty(end - pos, dtype=np.complex64)
        out.real = inter[0::2]
        out.imag = inter[1::2]
        if self._ram_div != 1.0:
            out /= self._ram_div
        self._ram_pos = end
        return out

    def read_all(self) -> np.ndarray:  # pragma: no cover - convenience
        self.seek_start()
        return self.read(self.total_samples)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    @property
    def in_ram(self) -> bool:
        """True when the file has been loaded into memory."""
        return self._ram is not None

    @property
    def duration_s(self) -> float:
        return (self.total_samples / self.fs) if self.fs else 0.0

    def describe(self) -> str:
        where = " [RAM]" if self.in_ram else ""
        return (f"IQ-файл {self.path} ({self.fmt}, fs "
                f"{self.fs / 1e6:.3f} Мвыб/с, центр "
                f"{self.center_freq / 1e6:.3f} МГц, "
                f"{self.total_samples} отсч./{self.duration_s:.2f} с)"
                + (" [метаданные .json]" if self.metadata else " [настройки]")
                + where)
