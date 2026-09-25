"""Qt-виджет спектра (PSD) на pyqtgraph.

Виджет показывает две кривые: спектр сгенерированного TX-сигнала и (при
мониторинге RX) спектр принятого сигнала.  Оценка PSD выполняется чистым
NumPy (:func:`gnss_sim.power.welch_psd`), pyqtgraph нужен только для
отрисовки.  Модуль импортируется без железа и работает в offscreen-режиме.
"""

from __future__ import annotations

import numpy as np
from PyQt5 import QtGui, QtWidgets

from .power import welch_psd

try:  # pragma: no cover - зависит от окружения
    import pyqtgraph as pg
except Exception:  # noqa: BLE001
    pg = None


class SpectrumWidget(QtWidgets.QWidget):
    """PSD-график с осями «частота, МГц» / «дБFS» и кривыми TX/RX.

    When ``center_freq`` is non-zero the X axis shows the absolute frequency
    (centre + offset); otherwise it shows the baseband offset in MHz.  The
    axis labels are updated accordingly.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        vbox = QtWidgets.QVBoxLayout(self)
        self.plot = None
        self.curve_tx = None
        self.curve_rx = None
        self._count = {"TX": 0, "RX": 0}
        self._x_absolute = False

        if pg is None:
            vbox.addWidget(QtWidgets.QLabel(
                "pyqtgraph не установлен — график спектра недоступен.\n"
                "Установите: pip install pyqtgraph"))
            return

        self.plot = pg.PlotWidget()
        self.plot.setBackground("w")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self._set_x_label(False)
        self.plot.setLabel("left", "Уровень", units="дБFS")
        self.plot.addLegend(offset=(10, 10))
        self.curve_tx = self.plot.plot(
            pen=pg.mkPen("#1f77b4", width=1), name="TX")
        self.curve_rx = self.plot.plot(
            pen=pg.mkPen("#d62728", width=1), name="RX")
        # Readable tick labels.
        try:
            tick_font = QtGui.QFont("Arial", 10)
            for axis in ("bottom", "left"):
                self.plot.getAxis(axis).setStyle(tickFont=tick_font)
        except Exception:  # noqa: BLE001
            pass
        vbox.addWidget(self.plot)

    # ------------------------------------------------------------------
    def _set_x_label(self, absolute: bool) -> None:
        if self.plot is None:
            return
        self._x_absolute = absolute
        text = "Частота" if absolute else "Смещение"
        self.plot.setLabel("bottom", text, units="МГц")

    # ------------------------------------------------------------------
    def update_spectrum(self, samples: np.ndarray, fs: float,
                        channel: str = "TX", center_freq: float = 0.0) -> None:
        """Отрисовать PSD блока ``samples`` (частота дискретизации ``fs``).

        ``channel`` = ``"TX"`` или ``"RX"`` (регистр не важен).  ``center_freq``
        — центральная частота (Гц): при ненулевом значении ось X абсолютная
        (центр + смещение), иначе — смещение от нуля.
        """
        if self.plot is None:
            return
        x = np.asarray(samples).ravel()
        if x.size < 8:
            return
        try:
            freqs, psd = welch_psd(x, float(fs))
        except Exception:  # noqa: BLE001 - график не должен ломать генерацию
            return
        if freqs.size == 0:
            return
        label = "RX" if str(channel).upper().startswith("R") else "TX"
        curve = self.curve_rx if label == "RX" else self.curve_tx
        if curve is None:
            return
        if center_freq and float(center_freq) > 0:
            self._set_x_label(True)
            x_axis = (float(center_freq) + freqs) / 1e6
        else:
            self._set_x_label(False)
            x_axis = freqs / 1e6
        curve.setData(x_axis, psd)
        self._count[label] += 1

    def clear(self) -> None:
        for curve in (self.curve_tx, self.curve_rx):
            if curve is not None:
                curve.setData([], [])
        self._count = {"TX": 0, "RX": 0}

    def describe(self) -> str:
        return (f"spectrum TX={self._count['TX']} RX={self._count['RX']}")
