"""PyQt5 GUI for the GNSS simulator (GPS L1 C/A + L1C, Galileo E1, SBAS, QZSS).

Run with ``run.py`` (which imports UHD before Qt on Windows) or
``python -m gnss_sim.gui``.  The "Карта и трек" tab draws an OpenStreetMap map,
lets you click a haul route and turns it into a mining dump-truck motion file.
"""

from __future__ import annotations

from .nativelog import (install_native_stderr_filter, quiet_uhd_gui,
                        set_native_stderr_sink)

# UHD must be imported *before* PyQt5 on Windows (creating a USRP after Qt has
# loaded can crash with 0xC0000005); ask for its [INFO]/[WARNING] lines and
# capture its native stderr before it loads.
quiet_uhd_gui()
install_native_stderr_filter()
try:  # pragma: no cover - platform dependent
    import uhd  # noqa: F401
except Exception:
    pass

import math
import os
import sys
import threading

from PyQt5 import QtCore, QtGui, QtWidgets

from . import track as trackmod
from . import ublox
from .config import (SimConfig, derive_band_key, derive_band_plan,
                     derive_nav_mode)
from .mapview import OsmMap
from .runner import SimulationRunner
from .spectrum import SpectrumWidget

_APP_TITLE = "GNSS Sim — GPS L1 C/A + L1C + Galileo E1 + SBAS + QZSS"

_RINEX_FILTER = ("RINEX (*.n *.??n *.??g *.rnx *.nav *.gz);;"
                 "Все файлы (*)")
_RINEX_FILTER_ALL = "Все файлы (*)"
_IQ_INPUT_FILTER = "IQ (*.cs16 *.cf32 *.cs8 *.cs4 *.bin *.iq);;Все файлы (*)"

#: Defaults shared by the controls and the per-group reset buttons.
_DEFAULT_FS = 2600000
_DEFAULT_FC = 1575420000
_DEFAULT_OUT = "gnss_sim_output.cs16"

#: Output extensions that track the format combo (B5).
_IQ_EXTS = (".cs16", ".cf32", ".cs8", ".cs4", ".bin", ".iq")
#: Local prefix of the merged multi-GNSS RINEX (published for past days only).
_MERGED_RINEX_PREFIX = "BRDC00IGS_R_"
#: Log colours (B1): errors red, warnings amber.
_LOG_ERROR_COLOR = "#d32f2f"
_LOG_WARN_COLOR = "#b26a00"


class _DateLockedDateTimeEdit(QtWidgets.QDateTimeEdit):
    """``QDateTimeEdit`` where only the TIME can be edited by the user.

    The start DATE is resolved from the RINEX (or «Сейчас»/merged fallback) and
    must not be free-editable.  Programmatic ``setDateTime`` still works (the
    date is part of the value), but arrow keys, clicks and typing that would
    land on the year/month/day section are redirected to the hour.
    """

    @staticmethod
    def _is_date_section(section) -> bool:
        return section in (QtWidgets.QDateTimeEdit.YearSection,
                           QtWidgets.QDateTimeEdit.MonthSection,
                           QtWidgets.QDateTimeEdit.DaySection)

    def _lock_to_time(self) -> None:
        if self._is_date_section(self.currentSection()):
            self.setCurrentSection(QtWidgets.QDateTimeEdit.HourSection)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: N802
        self._lock_to_time()
        super().keyPressEvent(event)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        super().mousePressEvent(event)
        self._lock_to_time()

    def stepBy(self, steps: int) -> None:  # noqa: N802
        self._lock_to_time()
        super().stepBy(steps)


class _WheelGuard(QtCore.QObject):
    """Ignore mouse-wheel changes for a combo box unless its popup is open.

    Qt's default behaviour lets a stray wheel scroll silently change a combo
    value.  This filter swallows the wheel event while the dropdown is closed
    (so scrolling the settings page never edits values), while still allowing
    scrolling inside the open popup.
    """

    def __init__(self, combo: QtWidgets.QComboBox) -> None:
        super().__init__(combo)
        self._combo = combo

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.Wheel:
            try:
                popup_open = self._combo.view().isVisible()
            except Exception:  # noqa: BLE001
                popup_open = False
            if not popup_open:
                event.ignore()
                return True
        return False

_HELP_HTML = """
<h3>Что делает программа</h3>
<p>Симулятор формирует цифровой IQ-сигнал GNSS (GPS L1 C/A + L1C, Galileo E1,
QZSS, SBAS, BeiDou B1I) и по выбору записывает его в файл и/или передаёт через
USRP B210 в эфир.</p>
<ul>
<li><b>«Сгенерировать IQ»</b> — только записать IQ-файл (передача на B210
выключена).</li>
<li><b>«Старт»</b> — записать IQ-файл, если задан файл выхода; передать в эфир,
если отмечено «Передавать на B210»; если отмечено и то и другое — сделать
оба действия.</li>
<li><b>«Стоп»</b> — остановить генерацию/передачу.</li>
</ul>

<h3>Сигналы — единственный выбор систем</h3>
<p>Галочки сигналов полностью определяют сеанс. Диапазон
(<code>l1</code>/<code>b1i</code>/<code>all</code>), центральная частота, частота
дискретизации и источник RINEX выводятся из них автоматически и показаны только
для чтения (поля «Радиотракт (вычисляется)» и примечание в «Эфемериды и
позиция»). Ручная пересборка полосы осталась только в CLI
(<code>-s</code>/<code>-f</code>).</p>
<ul>
<li><b>GPS L1 C/A</b> — гражданский сигнал GPS.</li>
<li><b>GPS L1C</b> — современный GPS-сигнал (TMBOC + BOC(1,1)), данные L1Cd
(нули или CNAV-2).</li>
<li><b>Galileo E1</b> — сигнал Galileo (CBOC).</li>
<li><b>QZSS L1 C/A</b>, <b>SBAS L1 C/A</b> — японский и SBAS
(геостационарный) сигналы.</li>
<li><b>BeiDou B1I</b> — китайский сигнал (1561.098 МГц). Вместе с L1/E1 он
автоматически образует <b>единый поток</b> (<code>all</code>, центр/fs
вычисляются); один он даёт узкую сессию <code>b1i</code>.</li>
</ul>
<p>Источник RINEX тоже следует за системами: если отмечены Galileo/QZSS/BeiDou,
нужен мультисистемный merged-файл, иначе достаточно GPS-файла.</p>

<h3>Консоль, UHD и «U»/«O»</h3>
<p>UHD по умолчанию печатает служебные строки в stderr (в PowerShell это
оранжевый <code>NativeCommandError</code>); программа ставит
<code>UHD_LOG_LEVEL=fatal</code> и перехватывает нативный stderr. Одиночные
маркеры <code>U</code>/<code>O</code> (underflow/overflow B210) библиотека UHD
пишет в обход уровня логов — они отфильтровываются, а настоящие ошибки
попадают в журнал/на stdout.</p>

<h3>Повторное использование готового IQ-файла</h3>
<ul>
<li>Отметьте <b>«Использовать готовый IQ-файл»</b> и укажите файл — генерация
полностью пропускается (RINEX и синтез не нужны).</li>
<li>Если рядом есть side-car <code>&lt;файл&gt;.json</code>, из него берутся
<code>sample_rate</code>, <code>format</code>, <code>center_freq</code>;
если нет — используются значения из настроек (в журнал пишется
предупреждение).</li>
<li>При включённой передаче на B210 файл проигрывается (с зацикливанием —
возврат к началу файла на диске); при выключенной он только проверяется и
описывается.</li>
</ul>

<h3>Поля режима и памяти</h3>
<ul>
<li><b>Длительность</b> — сколько секунд сигнала сгенерировать/записать (0 —
бесконечно, до «Стоп»). При передаче на B210 вместе с зацикливанием
длительность не ограничивает эфир: сегмент передаётся непрерывно до «Стоп»
(для приёмника длительность по-прежнему задаёт длину файла и не-loop передачу).</li>
<li><b>Зацикливать сегмент (RAM)</b> — сгенерировать сегмент в оперативную
память и повторять его. Для файла сегмент пишется один раз; для передачи на
B210 (TX) это <b>непрерывная передача до кнопки «Стоп»</b> (длительность
игнорируется, пока включено зацикливание).</li>
<li><b>Длина сегмента</b> — длина зацикливаемого сегмента (0 — автоматически:
<code>min(длительность, что влезает в бюджет RAM)</code>). Если заданная
длительность помещается в RAM, генерируется ровно она (один сегмент) даже
при включённом зацикливании. Урезается (с предупреждением и округлением до
кратного 30/18 с, т.е. 90 с) только когда не влезает. В журнале указывается,
источник зацикливания — RAM или диск.</li>
<li><b>Бюджет RAM</b> — сколько оперативной памяти разрешено занять сегменту
(0 — 60 % доступной).</li>
<li><b>по умолчанию</b> у каждой группы сбрасывает только её настройки;
логин/пароль/токен CDDIS при этом не затрагиваются.</li>
</ul>

<h3>RINEX-эфемериды</h3>
<p>Имена файлов: RINEX&nbsp;2 — <code>brdc{doy}0.{yy}n</code>, где
<code>{yy}</code> — <b>двузначный ГОД</b>, например
<code>brdc2680.26n</code> (268-й день 2026 года); RINEX&nbsp;3 —
<code>BRDC00IGS_R_ГГГГДДД0000_01D_MN.rnx</code>. <b>Важно:</b>
<code>.26n</code> — только GPS, <code>.26g</code> — только GLONASS,
merged <code>BRDC00IGS_R_…rnx</code> — мульти-GNSS (G/E/J/C). Для
Galileo/BeiDou нужен merged-файл. Архивы <code>.gz</code> распаковываются
автоматически (по магии <code>1f 8b</code> или расширению). Архив CDDIS:
<code>…/daily/{год}/brdc/</code> (без папок по дням).</p>

<h3>RINEX: повторное использование</h3>
<ul>
<li>Заданный вручную файл используется всегда, без скачивания; в журнал
пишется, какие системы (G/E/J/C) он реально содержит.</li>
<li>Файл за прошлую дату берётся из кэша как есть.</li>
<li>Файл за сегодня моложе 1 часа берётся из кэша; если старше — будет
предложено обновить.</li>
<li>Если кэша нет — файл скачивается автоматически (сначала merged
мульти-GNSS).</li>
</ul>

<h3>Время старта</h3>
<p>Дата старта берётся из <b>номера RINEX-файла</b> (его DOY), например
<code>BRDC00IGS_R_20262670000_01D_MN.rnx</code> (DOY&nbsp;267) даёт
<b>2026/09/24</b>; редактируется только <b>время</b> (ЧЧ:ММ:СС), дата —
заблокирована. Окно покрытия — фактические эпохи RINEX
<code>[начало, конец]</code>; старт вне окна отклоняется с понятным сообщением,
а галочка <b>«Сейчас (UTC)»</b> снимается/блокируется, если файл не покрывает
текущее время. При автоскачивании проверка выполняется после загрузки файла.</p>

<h3>Списки и прокрутка</h3>
<p>Колесо мыши над списком (частота, центр, источник, формат и т.п.) больше
не меняет значение — прокрутка действует только при открытом выпадающем
списке. Первый пункт списков частоты/центра — «по умолчанию».</p>

<h3>Мониторинг RX / холодный старт</h3>
<p>Для дуплексного контроля TX и RX должны быть на <b>разных</b> каналах
(например TX=0, RX=1) и разнесены по портам (TX/RX и RX2). «Холодный старт»
отправляет приёмнику u-blox команду UBX-CFG-RST (очистка BBR).</p>
"""


class Bridge(QtCore.QObject):
    """Marshals runner callbacks onto the GUI thread."""

    log = QtCore.pyqtSignal(str)
    #: Native UHD stderr text, forwarded from the fd-2 reader thread.
    native = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(float, float, float, float)
    #: TX phase label for the single bar: ``(kind, frac, loops, sim_s, cyclic)``.
    phase = QtCore.pyqtSignal(str, float, int, float, bool)
    channels = QtCore.pyqtSignal(list)
    finished = QtCore.pyqtSignal(object)
    spectrum = QtCore.pyqtSignal(str, object, float)
    #: TX/RX level: ``(label, peak_dbfs, rms_dbfs, clips)``.
    level = QtCore.pyqtSignal(str, float, float, int)
    askNav = QtCore.pyqtSignal(str, float)


def _row(layout: QtWidgets.QGridLayout, r: int, label: str,
         widget: QtWidgets.QWidget) -> None:
    layout.addWidget(QtWidgets.QLabel(label), r, 0)
    layout.addWidget(widget, r, 1)


class _CddisCredsDialog(QtWidgets.QDialog):
    """Модальный диалог ввода учётных данных CDDIS (Earthdata)."""

    def __init__(self, parent: QtWidgets.QWidget | None = None,
                 user: str = "", token: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle("Учётные данные CDDIS (Earthdata)")
        form = QtWidgets.QFormLayout(self)
        self.ed_user = QtWidgets.QLineEdit(user)
        self.ed_pass = QtWidgets.QLineEdit()
        self.ed_pass.setEchoMode(QtWidgets.QLineEdit.Password)
        self.ed_token = QtWidgets.QLineEdit(token)
        form.addRow("Логин Earthdata", self.ed_user)
        form.addRow("Пароль", self.ed_pass)
        form.addRow("Bearer-токен (необязательно)", self.ed_token)
        note = QtWidgets.QLabel(
            "Нужен логин/пароль Earthdata <b>или</b> bearer-токен. "
            "Данные сохранятся в настройках этой сессии.")
        note.setWordWrap(True)
        form.addRow(note)
        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

    def values(self) -> tuple[str, str, str]:
        return (self.ed_user.text().strip(), self.ed_pass.text(),
                self.ed_token.text().strip())


class _CddisDownloadThread(QtCore.QThread):
    """Фоновое скачивание файла CDDIS, чтобы не морозить интерфейс."""

    message = QtCore.pyqtSignal(str)
    done = QtCore.pyqtSignal(object, object)  # (path | None, error | None)

    def __init__(self, year: int, month: int, day: int, username: str,
                 password: str, token: str,
                 parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self.year = int(year)
        self.month = int(month)
        self.day = int(day)
        self.username = username
        self.password = password
        self.token = token

    def run(self) -> None:  # noqa: D102
        try:
            from .rinexfetch import download_cddis, doy_from_date
            doy = doy_from_date(self.year, self.month, self.day)
            yy = self.year % 100
            self.message.emit(
                f"CDDIS: {self.year:04d}-{self.month:02d}-{self.day:02d} -> "
                f"DOY {doy:03d}; merged BRDC00IGS_R_{self.year}{doy:03d}0000_"
                f"01D_MN.rnx.gz доступен только за прошедшие сутки, иначе "
                f"берём посистемные brdc{doy:03d}0.{yy:02d}{{n,g,l,c,j}}")
            path = download_cddis(
                self.year, doy, username=self.username, password=self.password,
                token=self.token, log=self.message.emit)
            self.done.emit(path, None)
        except Exception as exc:  # noqa: BLE001
            self.done.emit(None, str(exc))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(_APP_TITLE)
        self.resize(1180, 820)
        self.runner: SimulationRunner | None = None
        self.bridge = Bridge()
        self.bridge.log.connect(self._append_log)
        self.bridge.native.connect(self._append_native_log)
        self.bridge.progress.connect(self._on_progress)
        self.bridge.phase.connect(self._on_phase)
        self.bridge.channels.connect(self._on_channels)
        self.bridge.finished.connect(self._on_finished)
        self.bridge.spectrum.connect(self._on_spectrum)
        self.bridge.level.connect(self._on_level)
        self.bridge.askNav.connect(self._on_ask_nav)
        self._ask_event = threading.Event()
        self._ask_result = [False]
        self._updating_range = False
        #: True while a cyclic B210 transmission drives the single bar per pass.
        self._cyclic_tx = False
        #: Progress-widget state: ``_bar_switched`` is False while the
        #: preparation/generation bar is visible and True once the transmission
        #: bar replaced it; ``_pregen_seen`` records whether the runner already
        #: emitted a distinct preparation phase for this run.
        self._bar_switched = False
        self._pregen_seen = False
        self._phase_seen = False
        self._run_cfg: SimConfig | None = None
        self._truck_result: dict | None = None
        self._anim_idx = 0.0
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._anim_tick)
        self._nmea: ublox.NmeaReader | None = None
        self._cddis_thread: _CddisDownloadThread | None = None
        self._nmea_timer = QtCore.QTimer(self)
        self._nmea_timer.setInterval(500)
        self._nmea_timer.timeout.connect(self._update_ublox)
        self._build_ui()
        # Native UHD text (minus the U/O markers) goes to the GUI log, never to
        # the console.  run.py installs the fd-2 capture before importing UHD;
        # this just points it at the log widget.
        install_native_stderr_filter()
        set_native_stderr_sink(self.bridge.native.emit)

    @property
    def progress(self) -> QtWidgets.QProgressBar:
        """The currently visible bar (preparation/generation or transmission).

        Tests and the rest of the window use ``win.progress`` as a single handle;
        it resolves to the bar the user is actually looking at.
        """
        return self.progress_tx if self._bar_switched else self.progress_prep

    # ==================================================================
    # UI construction
    # ==================================================================
    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)

        # ---------------- left: settings (Базовые / Дополнительные) ----
        self.left_tabs = QtWidgets.QTabWidget()
        self.left_tabs.addTab(self._build_basic_tab(), "Базовые")
        self.left_tabs.addTab(self._build_advanced_tab(), "Дополнительные")
        root.addWidget(self.left_tabs, 3)

        # ---------------- right: tabs ----------------
        self.tabs = QtWidgets.QTabWidget()
        root.addWidget(self.tabs, 4)
        self.tabs.addTab(self._build_gen_tab(), "Генерация")
        self.tabs.addTab(self._build_map_tab(), "Карта и трек")
        self.tabs.addTab(self._build_spectrum_tab(), "Спектр")

        self._build_menu()
        self._update_channel_validity()
        self._update_start_range()

    # ------------------------------------------------------------------
    @staticmethod
    def _scroll(inner: QtWidgets.QWidget) -> QtWidgets.QScrollArea:
        area = QtWidgets.QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(inner)
        return area

    def _group_reset_btn(self, slot, tooltip: str | None = None):
        btn = QtWidgets.QPushButton("по умолчанию")
        btn.setMaximumWidth(140)
        btn.setToolTip(tooltip or "Сбросить только настройки этой группы "
                                  "(учётные данные CDDIS не затрагиваются)")
        btn.clicked.connect(slot)
        return btn

    def _guard_combo(self, combo: QtWidgets.QComboBox):
        """Install the wheel guard and keep a reference for direct testing."""
        guard = _WheelGuard(combo)
        combo.installEventFilter(guard)
        combo._wheel_guard = guard  # type: ignore[attr-defined]
        return guard

    # ------------------------------------------------------------------
    def _build_basic_tab(self) -> QtWidgets.QWidget:
        inner = QtWidgets.QWidget()
        left = QtWidgets.QVBoxLayout(inner)

        left.addWidget(self._build_nav_group())
        left.addWidget(self._build_signals_group())
        left.addWidget(self._build_tx_group())
        left.addWidget(self._build_output_group())
        left.addWidget(self._build_rf_group())
        # The «Сигналы» checkboxes are the single source of truth: every toggle
        # refreshes the read-only band/centre/fs and RINEX-source note.
        for cb in (self.cb_ca, self.cb_l1c, self.cb_gal, self.cb_qzss,
                   self.cb_sbas, self.cb_bds):
            cb.stateChanged.connect(self._refresh_derived)
        self._refresh_derived()
        left.addStretch(1)
        return self._scroll(inner)

    def _build_advanced_tab(self) -> QtWidgets.QWidget:
        inner = QtWidgets.QWidget()
        left = QtWidgets.QVBoxLayout(inner)
        left.addWidget(self._build_band_group())
        left.addWidget(self._build_cddis_group())
        left.addWidget(self._build_uhd_group())
        left.addWidget(self._build_monitor_group())
        left.addStretch(1)
        return self._scroll(inner)

    # ------------------------------------------------------------------
    def _build_nav_group(self) -> QtWidgets.QGroupBox:
        g_nav = QtWidgets.QGroupBox("Эфемериды и позиция")
        f = QtWidgets.QGridLayout(g_nav)
        self.ed_nav = QtWidgets.QLineEdit("")
        self.ed_nav.setPlaceholderText("пусто = скачать автоматически")
        btn_nav = QtWidgets.QPushButton("…")
        btn_nav.setFixedWidth(32)
        btn_nav.clicked.connect(self._browse_nav)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.ed_nav)
        row.addWidget(btn_nav)
        _row(f, 0, "RINEX navigation", self._wrap(row))
        self.cb_auto = QtWidgets.QCheckBox("Скачивать RINEX автоматически")
        self.cb_auto.setChecked(True)
        f.addWidget(self.cb_auto, 1, 0, 1, 2)

        # The RINEX source mode (multi-GNSS vs GPS-only) is derived from the
        # «Сигналы» checkboxes below and shown here read-only; there is no
        # separate «Режим эфемерид» choice any more.
        self.lbl_nav_mode = QtWidgets.QLabel()
        self.lbl_nav_mode.setTextFormat(QtCore.Qt.RichText)
        self.lbl_nav_mode.setWordWrap(True)
        self.lbl_nav_mode.setText(
            "Источник RINEX будет выбран автоматически по системам.")
        f.addWidget(self.lbl_nav_mode, 2, 0, 1, 2)

        self.ed_lat = QtWidgets.QDoubleSpinBox(); self.ed_lat.setRange(-90, 90)
        self.ed_lat.setDecimals(6); self.ed_lat.setValue(35.681298)
        self.ed_lon = QtWidgets.QDoubleSpinBox(); self.ed_lon.setRange(-180, 180)
        self.ed_lon.setDecimals(6); self.ed_lon.setValue(139.766247)
        self.ed_hgt = QtWidgets.QDoubleSpinBox(); self.ed_hgt.setRange(-1000, 20000)
        self.ed_hgt.setDecimals(1); self.ed_hgt.setValue(10.0)
        _row(f, 3, "Широта, °", self.ed_lat)
        _row(f, 4, "Долгота, °", self.ed_lon)
        _row(f, 5, "Высота, м", self.ed_hgt)
        self.ed_motion = QtWidgets.QLineEdit("")
        self.ed_motion.setPlaceholderText("пусто = статичная позиция")
        btn_mot = QtWidgets.QPushButton("…")
        btn_mot.setFixedWidth(32)
        btn_mot.clicked.connect(
            lambda: self._browse(self.ed_motion, "Motion",
                                 "Motion (*.csv *.txt);;Все файлы (*)",
                                 "Все файлы (*)"))
        rmot = QtWidgets.QHBoxLayout(); rmot.addWidget(self.ed_motion); rmot.addWidget(btn_mot)
        _row(f, 6, "Файл движения", self._wrap(rmot))

        # Start time: only the TIME is editable — the DATE is resolved from
        # the RINEX (read-only label) and from «Сейчас»/merged fallback.
        self.chk_now = QtWidgets.QCheckBox("Сейчас (UTC)")
        self.chk_now.setChecked(True)
        self.ed_start = _DateLockedDateTimeEdit()
        self.ed_start.setDisplayFormat("yyyy/MM/dd HH:mm:ss")
        self.ed_start.setCalendarPopup(False)
        self.ed_start.setTimeSpec(QtCore.Qt.UTC)
        self.ed_start.setDateTime(QtCore.QDateTime.currentDateTimeUtc())
        self.ed_start.setEnabled(False)
        self.chk_now.toggled.connect(
            lambda on: self.ed_start.setEnabled(not on))
        self.lbl_start_date = QtWidgets.QLabel()
        self.lbl_start_date.setTextFormat(QtCore.Qt.RichText)
        self.lbl_start_date.setWordWrap(True)
        start_row = QtWidgets.QHBoxLayout()
        start_row.addWidget(QtWidgets.QLabel("Время (UTC):"))
        start_row.addWidget(self.ed_start)
        start_row.addWidget(self.chk_now)
        _row(f, 7, "Время старта", self._wrap(start_row))
        f.addWidget(self.lbl_start_date, 8, 0, 1, 2)
        self.lbl_start_cover = QtWidgets.QLabel(
            "Покрытие эфемерид: неизвестно (будет автоскачивание)")
        self.lbl_start_cover.setWordWrap(True)
        f.addWidget(self.lbl_start_cover, 9, 0, 1, 2)
        self.sp_dur = QtWidgets.QDoubleSpinBox(); self.sp_dur.setRange(0, 86400)
        self.sp_dur.setValue(120.0); self.sp_dur.setSuffix(" с")
        self.sp_dur.setToolTip(
            "Сколько секунд записать/сгенерировать (0=∞, до «Стоп»). При TX с "
            "включённым зацикливанием эфир идёт непрерывно до «Стоп», а это "
            "поле задаёт длину файла / не-loop передачу.")
        _row(f, 10, "Длительность (0=∞)", self.sp_dur)
        f.addWidget(self._group_reset_btn(
            self._reset_nav_group,
            "Сбросить параметры эфемерид/позиции (CDDIS не затрагивается)"),
            11, 0, 1, 2)

        self._refresh_start_date_label()
        self.ed_start.dateTimeChanged.connect(
            lambda *_: self._refresh_start_date_label())

        # Coverage binding (B4 / issue 3): re-evaluate whenever the RINEX path
        # changes — browsing, typing, a CDDIS download or the «now» checkbox.
        self.ed_nav.textChanged.connect(self._on_nav_changed)
        self.ed_nav.editingFinished.connect(self._update_start_range)
        self.chk_now.toggled.connect(self._on_now_toggled)
        return g_nav

    # ------------------------------------------------------------------
    def _reset_nav_group(self) -> None:
        self.ed_nav.setText("")
        self.cb_auto.setChecked(True)
        self.ed_lat.setValue(35.681298)
        self.ed_lon.setValue(139.766247)
        self.ed_hgt.setValue(10.0)
        self.ed_motion.setText("")
        self.chk_now.setChecked(True)
        self.ed_start.setDateTime(QtCore.QDateTime.currentDateTimeUtc())
        self.sp_dur.setValue(120.0)
        self._update_start_range()

    # ------------------------------------------------------------------
    def _on_now_toggled(self, on: bool) -> None:
        self._refresh_start_date_label()
        if not on:
            self._update_start_range()

    def _refresh_start_date_label(self) -> None:
        """Show the resolved start DATE read-only (only the time is editable).

        The date comes from the resolved RINEX / merged fallback / «сейчас»;
        the user cannot free-edit it, so it is displayed as a label instead of
        a calendar section.
        """
        if not hasattr(self, "lbl_start_date"):
            return
        date_txt = "--"
        try:
            from .config import parse_start_time
            from .gpstime import gps2date
            y, m, d, _hh, _mm, _ss = gps2date(
                parse_start_time(self._start_text()))
            date_txt = f"{y:04d}/{m:02d}/{d:02d}"
        except Exception:  # noqa: BLE001
            pass
        src = ("сейчас, UTC" if self.chk_now.isChecked()
               else "из эфемерид/RINEX")
        self.lbl_start_date.setText(
            f"Дата старта (только чтение, {src}): <b>{date_txt}</b>")

    @staticmethod
    def _gps_to_qdt(g, offset_s: float = 0.0) -> QtCore.QDateTime:
        """Convert a :class:`GpsTime` (+offset) to a UTC ``QDateTime``.

        The time spec must be UTC: ``QDateTimeEdit.setDateTimeRange`` converts a
        local-spec datetime to the widget's spec, which shifted the coverage
        window by the machine's UTC offset (making the start time look
        uneditable/uncovered).
        """
        from .gpstime import gps2date, inc_gps_time
        y, mo, d, hh, mi, ss = gps2date(inc_gps_time(g, float(offset_s)))
        dt = QtCore.QDateTime(y, mo, d, hh, mi, int(ss))
        dt.setTimeSpec(QtCore.Qt.UTC)
        return dt

    def _nav_by_sv(self, path: str):
        """Parse (and cache) a RINEX file into ``by_sv``; ``None`` on failure."""
        if not path or not os.path.exists(path):
            return None
        try:
            key = (path, os.path.getmtime(path))
            cached = getattr(self, "_nav_span_cache", None)
            if cached and cached[0] == key:
                return cached[1]
            from .rinex import parse_nav_file
            by_sv, _iono = parse_nav_file(path)
            self._nav_span_cache = (key, by_sv)
            return by_sv
        except Exception:  # noqa: BLE001 - разбор покажет раннер
            return None

    def _coverage_by_sv(self, start):
        """Return ``by_sv`` for the RINEX the runner would use, or ``None``."""
        text = (self.ed_nav.text() or "").strip()
        parts = [p.strip() for p in text.split(";") if p.strip()]
        if parts and all(os.path.exists(p) for p in parts):
            if len(parts) == 1:
                by_sv = self._nav_by_sv(parts[0])
            else:
                by_sv = {}
                try:
                    from .rinex import parse_nav_file
                    by_sv, _ = parse_nav_file(parts)
                except Exception:  # noqa: BLE001
                    by_sv = {}
            if by_sv:
                return by_sv
        try:
            from .gpstime import gps2date
            from .rinexfetch import cached_nav_for_date, doy_from_date
            y, m, d, _h, _mi, _s = gps2date(start)
            path = cached_nav_for_date(y, doy_from_date(y, m, d))
            if path:
                return self._nav_by_sv(path)
        except Exception:  # noqa: BLE001
            pass
        return None

    def _rinex_name_date(self):
        """UTC date encoded in the selected RINEX file name(s), or ``None``.

        ``ed_nav`` may hold a single path or a ``;``-joined per-system set; the
        first name that parses wins.  Used to pin the read-only start DATE to
        the file's own day (issue 3) instead of the earliest record epoch.
        """
        text = (self.ed_nav.text() or "").strip()
        parts = [p.strip() for p in text.split(";") if p.strip()]
        if not parts:
            return None
        try:
            from .rinexfetch import date_from_rinex_name
        except Exception:  # noqa: BLE001
            return None
        for part in parts:
            found = date_from_rinex_name(part)
            if found is not None:
                return found
        return None

    def _update_start_range(self) -> None:
        """Bind the start-time widget to the cached ephemeris coverage (B4).

        When the selected RINEX does not cover «now» (for example an old
        ephemeris) the «Сейчас (UTC)» checkbox is disabled/unchecked and the
        start is moved to the RINEX start, so a run can never silently start
        outside the file.  The window is the actual RINEX epoch span
        (earliest…latest record epoch) and is always shown in the label.
        """
        if not hasattr(self, "ed_start") or not hasattr(self, "lbl_start_cover"):
            return
        if getattr(self, "_updating_range", False):
            return  # re-entrancy guard (chk_now.setChecked triggers us again)
        self._updating_range = True
        try:
            self._update_start_range_inner()
        finally:
            self._updating_range = False

    def _update_start_range_inner(self) -> None:
        from .config import parse_start_time
        from .gpstime import gps2date
        try:
            now = parse_start_time("now")
            start = parse_start_time(self._start_text())
        except Exception:  # noqa: BLE001
            return
        by_sv = self._coverage_by_sv(start)
        if not by_sv:
            self.lbl_start_cover.setText(
                "Покрытие эфемерид: неизвестно (будет автоскачивание)")
            if hasattr(self, "chk_now"):
                self.chk_now.setEnabled(True)
            self._refresh_start_date_label()
            return
        from .rinex import check_start_coverage, ephemeris_epoch_span
        span = ephemeris_epoch_span(by_sv)
        if span is None:
            self.lbl_start_cover.setText("Покрытие эфемерид: нет данных")
            if hasattr(self, "chk_now"):
                self.chk_now.setEnabled(True)
            self._refresh_start_date_label()
            return
        lo, hi = span
        # The window is the ACTUAL RINEX epoch span [earliest, latest], never
        # the old ``toe ± 6 h`` heuristic: the label and the widget must show
        # one consistent source of truth (the file), or a start can slip
        # outside the real ephemeris validity.
        lo_dt = self._gps_to_qdt(lo)
        hi_dt = self._gps_to_qdt(hi)
        if not (lo_dt.isValid() and hi_dt.isValid() and lo_dt <= hi_dt):
            self.lbl_start_cover.setText("Покрытие эфемерид: нет данных")
            self._refresh_start_date_label()
            return
        # A single-epoch file has a degenerate window: pin the widget to it so
        # the one valid start time is still selectable.
        self.ed_start.setDateTimeRange(lo_dt, hi_dt)
        window = (f"{lo_dt.toString('yyyy/MM/dd HH:mm')} … "
                  f"{hi_dt.toString('yyyy/MM/dd HH:mm')}")
        start_dt = self._gps_to_qdt(start)
        note = check_start_coverage(start, by_sv)
        now_note = check_start_coverage(now, by_sv)
        if now_note is not None:
            # The RINEX does not cover «now»: the checkbox cannot be used.
            if hasattr(self, "chk_now"):
                self.chk_now.setChecked(False)
                self.chk_now.setEnabled(False)
        elif hasattr(self, "chk_now"):
            self.chk_now.setEnabled(True)

        # Issue 3: the DATE comes from the RINEX file NUMBER (its day-of-year),
        # not from the earliest record epoch.  A merged/per-system file for DOY
        # 267 covers 2026-09-24 but its earliest ``toc`` can be 2026-09-23
        # 23:30; pinning the start to the previous day made a ZED-F9P reject
        # the ephemerides.  The [RINEX start, RINEX end] span is still kept for
        # validation, and the time-of-day stays editable.
        file_date = self._rinex_name_date()
        now_date = tuple(gps2date(now)[:3])
        moved_dt: QtCore.QDateTime | None = None
        if file_date is not None:
            if ((file_date.year, file_date.month, file_date.day) != now_date
                    and hasattr(self, "chk_now")):
                # A file from another day: «now» must not override its date.
                self.chk_now.setChecked(False)
            _, _, _, sh, sm, ss = gps2date(start)
            target = QtCore.QDateTime(file_date.year, file_date.month,
                                      file_date.day, sh, sm, int(ss))
            target.setTimeSpec(QtCore.Qt.UTC)
            if target < lo_dt:
                target = lo_dt
            elif target > hi_dt:
                target = hi_dt
            moved_dt = target
            self.ed_start.setDateTime(target)
        elif note is not None:
            # No RINEX name date (e.g. a synthetic/hand-made file): snap the
            # out-of-range start to the nearest span edge (previous behaviour).
            moved = lo if start_dt < lo_dt else hi
            self.ed_start.setDateTime(self._gps_to_qdt(moved))
        text = "Покрытие эфемерид (RINEX): " + window
        if file_date is not None:
            text += (". Дата старта из имени файла: "
                     f"{file_date.year:04d}/{file_date.month:02d}/"
                     f"{file_date.day:02d}")
        if moved_dt is not None:
            text += ("; старт на "
                     f"{moved_dt.toString('yyyy/MM/dd HH:mm')}")
        if now_note is not None:
            text += " («Сейчас (UTC)» недоступно: RINEX не покрывает now)"
        elif note is not None and file_date is None:
            text += ("; старт вне диапазона — переведён на ближайшую границу")
        self.lbl_start_cover.setText(text)
        # The date may have been clamped into the coverage range.
        self._refresh_start_date_label()

    def _on_nav_changed(self, _text: str = "") -> None:
        """Refresh the coverage window when the RINEX path changes (issue 3).

        Called on ``ed_nav.textChanged`` (browse button, typing, CDDIS
        completion) so the read-only date label and the editable start time
        immediately reflect the file's actual epoch span.  While the user is
        still typing a path that does not exist yet, only the date label is
        refreshed (no RINEX parse); ``_update_start_range`` already guards
        against recursion.
        """
        text = (self.ed_nav.text() or "").strip()
        parts = [p.strip() for p in text.split(";") if p.strip()]
        if parts and not all(os.path.exists(p) for p in parts):
            self._refresh_start_date_label()
            return
        self._update_start_range()

    def _browse_nav(self) -> None:
        """Browse for a RINEX file; always refresh the coverage/date (issue 3)."""
        before = self.ed_nav.text()
        self._browse(self.ed_nav, "RINEX navigation",
                     _RINEX_FILTER, _RINEX_FILTER_ALL)
        if self.ed_nav.text() == before:
            # Re-picking the same path emits no ``textChanged``; refresh anyway.
            self._on_nav_changed()

    # ------------------------------------------------------------------
    def _set_nav_mode_note(self, moved=None) -> None:
        """Read-only note describing the derived RINEX source mode.

        The mode is no longer a user choice: it follows the «Сигналы»
        checkboxes (multi-GNSS if Galileo/QZSS/BeiDou are enabled, otherwise
        GPS-only).  Optionally append the auto-moved start date.
        """
        if not hasattr(self, "lbl_nav_mode"):
            return
        multi = (getattr(self, "cb_gal", None) is not None
                 and (self.cb_gal.isChecked() or self.cb_qzss.isChecked()
                      or self.cb_bds.isChecked()))
        if multi:
            text = ("Источник RINEX: <b>мультисистемный merged (G/E/J/C)</b> — "
                    "выбраны Galileo/QZSS/BeiDou. Такой файл публикуется "
                    "только за <b>прошедшие</b> сутки; за сегодня берутся "
                    "посистемные brdc{DOY}0.{yy}{n,g,l,c,j}.")
        else:
            text = ("Источник RINEX: <b>GPS-only</b> — выбраны только GPS "
                    "L1/SBAS. Достаточно файла brdc{DOY}0.{yy}n; "
                    "мультисистемный merged не требуется.")
        if moved is not None:
            text += (" Дата старта переведена на "
                     f"<b>{moved.year:04d}/{moved.month:02d}/{moved.day:02d}</b>.")
        self.lbl_nav_mode.setText(text)

    def _latest_multignss_date(self, now=None, cache_dir: str | None = None):
        """Most recent UTC date (< today) with a merged multi-GNSS RINEX.

        A cached merged file (``BRDC00IGS_R_…_MN.rnx``) is preferred, scanning
        backwards up to a week; when none is cached the default is yesterday
        (UTC today − 1).  Never downloads (the GUI thread must stay offline).
        """
        try:
            from .rinexfetch import latest_merged_date
        except Exception:  # noqa: BLE001
            from datetime import datetime, timedelta, timezone
            now_dt = now or datetime.now(timezone.utc)
            if now_dt.tzinfo is None:
                now_dt = now_dt.replace(tzinfo=timezone.utc)
            return now_dt.astimezone(timezone.utc).date() - timedelta(days=1)
        return latest_merged_date(now=now, cache_dir=cache_dir)

    def _apply_multignss_date(self) -> None:
        """Move ``ed_start`` to the latest day with a merged multi-GNSS file."""
        if not hasattr(self, "ed_start"):
            return
        latest = self._latest_multignss_date()
        cur = self.ed_start.dateTime()
        hh, mi, ss = 0, 0, 0
        if cur.isValid():
            t = cur.time()
            hh, mi, ss = t.hour(), t.minute(), t.second()
        moved = QtCore.QDateTime(latest.year, latest.month, latest.day,
                                 hh, mi, ss)
        if not moved.isValid():
            moved = QtCore.QDateTime(latest.year, latest.month, latest.day,
                                     0, 0, 0)
        self.ed_start.setDateTime(moved)
        self._set_nav_mode_note(latest)
        self._update_start_range()
        self._refresh_start_date_label()

    def _correct_multignss_start(self, cfg: SimConfig) -> bool:
        """Move a "today" start to the latest merged-RINEX day (no network).

        Called from ``_start``/``_generate_iq`` after the CDDIS-credentials
        step.  Only the cache and the yesterday fallback are consulted, so the
        GUI thread never blocks on the network.  Always returns ``True`` (the
        correction is applied, not aborted).
        """
        if getattr(cfg, "nav_mode", "auto") != "merged":
            return True
        nav = str(getattr(cfg, "nav_file", "") or "").strip()
        parts = [p.strip() for p in nav.split(";") if p.strip()]
        if parts and all(os.path.exists(p) for p in parts):
            return True  # explicit user files: respect the selection

        from datetime import datetime, timezone
        now_utc = datetime.now(timezone.utc)
        from .config import parse_start_time
        from .gpstime import gps2date
        from .rinexfetch import has_merged_for_start, merged_start_for_date
        try:
            start = parse_start_time(self._start_text())
        except Exception:  # noqa: BLE001
            return True
        if has_merged_for_start(start):
            return True  # the chosen date already has a merged file
        y, m, d, _hh, _mm, _ss = gps2date(start)
        latest = self._latest_multignss_date(now=now_utc)
        if (latest.year, latest.month, latest.day) == (y, m, d):
            return True

        moved_gps = merged_start_for_date(start, latest)
        my, mm, md, mhh, mmi, mss = gps2date(moved_gps)
        self.chk_now.setChecked(False)
        moved = QtCore.QDateTime(my, mm, md, mhh, mmi, int(mss))
        if not moved.isValid():
            moved = QtCore.QDateTime(latest.year, latest.month, latest.day,
                                     0, 0, 0)
        self.ed_start.setDateTime(moved)
        self._set_nav_mode_note(latest)
        self._update_start_range()
        self._append_log(
            "Мультисистемный merged RINEX (G/E/J/C) публикуется только за "
            "прошедшие сутки; время старта переведено на "
            f"{latest.year:04d}/{latest.month:02d}/{latest.day:02d}.")
        cfg.start_text = self._start_text()
        return True

    # ------------------------------------------------------------------
    def _build_signals_group(self) -> QtWidgets.QGroupBox:
        g_sig = QtWidgets.QGroupBox("Сигналы")
        f2 = QtWidgets.QGridLayout(g_sig)
        self.cb_ca = QtWidgets.QCheckBox("GPS L1 C/A")
        self.cb_ca.setChecked(True)
        self.cb_l1c = QtWidgets.QCheckBox("GPS L1C (TMBOC + BOC(1,1))")
        self.cb_l1c.setChecked(True)
        self.cb_gal = QtWidgets.QCheckBox("Galileo E1 (CBOC)")
        self.cb_gal.setChecked(True)
        self.cb_qzss = QtWidgets.QCheckBox("QZSS L1 C/A")
        self.cb_qzss.setChecked(True)
        self.cb_sbas = QtWidgets.QCheckBox("SBAS L1 C/A")
        self.cb_sbas.setChecked(True)
        self.cb_bds = QtWidgets.QCheckBox("BeiDou B1I (1561.098 МГц)")
        self.cb_bds.setChecked(False)  # off by default (narrow L1 session)
        _sig_tip = ("Системы определяют диапазон (l1/b1i/all), центр/fs и "
                    "источник RINEX автоматически. Подробнее — в «Справке».")
        for i, w in enumerate((self.cb_ca, self.cb_l1c, self.cb_gal,
                               self.cb_qzss, self.cb_sbas, self.cb_bds)):
            w.setToolTip(_sig_tip)
            f2.addWidget(w, i, 0, 1, 2)
        self.cmb_l1c_data = QtWidgets.QComboBox()
        self.cmb_l1c_data.addItems(["zeros", "cnav2"])
        self._guard_combo(self.cmb_l1c_data)
        _row(f2, 6, "Данные L1Cd", self.cmb_l1c_data)
        self.cb_iono = QtWidgets.QCheckBox("Ионосферная задержка (Клобучар)")
        self.cb_iono.setChecked(True)
        f2.addWidget(self.cb_iono, 7, 0, 1, 2)
        self.sp_el = QtWidgets.QDoubleSpinBox(); self.sp_el.setRange(0, 90)
        self.sp_el.setValue(5.0); self.sp_el.setSuffix(" °")
        _row(f2, 8, "Маска элевации", self.sp_el)
        # Amplitude: auto by default (derived from the number of channels /
        # summed per-channel amplitudes so a multi-system scene has headroom);
        # unchecking «Авто» exposes the historic explicit single factor.
        self.sp_amp = QtWidgets.QDoubleSpinBox(); self.sp_amp.setRange(0.001, 0.9)
        self.sp_amp.setSingleStep(0.05); self.sp_amp.setValue(0.15)
        self.sp_amp.setEnabled(False)
        self.cb_amp_auto = QtWidgets.QCheckBox("Авто")
        self.cb_amp_auto.setChecked(True)
        self.cb_amp_auto.setToolTip(
            "Автоматический единый масштаб по числу каналов/сумме амплитуд: "
            "многосистемная сцена получает ~0.06–0.10 (с запасом до клиппинга), "
            "GPS-only L1 C/A остаётся на прежнем уровне. Снимите галочку, чтобы "
            "задать один явный множитель (как раньше).")
        self.cb_amp_auto.toggled.connect(
            lambda on: self.sp_amp.setEnabled(not on))
        amp_row = QtWidgets.QHBoxLayout()
        amp_row.addWidget(self.sp_amp)
        amp_row.addWidget(self.cb_amp_auto)
        _row(f2, 9, "Амплитуда", self._wrap(amp_row))
        self.cb_headroom = QtWidgets.QCheckBox("Ограничивать пик (anti-clip)")
        self.cb_headroom.setChecked(True)
        self.cb_headroom.setToolTip(
            "Автоматический запас по уровню: пик композитного сигнала "
            "приводится к целевому значению, чтобы ЦАП B210 не клиппировал. "
            "Для предгенерённого сегмента применяется один точный масштаб.")
        self.sp_headroom = QtWidgets.QDoubleSpinBox()
        self.sp_headroom.setRange(0.1, 0.95)
        self.sp_headroom.setSingleStep(0.05)
        self.sp_headroom.setValue(0.7)
        self.sp_headroom.setToolTip("Целевой пик (доля полной шкалы), по умолч. 0.7")
        self.cb_headroom.toggled.connect(self.sp_headroom.setEnabled)
        hr_row = QtWidgets.QHBoxLayout()
        hr_row.addWidget(self.cb_headroom)
        hr_row.addWidget(self.sp_headroom)
        hr_row.addStretch(1)
        _row(f2, 10, "Запас уровня", self._wrap(hr_row))
        # Read-only derived session: band key, centre and fs.
        self.lbl_band = QtWidgets.QLabel()
        self.lbl_band.setTextFormat(QtCore.Qt.RichText)
        self.lbl_band.setWordWrap(True)
        f2.addWidget(self.lbl_band, 11, 0, 1, 2)
        f2.addWidget(self._group_reset_btn(self._reset_signals_group,
                                           "Сбросить только сигналы"),
                     12, 0, 1, 2)
        return g_sig

    def _reset_signals_group(self) -> None:
        for cb in (self.cb_ca, self.cb_l1c, self.cb_gal, self.cb_qzss,
                   self.cb_sbas, self.cb_iono):
            cb.setChecked(True)
        self.cb_bds.setChecked(False)  # default: B1I off (narrow L1 session)
        self.cmb_l1c_data.setCurrentText("zeros")
        self.sp_el.setValue(5.0)
        self.cb_amp_auto.setChecked(True)
        self.sp_amp.setValue(0.15)
        self.cb_headroom.setChecked(True)
        self.sp_headroom.setValue(0.7)
        if hasattr(self, "lbl_band"):
            self._refresh_derived()

    def _build_output_group(self) -> QtWidgets.QGroupBox:
        g_out = QtWidgets.QGroupBox("Выход IQ")
        f3 = QtWidgets.QGridLayout(g_out)
        self.ed_out = QtWidgets.QLineEdit(_DEFAULT_OUT)
        btn_out = QtWidgets.QPushButton("…")
        btn_out.setFixedWidth(32)
        btn_out.clicked.connect(self._save_browse)
        rout = QtWidgets.QHBoxLayout(); rout.addWidget(self.ed_out); rout.addWidget(btn_out)
        _row(f3, 0, "Файл", self._wrap(rout))
        self.cmb_fmt = QtWidgets.QComboBox()
        self.cmb_fmt.addItems(["cs16", "cf32", "cs8", "cs4"])
        self._guard_combo(self.cmb_fmt)
        self.cmb_fmt.currentTextChanged.connect(self._on_format_changed)
        _row(f3, 1, "Формат", self.cmb_fmt)
        self.sp_scale = QtWidgets.QDoubleSpinBox(); self.sp_scale.setRange(1, 1e7)
        self.sp_scale.setValue(10000.0)
        _row(f3, 2, "Масштаб в int", self.sp_scale)
        self.lbl_format_hint = QtWidgets.QLabel(
            "cs8/cs4 вдвое уменьшают поток (2 Б/отсчёт вместо 4); при "
            "30 Мвыб/с сегмент в RAM может упираться в бюджет — уменьшите "
            "длительность/сегмент или включите потоковую передачу.")
        self.lbl_format_hint.setWordWrap(True)
        f3.addWidget(self.lbl_format_hint, 3, 0, 1, 2)

        # --- reuse an existing IQ file (generation is skipped) ------------
        self.cb_iq_in = QtWidgets.QCheckBox("Использовать готовый IQ-файл")
        self.cb_iq_in.setToolTip(
            "Генерация пропускается: файл читается и, если включён B210, "
            "передаётся в эфир; иначе только проверяется и описывается.\n"
            "Если рядом есть <файл>.json — sample_rate/format/center_freq "
            "берутся из него; иначе используются значения из настроек.")
        f3.addWidget(self.cb_iq_in, 4, 0, 1, 2)
        self.ed_iq_in = QtWidgets.QLineEdit("")
        self.ed_iq_in.setPlaceholderText("путь к готовому IQ-файлу")
        self.ed_iq_in.setEnabled(False)
        btn_iq = QtWidgets.QPushButton("…")
        btn_iq.setFixedWidth(32)
        btn_iq.clicked.connect(
            lambda: self._browse(self.ed_iq_in, "Готовый IQ-файл",
                                 _IQ_INPUT_FILTER, "Все файлы (*)"))
        btn_iq.setEnabled(False)
        r_iq = QtWidgets.QHBoxLayout()
        r_iq.addWidget(self.ed_iq_in)
        r_iq.addWidget(btn_iq)
        self._btn_iq_in = btn_iq
        self.cb_iq_in.toggled.connect(self.ed_iq_in.setEnabled)
        self.cb_iq_in.toggled.connect(btn_iq.setEnabled)
        _row(f3, 5, "Готовый IQ", self._wrap(r_iq))
        # The verbose side-car note moved into the tooltip of the checkbox.
        self.cb_iq_ram = QtWidgets.QCheckBox(
            "Загрузить IQ в RAM (зацикливать из памяти)")
        self.cb_iq_ram.setToolTip(
            "Прочитать готовый IQ-файл целиком в оперативную память и "
            "зацикливать из неё (быстрее, но нужен объём RAM под весь файл). "
            "По умолчанию — потоковое чтение с диска.")
        self.cb_iq_ram.setChecked(False)
        self.cb_iq_ram.setEnabled(False)
        self.cb_iq_in.toggled.connect(self.cb_iq_ram.setEnabled)
        f3.addWidget(self.cb_iq_ram, 6, 0, 1, 2)

        self.cb_loop = QtWidgets.QCheckBox("Зацикливать сегмент (RAM)")
        self.cb_loop.setChecked(True)
        self.cb_loop.setToolTip(
            "Для файла: сегмент генерируется один раз и пишется один раз.\n"
            "Для передачи на B210 (TX): непрерывная передача сегмента до "
            "кнопки «Стоп» — «Длительность» в эфире не ограничивает.")
        f3.addWidget(self.cb_loop, 7, 0, 1, 2)
        self.sp_loop = QtWidgets.QDoubleSpinBox(); self.sp_loop.setRange(0, 86400)
        self.sp_loop.setValue(0.0); self.sp_loop.setSuffix(" с (0=авто)")
        self.sp_loop.setToolTip(
            "Длина зацикливаемого сегмента в RAM (0 — авто). При TX с "
            "зацикливанием передача идёт до «Стоп».")
        _row(f3, 8, "Длина сегмента", self.sp_loop)
        self.sp_mem = QtWidgets.QDoubleSpinBox(); self.sp_mem.setRange(0, 1024)
        self.sp_mem.setValue(0.0); self.sp_mem.setSuffix(" ГБ (0=60%)")
        _row(f3, 9, "Бюджет RAM", self.sp_mem)
        self.lbl_ram = QtWidgets.QLabel()
        f3.addWidget(self.lbl_ram, 10, 0, 1, 2)
        try:
            from .sysinfo import available_ram, human, total_ram
            self.lbl_ram.setText(
                f"RAM: {human(available_ram())} свободно из {human(total_ram())}")
        except Exception:
            pass
        f3.addWidget(self._group_reset_btn(
            self._reset_output_group,
            "Сбросить только выход IQ (готовый файл, формат, лууп)"),
            11, 0, 1, 2)
        return g_out

    def _on_format_changed(self, fmt: str) -> None:
        """Track the output file extension with the format combo (B5)."""
        path = (self.ed_out.text() or "").strip()
        if not path:
            return
        stem, ext = os.path.splitext(path)
        if ext.lower() in _IQ_EXTS:
            self.ed_out.setText(stem + "." + str(fmt).lower())

    def _reset_output_group(self) -> None:
        self.ed_out.setText(_DEFAULT_OUT)
        self.cmb_fmt.setCurrentText("cs16")
        self.sp_scale.setValue(10000.0)
        self.cb_iq_in.setChecked(False)
        self.ed_iq_in.setText("")
        self.cb_iq_ram.setChecked(False)
        self.cb_loop.setChecked(True)
        self.sp_loop.setValue(0.0)
        self.sp_mem.setValue(0.0)

    def _build_rf_group(self) -> QtWidgets.QGroupBox:
        g_rf = QtWidgets.QGroupBox("Радиотракт (вычисляется)")
        f4 = QtWidgets.QGridLayout(g_rf)
        self.lbl_fs = QtWidgets.QLabel()
        self.lbl_fs.setTextFormat(QtCore.Qt.RichText)
        _row(f4, 0, "Частота дискр.", self.lbl_fs)
        self.lbl_fc = QtWidgets.QLabel()
        self.lbl_fc.setTextFormat(QtCore.Qt.RichText)
        _row(f4, 1, "Центр. частота", self.lbl_fc)
        self.cmb_backend = QtWidgets.QComboBox()
        self.cmb_backend.addItems(["auto", "cpu", "cuda"])
        self._guard_combo(self.cmb_backend)
        _row(f4, 2, "Бэкенд синтеза", self.cmb_backend)
        f4.addWidget(self._group_reset_btn(self._reset_rf_group,
                                           "Сбросить только радиотракт"),
                     3, 0, 1, 2)
        return g_rf

    def _reset_rf_group(self) -> None:
        self.cmb_backend.setCurrentText("auto")

    def _build_band_group(self) -> QtWidgets.QGroupBox:
        g = QtWidgets.QGroupBox("BeiDou B1I")
        f = QtWidgets.QGridLayout(g)
        self.cmb_b1i_data = QtWidgets.QComboBox()
        self.cmb_b1i_data.addItems(["d1", "placeholder"])
        self.cmb_b1i_data.setToolTip(
            "Данные BeiDou B1I: d1 — реальное сообщение D1 (NH20+BCH+эфемериды), "
            "placeholder — постоянный +1 (но со структурой NH20).")
        self._guard_combo(self.cmb_b1i_data)
        _row(f, 0, "Данные B1I", self.cmb_b1i_data)

        self.cb_combine = QtWidgets.QCheckBox(
            "Объединять L1+B1I (широкая полоса ~25 Мвыб/с)")
        self.cb_combine.setChecked(False)
        self.cb_combine.setToolTip(
            "По умолчанию L1 и B1I несовместимы по полосе: при обеих группах "
            "выбирается узкая L1 (2.6 Мвыб/с, центр 1575.42 МГц), а B1I "
            "отключается. Включите эту галочку (или CLI --combine / --band all), "
            "чтобы получить один широкий поток ~1571.33 МГц / ~25 Мвыб/с с "
            "сохранением боковых лепестков BOC(6,1). Предгенерация такого "
            "сегмента в RAM занимает минуты и несколько ГБ.")
        f.addWidget(self.cb_combine, 1, 0, 1, 2)

        self.cb_auto_b1i = QtWidgets.QCheckBox("Авто-подбор fs/центра под B1I")
        self.cb_auto_b1i.setChecked(True)
        self.cb_auto_b1i.setToolTip(
            "Устаревшее: авто-подбор широкой полосы под B1I, если он не "
            "помещается в текущую полосу (действует только при ручных -s/-f). "
            "В обычном режиме полоса выводится из галочек «Сигналы».")
        f.addWidget(self.cb_auto_b1i, 2, 0, 1, 2)
        self.lbl_auto_b1i = QtWidgets.QLabel(
            "B1I (1561.098 МГц) не помещается в узкую полосу L1: без галочки "
            "«Объединять L1+B1I» B1I отключается, fs остаётся 2.6 Мвыб/с. "
            "Для объединённого широкого потока включите эту галочку.")
        self.lbl_auto_b1i.setWordWrap(True)
        f.addWidget(self.lbl_auto_b1i, 3, 0, 1, 2)
        f.addWidget(self._group_reset_btn(
            self._reset_band_group,
            "Сбросить данные B1I и объединение"),
            4, 0, 1, 2)
        self.cb_combine.stateChanged.connect(self._on_combine_toggled)
        self._refresh_derived()
        return g

    # ------------------------------------------------------------------
    def _on_combine_toggled(self, on: bool) -> None:
        """Sync the Advanced «Объединять L1+B1I» opt-in with Basic «B1I».

        The Advanced control only makes sense when BeiDou B1I is selected.  When
        it is ticked, B1I is selected in the Basic tab too, so the two tabs can
        never disagree (the other direction — B1I off collapses the opt-in — is
        handled in :meth:`_refresh_derived`).
        """
        if on and not self.cb_bds.isChecked():
            self.cb_bds.setChecked(True)  # triggers _refresh_derived
        self._refresh_derived()

    def _refresh_derived(self, *_args) -> None:
        """Recompute the read-only band/centre/fs and RINEX note.

        The «Сигналы» checkboxes are the single source of truth for the band;
        the only Advanced state this changes is collapsing the «Объединять
        L1+B1I» opt-in (and disabling the B1I-only controls) when B1I is off, so
        the Basic and Advanced tabs stay consistent.
        """
        if not hasattr(self, "lbl_band") or not hasattr(self, "lbl_fs"):
            return
        # Cross-tab B1I sync (bidirectional invariant):
        #   «Объединять L1+B1I» checked  ->  B1I must be selected in Basic.
        #   B1I unchecked in Basic        ->  collapse the Advanced opt-in.
        if (hasattr(self, "cb_combine") and self.cb_combine.isChecked()
                and not self.cb_bds.isChecked()):
            self.cb_combine.blockSignals(True)
            self.cb_combine.setChecked(False)
            self.cb_combine.blockSignals(False)
        combine = (self.cb_combine.isChecked()
                   if hasattr(self, "cb_combine") else False)
        enables = {
            "enable_ca": self.cb_ca.isChecked(),
            "enable_l1c": self.cb_l1c.isChecked(),
            "enable_galileo": self.cb_gal.isChecked(),
            "enable_qzss": self.cb_qzss.isChecked(),
            "enable_sbas": self.cb_sbas.isChecked(),
            "enable_beidou": self.cb_bds.isChecked(),
        }
        # Enable the Advanced B1I controls only while B1I is selected; the
        # combine opt-in additionally needs an L1/E1 carrier to merge with.
        if hasattr(self, "cb_combine"):
            bds = enables["enable_beidou"]
            l1_family = any(enables[k] for k in (
                "enable_ca", "enable_l1c", "enable_galileo",
                "enable_qzss", "enable_sbas"))
            self.cb_combine.setEnabled(bool(bds and l1_family))
            if hasattr(self, "cmb_b1i_data"):
                self.cmb_b1i_data.setEnabled(bool(bds))
            if hasattr(self, "cb_auto_b1i"):
                self.cb_auto_b1i.setEnabled(bool(bds))
        band = derive_band_key(combine=combine, **enables)
        plan = derive_band_plan(combine=combine, **enables)
        if not any(enables.values()):
            self.lbl_band.setText(
                "<b>Системы не выбраны</b> — отметьте хотя бы один сигнал.")
            self.lbl_fs.setText("—")
            self.lbl_fc.setText("—")
        else:
            self.lbl_band.setText(
                f"<b>Диапазон (вычислен): {band}</b><br>{plan.describe()}")
            self.lbl_fs.setText(f"<b>{plan.fs / 1e6:g}</b> Мвыб/с")
            self.lbl_fc.setText(f"<b>{plan.center_freq / 1e6:.3f}</b> МГц")
        if hasattr(self, "lbl_auto_b1i"):
            if enables["enable_beidou"] and band == "l1":
                self.lbl_auto_b1i.setText(
                    "BeiDou B1I отключён: он не входит в узкую полосу L1 "
                    "(fs останется 2.6 Мвыб/с). Включите «Объединять L1+B1I», "
                    "чтобы передать L1 и B1I одним широким потоком ~25 Мвыб/с.")
            elif band == "all":
                self.lbl_auto_b1i.setText(
                    "Объединённый поток L1+B1I (~25 Мвыб/с): предгенерация "
                    "TX-сегмента в RAM занимает минуты и несколько ГБ.")
            else:
                self.lbl_auto_b1i.setText(
                    "B1I (1561.098 МГц) не помещается в узкую полосу L1: без "
                    "галочки «Объединять L1+B1I» B1I отключается, fs остаётся "
                    "2.6 Мвыб/с.")
        self._set_nav_mode_note()

    def _reset_band_group(self) -> None:
        if hasattr(self, "cb_combine"):
            self.cb_combine.setChecked(False)
        self.cb_auto_b1i.setChecked(True)
        self.cmb_b1i_data.setCurrentText("d1")
        self._refresh_derived()

    def _build_cddis_group(self) -> QtWidgets.QGroupBox:
        g = QtWidgets.QGroupBox("Источник эфемерид (CDDIS / BKG)")
        f = QtWidgets.QGridLayout(g)
        self.cmb_source = QtWidgets.QComboBox()
        self.cmb_source.addItems(["auto", "cddis", "bkg"])
        self._guard_combo(self.cmb_source)
        _row(f, 0, "Источник", self.cmb_source)
        self.ed_cddis_user = QtWidgets.QLineEdit("")
        self.ed_cddis_user.setPlaceholderText("логин Earthdata")
        _row(f, 1, "CDDIS логин", self.ed_cddis_user)
        self.ed_cddis_pass = QtWidgets.QLineEdit("")
        self.ed_cddis_pass.setEchoMode(QtWidgets.QLineEdit.Password)
        self.ed_cddis_pass.setPlaceholderText("пароль Earthdata")
        _row(f, 2, "CDDIS пароль", self.ed_cddis_pass)
        self.ed_cddis_token = QtWidgets.QLineEdit("")
        self.ed_cddis_token.setPlaceholderText("bearer-токен (необязательно)")
        _row(f, 3, "CDDIS токен", self.ed_cddis_token)
        self.ed_date = QtWidgets.QLineEdit("")
        self.ed_date.setPlaceholderText("YYYY/MM/DD")
        _row(f, 4, "Дата CDDIS", self.ed_date)
        self.btn_cddis = QtWidgets.QPushButton("Скачать за дату")
        self.btn_cddis.clicked.connect(self._download_cddis_date)
        f.addWidget(self.btn_cddis, 5, 0, 1, 2)
        self.lbl_rinex_kind = QtWidgets.QLabel(
            "Тип файла RINEX: <b>merged</b> "
            "BRDC00IGS_R_…rnx = мульти-GNSS (G/E/J/C, предпочтительно); "
            "<b>.26n</b> = только GPS; <b>.26g</b> = только GLONASS. "
            "Для Galileo/BeiDou выберите merged-файл; .gz распаковывается "
            "автоматически.")
        self.lbl_rinex_kind.setWordWrap(True)
        f.addWidget(self.lbl_rinex_kind, 6, 0, 1, 2)
        self.lbl_cddis_days = QtWidgets.QLabel(
            "<b>Важно:</b> мультисистемный файл "
            "<code>BRDC00IGS_R_…_MN.rnx</code> публикуется на CDDIS только за "
            "<b>прошедшие</b> сутки. Для текущей даты (UTC) merged обычно "
            "отсутствует, и программа берёт посистемные файлы "
            "<code>brdc{DOY}0.{yy}{n,g,l,c,j}</code> (GPS/ГЛОНАСС/Galileo/"
            "BeiDou/QZSS), какие уже выложены.")
        self.lbl_cddis_days.setWordWrap(True)
        f.addWidget(self.lbl_cddis_days, 7, 0, 1, 2)
        f.addWidget(self._group_reset_btn(
            self._reset_cddis_group,
            "Сбросить выбор источника/даты; логин, пароль и токен "
            "не затрагиваются"),
            8, 0, 1, 2)
        self.ed_date.setText(self._default_date_text())
        return g

    def _reset_cddis_group(self) -> None:
        # ВАЖНО: учётные данные (логин/пароль/токен) НЕ сбрасываются.
        self.cmb_source.setCurrentText("auto")
        self.ed_date.setText(self._default_date_text())

    def _build_tx_group(self) -> QtWidgets.QGroupBox:
        """B210 TX checkbox on the main tab (B6); wiring is unchanged."""
        g = QtWidgets.QGroupBox("Передача на B210")
        f = QtWidgets.QGridLayout(g)
        self.cb_tx = QtWidgets.QCheckBox(
            "Передавать на B210 (в эфир, в дополнение к IQ-файлу)")
        self.cb_tx.setChecked(False)
        self.cb_tx.setToolTip(
            "Снято: только IQ-файл. Отмечено: передавать в эфир через B210; "
            "если файл выхода тоже задан — и файл, и эфир. Параметры B210 — "
            "на вкладке «Дополнительные».")
        f.addWidget(self.cb_tx, 0, 0, 1, 2)
        return g

    def _build_uhd_group(self) -> QtWidgets.QGroupBox:
        g_tx = QtWidgets.QGroupBox("USRP B210 (параметры передачи)")
        f5 = QtWidgets.QGridLayout(g_tx)
        self.ed_args = QtWidgets.QLineEdit("type=b200")
        _row(f5, 0, "UHD args", self.ed_args)
        self.sp_ch = QtWidgets.QSpinBox(); self.sp_ch.setRange(0, 1)
        _row(f5, 1, "TX канал", self.sp_ch)
        self.sp_txg = QtWidgets.QDoubleSpinBox(); self.sp_txg.setRange(-90, 90)
        self.sp_txg.setValue(0.0); self.sp_txg.setSuffix(" дБ")
        _row(f5, 2, "TX усиление", self.sp_txg)
        self.ed_ant = QtWidgets.QLineEdit("TX/RX")
        _row(f5, 3, "Антенна", self.ed_ant)
        self.sp_bw = QtWidgets.QDoubleSpinBox(); self.sp_bw.setRange(0, 56e6)
        self.sp_bw.setDecimals(0); self.sp_bw.setValue(0.0)
        self.sp_bw.setSuffix(" Гц (0=авто)")
        _row(f5, 4, "Полоса TX", self.sp_bw)
        self.cmb_clk = QtWidgets.QComboBox()
        self.cmb_clk.addItems(["internal", "external", "gpsdo"])
        self._guard_combo(self.cmb_clk)
        _row(f5, 5, "Опорная частота", self.cmb_clk)
        btn_find = QtWidgets.QPushButton("Найти устройства")
        btn_find.clicked.connect(self._find_devices)
        f5.addWidget(btn_find, 6, 0, 1, 3)
        self.lbl_tx_info = QtWidgets.QLabel(
            "Фактические частота дискретизации и полоса читаются с B210 "
            "после запуска и пишутся в журнал (TX gain зажимается в диапазон "
            "устройства с предупреждением).")
        self.lbl_tx_info.setWordWrap(True)
        f5.addWidget(self.lbl_tx_info, 7, 0, 1, 3)
        f5.addWidget(self._group_reset_btn(self._reset_uhd_group,
                                           "Сбросить только параметры B210"),
                     8, 0, 1, 3)
        return g_tx

    def _reset_uhd_group(self) -> None:
        self.cb_tx.setChecked(False)
        self.ed_args.setText("type=b200")
        self.sp_ch.setValue(0)
        self.sp_txg.setValue(0.0)
        self.ed_ant.setText("TX/RX")
        self.sp_bw.setValue(0.0)
        self.cmb_clk.setCurrentText("internal")
        self._update_channel_validity()

    def _build_monitor_group(self) -> QtWidgets.QGroupBox:
        g_mon = QtWidgets.QGroupBox("Мониторинг RX / мощность")
        f6 = QtWidgets.QGridLayout(g_mon)
        self.cb_monitor = QtWidgets.QCheckBox(
            "Слушать RX на другом канале (контроль TX)")
        self.cb_monitor.setChecked(False)
        f6.addWidget(self.cb_monitor, 0, 0, 1, 2)
        self.sp_target = QtWidgets.QDoubleSpinBox(); self.sp_target.setRange(-120, 0)
        self.sp_target.setValue(-30.0); self.sp_target.setSuffix(" dBFS")
        _row(f6, 1, "Цель RX", self.sp_target)
        self.cb_tx_auto = QtWidgets.QCheckBox("Авторегулятор TX gain")
        self.cb_tx_auto.setChecked(True)
        f6.addWidget(self.cb_tx_auto, 2, 0, 1, 2)
        self.sp_rx_ch = QtWidgets.QSpinBox(); self.sp_rx_ch.setRange(0, 1)
        self.sp_rx_ch.setValue(1)
        _row(f6, 3, "RX канал", self.sp_rx_ch)
        self.ed_rx_ant = QtWidgets.QLineEdit("RX2")
        _row(f6, 4, "RX антенна", self.ed_rx_ant)
        self.sp_rx_gain = QtWidgets.QDoubleSpinBox(); self.sp_rx_gain.setRange(-30, 90)
        self.sp_rx_gain.setValue(30.0); self.sp_rx_gain.setSuffix(" дБ")
        _row(f6, 5, "RX усиление", self.sp_rx_gain)
        self.sp_rx_freq = QtWidgets.QDoubleSpinBox(); self.sp_rx_freq.setRange(0, 6e9)
        self.sp_rx_freq.setDecimals(0); self.sp_rx_freq.setValue(0.0)
        self.sp_rx_freq.setSuffix(" Гц (0 = как TX)")
        _row(f6, 6, "Частота RX", self.sp_rx_freq)
        self.cb_echo = QtWidgets.QCheckBox("Анализ эха (профиль задержек)")
        self.cb_echo.setChecked(False)
        f6.addWidget(self.cb_echo, 7, 0, 1, 2)
        self.cb_tx_check = QtWidgets.QCheckBox(
            "Контроль передачи: предупредить, если RX не видит сигнал")
        self.cb_tx_check.setChecked(False)
        self.cb_tx_check.setToolTip(
            "Один раз во время TX принять блок на RX-канале и проверить, "
            "что уровень поднялся над шумом (нужен RX-канал, отличный от TX)")
        f6.addWidget(self.cb_tx_check, 8, 0, 1, 2)
        self.lbl_mon_warn = QtWidgets.QLabel(
            "TX и RX каналы совпадают — выберите разные каналы")
        self.lbl_mon_warn.setStyleSheet("color: #b00020;")
        self.lbl_mon_warn.setWordWrap(True)
        self.lbl_mon_warn.setVisible(False)
        f6.addWidget(self.lbl_mon_warn, 9, 0, 1, 2)
        f6.addWidget(self._group_reset_btn(
            self._reset_monitor_group, "Сбросить только мониторинг RX"),
            10, 0, 1, 2)

        # Channel selection must keep TX != RX for duplex.
        self.sp_ch.valueChanged.connect(self._on_tx_channel_changed)
        self.sp_rx_ch.valueChanged.connect(self._update_channel_validity)
        self.cb_monitor.toggled.connect(self._update_channel_validity)
        self.cb_tx_check.toggled.connect(self._update_channel_validity)
        return g_mon

    def _reset_monitor_group(self) -> None:
        self.cb_monitor.setChecked(False)
        self.sp_target.setValue(-30.0)
        self.cb_tx_auto.setChecked(True)
        self.sp_rx_ch.setValue(1)
        self.ed_rx_ant.setText("RX2")
        self.sp_rx_gain.setValue(30.0)
        self.sp_rx_freq.setValue(0.0)
        self.cb_echo.setChecked(False)
        self.cb_tx_check.setChecked(False)
        self._update_channel_validity()

    # ------------------------------------------------------------------
    def _build_menu(self) -> None:
        help_menu = self.menuBar().addMenu("Справка")
        act = QtWidgets.QAction("Справка / как пользоваться…", self)
        act.triggered.connect(self._show_help)
        help_menu.addAction(act)

    def _show_help(self) -> None:
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Справка")
        box.setTextFormat(QtCore.Qt.RichText)
        box.setText(_HELP_HTML)
        box.setStandardButtons(QtWidgets.QMessageBox.Ok)
        box.exec_()

    # ==================================================================
    # Right-hand tabs
    # ==================================================================
    def _build_gen_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(page)
        ctrl = QtWidgets.QHBoxLayout()
        self.btn_gen = QtWidgets.QPushButton("Сгенерировать IQ")
        self.btn_gen.setToolTip("Записать IQ-файл без передачи на B210")
        self.btn_gen.clicked.connect(self._generate_iq)
        self.btn_start = QtWidgets.QPushButton("▶ Старт")
        self.btn_start.setToolTip(
            "Записать IQ-файл и/или передать в эфир (см. галочку B210)")
        self.btn_start.clicked.connect(self._start)
        self.btn_stop = QtWidgets.QPushButton("■ Стоп")
        self.btn_stop.clicked.connect(self._stop)
        self.btn_stop.setEnabled(False)
        ctrl.addWidget(self.btn_gen)
        ctrl.addWidget(self.btn_start)
        ctrl.addWidget(self.btn_stop)
        ctrl.addStretch(1)
        v.addLayout(ctrl)

        # One visible bar area (issue 1), implemented as a stack of two bars so
        # the widget can be REPLACED when the phase changes:
        #   * preparation/generation fills 0→100 % completely;
        #   * when preparation finishes (or right before the first sample is
        #     sent) the visible widget becomes the transmission bar, reset to 0,
        #     labelled «Передача», and fills as samples are sent;
        #   * a pure IQ-file run never shows the transmission bar and keeps the
        #     generation bar at 100 %;
        #   * a cyclic transmission keeps wrapping the transmission bar per pass
        #     («циклическая передача» + elapsed/pass).
        self.progress_stack = QtWidgets.QStackedWidget()
        self.progress_prep = QtWidgets.QProgressBar()
        self.progress_prep.setRange(0, 1000)
        self.progress_prep.setFormat("%p%")
        self.progress_tx = QtWidgets.QProgressBar()
        self.progress_tx.setRange(0, 1000)
        self.progress_tx.setFormat("%p%")
        self.progress_stack.addWidget(self.progress_prep)
        self.progress_stack.addWidget(self.progress_tx)
        self.progress_stack.setCurrentWidget(self.progress_prep)
        v.addWidget(self.progress_stack)
        self.lbl_progress = QtWidgets.QLabel("Готово")
        self.lbl_progress.setWordWrap(True)
        v.addWidget(self.lbl_progress)
        # Compact transmit/receive level indicator (peak/RMS dBFS + clipping/
        # underflow), updated periodically by the runner (never per sample).
        self.lbl_tx_level = QtWidgets.QLabel("Уровень TX: —")
        self.lbl_tx_level.setToolTip(
            "Уровень передаваемого сигнала (пик и RMS в dBFS, как в SDR_Scan: "
            "комплексный тон 1.0 = 0 dBFS). Показывается число отсчётов с "
            "|x|>1 (клип) и underflow B210. Обновляется ~2 раза/с.")
        v.addWidget(self.lbl_tx_level)

        v.addWidget(QtWidgets.QLabel("Видимые спутники"))
        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["SV", "Система", "Элевация, °", "Азимут, °", "Амплитуда"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        v.addWidget(self.table, 2)

        v.addWidget(self._build_ublox_group())

        v.addWidget(QtWidgets.QLabel("Журнал"))
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(3000)
        v.addWidget(self.log, 3)
        return page

    # ------------------------------------------------------------------
    def _build_map_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(page)

        self.map = OsmMap()
        self.map.set_center(self.ed_lat.value(), self.ed_lon.value(), 14)
        self.map.trackChanged.connect(self._on_track_changed)
        h.addWidget(self.map, 3)

        side = QtWidgets.QVBoxLayout()
        h.addLayout(side, 2)
        side.addWidget(QtWidgets.QLabel(
            "Кликните по карте, чтобы нарисовать маршрут\n"
            "(начало — зелёное, конец — красное)."))
        self.lbl_track = QtWidgets.QLabel("Точек: 0")
        side.addWidget(self.lbl_track)

        g = QtWidgets.QGroupBox("Параметры самосвала")
        f = QtWidgets.QGridLayout(g)
        self.sp_vl = QtWidgets.QDoubleSpinBox(); self.sp_vl.setRange(1, 60)
        self.sp_vl.setValue(40.0); self.sp_vl.setSuffix(" км/ч")
        self.sp_ve = QtWidgets.QDoubleSpinBox(); self.sp_ve.setRange(1, 80)
        self.sp_ve.setValue(60.0); self.sp_ve.setSuffix(" км/ч")
        self.sp_acc = QtWidgets.QDoubleSpinBox(); self.sp_acc.setRange(0.1, 3)
        self.sp_acc.setValue(0.5); self.sp_acc.setSuffix(" м/с²")
        self.sp_brk = QtWidgets.QDoubleSpinBox(); self.sp_brk.setRange(0.5, 6)
        self.sp_brk.setValue(1.5); self.sp_brk.setSuffix(" м/с²")
        self.sp_lat = QtWidgets.QDoubleSpinBox(); self.sp_lat.setRange(0.1, 5)
        self.sp_lat.setValue(0.8); self.sp_lat.setSuffix(" м/с²")
        self.sp_load = QtWidgets.QDoubleSpinBox(); self.sp_load.setRange(0, 900)
        self.sp_load.setValue(180.0); self.sp_load.setSuffix(" с")
        self.sp_dump = QtWidgets.QDoubleSpinBox(); self.sp_dump.setRange(0, 900)
        self.sp_dump.setValue(60.0); self.sp_dump.setSuffix(" с")
        self.sp_cycles = QtWidgets.QSpinBox(); self.sp_cycles.setRange(1, 50)
        self.sp_cycles.setValue(1)
        self.sp_grade = QtWidgets.QDoubleSpinBox(); self.sp_grade.setRange(-25, 25)
        self.sp_grade.setValue(6.0); self.sp_grade.setSuffix(" %")
        self.sp_alt0 = QtWidgets.QDoubleSpinBox(); self.sp_alt0.setRange(-500, 6000)
        self.sp_alt0.setValue(10.0); self.sp_alt0.setSuffix(" м")
        self.cb_loaded = QtWidgets.QCheckBox("Старт с грузом")
        self.cb_loaded.setChecked(True)
        rows = (("С грузом", self.sp_vl), ("Порожний", self.sp_ve),
                ("Разгон", self.sp_acc), ("Торможение", self.sp_brk),
                ("Боковое ускор.", self.sp_lat), ("Погрузка", self.sp_load),
                ("Разгрузка", self.sp_dump), ("Циклов", self.sp_cycles),
                ("Уклон", self.sp_grade), ("Высота 0", self.sp_alt0))
        for i, (lbl, w) in enumerate(rows):
            _row(f, i, lbl, w)
        f.addWidget(self.cb_loaded, len(rows), 0, 1, 2)
        side.addWidget(g)

        btns = QtWidgets.QHBoxLayout()
        self.btn_calc = QtWidgets.QPushButton("Рассчитать трек")
        self.btn_calc.clicked.connect(self._calc_track)
        self.btn_play = QtWidgets.QPushButton("▶ Проиграть")
        self.btn_play.clicked.connect(self._play_track)
        btn_clear = QtWidgets.QPushButton("Очистить")
        btn_clear.clicked.connect(self._clear_track)
        btns.addWidget(self.btn_calc)
        btns.addWidget(self.btn_play)
        btns.addWidget(btn_clear)
        side.addLayout(btns)

        self.lbl_truck = QtWidgets.QLabel("Трек не рассчитан")
        self.lbl_truck.setWordWrap(True)
        side.addWidget(self.lbl_truck)
        side.addStretch(1)
        return page

    # ------------------------------------------------------------------
    def _build_spectrum_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(page)
        self.spectrum_widget = SpectrumWidget()
        v.addWidget(self.spectrum_widget, 1)
        self.lbl_spectrum = QtWidgets.QLabel(
            "Спектр TX обновляется при генерации (~2 раза/с, блок ≤65536 "
            "отсчётов).  Ось X — частота в МГц; ось Y — уровень в дБFS.  При "
            "мониторинге RX показывается и спектр принятого сигнала "
            "(смещение от центра приёма).  Без мониторинга ось X — абсолютная "
            "частота (центр + смещение).")
        self.lbl_spectrum.setWordWrap(True)
        v.addWidget(self.lbl_spectrum)
        return page

    # ------------------------------------------------------------------
    def _build_ublox_group(self) -> QtWidgets.QGroupBox:
        g = QtWidgets.QGroupBox("Приёмник u-blox (COM)")
        f = QtWidgets.QGridLayout(g)

        self.cmb_port = QtWidgets.QComboBox()
        ports = ublox.list_ports()
        if not ports:
            ports = ["COM3"]
        self.cmb_port.addItems(ports)
        if "COM3" in ports:
            self.cmb_port.setCurrentText("COM3")
        self._guard_combo(self.cmb_port)
        _row(f, 0, "Порт", self.cmb_port)

        self.cmb_baud = QtWidgets.QComboBox()
        self.cmb_baud.addItems(["9600", "38400", "115200", "460800"])
        self.cmb_baud.setCurrentText("38400")
        self._guard_combo(self.cmb_baud)
        _row(f, 1, "Скорость", self.cmb_baud)

        self.btn_connect = QtWidgets.QPushButton("Подключить")
        self.btn_connect.clicked.connect(self._ublox_connect)
        self.btn_disconnect = QtWidgets.QPushButton("Отключить")
        self.btn_disconnect.clicked.connect(self._ublox_disconnect)
        self.btn_disconnect.setEnabled(False)
        rconn = QtWidgets.QHBoxLayout()
        rconn.addWidget(self.btn_connect)
        rconn.addWidget(self.btn_disconnect)
        f.addWidget(self._wrap(rconn), 2, 0, 1, 2)

        self.btn_cold = QtWidgets.QPushButton("Холодный старт")
        self.btn_cold.setToolTip(
            "Отправить UBX-CFG-RST (cold start: очистка BBR) на выбранный порт")
        self.btn_cold.clicked.connect(self._ublox_cold_start)
        f.addWidget(self.btn_cold, 3, 0, 1, 2)

        self.lbl_ublox = QtWidgets.QLabel("Не подключён")
        self.lbl_ublox.setWordWrap(True)
        f.addWidget(self.lbl_ublox, 4, 0, 1, 2)

        self.tbl_ublox = QtWidgets.QTableWidget(0, 5)
        self.tbl_ublox.setHorizontalHeaderLabels(
            ["Система", "PRN", "Элевация, °", "Азимут, °", "SNR"])
        self.tbl_ublox.horizontalHeader().setStretchLastSection(True)
        self.tbl_ublox.setEditTriggers(
            QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tbl_ublox.setMaximumHeight(140)
        f.addWidget(self.tbl_ublox, 5, 0, 1, 2)

        rtake = QtWidgets.QHBoxLayout()
        self.btn_take_pos = QtWidgets.QPushButton("Взять позицию из приёмника")
        self.btn_take_pos.clicked.connect(self._take_position)
        self.btn_take_time = QtWidgets.QPushButton("Взять время из приёмника")
        self.btn_take_time.clicked.connect(self._take_time)
        rtake.addWidget(self.btn_take_pos)
        rtake.addWidget(self.btn_take_time)
        f.addWidget(self._wrap(rtake), 6, 0, 1, 2)
        return g

    # ------------------------------------------------------------------
    def _reset_defaults(self) -> None:
        """Reset every settings group; CDDIS credentials are preserved.

        Kept for compatibility: it delegates to the per-group resets.  It is
        deliberately *not* wired to a single global button.
        """
        self._reset_nav_group()
        self._reset_signals_group()
        self._reset_output_group()
        self._reset_rf_group()
        self._reset_band_group()
        self._reset_cddis_group()
        self._reset_uhd_group()
        self._reset_monitor_group()

    # ==================================================================
    # RINEX downloads
    # ==================================================================
    def _start_text(self) -> str:
        """Start time as ``now`` or a full ``YYYY/MM/DD HH:mm:ss`` string."""
        if getattr(self, "chk_now", None) is None or self.chk_now.isChecked():
            return "now"
        return self.ed_start.dateTime().toString("yyyy/MM/dd HH:mm:ss")

    def _default_date_text(self) -> str:
        try:
            from .config import parse_start_time
            from .gpstime import gps2date
            y, m, d, _hh, _mm, _ss = gps2date(parse_start_time(self._start_text()))
            return f"{y:04d}/{m:02d}/{d:02d}"
        except Exception:  # noqa: BLE001
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            return f"{now.year:04d}/{now.month:02d}/{now.day:02d}"

    def _validate_start(self, cfg: SimConfig) -> bool:
        """Reject a start time not covered by the local RINEX ephemerides."""
        if cfg.iq_input:
            return True
        try:
            start = cfg.resolved_start()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(
                self, "Время старта", f"Неверная дата/время: {exc}")
            return False

        path = str(cfg.nav_file or "").strip()
        if not path or not os.path.exists(path):
            try:
                from .gpstime import gps2date
                from .rinexfetch import cached_nav_for_date, doy_from_date
                y, m, d, _h, _mi, _s = gps2date(start)
                path = cached_nav_for_date(y, doy_from_date(y, m, d)) or ""
            except Exception:  # noqa: BLE001
                path = ""
        if not path or not os.path.exists(path):
            return True  # будет автоскачивание — проверим после загрузки

        try:
            key = (path, os.path.getmtime(path))
            cached = getattr(self, "_nav_span_cache", None)
            if cached and cached[0] == key:
                by_sv = cached[1]
            else:
                from .rinex import parse_nav_file
                by_sv, _iono = parse_nav_file(path)
                self._nav_span_cache = (key, by_sv)
            from .rinex import check_start_coverage
            msg = check_start_coverage(start, by_sv)
        except Exception:  # noqa: BLE001 - ошибки разбора покажет раннер
            return True
        if msg:
            QtWidgets.QMessageBox.warning(self, "Время старта", msg)
            return False
        return True

    def _download_cddis_date(self) -> None:
        raw = (self.ed_date.text() or "").strip()
        raw = raw.replace("-", "/").replace(".", "/")
        try:
            y, m, d = (int(x) for x in raw.split("/"))
        except Exception:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "CDDIS",
                                          "Формат даты: YYYY/MM/DD")
            return
        try:
            from .rinexfetch import doy_from_date
            doy = doy_from_date(y, m, d)
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "CDDIS", f"Неверная дата: {exc}")
            return
        user = self.ed_cddis_user.text().strip()
        password = self.ed_cddis_pass.text()
        token = self.ed_cddis_token.text().strip()
        if not (token or (user and password)):
            # CDDIS credentials missing: prompt in a dedicated modal (B2).
            if not self._prompt_cddis_credentials():
                return
            user = self.ed_cddis_user.text().strip()
            password = self.ed_cddis_pass.text()
            token = self.ed_cddis_token.text().strip()
            if not (token or (user and password)):
                QtWidgets.QMessageBox.warning(
                    self, "CDDIS",
                    "Логин/пароль Earthdata или bearer-токен обязательны.")
                return
        yy = y % 100
        self._append_log(
            f"CDDIS: дата {y:04d}-{m:02d}-{d:02d} -> DOY {doy:03d}; merged "
            f"BRDC00IGS_R_{y}{doy:03d}0000_01D_MN.rnx.gz (только за прошлые "
            f"сутки) либо посистемные brdc{doy:03d}0.{yy:02d}{{n,g,l,c,j}}")
        self.btn_cddis.setEnabled(False)
        self._cddis_thread = _CddisDownloadThread(
            y, m, d, user, password, token, self)
        self._cddis_thread.message.connect(self._append_log)
        self._cddis_thread.done.connect(self._on_cddis_done)
        self._cddis_thread.start()

    def _on_cddis_done(self, path, err) -> None:
        self.btn_cddis.setEnabled(True)
        if err:
            self._append_log(f"CDDIS: ошибка: {err}")
            QtWidgets.QMessageBox.critical(self, "CDDIS", str(err))
            return
        if isinstance(path, (list, tuple)):
            paths = [str(p) for p in path]
            self.ed_nav.setText(";".join(paths))
            self._append_log(
                "CDDIS: загружено " + str(len(paths)) + " посистемных "
                "файлов: " + ", ".join(paths))
        else:
            self.ed_nav.setText(str(path))
            self._append_log(f"CDDIS: загружено -> {path}")
        # The downloaded RINEX resolves the coverage window: refresh the
        # start-time range/date at once so no stale date is left behind.
        self._update_start_range()
        self._refresh_start_date_label()

    # ==================================================================
    # RINEX reuse question (thread-safe) and runner launch
    # ==================================================================
    def _ask_nav_update(self, path: str, age: float) -> bool:
        """Called from the runner thread; blocks until the user answers."""
        self._ask_result[0] = False
        self._ask_event.clear()
        self.bridge.askNav.emit(str(path), float(age))
        if not self._ask_event.wait(timeout=600.0):
            return False
        return self._ask_result[0]

    def _on_ask_nav(self, path: str, age: float) -> None:
        reply = QtWidgets.QMessageBox.question(
            self, "Обновить эфемериды?",
            f"Кэшированный файл RINEX за сегодня старше {age / 60.0:.0f} мин:\n"
            f"{path}\n\nОбновить его из сети?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No)
        self._ask_result[0] = (reply == QtWidgets.QMessageBox.Yes)
        self._ask_event.set()

    def _check_nav(self, cfg: SimConfig) -> bool:
        if not cfg.nav_file and not cfg.auto_download:
            QtWidgets.QMessageBox.warning(
                self, "Нет эфемерид",
                "Укажите RINEX navigation файл или включите автоскачивание.")
            return False
        return True

    def _prompt_cddis_credentials(self) -> bool:
        """Ask for CDDIS login/password/token in a modal dialog (B2)."""
        dlg = _CddisCredsDialog(
            self, self.ed_cddis_user.text().strip(),
            self.ed_cddis_token.text().strip())
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return False
        user, password, token = dlg.values()
        self.ed_cddis_user.setText(user)
        self.ed_cddis_pass.setText(password)
        self.ed_cddis_token.setText(token)
        self._append_log("CDDIS: учётные данные сохранены в настройках сессии")
        return True

    def _ensure_cddis_credentials(self, cfg: SimConfig) -> bool:
        """Return True if CDDIS credentials are present (prompt if needed)."""
        if cfg.download_source != "cddis":
            return True
        if cfg.cddis_token or (cfg.cddis_user and cfg.cddis_password):
            return True
        self._append_log("CDDIS: нет логина/пароля или токена — запрашиваю")
        if not self._prompt_cddis_credentials():
            return False
        cfg.cddis_user = self.ed_cddis_user.text().strip()
        cfg.cddis_password = self.ed_cddis_pass.text()
        cfg.cddis_token = self.ed_cddis_token.text().strip()
        return bool(cfg.cddis_token or (cfg.cddis_user and cfg.cddis_password))

    def _launch(self, cfg: SimConfig) -> None:
        self.log.clear()
        self.table.setRowCount(0)
        self._cyclic_tx = False
        self._pregen_seen = False
        self._phase_seen = False
        self._bar_switched = False
        self.progress_stack.setCurrentWidget(self.progress_prep)
        self.progress_prep.setValue(0)
        self.progress_prep.setFormat("%p%")
        self.progress_tx.setValue(0)
        self.progress_tx.setFormat("%p%")
        if hasattr(self, "lbl_progress"):
            self.lbl_progress.setText("Подготовка…")
        if hasattr(self, "lbl_tx_level"):
            self.lbl_tx_level.setText("Уровень TX: —")
        # A reused IQ file has no synthesis step: when it is streamed to the
        # radio show the transmission bar straight away (the runner emits only
        # an overall fraction there, which ``_on_progress`` feeds to it).
        if cfg.iq_input and cfg.use_usrp:
            self._switch_to_tx_bar()
        self.btn_start.setEnabled(False)
        self.btn_gen.setEnabled(False)
        self.btn_stop.setEnabled(True)
        try:
            self.spectrum_widget.clear()
        except Exception:  # noqa: BLE001
            pass
        self._run_cfg = cfg
        self.runner = SimulationRunner(
            cfg,
            log=self.bridge.log.emit,
            progress=self.bridge.progress.emit,
            channels=self.bridge.channels.emit,
            finished=self.bridge.finished.emit,
            spectrum=self.bridge.spectrum.emit,
            ask=self._ask_nav_update,
            phase=self.bridge.phase.emit,
            level=self.bridge.level.emit,
        )
        self.runner.start()

    def _start(self) -> None:
        cfg = self._collect()
        if cfg.iq_input:
            if not os.path.exists(cfg.iq_input):
                QtWidgets.QMessageBox.warning(
                    self, "Готовый IQ-файл",
                    f"Файл не найден: {cfg.iq_input}")
                return
            self._append_log(
                "Повторное использование IQ-файла: генерация пропускается.")
            self._launch(cfg)
            return
        if not self._check_nav(cfg):
            return
        if not self._ensure_cddis_credentials(cfg):
            return
        self._correct_multignss_start(cfg)
        if not self._validate_start(cfg):
            return
        if (cfg.monitor or cfg.tx_check) and cfg.tx_channel == cfg.rx_channel:
            self._update_channel_validity()
            QtWidgets.QMessageBox.warning(
                self, "Каналы TX/RX",
                "Для мониторинга/контроля RX TX и RX каналы должны "
                "различаться.")
            return
        self._launch(cfg)

    def _generate_iq(self) -> None:
        cfg = self._collect()
        if cfg.iq_input:
            cfg.use_usrp = False
            cfg.monitor = False
            cfg.tx_check = False
            if not os.path.exists(cfg.iq_input):
                QtWidgets.QMessageBox.warning(
                    self, "Готовый IQ-файл",
                    f"Файл не найден: {cfg.iq_input}")
                return
            self._append_log(
                "Проверка готового IQ-файла (генерация пропускается).")
            self._launch(cfg)
            return
        if not cfg.output:
            QtWidgets.QMessageBox.warning(
                self, "Выход IQ", "Укажите файл для IQ-сигнала (поле «Файл»).")
            return
        if not self._check_nav(cfg):
            return
        if not self._ensure_cddis_credentials(cfg):
            return
        self._correct_multignss_start(cfg)
        if not self._validate_start(cfg):
            return
        cfg.use_usrp = False
        cfg.monitor = False
        cfg.tx_check = False
        self._append_log("Генерация IQ в файл (без передачи на B210).")
        self._launch(cfg)

    def _stop(self) -> None:
        if self.runner is not None:
            self.runner.stop()

    # ==================================================================
    # Spectrum
    # ==================================================================
    def _on_spectrum(self, channel, samples, fs) -> None:
        try:
            center = 0.0
            if self._run_cfg is not None and not self._run_cfg.monitor:
                center = self._run_cfg.center_freq
            self.spectrum_widget.update_spectrum(samples, fs, channel,
                                                 center_freq=center)
            self.lbl_spectrum.setText(self.spectrum_widget.describe())
        except Exception as exc:  # noqa: BLE001
            self._append_log(f"Спектр: {exc}")

    def _on_level(self, label: str, peak_dbfs: float, rms_dbfs: float,
                  clips: int) -> None:
        """Update the compact TX/RX level label (peak/RMS dBFS + клип/underflow)."""
        if not hasattr(self, "lbl_tx_level"):
            return
        txt = (f"{label}: пик {peak_dbfs:.1f} dBFS, RMS {rms_dbfs:.1f} dBFS")
        if clips:
            txt += f", КЛИП {int(clips)}"
        uf = 0
        try:
            sink = getattr(self.runner, "sink", None)
            if sink is not None:
                uf = int(getattr(sink, "underflows", 0) or 0)
        except Exception:  # noqa: BLE001
            uf = 0
        if uf:
            txt += f", underflow {uf}"
        self.lbl_tx_level.setText(txt)

    # ==================================================================
    # u-blox
    # ==================================================================
    def _ublox_connect(self) -> None:
        port = self.cmb_port.currentText().strip()
        try:
            baud = int(self.cmb_baud.currentText())
        except ValueError:
            baud = 38400
        try:
            self._nmea = ublox.NmeaReader(port, baud)
            self._nmea.start()
        except Exception as exc:  # noqa: BLE001
            self._nmea = None
            self._append_log(f"u-blox: не удалось открыть {port}: {exc}")
            QtWidgets.QMessageBox.warning(
                self, "u-blox", f"Не удалось открыть {port}: {exc}")
            return
        self._nmea_timer.start()
        self.btn_connect.setEnabled(False)
        self.btn_disconnect.setEnabled(True)
        self._append_log(f"u-blox: подключён {port} @ {baud} бод")

    def _ublox_disconnect(self) -> None:
        self._nmea_timer.stop()
        if self._nmea is not None:
            try:
                self._nmea.stop()
            except Exception as exc:  # noqa: BLE001
                self._append_log(f"u-blox: {exc}")
            self._nmea = None
        self.btn_connect.setEnabled(True)
        self.btn_disconnect.setEnabled(False)
        self.lbl_ublox.setText("Не подключён")
        self.tbl_ublox.setRowCount(0)
        self._append_log("u-blox: отключён")

    def _ublox_cold_start(self) -> None:
        port = self.cmb_port.currentText().strip()
        if not port:
            QtWidgets.QMessageBox.warning(self, "u-blox", "Выберите COM-порт.")
            return
        try:
            baud = int(self.cmb_baud.currentText())
        except ValueError:
            baud = 38400
        reply = QtWidgets.QMessageBox.question(
            self, "Холодный старт",
            f"Отправить UBX-CFG-RST (cold start) на {port} @ {baud}?\n"
            "Приёмник сбросит эфемериды/альманах и начнёт поиск заново.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No)
        if reply != QtWidgets.QMessageBox.Yes:
            return
        try:
            from .ublox import cold_start, ubx_cfg_rst_frame
            if self._nmea is not None and getattr(self._nmea, "port", "") == port:
                frame = ubx_cfg_rst_frame()
                self._nmea.send(frame)
            else:
                frame = cold_start(port, baud)
        except Exception as exc:  # noqa: BLE001
            self._append_log(f"u-blox: холодный старт не удался: {exc}")
            QtWidgets.QMessageBox.critical(self, "u-blox", str(exc))
            return
        self._append_log(
            f"u-blox: холодный старт (UBX-CFG-RST) отправлен на {port}, "
            f"кадр {frame.hex(' ').upper()}")

    def _update_ublox(self) -> None:
        if self._nmea is None:
            return
        try:
            p = self._nmea.position()
            dt = self._nmea.datetime_utc()
            sats = self._nmea.satellites()
        except Exception as exc:  # noqa: BLE001
            self._append_log(f"u-blox: {exc}")
            return
        self.lbl_ublox.setText(
            f"lat {p['lat']}  lon {p['lon']}  h {p['height']} м | "
            f"UTC {dt or '—'} | fix {p['fix']} | sats {p['num_sats']} | "
            f"HDOP {p['hdop']}")
        self.tbl_ublox.setRowCount(len(sats))
        for i, s in enumerate(sats):
            values = [s.get("system", ""), s.get("prn", ""),
                      s.get("elev", ""), s.get("az", ""), s.get("snr", "")]
            for j, v in enumerate(values):
                self.tbl_ublox.setItem(
                    i, j, QtWidgets.QTableWidgetItem(str(v)))

    def _take_position(self) -> None:
        if self._nmea is None:
            QtWidgets.QMessageBox.warning(self, "u-blox", "Приёмник не подключён.")
            return
        p = self._nmea.position()
        if p.get("lat") is None or p.get("lon") is None:
            QtWidgets.QMessageBox.warning(self, "u-blox", "Нет координат (fix?).")
            return
        self.ed_lat.setValue(float(p["lat"]))
        self.ed_lon.setValue(float(p["lon"]))
        if p.get("height") is not None:
            self.ed_hgt.setValue(float(p["height"]))
        self._append_log(
            f"u-blox: позиция {p['lat']:.6f}, {p['lon']:.6f}, {p['height']} м")

    def _take_time(self) -> None:
        if self._nmea is None:
            QtWidgets.QMessageBox.warning(self, "u-blox", "Приёмник не подключён.")
            return
        dt = self._nmea.datetime_utc()
        if not dt:
            QtWidgets.QMessageBox.warning(self, "u-blox", "Нет времени UTC (RMC?).")
            return
        qdt = QtCore.QDateTime.fromString(str(dt), "yyyy-MM-dd HH:mm:ss")
        if not qdt.isValid():
            qdt = QtCore.QDateTime.fromString(str(dt), "yyyy/MM/dd HH:mm:ss")
        if qdt.isValid():
            qdt.setTimeSpec(QtCore.Qt.UTC)
            self.ed_start.setDateTime(qdt)
            self.chk_now.setChecked(False)
        self._append_log(
            f"u-blox: время старта -> {self._start_text()}")

    # ==================================================================
    # Channels / progress / collect
    # ==================================================================
    def _on_tx_channel_changed(self) -> None:
        if self.sp_rx_ch.value() == self.sp_ch.value():
            self.sp_rx_ch.setValue(1 if self.sp_ch.value() == 0 else 0)
        self._update_channel_validity()

    def _update_channel_validity(self) -> None:
        if not hasattr(self, "btn_start"):
            return
        bad = ((self.cb_monitor.isChecked() or self.cb_tx_check.isChecked())
               and self.sp_ch.value() == self.sp_rx_ch.value())
        self.lbl_mon_warn.setVisible(bad)
        if bad:
            self.btn_start.setToolTip(
                "TX и RX каналы совпадают — выберите разные каналы")
            self.btn_start.setEnabled(False)
        else:
            self.btn_start.setToolTip(
                "Записать IQ-файл и/или передать в эфир (см. галочку B210)")
            running = self.runner is not None and self.runner.is_running()
            if not running:
                self.btn_start.setEnabled(True)

    # ------------------------------------------------------------------
    @staticmethod
    def _wrap(layout: QtWidgets.QLayout) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        w.setLayout(layout)
        return w

    def _browse(self, edit: QtWidgets.QLineEdit, title: str, pattern: str,
                selected: str | None = None) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, title, os.path.dirname(edit.text()) or ".", pattern,
            selected or "")
        if path:
            edit.setText(path)

    def _save_browse(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Выходной IQ-файл", self.ed_out.text(),
            "IQ (*.cs16 *.cf32 *.cs8 *.cs4 *.bin *.iq);;Все файлы (*)")
        if path:
            self.ed_out.setText(path)

    # ==================================================================
    # Track
    # ==================================================================
    def _on_track_changed(self, points: list) -> None:
        self.lbl_track.setText(f"Точек: {len(points)}")

    def _truck_params(self) -> trackmod.TruckParams:
        return trackmod.TruckParams(
            v_laden_kmh=self.sp_vl.value(), v_empty_kmh=self.sp_ve.value(),
            accel=self.sp_acc.value(), brake=self.sp_brk.value(),
            lat_acc=self.sp_lat.value(), load_s=self.sp_load.value(),
            dump_s=self.sp_dump.value(), cycles=self.sp_cycles.value(),
            grade_pct=self.sp_grade.value(), altitude0=self.sp_alt0.value(),
            start_loaded=self.cb_loaded.isChecked())

    def _calc_track(self) -> None:
        pts = self.map.track
        if len(pts) < 2:
            QtWidgets.QMessageBox.warning(self, "Мало точек",
                                          "Нарисуйте минимум две точки на карте.")
            return
        self._timer.stop()
        try:
            lat, lon, s = trackmod.resample_track(pts, spacing=2.0)
            result = trackmod.simulate_truck(lat, lon, s, self._truck_params())
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Ошибка", str(exc))
            return
        self._truck_result = result
        path = os.path.join(os.getcwd(), "truck_track.csv")
        trackmod.write_motion_csv(path, result)
        self.ed_motion.setText(path)
        dur = result["t"][-1]
        spd = [v * 3.6 for v in result["v"]]
        self.lbl_truck.setText(
            f"Трек: {len(result['t'])} отсчётов (~{dur:.0f} с, "
            f"{len(s)*0.0 + s[-1]:.0f} м); v {min(spd):.0f}..{max(spd):.0f} км/ч; "
            f"h {min(result['h']):.0f}..{max(result['h']):.0f} м\n"
            f"Motion-файл: {path}")
        self._append_log(f"Трек самосвала: {len(result['t'])} отсчётов, "
                         f"{dur:.1f} с -> {path}")
        self._anim_idx = 0.0
        self._play_track()

    def _play_track(self) -> None:
        if self._truck_result is None:
            return
        self._anim_idx = 0.0
        self._timer.start(40)

    def _anim_tick(self) -> None:
        r = self._truck_result
        if r is None:
            self._timer.stop()
            return
        n = len(r["t"])
        self._anim_idx += max(1.0, n / 500.0)
        i = int(self._anim_idx)
        if i >= n:
            i = n - 1
            self._timer.stop()
        self.map.set_truck(r["lat"][i], r["lon"][i])
        self.lbl_truck.setText(
            f"t = {r['t'][i]:.1f} с · v = {r['v'][i]*3.6:.1f} км/ч · "
            f"h = {r['h'][i]:.1f} м · фаза: {r['phase'][i]}")

    def _clear_track(self) -> None:
        self.map.clear_track()
        self._truck_result = None
        self.lbl_truck.setText("Трек не рассчитан")

    # ==================================================================
    # Log / callbacks
    # ==================================================================
    def _append_log(self, text: str) -> None:
        """Append log lines, colouring only genuine warnings/errors.

        Native UHD stderr is captured and forwarded in arbitrary multi-line
        chunks, so the payload is split first: otherwise a single ``[WARNING]``
        line would colour the *whole* chunk (and Qt ``appendHtml`` renders it as
        one orange block, hiding the plain native lines).  Each line is added as
        its own paragraph, and plain lines use ``appendPlainText`` so the colour
        can never leak to the following lines.
        """
        import html
        lines = str(text).splitlines()
        if not lines:
            lines = [""]
        for line in lines:
            low = line.lower()
            color = None
            if ("ошибк" in low or "error" in low or "исключени" in low):
                color = _LOG_ERROR_COLOR
            elif ("вниман" in low or "warning" in low or "предупреж" in low):
                color = _LOG_WARN_COLOR
            if color and hasattr(self.log, "appendHtml"):
                safe = html.escape(line)
                self.log.appendHtml(
                    f'<span style="color:{color};">{safe}</span>')
            else:
                self.log.appendPlainText(line)

    def _append_native_log(self, text: str) -> None:
        """Forward captured native UHD stderr into the journal (issue 2).

        Native lines are shown as plain text; only genuine UHD warnings/errors
        (``[WARNING]``/``[ERROR]`` tags, or an explicit Russian marker) are
        coloured, so an ``[INFO]`` line that happens to contain the substring
        ``error`` stays uncoloured.  The chunk is split first: the fd-2 reader
        forwards arbitrary multi-line chunks.
        """
        import html
        lines = str(text).splitlines()
        if not lines:
            lines = [""]
        for line in lines:
            stripped = line.lstrip()
            genuine = (stripped.startswith(("[WARNING]", "[ERROR]"))
                       or "ВНИМАН" in line or "ОШИБК" in line)
            color = None
            if genuine:
                low = line.lower()
                if "error" in low or "ошибк" in low:
                    color = _LOG_ERROR_COLOR
                else:
                    color = _LOG_WARN_COLOR
            if color and hasattr(self.log, "appendHtml"):
                self.log.appendHtml(
                    f'<span style="color:{color};">{html.escape(line)}</span>')
            else:
                self.log.appendPlainText(line)

    def _on_progress(self, frac: float, sim_s: float, wall: float,
                     rate: float) -> None:
        """Drive the visible bar from the runner's overall fraction.

        The overall fraction is only used while no phase signal has taken over:
        a pure IQ-file run (no radio) has no ``phase`` callback, so this fills
        the preparation/generation bar to 100 % and it stays there.  A B210 run
        is driven by :meth:`_on_phase` (preparation 0…100 %, then the
        transmission bar), so late overall values must not clobber it.
        """
        try:
            value = int(round(float(frac or 0.0) * 1000.0))
        except (TypeError, ValueError):
            value = 0
        value = max(0, min(1000, value))
        # Until the runner reports a distinct phase, the overall fraction drives
        # whichever bar is visible: preparation/generation by default, the
        # transmission bar for a reused IQ file streamed straight to the radio.
        # Once a phase signal arrived the phase owns the bar.
        if not self._phase_seen:
            tx = bool(self._run_cfg is not None and self._run_cfg.use_usrp)
            bar = self.progress_tx if self._bar_switched else self.progress_prep
            bar.setValue(value)
            if hasattr(self, "lbl_progress"):
                if self._bar_switched:
                    bar.setFormat("передача %p%")
                    self.lbl_progress.setText(
                        f"Передача: {value / 10:.0f}% ({sim_s:.1f} с)")
                elif tx:
                    bar.setFormat("подготовка %p%")
                    self.lbl_progress.setText(
                        f"Подготовка: {value / 10:.0f}% ({sim_s:.1f} с)")
                else:
                    bar.setFormat("генерация %p%")
                    self.lbl_progress.setText(
                        f"Генерация: {value / 10:.0f}% ({sim_s:.1f} с)")
        self.setWindowTitle(
            f"{_APP_TITLE} — {sim_s:.1f} с / {wall:.1f} с ({rate:.2f}x)")

    def _switch_to_tx_bar(self) -> None:
        """Replace the preparation widget with the transmission bar (reset to 0)."""
        if not self._bar_switched:
            self.progress_tx.setValue(0)
            self.progress_tx.setFormat("%p%")
            if hasattr(self, "lbl_progress"):
                self.lbl_progress.setText("Передача…")
        self._bar_switched = True
        self.progress_stack.setCurrentWidget(self.progress_tx)

    def _on_phase(self, kind: str, frac: float,
                  loops: int, sim_s: float, cyclic: bool) -> None:
        """Update the preparation or transmission bar for the current phase.

        ``pregen`` fills the preparation bar 0…100 %; when it completes (or on
        the first ``tx`` update, i.e. right before sending) the visible widget is
        replaced by the transmission bar, reset to 0.  A cyclic transmission
        wraps that bar every pass and labels it «циклическая передача» with the
        elapsed time and pass number, so it never freezes at 100 %.
        """
        try:
            value = int(round(float(frac or 0.0) * 1000.0))
        except (TypeError, ValueError):
            value = 0
        value = max(0, min(1000, value))
        lab = getattr(self, "lbl_progress", None)
        self._phase_seen = True
        if kind == "pregen":
            self._cyclic_tx = False
            self._pregen_seen = True
            if not self._bar_switched:
                self.progress_prep.setValue(value)
                self.progress_prep.setFormat("подготовка %p%")
                if lab is not None:
                    lab.setText(f"Подготовка: {value / 10:.0f}% "
                                f"({sim_s:.1f} с сигнала)")
            if value >= 1000:
                # Preparation is complete: replace the widget with the TX bar.
                self._switch_to_tx_bar()
            return
        if kind != "tx":
            return
        # First TX update (or end of pre-generation): show the TX bar, reset.
        self._switch_to_tx_bar()
        if cyclic:
            self._cyclic_tx = True
            self.progress_tx.setValue(value)
            self.progress_tx.setFormat("циклическая передача %p%")
            if lab is not None:
                lab.setText(f"Циклическая передача: проход {int(loops) + 1}, "
                            f"{sim_s:.1f} с (цикл {value / 10:.0f}%)")
        else:
            self._cyclic_tx = False
            self.progress_tx.setValue(value)
            self.progress_tx.setFormat("передача %p%")
            if lab is not None:
                lab.setText(f"Передача: {value / 10:.0f}% ({sim_s:.1f} с)")

    def _on_channels(self, chans: list) -> None:
        self.table.setRowCount(len(chans))
        for i, c in enumerate(chans):
            values = [c.get("name", c.get("prn")), c.get("kind", ""),
                      f"{math.degrees(c['elev']):.1f}",
                      f"{math.degrees(c['azim']):.1f}", f"{c['amp']:.4f}"]
            for j, v in enumerate(values):
                self.table.setItem(i, j, QtWidgets.QTableWidgetItem(str(v)))

    def _on_finished(self, err) -> None:
        self.btn_start.setEnabled(True)
        self.btn_gen.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if err:
            QtWidgets.QMessageBox.critical(self, "Ошибка", str(err))
            if hasattr(self, "lbl_progress"):
                self.lbl_progress.setText(f"Ошибка: {err}")
        else:
            self._cyclic_tx = False
            self.progress.setFormat("%p%")
            self.progress.setValue(1000)
            if hasattr(self, "lbl_progress"):
                self.lbl_progress.setText("Готово")
        self._update_channel_validity()

    # ------------------------------------------------------------------
    def _collect(self) -> SimConfig:
        # «Сигналы» are the single source of truth: derive band, centre/fs and
        # the RINEX source mode from the checked systems.
        enables = dict(
            enable_ca=self.cb_ca.isChecked(), enable_l1c=self.cb_l1c.isChecked(),
            enable_galileo=self.cb_gal.isChecked(),
            enable_qzss=self.cb_qzss.isChecked(),
            enable_sbas=self.cb_sbas.isChecked(),
            enable_beidou=self.cb_bds.isChecked())
        combine = (self.cb_combine.isChecked()
                   if hasattr(self, "cb_combine") else False)
        band = derive_band_key(combine=combine, **enables)
        plan = derive_band_plan(combine=combine, **enables)
        fs = plan.fs
        center = plan.center_freq
        nav_mode = derive_nav_mode(enable_galileo=enables["enable_galileo"],
                                   enable_qzss=enables["enable_qzss"],
                                   enable_beidou=enables["enable_beidou"])
        return SimConfig(
            nav_file=self.ed_nav.text().strip(),
            lat=self.ed_lat.value(), lon=self.ed_lon.value(),
            height=self.ed_hgt.value(),
            motion_file=self.ed_motion.text().strip(),
            start_text=self._start_text(),
            duration=self.sp_dur.value(),
            fs=fs,
            center_freq=center,
            band=band,
            fs_override=False,
            center_override=False,
            combine=combine,
            nav_mode=nav_mode,
            **enables,
            el_mask=self.sp_el.value(),
            amp_scale=(None if self.cb_amp_auto.isChecked()
                       else self.sp_amp.value()),
            headroom=self.cb_headroom.isChecked(),
            headroom_target=self.sp_headroom.value(),
            iono_enable=self.cb_iono.isChecked(),
            l1c_data=self.cmb_l1c_data.currentText(),
            b1i_data=self.cmb_b1i_data.currentText(),
            auto_download=self.cb_auto.isChecked(),
            download_source=self.cmb_source.currentText(),
            auto_b1i=self.cb_auto_b1i.isChecked(),
            cddis_user=self.ed_cddis_user.text().strip(),
            cddis_password=self.ed_cddis_pass.text(),
            cddis_token=self.ed_cddis_token.text().strip(),
            output=self.ed_out.text().strip(),
            output_format=self.cmb_fmt.currentText(),
            output_scale=self.sp_scale.value(),
            iq_input=(self.ed_iq_in.text().strip()
                      if self.cb_iq_in.isChecked() else ""),
            iq_in_ram=self.cb_iq_ram.isChecked(),
            loop=self.cb_loop.isChecked(),
            loop_seconds=self.sp_loop.value(),
            memory_budget_gb=self.sp_mem.value(),
            use_usrp=self.cb_tx.isChecked(), uhd_args=self.ed_args.text().strip(),
            tx_channel=self.sp_ch.value(), tx_gain=self.sp_txg.value(),
            tx_antenna=self.ed_ant.text().strip(),
            tx_bandwidth=self.sp_bw.value(),
            clock_source=self.cmb_clk.currentText(),
            backend=self.cmb_backend.currentText(),
            monitor=self.cb_monitor.isChecked(),
            tx_power_target_dbfs=self.sp_target.value(),
            tx_power_auto=self.cb_tx_auto.isChecked(),
            rx_channel=self.sp_rx_ch.value(),
            rx_antenna=self.ed_rx_ant.text().strip(),
            rx_gain=self.sp_rx_gain.value(),
            rx_center_freq=self.sp_rx_freq.value(),
            echo_analysis=self.cb_echo.isChecked(),
            tx_check=self.cb_tx_check.isChecked(),
        )

    def _find_devices(self) -> None:
        from .uhd_tx import list_devices, uhd_version
        self._append_log(f"UHD: {uhd_version()}")
        try:
            for dev in list_devices(self.ed_args.text().strip()):
                self._append_log("  " + str(dict(dev)))
        except Exception as exc:  # noqa: BLE001
            self._append_log(f"  ошибка: {exc}")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        self._timer.stop()
        self._nmea_timer.stop()
        if self._nmea is not None:
            try:
                self._nmea.stop()
            except Exception:  # noqa: BLE001
                pass
            self._nmea = None
        if self.runner is not None:
            self.runner.stop()
        event.accept()


def main() -> int:
    # Re-assert the GUI UHD level/log capture in case this entry point is used
    # directly (``python -m gnss_sim.gui``); MainWindow points the sink at its
    # own log widget.
    quiet_uhd_gui()
    install_native_stderr_filter()
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
