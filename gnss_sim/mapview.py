"""Dependency-free OpenStreetMap widget for PyQt5 (tiles fetched from OSM).

Only the standard library (urllib) and Qt are used: tiles are downloaded as PNG,
cached on disk and drawn with ``QPixmap`` (Qt decodes PNG natively).  Supports
pan (drag), zoom (wheel), click-to-add track points, right-click to remove the
last point, and drawing a polyline plus an optional moving marker.
"""

from __future__ import annotations

import math
import os
import urllib.request

from PyQt5 import QtCore, QtGui, QtWidgets

_TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
_TILE_SIZE = 256
_MAX_LAT = 85.05112878
_USER_AGENT = "gnss-sim/0.2 (https://example.invalid)"
DEFAULT_TILE_CACHE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tile_cache")


def lonlat_to_pixel(lon: float, lat: float, z: int) -> tuple[float, float]:
    n = float(2 ** z)
    x = (lon + 180.0) / 360.0 * n * _TILE_SIZE
    lat = max(min(lat, _MAX_LAT), -_MAX_LAT)
    lat_r = math.radians(lat)
    y = (1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) \
        / 2.0 * n * _TILE_SIZE
    return x, y


def pixel_to_lonlat(x: float, y: float, z: int) -> tuple[float, float]:
    n = float(2 ** z)
    lon = x / (_TILE_SIZE * n) * 360.0 - 180.0
    lat_r = math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / (_TILE_SIZE * n))))
    return lon, math.degrees(lat_r)


class TileCache:
    """Disk cache for OSM tiles."""

    def __init__(self, root: str = DEFAULT_TILE_CACHE) -> None:
        self.root = root

    def path(self, z: int, x: int, y: int) -> str:
        return os.path.join(self.root, str(z), str(x), f"{y}.png")

    def ensure(self, z: int, x: int, y: int, timeout: float = 8.0) -> bool:
        dest = self.path(z, x, y)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            return True
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        try:
            req = urllib.request.Request(
                _TILE_URL.format(z=z, x=x, y=y),
                headers={"User-Agent": _USER_AGENT})
            data = urllib.request.urlopen(req, timeout=timeout).read()
            tmp = dest + ".part"
            with open(tmp, "wb") as fh:
                fh.write(data)
            os.replace(tmp, dest)
            return True
        except Exception:  # noqa: BLE001
            return False

    def pixmap(self, z: int, x: int, y: int) -> QtGui.QPixmap | None:
        pix = QtGui.QPixmap()
        if pix.load(self.path(z, x, y)):
            return pix
        return None


class _TileDone(QtCore.QObject):
    loaded = QtCore.pyqtSignal(int, int, int)


class _FetchTask(QtCore.QRunnable):
    def __init__(self, cache: TileCache, z: int, x: int, y: int,
                 done: _TileDone) -> None:
        super().__init__()
        self._cache, self._z, self._x, self._y, self._done = cache, z, x, y, done

    def run(self) -> None:  # pragma: no cover - thread
        if self._cache.ensure(self._z, self._x, self._y):
            self._done.loaded.emit(self._z, self._x, self._y)


class OsmMap(QtWidgets.QWidget):
    """Interactive OSM map with a polyline track editor."""

    trackChanged = QtCore.pyqtSignal(list)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(360, 300)
        self.setMouseTracking(True)
        self.z = 14
        self.center_lat = 35.681298
        self.center_lon = 139.766247
        self.track: list[tuple[float, float]] = []
        self.marker: tuple[float, float] | None = None
        self.truck: tuple[float, float] | None = None
        self._cache = TileCache()
        self._pix: dict[tuple[int, int, int], QtGui.QPixmap] = {}
        self._done = _TileDone()
        self._done.loaded.connect(self._on_tile_loaded)
        self._pending: set[tuple[int, int, int]] = set()
        self._pool = QtCore.QThreadPool.globalInstance()
        self._drag_pos: QtCore.QPoint | None = None
        self._press_center: tuple[float, float] | None = None

    # ------------------------------------------------------------------
    def set_center(self, lat: float, lon: float, z: int | None = None) -> None:
        self.center_lat, self.center_lon = float(lat), float(lon)
        if z is not None:
            self.z = max(1, min(19, int(z)))
        self.update()

    def set_track(self, points: list[tuple[float, float]]) -> None:
        self.track = [(float(a), float(b)) for a, b in points]
        self.update()

    def clear_track(self) -> None:
        self.track = []
        self.truck = None
        self.update()
        self.trackChanged.emit([])

    def set_truck(self, lat: float | None, lon: float | None) -> None:
        self.truck = None if lat is None else (float(lat), float(lon))
        self.update()

    # ------------------------------------------------------------------
    def _center_px(self) -> tuple[float, float]:
        return lonlat_to_pixel(self.center_lon, self.center_lat, self.z)

    def _widget_to_lonlat(self, sx: float, sy: float) -> tuple[float, float]:
        cx, cy = self._center_px()
        ox = cx - self.width() / 2.0
        oy = cy - self.height() / 2.0
        return pixel_to_lonlat(ox + sx, oy + sy, self.z)

    def _lonlat_to_widget(self, lon: float, lat: float) -> tuple[float, float]:
        wx, wy = lonlat_to_pixel(lon, lat, self.z)
        cx, cy = self._center_px()
        return wx - (cx - self.width() / 2.0), wy - (cy - self.height() / 2.0)

    # ------------------------------------------------------------------
    def _on_tile_loaded(self, z: int, x: int, y: int) -> None:
        self._pending.discard((z, x, y))
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor(210, 210, 210))

        cx, cy = self._center_px()
        ox = cx - self.width() / 2.0
        oy = cy - self.height() / 2.0
        n = 2 ** self.z
        x0 = int(math.floor(ox / _TILE_SIZE))
        y0 = int(math.floor(oy / _TILE_SIZE))
        x1 = int(math.floor((ox + self.width()) / _TILE_SIZE))
        y1 = int(math.floor((oy + self.height()) / _TILE_SIZE))
        queued = 0
        for tx in range(x0, x1 + 1):
            for ty in range(y0, y1 + 1):
                if not (0 <= ty < n):
                    continue
                wx = tx % n
                key = (self.z, wx, ty)
                pix = self._pix.get(key)
                if pix is None:
                    pix = self._cache.pixmap(*key)
                    if pix is not None:
                        self._pix[key] = pix
                px = int(tx * _TILE_SIZE - ox)
                py = int(ty * _TILE_SIZE - oy)
                if pix is not None:
                    painter.drawPixmap(px, py, pix)
                else:
                    painter.fillRect(px, py, _TILE_SIZE, _TILE_SIZE,
                                     QtGui.QColor(190, 190, 190))
                    if key not in self._pending and queued < 16:
                        self._pending.add(key)
                        self._pool.start(_FetchTask(self._cache, *key, self._done))
                        queued += 1

        # track
        if len(self.track) >= 2:
            pts = [self._lonlat_to_widget(lon, lat) for lat, lon in self.track]
            painter.setPen(QtGui.QPen(QtGui.QColor(200, 30, 30), 3))
            for i in range(len(pts) - 1):
                painter.drawLine(QtCore.QPointF(*pts[i]), QtCore.QPointF(*pts[i + 1]))
        for i, (lat, lon) in enumerate(self.track):
            sx, sy = self._lonlat_to_widget(lon, lat)
            color = QtGui.QColor(0, 120, 0) if i == 0 else (QtGui.QColor(0, 0, 200)
                                                           if i < len(self.track) - 1
                                                           else QtGui.QColor(200, 0, 0))
            painter.setBrush(color)
            painter.setPen(QtGui.QPen(QtCore.Qt.white, 1))
            painter.drawEllipse(QtCore.QPointF(sx, sy), 5, 5)

        if self.truck is not None:
            sx, sy = self._lonlat_to_widget(self.truck[1], self.truck[0])
            painter.setBrush(QtGui.QColor(255, 140, 0))
            painter.setPen(QtGui.QPen(QtCore.Qt.black, 2))
            painter.drawEllipse(QtCore.QPointF(sx, sy), 7, 7)

        painter.setPen(QtGui.QColor(20, 20, 20))
        painter.drawText(8, 18, f"z{self.z}  {self.center_lat:.5f}, "
                                f"{self.center_lon:.5f}  точек: {len(self.track)}")
        painter.drawText(8, self.height() - 8,
                         "ЛКМ: добавить точку · ПКМ: убрать · перетаскивание: сдвиг · колесо: зум")

    # ------------------------------------------------------------------
    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if event.button() == QtCore.Qt.LeftButton:
            self._drag_pos = event.pos()
            self._press_center = (self.center_lat, self.center_lon)
        elif event.button() == QtCore.Qt.RightButton:
            if self.track:
                self.track.pop()
                self.update()
                self.trackChanged.emit(list(self.track))

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if self._drag_pos is None or self._press_center is None:
            return
        dx = event.pos().x() - self._drag_pos.x()
        dy = event.pos().y() - self._drag_pos.y()
        if abs(dx) < 3 and abs(dy) < 3:
            return
        cx, cy = lonlat_to_pixel(self._press_center[1], self._press_center[0], self.z)
        lon, lat = pixel_to_lonlat(cx - dx, cy - dy, self.z)
        self.center_lat, self.center_lon = lat, lon
        self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if event.button() != QtCore.Qt.LeftButton:
            return
        moved = (self._drag_pos is not None
                 and (event.pos() - self._drag_pos).manhattanLength() > 4)
        self._drag_pos = None
        self._press_center = None
        if not moved:
            lon, lat = self._widget_to_lonlat(event.pos().x(), event.pos().y())
            self.track.append((lat, lon))
            self.update()
            self.trackChanged.emit(list(self.track))

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:  # noqa: N802
        cursor = event.pos()
        lon0, lat0 = self._widget_to_lonlat(cursor.x(), cursor.y())
        steps = event.angleDelta().y() / 120.0
        self.z = max(1, min(19, self.z + (1 if steps > 0 else -1)))
        wx, wy = lonlat_to_pixel(lon0, lat0, self.z)
        cx = wx - cursor.x() + self.width() / 2.0
        cy = wy - cursor.y() + self.height() / 2.0
        self.center_lon, self.center_lat = pixel_to_lonlat(cx, cy, self.z)
        self.update()
