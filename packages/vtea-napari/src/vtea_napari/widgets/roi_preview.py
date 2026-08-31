"""Running the protocol over what is on screen instead of over the dataset.

Tuning a threshold on forty gigabytes by running the whole protocol is not
tuning, it is waiting. What a user actually wants is the answer for the part
they are looking at, now, and to keep panning and zooming while they think.

Three things make that honest rather than merely fast, and each is a way the
obvious version would mislead:

- **The preview is a tile of the protocol's own tiling.** The visible region
  is grown by the same halo every other tile gets, run, and trimmed back -
  so the preview *is* what a full run would write there, rather than the
  answer a filter gives when it can see nothing beyond the edge of the view.
  A preview that disagrees with the run at the edges is worse than none,
  because the edges are what gets looked at.
- **It runs on the level being displayed**, not on level 0. Reading full
  resolution for a view that is showing every eighth voxel is exactly the
  I/O the pyramid exists to avoid. The layer says which level it used, so a
  preview computed on a coarse level is never mistaken for the real answer.
- **It is a preview and says so.** Its layers are named `preview: ...`, it
  never touches the run context, and it is replaced rather than
  accumulated.

Plus the thing that decides whether it is usable at all: panning must not
queue a hundred runs. Every view change restarts one timer, and only a view
that has been still fires it.
"""

from __future__ import annotations

import numpy as np
from qtpy.QtCore import QTimer, Signal
from qtpy.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QWidget

# What a preview layer is called. The prefix is load-bearing: a preview sits
# in the layer list beside committed results, and one that looked like a
# result would be exported, gated and reported as one.
PREVIEW_PREFIX = "preview: "

# How long the view has to be still before a preview runs. Long enough that
# a pan is one run rather than forty, short enough to feel like a response.
DEFAULT_DELAY_MS = 400


class Region:
    """Where a preview runs: a box, at a pyramid level, in that level's own
    coordinates.

    `scale` and `translate` place the result back on the image, which is
    what lets a preview computed on level 2 sit exactly over the level 0
    data the viewer is showing.
    """

    __slots__ = ("core", "level", "scale")

    def __init__(self, level: int, core: tuple[slice, ...], scale: tuple[float, ...]):
        self.level = int(level)
        self.core = tuple(core)
        self.scale = tuple(float(value) for value in scale)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(part.stop - part.start for part in self.core)

    @property
    def n_voxels(self) -> int:
        return int(np.prod(self.shape)) if self.shape else 0

    def translate(self, origin) -> list[float]:
        """Where a block read at `origin` (this level's coordinates) belongs
        in the viewer's."""
        return [start * step for start, step in zip(origin, self.scale)]

    def describe(self) -> str:
        box = "x".join(str(size) for size in self.shape)
        return f"{box} at level {self.level}" if self.level else f"{box}"

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, Region)
            and self.level == other.level
            and self.core == other.core
        )

    def __repr__(self) -> str:
        return f"Region(level={self.level}, core={self.core})"


def levels_of(layer) -> list:
    """A layer's data as a list of pyramid levels, finest first."""
    data = getattr(layer, "data", None)
    if data is None:
        return []
    if getattr(layer, "multiscale", False) or isinstance(data, (list, tuple)):
        return list(data)
    return [data]


def displayed_level(layer) -> int:
    """Which pyramid level the viewer is currently drawing.

    napari picks it from the zoom, so this is the level whose voxels are
    actually on screen - and therefore the one a preview should read.
    """
    level = getattr(layer, "data_level", 0)
    try:
        return max(0, int(level))
    except (TypeError, ValueError):
        return 0


def level_scale(layer, level: int, ndim: int) -> tuple[float, ...]:
    """How many level-0 voxels one voxel of `level` covers, per axis.

    Read from the layer rather than assumed to be two: a store written by
    another tool may downsample by three, or differently per axis, and a
    preview placed with the wrong factor sits somewhere the user is not
    looking.
    """
    factors = getattr(layer, "downsample_factors", None)
    try:
        row = np.asarray(factors)[level]
        return tuple(float(value) for value in row[:ndim])
    except (TypeError, IndexError, ValueError):
        return (1.0,) * ndim


def visible_region(layer, *, whole_axes: tuple[int, ...] = ()) -> Region | None:
    """The part of `layer` currently on screen, in the displayed level's
    coordinates.

    `whole_axes` are taken entire rather than cropped - the channel axis,
    because a step that measures every channel needs them all and slicing
    one out changes what the protocol computes rather than only where.

    An axis the viewer is not displaying (z, while looking at one plane)
    comes back as the single plane on screen, which is the honest reading of
    "what is being looked at". The halo added around it afterwards is what
    keeps a 3D step's answer for that plane correct.
    """
    corners = getattr(layer, "corner_pixels", None)
    if corners is None:
        return None
    corners = np.asarray(corners)
    if corners.ndim != 2 or corners.shape[0] != 2 or not corners.shape[1]:
        return None

    level = displayed_level(layer)
    levels = levels_of(layer)
    if not levels or level >= len(levels):
        return None
    shape = tuple(levels[level].shape)
    ndim = len(shape)
    if corners.shape[1] != ndim:
        return None

    core = []
    for axis, (low, high) in enumerate(zip(corners[0], corners[1])):
        if axis in whole_axes:
            core.append(slice(0, shape[axis]))
            continue
        start = max(0, int(low))
        # napari's corners are inclusive, and an axis that is not being
        # displayed reports the same index twice - one plane, not none.
        stop = min(int(high) + 1, shape[axis])
        core.append(slice(start, max(stop, start + 1)))
    region = Region(level, tuple(core), level_scale(layer, level, ndim))
    return region if region.n_voxels else None


class PreviewControl(QWidget):
    """The switch, the debounce, and one line saying what happened.

    Owns no data and runs nothing itself: it says *when* to preview, and the
    builder - which knows the protocol, the budget and the layer - says
    what that means.
    """

    requested = Signal()

    def __init__(self, parent: QWidget | None = None, delay_ms: int = DEFAULT_DELAY_MS):
        super().__init__(parent)
        self.viewer = None
        self._connected = False

        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(int(delay_ms))
        self.timer.timeout.connect(self._fire)

        self.checkbox = QCheckBox("Preview the view")
        self.checkbox.setToolTip(
            "Run the protocol over the region on screen, at the resolution on screen. "
            "The result is a preview layer, not a result: it is replaced as you move, "
            "and it is not what the plot or the tables read."
        )
        self.checkbox.toggled.connect(self._on_toggled)

        self.status = QLabel("")
        self.status.setWordWrap(True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.checkbox)
        layout.addWidget(self.status, 1)

    # -- wiring -----------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self.checkbox.isChecked()

    def attach(self, viewer) -> None:
        """Follow this viewer's camera and slider, while switched on."""
        self.viewer = viewer
        if self.enabled:
            self._connect()

    def set_status(self, text: str) -> None:
        self.status.setText(text)

    def request(self) -> None:
        """Ask for a preview once the view has been still for the delay.

        Restarting the timer rather than queueing is the whole point: a pan
        emits a camera event per frame, and each one should postpone the
        run rather than schedule another.
        """
        if self.enabled:
            self.timer.start()

    def _fire(self) -> None:
        if self.enabled:
            self.requested.emit()

    def _on_toggled(self, checked: bool) -> None:
        if checked:
            self._connect()
            self.request()
        else:
            self._disconnect()
            self.timer.stop()
            self.status.setText("")

    def _connect(self) -> None:
        if self.viewer is None or self._connected:
            return
        self.viewer.camera.events.zoom.connect(self._on_view_changed)
        self.viewer.camera.events.center.connect(self._on_view_changed)
        self.viewer.dims.events.current_step.connect(self._on_view_changed)
        self._connected = True

    def _disconnect(self) -> None:
        if self.viewer is None or not self._connected:
            return
        for event in (
            self.viewer.camera.events.zoom,
            self.viewer.camera.events.center,
            self.viewer.dims.events.current_step,
        ):
            try:
                event.disconnect(self._on_view_changed)
            except (ValueError, TypeError):  # already gone; nothing to undo
                pass
        self._connected = False

    def _on_view_changed(self, _event=None) -> None:
        self.request()
