"""Running a long protocol without freezing napari, and stopping it.

A blocked run is measured in tiles and can be measured in hours. Run on the
Qt thread it freezes the window, and a window that has stopped repainting is
indistinguishable from one that has crashed - so the first thing a user does
is kill it.

The shape here is deliberately not `thread_worker`. That would make
`run_processing` asynchronous, and every caller and test that treats it as a
function returning a result would have to change. Instead the work goes to a
single background thread and the *calling* thread pumps the Qt event loop
while it waits: the run still returns its result to the caller, napari keeps
repainting, and Cancel is a button that can actually be clicked. NumPy and
scipy release the GIL for the heavy operations, so the pumping thread and
the working thread genuinely overlap.

Two things this must not get wrong, and both are guarded:

- **Re-entrancy.** Pumping events means a user can click things mid-run,
  including Run again. A run in progress refuses to start another.
- **Which thread touches napari.** Nothing here publishes a layer. The
  worker computes, the caller receives, and everything that talks to the
  viewer happens after `run` returns, on the thread that called it.

The second of those is what `ProgressRelay` is for. A running step has
things to say - which tile it is on, which step of the protocol is next -
and it says them from the worker thread, where touching a QWidget is not a
bug that shows up as a wrong pixel but one that takes the process down. So
the worker writes into the relay and the pump loop, which is on the GUI
thread by construction, reads it back out and draws it.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any

from qtpy.QtCore import QCoreApplication, QObject, Signal
from qtpy.QtWidgets import QApplication, QPushButton, QWidget

# How long to wait on the future between event-loop pumps. Short enough that
# a click feels immediate, long enough not to spin.
_PUMP_SECONDS = 0.05


class CancelFlag:
    """A stop request one thread sets and another reads.

    A plain flag behind a lock rather than an Event, because what the
    executor wants is a callable it can poll between tiles, and what the
    button wants is something it can set from the GUI thread.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stopped = False

    def cancel(self) -> None:
        with self._lock:
            self._stopped = True

    def reset(self) -> None:
        with self._lock:
            self._stopped = False

    def __call__(self) -> bool:
        with self._lock:
            return self._stopped

    @property
    def cancelled(self) -> bool:
        return self()


class ProgressRelay:
    """What a worker thread has to say about its progress, for the GUI to read.

    Deliberately not a Qt signal. A signal emitted from a worker thread is
    delivered whenever the receiving thread next runs its event loop, which
    for a tight tile loop means a queue of thousands of stale messages to
    drain; a relay holds only the latest, which is the only one worth
    drawing. Everything on it is guarded by a lock, because the two threads
    genuinely do touch it at once.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._message = ""
        self._fraction: float | None = None
        self._name = ""

    def report(self, message: str = "", *, fraction: float | None = None, name: str = "") -> None:
        """Called from the worker thread. Never touches a widget."""
        with self._lock:
            self._message = message
            self._fraction = fraction
            self._name = name

    def clear(self) -> None:
        self.report("", fraction=None, name="")

    def snapshot(self) -> tuple[str, float | None, str]:
        """(message, fraction, step name) as of now - read from the GUI thread."""
        with self._lock:
            return self._message, self._fraction, self._name


class RunControl(QObject):
    """Runs a callable off the GUI thread, and offers a Cancel button.

    `busy` says whether a run is in progress, which is what the caller
    checks before starting another.
    """

    started = Signal()
    finished = Signal()

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self.flag = CancelFlag()
        self._busy = False
        self.button = QPushButton("Cancel")
        self.button.setToolTip("Stop the run after the tile it is working on")
        self.button.clicked.connect(self.cancel)
        self.button.setVisible(False)

    @property
    def busy(self) -> bool:
        return self._busy

    def cancel(self) -> None:
        """Ask the run to stop. Safe to call from any thread.

        It sets the flag and nothing else. Touching the button here would
        be touching a QWidget from whichever thread happened to call - and
        Qt does not merely misbehave at that, it segfaults. The button's
        feedback is applied by the pump loop below, which is on the GUI
        thread by construction.
        """
        self.flag.cancel()

    def run(
        self,
        work: Callable[[Callable[[], bool]], Any],
        *,
        on_tick: Callable[[float], None] | None = None,
        show_cancel: bool = True,
    ) -> Any:
        """Run `work(should_stop)` on a worker thread, pumping Qt meanwhile.

        Returns whatever `work` returned, and re-raises whatever it raised -
        including `Cancelled`, which the caller is expected to treat as a
        partial result rather than a finished one.

        `on_tick(elapsed_seconds)` is called on the *calling* thread every
        pump, which is what a progress bar is driven from: it is the one
        place in a run that is both regular and safely on the GUI thread.
        Errors from it are swallowed rather than allowed to abort the run -
        a bar that failed to draw is not a reason to lose an hour of
        computation.

        `show_cancel` hides the Cancel button for a run too short to need
        one; the flag is still honoured, so a step that does poll it can
        still be stopped.
        """
        if self._busy:
            raise RuntimeError("a run is already in progress")
        self._busy = True
        self.flag.reset()
        self._show_button(show_cancel)
        self.started.emit()
        started_at = time.monotonic()
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future: Future = pool.submit(work, self.flag)
                while True:
                    self._show_stopping()
                    if on_tick is not None:
                        try:
                            on_tick(time.monotonic() - started_at)
                        except Exception:  # noqa: BLE001 - drawing must not kill the run
                            pass
                    _pump()
                    try:
                        # Doubles as the wait and the result: whatever the
                        # work raised comes back out here, cancellation
                        # included.
                        return future.result(timeout=_PUMP_SECONDS)
                    except FutureTimeout:
                        continue
        finally:
            self._busy = False
            self._show_button(False)
            self.finished.emit()

    def _show_button(self, visible: bool) -> None:
        self.button.setVisible(visible)
        self.button.setEnabled(bool(visible))
        self.button.setText("Cancel")

    def _show_stopping(self) -> None:
        """Reflect a cancellation in the button, on the GUI thread.

        Called from the pump loop rather than from `cancel`, so that a
        request arriving from a worker or a timer still shows up without
        anything but this thread touching the widget.
        """
        if self.flag.cancelled and self.button.isEnabled():
            self.button.setEnabled(False)
            self.button.setText("Stopping...")

    def widget(self) -> QWidget:
        return self.button


def _pump() -> None:
    """Let Qt repaint and deliver the Cancel click.

    A no-op where there is no application, so the same code path works in a
    headless test as in the viewer.
    """
    app = QApplication.instance() or QCoreApplication.instance()
    if app is not None:
        app.processEvents()
