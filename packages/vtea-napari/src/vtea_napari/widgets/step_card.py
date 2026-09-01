"""A single pipeline step rendered as a card: thumbnail, position, name,
parameter summary, Run/Edit/Delete buttons and a progress bar.

Renders the same information vtea.protocol.blockstepgui's
AbstractMicroBlockStepGUI-derived cards showed (position number, headline,
comment, Edit/Delete) as a plain qtpy widget instead of hand-built Swing,
plus a thumbnail preview of that step's last-run output (see
protocol_builder.py's run_pipeline(), which threads a Pipeline.run()
result's per-step arrays back onto each card via the shared
thumbnail.array_to_pixmap() helper the gallery view also uses).

Every card carries its own progress bar, under its buttons and no wider or
taller than they are: a card is where a step is started, so it is where the
step should say how far along it is. A step whose duration follows from the
size of its input gets a real fraction and a countdown (see
vtea_core.workflow.cost); one whose does not - t-SNE, UMAP, a Leiden
partition - gets a continuous bar rather than a fraction that would have to
be invented. The bar is deliberately small: it reports on a step, it is not
the card's subject.
"""

from __future__ import annotations

import numpy as np
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from vtea_core.workflow import Step, format_duration

from vtea_napari.widgets.thumbnail import array_to_pixmap, max_projection

_THUMBNAIL_SIZE = 48

# How tall the progress bar may be. Capped against the buttons' own height
# as well (see _fit_progress_to_buttons), so this is an upper bound rather
# than the number: on a large-font desktop the buttons grow and the bar
# stays a slim line under them.
PROGRESS_HEIGHT = 8

# The resolution of the bar's fraction. Qt progress bars are integer-valued,
# and a thousand steps is finer than any display can show.
PROGRESS_STEPS = 1000


def summarize_params(step: Step) -> str:
    if not step.params:
        return "(default parameters)"
    return ", ".join(f"{key}={value}" for key, value in step.params.items())


def summarize_channel(step: Step) -> str:
    """Which channel this step runs on, shown on the card so a per-step
    channel choice is visible without opening the Edit dialog.

    A step that consumes the measured feature table rather than the image -
    clustering, reduction, gating - says so instead: it has no channel, and
    labelling it "all channels" would suggest it reads the image.
    """
    if step.feature_input is not None:
        if not step.features:
            return "all features"
        return f"{len(step.features)} feature(s)"
    if not step.channel_applies:
        # A label image or a per-object table: single-channel either way, so
        # there is nothing to pick.
        return "no channel"
    return "all channels" if step.channel is None else f"channel {step.channel}"


def summarize_inputs(step: Step) -> str:
    """Inputs this step has been pointed at a named result rather than the
    shared default - e.g. `labels <- watershed_split_2`. Empty when every
    input is on its default, which is the common case and would otherwise
    add a line of noise to every card."""
    redirected = [
        f"{argument} ← {key}" for argument, key in step.input_keys.items() if argument != key
    ]
    return ", ".join(redirected)


class StepCardWidget(QFrame):
    """Emits edit_requested/delete_requested; the parent widget owns the Pipeline."""

    edit_requested = Signal()
    delete_requested = Signal()
    run_requested = Signal()

    def __init__(self, position: int, step: Step, parent=None, thumbnail: np.ndarray | None = None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)

        outer = QHBoxLayout(self)

        # What this step was expected to take, once it is running - kept so
        # the bar can say how much is left rather than only how far it is.
        self._estimate: float | None = None

        self.thumbnail_label = QLabel()
        self.thumbnail_label.setFixedSize(_THUMBNAIL_SIZE, _THUMBNAIL_SIZE)
        self.set_thumbnail(thumbnail)
        outer.addWidget(self.thumbnail_label)

        position_label = QLabel(f"{position}.")
        outer.addWidget(position_label)

        text_column = QVBoxLayout()
        # The name leads: it's how other steps refer to this one's result, so
        # it's the thing to read off the card when wiring a measurement step
        # to one of several segmentations.
        self.name_label = QLabel(step.name or f"{step.category}.{step.function_name}")
        self.name_label.setStyleSheet("font-weight: bold;")
        text_column.addWidget(self.name_label)
        if step.name:
            function_label = QLabel(f"{step.category}.{step.function_name}")
            function_label.setStyleSheet("color: gray;")
            text_column.addWidget(function_label)
        comment_text = step.comment if step.comment else summarize_params(step)
        text_column.addWidget(QLabel(comment_text))
        sources = summarize_inputs(step)
        self.channel_label = QLabel(
            f"{summarize_channel(step)} · {sources}" if sources else summarize_channel(step)
        )
        self.channel_label.setStyleSheet("color: gray;")
        text_column.addWidget(self.channel_label)
        outer.addLayout(text_column)

        outer.addStretch()

        # Run this one step on its own, against whatever the pipeline has
        # produced so far, and display the result. Analysis steps are not a
        # chain - measurements feeds clustering, clustering feeds back in as
        # a feature - so each needs to be runnable independently.
        self.run_button = QPushButton("Run")
        self.run_button.setToolTip("Run just this step and show its result")
        self.run_button.clicked.connect(self.run_requested.emit)

        self.edit_button = QPushButton("Edit")
        self.edit_button.clicked.connect(self.edit_requested.emit)

        self.delete_button = QPushButton("Delete")
        self.delete_button.clicked.connect(self.delete_requested.emit)

        # The buttons and the bar are one column on the right of the card,
        # which is what keeps the bar tied to the step it reports on - and
        # what its size is measured against.
        controls = QWidget()
        control_column = QVBoxLayout(controls)
        control_column.setContentsMargins(0, 0, 0, 0)
        control_column.setSpacing(2)
        button_row = QHBoxLayout()
        button_row.setContentsMargins(0, 0, 0, 0)
        button_row.setSpacing(2)
        for button in (self.run_button, self.edit_button, self.delete_button):
            button_row.addWidget(button)
        control_column.addLayout(button_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setRange(0, PROGRESS_STEPS)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        control_column.addWidget(self.progress_bar)
        self._fit_progress_to_buttons()
        outer.addWidget(controls)

    # -- progress ---------------------------------------------------------

    def _fit_progress_to_buttons(self) -> None:
        """Keep the bar within the three buttons: no wider than they are
        together, no taller than one of them.

        Sized from the buttons' own size hints rather than a hard-coded
        number of pixels, so it still holds at a different font size or on a
        high-DPI display - which is the only way "no wider than the buttons"
        can be true rather than approximately true.
        """
        buttons = (self.run_button, self.edit_button, self.delete_button)
        hints = [button.sizeHint() for button in buttons]
        combined = sum(hint.width() for hint in hints) + 2 * 2  # the row's spacing
        tallest = max(hint.height() for hint in hints)
        self.progress_bar.setMaximumWidth(combined)
        self.progress_bar.setFixedHeight(min(PROGRESS_HEIGHT, tallest))

    def begin_progress(self, estimate_seconds: float | None = None) -> None:
        """Show the bar for a step that is starting.

        `estimate_seconds` is what the step is expected to take (see
        vtea_core.workflow.estimate_seconds). None - no honest estimate -
        gives a continuous bar, Qt's own busy indicator, rather than a
        fraction nobody can stand behind.
        """
        self._estimate = estimate_seconds
        if estimate_seconds is None:
            self.set_indeterminate()
        else:
            self.progress_bar.setRange(0, PROGRESS_STEPS)
            self.progress_bar.setValue(0)
            self.progress_bar.setToolTip(f"Estimated {format_duration(estimate_seconds)}")
        self.progress_bar.setVisible(True)

    def set_indeterminate(self) -> None:
        """A continuous bar: this step is running, and how far along it is
        cannot be said."""
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setToolTip("Running - this step's duration cannot be estimated")
        self.progress_bar.setVisible(True)

    def set_determinate(self) -> None:
        """Switch a continuous bar to a measured one.

        For a step that turns out to know exactly how far along it is after
        all - a tiled run counts tiles - having started with no estimate to
        show.
        """
        self.progress_bar.setRange(0, PROGRESS_STEPS)
        self.progress_bar.setToolTip("")
        self.progress_bar.setVisible(True)

    def set_progress(self, fraction: float, remaining: float | None = None) -> None:
        """Move the bar to `fraction` (0-1), optionally naming what is left."""
        if self.progress_bar.maximum() == 0:  # indeterminate; leave it pulsing
            return
        value = int(max(0.0, min(1.0, fraction)) * PROGRESS_STEPS)
        self.progress_bar.setValue(value)
        if remaining is not None:
            self.progress_bar.setToolTip(f"{format_duration(remaining)} left")

    def end_progress(self) -> None:
        """Hide the bar again. A finished step's card shows its result, not a
        full bar that never goes away."""
        self.progress_bar.setRange(0, PROGRESS_STEPS)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setToolTip("")

    def set_thumbnail(self, array: np.ndarray | None) -> None:
        """Shows a preview of a 2D+ numeric or boolean array (e.g. this
        step's last-run output); clears the preview for anything else
        (None, a scalar, a DataFrame, a fitted model, ...)."""
        previewable = isinstance(array, np.ndarray) and array.ndim >= 2 and (
            array.dtype == bool or np.issubdtype(array.dtype, np.number)
        )
        if not previewable:
            self.thumbnail_label.clear()
            return
        pixmap = array_to_pixmap(max_projection(array), size=_THUMBNAIL_SIZE)
        self.thumbnail_label.setPixmap(pixmap)
