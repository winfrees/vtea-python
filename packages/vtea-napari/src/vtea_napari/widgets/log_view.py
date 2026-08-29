"""The status/log strip at the bottom of a dock widget.

A plain QLabel was doing this job, and a QLabel does not wrap: a long
message - a step's traceback text, a wired-up context key list - stretched
the whole dock sideways to fit on one line. This wraps instead, keeps the
history rather than replacing it, and caps its own height at a fraction of
the pane so a burst of messages scrolls inside the strip instead of
squeezing the plot out of view.

`setText`/`text` are kept as the API so it drops in where a QLabel was.
"""

from __future__ import annotations

from qtpy.QtWidgets import QPlainTextEdit, QWidget

# The log is a footnote, not a panel: at most this share of the height of
# the widget it sits in.
MAX_HEIGHT_FRACTION = 0.10
# ...but never so short that the newest line is invisible.
MINIMUM_HEIGHT = 34


class LogView(QPlainTextEdit):
    """Read-only, word-wrapped, scrollable message log."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.setMinimumHeight(MINIMUM_HEIGHT)
        self.setMaximumHeight(MINIMUM_HEIGHT)
        self.setPlaceholderText("Messages appear here.")

    # setText/text keep the QLabel API this replaces, hence the Qt spelling.
    def setText(self, text: str) -> None:
        """Append a message and scroll to it."""
        if not text:
            return
        self.appendPlainText(text)
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

    def text(self) -> str:
        return self.toPlainText()

    def clear_log(self) -> None:
        self.setPlainText("")

    def apply_height_budget(self, available_height: int) -> None:
        """Cap the strip at MAX_HEIGHT_FRACTION of the pane it lives in.

        Called from the owner's resizeEvent - a widget can't read its own
        parent's final height reliably during construction.
        """
        budget = max(int(available_height * MAX_HEIGHT_FRACTION), MINIMUM_HEIGHT)
        self.setMaximumHeight(budget)
