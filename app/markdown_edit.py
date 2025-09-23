from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QTextCursor
from PyQt6.QtWidgets import QTextEdit, QMenu


class MarkdownTextEdit(QTextEdit):
    """
    QTextEdit-based widget that understands Markdown.

    - setPlainText(text): treats input as Markdown when render mode is on.
    - toPlainText(): returns the original Markdown source text.
    - appendPlainText(text): appends to the Markdown source and re-renders.
    - setRenderMarkdown(True/False): toggles preview mode (render vs raw markdown).

    This class mimics the key QPlainTextEdit API used in the app so it can be
    swapped in with minimal changes.
    """

    def __init__(self, render_markdown: bool = True, parent=None):
        super().__init__(parent)
        self._raw_text: str = ""
        self._render_markdown: bool = render_markdown
        self._updating: bool = False
        # For typing user content as plain Markdown without rich formatting tags
        self.setAcceptRichText(False)
        # Track edits only when not rendering (raw markdown edit mode)
        self.textChanged.connect(self._on_text_changed)

    # ---- Public controls ----
    def setRenderMarkdown(self, enabled: bool):
        self._render_markdown = bool(enabled)
        # Re-render or show raw based on current mode
        self._refresh_view()

    def isRenderMarkdown(self) -> bool:
        return self._render_markdown

    # ---- QPlainTextEdit-like overrides ----
    def setPlainText(self, text: str) -> None:  # type: ignore[override]
        self._raw_text = text or ""
        self._refresh_view()

    def toPlainText(self) -> str:  # type: ignore[override]
        # Return original Markdown source rather than rendered/plain version
        return self._raw_text

    def appendPlainText(self, text: str) -> None:
        self._raw_text += (text or "")
        self._refresh_view()
        try:
            self.moveCursor(QTextCursor.MoveOperation.End)
        except Exception:
            pass

    def clear(self) -> None:  # type: ignore[override]
        self._raw_text = ""
        super().clear()

    def setReadOnly(self, ro: bool) -> None:  # type: ignore[override]
        super().setReadOnly(ro)
        if ro and not self._render_markdown:
            # Read-only areas should default to markdown preview
            self.setRenderMarkdown(True)

    # ---- Internals ----
    def _refresh_view(self):
        self._updating = True
        try:
            if self._render_markdown:
                # Render markdown into the widget
                super().setMarkdown(self._raw_text)
            else:
                # Show raw markdown for editing
                super().setPlainText(self._raw_text)
        finally:
            self._updating = False

    def _on_text_changed(self):
        # When in raw-edit mode, keep source in sync with the widget text
        if self._updating:
            return
        if not self._render_markdown:
            try:
                self._raw_text = super().toPlainText()
            except Exception:
                pass

    # ---- Context menu: add toggle action ----
    def contextMenuEvent(self, event):  # type: ignore[override]
        menu: QMenu = self.createStandardContextMenu()
        menu.addSeparator()
        toggle = menu.addAction(
            "Preview Markdown" if not self._render_markdown else "Edit Raw Markdown"
        )
        action = menu.exec(event.globalPos())
        if action == toggle:
            self.setRenderMarkdown(not self._render_markdown)

