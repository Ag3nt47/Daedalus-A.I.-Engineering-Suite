"""Small native Python editor with highlighting, find, and line numbers."""

from __future__ import annotations

from PySide6.QtCore import QRect, QRegularExpression, QSize, Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QKeySequence,
    QPainter,
    QPalette,
    QShortcut,
    QSyntaxHighlighter,
    QTextCharFormat,
    QTextCursor,
    QTextDocument,
    QTextFormat,
)
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class PythonHighlighter(QSyntaxHighlighter):
    KEYWORDS = (
        "False None True and as assert async await break class continue def del elif else "
        "except finally for from global if import in is lambda nonlocal not or pass raise "
        "return try while with yield match case"
    ).split()

    def __init__(self, document) -> None:
        super().__init__(document)
        self.rules: list[tuple[QRegularExpression, QTextCharFormat, int]] = []

        def style(color: str, *, bold: bool = False, italic: bool = False):
            value = QTextCharFormat()
            value.setForeground(QColor(color))
            value.setFontWeight(QFont.Weight.Bold if bold else QFont.Weight.Normal)
            value.setFontItalic(italic)
            return value

        keyword = style("#38bdf8", bold=True)
        keyword_pattern = r"\b(?:" + "|".join(self.KEYWORDS) + r")\b"
        self.rules.append((QRegularExpression(keyword_pattern), keyword, 0))
        self.rules.extend(
            [
                (QRegularExpression(r"\bdef\s+(\w+)"), style("#c084fc", bold=True), 1),
                (QRegularExpression(r"\bclass\s+(\w+)"), style("#f59e0b", bold=True), 1),
                (QRegularExpression(r"@[A-Za-z_]\w*"), style("#f472b6"), 0),
                (QRegularExpression(r"\b\d+(?:\.\d+)?(?:e[+-]?\d+)?\b"), style("#fbbf24"), 0),
                (
                    QRegularExpression(r"[rubfRUBF]*'(?:[^'\\]|\\.)*'"),
                    style("#34d399"),
                    0,
                ),
                (
                    QRegularExpression(r'[rubfRUBF]*"(?:[^"\\]|\\.)*"'),
                    style("#34d399"),
                    0,
                ),
                (QRegularExpression(r"\b(?:self|cls)\b"), style("#22d3ee", italic=True), 0),
            ]
        )
        self.comment_style = style("#64748b", italic=True)

    def highlightBlock(self, text: str) -> None:  # noqa: N802 - Qt API
        for expression, formatting, capture in self.rules:
            iterator = expression.globalMatch(text)
            while iterator.hasNext():
                match = iterator.next()
                start = match.capturedStart(capture)
                length = match.capturedLength(capture)
                if start >= 0 and length > 0:
                    self.setFormat(start, length, formatting)
        comment_at = text.find("#")
        if comment_at >= 0:
            self.setFormat(comment_at, len(text) - comment_at, self.comment_style)


class LineNumberArea(QWidget):
    def __init__(self, editor: "CodeEditor") -> None:
        super().__init__(editor)
        self.editor = editor
        self.setAccessibleName("Editor line numbers")

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API
        return QSize(self.editor.line_number_area_width(), 0)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        self.editor.paint_line_numbers(event)


class CodeEditor(QPlainTextEdit):
    """A focused editor primitive; execution remains owned by SandboxRunner."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("CodeEditor")
        self.setAccessibleName("Python code editor")
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        font = QFont("Cascadia Code", 10)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.setFont(font)
        self.setTabStopDistance(4 * self.fontMetrics().horizontalAdvance(" "))
        self.line_numbers = LineNumberArea(self)
        self.highlighter = PythonHighlighter(self.document())
        self.blockCountChanged.connect(self._update_margin)
        self.updateRequest.connect(self._update_line_area)
        self.cursorPositionChanged.connect(self._highlight_current_line)
        self._update_margin()
        self._highlight_current_line()

    def line_number_area_width(self) -> int:
        digits = max(2, len(str(max(1, self.blockCount()))))
        return 10 + self.fontMetrics().horizontalAdvance("9") * digits

    def _update_margin(self, _count: int = 0) -> None:
        self.setViewportMargins(self.line_number_area_width(), 0, 0, 0)

    def _update_line_area(self, rect: QRect, dy: int) -> None:
        if dy:
            self.line_numbers.scroll(0, dy)
        else:
            self.line_numbers.update(0, rect.y(), self.line_numbers.width(), rect.height())
        if rect.contains(self.viewport().rect()):
            self._update_margin()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        contents = self.contentsRect()
        self.line_numbers.setGeometry(
            QRect(contents.left(), contents.top(), self.line_number_area_width(), contents.height())
        )

    def paint_line_numbers(self, event) -> None:
        painter = QPainter(self.line_numbers)
        base = self.palette().color(QPalette.ColorRole.Base)
        painter.fillRect(event.rect(), base.darker(112))
        block = self.firstVisibleBlock()
        number = block.blockNumber()
        top = round(self.blockBoundingGeometry(block).translated(self.contentOffset()).top())
        bottom = top + round(self.blockBoundingRect(block).height())
        muted = self.palette().color(QPalette.ColorRole.PlaceholderText)
        active = self.palette().color(QPalette.ColorRole.Highlight)
        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                painter.setPen(active if number == self.textCursor().blockNumber() else muted)
                painter.drawText(
                    0,
                    top,
                    self.line_numbers.width() - 5,
                    self.fontMetrics().height(),
                    Qt.AlignmentFlag.AlignRight,
                    str(number + 1),
                )
            block = block.next()
            top = bottom
            bottom = top + round(self.blockBoundingRect(block).height())
            number += 1

    def _highlight_current_line(self) -> None:
        if self.isReadOnly():
            self.setExtraSelections([])
            return
        selection = QTextEdit.ExtraSelection()
        color = self.palette().color(QPalette.ColorRole.Highlight)
        color.setAlpha(32)
        selection.format.setBackground(color)
        selection.format.setProperty(QTextFormat.Property.FullWidthSelection, True)
        selection.cursor = self.textCursor()
        selection.cursor.clearSelection()
        self.setExtraSelections([selection])


class CodeEditorPanel(QWidget):
    """Code editor plus a keyboard-friendly find bar."""

    HIGHLIGHT_CHARACTER_LIMIT = 384 * 1024
    HIGHLIGHT_BLOCK_LIMIT = 6_000

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        self.editor = CodeEditor()
        self.syntax_highlighting_enabled = True
        layout.addWidget(self.editor, 1)
        self.find_bar = QWidget()
        find_layout = QHBoxLayout(self.find_bar)
        find_layout.setContentsMargins(0, 0, 0, 0)
        self.find_input = QLineEdit()
        self.find_input.setPlaceholderText("Find in file…")
        self.find_input.setAccessibleName("Find text in editor")
        previous = QPushButton("Previous")
        previous.setAccessibleName("Find previous match")
        next_button = QPushButton("Next")
        next_button.setAccessibleName("Find next match")
        close = QPushButton("Close")
        close.setAccessibleName("Close find bar")
        find_layout.addWidget(self.find_input, 1)
        find_layout.addWidget(previous)
        find_layout.addWidget(next_button)
        find_layout.addWidget(close)
        layout.addWidget(self.find_bar)
        self.find_bar.hide()
        self.find_input.returnPressed.connect(lambda: self.find_next(True))
        previous.clicked.connect(lambda: self.find_next(False))
        next_button.clicked.connect(lambda: self.find_next(True))
        close.clicked.connect(self.find_bar.hide)
        QShortcut(QKeySequence.StandardKey.Find, self.editor).activated.connect(self.show_find)
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self.find_input).activated.connect(
            self.find_bar.hide
        )

    def show_find(self) -> None:
        cursor = self.editor.textCursor()
        if cursor.hasSelection():
            self.find_input.setText(cursor.selectedText())
        self.find_bar.show()
        self.find_input.setFocus()
        self.find_input.selectAll()

    def find_next(self, forward: bool = True) -> bool:
        text = self.find_input.text()
        if not text:
            return False
        flags = QTextDocument.FindFlag(0)
        if not forward:
            flags |= QTextDocument.FindFlag.FindBackward
        if self.editor.find(text, flags):
            return True
        cursor = self.editor.textCursor()
        cursor.movePosition(
            QTextCursor.MoveOperation.Start if forward else QTextCursor.MoveOperation.End
        )
        self.editor.setTextCursor(cursor)
        return self.editor.find(text, flags)

    def setPlainText(self, text: str) -> None:  # noqa: N802 - mirrors QTextEdit
        highlight = (
            len(text) <= self.HIGHLIGHT_CHARACTER_LIMIT
            and text.count("\n") < self.HIGHLIGHT_BLOCK_LIMIT
        )
        # Detaching first prevents QSyntaxHighlighter from re-running on every
        # document mutation while QPlainTextEdit ingests a large file.
        self.editor.highlighter.setDocument(None)
        self.editor.setPlainText(text)
        self.syntax_highlighting_enabled = highlight
        if highlight:
            self.editor.highlighter.setDocument(self.editor.document())
        self.editor.setAccessibleDescription(
            "Python syntax highlighting active."
            if highlight
            else "Syntax highlighting paused for this large file to keep the editor responsive."
        )

    def toPlainText(self) -> str:  # noqa: N802 - mirrors QTextEdit
        return self.editor.toPlainText()
