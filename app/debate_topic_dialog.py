from __future__ import annotations
from typing import Optional, List, Dict

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QListWidget, QListWidgetItem, QMessageBox,
    QSpinBox, QPlainTextEdit
)

try:
    import curiosity_drive_node as cdn  # provides disciplines list and subtopics dict
except Exception:
    cdn = None  # type: ignore


class DebateTopicDialog(QDialog):
    """Dialog to pick discipline/subtopic and fetch 20 controversial questions via LLM."""

    def __init__(self, *, lang: str, llm_generate_text, parent=None, lang_name: Optional[str] = None):
        super().__init__(parent)
        self.setWindowTitle("Debate Topic")
        # Make dialog wider/taller by default
        try:
            self.setMinimumWidth(820)
            self.setMinimumHeight(560)
        except Exception:
            pass
        self.lang = (lang or "en").lower()
        # Human-readable language name for LLM directive
        self.lang_name = lang_name or ("Русский" if self.lang == 'ru' else "English")
        self.llm_generate_text = llm_generate_text  # callable(prompt:str)->str
        self.selected_question: Optional[str] = None

        self.disciplines: List[str] = []
        self.subtopics_map: Dict[str, List[str]] = {}
        if cdn is not None:
            try:
                self.disciplines = list(getattr(cdn, 'disciplines', []))
                self.subtopics_map = dict(getattr(cdn, 'subtopics', {}))
            except Exception:
                pass

        layout = QVBoxLayout(self)

        top = QHBoxLayout()
        layout.addLayout(top)
        self.disc_label = QLabel("Discipline")
        self.disc_combo = QComboBox()
        self.disc_combo.addItems(self.disciplines or [])
        self.disc_combo.currentIndexChanged.connect(self._on_disc_changed)
        top.addWidget(self.disc_label)
        top.addWidget(self.disc_combo)
        # Count spinbox
        self.count_label = QLabel("Count")
        self.count_spin = QSpinBox()
        self.count_spin.setRange(1, 100)
        self.count_spin.setValue(20)
        top.addWidget(self.count_label)
        top.addWidget(self.count_spin)

        sub_row = QHBoxLayout()
        layout.addLayout(sub_row)
        self.sub_label = QLabel("Subtopic")
        self.sub_combo = QComboBox()
        sub_row.addWidget(self.sub_label)
        sub_row.addWidget(self.sub_combo)

        gen_row = QHBoxLayout()
        layout.addLayout(gen_row)
        self.gen_btn = QPushButton("Generate")
        self.gen_btn.clicked.connect(self.on_generate)
        gen_row.addStretch(1)
        gen_row.addWidget(self.gen_btn)

        # Custom user question
        custom_row = QHBoxLayout()
        layout.addLayout(custom_row)
        self.custom_label = QLabel("Custom question / context")
        self.custom_edit = QPlainTextEdit()
        custom_row.addWidget(self.custom_label)
        custom_row.addWidget(self.custom_edit)

        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget, stretch=1)

        btn_row = QHBoxLayout()
        layout.addLayout(btn_row)
        self.ok_btn = QPushButton("OK")
        self.ok_btn.clicked.connect(self.on_accept)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        btn_row.addStretch(1)
        btn_row.addWidget(self.ok_btn)
        btn_row.addWidget(self.cancel_btn)

        # initial subtopics
        self._on_disc_changed(self.disc_combo.currentIndex())

    def _on_disc_changed(self, idx: int):
        disc = self.disc_combo.currentText()
        subs = self.subtopics_map.get(disc, [])
        self.sub_combo.clear()
        if subs:
            self.sub_combo.addItems(subs)

    def on_generate(self):
        disc = self.disc_combo.currentText().strip()
        sub = self.sub_combo.currentText().strip()
        if not disc or not sub:
            QMessageBox.warning(self, "Debate", "Select discipline and subtopic")
            return
        n = int(self.count_spin.value())
        prompt = self._build_prompt(disc, sub, n)
        try:
            text = self.llm_generate_text(prompt)
        except Exception as e:
            QMessageBox.critical(self, "LLM", f"Generation failed: {e}")
            return
        items = [ln.strip(" -\t") for ln in (text or "").splitlines() if ln.strip()]
        # Keep exactly N where possible
        if len(items) > n:
            items = items[:n]
        self.list_widget.clear()
        for q in items:
            QListWidgetItem(q, self.list_widget)
        if not items:
            QMessageBox.information(self, "LLM", "No questions generated.")

    def on_accept(self):
        custom = (self.custom_edit.toPlainText() or "").strip()
        if custom:
            self.selected_question = custom
        else:
            it = self.list_widget.currentItem()
            if it is None:
                QMessageBox.information(self, "Debate", "Pick a question from the list or write a custom one.")
                return
            self.selected_question = it.text()
        self.accept()

    def _build_prompt(self, disc: str, sub: str, n: int) -> str:
        prefix = f"Respond strictly in {self.lang_name}.\n\n"
        if self.lang == 'ru':
            body = (
                f"Выбери наиболее спорные вопросы по дисциплине {disc} и подтеме {sub}. "
                f"Выбери ровно {n} вопросов. Выведи каждый вопрос с новой строки."
            )
        else:
            body = (
                f"Pick the most controversial questions for the discipline {disc} and subtopic {sub}. "
                f"Choose exactly {n} questions. Output one question per line."
            )
        return prefix + body
