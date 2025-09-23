from __future__ import annotations
from typing import Optional, List, Dict

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QListWidget, QListWidgetItem, QMessageBox
)

try:
    import curiosity_drive_node as cdn  # provides disciplines list and subtopics dict
except Exception:
    cdn = None  # type: ignore


class DebateTopicDialog(QDialog):
    """Dialog to pick discipline/subtopic and fetch 20 controversial questions via LLM."""

    def __init__(self, *, lang: str, llm_generate_text, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Debate Topic")
        self.lang = (lang or "en").lower()
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
        prompt = self._build_prompt(disc, sub)
        try:
            text = self.llm_generate_text(prompt)
        except Exception as e:
            QMessageBox.critical(self, "LLM", f"Generation failed: {e}")
            return
        items = [ln.strip(" -\t") for ln in (text or "").splitlines() if ln.strip()]
        # Keep exactly 20 where possible
        if len(items) > 20:
            items = items[:20]
        self.list_widget.clear()
        for q in items:
            QListWidgetItem(q, self.list_widget)
        if not items:
            QMessageBox.information(self, "LLM", "No questions generated.")

    def on_accept(self):
        it = self.list_widget.currentItem()
        if it is None:
            QMessageBox.information(self, "Debate", "Pick a question from the list.")
            return
        self.selected_question = it.text()
        self.accept()

    def _build_prompt(self, disc: str, sub: str) -> str:
        if self.lang == 'ru':
            return f"Выбери наиболее спорные вопросы по дисциплине {disc} и подтеме {sub}. Выбери ровно 20 вопросов. Выведи каждый вопрос с новой строки."
        else:
            return f"Pick the most controversial questions for the discipline {disc} and subtopic {sub}. Choose exactly 20 questions. Output one question per line."

