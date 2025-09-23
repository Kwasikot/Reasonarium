from __future__ import annotations
import os
import traceback
from typing import List, Dict, Optional, Callable, Iterator
import re

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QTextCursor
from PyQt6.QtWidgets import (
    QWidget, QMainWindow, QVBoxLayout, QHBoxLayout, QTextBrowser, QPlainTextEdit,
    QPushButton, QLabel, QComboBox, QLineEdit, QDoubleSpinBox, QFileDialog, QSplitter,
    QGroupBox, QFormLayout, QMessageBox, QInputDialog, QTabWidget, QSpinBox, QDialog
)
from PyQt6.QtCore import QUrl

from app.prompts_loader import read_prompt
from app.settings_loader import Settings
from llm.openai_client import LLMClient as OpenAIClient
from app.audio_recorder import AudioRecorder
from app.local_stt import transcribe_whisper
from llm.ollama_client import OllamaClient
from app.debate_topic_dialog import DebateTopicDialog
try:
    import curiosity_drive_node as cdn  # disciplines/subtopics lists
except Exception:
    cdn = None  # type: ignore


class StreamWorker(QObject):
    token = pyqtSignal(str)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, gen_factory: Callable[[], Iterator[str]]):
        super().__init__()
        self._gen_factory = gen_factory

    def run(self):
        try:
            for piece in self._gen_factory():
                if piece:
                    self.token.emit(piece)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


class ChatInput(QPlainTextEdit):
    sendRequested = pyqtSignal()

    def keyPressEvent(self, event):  # type: ignore
        try:
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
                self.sendRequested.emit()
                return
        except Exception:
            pass
        super().keyPressEvent(event)


class ChatWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Reasonarium")
        self.resize(1000, 700)

        # State
        self.messages: List[Dict[str, str]] = []
        self.current_system_prompt: str = ""
        self.openai_client: Optional[OpenAIClient] = None
        self.ollama_client: Optional[OllamaClient] = None
        self._stream_thread: Optional[QThread] = None
        self._stream_worker: Optional[StreamWorker] = None
        self.settings = Settings()
        self.lang = self.settings.default_language() if self.settings.ok() else "en"
        # Topic and recording state (no QtMultimedia; we use sounddevice)
        self._last_topic = None
        self._recorder: Optional[AudioRecorder] = None
        self._last_topic: Optional[str] = None
        self.audio_source: Optional[QAudioSource] = None
        self.audio_io = None
        self.audio_wave = None
        self._record_tmp_path: Optional[str] = None

        # UI
        self._build_ui()
        self._apply_language()
        self._populate_openai_models()
        # Populate microphones initially
        try:
            self._populate_mics()
        except Exception:
            pass
        # Populate whisper models
        try:
            self._populate_whisper_models()
        except Exception:
            pass
        self._load_prompts_from_settings()

    # --- UI Construction ---
    def _build_ui(self):
        central = QWidget()
        root = QVBoxLayout(central)
        self.setCentralWidget(central)

        # Tab container
        self.tabs = QTabWidget()
        root.addWidget(self.tabs, stretch=1)

        # Chat tab content container
        chat_tab = QWidget()
        chat_tab_layout = QVBoxLayout(chat_tab)
        self.tabs.addTab(chat_tab, "Chat")

        # Controls row (in chat tab)
        controls = QHBoxLayout()
        chat_tab_layout.addLayout(controls)

        # Language selector
        self.lang_combo = QComboBox()
        # Will be populated in _apply_language()
        self.lang_combo.currentIndexChanged.connect(self.on_language_changed)
        self.lang_label = QLabel("Language:")
        controls.addWidget(self.lang_label)
        controls.addWidget(self.lang_combo)

        # Engine and model
        self.engine_combo = QComboBox()
        self.engine_combo.addItems(["OpenAI", "Ollama"])
        self.engine_combo.setCurrentIndex(0)
        self.engine_combo.currentTextChanged.connect(self.on_engine_changed)

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.editTextChanged.connect(self.on_model_changed)
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.0, 2.0)
        self.temp_spin.setSingleStep(0.1)
        self.temp_spin.setValue(1.0)

        self.engine_label = QLabel("Engine:")
        controls.addWidget(self.engine_label)
        controls.addWidget(self.engine_combo)
        self.model_label = QLabel("Model:")
        controls.addWidget(self.model_label)
        controls.addWidget(self.model_combo)
        self.temp_label = QLabel("Temp:")
        controls.addWidget(self.temp_label)
        controls.addWidget(self.temp_spin)

        # API key input (OpenAI)
        self.api_key_edit = QLineEdit(os.getenv("OPENAI_API_KEY", ""))
        self.api_key_edit.setPlaceholderText("OPENAI_API_KEY")
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_label = QLabel("API Key:")
        controls.addWidget(self.api_key_label)
        controls.addWidget(self.api_key_edit, stretch=1)

        # Optional proxy URL and trust_env toggle
        self.proxy_edit = QLineEdit(os.getenv("USEFULCLICKER_OPENAI_PROXY", ""))
        self.proxy_edit.setPlaceholderText("http(s)://user:pass@host:port (optional)")
        self.proxy_label = QLabel("Proxy:")
        controls.addWidget(self.proxy_label)
        controls.addWidget(self.proxy_edit, stretch=1)

        # Test connection button
        self.test_btn = QPushButton("Test OpenAI")
        self.test_btn.clicked.connect(self.on_test_openai)
        controls.addWidget(self.test_btn)

        # (Mode selector removed)

        # Prompts
        self.prompt_combo = QComboBox()
        self.reload_prompts_btn = QPushButton("Reload Prompts")
        self.reload_prompts_btn.clicked.connect(self._load_prompts_from_settings)
        self.prompt_label = QLabel("Prompt:")
        controls.addWidget(self.prompt_label)
        controls.addWidget(self.prompt_combo, stretch=1)
        controls.addWidget(self.reload_prompts_btn)
        self.start_session_btn = QPushButton("Start Session…")
        self.start_session_btn.clicked.connect(self.on_start_session)
        controls.addWidget(self.start_session_btn)

        # Splitter: chat and prompt preview
        splitter = QSplitter(Qt.Orientation.Vertical)
        chat_tab_layout.addWidget(splitter, stretch=1)

        # Chat area
        chat_box = QWidget()
        chat_layout = QVBoxLayout(chat_box)
        self.chat_view = QTextBrowser()
        # Increase font size for LLM output area
        try:
            self.chat_view.setStyleSheet("font-size: 14pt;")
        except Exception:
            pass
        self.chat_view.setOpenExternalLinks(True)
        self.chat_view.setReadOnly(True)
        chat_layout.addWidget(self.chat_view, stretch=1)

        # Input area
        input_row = QHBoxLayout()
        self.input_edit = ChatInput()
        self.input_edit.setPlaceholderText("Type your message…")
        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self.on_send)
        self.input_edit.sendRequested.connect(self.on_send)
        self.new_btn = QPushButton("New Chat")
        self.new_btn.clicked.connect(self.on_new_chat)
        self.auto_btn = QPushButton("Auto answer")
        self.auto_btn.clicked.connect(self.on_auto_answer)
        self.nextq_btn = QPushButton("Next question")
        self.nextq_btn.clicked.connect(self.on_next_question)
        input_row.addWidget(self.input_edit, stretch=1)
        input_row.addWidget(self.send_btn)
        input_row.addWidget(self.new_btn)
        input_row.addWidget(self.auto_btn)
        input_row.addWidget(self.nextq_btn)
        chat_layout.addLayout(input_row)
        # Microphone row: selector + Record toggle (sounddevice backend)
        mic_row = QHBoxLayout()
        self.mic_label = QLabel("Microphone")
        self.mic_combo = QComboBox()
        self.whisper_label = QLabel("Whisper")
        self.whisper_combo = QComboBox()
        self.whisper_label = QLabel("Whisper model")
        self.whisper_combo = QComboBox()
        self.record_btn = QPushButton("Record")
        self.record_btn.clicked.connect(self.on_record_toggle)
        mic_row.addWidget(self.mic_label)
        mic_row.addWidget(self.mic_combo, stretch=1)
        mic_row.addWidget(self.whisper_label)
        mic_row.addWidget(self.whisper_combo)
        mic_row.addWidget(self.record_btn)
        chat_layout.addLayout(mic_row)

        splitter.addWidget(chat_box)

        # Prompt preview panel
        self.preview_box = QGroupBox("Prompt Preview")
        pv_layout = QVBoxLayout(self.preview_box)
        self.prompt_preview = QPlainTextEdit()
        self.prompt_preview.setReadOnly(True)
        pv_layout.addWidget(self.prompt_preview)

        splitter.addWidget(self.preview_box)
        splitter.setSizes([500, 200])

        # Wire prompt selection
        self.prompt_combo.currentIndexChanged.connect(self._on_prompt_selected)

        # --- Curiosity Drive tab ---
        cd_tab = QWidget()
        cd_layout = QVBoxLayout(cd_tab)
        form = QFormLayout()
        cd_layout.addLayout(form)

        # Disciplines
        self.cd_lbl_disciplines = QLabel("Disciplines")
        self.cd_disc_combo = QComboBox()
        form.addRow(self.cd_lbl_disciplines, self.cd_disc_combo)

        # Audience / Rarity / Novelty / Count
        self.cd_lbl_audience = QLabel("Audience")
        self.cd_audience = QLineEdit("general")
        form.addRow(self.cd_lbl_audience, self.cd_audience)
        self.cd_lbl_rarity = QLabel("Rarity")
        self.cd_rarity = QLineEdit("medium-rare")
        form.addRow(self.cd_lbl_rarity, self.cd_rarity)
        self.cd_lbl_novelty = QLabel("Novelty")
        self.cd_novelty = QLineEdit("balanced")
        form.addRow(self.cd_lbl_novelty, self.cd_novelty)
        self.cd_lbl_count = QLabel("Items")
        self.cd_count = QSpinBox()
        self.cd_count.setRange(1, 50)
        self.cd_count.setValue(5)
        form.addRow(self.cd_lbl_count, self.cd_count)

        self.cd_generate_btn = QPushButton("Generate")
        self.cd_generate_btn.clicked.connect(self.on_cd_generate)
        cd_layout.addWidget(self.cd_generate_btn)

        self.cd_output = QPlainTextEdit()
        self.cd_output.setReadOnly(True)
        cd_layout.addWidget(self.cd_output, stretch=1)

        self.tabs.addTab(cd_tab, "Curiosity Drive")

        # --- Popper Challenge tab ---
        pop_tab = QWidget()
        pop_layout = QVBoxLayout(pop_tab)
        pop_controls = QHBoxLayout()
        pop_layout.addLayout(pop_controls)
        self.pop_lbl_d1 = QLabel("Discipline A")
        self.pop_d1 = QComboBox()
        self.pop_lbl_d2 = QLabel("Discipline B")
        self.pop_d2 = QComboBox()
        self.pop_syn_btn = QPushButton("Synthesize theory")
        self.pop_syn_btn.clicked.connect(self.on_popper_synthesize)
        # Levels: theory type and education level
        self.pop_lvl_theory_lbl = QLabel("Theory type")
        self.pop_lvl_theory = QComboBox()
        self.pop_lvl_edu_lbl = QLabel("Education level")
        self.pop_lvl_edu = QComboBox()
        pop_controls.addWidget(self.pop_lbl_d1)
        pop_controls.addWidget(self.pop_d1)
        pop_controls.addWidget(self.pop_lbl_d2)
        pop_controls.addWidget(self.pop_d2)
        pop_controls.addWidget(self.pop_lvl_theory_lbl)
        pop_controls.addWidget(self.pop_lvl_theory)
        pop_controls.addWidget(self.pop_lvl_edu_lbl)
        pop_controls.addWidget(self.pop_lvl_edu)
        pop_controls.addWidget(self.pop_syn_btn)

        self.pop_theory = QPlainTextEdit()
        self.pop_theory.setReadOnly(True)
        pop_layout.addWidget(self.pop_theory, stretch=1)

        self.pop_user_lbl = QLabel("Your experiments / observations")
        self.pop_user_edit = QPlainTextEdit()
        pop_layout.addWidget(self.pop_user_lbl)
        pop_layout.addWidget(self.pop_user_edit)

        # Optional AI suggestions
        from PyQt6.QtWidgets import QCheckBox
        self.pop_ai_chk = QCheckBox("Show AI suggestions")
        self.pop_ai_chk.stateChanged.connect(self.on_popper_ai_toggle)
        pop_layout.addWidget(self.pop_ai_chk)
        self.pop_ai_lbl = QLabel("AI experiments / observations")
        self.pop_ai_edit = QPlainTextEdit()
        self.pop_ai_edit.setReadOnly(True)
        self.pop_ai_lbl.setVisible(False)
        self.pop_ai_edit.setVisible(False)
        pop_layout.addWidget(self.pop_ai_lbl)
        pop_layout.addWidget(self.pop_ai_edit)

        row_eval = QHBoxLayout()
        self.pop_check_btn = QPushButton("Evaluate falsification")
        self.pop_check_btn.clicked.connect(self.on_popper_check)
        self.pop_confirm_btn = QPushButton("Confirm experiment")
        self.pop_confirm_btn.clicked.connect(self.on_popper_confirm)
        row_eval.addStretch(1)
        row_eval.addWidget(self.pop_check_btn)
        row_eval.addWidget(self.pop_confirm_btn)
        pop_layout.addLayout(row_eval)

        self.pop_result_lbl = QLabel("Evaluation")
        self.pop_result = QPlainTextEdit()
        self.pop_result.setReadOnly(True)
        pop_layout.addWidget(self.pop_result_lbl)
        pop_layout.addWidget(self.pop_result, stretch=1)

        self.tabs.addTab(pop_tab, "Popper Challenge")
        # (No subtopic comboboxes; only disciplines)

    def _load_prompts(self):
        # Fallback scanner if settings are not used
        from app.prompts_loader import list_prompt_files
        files = list_prompt_files()
        self.prompt_combo.blockSignals(True)
        self.prompt_combo.clear()
        self.prompt_combo.addItem("(none)", userData=None)
        for name, path in files:
            self.prompt_combo.addItem(name, userData=path)
        self.prompt_combo.blockSignals(False)
        self._on_prompt_selected(self.prompt_combo.currentIndex())

    def _load_prompts_from_settings(self):
        if not self.settings.ok():
            self._load_prompts()
            return
        entries = self.settings.prompts(self.lang)
        # Remember current selection by pid if possible
        prev_pid = None
        cur_data = self.prompt_combo.currentData()
        if isinstance(cur_data, tuple) and len(cur_data) == 2:
            prev_pid = cur_data[0]
        self.prompt_combo.blockSignals(True)
        self.prompt_combo.clear()
        self.prompt_combo.addItem("(none)", userData=None)
        for pid, title, path in entries:
            self.prompt_combo.addItem(title, userData=(pid, path))
        self.prompt_combo.blockSignals(False)
        # Restore selection by pid if possible
        if prev_pid is not None:
            for i in range(self.prompt_combo.count()):
                data = self.prompt_combo.itemData(i)
                if isinstance(data, tuple) and len(data) == 2 and data[0] == prev_pid:
                    self.prompt_combo.setCurrentIndex(i)
                    break
        self._on_prompt_selected(self.prompt_combo.currentIndex())

    def _on_prompt_selected(self, idx: int):
        data = self.prompt_combo.currentData()
        path = None
        if isinstance(data, tuple) and len(data) == 2:
            _, path = data
        elif isinstance(data, str):
            path = data
        if path:
            text = read_prompt(path)
            # Filter multilingual prompt content by chosen language if file bundles multiple langs
            text = self._filter_prompt_by_language(text, self.lang)
            self.current_system_prompt = text
            self.prompt_preview.setPlainText(text)
        else:
            self.current_system_prompt = ""
            self.prompt_preview.setPlainText("")

    # --- Session start from prompt with topic ---
    def on_start_session(self):
        # Ensure engine credentials are applied
        api_key = (self.api_key_edit.text() or "").strip()
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        proxy = (self.proxy_edit.text() or "").strip()
        if proxy:
            os.environ["USEFULCLICKER_OPENAI_PROXY"] = proxy

        # Require a prompt to be selected (optional: allow none)
        sys_prompt = (self.current_system_prompt or "").strip()
        if not sys_prompt:
            self._append_info("Select a prompt first or proceed with General Chat via Send.")
            return

        # Ask for topic depending on selected prompt id
        pdata = self.prompt_combo.currentData()
        pid = pdata[0] if isinstance(pdata, tuple) and len(pdata) == 2 else None
        topic_title = self._t("topic_generic", "Enter topic (optional)")
        if pid == "virtual_opponent":
            topic_title = self._t("topic_virtual", "Enter debate topic")
        elif pid == "aggressive_opponent":
            topic_title = self._t("topic_aggressive", "Enter aggressive debate topic")
        elif pid == "philosophy_reflection":
            topic_title = self._t("topic_philo", "Enter philosophical topic")

        # Use debate dialog for some modes
        use_dialog = pid in {"virtual_opponent", "aggressive_opponent"}
        if use_dialog:
            def _llm_generate_text(prompt: str) -> str:
                model = (self.model_combo.currentText() or None)
                temp = float(self.temp_spin.value())
                eng = self.engine_combo.currentText().strip().lower()
                if eng == 'openai':
                    if self.openai_client is None:
                        self.openai_client = OpenAIClient()
                    return self.openai_client.generate_text(prompt, model=model, temperature=temp)
                else:
                    if self.ollama_client is None:
                        self.ollama_client = OllamaClient()
                    return self.ollama_client.generate_text(prompt, model=model, temperature=temp)

            dlg = DebateTopicDialog(lang=self.lang, llm_generate_text=_llm_generate_text, parent=self)
            # Localize labels
            t = self.settings.ui_texts(self.lang) if self.settings.ok() else {}
            dlg.setWindowTitle(t.get("debate_dialog") or "Debate Topic")
            dlg.disc_label.setText(t.get("debate_discipline") or "Discipline")
            dlg.sub_label.setText(t.get("debate_subtopic") or "Subtopic")
            dlg.gen_btn.setText(t.get("debate_generate") or "Generate")
            if hasattr(dlg, 'count_label'):
                dlg.count_label.setText(t.get("debate_count") or "Count")
            if hasattr(dlg, 'custom_label'):
                dlg.custom_label.setText(t.get("debate_custom") or "Custom question")
            dlg.ok_btn.setText(t.get("debate_ok") or "OK")
            dlg.cancel_btn.setText(t.get("debate_cancel") or "Cancel")
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return
            topic = dlg.selected_question or ""
            if not topic:
                return
        else:
            topic, ok = QInputDialog.getText(self, "Reasonarium", topic_title)
            if not ok:
                return

        # Apply template replacement if present
        self._last_topic = (topic or "").strip() or None
        sys_prompt_applied = self._apply_topic_to_prompt(sys_prompt, topic)

        # Reset chat and seed messages
        self.chat_view.clear()
        self.messages = []
        if sys_prompt_applied.strip():
            self.messages.append({"role": "system", "content": sys_prompt_applied})
        # Seed user message to trigger the assistant per instructions (localized)
        if (self.lang or '').lower() == 'ru':
            seed_user = f"Начнём. Тема: {topic.strip()}" if topic.strip() else "Начнём."
        else:
            seed_user = f"Begin. Topic: {topic.strip()}" if topic.strip() else "Begin."
        self.messages.append({"role": "user", "content": seed_user})
        self._append_user(seed_user)

        # Start streaming
        try:
            gen_factory = self._make_stream_factory()
        except Exception as e:
            self._append_info(f"Cannot start streaming: {e}")
            QMessageBox.critical(self, "Error", f"Cannot start streaming: {e}")
            return
        self._append_assistant("")
        self._start_stream(gen_factory)

    @staticmethod
    def _apply_topic_to_prompt(prompt_text: str, topic: str) -> str:
        t = (topic or "").strip()
        if not t:
            return prompt_text
        # Replace common placeholders if author used them
        variants = ["{topic}", "{{TOPIC}}", "[[TOPIC]]", "<TOPIC>", "$TOPIC"]
        for v in variants:
            if v in prompt_text:
                return prompt_text.replace(v, t)
        # Otherwise, append a clear topic declaration
        return prompt_text + f"\n\nCurrent topic: {t}"

    def _ensure_prompt_topic(self, prompt_text: str) -> str:
        """If prompt has a topic placeholder and topic is unknown, ask user and substitute."""
        if not prompt_text:
            return prompt_text
        placeholders = ("{topic}", "{{TOPIC}}", "[[TOPIC]]", "<TOPIC>", "$TOPIC")
        if not any(ph in prompt_text for ph in placeholders):
            return prompt_text
        topic = (self._last_topic or "").strip()
        if not topic:
            pid = self._get_selected_prompt_id()
            title = self._t("topic_generic", "Enter topic (optional)")
            if pid == "virtual_opponent":
                title = self._t("topic_virtual", "Enter debate topic")
            elif pid == "aggressive_opponent":
                title = self._t("topic_aggressive", "Enter aggressive debate topic")
            elif pid == "philosophy_reflection":
                title = self._t("topic_philo", "Enter philosophical topic")
            t, ok = QInputDialog.getText(self, "Reasonarium", title)
            if not ok:
                return prompt_text
            topic = (t or "").strip()
            self._last_topic = topic or None
        return self._apply_topic_to_prompt(prompt_text, topic)

    def _get_selected_prompt_id(self) -> Optional[str]:
        data = self.prompt_combo.currentData()
        if isinstance(data, tuple) and len(data) == 2:
            return data[0]
        return None

    # --- Chat actions ---
    def on_new_chat(self):
        self.messages = []
        self.chat_view.clear()
        if self.current_system_prompt.strip():
            self.messages.append({"role": "system", "content": self._compose_system_prompt()})
        self._append_info("New chat started.")

    def on_auto_answer(self):
        # Build a suggested user answer to the latest assistant question (or input text)
        try:
            # Find last assistant message as the question context
            question = (self.input_edit.toPlainText() or "").strip()
            for m in reversed(self.messages):
                if m.get("role") == "assistant" and (m.get("content") or "").strip():
                    question = m.get("content").strip()
                    break
            if not question:
                QMessageBox.information(self, "Auto answer", "No question to answer.")
                return

            # Instruction by language
            if (self.lang or '').lower() == 'ru':
                instr = (
                    "Сформулируй максимально адекватный, аргументированный и краткий ответ (2–4 абзаца) на вопрос ниже,"
                    " выступая в роли пользователя. Учитывай контекст переписки. Верни только текст ответа без префиксов."
                )
                ucontent = f"{instr}\n\nВопрос:\n{question}"
            else:
                instr = (
                    "Write the most reasonable, well‑argued, concise answer (2–4 paragraphs) to the question below,"
                    " acting as the user. Consider chat context. Return only the answer, no prefixes."
                )
                ucontent = f"{instr}\n\nQuestion:\n{question}"

            # Prepare messages with current context + instruction
            msgs = list(self.messages)
            # Ensure a system prompt exists
            if not msgs and self.current_system_prompt.strip():
                msgs.append({"role": "system", "content": self._compose_system_prompt()})
            msgs.append({"role": "user", "content": ucontent})

            model = (self.model_combo.currentText() or None)
            temp = float(self.temp_spin.value())
            eng = self.engine_combo.currentText().strip().lower()

            if eng == 'openai':
                if self.openai_client is None:
                    self.openai_client = OpenAIClient()
                suggestion = self.openai_client.generate_chat(msgs, model=model, temperature=temp)
            else:
                if self.ollama_client is None:
                    self.ollama_client = OllamaClient()
                suggestion = self.ollama_client.generate_chat(msgs, model=model, temperature=temp)

            suggestion = (suggestion or "").strip()
            if not suggestion:
                QMessageBox.information(self, "Auto answer", "No suggestion produced.")
                return
            # Put into input for user to review/edit
            self.input_edit.setPlainText(suggestion)
        except Exception as e:
            QMessageBox.critical(self, "Auto answer", f"Failed: {e}")

    def on_next_question(self):
        try:
            # Prepare minimal instruction based on UI language
            if (self.lang or '').lower() == 'ru':
                instr = "Пропусти критику: пользователь ничего не ответил. Задай следующий уточняющий вопрос."
                display = "→ Следующий вопрос"
            else:
                instr = "Skip critique: the user did not answer. Ask the next clarifying question."
                display = "→ Next question"

            # Ensure system prompt on first turn
            if not self.messages and self.current_system_prompt.strip():
                self.messages.append({"role": "system", "content": self._compose_system_prompt()})

            self.messages.append({"role": "user", "content": instr})
            self._append_user(display)

            # Start streaming reply
            gen_factory = self._make_stream_factory()
            self._append_assistant("")
            self._start_stream(gen_factory)
        except Exception as e:
            QMessageBox.critical(self, "Next question", f"Failed: {e}")

    def on_send(self):
        text = (self.input_edit.toPlainText() or "").strip()
        if not text:
            return
        self.input_edit.clear()

        # Apply API key (if provided) before client init
        api_key = (self.api_key_edit.text() or "").strip()
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        proxy = (self.proxy_edit.text() or "").strip()
        if proxy:
            os.environ["USEFULCLICKER_OPENAI_PROXY"] = proxy

        # Ensure system prompt present on first send
        if not self.messages and self.current_system_prompt.strip():
            self.messages.append({"role": "system", "content": self._compose_system_prompt()})

        self.messages.append({"role": "user", "content": text})
        self._append_user(text)

        # Kick off streaming
        try:
            gen_factory = self._make_stream_factory()
        except Exception as e:
            self._append_info(f"Cannot start streaming: {e}")
            QMessageBox.critical(self, "Error", f"Cannot start streaming: {e}")
            return
        self._append_assistant("")  # placeholder; tokens will fill in
        self._start_stream(gen_factory)

    def _compose_system_prompt(self) -> str:
        # Ensure topic is substituted if template contains a placeholder
        base = self.current_system_prompt or ""
        return self._ensure_prompt_topic(base)

    # UI helpers
    def _append_info(self, text: str):
        self.chat_view.append(f"<div style='color: gray;'>• {self._escape(text)}</div>")

    def _append_user(self, text: str):
        self.chat_view.append(f"<b>You:</b> {self._escape(text)}")
        self.chat_view.moveCursor(QTextCursor.MoveOperation.End)

    def _append_assistant(self, text: str):
        # Start a new assistant message region we can update
        self.chat_view.append("<b>Opponent:</b>")
        cursor = self.chat_view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self._assistant_anchor_pos = cursor.position()
        self._assistant_buffer = []
        self._flush_assistant(text)

    def _flush_assistant(self, more_text: str):
        if more_text:
            self._assistant_buffer.append(more_text)
        buf_text = "".join(self._assistant_buffer)
        rendered = self._render_assistant_html(buf_text)
        cursor = self.chat_view.textCursor()
        # Replace from anchor to end with formatted HTML
        try:
            cursor.setPosition(getattr(self, "_assistant_anchor_pos", cursor.position()))
            cursor.movePosition(QTextCursor.MoveOperation.End, QTextCursor.MoveMode.KeepAnchor)
            cursor.removeSelectedText()
            # Indent rendered block by ~2 spaces after the Opponent: line
            cursor.insertHtml(f"<div style='margin-left: 1.5ch'>{rendered}</div>")
            self.chat_view.moveCursor(QTextCursor.MoveOperation.End)
        except Exception:
            # Fallback: append as plain
            self.chat_view.insertPlainText(more_text)
            self.chat_view.moveCursor(QTextCursor.MoveOperation.End)

    def _render_assistant_html(self, text: str) -> str:
        """Render assistant buffer with colored A:/B: lines."""
        lines = text.splitlines()
        out: List[str] = []
        re_a = re.compile(r"^\s*A\s*[:\.-]\s*(.*)$", re.IGNORECASE)
        re_b = re.compile(r"^\s*B\s*[:\.-]\s*(.*)$", re.IGNORECASE)
        for ln in lines:
            m = re_a.match(ln)
            if m:
                body = self._escape(m.group(1))
                out.append(f"<div style='color:#fbc02d'><b>A:</b> {body}</div>")
                continue
            m = re_b.match(ln)
            if m:
                body = self._escape(m.group(1))
                out.append(f"<div style='color:#2e7d32'><b>B:</b> {body}</div>")
                continue
            out.append(f"<div>{self._escape(ln)}</div>")
        if not lines:
            return ""
        return "".join(out)

    # (subtopic handlers removed)

    # --- Popper Challenge actions ---
    def on_popper_synthesize(self):
        try:
            d1 = (self.pop_d1.currentText() or "").strip()
            d2 = (self.pop_d2.currentText() or "").strip()
        except Exception:
            d1 = d2 = ""
        if not d1 and not d2:
            QMessageBox.information(self, "Popper", "Select at least one discipline")
            return
        eng = self.engine_combo.currentText().strip().lower()
        model = (self.model_combo.currentText() or None)
        temp = float(self.temp_spin.value())
        # Compose domain without subtopics
        domain_str = f"{d1}{' and ' + d2 if d2 else ''}".strip()
        lvl_theory = (self.pop_lvl_theory.currentText() or "").strip()
        lvl_edu = (self.pop_lvl_edu.currentText() or "").strip()
        if (self.lang or '').lower() == 'ru':
            prompt = (
                "Синтезируй краткую научно‑ориентированную теорию (3–6 предложений) в случайно выбранной области "
                f"({domain_str}). Теория не должна повторять предыдущие темы (избегай фруктов, деревьев и чрезмерно узких мотивов).\n\n"
                f"Тип теории: {lvl_theory}. Уровень изложения: {lvl_edu}.\n\n"
                "Затем добавь три раздела:\n"
                "A) Предсказания — не менее 2 чётких, проверяемых предсказаний, вытекающих из теории.\n"
                "C) Нефальсифицируемое — укажи части теории, которые нельзя проверить, и объясни, почему это проблематично.\n\n"
                "Теория может быть серьёзной, игривой, причудливой или абсурдной — но она всё равно должна соответствовать попперовскому критерию научной проверяемости."
            )
        else:
            prompt = (
                "Synthesize a short science‑oriented theory (3–6 sentences) in a randomly chosen domain "
                f"({domain_str}). The theory should not repeat previous themes (avoid fruits, trees, or overly narrow motifs).\n\n"
                f"Theory type: {lvl_theory}. Education level: {lvl_edu}.\n\n"
                "Then add three sections:\n"
                "A) Predictions — at least 2 clear, testable predictions derived from the theory.\n"
                "C) Unfalsifiable — identify any parts of the theory that cannot be tested, and explain why that is problematic.\n\n"
                "The theory may be serious, playful, whimsical, or absurd — but it must still follow Popper’s criterion of scientific testability."
            )
        try:
            if eng == 'openai':
                if self.openai_client is None:
                    self.openai_client = OpenAIClient()
                text = self.openai_client.generate_text(prompt, model=model, temperature=temp)
            else:
                if self.ollama_client is None:
                    self.ollama_client = OllamaClient()
                text = self.ollama_client.generate_text(prompt, model=model, temperature=temp)
        except Exception as e:
            QMessageBox.critical(self, "Popper", f"Synthesis failed: {e}")
            return
        self.pop_theory.setPlainText(text or "")

    def on_popper_check(self):
        theory = (self.pop_theory.toPlainText() or "").strip()
        attempt = (self.pop_user_edit.toPlainText() or "").strip()
        if not theory:
            QMessageBox.information(self, "Popper", "Synthesize a theory first")
            return
        if not attempt:
            QMessageBox.information(self, "Popper", "Provide your experiments / observations")
            return
        eng = self.engine_combo.currentText().strip().lower()
        model = (self.model_combo.currentText() or None)
        temp = float(self.temp_spin.value())
        if (self.lang or '').lower() == 'ru':
            prompt = (
                "Оцени предложенные эксперименты/наблюдения с позиций Поппера. Сначала теория, затем список экспериментов. "
                "Оцени по трём критериям (0–2 балла каждый): (1) есть ли чёткие тестируемые предсказания, к которым привязаны эксперименты; "
                "(2) действительно ли предложенные эксперименты/наблюдения могут сфальсифицировать предсказания; "
                "(3) есть ли нефальсифицируемые элементы. Дай краткий разбор и итоговую сумму баллов.\n\n"
                f"Теория:\n{theory}\n\nЭксперименты/наблюдения пользователя:\n{attempt}"
            )
        else:
            prompt = (
                "Evaluate the proposed experiments/observations in Popper's terms. The theory is below, then the list of experiments. "
                "Score three criteria (0–2 each): (1) Are there clear testable predictions that the experiments target? (2) Can the proposed "
                "experiments/observations actually falsify those predictions? (3) Are there unfalsifiable parts? Provide a brief critique and a total score.\n\n"
                f"Theory:\n{theory}\n\nUser experiments/observations:\n{attempt}"
            )
        try:
            if eng == 'openai':
                if self.openai_client is None:
                    self.openai_client = OpenAIClient()
                text = self.openai_client.generate_text(prompt, model=model, temperature=temp)
            else:
                if self.ollama_client is None:
                    self.ollama_client = OllamaClient()
                text = self.ollama_client.generate_text(prompt, model=model, temperature=temp)
        except Exception as e:
            QMessageBox.critical(self, "Popper", f"Evaluation failed: {e}")
            return
        self.pop_result.setPlainText(text or "")

    def on_popper_confirm(self):
        theory = (self.pop_theory.toPlainText() or "").strip()
        experiments = (self.pop_user_edit.toPlainText() or "").strip()
        if not theory or not experiments:
            QMessageBox.information(self, "Popper", "Provide theory and experiments first")
            return
        eng = self.engine_combo.currentText().strip().lower()
        model = (self.model_combo.currentText() or None)
        temp = float(self.temp_spin.value())
        if (self.lang or '').lower() == 'ru':
            instr = "Дай максимально честную и конструктивную критику."
            prompt = (
                f"{instr} Оцени валидность предложенных экспериментов в рамках предсказаний данной теории. "
                "Укажи сильные стороны, логические дыры, риски некорректной интерпретации, и как улучшить дизайн эксперимента.\n\n"
                f"Теория:\n{theory}\n\nЭксперименты/наблюдения:\n{experiments}"
            )
        else:
            instr = "Give me the most brutally honest constructive criticism you can."
            prompt = (
                f"{instr} Evaluate whether the proposed experiments are valid within the predictions of the theory. "
                "Call out strengths, logical holes, risks of misinterpretation, and how to improve the experiment design.\n\n"
                f"Theory:\n{theory}\n\nExperiments/observations:\n{experiments}"
            )
        try:
            if eng == 'openai':
                if self.openai_client is None:
                    self.openai_client = OpenAIClient()
                text = self.openai_client.generate_text(prompt, model=model, temperature=temp)
            else:
                if self.ollama_client is None:
                    self.ollama_client = OllamaClient()
                text = self.ollama_client.generate_text(prompt, model=model, temperature=temp)
        except Exception as e:
            QMessageBox.critical(self, "Popper", f"Critique failed: {e}")
            return
        self.pop_result.setPlainText(text or "")

    def on_popper_ai_toggle(self, state: int):
        show = state != 0
        self.pop_ai_lbl.setVisible(show)
        self.pop_ai_edit.setVisible(show)
        if not show:
            self.pop_ai_edit.clear()
            return
        # Generate AI suggestions for experiments
        theory = (self.pop_theory.toPlainText() or "").strip()
        if not theory:
            return
        eng = self.engine_combo.currentText().strip().lower()
        model = (self.model_combo.currentText() or None)
        temp = float(self.temp_spin.value())
        if (self.lang or '').lower() == 'ru':
            prompt = (
                "Предложи 3–6 экспериментальных проверок или наблюдений, которые максимально прямо сфальсифицируют предсказания "
                "данной теории. Кратко, по одному пункту в строке.\n\nТеория:\n" + theory
            )
        else:
            prompt = (
                "Propose 3–6 experimental tests or observations that could most directly falsify the predictions of this theory. "
                "Keep it concise, one per line.\n\nTheory:\n" + theory
            )
        try:
            if eng == 'openai':
                if self.openai_client is None:
                    self.openai_client = OpenAIClient()
                text = self.openai_client.generate_text(prompt, model=model, temperature=temp)
            else:
                if self.ollama_client is None:
                    self.ollama_client = OllamaClient()
                text = self.ollama_client.generate_text(prompt, model=model, temperature=temp)
        except Exception as e:
            self.pop_ai_edit.setPlainText(f"Error: {e}")
            return
        self.pop_ai_edit.setPlainText(text or "")

    @staticmethod
    def _escape(text: str) -> str:
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    # --- Streaming machinery ---
    def _start_stream(self, gen_factory: Callable[[], Iterator[str]]):
        # Clean up prior thread if any
        if self._stream_thread is not None:
            try:
                self._stream_thread.quit()
                self._stream_thread.wait(100)
            except Exception:
                pass
        self._stream_thread = QThread()
        self._stream_worker = StreamWorker(gen_factory)
        self._stream_worker.moveToThread(self._stream_thread)
        self._stream_thread.started.connect(self._stream_worker.run)
        self._stream_worker.token.connect(lambda t: self._flush_assistant(t))
        self._stream_worker.finished.connect(self._on_stream_finished)
        self._stream_worker.error.connect(self._on_stream_error)
        self._stream_thread.start()

    def _on_stream_finished(self):
        # Finalize assistant message into history
        final_text = "".join(getattr(self, "_assistant_buffer", []))
        self.messages.append({"role": "assistant", "content": final_text})
        self._cleanup_stream()

    def _on_stream_error(self, msg: str):
        self._append_info(f"Error: {msg}")
        self._cleanup_stream()

    def _cleanup_stream(self):
        if self._stream_thread:
            self._stream_thread.quit()
            self._stream_thread.wait(100)
        self._stream_thread = None
        self._stream_worker = None

    # --- LLM streaming factory ---
    def _make_stream_factory(self) -> Callable[[], Iterator[str]]:
        engine = self.engine_combo.currentText()
        model = (self.model_combo.currentText() or "").strip() or None
        temp = float(self.temp_spin.value())
        messages = list(self.messages)  # copy current history

        if engine == "OpenAI":
            if self.openai_client is None:
                # Instantiate lazily to surface API key errors early
                self.openai_client = OpenAIClient()
            client = self.openai_client

            def gen() -> Iterator[str]:
                try:
                    for piece in client.stream_chat(messages, model=model, temperature=temp):
                        yield piece
                except Exception as e:
                    # Surface explicit connection issue
                    yield f"\n[Connection error: {e}]\n"

            return gen

        elif engine == "Ollama":
            if self.ollama_client is None:
                self.ollama_client = OllamaClient()
            client = self.ollama_client

            def gen() -> Iterator[str]:
                for piece in client.stream_chat(messages, model=model, temperature=temp):
                    yield piece

            return gen

        else:
            raise RuntimeError(f"Unknown engine: {engine}")

    # --- Connection test ---
    def on_test_openai(self):
        # Apply potential edits
        api_key = (self.api_key_edit.text() or "").strip()
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        proxy = (self.proxy_edit.text() or "").strip()
        if proxy:
            os.environ["USEFULCLICKER_OPENAI_PROXY"] = proxy
        # Quick non-stream test
        try:
            client = OpenAIClient()
            txt = client.generate_chat([
                {"role": "user", "content": "Reply with: pong"}
            ], model=(self.model_combo.currentText() or None), temperature=0)
            if txt:
                QMessageBox.information(self, "OpenAI", f"OK: {txt[:200]}")
            else:
                QMessageBox.warning(self, "OpenAI", "Received empty response.")
        except Exception as e:
            QMessageBox.critical(self, "OpenAI", f"Connection error: {e}")

    def on_speech_to_text(self):
        # Initialize OpenAI client
        if self.openai_client is None:
            try:
                self.openai_client = OpenAIClient()
            except Exception as e:
                QMessageBox.critical(self, "OpenAI", f"Init failed: {e}")
                return
        # Pick audio file
        path, _ = QFileDialog.getOpenFileName(self, self._t("speech_to_text", "Speech to text"), "", "Audio Files (*.wav *.mp3 *.m4a *.ogg)")
        if not path:
            return
        try:
            text = self.openai_client.transcribe_file(path, model=None)
        except Exception as e:
            QMessageBox.critical(self, "STT", f"Failed: {e}")
            return
        if text:
            self.input_edit.setPlainText(text)
        else:
            QMessageBox.information(self, "STT", "No text recognized.")

    def _populate_mics(self):
        # Enumerate microphones using sounddevice helper
        try:
            from app.audio_recorder import AudioRecorder
            inputs = AudioRecorder.list_input_devices()
        except Exception as e:
            print(f"[Audio] Microphone enumeration failed: {e}")
            traceback.print_exc()
            inputs = []
        try:
            self.mic_combo.blockSignals(True)
            self.mic_combo.clear()
            if inputs:
                for dev in inputs:
                    name = dev.get('name') or f"Device {dev.get('index')}"
                    self.mic_combo.addItem(name, userData=dev.get('index'))
                self.mic_combo.setCurrentIndex(0)
                self.record_btn.setEnabled(True)
                print(f"[Audio] {len(inputs)} microphone(s) available")
            else:
                self.mic_combo.addItem("(no inputs)", userData=None)
                self.record_btn.setEnabled(False)
                print("[Audio] No microphones found; check OS permissions and drivers")
        finally:
            try:
                self.mic_combo.blockSignals(False)
            except Exception:
                pass

    # --- Microphone recording ---
    def on_record_toggle(self):
        # Toggle recording with sounddevice; on stop — transcribe
        if self._recorder is not None:
            # Stop
            try:
                path = None
                try:
                    path = self._recorder.stop()
                except Exception:
                    # WhisperStream.stop() returns None; AudioRecorder.stop() returns path
                    path = None
            except Exception:
                path = None
            self._recorder = None
            self.record_btn.setText(self._t("record", "Record"))
            if path and os.path.exists(path):
                try:
                    text = ""
                    if self.engine_combo.currentText().strip().lower() == "ollama":
                        # Local STT via Whisper for Ollama engine
                        # Choose language hint from UI language
                        lang_hint = 'ru' if (self.lang or '').lower() == 'ru' else 'en'
                        whisper_model = (self.whisper_combo.currentText() or os.getenv('WHISPER_MODEL') or 'base')
                        text = transcribe_whisper(path, model=whisper_model, language=lang_hint)
                    else:
                        if self.openai_client is None:
                            self.openai_client = OpenAIClient()
                        text = self.openai_client.transcribe_file(path)
                    if text:
                        self.input_edit.setPlainText(text)
                    else:
                        QMessageBox.information(self, "STT", "No text recognized.")
                except Exception as e:
                    QMessageBox.critical(self, "STT", f"Failed: {e}")
                finally:
                    try:
                        os.unlink(path)
                    except Exception:
                        pass
            return

        dev_index = self.mic_combo.currentData()
        if dev_index is None:
            QMessageBox.warning(self, "Audio", "No input device available.")
            return
        # Start simple recorder (Whisper transcription on stop for Ollama)
        try:
            self._recorder = AudioRecorder(device=int(dev_index), samplerate=16000, channels=1)  # type: ignore
            self._recorder.start()
            self.record_btn.setText(self._t("stop", "Stop"))
        except Exception as e:
            self._recorder = None
            QMessageBox.critical(self, "Audio", f"Cannot start recording: {e}")

    # No QtMultimedia streaming callback; handled by AudioRecorder

    def _append_stt_text(self, txt: str):
        cur = self.input_edit.toPlainText().strip()
        if cur:
            self.input_edit.setPlainText(cur + "\n" + txt)
        else:
            self.input_edit.setPlainText(txt)

    # Voice playback and legacy functions removed (no QtMultimedia)

    # --- Language support ---
    def _apply_language(self):
        # Populate language combo
        self.lang_combo.blockSignals(True)
        self.lang_combo.clear()
        langs = self.settings.languages() if self.settings.ok() else [("en", "English"), ("ru", "Русский")]
        current_index = 0
        for i, (code, name) in enumerate(langs):
            self.lang_combo.addItem(name, userData=code)
            if code == self.lang:
                current_index = i
        self.lang_combo.setCurrentIndex(current_index)
        self.lang_combo.blockSignals(False)

        # Update UI texts
        self._apply_ui_texts()
        # Refresh OpenAI models (in case language switch requires different defaults later)
        self._populate_openai_models()
        # Populate mics
        try:
            self._populate_mics()
        except Exception:
            pass
        # Populate disciplines for Curiosity tab
        dis = self.settings.disciplines(self.lang) if self.settings.ok() else []
        self.cd_disc_combo.clear()
        if dis:
            self.cd_disc_combo.addItem("(random)", userData=None)
            for d in dis:
                self.cd_disc_combo.addItem(d, userData=d)
        else:
            self.cd_disc_combo.addItem("(random)", userData=None)

        # No mode selector

    # def _populate_modes(self):
    #     pass

    def _apply_ui_texts(self):
        t = self.settings.ui_texts(self.lang) if self.settings.ok() else {}
        def tx(key: str, fallback: str) -> str:
            return t.get(key) or fallback
        self.lang_label.setText(tx("language", "Language"))
        self.engine_label.setText(tx("engine", "Engine"))
        self.model_label.setText(tx("model", "Model"))
        self.temp_label.setText(tx("temp", "Temp"))
        self.api_key_label.setText(tx("api_key", "API Key"))
        self.proxy_label.setText(tx("proxy", "Proxy"))
        self.prompt_label.setText(tx("prompt", "Prompt"))
        self.reload_prompts_btn.setText(tx("reload_prompts", "Reload Prompts"))
        self.start_session_btn.setText(tx("start_session", "Start Session…"))
        self.send_btn.setText(tx("send", "Send"))
        self.new_btn.setText(tx("new_chat", "New Chat"))
        if hasattr(self, 'auto_btn'):
            self.auto_btn.setText(tx("auto_answer", "Auto answer"))
        if hasattr(self, 'nextq_btn'):
            self.nextq_btn.setText(tx("next_question", "Next question"))
        self.preview_box.setTitle(tx("prompt_preview", "Prompt Preview"))
        self.test_btn.setText(tx("test_openai", "Test OpenAI"))
        # Tabs
        self.tabs.setTabText(0, tx("tab_chat", "Chat"))
        # Curiosity tab texts
        # Find curiosity tab index (assumed 1)
        self.tabs.setTabText(1, tx("tab_curiosity", "Curiosity Drive"))
        # Popper tab texts and data
        if cdn is not None:
            try:
                dis = list(getattr(cdn, 'disciplines', []))
            except Exception:
                dis = []
        else:
            dis = []
        try:
            self.pop_d1.clear(); self.pop_d2.clear()
            if dis:
                # Discipline A: regular list
                self.pop_d1.addItems(dis)
                # Discipline B: add empty option first, then list
                self.pop_d2.addItem("")
                self.pop_d2.addItems(dis)
                self.pop_d2.setCurrentIndex(0)
        except Exception:
            pass
        try:
            self.tabs.setTabText(2, tx("tab_popper", "Popper Challenge"))
            self.pop_lbl_d1.setText(tx("popper_d1", "Discipline A"))
            self.pop_lbl_d2.setText(tx("popper_d2", "Discipline B"))
            self.pop_lvl_theory_lbl.setText(tx("popper_level_theory", "Theory type"))
            self.pop_lvl_edu_lbl.setText(tx("popper_level_edu", "Education level"))
            # Populate options per language
            self._populate_popper_levels()
            self.pop_syn_btn.setText(tx("popper_synthesize", "Synthesize theory"))
            self.pop_user_lbl.setText(tx("popper_user_experiments", "Your experiments / observations"))
            self.pop_ai_chk.setText(tx("popper_ai_show", "Show AI suggestions"))
            self.pop_ai_lbl.setText(tx("popper_ai_experiments", "AI experiments / observations"))
            self.pop_check_btn.setText(tx("popper_check", "Evaluate falsification"))
            self.pop_confirm_btn.setText(tx("popper_confirm", "Confirm experiment"))
            self.pop_result_lbl.setText(tx("popper_result", "Evaluation"))
        except Exception:
            pass
        # Curiosity labels (form created with static labels; adjust)
        self.cd_lbl_disciplines.setText(tx("cd_disciplines", "Disciplines"))
        self.cd_lbl_audience.setText(tx("cd_audience", "Audience"))
        self.cd_lbl_rarity.setText(tx("cd_rarity", "Rarity"))
        self.cd_lbl_novelty.setText(tx("cd_novelty", "Novelty"))
        self.cd_lbl_count.setText(tx("cd_count", "Items"))
        self.cd_audience.setPlaceholderText(tx("cd_audience", "Audience"))
        self.cd_rarity.setPlaceholderText(tx("cd_rarity", "Rarity"))
        self.cd_novelty.setPlaceholderText(tx("cd_novelty", "Novelty"))
        self.cd_generate_btn.setText(tx("cd_generate", "Generate"))
        # Mic controls
        if hasattr(self, 'mic_label'):
            self.mic_label.setText(tx("microphone", "Microphone"))
        if hasattr(self, 'whisper_label'):
            self.whisper_label.setText(tx("whisper_model", "Whisper model"))
        if hasattr(self, 'record_btn'):
            if getattr(self, '_recorder', None) is None:
                self.record_btn.setText(tx("record", "Record"))
            else:
                self.record_btn.setText(tx("stop", "Stop"))

    def _populate_openai_models(self):
        # Populate with settings; allow custom entry via editable combo
        models = self.settings.openai_models() if self.settings.ok() else []
        current = (self.model_combo.currentText() or "").strip()
        env_default = os.getenv("USEFULCLICKER_OPENAI_MODEL", "")
        xml_default = self.settings.default_openai_model() if self.settings.ok() else None
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        if models:
            self.model_combo.addItems(models)
        # Set initial text/selection priority: env var > xml default > existing > hardcoded
        target = env_default or (xml_default or current) or "gpt-4o-mini"
        self.model_combo.setEditText(target)
        self.model_combo.blockSignals(False)

    def _populate_whisper_models(self):
        try:
            models = ["tiny", "base", "small", "medium", "large-v2"]
            env_default = os.getenv("WHISPER_MODEL", "base")
            self.whisper_combo.blockSignals(True)
            self.whisper_combo.clear()
            self.whisper_combo.addItems(models)
            # pick env default if present
            if env_default in models:
                self.whisper_combo.setCurrentText(env_default)
            self.whisper_combo.blockSignals(False)
        except Exception:
            pass

    def _populate_popper_levels(self):
        # Save selections
        cur_theory = self.pop_lvl_theory.currentText() if self.pop_lvl_theory.count() else ""
        cur_edu = self.pop_lvl_edu.currentText() if self.pop_lvl_edu.count() else ""
        # Options per language
        if (self.lang or '').lower() == 'ru':
            theory_opts = [
                "Trivial / Absurd — намеренно простая или сказочная",
                "Folk / Intuitive — бытовое или интуитивное",
                "Speculative / Pseudoscientific — правдоподобно, но с дырами",
                "Scientific‑Style — близко к научной гипотезе",
                "Advanced / Cross‑disciplinary — сложная, междисциплинарная",
            ]
            edu_opts = [
                "School — базовый школьный уровень",
                "Undergraduate (Bachelor) — базовые модели науки",
                "Graduate (Master) — углублённые рассуждения",
                "Doctoral / PhD — профессиональная детализация",
            ]
        else:
            theory_opts = [
                "Trivial / Absurd — deliberately simple or whimsical",
                "Folk / Intuitive — everyday or intuitive",
                "Speculative / Pseudoscientific — plausible but leaky",
                "Scientific‑Style — resembles a scientific hypothesis",
                "Advanced / Cross‑disciplinary — complex, multi‑domain",
            ]
            edu_opts = [
                "School — very simple language",
                "Undergraduate (Bachelor) — basic scientific models",
                "Graduate (Master) — advanced reasoning",
                "Doctoral / PhD — highly technical detail",
            ]
        # Fill combos
        self.pop_lvl_theory.blockSignals(True)
        self.pop_lvl_theory.clear()
        self.pop_lvl_theory.addItems(theory_opts)
        self.pop_lvl_theory.blockSignals(False)

        self.pop_lvl_edu.blockSignals(True)
        self.pop_lvl_edu.clear()
        self.pop_lvl_edu.addItems(edu_opts)
        self.pop_lvl_edu.blockSignals(False)
        # Restore selection if possible
        if cur_theory:
            idx = self.pop_lvl_theory.findText(cur_theory)
            if idx >= 0:
                self.pop_lvl_theory.setCurrentIndex(idx)
        if cur_edu:
            idx = self.pop_lvl_edu.findText(cur_edu)
            if idx >= 0:
                self.pop_lvl_edu.setCurrentIndex(idx)

    def _populate_whisper_models(self):
        models = self.settings.whisper_models() if self.settings.ok() else []
        default = self.settings.default_whisper_model() if self.settings.ok() else None
        env_default = os.getenv('WHISPER_MODEL', '')
        self.whisper_combo.blockSignals(True)
        self.whisper_combo.clear()
        if models:
            self.whisper_combo.addItems(models)
        target = env_default or default or (models[0] if models else 'base')
        self.whisper_combo.setCurrentText(target)
        self.whisper_combo.blockSignals(False)

    def _populate_ollama_models(self):
        models = self.settings.ollama_models() if self.settings.ok() else []
        # Set Ollama base URL to env for client
        base = self.settings.ollama_endpoint() if self.settings.ok() else None
        if base:
            os.environ["USEFULCLICKER_OLLAMA_BASE"] = base
        current = (self.model_combo.currentText() or "").strip()
        env_default = os.getenv("USEFULCLICKER_OLLAMA_MODEL", "")
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        if models:
            self.model_combo.addItems(models)
        target = env_default or (models[0] if models else current) or "llama3.2:latest"
        self.model_combo.setEditText(target)
        self.model_combo.blockSignals(False)

    def on_engine_changed(self, value: str):
        eng = (value or "").strip().lower()
        if eng == "openai":
            self._populate_openai_models()
        elif eng == "ollama":
            self._populate_ollama_models()
        else:
            pass

    def _t(self, key: str, fallback: Optional[str] = None) -> str:
        t = self.settings.ui_texts(self.lang) if self.settings.ok() else {}
        return t.get(key) or fallback or key

    def on_model_changed(self, text: str):
        name = (text or "").strip()
        if not name:
            return
        # Heuristic engine switch based on known lists and tag pattern
        try:
            ollama_list = set(self.settings.ollama_models() or []) if self.settings.ok() else set()
        except Exception:
            ollama_list = set()
        try:
            openai_list = set(self.settings.openai_models() or []) if self.settings.ok() else set()
        except Exception:
            openai_list = set()
        if name in ollama_list or ":" in name:
            if self.engine_combo.currentText() != "Ollama":
                self.engine_combo.setCurrentText("Ollama")
        elif name in openai_list:
            if self.engine_combo.currentText() != "OpenAI":
                self.engine_combo.setCurrentText("OpenAI")

    def on_language_changed(self, idx: int):
        code = self.lang_combo.currentData()
        if not code:
            return
        self.lang = str(code)
        self._apply_language()
        self._load_prompts_from_settings()

    @staticmethod
    def _filter_prompt_by_language(text: str, lang: str) -> str:
        """If a prompt file contains multiple language blocks separated by lines with '---',
        try to pick the block matching the target language. Heuristic: choose the block with
        more Cyrillic for ru; otherwise choose the block with fewer Cyrillic characters.
        If no separators, return as is.
        """
        if not text:
            return text
        parts = []
        cur = []
        for line in text.splitlines():
            if line.strip() == '---':
                parts.append("\n".join(cur).strip())
                cur = []
            else:
                cur.append(line)
        if cur:
            parts.append("\n".join(cur).strip())
        if len(parts) <= 1:
            return text
        def cyr_ratio(s: str) -> float:
            total = len(s)
            if total == 0:
                return 0.0
            cyr = sum(1 for ch in s if 'А' <= ch <= 'я' or ch == 'ё' or ch == 'Ё')
            return cyr / total
        if (lang or '').lower() == 'ru':
            parts_sorted = sorted(parts, key=cyr_ratio, reverse=True)
        else:
            parts_sorted = sorted(parts, key=cyr_ratio)  # prefer least Cyrillic
        return parts_sorted[0]

    # --- Curiosity Drive fallback ---
    def on_cd_generate(self):
        import json, datetime, random
        # disciplines from settings filtered by combo selection
        dis_all = self.settings.disciplines(self.lang) if self.settings.ok() else []
        pick = self.cd_disc_combo.currentData()
        disciplines = dis_all if pick is None else [pick]
        audience = (self.cd_audience.text() or "general").strip()
        rarity = (self.cd_rarity.text() or "medium-rare").strip()
        novelty = (self.cd_novelty.text() or "balanced").strip()
        n = int(self.cd_count.value())

        data = self._curiosity_fallback_json(disciplines, audience, rarity, novelty, n)
        self.cd_output.setPlainText(json.dumps(data, ensure_ascii=False, indent=2))

    def _curiosity_fallback_json(self, disciplines, audience, rarity, novelty, n):
        # deterministic simple filler
        import datetime, random
        if not disciplines:
            disciplines = ['General Science']
        picked = random.choice(disciplines)
        meta = {
            "audience": audience,
            "rarity": rarity,
            "novelty": novelty,
            "discipline_pool": disciplines,
            "picked_discipline": picked,
            "n": n,
            "timestamp": datetime.datetime.now().isoformat()
        }
        items = []
        for i in range(n):
            items.append({
                'concept': f'{picked} concept {i+1}',
                'rare_term': None,
                'kid_gloss': f'A short explanation for item {i+1}',
                'hook_question': f'What if {picked} {i+1}?',
                'mini_task': f'Try a small experiment {i+1}',
                'yt_query': f'{picked} intro'
            })
        return {'meta': meta, 'items': items}
