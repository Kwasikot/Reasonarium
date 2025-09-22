from __future__ import annotations
import os
import traceback
from typing import List, Dict, Optional, Callable, Iterator

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QTextCursor
from PyQt6.QtWidgets import (
    QWidget, QMainWindow, QVBoxLayout, QHBoxLayout, QTextBrowser, QPlainTextEdit,
    QPushButton, QLabel, QComboBox, QLineEdit, QDoubleSpinBox, QFileDialog, QSplitter,
    QGroupBox, QFormLayout, QMessageBox, QInputDialog, QTabWidget, QSpinBox
)

from app.prompts_loader import read_prompt
from app.settings_loader import Settings
from llm.openai_client import LLMClient as OpenAIClient
from llm.ollama_client import OllamaClient


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

        # UI
        self._build_ui()
        self._apply_language()
        self._populate_openai_models()
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

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
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
        self.chat_view.setOpenExternalLinks(True)
        self.chat_view.setReadOnly(True)
        chat_layout.addWidget(self.chat_view, stretch=1)

        # Input area
        input_row = QHBoxLayout()
        self.input_edit = QPlainTextEdit()
        self.input_edit.setPlaceholderText("Type your message…")
        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self.on_send)
        self.new_btn = QPushButton("New Chat")
        self.new_btn.clicked.connect(self.on_new_chat)
        input_row.addWidget(self.input_edit, stretch=1)
        input_row.addWidget(self.send_btn)
        input_row.addWidget(self.new_btn)
        chat_layout.addLayout(input_row)

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

        topic, ok = QInputDialog.getText(self, "Reasonarium", topic_title)
        if not ok:
            return

        # Apply template replacement if present
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
        variants = ["{{TOPIC}}", "[[TOPIC]]", "<TOPIC>", "{topic}", "$TOPIC"]
        for v in variants:
            if v in prompt_text:
                return prompt_text.replace(v, t)
        # Otherwise, append a clear topic declaration
        return prompt_text + f"\n\nCurrent topic: {t}"

    # --- Chat actions ---
    def on_new_chat(self):
        self.messages = []
        self.chat_view.clear()
        if self.current_system_prompt.strip():
            self.messages.append({"role": "system", "content": self._compose_system_prompt()})
        self._append_info("New chat started.")

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
        # No mode header; just return the selected system prompt
        return self.current_system_prompt

    # UI helpers
    def _append_info(self, text: str):
        self.chat_view.append(f"<div style='color: gray;'>• {self._escape(text)}</div>")

    def _append_user(self, text: str):
        self.chat_view.append(f"<b>You:</b> {self._escape(text)}")
        self.chat_view.moveCursor(QTextCursor.MoveOperation.End)

    def _append_assistant(self, text: str):
        self.chat_view.append("<b>Assistant:</b> <span id='assistant-current'></span>")
        self._assistant_buffer = []  # internal buffer for streaming
        self._flush_assistant(text)

    def _flush_assistant(self, more_text: str):
        if more_text:
            self._assistant_buffer.append(more_text)
        html = self._escape("".join(self._assistant_buffer)).replace("\n", "<br>")
        # Replace last block's span content
        cursor = self.chat_view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.chat_view.insertPlainText("")  # ensure layout updated
        # The simple way: append incremental text directly
        # For simplicity, append tokens as they arrive:
        if more_text:
            self.chat_view.insertPlainText(more_text)
            self.chat_view.moveCursor(QTextCursor.MoveOperation.End)

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
        self.preview_box.setTitle(tx("prompt_preview", "Prompt Preview"))
        self.test_btn.setText(tx("test_openai", "Test OpenAI"))
        # Tabs
        self.tabs.setTabText(0, tx("tab_chat", "Chat"))
        # Curiosity tab texts
        # Find curiosity tab index (assumed 1)
        self.tabs.setTabText(1, tx("tab_curiosity", "Curiosity Drive"))
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

    def _t(self, key: str, fallback: Optional[str] = None) -> str:
        t = self.settings.ui_texts(self.lang) if self.settings.ok() else {}
        return t.get(key) or fallback or key

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
