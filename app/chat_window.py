from __future__ import annotations
import os
import traceback
from typing import List, Dict, Optional, Callable, Iterator

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QTextCursor
from PyQt6.QtWidgets import (
    QWidget, QMainWindow, QVBoxLayout, QHBoxLayout, QTextBrowser, QPlainTextEdit,
    QPushButton, QLabel, QComboBox, QLineEdit, QDoubleSpinBox, QFileDialog, QSplitter,
    QGroupBox, QFormLayout, QMessageBox, QInputDialog
)

from app.prompts_loader import list_prompt_files, read_prompt
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

        # UI
        self._build_ui()
        self._load_prompts()

    # --- UI Construction ---
    def _build_ui(self):
        central = QWidget()
        root = QVBoxLayout(central)
        self.setCentralWidget(central)

        # Controls row
        controls = QHBoxLayout()
        root.addLayout(controls)

        # Engine and model
        self.engine_combo = QComboBox()
        self.engine_combo.addItems(["OpenAI", "Ollama"])
        self.engine_combo.setCurrentIndex(0)

        self.model_edit = QLineEdit(os.getenv("USEFULCLICKER_OPENAI_MODEL", "gpt-4o-mini"))
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.0, 2.0)
        self.temp_spin.setSingleStep(0.1)
        self.temp_spin.setValue(1.0)

        controls.addWidget(QLabel("Engine:"))
        controls.addWidget(self.engine_combo)
        controls.addWidget(QLabel("Model:"))
        controls.addWidget(self.model_edit)
        controls.addWidget(QLabel("Temp:"))
        controls.addWidget(self.temp_spin)

        # API key input (OpenAI)
        self.api_key_edit = QLineEdit(os.getenv("OPENAI_API_KEY", ""))
        self.api_key_edit.setPlaceholderText("OPENAI_API_KEY")
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        controls.addWidget(QLabel("API Key:"))
        controls.addWidget(self.api_key_edit, stretch=1)

        # Optional proxy URL and trust_env toggle
        self.proxy_edit = QLineEdit(os.getenv("USEFULCLICKER_OPENAI_PROXY", ""))
        self.proxy_edit.setPlaceholderText("http(s)://user:pass@host:port (optional)")
        controls.addWidget(QLabel("Proxy:"))
        controls.addWidget(self.proxy_edit, stretch=1)

        # Test connection button
        self.test_btn = QPushButton("Test OpenAI")
        self.test_btn.clicked.connect(self.on_test_openai)
        controls.addWidget(self.test_btn)

        # Modes
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "General Chat",
            "Rationality Drive",
            "Philosophical Reflection",
            "Virtual Opponent",
            "Aggressive Opponent",
            "Curiosity Drive",
            "Technical Reflection",
            "Popper Challenge",
        ])
        controls.addWidget(QLabel("Mode:"))
        controls.addWidget(self.mode_combo)

        # Prompts
        self.prompt_combo = QComboBox()
        self.reload_prompts_btn = QPushButton("Reload Prompts")
        self.reload_prompts_btn.clicked.connect(self._load_prompts)
        controls.addWidget(QLabel("Prompt:"))
        controls.addWidget(self.prompt_combo, stretch=1)
        controls.addWidget(self.reload_prompts_btn)
        self.start_session_btn = QPushButton("Start Session…")
        self.start_session_btn.clicked.connect(self.on_start_session)
        controls.addWidget(self.start_session_btn)

        # Splitter: chat and prompt preview
        splitter = QSplitter(Qt.Orientation.Vertical)
        root.addWidget(splitter, stretch=1)

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
        preview_box = QGroupBox("Prompt Preview")
        pv_layout = QVBoxLayout(preview_box)
        self.prompt_preview = QPlainTextEdit()
        self.prompt_preview.setReadOnly(True)
        pv_layout.addWidget(self.prompt_preview)

        splitter.addWidget(preview_box)
        splitter.setSizes([500, 200])

        # Wire prompt selection
        self.prompt_combo.currentIndexChanged.connect(self._on_prompt_selected)

    def _load_prompts(self):
        files = list_prompt_files()
        self.prompt_combo.blockSignals(True)
        self.prompt_combo.clear()
        self.prompt_combo.addItem("(none)", userData=None)
        for name, path in files:
            self.prompt_combo.addItem(name, userData=path)
        self.prompt_combo.blockSignals(False)
        self._on_prompt_selected(self.prompt_combo.currentIndex())

    def _on_prompt_selected(self, idx: int):
        path = self.prompt_combo.currentData()
        if path:
            text = read_prompt(path)
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

        # Ask for topic depending on mode
        mode = self.mode_combo.currentText()
        topic_title = {
            "Virtual Opponent": "Введите тему дебатов",
            "Aggressive Opponent": "Введите тему жёстких дебатов",
            "Philosophical Reflection": "Введите тему философской беседы",
        }.get(mode, "Введите тему (опционально)")

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
        # Seed user message to trigger the assistant per instructions
        seed_user = ("Начнём. Тема: " + topic.strip()) if topic.strip() else "Начнём."
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
        mode = self.mode_combo.currentText()
        if mode == "General Chat":
            return self.current_system_prompt
        # Prepend a lightweight mode header
        header = f"[Mode: {mode}]\n"
        return header + (self.current_system_prompt or "")

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
        model = (self.model_edit.text() or "").strip() or None
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
            ], model=(self.model_edit.text() or None), temperature=0)
            if txt:
                QMessageBox.information(self, "OpenAI", f"OK: {txt[:200]}")
            else:
                QMessageBox.warning(self, "OpenAI", "Received empty response.")
        except Exception as e:
            QMessageBox.critical(self, "OpenAI", f"Connection error: {e}")
