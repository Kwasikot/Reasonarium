from __future__ import annotations
import sys
from PyQt6.QtWidgets import QApplication
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    load_dotenv = None

from app.chat_window import ChatWindow


def main():
    # Load .env from project root if available
    if load_dotenv is not None:
        try:
            load_dotenv()
        except Exception:
            pass
    app = QApplication(sys.argv)
    w = ChatWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
