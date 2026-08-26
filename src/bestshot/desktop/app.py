"""Point d'entrée de l'application de bureau BestShotAI."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from bestshot.desktop.main_window import MainWindow


def main() -> None:
    """Démarre l'interface graphique locale."""
    application = QApplication.instance() or QApplication(sys.argv)
    application.setApplicationName("BestShotAI")
    window = MainWindow()
    window.show()
    sys.exit(application.exec())


if __name__ == "__main__":
    main()
