"""
Main application program.
"""
import sys

from PySide6.QtWidgets import QApplication

from colors_clustering.gui import MainWindow


if __name__ == "__main__":
    app = QApplication([])

    main_window = MainWindow()
    main_window.resize(800, 600)
    main_window.show()

    sys.exit(app.exec())
