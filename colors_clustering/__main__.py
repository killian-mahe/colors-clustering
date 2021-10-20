"""
Main application program.
"""
import sys

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from colors_clustering.gui import MainWindow


if __name__ == "__main__":
    app = QApplication([])
    app.setWindowIcon(QIcon("./colors_clustering/assets/logo.jpg"))

    main_window = MainWindow()
    main_window.resize(1000, 600)
    main_window.show()

    sys.exit(app.exec())
