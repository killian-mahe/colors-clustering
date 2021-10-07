from GUI import MainWindow
from PySide6.QtWidgets import QApplication

import sys


if __name__ == "__main__":
    app = QApplication([])

    main_window = MainWindow()
    main_window.resize(800, 600)
    main_window.show()

    sys.exit(app.exec())
