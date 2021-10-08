"""
Application related GUI.
"""

from PySide6.QtGui import QAction, QIcon, QPixmap
from PySide6.QtWidgets import (
    QMainWindow,
    QGridLayout,
    QGraphicsScene,
    QGraphicsView,
    QGraphicsPixmapItem,
    QPushButton
)
from PySide6 import QtWidgets


class MainWindow(QMainWindow):
    """
    A class to represent the app main window.

    Attributes
    ----------
    layout : QGridLayout
        app grid layout.
    pixmap_item : QGraphicsPixmapItem
        original picture representation.
    scene : QGraphicsScene
        graphical scene for the original picture representation.
    view : QGraphicView
        view that display the scene in the main window.

    Methods
    -------
    create_image_view()
        Create the original picture view.
    create_menus()
        Create the menus displayed in the main window.
    open_file()
        Create a file dialog and update the selected picture.
    """

    def __init__(self):
        """
        Constructs all the necessary attributes for the main window object.
        """
        super().__init__()
        self.setWindowTitle("Colors clustering")
        self.layout = QGridLayout()
        self.pixmap_item = None
        self.create_image_view()
        self.create_menus()
        self.setCentralWidget(QtWidgets.QWidget())
        self.centralWidget().setLayout(self.layout)

    # noinspection PyAttributeOutsideInit
    def create_image_view(self):
        """
        Create all the necessary scene and view for the original picture.

        Returns
        -------
        None
        """
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.layout.addWidget(self.view, 0, 0)

    def create_menus(self):
        """
        Create all the menus.

        Returns
        -------
        None
        """
        button_action = QAction(QIcon("bug.png"), "&Open", self)
        button_action.setStatusTip("Open a new picture")
        button_action.triggered.connect(self.open_file)
        button_action.setCheckable(True)
        self.layout.addWidget(
            QPushButton("Hello world"),
            0, 1)

        menu = self.menuBar()
        file_menu = menu.addMenu("&File")
        file_menu.addAction(button_action)

    def open_file(self):
        """
        Open a Qt file dialog and update the graphic view.

        Returns
        -------
        None
        """
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self,
                                                             "Open Image", "/",
                                                             "Image Files (*.png *.jpg *.bmp)")
        pixmap = QPixmap(file_path)
        if self.pixmap_item:
            self.scene.removeItem(self.pixmap_item)
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.pixmap_item)
