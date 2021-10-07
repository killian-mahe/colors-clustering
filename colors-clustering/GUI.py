from PySide6.QtGui import QAction, QIcon, QPixmap
from PySide6.QtWidgets import QMainWindow, QGridLayout, QGraphicsScene, QGraphicsView, QGraphicsPixmapItem, QPushButton
from PySide6 import QtWidgets


class MainWindow(QMainWindow):

    def __init__(self):
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
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.layout.addWidget(self.view, 0, 0)

    def create_menus(self):
        button_action = QAction(QIcon("bug.png"), "&Open", self)
        button_action.setStatusTip("Open a new picture")
        button_action.triggered.connect(self.openFile)
        button_action.setCheckable(True)
        self.layout.addWidget(QPushButton("Hello world"), 0, 1)

        menu = self.menuBar()
        file_menu = menu.addMenu("&File")
        file_menu.addAction(button_action)

    def openFile(self, s):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Image", "/", "Image Files (*.png *.jpg *.bmp)")
        pixmap = QPixmap(file_path)
        if self.pixmap_item:
            self.scene.removeItem(self.pixmap_item)
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.pixmap_item)
