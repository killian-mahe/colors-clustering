"""
Application related GUI.
"""
from copy import deepcopy
import os

from PySide6.QtCore import Signal, QObject, QThread
from PySide6.QtGui import QAction, QIcon, QPixmap
from PySide6.QtWidgets import (
    QMainWindow,
    QGridLayout,
    QGraphicsScene,
    QGraphicsView,
    QGraphicsPixmapItem,
    QPushButton,
    QComboBox,
    QLabel,
    QDialog,
)
from PySide6 import QtWidgets, QtCore

from colors_clustering.algorithms import KMeans
from interfaces import AlgorithmType, KMeansOptions, AlgorithmOptions


class AlgorithmWorker(QObject):

    result_ready = Signal(str)

    def do_work(self, algorithm, file_path, options):
        if algorithm == AlgorithmType.KMEANS:
            algo = KMeans(file_path)
            algo.fit(n_clusters=options.clusters)
            self.result_ready.emit(algo.save(
                os.path.split(file_path)[0] + "edited.png"
            ))
        else:
            raise NotImplementedError()


class KMeansOptionsDialog(QDialog):
    """
    A class to let the user choose KMeans algorithm options.

    Attributes
    ----------
    layout : QGridLayout
        Dialog grid layout.
    initial_options : KMeansOptions
        Initial options (by default or not).
    options : KMeansOptions
        Selected options.

    Methods
    -------
    button_pressed()
        Handle the press of the dialog buttons.
    apply_style()
        Apply the custom style on the dialog.
    """

    def __init__(self, parent, options: KMeansOptions = None):
        """
        Construct the dialog instance.

        Parameters
        ----------
        parent : QtWidget
            Qt parent.
        options : KMeansOptions
            Default options.
        """
        super().__init__(parent)
        if options is None:
            options = KMeansOptions()
        self.initial_options = options
        self.options = deepcopy(options)

        self.layout = QGridLayout()
        self.setLayout(self.layout)

        self.k_value_combo_box = QComboBox()

        for i in [8, 16, 32, 64]:
            self.k_value_combo_box.addItem(str(i))
        self.k_value_combo_box.setCurrentText(str(self.options.clusters))

        self.layout.addWidget(QLabel("Number of clusters"), 0, 0)
        self.layout.addWidget(self.k_value_combo_box, 0, 1)

        self.save_button = QPushButton("Save", self)
        self.save_button.clicked.connect(self.button_pressed)
        self.layout.addWidget(self.save_button, 1, 0)

        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.clicked.connect(self.button_pressed)
        self.layout.addWidget(self.cancel_button, 1, 1)

        self.apply_style()

    def button_pressed(self):
        """
        Handle the press of a button.

        Returns
        -------
        None
        """
        if self.sender() == self.save_button:
            self.options.clusters = int(self.k_value_combo_box.currentText())
        else:
            self.options = self.initial_options
        self.close()

    def apply_style(self):
        """
        Apply the style on the dialog.

        Returns
        -------
        None
        """
        self.setWindowTitle("K-Means options")
        self.setMinimumWidth(300)
        self.layout.setAlignment(QtCore.Qt.AlignTop)

    @classmethod
    def GetOptions(cls, parent, *args):
        """
        Open a KMeansOptionsDialog to get the algorithm parameters.

        Parameters
        ----------
        parent : QtWidget
            Qt parent.
        args
            Additional args.

        Returns
        -------
        KMeansOptions
        """
        dialog = cls(parent, *args)
        dialog.exec_()

        return dialog.options


class MainWindow(QMainWindow):
    """
    A class to represent the app main window.

    Attributes
    ----------
    layout : QGridLayout
        app grid layout.
    original_pixmap_item : QGraphicsPixmapItem
        original picture representation.
    original_scene : QGraphicsScene
        graphical scene for the original picture representation.
    original_view : QGraphicView
        view that display the scene in the main window.
    algorithm_select_combo_box : QComboBox
        Qt Combobox that handle algorithm selection.

    Methods
    -------
    create_image_view()
        Create the original picture view.
    create_menus()
        Create the menus displayed in the main window.
    open_file()
        Create a file dialog and update the selected picture.
    apply()
        Apply the selected algorithm.
    """
    compute = Signal(AlgorithmType, str, AlgorithmOptions)

    def __init__(self):
        """
        Constructs all the necessary attributes for the main window object.
        """
        super().__init__()
        self.setWindowTitle("Colors clustering")
        self.layout = QGridLayout()
        self.original_pixmap_item = None
        self.original_file_path = ""
        self.edited_pixmap_item = None
        self.edited_file_path = ""
        self.selected_algorithm = None
        self.algorithm_thread = QThread()
        self.algorithm_worker = AlgorithmWorker()
        self.kmeans_options = KMeansOptions()
        self.create_image_view()
        self.create_menus()
        self.create_controls()
        self.setCentralWidget(QtWidgets.QWidget())
        self.centralWidget().setLayout(self.layout)

        self.algorithm_worker.result_ready.connect(self.handle_result)
        self.compute.connect(self.algorithm_worker.do_work)
        self.algorithm_worker.moveToThread(self.algorithm_thread)
        self.algorithm_thread.start()

    # noinspection PyAttributeOutsideInit
    def create_image_view(self):
        """
        Create all the necessary scene and view for the original picture.

        Returns
        -------
        None
        """
        self.original_scene = QGraphicsScene()
        self.original_view = QGraphicsView(self.original_scene)
        self.layout.addWidget(self.original_view, 2, 0, 1, 3)

        self.edited_scene = QGraphicsScene()
        self.edited_view = QGraphicsView(self.edited_scene)
        self.layout.addWidget(self.edited_view, 2, 3, 1, 3)

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

        menu = self.menuBar()
        file_menu = menu.addMenu("&File")
        file_menu.addAction(button_action)

    # noinspection PyAttributeOutsideInit
    def create_controls(self):
        """
        Create all the controls (buttons, combobox, etc).

        Returns
        -------
        None
        """
        self.algorithm_select_combo_box = QComboBox(self)
        self.layout.addWidget(QLabel("Selected algorithm"), 0, 0, 1, 4)
        self.layout.addWidget(self.algorithm_select_combo_box, 1, 0, 1, 4)

        for algorithm in AlgorithmType:
            if not self.selected_algorithm:
                self.selected_algorithm = algorithm
            self.algorithm_select_combo_box.addItem(algorithm.value)
        self.algorithm_select_combo_box.currentTextChanged.connect(
            self.update_algorithm
        )

        options_button = QPushButton("Options")
        self.layout.addWidget(options_button, 1, 4)
        options_button.clicked.connect(self.open_settings_menu)

        apply_button = QPushButton("Apply")
        self.layout.addWidget(apply_button, 1, 5)
        apply_button.clicked.connect(lambda x: self.compute.emit(self.selected_algorithm, self.original_file_path, self.kmeans_options))

    def open_settings_menu(self):
        self.kmeans_options = KMeansOptionsDialog.GetOptions(self, self.kmeans_options)

    def update_algorithm(self, algorithm_name: str):
        """
        Update the selected algorithm.

        Parameters
        ----------
        algorithm_name : str
            Selected algorithm name.

        Returns
        -------
        None
        """
        self.selected_algorithm = AlgorithmType(algorithm_name)

    def handle_result(self, edited_file_path):
        """
        Apply the selected algorithm.

        Returns
        -------
        None
        """
        self.edited_file_path = edited_file_path
        pixmap = QPixmap(self.edited_file_path)
        if self.edited_pixmap_item:
            self.edited_scene.removeItem(self.edited_pixmap_item)
        self.edited_pixmap_item = QGraphicsPixmapItem(pixmap)
        self.edited_scene.addItem(self.edited_pixmap_item)

    def open_file(self):
        """
        Open a Qt file dialog and update the graphic view.

        Returns
        -------
        None
        """
        self.original_file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Image", "/", "Image Files (*.png *.jpg *.bmp)"
        )
        pixmap = QPixmap(self.original_file_path)
        if self.original_pixmap_item:
            self.original_scene.removeItem(self.original_pixmap_item)
        if self.edited_pixmap_item:
            self.edited_scene.removeItem(self.edited_pixmap_item)
        self.original_pixmap_item = QGraphicsPixmapItem(pixmap)
        self.original_scene.addItem(self.original_pixmap_item)
