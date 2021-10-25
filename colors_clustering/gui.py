"""
Application related GUI.
"""
from shutil import copyfile
from copy import deepcopy
import traceback
import os

from PySide6.QtCore import Signal, QObject, QThread
from PySide6.QtGui import QAction, QIcon, QPixmap, QIntValidator
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
    QLineEdit,
    QProgressBar,
    QStatusBar,
)
from PySide6 import QtWidgets, QtCore

from algorithms import KMeans, DBScan
from interfaces import (
    AlgorithmType,
    KMeansOptions,
    AlgorithmOptions,
    DBScanOptions,
    DistanceType,
)


class AlgorithmWorker(QObject):
    """
    A worker who manages the calculation of colour clustering.
    """

    result_ready = Signal(str)
    progress = Signal(int)
    stateChanged = Signal(str)

    def do_work(
        self, algorithm: AlgorithmType, file_path: str, options: AlgorithmOptions
    ):
        """
        Do the work.

        Parameters
        ----------
        algorithm : AlgorithmType
            Type of the algorithm.
        file_path : str
            Path to picture.
        options : AlgorithmOptions
            Options of the algorithm.

        Returns
        -------
        None
        """
        try:
            if algorithm == AlgorithmType.KMEANS:
                algo = KMeans(file_path)
                self.stateChanged.emit("Applying KMeans algorithm")
                algo.fit(
                    n_clusters=options.clusters,
                    accuracy=options.accuracy,
                    distance_order=options.distance_type.value,
                )
                self.stateChanged.emit("Exporting result")
                self.result_ready.emit(
                    algo.save(os.path.split(file_path)[0] + "edited.png")
                )
                self.stateChanged.emit("K-Means clustering completed.")
            elif algorithm == AlgorithmType.DBSCAN:
                algo = DBScan(file_path)
                self.stateChanged.emit("Applying DBScan algorithm")
                algo.progress.connect(lambda x: self.progress.emit(x))
                algo.fit(
                    options.minimum_points,
                    options.epsilon,
                    distance_order=options.distance_type.value,
                )
                self.stateChanged.emit("Exporting result")
                self.result_ready.emit(
                    algo.save(os.path.split(file_path)[0] + "edited.png")
                )
                self.stateChanged.emit("DBScan clustering completed.")
        except Exception:
            self.stateChanged.emit("An error as occurred. Please retry.")
            print(traceback.format_exc())


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
        self.accuracy_line_edit = QLineEdit()
        self.distance_type_combo_box = QComboBox()
        self.accuracy_line_edit.setText(str(options.accuracy))

        for i in [8, 16, 32, 64]:
            self.k_value_combo_box.addItem(str(i))
        self.k_value_combo_box.setCurrentText(str(self.options.clusters))

        for distance_type in DistanceType:
            self.distance_type_combo_box.addItem(str(distance_type.name))
        self.distance_type_combo_box.setCurrentText(
            str(self.options.distance_type.name)
        )

        self.layout.addWidget(QLabel("Distance type"), 0, 0)
        self.layout.addWidget(self.distance_type_combo_box, 0, 1)

        self.layout.addWidget(QLabel("Number of clusters"), 1, 0)
        self.layout.addWidget(self.k_value_combo_box, 1, 1)

        self.layout.addWidget(QLabel("Accuracy [0; 255]"), 2, 0)
        self.layout.addWidget(self.accuracy_line_edit, 2, 1)

        self.save_button = QPushButton("Save", self)
        self.save_button.clicked.connect(self.button_pressed)
        self.layout.addWidget(self.save_button, 3, 0)

        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.clicked.connect(self.button_pressed)
        self.layout.addWidget(self.cancel_button, 3, 1)

        self.apply_style()

    def button_pressed(self):
        """
        Handle the press of a button.

        Returns
        -------
        None
        """
        if self.sender() == self.save_button:
            self.options.distance_type = DistanceType[
                self.distance_type_combo_box.currentText()
            ]
            self.options.clusters = int(self.k_value_combo_box.currentText())
            self.options.accuracy = float(self.accuracy_line_edit.text())
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


class DBScanOptionsDialog(QDialog):
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

    def __init__(self, parent, options: DBScanOptions = None):
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

        self.epsilon_value_line_edit = QLineEdit()
        self.nb_points_value_line_edit = QLineEdit()
        self.distance_type_combo_box = QComboBox()

        for distance_type in DistanceType:
            self.distance_type_combo_box.addItem(str(distance_type.name))
        self.distance_type_combo_box.setCurrentText(
            str(self.options.distance_type.name)
        )

        self.epsilon_value_line_edit.setText(str(self.options.epsilon))

        self.layout.addWidget(QLabel("Distance type"), 0, 0)
        self.layout.addWidget(self.distance_type_combo_box, 0, 1)

        self.layout.addWidget(QLabel("Epsilon"), 1, 0)
        self.layout.addWidget(self.epsilon_value_line_edit, 1, 1)

        self.nb_points_value_line_edit.setText(str(self.options.minimum_points))

        self.layout.addWidget(QLabel("Minimum number of points"), 2, 0)
        self.layout.addWidget(self.nb_points_value_line_edit, 2, 1)

        self.save_button = QPushButton("Save", self)
        self.save_button.clicked.connect(self.button_pressed)
        self.layout.addWidget(self.save_button, 3, 0)

        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.clicked.connect(self.button_pressed)
        self.layout.addWidget(self.cancel_button, 3, 1)

        self.apply_style()

    def button_pressed(self):
        """
        Handle the press of a button.

        Returns
        -------
        None
        """
        if self.sender() == self.save_button:
            self.options.distance_type = DistanceType[
                self.distance_type_combo_box.currentText()
            ]
            self.options.epsilon = float(self.epsilon_value_line_edit.text())
            self.options.minimum_points = int(self.nb_points_value_line_edit.text())
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
        self.setWindowTitle("DBScan options")
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
    handle_result()
        Handle the results of finished algorithms.
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
        self.apply_button = QPushButton("Apply")
        self.kmeans_options = KMeansOptions()
        self.dbscan_options = DBScanOptions()
        self.create_image_view()
        self.create_menus()
        self.create_controls()
        self.setCentralWidget(QtWidgets.QWidget())
        self.centralWidget().setLayout(self.layout)

        self.setWindowIcon(QIcon("./colors_clustering/assets/logo.jpg"))

        self.algorithm_worker.result_ready.connect(self.handle_result)
        self.algorithm_worker.progress.connect(self.handle_algorithm_progress)
        self.algorithm_worker.stateChanged.connect(
            lambda x: self.statusBar().showMessage(x)
        )
        self.compute.connect(self.algorithm_worker.do_work)
        self.algorithm_worker.moveToThread(self.algorithm_thread)
        self.algorithm_thread.start()

    def handle_algorithm_progress(self, value):
        self.progress_bar.setValue(value)

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
        self.layout.addWidget(self.original_view, 2, 1, 1, 3)

        self.edited_scene = QGraphicsScene()
        self.edited_view = QGraphicsView(self.edited_scene)
        self.layout.addWidget(self.edited_view, 2, 4, 1, 3)

    def create_menus(self):
        """
        Create all the menus.

        Returns
        -------
        None
        """
        import_action = QAction("&Import", self)
        import_action.triggered.connect(self.open_file)

        self.export_action = QAction("&Export", self)
        self.export_action.triggered.connect(self.export_file)
        self.export_action.setEnabled(False)

        menu = self.menuBar()
        menu.addActions([
            import_action,
            self.export_action
        ])

        self.setStatusBar(QStatusBar())

    # noinspection PyAttributeOutsideInit
    def create_controls(self):
        """
        Create all the controls (buttons, combobox, etc).

        Returns
        -------
        None
        """
        self.algorithm_select_combo_box = QComboBox(self)
        self.layout.addWidget(
            QLabel(
                "Import a picture  >  Selected an algorithm  >  Change the algorithm options  >  Apply it on your picture !"
            ),
            0,
            1,
            1,
            4,
        )
        self.layout.addWidget(self.algorithm_select_combo_box, 1, 1, 1, 4)

        for algorithm in AlgorithmType:
            if not self.selected_algorithm:
                self.selected_algorithm = algorithm
            self.algorithm_select_combo_box.addItem(algorithm.value)
        self.algorithm_select_combo_box.currentTextChanged.connect(
            self.update_algorithm
        )

        self.progress_bar = QProgressBar()
        self.layout.addWidget(self.progress_bar, 3, 1, 1, 6)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        options_button = QPushButton("Options")
        self.layout.addWidget(options_button, 1, 5)
        options_button.clicked.connect(self.open_settings_menu)

        self.layout.addWidget(self.apply_button, 1, 6)
        self.apply_button.clicked.connect(self.apply)

    def open_settings_menu(self):
        """
        Open the settings menu corresponding to the selected algorithm.

        Returns
        -------
        None
        """
        if self.selected_algorithm == AlgorithmType.KMEANS:
            self.kmeans_options = KMeansOptionsDialog.GetOptions(
                self, self.kmeans_options
            )
        elif self.selected_algorithm == AlgorithmType.DBSCAN:
            self.dbscan_options = DBScanOptionsDialog.GetOptions(
                self, self.dbscan_options
            )

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
        Handle the results of finished algorithms.

        Returns
        -------
        None
        """
        self.apply_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.edited_file_path = edited_file_path
        pixmap = QPixmap(self.edited_file_path)
        if self.edited_pixmap_item:
            self.edited_scene.removeItem(self.edited_pixmap_item)
        self.edited_pixmap_item = QGraphicsPixmapItem(pixmap)
        self.edited_scene.addItem(self.edited_pixmap_item)
        self.export_action.setEnabled(True)

    def apply(self):
        """
        Apply the selected algorithm.

        Returns
        -------
        None
        """
        self.compute.emit(
            self.selected_algorithm,
            self.original_file_path,
            self.kmeans_options
            if self.selected_algorithm == AlgorithmType.KMEANS
            else self.dbscan_options,
        )
        if self.edited_pixmap_item:
            self.edited_scene.removeItem(self.edited_pixmap_item)
        self.apply_button.setEnabled(False)
        self.export_action.setEnabled(False)

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
        if self.original_file_path:
            self.edited_file_path = ""
            pixmap = QPixmap(self.original_file_path)
            if self.original_pixmap_item:
                self.original_scene.removeItem(self.original_pixmap_item)
            if self.edited_pixmap_item:
                self.edited_scene.removeItem(self.edited_pixmap_item)
            self.original_pixmap_item = QGraphicsPixmapItem(pixmap)
            self.original_scene.addItem(self.original_pixmap_item)

    def export_file(self):
        if self.edited_file_path:
            storage_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save image", "/", "Image Files (*.png *.jpg *.bmp)"
            )
            copyfile(self.edited_file_path, storage_path)
