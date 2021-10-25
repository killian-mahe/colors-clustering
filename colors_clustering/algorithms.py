import itertools
from functools import partial
import multiprocessing as mp
import numpy as np
import random
import copy

from PySide6.QtCore import QObject, Signal
import skimage.io

from interfaces import Algorithm

NOISE_POINT = -3


def get_neighbours(
    pixels_map: np.array, point: np.array, epsilon: float, distance_order: int
):
    """
    Get all the neighbours of a point in the given radius.

    Parameters
    ----------
    pixels_map : np.array
        The pixels map of the picture.
    point : np.array
        The point to get the neighbours of.
    epsilon : float
        The radius where to search.
    distance_order : int
        Order of the Frobenius norm.

    Returns
    -------
    list of coordinates.
    """

    distances = np.linalg.norm(
        pixels_map - pixels_map[point[0], point[1]], ord=distance_order, axis=2
    )
    x, y = np.where(distances <= epsilon)
    return list(zip(x, y))


def get_neighbours_multiprocess(pixels_map: np.array, epsilon: float, distance_order: int, point: np.array):
    return [(point[0], point[1]), get_neighbours(pixels_map, point, epsilon, distance_order)]


def picture_to_pixelmap(picture_path: str) -> np.array:
    """
    Convert a image to a np.array of pixels with RGB components.

    Parameters
    ----------
    picture_path : str
        Image file path.

    Returns
    -------
    np.array
    """
    pixel_map = skimage.io.imread(picture_path).copy()

    # Remove alpha component if it's a PNG file.
    if picture_path[-3::] == "png":
        width, height, _ = pixel_map.shape
        return pixel_map[:, :, :3]
    return pixel_map


def color_in_array(array: list[list[int]], subarray: list[int]) -> bool:
    """
    Check if a color is in the given array of colors.

    Parameters
    ----------
    array : list[list[int]]
        Array of colors.
    subarray : list[int]
        Color to search.

    Returns
    -------
    bool
    """
    for i in range(len(array)):
        if (
            subarray[0] == array[i][0]
            and subarray[1] == array[i][1]
            and subarray[2] == array[i][2]
        ):
            return True
    return False


def choose_best_cluster(
    distance_order: int, data: tuple[int, int, list, list]
) -> tuple[int, int, int]:
    """
    Choose the nearest cluster point.

    Parameters
    ----------
    distance_order : int
        Order of the Frobenius norm.
    data : (int, int, list, list)
        Data as (x, y, [R, G, B], list_of_clusters)

    Returns
    -------
    (int, int, int)
    """
    x, y, color, clusters_points = data
    return (
        x,
        y,
        np.argmin(np.linalg.norm(clusters_points - color, ord=distance_order, axis=1)),
    )


class KMeans(Algorithm):
    """
    KMeans algorithm.

    Attributes
    ----------
    pixels_map : np.array
        Pixels map (width, height, 3) with RBG components?
    n_clusters : int
        Number of clusters.
    trained : bool
        Model trained or not.

    Methods
    -------
    fit()
        Train and fit the algorithm.
    export()
        Export the picture and return the pixels map.
    save()
        Export and save the picture.
    create_random_clusters()
        Create new random clusters.
    """

    def __init__(self, picture_path: str):
        """
        Create a KMeans model instance.

        Parameters
        ----------
        picture_path : str
            Path of the picture.
        """
        if picture_path:
            self.pixels_map = picture_to_pixelmap(picture_path)
        else:
            self.pixels_map = None
        self.n_clusters = 0
        self.clusters_points = []
        self.trained = False

    def fit(
        self,
        n_clusters: int = 8,
        accuracy: float = 1,
        distance_order: int = 2,
    ):
        """
        Train and fit the algorithm.

        Parameters
        ----------
        n_clusters : int
            Number of clusters to create.
        accuracy : float
            Maximum mean distance between cluster centers
        distance_order : int
            Order of the Frobenius norm.

        Returns
        -------
        None
        """
        self.n_clusters = n_clusters

        if not self.pixels_map.any():
            raise RuntimeError("You must provide the path of the picture.")

        self.create_random_clusters()

        width, height, _ = self.pixels_map.shape

        self.nearest_cluster = np.zeros((width, height), dtype=np.byte)

        while True:
            colors_map = []
            for x in range(width):
                for y in range(height):
                    colors_map.append(
                        (x, y, self.pixels_map[x][y], self.clusters_points)
                    )

            with mp.Pool(8) as p:
                result = p.map(partial(choose_best_cluster, distance_order), colors_map)

            for (x, y, cluster) in result:
                self.nearest_cluster[x][y] = cluster

            last_clusters = copy.deepcopy(self.clusters_points)

            for j in range(len(self.clusters_points)):
                row, col = np.where(self.nearest_cluster == j)
                if len(self.pixels_map[row, col]):
                    self.clusters_points[j] = np.mean(
                        self.pixels_map[row, col], axis=0
                    ).round()

            if (
                np.mean(abs(np.subtract(last_clusters, self.clusters_points)))
                < accuracy
            ):
                break

        self.trained = True

    def export(self) -> np.array:
        """
        Export the edited picture by applying the trained model and return the new pixels map.

        Returns
        -------
        np.array
        """
        if not self.trained:
            raise RuntimeError("You must train the model first.")

        width, height, _ = self.pixels_map.shape

        result = np.zeros((width, height, 3))

        for x in range(width):
            for y in range(height):
                result[x][y] = self.clusters_points[self.nearest_cluster[x][y]]

        return result

    def save(self, file_path) -> str:
        """
        Export and save the new edited picture in file_path.

        Parameters
        ----------
        file_path : str

        Returns
        -------
        str
        """
        skimage.io.imsave(file_path, self.export())
        return file_path

    def create_random_clusters(self):
        """
        Create new random clusters.

        Returns
        -------
        None
        """
        width, height, _ = self.pixels_map.shape
        for i in range(self.n_clusters):
            r = random.randrange(0, 255)
            g = random.randrange(0, 255)
            b = random.randrange(0, 255)
            while color_in_array(self.clusters_points, [r, g, b]):
                r = random.randrange(0, 255)
                g = random.randrange(0, 255)
                b = random.randrange(0, 255)
            self.clusters_points.append(np.array([r, g, b], dtype=np.short))


class DBScan(Algorithm, QObject):
    progress = Signal(int)

    def __init__(self, picture_path: str):
        super().__init__()

        if picture_path:
            self.pixels_map = picture_to_pixelmap(picture_path)
        else:
            self.pixels_map = None

        self.cluster_mapping = None
        self.nb_cluster = 0
        self.minimum_points = 5
        self.epsilon = 3
        self.trained = False

    def fit(
        self,
        minimum_points: int = 3,
        epsilon: float = 5,
        distance_order: int = 2,
    ):
        if not self.pixels_map.any():
            raise RuntimeError("You must provide the path of the picture.")

        self.minimum_points = minimum_points
        self.epsilon = epsilon

        width, height, _ = self.pixels_map.shape

        self.cluster_mapping = np.zeros((width, height))

        cluster = 0
        to_label_x, to_label_y = np.where(self.cluster_mapping == 0)
        to_label = list(zip(to_label_x, to_label_y))

        initial_number_of_pixels = len(to_label)

        while len(to_label):
            point = to_label[0]
            neighbours = get_neighbours(
                self.pixels_map, point, self.epsilon, distance_order
            )
            if len(neighbours) < self.minimum_points:
                self.cluster_mapping[point[0], point[1]] = NOISE_POINT
            else:
                cluster += 1
                self.cluster_mapping[point[0], point[1]] = cluster
                while len(neighbours):
                    point = neighbours.pop(0)
                    if self.cluster_mapping[point[0], point[1]] == 0:
                        neighbours_bis = get_neighbours(
                            self.pixels_map, point, self.epsilon, distance_order
                        )
                        if len(neighbours_bis) >= self.minimum_points:
                            neighbours = list(set(neighbours_bis) | set(neighbours))
                        else:
                            self.cluster_mapping[point[0], point[1]] = NOISE_POINT
                        self.cluster_mapping[point[0], point[1]] = cluster

            to_label_x, to_label_y = np.where(self.cluster_mapping == 0)
            to_label = list(zip(to_label_x, to_label_y))
            self.progress.emit(
                (initial_number_of_pixels - len(to_label))
                * 100
                / initial_number_of_pixels
            )

        self.nb_cluster = cluster
        self.trained = True

    def save(self, file_path) -> str:
        """
        Export and save the new edited picture in file_path.

        Parameters
        ----------
        file_path : str

        Returns
        -------
        str
        """
        skimage.io.imsave(file_path, self.export())
        return file_path

    def export(self):
        if not self.trained:
            raise RuntimeError("You must train the model first.")

        result = copy.deepcopy(self.pixels_map)

        for i in np.arange(1, self.nb_cluster + 1):
            x, y = np.where(self.cluster_mapping == i)
            colors = result[x, y]
            mean_color = np.sum(colors, axis=0) / len(colors)
            result[x, y] = mean_color

        x, y = np.where(self.cluster_mapping == NOISE_POINT)
        result[x, y] = [0, 0, 0]

        return result
