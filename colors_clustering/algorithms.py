import queue
from functools import partial
import multiprocessing as mp
import itertools
import random
import copy
import math

import numpy as np
import skimage.io

from interfaces import Algorithm

CORE_POINT = -1
EDGE_POINT = -2
NOISE_POINT = -3


def neighbour_points(data, epsilon, coord: tuple):
    point = data[coord[0], coord[1]]

    distances = np.linalg.norm(data - point, axis=2)
    x, y = np.where(distances <= epsilon)
    return [coord[0], coord[1]] + list(zip(x, y))


def get_neighbours(pixels_map, point, epsilon):

    distances = np.linalg.norm(pixels_map - pixels_map[point[0], point[1]], axis=2)
    x, y = np.where(distances <= epsilon)
    return list(zip(x, y))


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
        pixel_map.resize((width, height, 3))
    return pixel_map


def color_in_array(array: list[list[int]], subarray: list[int]) -> bool:
    """
    Check if a color is in the given array of colors.

    Parameters
    ----------
    array : list[list[int]]
        Array of colors.
    subarray : list[int]
        Color tp search.

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


def color_distance(r1: int, g1: int, b1: int, r2: int, g2: int, b2: int) -> float:
    """
    Compute the euclidean distance between two RGB colors.

    Parameters
    ----------
    r1 : int
        RED component of the first color.
    g1 : int
        GREEN component of the first color.
    b1 : int
        BLUE component of the first color.
    r2 : int
        RED component of the second color.
    g2 : int
        GREEN component of the second color.
    b2 : int
        BLUE component of the second color.

    Returns
    -------
    float
    """
    return math.sqrt(
        pow(float(r1) - r2, 2) + pow(float(g1) - g2, 2) + pow(float(b1) - b2, 2)
    )


def color_distance_matrix(color_a, color_b):
    return np.linalg.norm(color_b - color_a)


def choose_best_cluster(data: tuple[int, int, list, list]) -> tuple[int, int, int]:
    """
    Choose the nearest cluster point.

    Parameters
    ----------
    data : (int, int, list, list)
        Data as (x, y, [R, G, B], list_of_clusters)
    Returns
    -------
    (int, int, int)
    """
    distances = []
    x, y, color, clusters_points = data
    for i in range(len(clusters_points)):
        distances.append(color_distance(*clusters_points[i][2], *color))
    return x, y, distances.index(min(distances))


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

    def __init__(self, picture_path: str = None):
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

    def fit(self, picture_path: str = None, n_clusters: int = 8):
        """
        Train and fit the algorithm.

        Parameters
        ----------
        picture_path : str
            Path of the picture.
        n_clusters : int
            Number of clusters to create.

        Returns
        -------
        None
        """
        self.n_clusters = n_clusters

        if picture_path:
            self.pixels_map = picture_to_pixelmap(picture_path)
        if not self.pixels_map.any():
            raise RuntimeError("You must provide the path of the picture.")

        self.create_random_clusters()

        width, height, _ = self.pixels_map.shape

        new_clusters_points = [[0, 0, 0] for i in range(n_clusters)]
        self.nearest_cluster = copy.deepcopy(self.pixels_map)
        self.nearest_cluster.resize((width, height, 1))

        colors_map = []
        for x in range(width):
            for y in range(height):
                colors_map.append((x, y, self.pixels_map[x][y], self.clusters_points))

        with mp.Pool(8) as p:
            result = p.map(choose_best_cluster, colors_map)

        for (x, y, cluster) in result:
            self.nearest_cluster[x][y] = cluster

        for x in range(width):
            for y in range(height):
                cluster_index = self.nearest_cluster[x][y][0]
                new_clusters_points[cluster_index] = np.add(
                    new_clusters_points[cluster_index], self.pixels_map[x][y]
                )

        unique, counts = np.unique(self.nearest_cluster, return_counts=True)
        for index, count in list(zip(unique, counts)):
            new_clusters_points[index] = (new_clusters_points[index] / count).round()

        self.clusters_points = new_clusters_points
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

        result = copy.deepcopy(self.pixels_map)

        for x in range(width):
            for y in range(height):
                result[x][y] = self.clusters_points[self.nearest_cluster[x][y][0]]

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
            x = random.randrange(0, width)
            y = random.randrange(0, height)
            while color_in_array(self.clusters_points, self.pixels_map[x][y]):
                x = random.randrange(0, width)
                y = random.randrange(0, height)
            self.clusters_points.append([x, y, self.pixels_map[x][y]])


class DBScan(Algorithm):

    def __init__(self, picture_path: str, minimum_points: int = 3, epsilon: float = 5):
        super().__init__()

        if picture_path:
            self.pixels_map = picture_to_pixelmap(picture_path)
        else:
            self.pixels_map = None

        self.cluster_mapping = None
        self.nb_cluster = 0
        self.minimum_points = minimum_points
        self.epsilon = epsilon
        self.trained = False

    def fit_old(self, picture_path: str = None):
        if picture_path:
            self.pixels_map = picture_to_pixelmap(picture_path)
        if not self.pixels_map.any():
            raise RuntimeError("You must provide the path of the picture.")

        width, height, _ = self.pixels_map.shape

        points = list(itertools.product(np.arange(width), np.arange(height)))

        func = partial(neighbour_points, self.pixels_map, self.epsilon)

        with mp.Pool(mp.cpu_count()) as p:
            neighbours_map = p.map(func, points)

        print("Finished")

        neighbours = np.empty((width, height), dtype=object)

        for point in neighbours_map:
            neighbours[point[0], point[1]] = point[2:]

        point_label = np.empty((width, height))

        for x in range(width):
            for y in range(height):
                if len(neighbours[x, y]) >= self.minimum_points:
                    point_label[x, y] = CORE_POINT
                else:
                    point_label[x, y] = NOISE_POINT

        x, y = np.where(point_label == NOISE_POINT)

        for point in zip(x, y):
            for neighbour in neighbours[point[0], point[1]]:
                if point_label[neighbour[0], neighbour[1]] == CORE_POINT:
                    point_label[point[0], point[1]] = EDGE_POINT
                    break

        print(f"(55, 196) label : {point_label[55, 196]}")

        cluster = 1

        for x in range(width):
            for y in range(height):
                q = queue.Queue()
                if point_label[x, y] == CORE_POINT:
                    q.put((x, y))

                    while not q.empty():
                        core_x, core_y = q.get()
                        for neighbour in neighbours[core_x, core_y]:
                            if (neighbour[0], neighbour[1]) == (55, 204):
                                print(
                                    f"(55, 204) is tested as {point_label[neighbour[0], neighbour[1]]} as the neighbour of ({core_x}, {core_y})")
                            if point_label[neighbour[0], neighbour[1]] == CORE_POINT:
                                q.put(neighbour)
                                point_label[neighbour[0], neighbour[1]] = cluster
                            elif point_label[neighbour[0], neighbour[1]] == EDGE_POINT:
                                point_label[neighbour[0], neighbour[1]] = cluster
                    cluster += 1

        print(f"{cluster} clusters")

        self.nb_cluster = cluster - 1

        print(point_label)

        x, y = np.where(point_label == EDGE_POINT)
        print(f"Neighbours of (55, 204) (Edge) : {neighbours[55, 204]}")
        print(f"Neighbours of (55, 196) (Core) : {neighbours[55, 196]}")

        self.cluster_mapping = point_label
        self.trained = True

    def fit(self, picture_path: str = None):
        if picture_path:
            self.pixels_map = picture_to_pixelmap(picture_path)
        if not self.pixels_map.any():
            raise RuntimeError("You must provide the path of the picture.")

        width, height, _ = self.pixels_map.shape

        self.cluster_mapping = np.zeros((width, height))

        cluster = 0
        to_label_x, to_label_y = np.where(self.cluster_mapping == 0)
        to_label = list(zip(to_label_x, to_label_y))

        print(len(to_label))
        while len(to_label):
            point = to_label[0]
            neighbours = get_neighbours(self.pixels_map, point, self.epsilon)
            if len(neighbours) < self.minimum_points:
                self.cluster_mapping[point[0], point[1]] = -1
            else:
                cluster += 1
                self.cluster_mapping[point[0], point[1]] = cluster
                for point in neighbours:
                    if self.cluster_mapping[point[0], point[1]] == 0:
                        neighbours_bis = get_neighbours(self.pixels_map, point, self.epsilon)
                        if len(neighbours_bis) >= self.minimum_points:
                            neighbours += neighbours_bis
                    if self.cluster_mapping[point[0], point[1]] == 0:
                        self.cluster_mapping[point[0], point[1]] = cluster
            to_label_x, to_label_y = np.where(self.cluster_mapping == 0)
            to_label = list(zip(to_label_x, to_label_y))
            print(len(to_label))
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
        result[x, y] = [255, 255, 255]

        return result
