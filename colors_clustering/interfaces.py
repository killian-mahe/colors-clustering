from enum import Enum


class AlgorithmType(Enum):
    """All the different types of supported algorithms."""

    KMEANS = "K-Means"
    DBSCAN = "DBScan"


class DistanceType(Enum):
    EUCLIDEAN = 2
    MANHATTAN = 1


class Algorithm:
    """
    Abstract class for algorithm.
    """

    def fit(self):
        """
        Train and fit the algorithm.

        Returns
        -------
        None
        """
        return NotImplementedError

    def export(self):
        """
        Export the picture and return the pixels map.

        Returns
        -------
        None
        """
        return NotImplementedError


class AlgorithmOptions:
    def __init__(self):
        self.distance_type = DistanceType.EUCLIDEAN


class KMeansOptions(AlgorithmOptions):
    def __init__(self):
        super().__init__()
        self.clusters = 8
        self.accuracy = 1

    def __str__(self):
        return f"K-Means options : {self.__dict__}"


class DBScanOptions(AlgorithmOptions):
    def __init__(self):
        super().__init__()
        self.minimum_points = 3
        self.epsilon = 1.5

    def str(self):
        return f"DBScan options : {self.__dict__}"
