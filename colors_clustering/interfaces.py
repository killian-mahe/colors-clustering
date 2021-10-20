from enum import Enum


class AlgorithmType(Enum):
    KMEANS = "K-Means"
    DBSCAN = "DBScan"


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
    pass


class KMeansOptions(AlgorithmOptions):
    def __init__(self):
        self.clusters = 8

    def __str__(self):
        return f"K-Means options : {self.__dict__}"


class DBScanOptions(AlgorithmOptions):
    def __init__(self):
        self.minimum_points = 3
        self.epsilon = 1.5

    def str(self):
        return f"DBScan options : {self.__dict__}"
