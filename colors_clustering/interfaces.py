from enum import Enum


class Algorithm(Enum):
    KMEANS = "K-Means"
    DBSCAN = "DBScan"


class KMeansOptions:
    def __init__(self):
        self.clusters = 8

    def __str__(self):
        return f"K-Means options : {self.__dict__}"
