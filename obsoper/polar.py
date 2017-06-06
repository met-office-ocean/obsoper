"""Polar search functions"""
import numpy as np
from scipy import spatial


def search():
    """Search function"""


def grid_structure():
    """Estimate grid from points"""


def join_simplices(simplex_1, simplex_2, face_index):
    """Join 2 simplices given face index

    Insert extra index in correct position of first simplex

    :return: simplex representing joined simplices
    """
    extra_index = list(set(simplex_2) - set(simplex_1))[0]
    result = list(simplex_1)
    result.insert(face_index + 1, extra_index)
    return result


class Grid(object):
    """Estimate grid from points
    
    .. note: Uses Delaunay triangulation to search grid
    """
    def __init__(self, points):
        self.points = points
        self.triangulation = spatial.Delaunay(points)
        self.cells = [[0, 1, 2, 3]]

    def find_cell(self, position):
        return 0


def longest_side(points):
    """Longest side of a polygon"""
    return np.argmax(side_lengths(points))


def side_lengths(points):
    """Polygon side lengths"""
    n_sides = len(points)
    lengths = []
    for iside in range(n_sides):
        point_1 = points[iside]
        point_2 = points[(iside + 1) % n_sides]
        length = distance(point_1, point_2)
        lengths.append(length)
    return lengths


def distance(point_1, point_2):
    """Cartesian distance between two points"""
    return np.sqrt(((point_2 - point_1)**2).sum())
