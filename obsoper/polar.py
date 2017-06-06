"""Polar search functions"""
import numpy as np


def search():
    """Search function"""


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
