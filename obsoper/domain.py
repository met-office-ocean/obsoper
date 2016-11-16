# pylint: disable=invalid-name
"""
Model domains
=============

Regular latitude/longitude
--------------------------

For regular latitude/longitude grids a simple bounding box test is all
that is required to determine if a point lies inside the domain.

Irregular boundaries
--------------------

Regional ocean models with irregular boundaries can perform an additional
point in polygon check to determine if a point is inside the domain.

Global models
-------------

Ocean models of global extent typically have a southern boundary, since
Antarctica is a land mass covering the South Pole. A North/South extent
check may be sufficient to determine whether a point belongs to the domain or
not.
"""
from itertools import izip
import numpy as np


class Domain(object):
    """Grid domain definition"""
    def __init__(self, longitudes, latitudes):
        self.bounding_box = Box(np.min(longitudes),
                                np.max(longitudes),
                                np.min(latitudes),
                                np.max(latitudes))
        self.point_in_polygon = PointInPolygon.from2d(longitudes,
                                                      latitudes)

    def contains(self, longitudes, latitudes):
        """check observations are contained within domain"""
        longitudes = np.asarray(longitudes, dtype="d")
        latitudes = np.asarray(latitudes, dtype="d")

        if longitudes.ndim == 0:
            return self.point_in_polygon.inside(longitudes,
                                                latitudes)
        # Optimal vector algorithm
        result = np.zeros_like(longitudes, dtype=np.bool)
        in_box = self.bounding_box.inside(longitudes, latitudes)
        result[in_box] = self.point_in_polygon.inside(longitudes[in_box],
                                                      latitudes[in_box])
        return result

class Box(object):
    """Bounding box surrounding collection of vertices"""
    def __init__(self, xmin, xmax, ymin, ymax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    def inside(self, x, y):
        """Check point lies inside box

        :param x: x-coordinate to test
        :param y: y-coordinate to test
        :returns: logical indicating coordinates contained in box
        """
        return ((x <= self.xmax) &
                (x >= self.xmin) &
                (y <= self.ymax) &
                (y >= self.ymin))


def boundary(values):
    """Extract boundary from grid

    A boundary of a grid shaped N x M consists of 2N + 2M - 4 points.
    2 rows, 2 columns minus 4 corners that have been double counted.

    :param values: 2D array shaped (N, M)
    :returns: array shaped (B, 2) where B is the number of points on the
              boundary (2N + 2M - 4).
    """
    values = np.asarray(values)
    return np.asarray(list(values[:, 0]) +
                      list(values[-1, 1:-1]) +
                      list(values[::-1, -1]) +
                      list(values[0, -2:0:-1]),
                      dtype="d")


def point_in_polygon(x, y, xp, yp):
    """Point in polygon algorithm"""
    return PointInPolygon(x, y).inside(xp, yp)


class PointInPolygon(object):
    """Point in polygon search algorithm

    The algorithm proceeds as follow:

        - Detect faces that contain the y-coordinate of the test point
        - Find x-coordinate (nodes) for the faces at the test y-coordinate
        - Count nodes either side of the test point
        - An odd number of nodes on both sides means the point is inside
        - If the test point is a node (lies on boundary) it is also
          counted as inside

    In cases where two faces meet the lower face is counted as having a node.
    This convention removes double counting. As a result,
    points at the top of the polygon need to be treated separately.

    :param x: array of x coordinates of polygon vertices
    :param y: array of y coordinates of polygon vertices
    """
    def __init__(self, x, y):
        self.x, self.y = np.asarray(x), np.asarray(y)

        # Define valid line segments
        x1, y1, x2, y2 = as_segments(self.x, self.y)
        self.x1, self.y1, self.x2, self.y2 = remove_horizontal(x1, y1, x2, y2)

        # Detect intervals containing f(x) = yp
        self.y_min, self.y_max = order_intervals(self.y1, self.y2)

        # Determine y-axis grid limit
        self.y_limit = np.max([self.y1, self.y2])

    @classmethod
    def from2d(cls, longitudes, latitudes):
        """Construct point in polygon search from 2D longitudes and latitudes

        Conveniently maps array boundaries to polygon definition

        :param longitudes: array shape (X, Y)
        :param latitudes: array shape (X, Y)
        """
        return cls(boundary(longitudes), boundary(latitudes))

    def inside(self, xp, yp):
        """Check point(s) lie inside polygon"""
        xp, yp = np.asarray(xp), np.asarray(yp)
        if xp.ndim == 0:
            return self._scalar_inside(xp, yp)
        return self._vector_inside(xp, yp)

    def _vector_inside(self, xp, yp):
        return np.array([self._scalar_inside(x, y) for x, y in izip(xp, yp)],
                        dtype=np.bool)

    def _scalar_inside(self, xp, yp):
        # Apply algorithm to points at top of domain
        if yp == self.y_limit:
            nodes = self.x[self.y == self.y_limit]
            return self.between_nodes(nodes, xp)

        # Detect intervals containing f(x) = yp
        points = interval_contains(self.y_min, self.y_max, yp)

        # Check that nodes exist
        if len(self.x1[points]) == 0:
            return False

        # Find x-values corresponding to yp for each segment
        nodes = solve(self.x1[points],
                      self.y1[points],
                      self.x2[points],
                      self.y2[points],
                      yp)
        return self.between_nodes(nodes, xp)

    @staticmethod
    def between_nodes(nodes, position):
        """Check position in relation to node positions

        A point is inside the domain for one of two reasons, either:
           - there are an odd number of nodes on either side of the point
           - the point is on the boundary (is a node position)

        :returns: True if position is in domain
        """
        # Include solutions on boundary
        if position in nodes:
            return True
        # Count nodes left/right of xp
        return (odd(count_below(nodes, position)) and
                odd(count_above(nodes, position)))


def as_segments(x, y):
    """Convert coordinates representing polygon to segments"""
    return x, y, cycle(x), cycle(y)


def remove_horizontal(x1, y1, x2, y2):
    """Remove segments with zero slope"""
    keep = y1 != y2
    return x1[keep], y1[keep], x2[keep], y2[keep]


def cycle(values):
    """Shift array view in a cyclic manner"""
    return np.append(values[1:], values[0])


def order_intervals(left, right):
    """Rearrange intervals into ascending order

    :param left: left interval values
    :param right: right interval values
    :returns: (minimum, maximum) arrays sorted into lower/upper values
    """
    return np.min([left, right], axis=0), np.max([left, right], axis=0)


def interval_contains(minimum, maximum, value):
    """Determine if interval contains point

    .. note:: zero sized intervals do not contain points

    .. note:: interval is closed to the left and open on the right
    """
    minimum, maximum = np.asarray(minimum), np.asarray(maximum)
    return (minimum <= value) & (value < maximum)


def solve(x1, y1, x2, y2, y):
    """Solve equation of line for x given y

    This is the inverse of the usual approach to solving a linear equation.
    Linear equations can be solved forward for y or backward for x using the
    same form of equation, y0 + (dy / dx) * (x - x0). In this case with
    y and x switched, the equation reads x0 + (dx / dy) * (y - y0).

    :returns: value that satisfies line defined by (x1, y1), (x2, y2)
    """
    dxdy = (x2 - x1) / (y2 - y1)
    return x1 + (dxdy * (y - y1))


def count_below(values, threshold):
    """Count number of values lying below some threshold"""
    return (values < threshold).sum()


def count_above(values, threshold):
    """Count number of values lying above some threshold"""
    return (values > threshold).sum()


def odd(number):
    """Determine if number is odd"""
    return (number % 2) == 1
