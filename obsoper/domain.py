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
import numpy as np
from . import box


class Domain(object):
    """Grid domain definition"""
    def __init__(self, longitudes, latitudes):
        self.bounding_box = box.Box(np.min(longitudes),
                                    np.max(longitudes),
                                    np.min(latitudes),
                                    np.max(latitudes))

    def contains(self, longitudes, latitudes):
        """check observations are contained within domain"""
        return self.bounding_box.inside(longitudes, latitudes)


def boundary(longitudes, latitudes):
    """Extract boundary from grid

    A boundary of a grid shaped N x M consists of 2N + 2M - 4 points.
    2 rows, 2 columns minus 4 corners that have been double counted.

    :param longitudes: 2D array shaped (N, M)
    :param latitudes: 2D array shaped (N, M)
    :returns: array shaped (B, 2) where B is the number of points on the
              boundary (2N + 2M - 4).
    """
    return np.asarray(zip(longitudes[:, 0], latitudes[:, 0]) +
                      zip(longitudes[-1, 1:-1], latitudes[-1, 1:-1]) +
                      zip(longitudes[::-1, -1], latitudes[::-1, -1]) +
                      zip(longitudes[0, -2:0:-1], latitudes[0, -2:0:-1]),
                      dtype="d")


def point_in_polygon(polygon, point):
    """Determine if points lie inside polygon"""
    return algorithm(polygon[:, 0], polygon[:, 1], point[0], point[1])


def algorithm(x, y, xp, yp):
    """Point in polygon algorithm"""
    search = PolygonSearch(x, y)
    return search.inside(xp, yp)


class PolygonSearch(object):
    """Point in polygon search algorithm"""
    def __init__(self, x, y):
        self.x, self.y = np.asarray(x), np.asarray(y)

        # Define valid line segments
        self.x1, self.y1, self.x2, self.y2 = valid_segments(self.x, self.y)

        # Detect intervals containing f(x) = yp
        self.y_min, self.y_max = order_intervals(self.y1, self.y2)

        # Determine y-axis grid limit
        self.y_limit = np.max([self.y1, self.y2])

    def inside(self, xp, yp):
        """Check point(s) lie inside polygon"""
        xp, yp = np.asarray(xp), np.asarray(yp)
        if xp.ndim == 0:
            return self._scalar_inside(xp, yp)
        return self._vector_inside(xp, yp)

    def _vector_inside(self, xp, yp):
        result = np.zeros_like(xp, dtype=np.bool)
        for i, (x, y) in enumerate(zip(xp, yp)):
            result[i] = self._scalar_inside(x, y)
        return result

    def _scalar_inside(self, xp, yp):
        # Apply algorithm to points at top of domain
        if yp == self.y_limit:
            nodes = self.x[self.y == self.y_limit]
            if xp in nodes:
                return True
            return odd(count_below(nodes, xp)) and odd(count_above(nodes, xp))

        # Detect intervals containing f(x) = yp
        points = interval_contains(self.y_min, self.y_max, yp)

        # Check that nodes exist
        if not points.any():
            return False

        # Find x-values corresponding to yp for each segment
        nodes = solve(self.x1[points],
                      self.y1[points],
                      self.x2[points],
                      self.y2[points],
                      yp)

        # Include solutions on boundary
        if xp in nodes:
            return True

        # Count nodes left/right of xp
        return odd(count_below(nodes, xp)) and odd(count_above(nodes, xp))


def valid_segments(x, y):
    """Convert coordinates representing polygon to segments used by algorithm

    .. note:: segments parallel to x-axis are removed since algorithm
              can't search line segments with dy equal to zero.
    """
    # pylint: disable=invalid-name
    x1, y1 = x, y
    x2, y2 = cycle(x), cycle(y)
    return remove_horizontal(x1, y1, x2, y2)


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
