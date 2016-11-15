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
    for vertex in polygon:
        if np.allclose(vertex, point):
            return True
    return algorithm(polygon[:, 0], polygon[:, 1], point[0], point[1])


def algorithm(x, y, xp, yp):
    """Point in polygon algorithm"""
    x, y = np.asarray(x), np.asarray(y)

    # Detect intervals containing f(x) = yp
    ymin, ymax = order_intervals(y, cycle(y))
    points = interval_contains(ymin, ymax, yp)

    # Check that nodes exist
    if not points.any():
        return False

    # Find x-values corresponding to yp for each segment
    nodes = solve(x[points], y[points], cycle(x)[points], cycle(y)[points], yp)

    # Count nodes left/right of xp
    return odd(count_below(nodes, xp)) and odd(count_above(nodes, xp))


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
    """Determine if interval contains point"""
    minimum, maximum = np.asarray(minimum), np.asarray(maximum)
    return (minimum < value) & (value < maximum)


def solve(x1, y1, x2, y2, y):
    """Solve equation of line for x given y

    This is the inverse of the usual approach to solving a linear equation.
    Linear equations can be solved forward for y or backward for x using the
    same form of equation, y0 + (dy / dx) * (x - x0). In this case with
    y and x switched, the equation reads x0 + (dx / dy) * (y - y0).

    :returns: value that satisfies line defined by (x1, y1), (x2, y2)
    """
    dxdy = (x2 - x1) / (y2 - y1)
    return  x1 + (dxdy * (y - y1))


def count_below(values, threshold):
    """Count number of values lying below some threshold"""
    return (values < threshold).sum()


def count_above(values, threshold):
    """Count number of values lying above some threshold"""
    return (values > threshold).sum()


def odd(number):
    """Determine if number is odd"""
    return (number % 2) == 1
