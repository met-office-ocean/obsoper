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
    :returns: array shaped (2, B) where B is the number of points on the
              boundary (2N + 2M - 4).
    """
    return np.asarray(zip(longitudes[:, 0], latitudes[:, 0]) +
                      zip(longitudes[-1, 1:-1], latitudes[-1, 1:-1]) +
                      zip(longitudes[::-1, -1], latitudes[::-1, -1]) +
                      zip(longitudes[0, -2:0:-1], latitudes[0, -2:0:-1]),
                      dtype="d")


def interval_contains(x1, x2, x):
    """Determine if interval contains point"""
    return x1 < x < x2


def solve(x1, y1, x2, y2, x):
    """Solve equation of line for y given x

    :returns: value of y that satisfies line defined by (x1, y1), (x2, y2)
    """
    dydx = (y2 - y1) / (x2 - x1)
    return  y1 + (dydx * (x - x1))


def point_in_polygon(polygon, point):
    """Determine if points lie inside polygon"""
    for vertex in polygon:
        if np.allclose(vertex, point):
            return True

    # Ray casting algorithm
    ray = np.asarray([point_outside(polygon), point], dtype="d")
    return (count_intersects(polygon, ray) % 2) == 1


def count_intersects(polygon, ray):
    """Check that a ray cast through a polygon counts intersects correctly"""
    intersects = 0
    for face in faces(polygon):
        if segments_intersect(face, ray):
            if np.allclose(crossing_point(face, ray), face[0]):
                continue
            intersects += 1
    return intersects


def point_outside(polygon, epsilon=0.1):
    """Generate point that lies outside of a polygon"""
    return np.mean(polygon[:, 0]), np.min(polygon[:, 1]) - epsilon


def faces(polygon):
    """Iterate over polygon"""
    nsides = len(polygon)
    for iside in range(nsides):
        yield polygon[iside], polygon[(iside + 1) % nsides]


def crossing_point(line_1, line_2):
    """calculate crossing point of two intersecting line segments"""
    # pylint: disable=invalid-name
    line_1, line_2 = np.asarray(line_1), np.asarray(line_2)
    p = line_1[0]
    r = line_1[1] - line_1[0]
    q = line_2[0]
    s = line_2[1] - line_2[0]
    t = np.cross((q - p), s) / np.cross(r, s)
    return p + (t * r)


def segments_intersect(line_1, line_2):
    """Determine if two line segments intersect

    A line is represented as a 2 by 2 array, 2 points and 2 dimensions.

    :param line_1: array shaped (2, 2)
    :param line_2: array shaped (2, 2)
    :returns: True is line segments touch or intersect
    """
    line_1, line_2 = np.asarray(line_1), np.asarray(line_2)

    # X axis boundary check
    if disjoint(line_1[:, 0], line_2[:, 0]):
        return False

    # Y axis boundary check
    if disjoint(line_1[:, 1], line_2[:, 1]):
        return False

    # Both points not on same side of line
    # if side_1 or side_2 are zero then point is on a line
    side_1 = side(line_1, line_2[0])
    side_2 = side(line_1, line_2[1])
    return (side_1 * side_2) <= 0


def disjoint(interval_1, interval_2):
    """Determine if 1D intervals are disjoint

    An interval is defined by two values on a 1D axis. Two intervals
    are disjoint if their nearest end points have space between them.

    :param interval_1: length 2 array
    :param interval_2: length 2 array
    :returns: True is intervals do not overlap
    """
    return ((np.min(interval_1) > np.max(interval_2)) or
            (np.max(interval_1) < np.min(interval_2)))


def side(line, point):
    """Determine which side a point lies on

    A side is defined by taking the cross product of the point with the line
    after both have been translated so that the line touches the origin.

    :returns: number where positive, zero and negative indicate right, on and
              left sides of a line respectively
    """
    # pylint: disable=invalid-name
    xp, yp = point
    (x1, y1), (x2, y2) = line
    return ((yp - y1) * (x2 - x1)) - ((xp - x1) * (y2 - y1))
