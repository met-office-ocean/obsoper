"""
Grid cells
"""
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport abs
from . import spherical
from . import box
from . cimport (box,
                spherical)


def contains(vertices, x, y):
    """Vectorized approach to Cell membership testing

    .. note:: vertices first dimension must equal x, y lengths
    """
    vertices = np.asarray(vertices, dtype="d")
    x = np.asarray(x, dtype="d")
    y = np.asarray(y, dtype="d")
    return np.array(cy_contains(vertices, x, y), dtype=np.bool)


def contains(vertices, x, y):
    """Vectorized approach to Cell membership testing

    .. note:: vertices first dimension must equal x, y lengths
    """
    vertices = np.asarray(vertices, dtype="d")
    x = np.asarray(x, dtype="d")
    y = np.asarray(y, dtype="d")
    return np.array(cy_contains(vertices, x, y), dtype=np.bool)


cdef np.uint8_t[:] cy_contains(
        double[:, :, :] vertices,
        double[:] x,
        double[:] y):
    cdef Py_ssize_t i
    cdef np.uint8_t[:] inside = np.zeros(vertices.shape[0], dtype=np.uint8)
    for i in range(vertices.shape[0]):
        inside[i] = same_side_test(vertices[i], x[i], y[i])
    return inside


cdef class Cell:
    """Grid cell"""
    def __init__(self, double[:, :] vertices, box.Box bounding_box=None):
        self.vertices = vertices

        if bounding_box is None:
            bounding_box = box.Box.frompolygon(vertices)
        self.bounding_box = bounding_box

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__,
                               repr(np.asarray(self.vertices)))

    @classmethod
    def from_positions(cls, double[:, :, :] positions,
                       Py_ssize_t i,
                       Py_ssize_t j):
        """Construct cell from model grid index"""
        cdef:
            int ni = positions.shape[0]
            double[:, :] polygon

        polygon = np.array([positions[i % ni, j],
                            positions[(i + 1) % ni, j],
                            positions[(i + 1) % ni, j + 1],
                            positions[i % ni, j + 1]],
                           dtype="d")

        if crosses_dateline(polygon[:, 0]):
            return Dateline(polygon)
        return cls(polygon)

    cpdef bint contains(self, double x, double y):
        """Detects coordinates that lie inside the grid cell"""
        return self.bounding_box.inside(x, y) and self.same_side_test(x, y)

    cdef bint same_side_test(self, double x, double y):
        """Detects coordinates that lie inside a polygon"""
        return same_side_test(self.vertices, x, y)

    property center:
        def __get__(self):
            if self._center is None:
                self._center = self._compute_center()
            return self._center

    property polygon:
        def __get__(self):
            return self.vertices

    cdef tuple _compute_center(self):
        """Estimate center created by great circle diagonal intersection"""
        cdef:
            double[:, :] line_1 = np.array([self.vertices[0],
                                            self.vertices[2]], dtype="d")
            double[:, :] line_2 = np.array([self.vertices[1],
                                            self.vertices[3]], dtype="d")
        return spherical.intercept(line_1, line_2)


cpdef bint crosses_dateline(double[:] longitudes):
    """Detect Polygon that crosses 180th meridian"""
    return abs(min(longitudes) - max(longitudes)) > 180


@cython.cdivision(True)
@cython.boundscheck(False)
cpdef bint same_side_test(double[:, :] vertices, double x, double y):
    """Points lie inside convex polygons if they lie on the same side as every
    face.

    :returns: logical indicating point inside polygon
    """
    cdef double x1, y1, x2, y2, current, previous
    cdef int ipoint, npoints
    cdef double xmin, xmax, ymin, ymax

    npoints = vertices.shape[0]

    previous = 0
    for ipoint in range(npoints):
        x1 = vertices[ipoint, 0]
        y1 = vertices[ipoint, 1]
        x2 = vertices[(ipoint + 1) % npoints, 0]
        y2 = vertices[(ipoint + 1) % npoints, 1]

        current = detect_side(x1, y1,
                              x2, y2,
                              x, y)
        if (current == 0):
            # Point on line defined by boundary is included
            # if it lies between the corner points
            if (x1 < x2):
                xmin, xmax = x1, x2
            else:
                xmin, xmax = x2, x1
            if (y1 < y2):
                ymin, ymax = y1, y2
            else:
                ymin, ymax = y2, y1
            return ((x >= xmin) and
                    (x <= xmax) and
                    (y >= ymin) and
                    (y <= ymax))

        if (previous * current) < 0:
            return False

        previous = current

    return True


cdef double detect_side(double x1,
                        double y1,
                        double x2,
                        double y2,
                        double x,
                        double y):
    """Check the which side a point is on relative to a line segment"""
    return (y - y1) * (x2 - x1) - (x - x1) * (y2 - y1)


class Dateline(Cell):
    """Grid cell containing 180th meridian

    The longitude axis inside these cells has a discontinuity at the 180th
    meridian.
    """
    def __init__(self, polygon):
        self.west, self.east = self.split_cell(polygon)
        bounding_box = box.Dateline(polygon)
        super(Dateline, self).__init__(polygon, bounding_box)

    def contains(self, longitude, latitude):
        """Determines whether point in interior of cell"""
        if self.bounding_box.inside(longitude, latitude):
            if longitude > 0:
                return same_side_test(self.west, longitude, latitude)
            else:
                return same_side_test(self.east, longitude, latitude)
        else:
            return False

    @staticmethod
    def split_cell(polygon):
        """Split cell into West/East cells either side of 180th meridian"""
        return split_cell(polygon)


def split_cell(polygon):
    """Split a cell containing the 180th meridian into two sub-cells

    :param polygon: polygon definition of grid cell
    :returns: west cell, east cell
    """
    # Insert extra nodes
    nodes = []
    for (x1, y1), (x2, y2) in segments(polygon):
        nodes.append((x1, y1))
        if np.abs(x2 - x1) > 180:
            if x1 > 0:
                # Translate x2 from near -180 to +180
                x2 += 360.
            else:
                # Translate x1 from near -180 to +180
                x1 += 360.
            y = y_intercept(x1, y1, x2, y2, 180)
            nodes += [(180, y), (-180, y)]

    # Split nodes into east/west sub-cycles
    west, east = [], []
    for node in nodes:
        longitude, _ = node
        if longitude > 0:
            west.append(node)
        else:
            east.append(node)
    return np.asarray(west, dtype="d"), np.asarray(east, dtype="d")


cdef double y_intercept(double x1, double y1, double x2, double y2, double x):
    """y axis intercept formula

    :param x1: x coordinate of first point
    :param y1: y coordinate of first point
    :param x2: x coordinate of second point
    :param y2: y coordinate of second point
    :returns: y value that satisfies line at x
    """
    cdef double slope

    if x1 == x2:
        return y1
    slope = (y2 - y1) / (x2 - x1)
    return y1 + slope * (x - x1)


cpdef segments(polygon):
    """iterate over line segments defining a polygon"""
    cdef int ipoint
    cdef int npoints = len(polygon)
    cdef np.ndarray[double, ndim=3] result = np.empty((npoints, 2, 2))

    polygon = np.asarray(polygon, dtype="d")
    for ipoint in range(npoints):
        result[ipoint] = polygon[ipoint], polygon[(ipoint + 1) % npoints]
    return result


cdef class Collection:
    """Convenient cache of ocean model grid cells"""
    def __init__(self, positions):
        self.positions = np.asarray(positions, dtype="d")
        self.cache = {}

    @classmethod
    def from_lonlats(cls, grid_longitudes, grid_latitudes):
        """Construct grid from longitude/latitude arrays"""
        positions = np.dstack((grid_longitudes,
                               grid_latitudes))
        return cls(positions)

    def __getitem__(self, key):
        i, j = key
        return self.find(i, j)

    cpdef Cell find(self, int i, int j):
        """Locate cell from collection"""
        try:
            return self.cache[(i, j)]
        except KeyError:
            grid_cell = Cell.from_positions(self.positions, i, j)
            self.cache[(i, j)] = grid_cell
            return grid_cell

    @property
    def shape(self):
        """Shape of grid"""
        ni = self.positions.shape[0]
        nj = self.positions.shape[1]
        return ni, nj


@cython.boundscheck(False)
cpdef bint same(Cell cell_1, Cell cell_2):
    """Determine if two cell objects refer to the same grid cell in space

    :returns: True if both cells represent same cell
    """
    cdef:
        int i, j, n = cell_1.vertices.shape[0]
        double x, y, tolerance = 1e-6
    for i in range(n):
        for j in range(2):
            x = cell_1.vertices[i, j]
            y = cell_2.vertices[i, j]
            if abs(x - y) > tolerance:
                return False
    return True
