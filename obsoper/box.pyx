"""
Boxes that surround grid cells.
"""
cimport cython


cdef class Box:
    """Bounding box surrounding collection of vertices"""
    def __init__(self,
                 double xmin,
                 double xmax,
                 double ymin,
                 double ymax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    @classmethod
    def frompolygon(cls, double[:, :] vertices):
        """Construct box from vertices

        :param vertices: array of pairs of points
        """
        cdef:
            double xmin, xmax, ymin, ymax
        xmin = min(vertices[:, 0])
        xmax = max(vertices[:, 0])
        ymin = min(vertices[:, 1])
        ymax = max(vertices[:, 1])
        return cls(xmin, xmax, ymin, ymax)

    cpdef bint inside(self, double x, double y):
        """Check point lies inside box

        :param x: x-coordinate to test
        :param y: y-coordinate to test
        :returns: logical indicating coordinates contained in box
        """
        if x > self.xmax:
            return False
        if x < self.xmin:
            return False
        if y > self.ymax:
            return False
        if y < self.ymin:
            return False
        return True


cdef class Dateline(Box):
    """Bounding box for grid cells that contain 180th Meridian"""
    def __init__(self, double[:, :] polygon):
        cdef:
            double[:] longitudes = polygon[:, 0]
            double[:] latitudes = polygon[:, 1]

        self.north = max(latitudes)
        self.south = min(latitudes)
        self.east = maximum_negative(longitudes)
        self.west = minimum_positive(longitudes)

    cpdef bint inside(self, double longitude, double latitude):
        """Check point inside bounding box containing 180th meridian

        Cells containing the 180th Meridian wrap around the planet in the
        opposite sense to regular grid cells.

        :param longitude: longitude of query point
        :param latitude: latitude of query point
        :returns: logical indicating point inside box
        """
        return self.inside_x(longitude) and self.inside_y(latitude)

    cpdef bint inside_x(self, double longitude):
        """Check point inside east/west extents"""
        if longitude > 0:
            return longitude > self.west
        else:
            return longitude < self.east

    cpdef bint inside_y(self, double latitude):
        """Check point inside north/south extents"""
        if latitude > self.north:
            return False
        elif latitude < self.south:
            return False
        else:
            return True


@cython.boundscheck(False)
cdef double minimum_positive(double[:] values):
    """Select minimum positive value"""
    cdef:
        int i, n
        double value
        list positives = []

    n = values.shape[0]
    for i in range(n):
        value = values[i]
        if value > 0:
            positives.append(value)

    return min(positives)


@cython.boundscheck(False)
cdef double maximum_negative(double[:] values):
    """Select maximum negative value"""
    cdef:
        int i, n
        double value
        list negatives = []

    n = values.shape[0]
    for i in range(n):
        value = values[i]
        if value < 0:
            negatives.append(value)

    return max(negatives)
