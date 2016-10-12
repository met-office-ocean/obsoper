"""Cython extension of spherical geometry functions"""
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport (sin,
                        cos,
                        atan2,
                        sqrt,
                        M_PI)


cdef struct Cartesian:
    double x
    double y
    double z


cdef struct Spherical:
    double longitude
    double latitude


cpdef bint intersect(double[:, :] line_1, double[:, :] line_2):
    """Check that two great circles intersect

    :returns: Logical indicating segments intersect
    """
    return _intersect(np.asarray(line_1, dtype="d"),
                      np.asarray(line_2, dtype="d"))


cpdef tuple intercept(double[:, :] line_1, double[:, :] line_2):
    """Find where two great circles intersect

    :returns: Location of intercept between two great circle arcs
    """
    coordinate = _intercept(np.asarray(line_1, dtype="d"),
                            np.asarray(line_2, dtype="d"))
    return coordinate.longitude, coordinate.latitude


def to_cartesian(point):
    """convert point given in latitude and longitude to x, y, z

    :param point: (longitude, latitude) in degrees
    :returns: x, y, z
    """
    longitude, latitude = point
    coordinate = _to_cartesian({"longitude": longitude,
                                "latitude": latitude})
    return coordinate.x, coordinate.y, coordinate.z


def to_spherical(point):
    """Convert cartesian point to longitude latitude in degrees

    :param point: (x, y, z) coordinates
    :returns: (longitude, latitude) in degrees
    """
    x, y, z = point
    coordinate = _to_spherical({"x": x, "y": y, "z": z})
    return coordinate.longitude, coordinate.latitude


@cython.boundscheck(False)
cdef Spherical _intercept(double[:, :] line_1,
                          double[:, :] line_2) except *:
    """Find where two great circles intersect

    :returns: Location of intercept between two great circle arcs
    """
    cdef:
        int n = 4
        double[4] scalars
        Spherical coordinate
        Cartesian point_1, point_2, point_3, point_4, vector_t

    point_1 = _to_cartesian({"longitude": line_1[0, 0],
                             "latitude": line_1[0, 1]})
    point_2 = _to_cartesian({"longitude": line_1[1, 0],
                             "latitude": line_1[1, 1]})
    point_3 = _to_cartesian({"longitude": line_2[0, 0],
                             "latitude": line_2[0, 1]})
    point_4 = _to_cartesian({"longitude": line_2[1, 0],
                             "latitude": line_2[1, 1]})

    scalars = c_scalars(point_1, point_2, point_3, point_4)
    vector_t = c_vector_t(point_1, point_2, point_3, point_4)

    # Check sign convention
    if all_positive(scalars, n):
        return _to_spherical(vector_t)
    if all_negative(scalars, n):
        return _to_spherical({"x": -vector_t.x,
                              "y": -vector_t.y,
                              "z": -vector_t.z})
    else:
        message = "Great circle arcs do not intersect:\n{}\n{}".format(line_1,
                                                                       line_2)
        raise Exception(message)


@cython.boundscheck(False)
cdef bint _intersect(double[:, :] line_1, double[:, :] line_2):
    """Check that two great circles intersect

    :returns: Logical indicating segments intersect
    """
    cdef:
        int n = 4
        double[4] scalars
        Cartesian point_1, point_2, point_3, point_4

    point_1 = _to_cartesian({"longitude": line_1[0, 0],
                             "latitude": line_1[0, 1]})
    point_2 = _to_cartesian({"longitude": line_1[1, 0],
                             "latitude": line_1[1, 1]})
    point_3 = _to_cartesian({"longitude": line_2[0, 0],
                             "latitude": line_2[0, 1]})
    point_4 = _to_cartesian({"longitude": line_2[1, 0],
                             "latitude": line_2[1, 1]})

    scalars = c_scalars(point_1, point_2, point_3, point_4)

    # Check sign convention
    if all_positive(scalars, n):
        return True
    elif all_negative(scalars, n):
        return True
    else:
        return False


cdef double* c_scalars(Cartesian point_1,
                       Cartesian point_2,
                       Cartesian point_3,
                       Cartesian point_4):
    """Define scalar values that determine great circle intersection"""
    cdef:
        double[4] scalars
        Cartesian vector_p = cross(point_1, point_2)
        Cartesian vector_q = cross(point_3, point_4)
        Cartesian vector_t = cross(vector_p, vector_q)

    # Calculate projections onto t vector
    scalars[0] = -1. * dot(cross(point_1, vector_p), vector_t)
    scalars[1] = +1. * dot(cross(point_2, vector_p), vector_t)
    scalars[2] = -1. * dot(cross(point_3, vector_q), vector_t)
    scalars[3] = +1. * dot(cross(point_4, vector_q), vector_t)
    return scalars


cdef bint all_positive(double* values, int n):
    """Boolean check that all values are positive"""
    cdef int i
    for i in range(n):
        if values[i] <= 0:
            return False
    return True


cdef bint all_negative(double* values, int n):
    """Boolean check that all values are negative"""
    cdef int i
    for i in xrange(n):
        if values[i] >= 0:
            return False
    return True


cdef Cartesian  c_vector_t(Cartesian point_1,
                           Cartesian point_2,
                           Cartesian point_3,
                           Cartesian point_4):
    """Calculate vector t needed to determine great circle overlap"""
    cdef Cartesian vector_p = cross(point_1, point_2)
    cdef Cartesian vector_q = cross(point_3, point_4)
    return normalise(cross(vector_p, vector_q))


cdef Spherical _to_spherical(Cartesian point):
    """Convert cartesian point to longitude latitude in degrees

    :param point: (x, y, z) coordinates
    :returns: (longitude, latitude) in degrees
    """
    cdef double radius, longitude, latitude
    radius = sqrt(point.x**2 + point.y**2)
    longitude = atan2(point.y, point.x)
    latitude = atan2(point.z, radius)
    return {"longitude": degrees(longitude),
            "latitude": degrees(latitude)}


cdef Cartesian _to_cartesian(Spherical point):
    """Cartesian coordinate representation of longitude/latitude"""
    cdef double x, y, z
    cdef double longitude, latitude

    longitude = radians(point.longitude)
    latitude = radians(point.latitude)

    x = cos(longitude) * cos(latitude)
    y = sin(longitude) * cos(latitude)
    z = sin(latitude)

    return {"x": x,
            "y": y,
            "z": z}


@cython.cdivision(True)
cdef Cartesian normalise(Cartesian vector):
    """Normalise a vector"""
    cdef:
        double length = norm(vector)
    return {"x": vector.x / length,
            "y": vector.y / length,
            "z": vector.z / length}


cdef double norm(Cartesian vector):
    """Vector length"""
    return sqrt(dot(vector, vector))


cdef Cartesian cross(Cartesian u, Cartesian v):
    """Cross product"""
    return {"x": (u.y * v.z) - (u.z * v.y),
            "y": (u.z * v.x) - (u.x * v.z),
            "z": (u.x * v.y) - (u.y * v.x)}


cdef double dot(Cartesian u, Cartesian v):
    """Cross product"""
    return (u.x * v.x) + (u.y * v.y) + (u.z * v.z)


cdef double radians(double angle):
    """Radian converter"""
    return angle * (M_PI / 180.)


@cython.cdivision(True)
cdef double degrees(double angle):
    """Degree converter"""
    return (angle * 180.) / M_PI
