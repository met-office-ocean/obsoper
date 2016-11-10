"""
Spherical linear interpolation (Slerp)
--------------------------------------

Interpolation on the surface of a sphere is analogous to interpolation
on a plane. The only differences being straight lines are replaced by great
circles and distances are proportional to angles.
"""
import numpy as np


def bilinear(corners, values, point):
    """Two dimensional linear interpolation on sphere

    Calculates spherical analogue to Euclidean bilinear interpolation

    :param corners: 4 cartesian coordinates representing surrounding corners
    :param values: values associated with each corner
    :param point: cartesian coordinate to be estimated
    :returns: value(s) at observed points
    """
    corners = np.asarray(corners, dtype="d")
    for icorner in range(4):
        if np.allclose(point, corners[icorner]):
            return values[icorner]
    # Note: interpolate general positions
    method = "topbottom"
    if method == "topbottom":
        lower_point = great_circle_nearest_point(corners[0],
                                                 corners[1],
                                                 point)
        lower_value = linear(corners[0],
                             corners[1],
                             values[0],
                             values[1],
                             lower_point)

        upper_point = great_circle_nearest_point(corners[2],
                                                 corners[3],
                                                 point)
        upper_value = linear(corners[2],
                             corners[3],
                             values[2],
                             values[3],
                             upper_point)

        return linear(lower_point,
                      upper_point,
                      lower_value,
                      upper_value,
                      point)
    else:
        left_point = great_circle_nearest_point(corners[0],
                                                corners[3],
                                                point)
        left_value = linear(corners[0],
                            corners[3],
                            values[0],
                            values[3],
                            left_point)

        right_point = great_circle_nearest_point(corners[1],
                                                 corners[2],
                                                 point)
        right_value = linear(corners[1],
                             corners[2],
                             values[1],
                             values[2],
                             right_point)

        return linear(left_point,
                      right_point,
                      left_value,
                      right_value,
                      point)


def linear(vertex_a,
           vertex_b,
           value_a,
           value_b,
           vertex_c):
    """One dimensional linear interpolation on sphere

    Calculates linear interpolated value at point C along great circle
    arc joining points A and B.

    Interpolation weights are given by the angle subtended by point A and C
    as a fraction of the total angle subtended by point A and B.

    :param vertex_a: (x, y, z) coordinate of point A on sphere
    :param vertex_b: (x, y, z) coordinate of point B on sphere
    :param value_a: value at vertex A
    :param value_b: value at vertex B
    :param vertex_c: (x, y, z) coordinate to evaluate interpolated value
    :returns: value at vertex C
    """
    if np.allclose(vertex_c, vertex_a):
        return value_a
    if np.allclose(vertex_c, vertex_b):
        return value_b
    omega = np.arccos(np.dot(vertex_a, vertex_b))
    theta = np.arccos(np.dot(vertex_a, vertex_c))
    weight = theta / omega
    return ((1 - weight) * value_a) +  (weight * value_b)


# pylint: disable=invalid-name
def great_circle_nearest_point(A, B, C):
    """Find point on great circle that is nearest to point not on great circle

    Solution involves the following simple geometric argument:
       - A cross B defines vector G perpendicular to plane AB (great circle AB)
       - G cross C defines vector F co-planar to plane AB but perpendicular to
         plane CG
       - F cross G defines vector T representing intersection of plane AB with
         plane CG
       - T when scaled to unit length defined closest point to great circle AB

    .. note:: Great circle definition is ambiguous for antipodal points
    .. note:: A point that is perpendicular to the plane defined by the great
              circle is equidistant to all points on the great circle

    :param A: (x, y, z) coordinate of point A defining great circle
    :param B: (x, y, z) coordinate of point B defining great circle
    :param C: (x, y, z) coordinate of point C not on great circle
    :returns: Point T lying on great circle defined by AB and closest to C
    """
    G = np.cross(A, B)
    F = np.cross(G, C)
    T = np.cross(F, G)
    return T / np.linalg.norm(T)
