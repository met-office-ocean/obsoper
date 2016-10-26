"""
Bilinear interpolators
========================
"""
# pylint: disable=invalid-name
import numpy as np


class BilinearTransform(object):
    """Bilinear interpolator

    Maps from general quadrilateral to unit square.

    Corners are chosen anti-clockwise starting from lower left.

    .. note:: Corner ordering is highly important.

    .. note:: If evaluating multiple cells and multiple positions, the
              number of cells must form a 1-2-1 correspondence with
              (x, y) positions.

    :param corners: array shaped (4, 2, [N]) representing 4 corners, 2
                    coordinates and optionally N cells
    :param x: array shapes ([N],) representing multiple positions
    :param y: array shapes ([N],) representing multiple positions
    """
    def __init__(self, corners, x, y):
        self.corners = np.asarray(corners, dtype="d")

        # Calculate alpha, beta coefficients
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = self.corners
        self.alpha = np.array([x1,
                               x2 - x1,
                               x4 - x1,
                               x1 - x2 + x3 - x4], dtype="d")
        self.beta = np.array([y1,
                              y2 - y1,
                              y4 - y1,
                              y1 - y2 + y3 - y4], dtype="d")

        # Calculate weights
        self.weights = self._weights(x, y)

    def _weights(self, x, y):
        di, dj = self.to_unit_square(x, y)
        return np.array([(1 - di) * (1 - dj),
                         di * (1 - dj),
                         di * dj,
                         (1 - di) * dj])

    def to_unit_square(self, x, y):
        """Transform from x, y coordinates to unit-square coordinates"""
        x, y = np.asarray(x, dtype="d"), np.asarray(y, dtype="d")

        a1, a2, a3, a4 = self.alpha
        b1, b2, b3, b4 = self.beta

        coefficient_1 = (a4 * b3) - (a3 * b4)
        coefficient_2 = ((a4 * b1) - (a1 * b4) +
                         (a2 * b3) - (a3 * b2) +
                         (x * b4) - (y * a4))
        coefficient_3 = ((a2 * b1) - (a1 * b2) +
                         (x * b2) - (y * a2))

        m = self.quadratic_root(coefficient_1,
                                coefficient_2,
                                coefficient_3)
        l = (x - a1 - a3 * m) / (a2 + a4 * m)
        return l, m

    @staticmethod
    def quadratic_root(a, b, c):
        """Simple quadratic positive root solution"""
        return quadratic_root(a, b, c)

    def __call__(self, values):
        # Note: Transpose could be eliminated if rest of library updated
        # to handle ([[Z], N], 4) shapes

        values = np.ma.asarray(values, dtype="d")
        weights = self.weights

        if values.ndim == 2:
            values = values.T

        if weights.ndim == 2:
            weights = weights.T

        return interpolate(values, weights).T


def interpolate(values, weights):
    """Apply interpolation weights to estimate field

    Applies interpolation weights across corner values. Surface and 3D
    data can be interpolated as one-to-many or many-to-one. Both arrays
    must have their last dimension length equal to 4.

    Takes advantage of numpy's broadcasting rules to multiply and
    sum values with weights.

    :param values: array shaped ([[Z], N], 4) where N is number
                   of observations and Z is model levels. Square brackets
                   indicate optional dimensions.
    :param weights: array shaped ([N], 4) where N is number of
                    observations
    :returns: scalar/array shaped ([[Z], N]) representing the field value
              at location interior to the cell
    """
    values, weights = np.ma.asarray(values), np.asarray(weights)
    return np.ma.sum(weights * values, axis=-1)


def interpolation_weights(corners, x, y):
    """Calculate interpolation weights

    General purpose 2D quadrilateral interpolation weights.

    :param corners: array shaped ([N,] 4, 2) where N is the number of
                    observations
    :param x: scalar/1D array of x-axis positions
    :param y: scalar/1D array of y-axis positions
    :returns: array shapes ([N,] 4) weights for each corner and
              optionally each observation
    """
    corners = np.asarray(corners)
    if corners.ndim == 3:
        corners = np.transpose(corners, (1, 2, 0))
    return BilinearTransform(corners, x, y).weights.T


def quadratic_root(a, b, c):
    """Simple quadratic positive root solution"""
    if (np.size(a) == 0) or (np.size(b) == 0) or (np.size(c) == 0):
        return np.array([], dtype="d")

    if np.all(a == 0):
        return _linear(b, c)

    if np.any(a == 0):
        # Handle linear and quadratic systems of equations
        linear = a == 0
        quadratic = ~linear

        result = np.empty_like(b)

        result[linear] = _linear(b[linear],
                                 c[linear])

        result[quadratic] = _quadratic(a[quadratic],
                                       b[quadratic],
                                       c[quadratic])
        return result

    return _quadratic(a, b, c)


def _quadratic(a, b, c):
    """Simple quadratic positive root solution"""
    return (-b + np.sqrt(b*b - 4*a*c)) / (2 * a)


def _linear(b, c):
    """Simple linear root solution"""
    return -c / b
