"""
Horizontal interpolators
========================
"""
# pylint: disable=invalid-name
import numpy as np


class UnitSquare(object):
    """bilinear interpolator

    Handles horizontal interpolation of surface and profile variables.

    .. note:: depth dimension should be last dimension e.g. (x, y, z)
    """
    def __init__(self, ilon, jlat, dilon, djlat):
        self.ilon = ilon
        self.jlat = jlat
        self.dilon = dilon
        self.djlat = djlat

    def __call__(self, field):
        """bilinear interpolation"""
        values, weights = self.values(field), self.weights
        if values.ndim == 3:
            weights = weights[..., None]
        result = np.ma.sum(values * weights, axis=0)
        return np.ma.masked_array(result, mask=self.masked(field))

    def values(self, field):
        """bilinear interpolation scheme model values

        :param field: 2D/3D model field
        :returns: 3D/4D array with first dimension length 4
        """
        return np.ma.vstack([field[None, self.ilon, self.jlat],
                             field[None, self.ilon, self.jlat + 1],
                             field[None, self.ilon + 1, self.jlat],
                             field[None, self.ilon + 1, self.jlat + 1]])

    @property
    def weights(self):
        """bilinear unit square scheme interpolation weights

        .. note:: dilon, dilat are unit square fractions
        """
        return np.ma.vstack([(1 - self.dilon) * (1 - self.djlat),
                             (1 - self.dilon) * self.djlat,
                             self.dilon * (1 - self.djlat),
                             self.dilon * self.djlat])

    def masked(self, field):
        """screens incomplete interpolations

        masks computation dimension for any computations with
        fewer than 4 valid corners
        """
        return self.values(field).mask.any(axis=0)
