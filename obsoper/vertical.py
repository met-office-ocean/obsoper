"""
Vertical interpolators
======================

Vertical interpolation is achieved through use of :py:mod:`scipy.interpolate`
module. The module is a wrapper of the Fortran library FITPACK. However, it
does not preserve or indeed handle masked arrays elegantly.

"""
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from .window import Window, EmptyWindow


class Vertical2DInterpolator(object):
    """Vertical section interpolator

    Interpolates model grid section to observed section depths.
    Both sections should be 2D arrays whose first dimension represents
    the number of water columns. The second dimension of each section
    should be the depth dimension.

    :param depths: 2D array (N, Z)
    :param field: 2D array (N, Z)

    :returns: interpolator function
    """
    def __init__(self, depths, field):
        self.depths = depths
        self.field = field

    def __call__(self, observed_depths):
        result = []
        for depths_1d, field_1d, observed_1d in zip(self.depths, self.field,
                                                    observed_depths):
            interpolator = Vertical1DInterpolator(depths_1d, field_1d)
            result.append(interpolator(observed_1d))
        return np.ma.MaskedArray(result)


class Vertical1DInterpolator(object):
    """Vertical water column interpolator

    Interpolates a single model water column to an observed profile.
    Masked data points are excluded from the interpolation but included
    in the result.

    :param depths: 1D array
    :param field: 1D array
    """
    def __init__(self, depths, field):
        depths, field = self.match(depths, field)
        self.interpolator = self.select_interpolator(depths, field)
        self.window = self.select_window(depths)

    @staticmethod
    def match(depths, field):
        """combine masks from both depths and field"""
        depths, field = np.ma.asarray(depths), np.ma.asarray(field)
        common = depths.mask | field.mask
        return (np.ma.masked_array(depths, common),
                np.ma.masked_array(field, common))

    @staticmethod
    def select_window(depths):
        """select appropriate window function

        :returns: either :class:`obsoper.window.EmptyWindow`
                  or :class:`obsoper.window.Window`
        """
        if depths.mask.all():
            return EmptyWindow()
        return Window(depths.min(), depths.max())

    def select_interpolator(self, depths, field):
        """Select appropriate interpolation function

        ========= ===========================
        Knots     Interpolator
        ========= ===========================
        0 or 1    Masked data always returned
        2 or 3    Linear spline interpolator
        4 or more Cubic spline interpolator
        ========= ===========================

        Spline interpolators are created using non-masked data points

        :returns: interpolator chosen from above table
        """
        if self.knots(depths) < 2:
            # Masked data interpolator
            return lambda x: np.ma.masked_all(len(x))
        if self.knots(depths) < 4:
            # Linear spline interpolator
            return InterpolatedUnivariateSpline(depths.compressed(),
                                                field.compressed(), k=1)
        # Cubic spline interpolator
        return InterpolatedUnivariateSpline(depths.compressed(),
                                            field.compressed())

    @staticmethod
    def knots(depths):
        """count number of knots"""
        return np.ma.count(depths)

    def __call__(self, observed_depths):
        """interpolate field to observed depths"""
        observed_depths = self.screen_depths(observed_depths)
        if observed_depths.mask.any():
            return self.masked_interpolate(observed_depths)
        else:
            return self.interpolator(observed_depths)

    def screen_depths(self, observed_depths):
        """mask depths outside model water column"""
        observed_depths = np.ma.asarray(observed_depths)
        mask = self.window.outside(observed_depths)
        return np.ma.masked_array(observed_depths, observed_depths.mask | mask)

    def masked_interpolate(self, observed_depths):
        """performs interpolation on valid data while preserving mask"""
        result = np.ma.masked_all(observed_depths.shape)
        points, values = (~observed_depths.mask,
                          observed_depths.compressed())
        result[points] = self.interpolator(values)
        return result
