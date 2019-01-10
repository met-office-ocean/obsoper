"""
Horizontal interpolators
========================

For sufficiently complex interpolations it is sometimes beneficial to declare
the observation locations up front, so that grid cells positions can be
calculated. Then repeated calls to interpolate can map different fields to
predefined locations.

For simple light weight interpolations specifying desired locations at
interpolation time is more convenient. In this use case, several sets of
observations can be interpolated to a single model layout.

"""
# pylint: disable=invalid-name
import numpy as np
from . import (grid,
               bilinear,
               domain,
               orca)
from obsoper.corners import (
    select_corners,
    select_field,
    correct_corners,
    mask_corners)


class Horizontal(object):
    """Interpolate ocean grid to observation locations

    Perform bilinear interpolation on ocean grids. The following combinations
    of boundary and search algorithm are recommended.

    ======== ========= =====================
    Boundary Search    Description
    ======== ========= =====================
    band     tripolar  ORCA family of models
    polygon  cartesian Regional models
    regular  cartesian Regular lon/lat models
    ======== ========= =====================

    .. note:: When analysing tripolar models it is important to specify
              whether or not the grid includes a halo via `has_halo`

    :param grid_longitudes: 2D array of longitudes shapes (x, y)
    :param grid_latitudes: 2D array of latitudes shapes (x, y)
    :param observed_longitudes: 1D array of longitudes
    :param observed_latitudes: 1D array of latitudes
    :param search: algorithm either 'tripolar' or 'cartesian'
                   see :func:`obsoper.grid.lower_left`
    :param boundary: domain shape either 'band', 'polygon' or 'regular',
                     see :func:`obsoper.domain.inside`
    """
    def __init__(self,
                 grid_longitudes,
                 grid_latitudes,
                 observed_longitudes,
                 observed_latitudes,
                 search="cartesian",
                 boundary="polygon"):
        # Cast positions as doubles
        observed_longitudes = np.asarray(observed_longitudes, dtype="d")
        observed_latitudes = np.asarray(observed_latitudes, dtype="d")
        grid_longitudes = np.asarray(grid_longitudes, dtype="d")
        grid_latitudes = np.asarray(grid_latitudes, dtype="d")
        self.n_observations = len(observed_longitudes)

        # Detect observations inside domain
        self.included = domain.inside(grid_longitudes,
                                      grid_latitudes,
                                      observed_longitudes,
                                      observed_latitudes,
                                      kind=boundary)

        if self.included.any():
            included_longitudes = observed_longitudes[self.included]
            included_latitudes = observed_latitudes[self.included]

            # Detect grid cells containing observations
            self.i, self.j = grid.lower_left(grid_longitudes,
                                             grid_latitudes,
                                             included_longitudes,
                                             included_latitudes,
                                             search=search)

            # Correct grid cell corner positions to account for dateline
            _grid = np.dstack((grid_longitudes,
                               grid_latitudes)).astype("d")
            corners = select_corners(_grid, self.i, self.j)
            corners = correct_corners(corners, included_longitudes)

            # Estimate interpolation weights from coordinates
            # Corner position shape (4, 2, [N]) --> (N, 4, 2)
            if corners.ndim == 3:
                corners = np.transpose(corners, (2, 0, 1))
            self.weights = bilinear.interpolation_weights(corners,
                                                          included_longitudes,
                                                          included_latitudes)
    def __call__(self, field):
        return self.interpolate(field)

    def interpolate(self, field):
        """Perform vectorised interpolation to observed positions

        :param field: array shaped (I, J, [K]) same shape as model domain
        :returns: array shaped (N, [K]) of interpolated field values
                  where N represents the number of observed positions
        """
        field = np.ma.asarray(field)

        # Interpolate field to observed positions
        if field.ndim == 3:
            shape = (self.n_observations, field.shape[2])
        else:
            shape = (self.n_observations,)
        result = np.ma.masked_all(shape, dtype="d")

        if self.included.any():
            corner_values = select_field(field, self.i, self.j)

            # Corner values shape (4, [Z, [N]]) --> (Z, N, 4)
            if corner_values.ndim == 2:
                corner_values = corner_values.T
            elif corner_values.ndim == 3:
                corner_values = np.transpose(corner_values, (1, 2, 0))

            corner_values = mask_corners(corner_values)

            result[self.included] = bilinear.interpolate(corner_values,
                                                         self.weights).T
        return result


class Regional(Horizontal):
    """Regional ocean model interpolator

    Computes horizontal interpolations for regional ocean models with
    irregular boundaries and/or rotated grid coordinates.

    Similar to :class:`obsoper.horizontal.Regular` in that the boundaries
    are of finite extent but is more general. A point in polygon test is
    used to determine if a point is inside the domain. And a KD tree search
    is used to locate grid points surrounding an observation.

    :param grid_longitudes: 2D array of longitudes shapes (x, y)
    :param grid_latitudes: 2D array of latitudes shapes (x, y)
    :param observed_longitudes: 1D array of longitudes
    :param observed_latitudes: 1D array of latitudes
    """
    def __init__(self,
                 grid_longitudes,
                 grid_latitudes,
                 observed_longitudes,
                 observed_latitudes):
        super(Regional, self).__init__(grid_longitudes,
                                       grid_latitudes,
                                       observed_longitudes,
                                       observed_latitudes,
                                       search="cartesian",
                                       boundary="polygon")


class Regular(object):
    """Regular grid horizontal interpolator

    :param grid_longitudes: Array shaped (X[, Y])
    :param grid_latitudes: Array shaped ([X,] Y)
    :param observed_longitudes: 1D array of longitudes
    :param observed_latitudes: 1D array of latitudes
    """
    def __init__(self,
                 grid_longitudes,
                 grid_latitudes,
                 observed_longitudes,
                 observed_latitudes):
        self.observed_longitudes = np.asarray(observed_longitudes)
        self.observed_latitudes = np.asarray(observed_latitudes)

        grid_longitudes = np.asarray(grid_longitudes, dtype="d")
        if grid_longitudes.ndim == 2:
            grid_longitudes = grid_longitudes[:, 0]

        grid_latitudes = np.asarray(grid_latitudes, dtype="d")
        if grid_latitudes.ndim == 2:
            grid_latitudes = grid_latitudes[0, :]

        self.grid = grid.Regular2DGrid(grid_longitudes,
                                       grid_latitudes)

    def interpolate(self, field):
        """interpolates model field to observed locations

        :param field: 2D/3D model array
        :param observed_longitudes: 1D array
        :param observed_latitudes: 1D array
        :returns: either 1D vector or 2D section of field in
                  observation space
        """
        field = np.ma.asarray(field)

        # Detect observations inside grid
        mask = self.grid.inside(self.observed_longitudes,
                                self.observed_latitudes)

        # np.ma.where is used to prevent masked elements being used as indices
        points = np.ma.where(mask)

        # Interpolate to observations inside grid
        search_result = self.grid.search(self.observed_longitudes[points],
                                         self.observed_latitudes[points])
        interpolator = UnitSquare(*search_result)
        interpolated = interpolator(field)

        # Assemble result
        result = np.ma.masked_all(self.section_shape(self.observed_longitudes,
                                                     field))
        result[points] = interpolated
        return result

    @staticmethod
    def section_shape(positions, field):
        """defines shape of bilinear interpolated section/surface"""
        if field.ndim == 2:
            return (len(positions),)
        return (len(positions), field.shape[2])


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
