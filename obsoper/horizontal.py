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
    :param has_halo: logical indicating existence of halo
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
                 has_halo=False,
                 search="cartesian",
                 boundary="polygon"):
        # Cast positions as doubles
        observed_longitudes = np.asarray(observed_longitudes, dtype="d")
        observed_latitudes = np.asarray(observed_latitudes, dtype="d")
        grid_longitudes = np.asarray(grid_longitudes, dtype="d")
        grid_latitudes = np.asarray(grid_latitudes, dtype="d")
        self.n_observations = len(observed_longitudes)

        # Screen grid cells inside halo
        self.has_halo = has_halo
        if self.has_halo:
            grid_longitudes = orca.remove_halo(grid_longitudes)
            grid_latitudes = orca.remove_halo(grid_latitudes)

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

    def interpolate(self, field):
        """Perform vectorised interpolation to observed positions

        .. note:: `has_halo` flag specified during construction trims field
                  appropriately

        :param field: array shaped (I, J, [K]) same shape as model domain
        :returns: array shaped (N, [K]) of interpolated field values
                  where N represents the number of observed positions
        """
        field = np.ma.asarray(field)

        if self.has_halo:
            field = orca.remove_halo(field)

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


class Tripolar(Horizontal):
    """Tri-polar interpolator

    Handles grids that are composed of quadrilaterals but may
    not be easily searchable.

    :param grid_longitudes: 2D array of longitudes shapes (x, y)
    :param grid_latitudes: 2D array of latitudes shapes (x, y)
    :param observed_longitudes: 1D array of longitudes
    :param observed_latitudes: 1D array of latitudes
    :param has_halo: flag indicating whether diagnostics have redundant halo
                     columns and row.
    """
    def __init__(self,
                 grid_longitudes,
                 grid_latitudes,
                 observed_longitudes,
                 observed_latitudes,
                 has_halo=False):
        super(Tripolar, self).__init__(grid_longitudes,
                                       grid_latitudes,
                                       observed_longitudes,
                                       observed_latitudes,
                                       has_halo=has_halo,
                                       search="tripolar",
                                       boundary="band")


def mask_corners(corner_values):
    """Masks cells which have 1 or more masked corner values

    :param corner_values: array shaped ([[N], M], 4)
    :returns: array with corner dimension partially masked cells fully masked
    """
    if hasattr(corner_values, "mask") and (corner_values.mask.any()):
        if corner_values.ndim == 1:
            return np.ma.masked_all_like(corner_values)

        # Multidimensional arrays
        invalid = corner_values.mask.any(axis=-1)
        corner_values[invalid] = np.ma.masked
    return corner_values


# Vectorised algorithm
def select_corners(values, i, j):
    """Select array of grid cell corner positions/columns

    Selects positions/water columns representing four corners by
    choosing (i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)

    .. note:: periodic in i coordinate

    :param values: 2D/3D array dimension (X, Y, [Z])
    :param i: array of integers shaped ([N],) representing lower left corner
    :param j: array of integers shaped ([N],) representing lower left corner
    :returns: array shaped (4, [Z, [N]]) representing values/columns
              for each cell whose lower left corner is (i, j)
    """
    #pylint: disable=invalid-name
    i, j = np.asarray(i, dtype="i"), np.asarray(j, dtype="i")

    if isinstance(values, np.ma.MaskedArray):
        array_constructor = np.ma.MaskedArray
    else:
        array_constructor = np.array
    ni = values.shape[0]
    vectors = [values[i % ni, j],
               values[(i+1) % ni, j],
               values[(i+1) % ni, j+1],
               values[i % ni, j+1]]
    if np.isscalar(i):
        return array_constructor(vectors, dtype="d")
    else:
        return array_constructor([vector.T for vector in vectors], dtype="d")


def select_field(field, i, j):
    """Select array of grid cell corner values"""
    return select_corners(field, i, j)


def correct_corners(vertices, longitudes):
    """Alter grid corners in-place to enable vectorised interpolation"""
    vertices = np.asarray(vertices, dtype="d")
    grid_longitudes = vertices[:, 0]

    # Apply East/West dateline corrections
    if vertices.ndim == 2:
        if is_dateline(vertices):
            if is_east(longitudes):
                correct_east(grid_longitudes)
            else:
                correct_west(grid_longitudes)

    elif vertices.ndim == 3:
        datelines = is_dateline(vertices)

        east = is_east(longitudes) & datelines
        west = is_west(longitudes) & datelines

        # Note: boolean indexing returns copy of array
        grid_longitudes[:, east] = correct_east(grid_longitudes[:, east])
        grid_longitudes[:, west] = correct_west(grid_longitudes[:, west])

    return vertices


def correct_east(longitudes):
    """Map cell lying on 180th meridian to eastern coordinates"""
    longitudes[longitudes > 0] -= 360.
    return longitudes


def correct_west(longitudes):
    """Map cell lying on 180th meridian to western coordinates"""
    longitudes[longitudes < 0] += 360.
    return longitudes


def is_dateline(vertices):
    """Detect dateline cells from corners"""
    vertices = np.asarray(vertices, dtype="d")
    longitudes = vertices[:, 0]
    return np.abs(longitudes.min(axis=0) - longitudes.max(axis=0)) > 180


def is_east(longitudes):
    """Detect positions on eastern side of greenwich meridian"""
    return np.asarray(longitudes, dtype="d") <= 0.


def is_west(longitudes):
    """Detect positions on western side of greenwich meridian"""
    return ~is_east(longitudes)


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
