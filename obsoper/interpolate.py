"""
Interpolators
=============
"""
import numpy as np
from . import (grid,
               bilinear,
               orca)


class Tripolar(object):
    """Tri-polar interpolator

    Handles grids that are composed of quadrilaterals but may
    not be easily searchable.

    :param grid_longitudes: 2D array of longitudes shapes (x, y)
    :param grid_latitudes: 2D array of latitudes shapes (x, y)
    :param observed_longitudes: 1D array of longitudes
    :param observed_latitudes: 1D array of latitudes
    :param has_halo: flag indicating whether diagnostics have redundant halo
                     columns and row.
    :param dimension_order: describe grid dimensions "xy" or "yx", interpolator
                            transposes arrays to "xy"
    """
    def __init__(self,
                 grid_longitudes,
                 grid_latitudes,
                 observed_longitudes,
                 observed_latitudes,
                 has_halo=False,
                 dimension_order="xy"):
        # Cast positions as doubles
        self.observed_longitudes = np.asarray(observed_longitudes, dtype="d")
        self.observed_latitudes = np.asarray(observed_latitudes, dtype="d")
        self.grid_longitudes = np.asarray(grid_longitudes, dtype="d")
        self.grid_latitudes = np.asarray(grid_latitudes, dtype="d")

        self.has_halo = has_halo

        # Transpose coordinates
        if dimension_order.lower() == "yx":
            self.grid_longitudes = self.grid_longitudes.T
            self.grid_latitudes = self.grid_latitudes.T

        # Screen grid cells inside halo
        if self.has_halo:
            self.grid_longitudes = orca.remove_halo(self.grid_longitudes)
            self.grid_latitudes = orca.remove_halo(self.grid_latitudes)

        self._grid = np.dstack((self.grid_longitudes,
                                self.grid_latitudes)).astype("d")

        self.n_observations = len(self.observed_longitudes)

        # Filter observations that are enclosed by grid
        self.minimum_latitude = np.ma.min(self.grid_latitudes)
        self.maximum_latitude = np.ma.max(self.grid_latitudes)
        self.included = self.inside_grid(self.observed_latitudes)

        if self.included.any():
            included_longitudes = self.observed_longitudes[self.included]
            included_latitudes = self.observed_latitudes[self.included]

            # Locate relevant grid cells
            search = grid.Search(self.grid_longitudes,
                                 self.grid_latitudes)
            self.i, self.j = search.lower_left(included_longitudes,
                                               included_latitudes)
            self.i = np.asarray(self.i, dtype="i")
            self.j = np.asarray(self.j, dtype="i")

            # Correct grid cell corner positions to account for dateline
            corners = select_corners(self._grid, self.i, self.j)
            corners = correct_corners(corners, included_longitudes)

            # Train interpolator on coordinates
            self.interpolator = bilinear.BilinearTransform(corners,
                                                           included_longitudes,
                                                           included_latitudes)

    def inside_grid(self, latitudes):
        """Determine observations inside grid"""
        return ((latitudes >= self.minimum_latitude) &
                (latitudes <= self.maximum_latitude))

    def interpolate(self, field):
        """Perform vectorised interpolation

        .. note:: `has_halo` flag specified during construction trims field
                  appropriately

        :param field: 2D array same shape as grid_longitudes/grid_latitudes
        :returns: 1D array of interpolated field values
        """
        if self.has_halo:
            field = orca.remove_halo(field)

        # Interpolate field to observed positions
        result = np.ma.masked_all(self.n_observations, dtype="d")
        if self.included.any():
            corner_values = self.select_field(field)
            if hasattr(corner_values, "mask") and (corner_values.mask.any()):
                invalid = corner_values.mask.any(axis=0)
                corner_values[:, invalid] = np.ma.masked
            result[self.included] = self.interpolator(corner_values)
        return result

    def select_field(self, field):
        """Select grid cell corner values corresponding to observations"""
        return select_field(field, self.i, self.j)


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


def _select_corners(values, i, j):
    """Select array of grid cell corner positions/columns

    Selects positions/water columns representing four corners by
    choosing (i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)

    .. note:: periodic in i coordinate

    :param values: 2D/3D array dimension (X, Y, [Z])
    :param i: array of integers shaped ([N],) representing lower left corner
    :param j: array of integers shaped ([N],) representing lower left corner
    :returns: array shaped (4, [N, [Z]]) representing values/columns
              for each cell whose lower left corner is (i, j)
    """
    #pylint: disable=invalid-name
    i, j = np.asarray(i, dtype="i"), np.asarray(j, dtype="i")
    if isinstance(values, np.ma.MaskedArray):
        array_constructor = np.ma.MaskedArray
    else:
        array_constructor = np.array
    i_corners = np.array([i, i+1, i+1, i]) % len(values)
    j_corners = np.array([j, j, j+1, j+1])
    return array_constructor(values[i_corners, j_corners], dtype="d")


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
