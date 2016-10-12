"""
Grids
======

Horizontal interpolation on a grid typically involves locating
the four grid cells surrounding an observed location. For a regular
grid a single (i, j) pair is enough to infer the existence of the
other 3 corners, namely (i+1, j), (i, j+1) and (i+1, j+1). This information
by itself is not sufficient to perform a bilinear interpolation. The fractional
displacement along the i-direction and j-direction is also required.


The following classes :class:`Regular2DGrid` and :class:`Regular1DGrid`
implement the required positional search functions.

"""
# pylint: disable=invalid-name
from collections import namedtuple
from scipy.spatial import cKDTree
import numpy as np
from .exceptions import NotInGrid
from . import walk


SearchResult = namedtuple("SearchResult", ("ilon", "ilat", "dilon", "dilat"))


class Search(object):
    """Finds lower-left hand grid point nearest observation"""
    def __init__(self, grid_longitudes, grid_latitudes):
        self.neighbour = NearestNeighbour(grid_longitudes, grid_latitudes)
        self.walk = walk.Walk.tripolar(grid_longitudes,
                                       grid_latitudes)

    def lower_left(self, longitudes, latitudes):
        """Detect lower left corner index of four corners surrounding
        observations.
        """
        i, j = self.nearest(longitudes, latitudes)
        return self.walk.query(longitudes, latitudes, i, j)

    def nearest(self, longitudes, latitudes):
        """Find nearest neighbour"""
        return self.neighbour.nearest(longitudes, latitudes)


class NearestNeighbour(object):
    """Nearest neighbour search

    Unstructured nearest neighbour search in 2D. Takes advantage of
    KD-Tree algorithm.
    """
    def __init__(self, longitudes, latitudes):
        self.shape = longitudes.shape
        self.tree = cKDTree(self.points(longitudes,
                                        latitudes))

    def nearest(self, longitudes, latitudes):
        """Find nearest neighbour

        :returns: i, j arrays of indices
        """
        _, indices = self.tree.query(self.points(longitudes,
                                                 latitudes))
        return np.unravel_index(indices,
                                self.shape)

    @staticmethod
    def points(longitudes, latitudes):
        """Convert arrays of longitudes and latitudes to single array"""
        return np.array([np.ravel(longitudes),
                         np.ravel(latitudes)]).T


class Regular2DGrid(object):
    """Regular 2D grid

    Two dimensional analog of :class:`Regular1DGrid`. Latitudes
    and longitudes defining the grid should be passed in as 1D arrays.

    :param longitudes: 1D array of longitudes
    :type longitudes: numpy.ndarray
    :param latitudes: 1D array of latitudes
    :type latitudes: numpy.ndarray
    """
    def __init__(self, longitudes, latitudes):
        self.longitudes = Regular1DGrid(longitudes)
        self.latitudes = Regular1DGrid(latitudes)

    def search(self, longitudes, latitudes):
        """Locate grid box south west corner relative locations

        Grid box corners are important for bilinear interpolation algorithms.
        The search result describes the grid index of the cells along with
        two fractions describing the position inside each cell.

        :returns: A tuple containing four arrays.
                  Longitude indices, latitude indices,
                  longitude fractions and latitude fractions.
        :rtype: :class:`SearchResult`
        """
        if self.outside(longitudes, latitudes).any():
            raise NotInGrid
        ilon, dilon = self.longitudes.search(longitudes)
        ilat, dilat = self.latitudes.search(latitudes)
        return SearchResult(ilon, ilat, dilon, dilat)

    def outside(self, longitudes, latitudes):
        """Detect points outside of grid

        A coordinate is outside of the 2D grid if its latitude
        or its longitude are outside the extent of the 2D grid.

        :returns: True if points lie outside 2D domain
        :rtype: Boolean array
        """
        return (self.longitudes.outside(longitudes) |
                self.latitudes.outside(latitudes))

    def inside(self, longitudes, latitudes):
        """Detect points inside grid

        Logical negation of outside grid criterion.

        :returns: True if points lie outside 2D domain
        :rtype: Boolean array
        """
        return ~self.outside(longitudes, latitudes)


class Regular1DGrid(object):
    """
    Defines a single dimension consisting of N vertices
    and N - 1 equally spaced cells.

    :param vertices: 1D array of vertices that define the grid.
    """
    def __init__(self, vertices):
        self.vertices = vertices

    @classmethod
    def fromcenters(cls, centers):
        """Constructs dimension given centered grid boxes"""
        return cls(centers - (cls.spacing(centers) / 2.))

    def search(self, points):
        """Searches dimension for cell numbers and positions inside each cell.

        Fractional positions inside each cell are useful for
        interpolation techniques, especially as weights for
        linear interpolations.

        Cell numbers are zero-indexed, this makes it easy to relate values
        at grid points to arbitrary positions.

        :param points: Array of positions along dimension
        :returns: indices, fractions
        :rtype: (int, float)
        """
        fractions, indices = np.modf(self.cell_space(points))
        return indices.astype(np.int), fractions

    def cell_space(self, points):
        """Maps points into grid cell counting space

        Cells are zero-indexed to ease usage of interpolation algorithms.

        :param points: Array of positions along dimension
        :returns: Array of positions in grid cell space
        """
        return (points - self.minimum) / self.grid_spacing

    def outside(self, points):
        """Detects points outside the grid.

        Points are deemed to lie outside of the grid if they
        are less than the minimum grid position or greater than
        or equal to the maximum grid position.

        Points lying on the minimum value are included. For symmetry reasons
        points that lie on the maximum value are excluded.

        :param points: Array of positions along dimension
        :returns: True if points are outside grid
        :rtype: Boolean array

        .. seealso:: Inside grid criterion :func:`Regular1DGrid.inside`
        """
        return (points < self.minimum) | (points >= self.maximum)

    def inside(self, points):
        """Detects points inside the grid.

        Logical negation of the outside grid criteria.

        :param points: Array of positions along dimension
        :returns: True if points are inside grid
        :rtype: Boolean array

        .. seealso:: Outside grid criterion :func:`Regular1DGrid.outside`
        """
        return ~self.outside(points)

    @property
    def minimum(self):
        """Calculates minimum position.

        :returns: Minimum allowed grid position
        :rtype: float
        """
        return self.vertices[0]

    @property
    def maximum(self):
        """Calculates maximum position.

        :returns: Maximum allowed grid position
        :rtype: float
        """
        return self.minimum + self.cells * self.grid_spacing

    @property
    def cells(self):
        """Number of grid cells.

        For N vertices defining a grid there are N - 1 grid cells.

        :returns: Number of cells
        :rtype: int
        """
        return len(self.vertices) - 1

    @property
    def grid_spacing(self):
        """Grid spacing

        Regular grids are not always defined to machine precision. The
        estimated grid spacing assumes the model grid is defined to within
        4 decimal places.

        :returns: Estimated grid spacing
        :rtype: float
        """
        return self.spacing(self.vertices)

    @staticmethod
    def spacing(vertices):
        """calculates grid spacing"""
        return np.round(vertices[1] - vertices[0], decimals=4)
