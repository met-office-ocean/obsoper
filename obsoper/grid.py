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
import warnings
from collections import namedtuple
from scipy.spatial import cKDTree
import numpy as np
from . import (exceptions,
               cell,
               coordinates,
               walk)


SearchResult = namedtuple("SearchResult", ("ilon", "ilat", "dilon", "dilat"))


def lower_left(grid_longitudes,
               grid_latitudes,
               observed_longitudes,
               observed_latitudes,
               search="cartesian"):
    """Find lower left grid indicies associated with observed locations

    :returns: i, j arrays of indices
    """
    return _algorithm(search)(grid_longitudes,
                              grid_latitudes).lower_left(observed_longitudes,
                                                         observed_latitudes)

def _algorithm(search):
    if search.lower() == "cartesian":
        return CartesianSearch
    elif search.lower() == "tripolar":
        return TripolarSearch


class TripolarSearch(object):
    """Finds lower-left hand grid point nearest observation

    Uses a longitude/latitude KD-Tree search followed by an iterative
    walk algorithm to converge on a Cell surrounding a point.

    The walk algorithm knows the layout of the grid and the great circle
    direction needed to move closer to the point.

    :param grid_longitudes: 2D array dimensioned (X, Y)
    :param grid_latitudes: 2D array dimensioned (X, Y)
    """
    def __init__(self, grid_longitudes, grid_latitudes):
        grid_longitudes = np.asarray(grid_longitudes)
        grid_latitudes = np.asarray(grid_latitudes)
        self.neighbour = LonLatNeighbour(grid_longitudes, grid_latitudes)
        self.walk = walk.Walk.tripolar(grid_longitudes,
                                       grid_latitudes)

    def lower_left(self, longitudes, latitudes):
        """Detect lower left corner index of four corners surrounding
        observations.

        :param longitudes: array shaped ([N],)
        :param latituds: array shaped ([N],)
        :returns: (i, j) index arrays representing lower left hand corners
        """
        i, j = self.nearest(longitudes, latitudes)
        ri, rj = self.walk.query(longitudes, latitudes, i, j)
        return np.asarray(ri, dtype="i"), np.asarray(rj, dtype="i")

    def nearest(self, longitudes, latitudes):
        """Find nearest neighbour"""
        return self.neighbour.nearest(longitudes, latitudes)


class Search(TripolarSearch):
    """Finds lower-left hand grid point nearest observation

    .. deprecated:: 0.0.5
       Use :class:`TripolarSearch` instead

    Uses a longitude/latitude KD-Tree search followed by an iterative
    walk algorithm to converge on a Cell surrounding a point.

    The walk algorithm knows the layout of the grid and the great circle
    direction needed to move closer to the point.

    :param grid_longitudes: 2D array dimensioned (X, Y)
    :param grid_latitudes: 2D array dimensioned (X, Y)
    """
    def __init__(self, *args, **kwargs):
        warnings.warn("Search deprecated use TripolarSearch instead")
        super(Search, self).__init__(*args, **kwargs)


class CartesianSearch(object):
    """Search for corners surrounding observations

    Uses a Cartesian coordinate KD-Tree search followed by an
    iteration through the N nearest neighbours to find the Cell
    that contains each point.

    .. note:: No further knowledge of the grid layout is needed since the
              KD-Tree search should by the pigeon hole principle detect
              the correct corner

    :param grid_longitudes: 2D array dimensioned (X, Y)
    :param grid_latitudes: 2D array dimensioned (X, Y)
    """
    def __init__(self, grid_longitudes, grid_latitudes):
        grid_longitudes = np.asarray(grid_longitudes)
        grid_latitudes = np.asarray(grid_latitudes)

        self.ni, self.nj = grid_longitudes.shape
        self._nearest_neighbour = CartesianNeighbour(grid_longitudes,
                                                     grid_latitudes)
        self._positions = np.dstack([grid_longitudes,
                                     grid_latitudes]).astype("d")

    @property
    def k(self):
        """number of nearest neighbours based on grid size"""
        return min(self.ni * self.nj, 8)

    def lower_left(self, longitudes, latitudes):
        """find lower left corner index of corners surrounding observations

        .. note:: assumes positions provided have already been checked
                  for domain membership

        :param longitudes: array shaped ([N],)
        :param latituds: array shaped ([N],)
        :returns: (i, j) index arrays representing lower left hand corners
        """
        i_near, j_near = self._nearest_neighbour.nearest(longitudes,
                                                         latitudes,
                                                         k=self.k)
        i_result, j_result = [], []
        for longitude, latitude, i_indices, j_indices in zip(longitudes,
                                                             latitudes,
                                                             i_near,
                                                             j_near):
            i, j = self.detect(longitude,
                               latitude,
                               i_indices,
                               j_indices)
            i_result.append(i)
            j_result.append(j)
        return np.array(i_result), np.array(j_result)

    def detect(self, longitude, latitude, i_indices, j_indices):
        """Detect lower left corner near a point given several candidates"""
        for i, j in zip(i_indices, j_indices):
            # Note: this check might be unnescessary for cyclic grids
            if (i == (self.ni - 1)) or (j == (self.nj - 1)):
                # point on boundary
                continue
            if self.detect_one(longitude, latitude, i, j):
                return i, j

        # Report search failure to user
        message = "could not find ({}, {}) given i={}, j={}".format(longitude,
                                                                    latitude,
                                                                    i_indices,
                                                                    j_indices)
        raise exceptions.SearchFailed(message)

    def detect_one(self, longitude, latitude, i, j):
        """Check that a particular point is contained in a cell"""
        grid_cell = cell.Cell.from_positions(self._positions, i, j)
        return grid_cell.contains(longitude, latitude)


class LonLatNeighbour(object):
    """Nearest neighbour search in longitude/latitude space

    KD-Tree algorithm using longitude and latitude as the dimensions.

    :param longitudes: 2D array shaped (X, Y)
    :param latitudes: 2D array shaped (X, Y)
    """
    def __init__(self, longitudes, latitudes):
        self.shape = longitudes.shape
        self.tree = cKDTree(self.points(longitudes,
                                        latitudes))

    def nearest(self, longitudes, latitudes, **query_args):
        """Find nearest neighbour grid indexes

        :param query_args: keyword arguments to be passed to cKDTree.query()
        :returns: i, j arrays of indices
        """
        _, indices = self.tree.query(self.points(longitudes,
                                                 latitudes),
                                     **query_args)
        return np.unravel_index(indices,
                                self.shape)

    @staticmethod
    def points(longitudes, latitudes):
        """Convert arrays of longitudes and latitudes to single array"""
        return np.array([np.ravel(longitudes),
                         np.ravel(latitudes)]).T


class NearestNeighbour(LonLatNeighbour):
    """Nearest neighbour search

    .. deprecated:: 0.0.5
       Use :class:`LonLatNeighbour` instead

    Unstructured nearest neighbour search in 2D. Takes advantage of
    KD-Tree algorithm.

    :param longitudes: 2D array shaped (X, Y)
    :param latitudes: 2D array shaped (X, Y)
    """
    def __init__(self, *args, **kwargs):
        warnings.warn("NearestNeighbour deprecated use LonLatNeighbour instead")
        super(NearestNeighbour, self).__init__(*args, **kwargs)


class CartesianNeighbour(object):
    """Nearest neighbour algorithm that works in Cartesian coordinates

    KD-Tree search in Cartesian space. Simplifies usage of scipy.spatial.KDTree
    by handling conversion of array shapes and mapping between geographic
    and Cartesian coordinates.

    .. note: Cartesian representation is on a unit sphere

    :param longitudes: 2D array shaped (X, Y)
    :param latitudes: 2D array shaped (X, Y)
    """
    def __init__(self, longitudes, latitudes):
        self.shape = longitudes.shape
        self.tree = cKDTree(self.points(longitudes,
                                        latitudes))

    def nearest(self, longitudes, latitudes, **query_args):
        """Find nearest neighbour grid indexes

        Converts longitudes and latitudes to (x, y, z) before calling
        cKDTree.query().

        :param query_args: keyword arguments to be passed to cKDTree.query()
        :returns: i, j arrays of indices of points nearest to given positions
        """
        _, indices = self.tree.query(self.points(longitudes,
                                                 latitudes),
                                     **query_args)
        return np.unravel_index(indices,
                                self.shape)

    @staticmethod
    def points(longitudes, latitudes):
        """Convert arrays of longitudes and latitudes to single array

        :param longitudes: array of longitudes
        :param latitudes: array of latitudes
        :returns: Cartesian representation array shaped [N, 3]
        """
        x, y, z = coordinates.cartesian(longitudes, latitudes)
        return np.array([np.ravel(x),
                         np.ravel(y),
                         np.ravel(z)]).T


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
            raise exceptions.NotInGrid
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
        return (points - self.initial) / self.grid_spacing

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
        return np.ma.min(self.vertices)

    @property
    def initial(self):
        """Calculates initial position.

        :returns: First grid position
        :rtype: float
        """
        return self.vertices[0]

    @property
    def maximum(self):
        """Calculates maximum position.

        :returns: Maximum allowed grid position
        :rtype: float
        """
        return self.minimum + self.cells * np.ma.abs(self.grid_spacing)

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
