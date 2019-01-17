"""Tripolar grid search/interpolation"""
import scipy.spatial
import numpy as np
from numpy import sin, cos, deg2rad
from obsoper import (grid, bilinear, orca, cell)
from obsoper.corners import (
    select_corners,
    select_field,
    correct_corners,
    mask_corners)
from pkg_resources import parse_version
import netCDF4


class SearchFailed(Exception):
    pass


class ORCA(object):
    """General purpose ORCA grid interpolator"""
    def __init__(self, lons, lats):
        if isinstance(lons, list):
            lons = np.array(lons, dtype="d")
        if isinstance(lats, list):
            lats = np.array(lats, dtype="d")
        self.lons = lons
        self.lats = lats
        nx, ny = self.lons.shape
        gi, gj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        self.gi = gi.flatten()
        self.gj = gj.flatten()
        x, y, z = ORCAExtended.cartesian(
            self.lons.flatten(),
            self.lats.flatten())
        self.tree = self._kdtree(np.array([x, y, z]).T)

    @classmethod
    def load(cls, path, lon="nav_lon", lat="nav_lat"):
        """Helper to load interpolator from file"""
        with netCDF4.Dataset(path) as dataset:
            lons = dataset.variables[lon][:]
            lats = dataset.variables[lat][:]
            return cls(lons, lats)

    @staticmethod
    def _kdtree(points):
        if parse_version(scipy.__version__) >= parse_version("0.16.0"):
            return scipy.spatial.cKDTree(points, balanced_tree=False)
        else:
            return scipy.spatial.cKDTree(points)

    def __call__(self, *args, **kwargs):
        return self.interpolate(*args, **kwargs)

    def interpolate(self, values, lon, lat):
        x, y, z = ORCAExtended.cartesian(lon, lat)
        eps, indices = self.tree.query([x, y, z], k=12)
        for ti in indices:
            i = self.gi[ti]
            j = self.gj[ti]
            corner_lons = self.corners(self.lons, i, j)
            corner_lats = self.corners(self.lats, i, j)
            cx, cy = ORCAExtended.gnomonic(
                corner_lons,
                corner_lats,
                central_lon=lon,
                central_lat=lat)
            px, py = 0., 0.
            vertices = np.asarray([cx, cy], dtype=np.double).T
            if cell.Cell(vertices).contains(px, py):
                weights = bilinear.interpolation_weights(vertices, px, py)
                return np.sum(self.corners(values, i, j) * weights)

    @staticmethod
    def corners(array, i, j, dtype="d"):
        return np.ma.asarray([
            array[i, j],
            array[i + 1, j],
            array[i + 1, j + 1],
            array[i, j + 1]], dtype=dtype)


class ORCAExtended(object):
    """General purpose extended grid interpolator"""
    def __init__(
            self,
            grid_lons,
            grid_lats,
            grid_mask):
        self.grid_lons = grid_lons
        self.grid_lats = grid_lats
        mask = grid_mask
        nx, ny = grid_lons.shape
        glons = np.ma.masked_array(grid_lons, mask).compressed()
        glats = np.ma.masked_array(grid_lats, mask).compressed()
        gi, gj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        self.gi = np.ma.masked_array(gi, mask).compressed()
        self.gj = np.ma.masked_array(gj, mask).compressed()
        x, y, z = self.cartesian(glons, glats)
        self.tree = scipy.spatial.cKDTree(np.array([x, y, z]).T)

    @classmethod
    def unmasked(cls, lons, lats):
        return cls(lons, lats, False)

    def __call__(self, *args):
        return self.interpolate(*args)

    def interpolate(self, field, lons, lats):
        result = np.ma.masked_all(len(lons), dtype=np.double)
        for io in range(len(lons)):
            try:
                i, j, weights = self.search(lons[io], lats[io])
            except SearchFailed:
                continue
            pts = (i + np.array([0, 1, 1, 0]),
                   j + np.array([0, 0, 1, 1]))
            values = field[pts]

            # Skip points not surrounded by four corners
            if isinstance(values, np.ma.masked_array):
                if values.mask.any():
                    continue

            result[io] = np.sum(weights * values)
        return result

    def search(self, lon, lat):
        x, y, z = self.cartesian(lon, lat)
        eps, self.i = self.tree.query([x, y, z], k=self.k)
        for i, j in zip(self.gi[self.i], self.gj[self.i]):
            pts = (i + np.array([0, 1, 1, 0]),
                   j + np.array([0, 0, 1, 1]))
            try:
                lons = self.grid_lons[pts]
                lats = self.grid_lats[pts]
            except IndexError:
                continue
            x, y = self.stereographic(
                lons,
                lats,
                central_lon=lon,
                central_lat=lat)
            vertices = np.asarray([x, y], dtype=np.double).T
            if self.contains(vertices, 0., 0.):
                return i, j, self.weights(vertices, 0., 0.)
        raise SearchFailed("{} {} not found".format(lon, lat))

    @property
    def k(self):
        return min(12, len(self.gi))

    def weights(self, vertices, x, y):
        return bilinear.interpolation_weights(vertices, x, y)

    def vector_interpolate(self, field, lons, lats):
        """Vectorised approach to Cartesian/Stereographic interpolation"""
        if isinstance(lons, list):
            lons = np.array(lons)
        if isinstance(lats, list):
            lats = np.array(lats)

        x, y, z = self.cartesian(lons, lats)
        eps, neighbours = self.tree.query(np.array([x, y, z], dtype="d").T,
                                          k=self.k)

        global_found = np.zeros(len(lons), dtype=bool)
        global_search_i = np.zeros(len(lons), dtype="i")
        global_search_j = np.zeros(len(lons), dtype="i")
        global_weights = np.ma.masked_all((len(lons), 4), dtype="d")
        for ni in range(self.k):
            nni = neighbours[:, ni]

            i, j = self.gi[nni], self.gj[nni]

            nx, ny = self.grid_lons.shape
            included = (i < (nx - 1)) & (j < (ny - 1))

            mask = ~global_found & included

            search_i = i[mask]
            search_j = j[mask]
            search_lons = lons[mask]
            search_lats = lats[mask]

            corner_lons = self.corners(self.grid_lons, search_i, search_j)
            corner_lats = self.corners(self.grid_lats, search_i, search_j)

            corners = np.empty((len(search_lons), 4, 2), dtype="d")
            x, y = self.stereographic(
                corner_lons,
                corner_lats,
                central_lon=search_lons,
                central_lat=search_lats)
            corners[:, :, 0] = x.T
            corners[:, :, 1] = y.T
            zeros = np.zeros(len(search_lons))
            contained = cell.contains(corners, zeros, zeros)
            if not any(contained):
                continue

            weights = bilinear.interpolation_weights(
                corners[contained],
                zeros[contained],
                zeros[contained])

            pts = np.where(mask)[0][contained]
            global_found[pts] = contained
            global_search_i[pts] = search_i[contained]
            global_search_j[pts] = search_j[contained]
            global_weights[pts] = weights

        i, j = global_search_i, global_search_j
        weights = global_weights
        values = self.corners(field, i, j)
        return np.ma.sum(values.T * weights, axis=1)

    @staticmethod
    def corners(array, i, j, dtype="d"):
        return np.ma.asarray([
            array[i, j],
            array[i + 1, j],
            array[i + 1, j + 1],
            array[i, j + 1]], dtype=dtype)

    @staticmethod
    def contains(vertices, x, y):
        if (np.ndim(vertices) == 3) and (np.ndim(x) == 1):
            return cell.contains(vertices, x, y)
        return cell.Cell(vertices).contains(x, y)

    @staticmethod
    def cartesian(lon, lat):
        """Project into 3D Cartesian space"""
        lon, lat = np.deg2rad(lon), np.deg2rad(lat)
        z = np.sin(lat)
        y = np.cos(lat) * np.sin(lon)
        x = np.cos(lat) * np.cos(lon)
        return x, y, z

    @staticmethod
    def stereographic(lon, lat,
                      central_lon=0,
                      central_lat=90):
        """Stereographic projection through point

        Conformal map centered on observation that can be used
        to locate surrounding grid points
        """
        lam, lam0 = deg2rad(lon), deg2rad(central_lon)
        phi, phi1 = deg2rad(lat), deg2rad(central_lat)
        d = (1 +
             (sin(phi1) * sin(phi)) +
             (cos(phi1) * cos(phi) * cos(lam - lam0)))
        k = 2 / d
        x = k * cos(phi) * sin(lam - lam0)
        y = k * (cos(phi1) * sin(phi) -
                 sin(phi1) * cos(phi) * cos(lam - lam0))
        return x, y

    @staticmethod
    def gnomonic(lon, lat,
            central_lon=0,
            central_lat=90):
        """Gnomonic projection through point"""
        lam, lam0 = deg2rad(lon), deg2rad(central_lon)
        phi, phi1 = deg2rad(lat), deg2rad(central_lat)
        d = (sin(phi1) * sin(phi) +
             cos(phi1) * cos(phi) * cos(lam - lam0))
        x = (cos(phi) * sin(lam - lam0)) / d
        y = (cos(phi1) * sin(phi) -
             sin(phi1) * cos(phi) * cos(lam - lam0)) / d
        return x, y

    @staticmethod
    def lambert_azimuthal_equal_area(
            lon,
            lat,
            central_lon=0,
            central_lat=90):
        """Lambert azimuthal equal area projection through point"""
        lam, lam0 = deg2rad(lon), deg2rad(central_lon)
        phi, phi1 = deg2rad(lat), deg2rad(central_lat)
        d = (1 +
             sin(phi1) * sin(phi) +
             cos(phi1) * cos(phi) * cos(lam - lam0))
        k = np.sqrt(2 / d)
        x = k * cos(phi) * sin(lam - lam0)
        y = k * (cos(phi1) * sin(phi) -
                 sin(phi1) * cos(phi) * cos(lam - lam0))
        return x, y


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
    """
    def __init__(self,
                 grid_longitudes,
                 grid_latitudes,
                 observed_longitudes,
                 observed_latitudes,
                 has_halo=False):
        # Screen grid cells inside halo
        self.has_halo = has_halo
        if self.has_halo:
            grid_longitudes = orca.remove_halo(grid_longitudes)
            grid_latitudes = orca.remove_halo(grid_latitudes)

        # Cast positions as doubles
        observed_longitudes = np.asarray(observed_longitudes, dtype="d")
        observed_latitudes = np.asarray(observed_latitudes, dtype="d")
        grid_longitudes = np.asarray(grid_longitudes, dtype="d")
        grid_latitudes = np.asarray(grid_latitudes, dtype="d")
        self.n_observations = len(observed_longitudes)

        # Detect observations inside domain
        southern_edge = np.ma.min(grid_latitudes)
        northern_edge = np.ma.max(grid_latitudes)
        self.included = ((observed_latitudes >= southern_edge) &
                         (observed_latitudes <= northern_edge))

        if self.included.any():
            included_longitudes = observed_longitudes[self.included]
            included_latitudes = observed_latitudes[self.included]

            # Detect grid cells containing observations
            self.i, self.j = grid.lower_left(grid_longitudes,
                                             grid_latitudes,
                                             included_longitudes,
                                             included_latitudes,
                                             search="tripolar")

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
