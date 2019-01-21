# pylint: disable=missing-docstring, invalid-name
import unittest
import os
import numpy as np
import obsoper
try:
    import netCDF4
except ImportError:
    netCDF4 = None


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
ORCA025_FILE = os.path.join(SCRIPT_DIR,
    "data/orca025_grid.nc")
ORCA025EXT_CICE_FILE = os.path.join(SCRIPT_DIR,
    "data/prodm_op_gl.cice_20180930_00.-36.nc")


class TestGrid(unittest.TestCase):
    def setUp(self):
        self.shape = (2, 2)
        lons, lats = [[5, 5], [6, 6]], [[2, 3], [2, 3]]
        self.grid = obsoper.Grid(lons, lats)

    def test_shape(self):
        result = self.grid.shape
        expect = self.shape
        self.assertEqual(expect, result)

    def test_flatten(self):
        rlons, rlats = self.grid.flatten()
        elons, elats = [5, 5, 6, 6], [2, 3, 2, 3]
        np.testing.assert_array_almost_equal(elons, rlons)
        np.testing.assert_array_almost_equal(elats, rlats)

    def test_flat_index(self):
        rlons, rlats = self.grid.flat_index()
        elons, elats = [0, 0, 1, 1], [0, 1, 0, 1]
        np.testing.assert_array_almost_equal(elons, rlons)
        np.testing.assert_array_almost_equal(elats, rlats)


class TestMaskedGrid(unittest.TestCase):
    def setUp(self):
        self.shape = (2, 2)
        lons, lats = [[7, 7], [9, 9]], [[1, 3], [1, 3]]
        mask = [[True, False], [False, False]]
        self.grid = obsoper.MaskedGrid(lons, lats, mask)

    def test_shape(self):
        result = self.grid.shape
        expect = self.shape
        self.assertEqual(expect, result)

    def test_flatten(self):
        rlons, rlats = self.grid.flatten()
        elons, elats = [7, 9, 9], [3, 1, 3]
        np.testing.assert_array_almost_equal(elons, rlons)
        np.testing.assert_array_almost_equal(elats, rlats)

    def test_flat_index(self):
        rlons, rlats = self.grid.flat_index()
        elons, elats = [0, 1, 1], [1, 0, 1]
        np.testing.assert_array_almost_equal(elons, rlons)
        np.testing.assert_array_almost_equal(elats, rlats)


class TestCorners(unittest.TestCase):
    def setUp(self):
        # Chosen to be ordered such that
        # (i, j), (i+1, j), (i+1, j+1), (i, j+1) --> 0, 1, 2, 3
        self.values = np.array([[0, 3],
                                [1, 2]])

    def test_fixed_corners(self):
        i, j = 0, 0
        result = obsoper.Fixed.corners(self.values, i, j)
        expect = [0, 1, 2, 3]
        np.testing.assert_array_equal(expect, result)

    def test_fixed_beyond_right_edge_raises_indexerror(self):
        i, j = 1, 0
        with self.assertRaises(IndexError):
            obsoper.Fixed.corners(self.values, i, j)

    def test_cyclic_beyond_right_edge_returns_cycled_rows(self):
        i, j = 1, 0
        result = obsoper.Cyclic.corners(self.values, i, j)
        expect = [1, 0, 3, 2]
        np.testing.assert_array_equal(expect, result)

    def test_valid_cyclic_index_returns_true(self):
        ni, nj = 2, 2
        i, j = 3, 0
        result = obsoper.Cyclic.valid(ni, nj, i, j)
        expect = True
        np.testing.assert_array_equal(expect, result)

    def test_valid_cyclic_index_returns_false(self):
        ni, nj = 2, 2
        i, j = 0, 2
        result = obsoper.Cyclic.valid(ni, nj, i, j)
        expect = False
        np.testing.assert_array_equal(expect, result)


@unittest.skipIf(netCDF4 is None or not os.path.exists(ORCA025_FILE),
                 "Skip ORCA025 tests")
class TestCartesianAzimuthalORCA025(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with netCDF4.Dataset(ORCA025_FILE) as dataset:
            cls.orca025_lons = np.asarray(dataset.variables["nav_lon"][:])
            cls.orca025_lats = np.asarray(dataset.variables["nav_lat"][:])

    def setUp(self):
        self.interpolator = obsoper.CartesianAzimuthal(
            self.orca025_lons,
            self.orca025_lats)

    def test_interpolator_given_constant(self):
        lon, lat = 0, 0
        constant = 42
        values = np.full(self.orca025_lons.shape, constant)
        self.check(values, lon, lat, constant)

    def test_interpolator_given_longitude(self):
        lon, lat = 100, 0
        self.check(self.orca025_lons, lon, lat, lon)

    def test_interpolator_given_latitude(self):
        lon, lat = 100, 45
        self.check(self.orca025_lats, lon, lat, lat)

    def test_interpolate_3d(self):
        no = 3  # Observations
        nk = 10  # Levels
        ni, nj = self.orca025_lons.shape
        values = np.ones((ni, nj, nk), dtype="d")
        lons, lats = np.arange(no), np.arange(no)
        result = self.interpolator(values, lons, lats)
        expect = np.ones((nk, no))
        np.testing.assert_array_almost_equal(expect, result)

    def check(self, values, lon, lat, expect):
        result = self.interpolator(values, lon, lat)
        np.testing.assert_array_almost_equal(expect, result, decimal=4)


@unittest.skipIf(netCDF4 is None or not os.path.exists(ORCA025EXT_CICE_FILE),
                 "Skip ORCA025ext CICE tests")
class TestCartesianAzimuthalORCA025EXTCICE(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with netCDF4.Dataset(ORCA025EXT_CICE_FILE) as dataset:
            cls.grid_lons = np.asarray(dataset.variables["TLON"][:])
            cls.grid_lats = np.asarray(dataset.variables["TLAT"][:])
            cls.grid_ice = dataset.variables["aice"][:]

    def setUp(self):
        self.operator = obsoper.CartesianAzimuthal(
            self.grid_lons,
            self.grid_lats,
            self.grid_ice.mask)

    def test_search(self):
        lon = -9.9305896759
        lat = -44.4005584717
        ri, rj, rw = self.operator.search(lon, lat)
        ei, ej, ew = 484, 1108, [0.44, 0.27, 0.1, 0.17]
        self.assertEqual((ei, ej), (ri, rj))
        np.testing.assert_array_almost_equal(ew, rw, decimal=2)

    def test_interpolate_equator(self):
        result = self.operator(self.grid_ice[0], [0], [0])
        expect = [0]
        self.assert_masked_array_equal(expect, result)

    def test_interpolate_southern_ocean(self):
        result = self.operator(self.grid_ice[0], [14], [-55])
        expect = [0.61351594]
        self.assert_masked_array_equal(expect, result)

    def test_search_southern_ocean(self):
        ri, rj, rw = self.operator.search(14, -55)
        ei, ej, ew = 418, 1204, [0.531521, 0.468479, 0, 0]
        self.assertEqual((ei, ej), (ri, rj))
        np.testing.assert_array_almost_equal(ew, rw)

    def test_contains_given_multiple_cells(self):
        vertices = np.array([
            [[0, 0], [1, 0], [1, 1], [0, 1]],
            [[1, 0], [2, 0], [2, 1], [1, 1]]
        ], dtype="d")
        x = np.array([0, 2], dtype="d")
        y = np.array([0, 0], dtype="d")
        result = obsoper.CartesianAzimuthal.contains(vertices, x, y)
        expect = [True, True]
        np.testing.assert_array_equal(expect, result)

    def test_vector_interpolate_southern_ocean(self):
        result = self.operator.vector_interpolate(self.grid_ice[0], [14], [-55])
        expect = [0.61351594]
        np.testing.assert_array_almost_equal(expect, result)

    def test_vector_interpolate_equator(self):
        result = self.operator.vector_interpolate(self.grid_ice[0], [0], [0])
        expect = [0]
        np.testing.assert_array_almost_equal(expect, result)

    @unittest.skip("exhaustive test")
    def test_serial_and_vector_algorithms_agree(self):
        lons, lats = np.meshgrid(
            np.linspace(0, 360, 200),
            np.linspace(-90, 90, 200)
        )
        lons, lats = lons.flatten(), lats.flatten()
        _, ni, nj = self.grid_ice.shape
        field = self.grid_lats
        result = self.operator.vector_interpolate(field, lons, lats)
        expect = self.operator.serial_interpolate(field, lons, lats)
        self.assert_masked_array_equal(expect, result)

    def assert_masked_array_equal(self, expect, result):
        expect = np.ma.asarray(expect)
        self.assertEqual(expect.shape, result.shape)
        np.testing.assert_array_almost_equal(expect.compressed(),
                                             result.compressed())


class TestCartesianAzimuthalUnitSquare(unittest.TestCase):
    def test_unit_square(self):
        lons, lats = np.meshgrid(
            np.array([0, 1], dtype="d"),
            np.array([0, 1], dtype="d")
        )
        operator = obsoper.CartesianAzimuthal(lons, lats)
        result = operator.interpolate(lons, [0.5], [0.5])
        expect = [0.5]
        np.testing.assert_array_almost_equal(expect, result)

    def test_interpolate_given_masked_corner_returns_masked(self):
        lons, lats = np.meshgrid(
            np.array([0, 1], dtype="d"),
            np.array([0, 1], dtype="d")
        )
        values = np.ma.masked_array(
            [[1, 2], [3, 4]],
            mask=[[False, False], [False, True]],
            dtype="d")
        operator = obsoper.CartesianAzimuthal(lons, lats)
        result = operator.interpolate(values, [0.5], [0.5])
        expect = np.ma.masked_all(1)
        self.assert_masked_array_equal(expect, result)

    def test_vector_interpolate_given_masked_corner_returns_masked(self):
        lons, lats = np.meshgrid(
            np.array([0, 1], dtype="d"),
            np.array([0, 1], dtype="d")
        )
        values = np.ma.masked_array(
            [[1, 2], [3, 4]],
            mask=[[False, False], [False, True]],
            dtype="d")
        operator = obsoper.CartesianAzimuthal(lons, lats)
        result = operator.vector_interpolate(values, [0.5], [0.5])
        expect = np.ma.masked_all(1)
        self.assert_masked_array_equal(expect, result)

    def test_train_interpolator_on_observed_positions(self):
        grid_lons, grid_lats = np.meshgrid(
            np.array([0, 1], dtype="d"),
            np.array([0, 1], dtype="d")
        )
        obs_lons, obs_lats = [0.1], [0.1]
        field = np.array([[0, 2],
                          [1, 3]], dtype="d")
        operator = obsoper.CartesianAzimuthal(grid_lons, grid_lats)
        interpolator = operator.train(obs_lons, obs_lats)
        result = interpolator(field)
        expect = [0.299993146]
        np.testing.assert_array_almost_equal(expect, result)

    def assert_masked_array_equal(self, expect, result):
        expect = np.ma.asarray(expect)
        self.assertEqual(expect.shape, result.shape)
        np.testing.assert_array_almost_equal(expect.compressed(),
                                             result.compressed())


class TestStereographic(unittest.TestCase):
    def setUp(self):
        self.radius_45 = 2 * (np.sqrt(2) - 1)

    def test_stereographic_given_zero_ninety(self):
        self.check(0, 90, (0, 0))

    def test_stereographic_given_zero_forty_five(self):
        self.check(0, 45, (0, -self.radius_45))

    def test_stereographic_given_one_eighty_forty_five(self):
        self.check(180, 45, (0, self.radius_45))

    def test_stereographic_given_ninety_forty_five(self):
        self.check(90, 45, (self.radius_45, 0))

    def test_stereographic_given_vector(self):
        self.check([90, 180], [45, 45], ([self.radius_45, 0],
                                         [0, self.radius_45]))

    def check(self, lon, lat, expect):
        ex, ey = expect
        rx, ry = obsoper.CartesianAzimuthal.stereographic(
            lon,
            lat)
        np.testing.assert_array_almost_equal(ex, rx)
        np.testing.assert_array_almost_equal(ey, ry)


class TestStereographicCentralLonLat(unittest.TestCase):
    def test_stereographic_given_central_lon_lat(self):
        self.check(
            lon=10,
            lat=10,
            central_lon=0,
            central_lat=0,
            ex=0.17362783,
            ey=0.17630632)

    def test_stereographic_given_different_central_lon_lat(self):
        self.check(
            lon=10,
            lat=10,
            central_lon=10,
            central_lat=0,
            ex=0.,
            ey=0.174977)

    def test_stereographic_given_central_lon_lat_zero_ten(self):
        self.check(
            lon=10,
            lat=10,
            central_lon=0,
            central_lat=10,
            ex=0.17227927,
            ey=0.00261731)

    def test_stereographic_given_vector_central_lon_lat(self):
        self.check(
            lon=10,
            lat=10,
            central_lon=[0, 10, 0],
            central_lat=[0, 0, 10],
            ex=[0.17362783, 0., 0.17227927],
            ey=[0.17630632, 0.174977, 0.00261731])

    def test_stereographic_given_vector_lons_lats_and_central_lons_lats(self):
        lons = np.array([[0, 10, 10],
                         [1, 10, 10],
                         [2, 10, 10],
                         [3, 10, 10]])
        lats = np.array([[0, 10, 10],
                         [1, 10, 10],
                         [2, 10, 10],
                         [3, 10, 10]])
        self.assertEqual(lons.shape, (4, 3))
        self.assertEqual(lats.shape, (4, 3))
        self.check(
            lon=lons,
            lat=lats,
            central_lon=[0, 10, 0],
            central_lat=[0, 0, 10],
            ex=[[0.,         0., 0.17227927],
                [0.01745241, 0., 0.17227927],
                [0.03489949, 0., 0.17227927],
                [0.05233591, 0., 0.17227927]],
            ey=[[0.,         0.174977, 0.00261731],
                [0.01745506, 0.174977, 0.00261731],
                [0.03492076, 0.174977, 0.00261731],
                [0.05240773, 0.174977, 0.00261731]])

    def check(self, lon, lat, central_lon, central_lat, ex, ey):
        rx, ry = obsoper.CartesianAzimuthal.stereographic(
            lon,
            lat,
            central_lon=central_lon,
            central_lat=central_lat)
        np.testing.assert_array_almost_equal(ex, rx)
        np.testing.assert_array_almost_equal(ey, ry)
