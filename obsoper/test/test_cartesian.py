# pylint: disable=missing-docstring, invalid-name
import unittest
import os
import netCDF4
import numpy as np
import obsoper

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
ORCA025EXT_CICE_FILE = os.path.join(SCRIPT_DIR,
                                    "data/prodm_op_gl.cice_20180930_00.-36.nc")


@unittest.skipIf(not os.path.exists(ORCA025EXT_CICE_FILE), "no ORCA025 CICE file available")
class TestORCA025EXTCICE(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with netCDF4.Dataset(ORCA025EXT_CICE_FILE) as dataset:
            cls.grid_lons = np.asarray(dataset.variables["TLON"][:])
            cls.grid_lats = np.asarray(dataset.variables["TLAT"][:])
            cls.grid_ice = dataset.variables["aice"][:]

    def setUp(self):
        self.interpolator = obsoper.ORCAExtended(
            self.grid_lons,
            self.grid_lats,
            self.grid_ice.mask)

    def test_search(self):
        lon = -9.9305896759
        lat = -44.4005584717
        ri, rj, rw = self.interpolator.search(lon, lat)
        ei, ej, ew = 484, 1108, [0.44, 0.27, 0.1, 0.17]
        self.assertEqual((ei, ej), (ri, rj))
        np.testing.assert_array_almost_equal(ew, rw, decimal=2)

    def test_interpolate_equator(self):
        result = self.interpolator(self.grid_ice[0], [0], [0])
        expect = [0]
        self.assert_masked_array_equal(expect, result)

    def test_interpolate_southern_ocean(self):
        result = self.interpolator(self.grid_ice[0], [14], [-55])
        expect = [0.61351594]
        self.assert_masked_array_equal(expect, result)

    def test_search_southern_ocean(self):
        ri, rj, rw = self.interpolator.search(14, -55)
        ei, ej, ew = 418, 1204, [0.53151717, 0.46848283, 0, 0]
        self.assertEqual((ei, ej), (ri, rj))
        np.testing.assert_array_almost_equal(ew, rw)

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
        rx, ry = obsoper.ORCAExtended.stereographic(
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
            ex=[[0., 0., 0.17227927],
                [0.01745241, 0., 0.17227927],
                [0.03489949, 0., 0.17227927],
                [0.05233591, 0., 0.17227927]],
            ey=[[0., 0.174977, 0.00261731],
                [0.01745506, 0.174977, 0.00261731],
                [0.03492076, 0.174977, 0.00261731],
                [0.05240773, 0.174977, 0.00261731]])

    def check(self, lon, lat, central_lon, central_lat, ex, ey):
        rx, ry = obsoper.ORCAExtended.stereographic(
            lon,
            lat,
            central_lon=central_lon,
            central_lat=central_lat)
        np.testing.assert_array_almost_equal(ex, rx)
        np.testing.assert_array_almost_equal(ey, ry)
