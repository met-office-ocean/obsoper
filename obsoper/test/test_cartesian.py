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
            cls.grid_lons = dataset.variables["TLON"][:]
            cls.grid_lats = dataset.variables["TLAT"][:]
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
        expect = [1]
        self.assert_masked_array_equal(expect, result)

    def assert_masked_array_equal(self, expect, result):
        expect = np.ma.asarray(expect)
        self.assertEqual(expect.shape, result.shape)
        np.testing.assert_array_almost_equal(expect.compressed(),
                                             result.compressed())


class TestStereographic(unittest.TestCase):
    def setUp(self):
        self.radius_45 = 2 * (np.sqrt(2) - 1)

    def test_stereographic_given_zero_ninety(self):
        self.check(0, 90, 0, 0)

    def test_stereographic_given_zero_forty_five(self):
        self.check(0, 45, 0, -self.radius_45)

    def test_stereographic_given_one_eighty_forty_five(self):
        self.check(180, 45, 0, self.radius_45)

    def test_stereographic_given_ninety_forty_five(self):
        self.check(90, 45, self.radius_45, 0)

    def check(self, lon, lat, ex, ey):
        rx, ry = obsoper.ORCAExtended.stereographic(
            lon,
            lat)
        np.testing.assert_array_almost_equal(ex, rx)
        np.testing.assert_array_almost_equal(ey, ry)
