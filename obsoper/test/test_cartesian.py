# pylint: disable=missing-docstring, invalid-name
import unittest
import os
import netCDF4
import numpy as np
import obsoper

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
ORCA025EXT_CICE_FILE = os.path.join(SCRIPT_DIR,
                                    "data/prodm_op_gl.cice_20180930_00.-36.nc")


class TestORCA025EXTCICE(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with netCDF4.Dataset(ORCA025EXT_CICE_FILE) as dataset:
            cls.grid_lons = dataset.variables["TLON"][:]
            cls.grid_lats = dataset.variables["TLAT"][:]
            cls.grid_ice = dataset.variables["aice"][:]

    def test_interpolate(self):
        lon = -9.9305896759
        lat = -44.4005584717
        interpolator = obsoper.ORCAExtended(
            self.grid_lons,
            self.grid_lats,
            self.grid_ice.mask,
            lon,
            lat
        )
        result = interpolator(self.grid_ice)
        expect = [0]
        np.testing.assert_array_almost_equal(expect, result)
