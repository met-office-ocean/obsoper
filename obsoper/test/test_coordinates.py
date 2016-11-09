# pylint: disable=missing-docstring, invalid-name
import unittest
import numpy as np
import obsoper


class TestCartesian(unittest.TestCase):
    def test_cartesian_given_lists_returns_arrays(self):
        x, y, z = obsoper.cartesian([], [])
        self.assertIsInstance(x, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsInstance(z, np.ndarray)

    def test_cartesian_given_empty_arrays_returns_empty_arrays(self):
        result = obsoper.cartesian([], [])
        expect = [], [], []
        self.assertCoordinatesEqual(expect, result)

    def test_cartesian_given_greenwich_equator_returns_unit_x(self):
        self.check_cartesian(longitudes=[0], latitudes=[0],
                             x=[1], y=[0], z=[0])

    def test_cartesian_given_north_pole_returns_unit_z(self):
        self.check_cartesian(longitudes=[0], latitudes=[90],
                             x=[0], y=[0], z=[1])

    def check_cartesian(self, longitudes, latitudes, x, y, z):
        result = obsoper.cartesian(longitudes, latitudes)
        expect = x, y, z
        self.assertCoordinatesEqual(expect, result)

    @staticmethod
    def assertCoordinatesEqual(expect, result):
        np.testing.assert_array_almost_equal(expect[0], result[0])
        np.testing.assert_array_almost_equal(expect[1], result[1])
        np.testing.assert_array_almost_equal(expect[2], result[2])
