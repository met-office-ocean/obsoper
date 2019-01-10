# pylint: disable=missing-docstring, invalid-name
import unittest
import obsoper.tripolar
import numpy as np


class TestStereographic(unittest.TestCase):
    def test_pole_returns_origin(self):
        self.check(0, 90, 0, 0)

    def test_45N_returns_one_zero(self):
        self.check(0, 45, 1, 0)

    def test_45N_180E_returns_minus_one_zero(self):
        self.check(180, 45, -1, 0)

    def check(self, lons, lats, ex, ey):
        rx, ry = obsoper.tripolar.stereographic(lons, lats)
        np.testing.assert_array_almost_equal(rx, ex)
        np.testing.assert_array_almost_equal(ry, ey)
