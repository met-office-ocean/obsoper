# pylint: disable=missing-docstring, invalid-name
import unittest
import numpy as np
from obsoper import Domain


class TestDomain(unittest.TestCase):
    def test_contains_given_bottom_left_corner_returns_true(self):
        self.check_contains([0, 1], [0, 1], 0, 0, True)

    def test_contains_given_bottom_right_corner_returns_true(self):
        self.check_contains([0, 1], [0, 1], 1, 0, True)

    def test_contains_given_top_left_corner_returns_true(self):
        self.check_contains([0, 1], [0, 1], 0, 1, True)

    def test_contains_given_top_right_corner_returns_true(self):
        self.check_contains([0, 1], [0, 1], 1, 1, True)

    def test_contains_given_point_too_east_returns_false(self):
        self.check_contains([0, 1], [0, 1], 1.1, 0, False)

    def test_contains_given_point_too_west_returns_false(self):
        self.check_contains([0, 1], [0, 1], -0.1, 0, False)

    def test_contains_given_point_too_north_returns_false(self):
        self.check_contains([0, 1], [0, 1], 0, 1.1, False)

    def test_contains_given_point_too_south_returns_false(self):
        self.check_contains([0, 1], [0, 1], 0, -0.1, False)

    def check_contains(self,
                       grid_longitudes,
                       grid_latitudes,
                       longitudes,
                       latitudes,
                       expect):
        grid_longitudes, grid_latitudes = np.meshgrid(grid_longitudes,
                                                      grid_latitudes,
                                                      indexing="ij")
        fixture = Domain(grid_longitudes,
                         grid_latitudes)
        result = fixture.contains(longitudes, latitudes)
        self.assertEqual(expect, result)
