# pylint: disable=missing-docstring, invalid-name
import unittest
import numpy as np
from obsoper import (domain,
                     Polygon)


class TestInside(unittest.TestCase):
    def setUp(self):
        # Trapezoid 2x2 grid
        self.grid_lons = [[0, 1],
                          [3, 2]]
        self.grid_lats = [[0, 1],
                          [0, 1]]
        self.obs_lons = [-1, 0.1, 1.5]
        self.obs_lats = [0.5, 0.5, 0.5]

    def test_inside_given_latitude_band(self):
        self.check_inside("band", [True, True, True])

    def test_inside_given_lonlat(self):
        self.check_inside("lonlat", [False, True, True])

    def test_inside_given_polygon(self):
        self.check_inside("polygon", [False, False, True])

    def check_inside(self, kind, expect):
        result = domain.inside(self.grid_lons,
                               self.grid_lats,
                               self.obs_lons,
                               self.obs_lats,
                               kind=kind)
        np.testing.assert_array_equal(expect, result)


class TestPolygon(unittest.TestCase):
    def test_inside_given_bottom_left_corner_returns_true(self):
        self.check_inside([0, 1], [0, 1], 0, 0, True)

    def test_inside_given_bottom_right_corner_returns_true(self):
        self.check_inside([0, 1], [0, 1], 1, 0, True)

    def test_inside_given_top_left_corner_returns_true(self):
        self.check_inside([0, 1], [0, 1], 0, 1, True)

    def test_inside_given_top_right_corner_returns_true(self):
        self.check_inside([0, 1], [0, 1], 1, 1, True)

    def test_inside_given_point_too_east_returns_false(self):
        self.check_inside([0, 1], [0, 1], 1.1, 0, False)

    def test_inside_given_point_too_west_returns_false(self):
        self.check_inside([0, 1], [0, 1], -0.1, 0, False)

    def test_inside_given_point_too_north_returns_false(self):
        self.check_inside([0, 1], [0, 1], 0, 1.1, False)

    def test_inside_given_point_too_south_returns_false(self):
        self.check_inside([0, 1], [0, 1], 0, -0.1, False)

    def check_inside(self,
                     grid_longitudes,
                     grid_latitudes,
                     longitudes,
                     latitudes,
                     expect):
        grid_longitudes, grid_latitudes = np.meshgrid(grid_longitudes,
                                                      grid_latitudes,
                                                      indexing="ij")
        fixture = Polygon(grid_longitudes,
                          grid_latitudes)
        result = fixture.inside(longitudes, latitudes)
        self.assertEqual(expect, result)


class TestIrregularPolygon(unittest.TestCase):
    def setUp(self):
        # Trapezoid domain
        longitudes = [[0, 3],
                      [1, 2]]
        latitudes = [[0, 0],
                     [1, 1]]
        self.fixture = Polygon(longitudes, latitudes)

    def test_inside_given_point_inside_bounding_box_outside_domain(self):
        result = self.fixture.inside(0.1, 0.9)
        expect = False
        self.assertEqual(expect, result)

    def test_inside_given_points_inside_bounding_box_and_domain(self):
        result = self.fixture.inside([-0.1, 0.1, 1.0, 2.4],
                                     [0.9, 0.9, 0.9, 0.9])
        expect = [False, False, True, False]
        np.testing.assert_array_equal(expect, result)
