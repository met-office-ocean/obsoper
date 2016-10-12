# pylint: disable=missing-docstring, invalid-name
import unittest
import numpy as np
from obsoper import grid
from obsoper.grid import (Regular2DGrid,
                          Regular1DGrid,
                          SearchResult)
from obsoper.exceptions import NotInGrid


class TestSearch(unittest.TestCase):
    def setUp(self):
        # NEMO netcdf arrays are dimensioned (y, x) or (latitude, longitude)
        # This is confusing as the natural correspondence between
        # (i, j) and (x, y) is reversed.

        # Longitudes vary with axis=0 or up/down in numpy notation
        self.longitudes = np.array([[0, 0, 0],
                                    [2, 2, 2],
                                    [5, 5, 5]])

        # Latitudes vary with axis=1 or left/right in numpy notation
        self.latitudes = np.array([[0, 1, 5],
                                   [0, 1, 5],
                                   [0, 1, 5]])

        # Irregular grid fixture looks like this in geospatial coordinates
        # 5 X--X----X
        #   |  |    |
        #   |  |    |
        # 1 X--X----X
        #   |  |    |
        # 0 X--X----X
        #   0  2    5
        self.fixture = grid.Search(self.longitudes, self.latitudes)

    def test_nearest_given_origin_returns_0_0(self):
        longitude, latitude = 0, 0
        result = self.fixture.nearest(longitude, latitude)
        expect = (np.array([0]),
                  np.array([0]))
        self.assertEqual(expect, result)

    def test_nearest_given_upper_right_corner_returns_3_3(self):
        longitude, latitude = 5, 5
        result = self.fixture.nearest(longitude, latitude)
        expect = (np.array([2]),
                  np.array([2]))
        self.assertEqual(expect, result)

    def test_nearest_given_point_nearest_upper_left_corner(self):
        longitude, latitude = 0.1, 4.9
        result = self.fixture.nearest(longitude, latitude)
        expect = (np.array([0]),
                  np.array([2]))
        self.assertEqual(expect, result)

    def test_lower_left_given_upper_right_corner(self):
        longitude, latitude = [4.9], [4.9]
        result = self.fixture.lower_left(longitude, latitude)
        expect = ([1], [1])
        np.testing.assert_array_equal(expect, result)

    def test_lower_left_given_lower_left_corner(self):
        longitude, latitude = [0.1], [1.1]
        result = self.fixture.lower_left(longitude, latitude)
        expect = ([0], [1])
        np.testing.assert_array_equal(expect, result)


class TestNearestNeighbour(unittest.TestCase):
    def test_nearest(self):
        grid_longitudes = np.array([[0, 1],
                                    [0, 1]])
        grid_latitudes = np.array([[0, 0],
                                   [1, 1]])

        fixture = grid.NearestNeighbour(grid_longitudes,
                                        grid_latitudes)
        result = fixture.nearest([0.1, 0.9, 0.9],
                                 [0.1, 0.1, 0.9])
        expect = (np.array([0, 0, 1]),
                  np.array([0, 1, 1]))
        self.assertIndexEqual(expect, result)

    def assertIndexEqual(self, expect, result):
        result_i, result_j = result
        expect_i, expect_j = expect
        np.testing.assert_array_equal(expect_i, result_i)
        np.testing.assert_array_equal(expect_j, result_j)


class TestNumpy(unittest.TestCase):
    def test_specifying_points(self):
        """check that points can be specified as a list of pairs"""
        result, _ = np.array([[1, 4],
                              [2, 8],
                              [3, 12]]).T
        expect = np.array([1, 2, 3])
        np.testing.assert_array_almost_equal(expect, result)


class TestRegular2DGrid(unittest.TestCase):
    def setUp(self):
        self.longitudes = [0, 1]
        self.latitudes = [2, 3]
        self.fixture = Regular2DGrid(self.longitudes, self.latitudes)

        # Point outside grid
        self.outside_longitude = np.array([180.])
        self.outside_latitude = np.array([90.])

    def test_search_given_single_longitude_and_latitude(self):
        result = self.fixture.search(np.array([0.8]), np.array([2.2]))
        expect = SearchResult([0], [0], [0.8], [0.2])
        self.assertSearchResultEqual(expect, result)

    def test_search_given_point_outside_domain_raises_exception(self):
        with self.assertRaises(NotInGrid):
            self.fixture.search(self.outside_longitude, self.outside_latitude)

    def test_outside_given_point_outside_grid_returns_true(self):
        result = self.fixture.outside(self.outside_longitude,
                                      self.outside_latitude)
        expect = np.array([True], dtype=np.bool)
        np.testing.assert_array_equal(expect, result)

    def test_inside_given_point_outside_grid_returns_false(self):
        result = self.fixture.inside(self.outside_longitude,
                                     self.outside_latitude)
        expect = np.array([False], dtype=np.bool)
        np.testing.assert_array_equal(expect, result)

    @staticmethod
    def assertSearchResultEqual(expect, result):
        np.testing.assert_array_equal(expect.ilon, result.ilon)
        np.testing.assert_array_equal(expect.ilat, result.ilat)
        np.testing.assert_array_almost_equal(expect.dilat, result.dilat)
        np.testing.assert_array_almost_equal(expect.dilon, result.dilon)


class TestRegular1DGrid(unittest.TestCase):
    def setUp(self):
        self.values = [0., 0.11111]
        self.minimum = 0.
        self.maximum = 0.11110
        self.fixture = Regular1DGrid(self.values)

    def test_cells_given_two_vertices_returns_one(self):
        result = self.fixture.cells
        expect = 1
        self.assertEqual(expect, result)

    def test_grid_spacing_rounds_to_the_nearest_4_decimal_places(self):
        result = self.fixture.grid_spacing
        expect = 0.11110
        self.assertEqual(expect, result)

    def test_minimum_returns_first_element_of_values(self):
        result = self.fixture.minimum
        expect = self.minimum
        self.assertEqual(expect, result)

    def test_maximum_returns_last_element_of_values_plus_grid_spacing(self):
        result = self.fixture.maximum
        expect = self.maximum
        self.assertEqual(expect, result)

    def test_outside_given_point_inside_grid_returns_false(self):
        result = self.fixture.outside(0.1)
        self.assertFalse(result)

    def test_outside_given_point_less_than_minimum_returns_true(self):
        result = self.fixture.outside(self.minimum - 0.1)
        self.assertTrue(result)

    def test_outside_given_point_greater_than_maximum_returns_true(self):
        result = self.fixture.outside(self.maximum + 0.1)
        self.assertTrue(result)

    def test_outside_given_point_equal_to_maximum_returns_true(self):
        result = self.fixture.outside(self.maximum)
        self.assertTrue(result)

    def test_search_given_empty_list_returns_empty_lists(self):
        self.check_search([], ([], []))

    def test_search_given_returns_indices_with_dtype_int(self):
        result, _ = self.fixture.search(np.asarray([]))
        self.assertEqual(np.int, result.dtype)

    def test_search_given_minimum_returns_index_zero_fraction_zero(self):
        self.check_search([0.], ([0], [0.]))

    def test_search_given_maximum_returns_index_one_fraction_zero(self):
        self.check_search([0.11110], ([1], [0.]))

    def check_search(self, points, expect):
        result = self.fixture.search(np.asarray(points))
        self.assertSearchResultEqual(expect, result)

    @staticmethod
    def assertSearchResultEqual(expect, result):
        np.testing.assert_array_equal(expect[0], result[0])
        np.testing.assert_array_almost_equal(expect[1], result[1])

    def test_cell_space_given_empty_list_returns_empty_list(self):
        result = self.fixture.cell_space(np.array([]))
        expect = []
        np.testing.assert_array_equal(expect, result)

    @staticmethod
    def test_cell_space_scales_point_by_grid_spacing():
        fixture = Regular1DGrid([0., 0.1])
        result = fixture.cell_space(np.array([0.05]))
        expect = [0.5]
        np.testing.assert_array_equal(expect, result)

    @staticmethod
    def test_cell_space_shifts_point_by_minimum():
        fixture = Regular1DGrid([1., 2.])
        result = fixture.cell_space(np.array([1.5]))
        expect = [0.5]
        np.testing.assert_array_equal(expect, result)

    def test_fromcenters_minimum(self):
        fixture = Regular1DGrid.fromcenters([1., 2.])
        result = fixture.minimum
        expect = 0.5
        self.assertEqual(expect, result)

    def test_fromcenters_grid_spacing(self):
        fixture = Regular1DGrid.fromcenters([1., 2.])
        result = fixture.grid_spacing
        expect = 1.
        self.assertEqual(expect, result)
