# pylint: disable=missing-docstring, invalid-name
import unittest
import numpy as np
import mock
from obsoper import (exceptions,
                     grid)


class TestLowerLeft(unittest.TestCase):
    def setUp(self):
        self.grid_lons = [[0, 0],
                          [1, 1]]
        self.grid_lats = [[0, 1],
                          [0, 1]]
        self.obs_lons = [0.1]
        self.obs_lats = [0.1]

    def test_lower_left_given_cartesian_search(self):
        self.check_lower_left("cartesian", ([0], [0]))

    def test_lower_left_given_tripolar_search(self):
        self.check_lower_left("tripolar", ([0], [0]))

    def check_lower_left(self, search, expect):
        result = grid.lower_left(self.grid_lons,
                                 self.grid_lats,
                                 self.obs_lons,
                                 self.obs_lats,
                                 search=search)
        np.testing.assert_array_equal(expect, result)


class TestSearch(unittest.TestCase):
    @staticmethod
    def test_deprecate():
        with mock.patch("obsoper.grid.warnings") as mock_warnings:
            grid.Search(np.ones((2, 2)),
                        np.ones((2, 2)))
            mock_warnings.warn.assert_called_once()


class TestTripolarSearch(unittest.TestCase):
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
        self.fixture = grid.TripolarSearch(self.longitudes, self.latitudes)

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


class TestCartesianSearch(unittest.TestCase):
    def setUp(self):
        self.dateline_cell = self.make_cartesian_search([100, -179], [50, 70])
        self.two_by_two = self.make_cartesian_search([0, 1, 2], [0, 1, 2])

    @staticmethod
    def make_cartesian_search(lons, lats):
        grid_lons, grid_lats = np.meshgrid(lons, lats, indexing="ij")
        return grid.CartesianSearch(grid_lons, grid_lats)

    def test_lower_left_given_points_in_dateline_cell(self):
        self.check_lower_left(self.dateline_cell,
                              [140, 160, -179.5], [50, 60, 70],
                              ([0, 0, 0], [0, 0, 0]))

    def test_lower_left_given_point_near_upper_right_corner_in_two_by_two(self):
        self.check_lower_left(self.two_by_two,
                              [1.9], [1.9],
                              ([1], [1]))

    @staticmethod
    def check_lower_left(search,
                         lons,
                         lats,
                         expect):
        result = search.lower_left(lons, lats)
        np.testing.assert_array_equal(expect, result)

    def test_number_of_neighbours_given_two_by_two_returns_4(self):
        self.check_number((2, 2), 4)

    def test_number_of_neighbours_given_3_by_3_returns_8(self):
        self.check_number((3, 3), 8)

    def check_number(self, shape, expect):
        fixture = grid.CartesianSearch(np.zeros(shape), np.zeros(shape))
        result = fixture.k
        self.assertEqual(expect, result)

    def test_lower_left_raises_exception_if_search_fails(self):
        """should report algorithm failure to user"""
        with self.assertRaises(exceptions.SearchFailed):
            grid.CartesianSearch(np.zeros((2, 2)),
                                 np.zeros((2, 2))).lower_left([1], [1])


class TestLonLatNeighbour(unittest.TestCase):
    def test_nearest(self):
        grid_longitudes = np.array([[0, 1],
                                    [0, 1]])
        grid_latitudes = np.array([[0, 0],
                                   [1, 1]])

        fixture = grid.LonLatNeighbour(grid_longitudes,
                                       grid_latitudes)
        result = fixture.nearest([0.1, 0.9, 0.9],
                                 [0.1, 0.1, 0.9])
        expect = (np.array([0, 0, 1]),
                  np.array([0, 1, 1]))
        self.assertIndexEqual(expect, result)

    @staticmethod
    def assertIndexEqual(expect, result):
        result_i, result_j = result
        expect_i, expect_j = expect
        np.testing.assert_array_equal(expect_i, result_i)
        np.testing.assert_array_equal(expect_j, result_j)


class TestNearestNeighbour(unittest.TestCase):
    @staticmethod
    def test_deprecated():
        with mock.patch("obsoper.grid.warnings") as mock_warnings:
            grid.NearestNeighbour(np.ones((2, 2)),
                                  np.ones((2, 2)))
            mock_warnings.warn.assert_called_once()


class TestCartesianNeighbour(unittest.TestCase):
    def setUp(self):
        grid_longitudes, grid_latitudes = np.meshgrid([10, 20],
                                                      [50, 70],
                                                      indexing="ij")
        self.fixture = grid.CartesianNeighbour(grid_longitudes,
                                               grid_latitudes)

    def test_nearest_given_lower_left_corner(self):
        self.check_nearest(10, 50, ([0], [0]))

    def test_nearest_given_lower_right_corner(self):
        self.check_nearest(20, 50, ([1], [0]))

    def test_nearest_given_upper_left_corner(self):
        self.check_nearest(10, 70, ([0], [1]))

    def test_nearest_given_upper_right_corner(self):
        self.check_nearest(20, 70, ([1], [1]))

    def test_nearest_given_point_on_other_side_of_180th_meridian(self):
        grid_longitudes, grid_latitudes = np.meshgrid([100, -179],
                                                      [50, 70],
                                                      indexing="ij")
        fixture = grid.CartesianNeighbour(grid_longitudes, grid_latitudes)
        result = fixture.nearest(179, 70)
        expect = ([1], [1])
        np.testing.assert_array_equal(expect, result)

    def test_nearest_given_multiple_nearest_neighbours(self):
        grid_longitudes, grid_latitudes = np.meshgrid([100, -179],
                                                      [50, 70],
                                                      indexing="ij")
        fixture = grid.CartesianNeighbour(grid_longitudes, grid_latitudes)
        result = fixture.nearest(179, 70, k=4)
        expect = ([[1, 1, 0, 0]], [[1, 0, 1, 0]])
        np.testing.assert_array_equal(expect, result)

    def check_nearest(self, longitudes, latitudes, expect):
        result = self.fixture.nearest(longitudes, latitudes)
        np.testing.assert_array_equal(expect, result)


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
        self.fixture = grid.Regular2DGrid(self.longitudes, self.latitudes)

        # Point outside grid
        self.outside_longitude = np.array([180.])
        self.outside_latitude = np.array([90.])

    def test_search_given_single_longitude_and_latitude(self):
        result = self.fixture.search(np.array([0.8]), np.array([2.2]))
        expect = grid.SearchResult([0], [0], [0.8], [0.2])
        self.assertSearchResultEqual(expect, result)

    def test_search_given_point_outside_domain_raises_exception(self):
        with self.assertRaises(exceptions.NotInGrid):
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
        self.fixture = grid.Regular1DGrid(self.values)

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

    def test_inside_given_point_outside_returns_false(self):
        result = self.fixture.inside(self.minimum - 0.1)
        self.assertEqual(result, False)

    def test_inside_given_point_inside_returns_true(self):
        result = self.fixture.inside(self.maximum - 0.1)
        self.assertEqual(result, True)

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
        self.check_cell_space([0., 0.11111], [], [])

    def test_cell_space_scales_point_by_grid_spacing(self):
        self.check_cell_space([0., 0.1], [0.05], [0.5])

    def test_cell_space_shifts_point_by_minimum(self):
        self.check_cell_space([1., 2.], [1.5], [0.5])

    def test_cell_space_given_descending_grid(self):
        self.check_cell_space([1., 0.], [0.1], [0.9])

    @staticmethod
    def check_cell_space(vertices, points, expect):
        fixture = grid.Regular1DGrid(vertices)
        result = fixture.cell_space(np.array(points))
        np.testing.assert_array_equal(expect, result)

    def test_fromcenters_minimum(self):
        fixture = grid.Regular1DGrid.fromcenters([1., 2.])
        result = fixture.minimum
        expect = 0.5
        self.assertEqual(expect, result)

    def test_fromcenters_grid_spacing(self):
        fixture = grid.Regular1DGrid.fromcenters([1., 2.])
        result = fixture.grid_spacing
        expect = 1.
        self.assertEqual(expect, result)

    def test_inside_given_point_inside_descending_grid_returns_true(self):
        self.check_inside([1., 0.], [0.5], [True])

    def test_inside_given_point_left_of_descending_grid_returns_false(self):
        self.check_inside([1., 0.], [1.1], [False])

    def test_inside_given_point_right_of_descending_grid_returns_false(self):
        self.check_inside([1., 0.], [-0.1], [False])

    def check_inside(self, vertices, points, expect):
        fixture = grid.Regular1DGrid(vertices)
        result = fixture.inside(points)
        np.testing.assert_array_equal(expect, result)

    def test_spacing_given_descending_coordinate(self):
        self.check_grid_spacing([1., 0.], -1)

    def check_grid_spacing(self, vertices, expect):
        fixture = grid.Regular1DGrid(vertices)
        result = fixture.grid_spacing
        self.assertEqual(expect, result)
