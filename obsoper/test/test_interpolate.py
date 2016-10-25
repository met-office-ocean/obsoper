# pylint: disable=missing-docstring, invalid-name
import unittest
import numpy as np
from obsoper import interpolate


class TestTripolarInterpolator(unittest.TestCase):
    def setUp(self):
        self.grid_lons = np.array([[10, 10],
                                   [20, 20]])
        self.grid_lats = np.array([[30, 40],
                                   [30, 40]])
        self.field = np.array([[1, 2],
                               [3, 4]])

    def test_interpolate_given_two_points(self):
        observed_longitudes = np.array([11, 19])
        observed_latitudes = np.array([31, 39])
        interpolator = interpolate.Tripolar(self.grid_lons,
                                            self.grid_lats,
                                            observed_longitudes,
                                            observed_latitudes)
        result = interpolator.interpolate(self.field)
        expect = np.array([1.3, 3.7])
        np.testing.assert_array_almost_equal(expect, result)

    def test_interpolate_given_point_west_of_dateline(self):
        grid_lons, grid_lats = np.meshgrid([179, -179],
                                           [10, 12],
                                           indexing="ij")

        observed_lons = np.array([179.2])
        observed_lats = np.array([10.2])
        interpolator = interpolate.Tripolar(grid_lons,
                                            grid_lats,
                                            observed_lons,
                                            observed_lats)

        result = interpolator.interpolate(self.field)

        expect = np.array([1.3])
        np.testing.assert_array_almost_equal(expect, result)

    def test_interpolate_given_point_south_of_grid_returns_masked(self):
        self.check_southern_edge([0], [-80], np.ma.masked_all(1))

    def test_interpolate_given_point_on_southern_edge_of_grid(self):
        self.check_southern_edge([-10], [-70], [1])

    def test_interpolate_given_two_points_one_south_of_grid(self):
        self.check_southern_edge([0, 0], [-80, -70],
                                 np.ma.MaskedArray([100, 4], [True, False]))

    def test_interpolate_given_point_inside_cyclic_longitude_cell(self):
        grid_lons, grid_lats = np.meshgrid([70, 140, -150, -80, -10, 60],
                                           [-70, -60, -50],
                                           indexing="ij")
        lons, lats = [65], [-60]
        field = np.zeros((6, 3))
        field[[0, -1], :] = 1

        fixture = interpolate.Tripolar(grid_lons,
                                       grid_lats,
                                       lons,
                                       lats)

        result = fixture.interpolate(field)
        expect = [1]
        self.assertMaskedArrayAlmostEqual(expect, result)

    def test_interpolate_given_masked_value_returns_masked(self):
        grid_lons, grid_lats = np.meshgrid([0, 1],
                                           [0, 1],
                                           indexing="ij")

        observed_lons = np.array([0.5])
        observed_lats = np.array([0.5])
        interpolator = interpolate.Tripolar(grid_lons,
                                            grid_lats,
                                            observed_lons,
                                            observed_lats)

        field = np.ma.MaskedArray([[1, 2], [2, 1]],
                                  [[False, False], [True, False]])

        result = interpolator.interpolate(field)

        expect = np.ma.masked_all(1)
        self.assertMaskedArrayAlmostEqual(expect, result)

    def test_interpolate_given_predetermined_positions(self):
        grid_lons, grid_lats = np.meshgrid([0, 1],
                                           [0, 1],
                                           indexing="ij")
        observed_lons, observed_lats = [0.5], [0.5]
        fixture = interpolate.Tripolar(grid_lons,
                                       grid_lats,
                                       observed_lons,
                                       observed_lats)
        result = fixture.interpolate(self.field)
        expect = [2.5]
        self.assertMaskedArrayAlmostEqual(expect, result)

    def check_southern_edge(self, lons, lats, expect):
        grid_lons, grid_lats = np.meshgrid([-10, 0, 10],
                                           [-70, -60, -50],
                                           indexing="ij")
        field = np.array([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9]])
        fixture = interpolate.Tripolar(grid_lons,
                                       grid_lats,
                                       lons,
                                       lats)
        result = fixture.interpolate(field)
        self.assertMaskedArrayAlmostEqual(expect, result)

    def assertMaskedArrayAlmostEqual(self, expect, result):
        expect, result = np.ma.asarray(expect), np.ma.asarray(result)
        self.assertEqual(expect.shape,
                         result.shape)
        np.testing.assert_array_almost_equal(expect.compressed(),
                                             result.compressed())

    def test_interpolate_given_unmasked_masked_array(self):
        grid_lons, grid_lats = np.meshgrid([0, 1], [0, 1], indexing="ij")
        obs_lons, obs_lats = np.array([0.1]), np.array([0.1])
        operator = interpolate.Tripolar(grid_lons,
                                        grid_lats,
                                        obs_lons,
                                        obs_lats)
        field = np.ma.masked_array([[1, 2], [3, 4]], dtype="d")
        result = operator.interpolate(field)
        expect = [1.3]
        np.testing.assert_array_equal(expect, result)

    def test_interpolate_given_halo_flag_ignores_halo_locations(self):
        """should only consider non-halo grid cells during interpolation

        .. note:: Halo refers to extra columns East/West of the grid
                  and an extra row North of the grid.

        In the contrived example below, only longitudes [1, 2] and
        latitudes [0, 1] are considered inside the halo. Cyclic east/west
        interpolation should not see ones in halo cells.
        """
        grid_lons, grid_lats = np.meshgrid([0, 1, 2, 0, 1],
                                           [0, 1, 2], indexing="ij")
        obs_lons, obs_lats = np.array([0.5]), np.array([0.5])
        operator = interpolate.Tripolar(grid_lons,
                                        grid_lats,
                                        obs_lons,
                                        obs_lats,
                                        has_halo=True)
        field = np.ma.masked_array([[1, 1, 1],
                                    [0, 0, 1],
                                    [0, 0, 1],
                                    [0, 0, 1],
                                    [1, 1, 1]], dtype="d")
        result = operator.interpolate(field)
        expect = [0]
        np.testing.assert_array_equal(expect, result)

    def test_interpolate_given_cyclic_longitude_simple_case(self):
        """should consider [n-1, 0] cell as a continuous cell in longitude"""
        grid_lons, grid_lats = np.meshgrid([1, 2, 0],
                                           [0, 1], indexing="ij")
        obs_lons, obs_lats = np.array([0.5]), np.array([0.5])
        operator = interpolate.Tripolar(grid_lons,
                                        grid_lats,
                                        obs_lons,
                                        obs_lats)
        field = np.ma.masked_array([[1, 1],
                                    [7, 7],
                                    [3, 3]], dtype="d")
        result = operator.interpolate(field)
        expect = [2]
        np.testing.assert_array_equal(expect, result)


class TestTripolarGridDimensions(unittest.TestCase):
    def setUp(self):
        self.shape_yx = (3, 2)
        self.shape_xy = (2, 3)
        self.grid_lons = np.full(self.shape_yx, 10)
        self.grid_lats = np.full(self.shape_yx, 11)
        self.lons, self.lats = [], []
        self.fixture = interpolate.Tripolar(self.grid_lons,
                                            self.grid_lats,
                                            self.lons,
                                            self.lats,
                                            dimension_order="yx")

    def test_constructor_transposes_grid_longitudes(self):
        result = self.fixture.grid_longitudes.shape
        expect = self.shape_xy
        self.assertEqual(expect, result)

    def test_constructor_transposes_grid_latitudes(self):
        result = self.fixture.grid_latitudes.shape
        expect = self.shape_xy
        self.assertEqual(expect, result)

    def test_constructor_given_list_and_yx(self):
        grid_lons = [[0, 1],
                     [0, 1]]
        grid_lats = [[0, 0],
                     [1, 1]]
        fixture = interpolate.Tripolar(grid_lons,
                                       grid_lats,
                                       self.lons,
                                       self.lats,
                                       dimension_order="yx")
        result = fixture.grid_longitudes, fixture.grid_latitudes
        expect = (np.array(grid_lons).T,
                  np.array(grid_lats).T)
        np.testing.assert_array_equal(expect, result)


class TestTripolar3D(unittest.TestCase):
    @unittest.skip("implementing bilinear interpolator")
    def test_interpolate_3d_data(self):
        nx, ny, nz = 2, 2, 3
        no = 10
        grid_lons, grid_lats = np.meshgrid([0, 1],
                                           [0, 1],
                                           indexing="ij")
        lons, lats = np.zeros(no), np.zeros(no)
        field = np.arange(nx*ny*nz).reshape((nx, ny, nz))
        fixture = interpolate.Tripolar(grid_lons,
                                       grid_lats,
                                       lons,
                                       lats)
        result = fixture.interpolate(field)
        expect = np.zeros((no, nz))
        np.testing.assert_array_equal(expect, result)


class TestSelectCorners(unittest.TestCase):
    def setUp(self):
        self.ni = 4
        longitudes, latitudes = np.meshgrid([10, 20, 30, 40],
                                            [45, 50, 55],
                                            indexing="ij")
        self.grid = np.dstack((longitudes, latitudes))
        self.cell_00 = np.array([(10, 45),
                                 (20, 45),
                                 (20, 50),
                                 (10, 50)])
        self.cell_10 = np.array([(20, 45),
                                 (30, 45),
                                 (30, 50),
                                 (20, 50)])
        self.cell_01 = np.array([(10, 50),
                                 (20, 50),
                                 (20, 55),
                                 (10, 55)])

        # Vector fixture
        self.i = np.array([0, 1])
        self.j = np.array([0, 0])
        self.cells = np.dstack([self.cell_00,
                                self.cell_10])

    def test_select_corners_given_i_0_j_0(self):
        self.check_select_corners(i=0, j=0, expect=self.cell_00)

    def test_select_corners_given_i_1_j_0(self):
        self.check_select_corners(i=1, j=0, expect=self.cell_10)

    def test_select_corners_given_i_0_j_1(self):
        self.check_select_corners(i=0, j=1, expect=self.cell_01)

    def test_select_corners_cyclic_i_coordinate(self):
        self.check_select_corners(i=self.ni, j=0, expect=self.cell_00)

    def test_select_corners_given_array_ij(self):
        self.check_select_corners(i=self.i, j=self.j, expect=self.cells)

    def check_select_corners(self, i, j, expect):
        result = interpolate.select_corners(self.grid, i, j)
        np.testing.assert_array_almost_equal(expect, result)


class TestSelectField(unittest.TestCase):
    def setUp(self):
        self.ni = 4
        self.field = np.array([[0, 1, 2],
                               [3, 4, 5],
                               [6, 7, 8],
                               [9, 10, 11]])
        self.cell_00 = np.array([0, 3, 4, 1])
        self.cell_10 = np.array([3, 6, 7, 4])
        self.cell_01 = np.array([1, 4, 5, 2])

        # Vector fixture
        self.i = np.array([0, 1])
        self.j = np.array([0, 0])
        self.cells = np.dstack([self.cell_00,
                                self.cell_10])[0, :]

    def test_select_field_given_i_0_j_0(self):
        self.check_select_field(i=0, j=0, expect=self.cell_00)

    def test_select_field_given_i_1_j_0(self):
        self.check_select_field(i=1, j=0, expect=self.cell_10)

    def test_select_field_given_i_0_j_1(self):
        self.check_select_field(i=0, j=1, expect=self.cell_01)

    def test_select_field_cyclic_i_coordinate(self):
        self.check_select_field(i=self.ni, j=0, expect=self.cell_00)

    def test_select_field_given_array_ij(self):
        self.check_select_field(i=self.i, j=self.j, expect=self.cells)

    def check_select_field(self, i, j, expect):
        result = interpolate.select_field(self.field, i, j)
        np.testing.assert_array_almost_equal(expect, result)

    def test_self_cells(self):
        """Assert test fixture shape (4, N)"""
        result = self.cells.shape
        expect = (4, 2)
        self.assertEqual(expect, result)


class TestCorrectCorners(unittest.TestCase):
    def setUp(self):
        self.dateline_corners = [(+179, 0),
                                 (-179, 0),
                                 (-179, 1),
                                 (+179, 1)]
        self.ordinary_corners = [(0, 0),
                                 (1, 0),
                                 (1, 1),
                                 (0, 1)]
        self.east_adjusted = [(-181, 0),
                              (-179, 0),
                              (-179, 1),
                              (-181, 1)]
        self.west_adjusted = [(+179, 0),
                              (+181, 0),
                              (+181, 1),
                              (+179, 1)]
        self.eastern_longitude = -179
        self.western_longitude = +179

        # Many cells fixture
        self.many_dateline_cells = np.dstack([self.dateline_corners,
                                              self.dateline_corners,
                                              self.dateline_corners])
        self.many_ordinary_cells = np.dstack([self.ordinary_corners,
                                              self.ordinary_corners,
                                              self.ordinary_corners])
        self.many_longitudes = [self.eastern_longitude,
                                self.western_longitude,
                                self.eastern_longitude]
        self.many_adjusted_cells = np.dstack([self.east_adjusted,
                                              self.west_adjusted,
                                              self.east_adjusted])

    def test_correct_corners_given_eastern_longitude(self):
        self.check_correct_corners(self.dateline_corners,
                                   self.eastern_longitude,
                                   self.east_adjusted)

    def test_correct_corners_given_western_longitude(self):
        self.check_correct_corners(self.dateline_corners,
                                   self.western_longitude,
                                   self.west_adjusted)

    def test_correct_corners_given_ordinary_cell_returns_ordinary_cell(self):
        self.check_correct_corners(self.ordinary_corners,
                                   self.eastern_longitude,
                                   self.ordinary_corners)

    def test_correct_corners_given_multiple_longitudes(self):
        self.check_correct_corners(self.many_dateline_cells,
                                   self.many_longitudes,
                                   self.many_adjusted_cells)

    def test_correct_corners_given_multiple_ordinary_cells(self):
        self.check_correct_corners(self.many_ordinary_cells,
                                   self.many_longitudes,
                                   self.many_ordinary_cells)

    def check_correct_corners(self, vertices, longitudes, expect):
        result = interpolate.correct_corners(vertices, longitudes)
        np.testing.assert_array_almost_equal(expect, result)


class TestIsDateline(unittest.TestCase):
    def setUp(self):
        self.dateline_corners = [(+179, 0),
                                 (-179, 0),
                                 (-179, 1),
                                 (+179, 1)]
        self.ordinary_corners = [(0, 0),
                                 (1, 0),
                                 (1, 1),
                                 (0, 1)]
        self.sequence = np.dstack([self.ordinary_corners,
                                   self.dateline_corners,
                                   self.ordinary_corners])

    def test_is_dateline_given_dateline_corners_returns_true(self):
        self.check_is_dateline(self.dateline_corners, True)

    def test_is_dateline_given_ordinary_corners_returns_false(self):
        self.check_is_dateline(self.ordinary_corners, False)

    def test_is_dateline_given_sequence_returns_boolean_array(self):
        self.check_is_dateline(self.sequence, [False, True, False])

    def check_is_dateline(self, corners, expect):
        result = interpolate.is_dateline(corners)
        np.testing.assert_array_almost_equal(expect, result)

    def test_self_sequence_shape(self):
        """Assert test fixture shape (4, 2, N)"""
        result = self.sequence.shape
        expect = (4, 2, 3)
        self.assertEqual(expect, result)


class TestIsEast(unittest.TestCase):
    def test_is_east_given_east_longitude_returns_true(self):
        self.check_is_east(-1., True)

    def test_is_east_given_west_longitude_returns_false(self):
        self.check_is_east(+1., False)

    def test_is_east_given_greenwich_meridian_returns_true(self):
        self.check_is_east(0., True)

    def test_is_east_given_multiple_values_returns_boolean_array(self):
        self.check_is_east([-1, 0, 1], [True, True, False])

    def check_is_east(self, longitudes, expect):
        result = interpolate.is_east(longitudes)
        np.testing.assert_array_almost_equal(expect, result)


class TestIsWest(unittest.TestCase):
    def test_is_west_given_west_longitude_returns_false(self):
        self.check_is_west(-1., False)

    def test_is_west_given_west_longitude_returns_true(self):
        self.check_is_west(+1., True)

    def test_is_west_given_greenwich_meridian_returns_false(self):
        self.check_is_west(0., False)

    def test_is_west_given_multiple_values_returns_boolean_array(self):
        self.check_is_west([-1, 0, 1], [False, False, True])

    def check_is_west(self, longitudes, expect):
        result = interpolate.is_west(longitudes)
        np.testing.assert_array_almost_equal(expect, result)


@unittest.skip("major refactoring")
class TestInterpolateDateline(unittest.TestCase):
    def setUp(self):
        self.corners = np.array([(179, 10),
                                 (-179, 10),
                                 (-179, 20),
                                 (179, 20)])
        self.corner_values = [1, 2, 3, 4]

    def test_interpolate_points_inside_cell_west_of_180th_meridian(self):
        self.check_dateline_interpolate(longitudes=[179.5],
                                        latitudes=[10],
                                        expect=[1.25])

    def test_interpolate_points_inside_cell_east_of_180th_meridian(self):
        self.check_dateline_interpolate(longitudes=[-179.5],
                                        latitudes=[10],
                                        expect=[1.75])

    def test_interpolate_points_inside_cell_east_and_west(self):
        self.check_dateline_interpolate(longitudes=[-179.5, 179.5],
                                        latitudes=[10, 10],
                                        expect=[1.75, 1.25])

    def check_dateline_interpolate(self, longitudes, latitudes, expect):
        result = interpolate.dateline_interpolate(self.corners,
                                                  self.corner_values,
                                                  longitudes,
                                                  latitudes)
        self.assertMaskedArrayAlmostEqual(expect, result)

    def assertMaskedArrayAlmostEqual(self, expect, result):
        expect, result = np.ma.asarray(expect), np.ma.asarray(result)
        self.assertEqual(expect.shape,
                         result.shape)
        np.testing.assert_array_almost_equal(expect.compressed(),
                                             result.compressed())
