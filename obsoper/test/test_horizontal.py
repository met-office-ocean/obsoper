# pylint: disable=missing-docstring, invalid-name
import unittest
import numpy as np
from obsoper.horizontal import UnitSquare
from obsoper import horizontal


class TestHorizontal(unittest.TestCase):
    def setUp(self):
        # Trapezoid 2x2 grid represents most of the issues encountered with
        # rotated grids
        self.grid_lons = [[0, 1],
                          [3, 2]]
        self.grid_lats = [[0, 1],
                          [0, 1]]
        self.field = [[1, 2],
                      [3, 4]]
        self.obs_lons = [1.5]
        self.obs_lats = [0.5]
        self.fixture = horizontal.Horizontal(self.grid_lons,
                                             self.grid_lats,
                                             self.obs_lons,
                                             self.obs_lats)

    def test_interpolate(self):
        result = self.fixture.interpolate(self.field)
        expect = 2.5
        self.assertEqual(expect, result)


class TestTripolar(unittest.TestCase):
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
        interpolator = horizontal.Tripolar(self.grid_lons,
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
        interpolator = horizontal.Tripolar(grid_lons,
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

    def test_interpolate_given_point_north_of_grid_returns_masked(self):
        self.check_southern_edge([0], [-49], np.ma.masked_all(1))

    def test_interpolate_given_point_inside_cyclic_longitude_cell(self):
        grid_lons, grid_lats = np.meshgrid([70, 140, -150, -80, -10, 60],
                                           [-70, -60, -50],
                                           indexing="ij")
        lons, lats = [65], [-60]
        field = np.zeros((6, 3))
        field[[0, -1], :] = 1

        fixture = horizontal.Tripolar(grid_lons,
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
        interpolator = horizontal.Tripolar(grid_lons,
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
        fixture = horizontal.Tripolar(grid_lons,
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
        fixture = horizontal.Tripolar(grid_lons,
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
        operator = horizontal.Tripolar(grid_lons,
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
        operator = horizontal.Tripolar(grid_lons,
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
        operator = horizontal.Tripolar(grid_lons,
                                       grid_lats,
                                       obs_lons,
                                       obs_lats)
        field = np.ma.masked_array([[1, 1],
                                    [7, 7],
                                    [3, 3]], dtype="d")
        result = operator.interpolate(field)
        expect = [2]
        np.testing.assert_array_equal(expect, result)


class TestTripolar3D(unittest.TestCase):
    def test_interpolate_3d_data(self):
        no = 5
        grid_lons, grid_lats = np.meshgrid([0, 1],
                                           [0, 1],
                                           indexing="ij")
        lons, lats = np.zeros(no), np.zeros(no)
        field = np.array([[[0, 1, 2],
                           [0, 1, 2]],
                          [[0, 1, 2],
                           [0, 1, 2]]])
        fixture = horizontal.Tripolar(grid_lons,
                                      grid_lats,
                                      lons,
                                      lats)
        result = fixture.interpolate(field)
        expect = np.array([[0, 1, 2],
                           [0, 1, 2],
                           [0, 1, 2],
                           [0, 1, 2],
                           [0, 1, 2]])
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
        self.i = np.array([0, 1, 0])
        self.j = np.array([0, 0, 1])
        self.cells = np.transpose(np.array([self.cell_00,
                                            self.cell_10,
                                            self.cell_01]),
                                  (1, 2, 0))

    def test_grid_shape(self):
        x, y, dims = 4, 3, 2
        self.assertEqual((x, y, dims), self.grid.shape)

    def test_cells_shape(self):
        corners, cells, dims = 4, 2, 3
        self.assertEqual((corners, cells, dims), self.cells.shape)

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
        result = horizontal.select_corners(self.grid, i, j)
        np.testing.assert_array_almost_equal(expect, result)


class TestSelectCornersShape(unittest.TestCase):
    def setUp(self):
        self.nz = 5
        self.surface = np.ones((2, 2))
        self.full_depths = np.ones((2, 2, self.nz))

    def test_select_corners_given_surface_and_scalar_returns_1d_shape(self):
        self.check_shape(self.surface, 0, 0, (4,))

    def test_select_corners_given_surface_and_vector_returns_2d_shape(self):
        self.check_shape(self.surface, [0], [0], (4, 1))

    def test_select_corners_given_full_depth_and_scalar_returns_2d_shape(self):
        self.check_shape(self.full_depths, 0, 0, (4, self.nz))

    def test_select_corners_given_full_depth_and_vector_returns_3d_shape(self):
        self.check_shape(self.full_depths, [0], [0], (4, self.nz, 1))

    def check_shape(self, values, i, j, expect):
        result = horizontal.select_corners(values, i, j).shape
        self.assertEqual(expect, result)


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
        result = horizontal.select_field(self.field, i, j)
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
        result = horizontal.correct_corners(vertices, longitudes)
        np.testing.assert_array_almost_equal(expect, result)


class TestMaskCorners(unittest.TestCase):
    def setUp(self):
        self.ndarray = np.arange(4)
        self.all_masked = np.ma.masked_all(4)
        self.single_masked = np.ma.masked_array([1, 2, 3, 4],
                                                mask=[True, False, False, False])

        # 2D array (N, 4)
        values = [[1, 2, 3, 4],
                  [5, 6, 7, 8]]
        mask_before = [[True, False, False, False],
                       [False, False, False, False]]
        mask_after = [[True, True, True, True],
                      [False, False, False, False]]
        self.mask2d = np.ma.masked_array(values, mask_before)
        self.mask2dfill = np.ma.masked_array(values, mask_after)

        # 3D array (Z, N, 4)
        values = [[[1, 2, 3, 4],
                   [5, 6, 7, 8]],
                  [[1, 2, 3, 4],
                   [5, 6, 7, 8]]]
        mask_before = [[[True, False, False, False],
                        [False, False, False, False]],
                       [[False, False, False, False],
                        [True, False, False, False]]]
        mask_after = [[[True, True, True, True],
                       [False, False, False, False]],
                      [[False, False, False, False],
                       [True, True, True, True]]]
        self.mask3d = np.ma.masked_array(values, mask_before)
        self.mask3dfill = np.ma.masked_array(values, mask_after)

    def test_mask_corners_given_ndarray_returns_ndarray(self):
        result = horizontal.mask_corners(self.ndarray)
        expect = self.ndarray
        np.testing.assert_array_equal(expect, result)

    def test_mask_corners_given_all_masked_returns_all_masked(self):
        self.check_mask_corners(self.all_masked, self.all_masked)

    def test_mask_corners_given_single_masked_value_returns_all_masked(self):
        self.check_mask_corners(self.single_masked, self.all_masked)

    def test_mask_corners_given_masked_2d_returns_masks_axis_zero(self):
        self.check_mask_corners(self.mask2d, self.mask2dfill)

    def test_mask_corners_given_masked_3d_returns_masks_axis_zero(self):
        self.check_mask_corners(self.mask3d, self.mask3dfill)

    def check_mask_corners(self, given, expect):
        result = horizontal.mask_corners(given)
        self.assertMaskedArrayEqual(expect, result)

    def assertMaskedArrayEqual(self, expect, result):
        self.assertEqual(expect.shape, result.shape)
        np.testing.assert_array_equal(expect.compressed(),
                                      result.compressed())


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
        result = horizontal.is_dateline(corners)
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
        result = horizontal.is_east(longitudes)
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
        result = horizontal.is_west(longitudes)
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
        result = horizontal.dateline_interpolate(self.corners,
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


class TestUnitSquare(unittest.TestCase):
    def setUp(self):
        self.empty = []
        self.data = np.array([1, 2, 3, 4]).reshape(2, 2)

    def test_weights_given_lower_left_corner(self):
        self.check_weights([0], [0], [1, 0, 0, 0])

    def test_weights_given_lower_right_corner(self):
        self.check_weights([0], [1], [0, 1, 0, 0])

    def test_weights_given_upper_left_corner(self):
        self.check_weights([1], [0], [0, 0, 1, 0])

    def test_weights_given_upper_right_corner(self):
        self.check_weights([1], [1], [0, 0, 0, 1])

    def check_weights(self, x, y, expect):
        fixture = UnitSquare(self.empty, self.empty,
                             np.array(x), np.array(y))
        result = fixture.weights
        expect = np.array(expect).reshape(4, 1)
        np.testing.assert_array_equal(expect, result)

    def test_weights_given_n_fractions_returns_4_by_n_shape(self):
        x, y = np.array([0, 1]), np.array([0, 1])
        fixture = UnitSquare(self.empty, self.empty, x, y)
        result = fixture.weights.shape
        expect = (4, 2)
        self.assertEqual(expect, result)

    def test_values_given_index_and_data(self):
        i, j = np.array([0]), np.array([0])
        fixture = UnitSquare(i, j, self.empty, self.empty)
        data = np.array([1, 2, 3, 4]).reshape(2, 2)
        result = fixture.values(data)
        expect = np.array([1, 2, 3, 4]).reshape(4, 1)
        np.testing.assert_array_equal(expect, result)

    def test_interpolation_given_lower_left_corner(self):
        self.check_interpolation([0], [0], [1])

    def test_interpolation_given_lower_right_corner(self):
        self.check_interpolation([0], [1], [2])

    def test_interpolation_given_upper_left_corner(self):
        self.check_interpolation([1], [0], [3])

    def test_interpolation_given_upper_right_corner(self):
        self.check_interpolation([1], [1], [4])

    @staticmethod
    def check_interpolation(x, y, expect):
        x, y = np.array(x), np.array(y)
        i, j = np.array([0]), np.array([0])
        fixture = UnitSquare(i, j, x, y)
        data = np.array([1, 2, 3, 4]).reshape(2, 2)
        result = fixture(data)
        np.testing.assert_array_equal(expect, result)

    def test_masked_given_upper_left_masked_returns_true(self):
        i, j = np.array([0]), np.array([0])
        fixture = UnitSquare(i, j, self.empty, self.empty)
        data = np.ma.masked_array([[1, 2], [3, 4]])
        data[1, 0] = np.ma.masked
        result = fixture.masked(data)
        expect = np.array([True])
        np.testing.assert_array_equal(expect, result)

    def test_interpolation_given_3d_data(self):
        data = np.zeros((3, 3, 2))
        data[..., 0] = 1
        data[..., 1] = 2
        x, y = np.array([0]), np.array([0])
        i, j = np.array([1]), np.array([1])
        fixture = UnitSquare(i, j, x, y)
        result = fixture(data)
        expect = np.array([1, 2]).reshape(1, 2)
        np.testing.assert_array_equal(expect, result)


class TestRegular(unittest.TestCase):
    def setUp(self):
        self.grid_longitudes = np.arange(3)
        self.grid_latitudes = np.arange(3)
        self.observed_longitudes = np.array([0.9, 1.9])
        self.observed_latitudes = np.array([1.5, 0.1])
        self.fixture = horizontal.Regular(self.grid_longitudes,
                                          self.grid_latitudes,
                                          self.observed_longitudes,
                                          self.observed_latitudes)
        self.field = np.array([[1, 2, 3],
                               [4, 5, 6],
                               [7, 8, 9]])
        self.counterparts = np.array([5.2, 6.8])

    def test_constructor_given_lists(self):
        horizontal.Regular([], [], [], [])

    def test_interpolate_given_2d_grid_lon_lat_arrays(self):
        grid_longitudes = np.array([[10, 10], [20, 20]])
        grid_latitudes = np.array([[30, 40], [30, 40]])
        field = np.array([[0, 0], [1, 1]])
        fixture = horizontal.Regular(grid_longitudes,
                                     grid_latitudes,
                                     [11],
                                     [31])
        result = fixture.interpolate(field)
        expect = [0.1]
        np.testing.assert_array_equal(expect, result)

    def test_interpolate(self):
        result = self.fixture.interpolate(self.field)
        expect = self.counterparts
        np.testing.assert_array_equal(expect, result)
