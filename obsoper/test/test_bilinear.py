# pylint: disable=missing-docstring, invalid-name
import unittest
import numpy as np
from obsoper import bilinear


class IntegrationTests(unittest.TestCase):
    def test_interpolator_returning_ice_concentration_greater_than_one(self):
        positions = [(74.322235, 81.057388),
                     (74.339172, 81.169792),
                     (73.668488, 81.1716),
                     (73.660034, 81.059174)]
        values = [0.75734591, 0.79080015, 0.8328464, 0.80316955]
        longitude, latitude = (np.float32(74.27739716),
                               np.float32(81.11838531))
        fixture = bilinear.BilinearTransform(positions,
                                             [longitude],
                                             [latitude])
        result = fixture(values)[0]
        expect = 0.77900270456782872
        self.assertAlmostEqual(expect, result)


class TestBilinearTransform(unittest.TestCase):
    def setUp(self):
        # General quadrilateral
        self.quadrilateral_corners = [(0.4, 0),
                                      (1.0, 0),
                                      (0.9, 1),
                                      (0.0, 1)]
        self.quadrilateral_values = [1, 0, 1, 0.5]

        # Unit square
        self.unit_corners = np.array([[0, 0],
                                      [1, 0],
                                      [1, 1],
                                      [0, 1]])
        self.unit_values = np.array([1, 2, 3, 4])

        # Rectangle
        self.rectangle_corners = np.array([[10, 30],
                                           [20, 30],
                                           [20, 40],
                                           [10, 40]])
        self.rectangle_values = np.array([1, 3, 4, 2])

    def test_interpolate_returns_bottom_left_corner_values(self):
        self.check_interpolate(0.4, 0, 1)

    def test_interpolate_returns_bottom_right_corner_values(self):
        self.check_interpolate(1.0, 0, 0)

    def test_interpolate_returns_top_left_corner_values(self):
        self.check_interpolate(0.9, 1, 1)

    def test_interpolate_returns_top_right_corner_values(self):
        self.check_interpolate(0.0, 1, 0.5)

    def test_interpolate_returns_value_on_bottom_edge(self):
        self.check_interpolate(0.5, 0, 5./6.)

    def test_interpolate_returns_value_on_right_edge(self):
        self.check_interpolate(0.99, 0.1, 0.1)

    def test_interpolate_returns_value_on_top_edge(self):
        self.check_interpolate(0.8, 1, 17./18.)

    def test_interpolate_returns_value_on_left_edge(self):
        self.check_interpolate(0.1, 0.75, 0.625)

    def test_interpolate_returns_value_in_interior(self):
        self.check_interpolate(0.5, 0.5, 0.65)

    def test_interpolate_given_multiple_positions(self):
        self.check_interpolate([0.5, 0.0], [0.5, 1.0], [0.65, 0.5])

    def check_interpolate(self, x, y, expect):
        fixture = bilinear.BilinearTransform(self.quadrilateral_corners, x, y)
        result = fixture(self.quadrilateral_values)
        np.testing.assert_array_almost_equal(expect, result)

    def test_interpolate_given_general_case_near_lower_left(self):
        self.check_rectangle(11, 31, 1.3)

    def test_interpolate_given_general_case_near_upper_right(self):
        self.check_rectangle(19, 39, 3.7)

    def check_rectangle(self, longitudes, latitudes, expect):
        fixture = bilinear.BilinearTransform(self.rectangle_corners,
                                             longitudes,
                                             latitudes)
        result = fixture(self.rectangle_values)
        self.assertAlmostEqual(expect, result)

    def test_interpolate_given_unit_square(self):
        longitude, latitude = 0.5, 0.5
        fixture = bilinear.BilinearTransform(self.unit_corners,
                                             longitude,
                                             latitude)
        result = fixture(self.unit_values)
        expect = 2.5
        self.assertAlmostEqual(expect, result)


class TestInterpolate(unittest.TestCase):
    def setUp(self):
        self.values1d = [1, 2, 3, 4]
        self.values2d = [[1, 2, 3, 4],
                         [1, 2, 3, 4]]
        self.values3d = [[[1, 2, 3, 4],
                          [1, 2, 3, 4]],
                         [[1, 2, 3, 4],
                          [1, 2, 3, 4]]]
        self.weights1d = [0.25, 0.25, 0.25, 0.25]
        self.weights2d = [[0.25, 0.25, 0.25, 0.25],
                          [1, 0, 0, 0]]

        # Various configurations of 3D values
        # Depth varying
        self.values_3d_z = [[[0, 0, 0, 0],
                             [0, 0, 0, 0]],
                            [[1, 1, 1, 1],
                             [1, 1, 1, 1]],
                            [[2, 2, 2, 2],
                             [2, 2, 2, 2]]]
        # Corner varying
        self.values_3d_c = [[[1, 2, 3, 4],
                             [1, 2, 3, 4]],
                            [[1, 2, 3, 4],
                             [1, 2, 3, 4]],
                            [[1, 2, 3, 4],
                             [1, 2, 3, 4]]]
        # Observation varying
        self.values_3d_o = [[[0, 0, 0, 0],
                             [1, 1, 1, 1]],
                            [[0, 0, 0, 0],
                             [1, 1, 1, 1]],
                            [[0, 0, 0, 0],
                             [1, 1, 1, 1]]]

    def test_interpolate_given_1d_values_1d_weights(self):
        result = bilinear.interpolate(self.values1d, self.weights1d)
        expect = 2.5
        self.assertEqual(expect, result)

    def test_interpolate_given_2d_values_1d_weights(self):
        result = bilinear.interpolate(self.values2d, self.weights1d)
        expect = [2.5, 2.5]
        np.testing.assert_array_equal(expect, result)

    def test_interpolate_given_1d_values_2d_weights(self):
        result = bilinear.interpolate(self.values1d, self.weights2d)
        expect = [2.5, 1]
        np.testing.assert_array_equal(expect, result)

    def test_interpolate_given_2d_values_2d_weights(self):
        values = [[1, 2, 3, 4],
                  [9, 9, 9, 9]]
        result = bilinear.interpolate(values, self.weights2d)
        expect = [2.5, 9]
        np.testing.assert_array_equal(expect, result)

    def test_interpolate_given_3d_values_1d_weights(self):
        result = bilinear.interpolate(self.values3d, self.weights1d)
        expect = [[2.5, 2.5],
                  [2.5, 2.5]]
        np.testing.assert_array_equal(expect, result)

    def test_interpolate_given_depth_varying_3d_values_2d_weights(self):
        result = bilinear.interpolate(self.values_3d_z, self.weights2d)
        expect = [[0, 0],
                  [1, 1],
                  [2, 2]]
        np.testing.assert_array_equal(expect, result)

    def test_interpolate_given_corner_varying_3d_values_2d_weights(self):
        result = bilinear.interpolate(self.values_3d_c, self.weights2d)
        expect = [[2.5, 1],
                  [2.5, 1],
                  [2.5, 1]]
        np.testing.assert_array_equal(expect, result)

    def test_interpolate_given_obs_varying_3d_values_2d_weights(self):
        result = bilinear.interpolate(self.values_3d_o, self.weights2d)
        expect = [[0, 1],
                  [0, 1],
                  [0, 1]]
        np.testing.assert_array_equal(expect, result)

    def test_interpolate_given_depth_varying_3d_values_3d_weights(self):
        weights = [[[1, 0, 0, 0],
                    [0, 1, 0, 0]],
                   [[0, 0, 1, 0],
                    [0, 0, 0, 1]],
                   [[0, 0, 0, 0],
                    [0.25, 0.25, 0.25, 0.25]]]
        result = bilinear.interpolate(self.values_3d_z, weights)
        expect = [[0, 0],
                  [1, 1],
                  [0, 2]]
        np.testing.assert_array_equal(expect, result)

    def test_interpolate_given_masked_values(self):
        N = 10
        result = bilinear.interpolate(np.ma.masked_all((N, 4)),
                                      self.weights1d)
        expect = np.ma.masked_all(N)
        self.assertMaskedArrayEqual(expect, result)

    def assertMaskedArrayEqual(self, expect, result):
        self.assertEqual(expect.shape, result.shape)
        np.testing.assert_array_equal(expect.compressed(),
                                      result.compressed())


class TestInterpolationWeights(unittest.TestCase):
    def setUp(self):
        self.unit_square = [[0, 0],
                            [1, 0],
                            [1, 1],
                            [0, 1]]

    def test_interpolation_weights(self):
        result = bilinear.interpolation_weights(self.unit_square,
                                                [0.5],
                                                [0.5])
        expect = [[0.25, 0.25, 0.25, 0.25]]
        np.testing.assert_array_almost_equal(expect, result)

    def test_interpolation_weights_given_multiple_cells(self):
        result = bilinear.interpolation_weights([self.unit_square,
                                                 self.unit_square],
                                                [0.5],
                                                [0.5])
        expect = [[0.25, 0.25, 0.25, 0.25],
                  [0.25, 0.25, 0.25, 0.25]]
        np.testing.assert_array_almost_equal(expect, result)

    def test_interpolation_weights_given_multiple_points(self):
        result = bilinear.interpolation_weights(self.unit_square,
                                                [0.5, 0.1],
                                                [0.5, 0.1])
        expect = [[0.25, 0.25, 0.25, 0.25],
                  [0.81, 0.09, 0.01, 0.09]]
        np.testing.assert_array_almost_equal(expect, result)

    def test_interpolation_weights_given_multiple_points_and_cells(self):
        result = bilinear.interpolation_weights([self.unit_square,
                                                 self.unit_square],
                                                [0.5, 0.1],
                                                [0.5, 0.1])
        expect = [[0.25, 0.25, 0.25, 0.25],
                  [0.81, 0.09, 0.01, 0.09]]
        np.testing.assert_array_almost_equal(expect, result)

    def test_interpolation_weights_asserts_number_cells_equals_points(self):
        with self.assertRaises(ValueError):
            bilinear.interpolation_weights([self.unit_square,
                                            self.unit_square],
                                           [0.5, 0.1, 0.6],
                                           [0.5, 0.1, 0.6])


class TestBeta(unittest.TestCase):
    def setUp(self):
        # General quadrilateral
        self.quadrilateral_corners = [(0.4, 0),
                                      (1.0, 0),
                                      (0.9, 1),
                                      (0.0, 1)]

        # Unit square
        self.unit_corners = np.array([[0, 0],
                                      [1, 0],
                                      [1, 1],
                                      [0, 1]])

        # Rectangle
        self.rectangle_corners = np.array([[10, 30],
                                           [20, 30],
                                           [20, 40],
                                           [10, 40]])

        # Observed position information not needed for alpha calculation
        self.x, self.y = np.array([]), np.array([])

    def test_beta_given_unit_square(self):
        self.check_beta(self.unit_corners, [0, 0, 1, 0])

    def test_beta_given_rectangle(self):
        self.check_beta(self.rectangle_corners, [30., 0., 10., 0.])

    def test_beta_given_quadrilateral(self):
        self.check_beta(self.quadrilateral_corners, [0., 0., 1., 0.])

    def check_beta(self, corners, expect):
        transform = bilinear.BilinearTransform(corners, self.x, self.y)
        result = transform.beta
        np.testing.assert_array_almost_equal(expect, result)


class TestAlpha(unittest.TestCase):
    def setUp(self):
        # General quadrilateral
        self.quadrilateral_corners = [(0.4, 0),
                                      (1.0, 0),
                                      (0.9, 1),
                                      (0.0, 1)]

        # Unit square
        self.unit_corners = np.array([[0, 0],
                                      [1, 0],
                                      [1, 1],
                                      [0, 1]])

        # Rectangle
        self.rectangle_corners = np.array([[10, 30],
                                           [20, 30],
                                           [20, 40],
                                           [10, 40]])

        # Observed position information not needed for alpha calculation
        self.x, self.y = np.array([]), np.array([])

    def test_alpha_given_unit_square(self):
        self.check_alpha(self.unit_corners, [0, 1, 0, 0])

    def test_alpha_given_rectangle(self):
        self.check_alpha(self.rectangle_corners, [10., 10., 0., 0.])

    def test_alpha_given_quadrilateral(self):
        self.check_alpha(self.quadrilateral_corners, [0.4, 0.6, -0.4, 0.3])

    def check_alpha(self, corners, expect):
        transform = bilinear.BilinearTransform(corners, self.x, self.y)
        result = transform.alpha
        np.testing.assert_array_almost_equal(expect, result)


class TestQuadraticRoot(unittest.TestCase):
    def test_quadratic_root_given_zero_descriminant_returns_neg_b_div_2a(self):
        self.check_quadratic_root(a=1, b=2, c=1, expect=-1)

    def test_quadratic_root_given_zero_a_returns_minus_c_over_b(self):
        self.check_quadratic_root(a=0, b=2., c=1., expect=-0.5)

    def test_quadratic_root_given_vector_b_and_c(self):
        a, b, c = 1, np.array([2, 2]), np.array([1, 1])
        expect = np.array([-1, -1])
        self.check_quadratic_root(a, b, c, expect)

    def test_quadratic_root_given_vector_a_b_and_c(self):
        a, b, c = np.array([1, 1]), np.array([2, 2]), np.array([1, 1])
        expect = np.array([-1, -1])
        self.check_quadratic_root(a, b, c, expect)

    def test_quadratic_root_given_vector_a_b_and_c_with_zero_valued_a(self):
        a, b, c = np.array([1., 0.]), np.array([2., 2.]), np.array([1., 1.])
        expect = np.array([-1, -0.5])
        self.check_quadratic_root(a, b, c, expect)

    def check_quadratic_root(self, a, b, c, expect):
        result = bilinear.quadratic_root(a, b, c)
        np.testing.assert_array_almost_equal(expect, result)


class TestBilinearTransformBroadcast(unittest.TestCase):
    def setUp(self):
        # Check centers of three displaced uniform unit cells
        first_corners = np.array([(0, 0), (1, 0), (1, 1), (0, 1)])
        second_corners = np.array([(10, 0), (11, 0), (11, 1), (10, 1)])
        third_corners = np.array([(0, 10), (1, 10), (1, 11), (0, 11)])

        first_values = np.array([1, 1, 1, 1])
        second_values = np.array([2, 2, 2, 2])
        third_values = np.array([3, 3, 3, 3])

        corners = np.dstack([first_corners,
                             second_corners,
                             third_corners])

        self.values = np.dstack([first_values,
                                 second_values,
                                 third_values])[0, :, :]

        self.longitudes = np.array([0.5, 10.5, 0.5])
        self.latitudes = np.array([0.5, 0.5, 10.5])

        self.fixture = bilinear.BilinearTransform(corners,
                                                  self.longitudes,
                                                  self.latitudes)

    def test_broadcasting(self):
        """Numpy broadcasting rules should allow interpolation across cells"""
        result = self.fixture(self.values)
        expect = np.array([1., 2., 3.])
        np.testing.assert_array_almost_equal(expect, result)

    def test_to_unit_square(self):
        """Numpy broadcasting rules should allow interpolation across cells"""
        result = self.fixture.to_unit_square(self.longitudes, self.latitudes)
        expect = np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5])
        np.testing.assert_array_almost_equal(expect, result)
