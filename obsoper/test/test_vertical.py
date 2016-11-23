# pylint: disable=missing-docstring, invalid-name
import unittest
import numpy as np
from obsoper import vertical
from obsoper.vertical import (Vertical2DInterpolator,
                              Vertical1DInterpolator)


class TestSection(unittest.TestCase):
    def test_repr(self):
        fixture = vertical.Section([], [])
        result = repr(fixture)
        expect = "Section(values=[],\ndepths=[])"
        self.assertEqual(expect, result)

    def test_construct_with_lists_returns_depths(self):
        fixture = vertical.Section([], [])
        result = fixture.depths
        expect = []
        np.testing.assert_array_equal(expect, result)

    def test_construct_with_lists_returns_values(self):
        fixture = vertical.Section([], [])
        result = fixture.values
        expect = []
        np.testing.assert_array_equal(expect, result)

    def test_construct_given_incompatible_depths_raises_assertion_error(self):
        with self.assertRaises(AssertionError):
            vertical.Section([], [[0, 0]])


class TestColumn(unittest.TestCase):
    def setUp(self):
        self.values = [0, 100]
        self.depths = [0, 1]
        self.fixture = vertical.Column(self.values, self.depths)

    def test_interpolate(self):
        result = self.fixture.interpolate(0.9)
        expect = 90.
        np.testing.assert_array_equal(expect, result)


class TestVertical1DInterpolator(unittest.TestCase):
    def setUp(self):
        # Depths
        self.masked_depths = np.ma.masked_all(5)
        self.partial_depths = np.ma.masked_array([100, 200, 300, 400, 500],
                                                 [False, True, False, False,
                                                  False])
        self.linear_depths = np.array([100, 200, 300, 400, 500])

        # Fields
        self.linear_field = np.array([10, 20, 30, 40, 50])
        self.partial_field = np.ma.masked_array([10, 20, 30, 40, 50],
                                                [False, True, False, False,
                                                 False])

        self.fixture = Vertical1DInterpolator(self.linear_depths,
                                              self.linear_field)
        self.sample_depth = np.array([150])
        self.sample_value = np.array([15])

    def test_constructor_raises_exception_if_zero_sized_array_given(self):
        with self.assertRaises(AssertionError):
            vertical.Vertical1DInterpolator([], [])

    def test_given_linear_function(self):
        result = self.fixture(self.sample_depth)
        expect = self.sample_value
        np.testing.assert_array_almost_equal(expect, result)

    def test_constructor_given_3_knots_reverts_to_linear_interpolation(self):
        three_depths = np.array([10, 20, 30])
        three_values = np.array([1, 2, 3])
        fixture = Vertical1DInterpolator(three_depths, three_values)
        result = fixture([15])
        expect = np.array([1.5])
        np.testing.assert_array_almost_equal(expect, result)

    def test_interpolate_all_masked_depths_returns_all_masked(self):
        fixture = Vertical1DInterpolator(self.masked_depths,
                                         self.linear_field)
        result = fixture(self.sample_depth)
        expect = np.ma.masked_all(1)
        self.assertMaskedArrayEqual(expect, result)

    def test_knots_given_partially_masked_depths_returns_correct_count(self):
        result = self.fixture.knots(self.partial_depths)
        expect = 4
        self.assertEqual(expect, result)

    def test_interpolate_given_partially_masked_depths(self):
        fixture = Vertical1DInterpolator(self.partial_depths,
                                         self.linear_field)
        result = fixture(self.sample_depth)
        expect = self.sample_value
        np.testing.assert_array_almost_equal(expect, result)

    def test_interpolate_given_masked_observed_depth_returns_masked(self):
        result = self.fixture(np.ma.masked_all(1))
        expect = np.ma.masked_all(1)
        self.assertMaskedArrayEqual(expect, result)

    def test_interpolate_preserves_masked_observed_depths(self):
        observed_depths = np.ma.masked_array([0., 150.], mask=[True, False])
        result = self.fixture(observed_depths)
        expect = np.ma.masked_array([9999., 15.], mask=[True, False])
        self.assertMaskedArrayEqual(expect, result)

    def assertMaskedArrayEqual(self, expect, result):
        self.assertEqual(expect.shape, result.shape)
        np.testing.assert_array_equal(expect.mask, result.mask)
        np.testing.assert_array_almost_equal(expect.compressed(),
                                             result.compressed())


class TestMissingDataInterpolation(unittest.TestCase):
    def setUp(self):
        self.x_knots = np.array([0., 1., 2., 3., 4.])
        self.y_knots = np.array([0., 0.5, 1., 0.5, 0.])
        self.mask = np.array([False, False, True, False, False], dtype=np.bool)
        self.point = np.array([2.5])
        self.tent_value = np.array([0.84375])
        self.trapezoid_value = np.array([0.625])

    def test_interpolate_given_clean_data_returns_tent_value(self):
        fixture = Vertical1DInterpolator(self.x_knots, self.y_knots)
        result = fixture(self.point)
        expect = self.tent_value
        np.testing.assert_array_almost_equal(expect, result)

    def test_interpolate_given_masked_x_returns_trapezoid_value(self):
        x = np.ma.masked_array(self.x_knots, self.mask)
        fixture = Vertical1DInterpolator(x, self.y_knots)
        result = fixture(self.point)
        expect = self.trapezoid_value
        np.testing.assert_array_almost_equal(expect, result)

    def test_interpolate_given_masked_y_returns_trapezoid_value(self):
        y = np.ma.masked_array(self.y_knots, self.mask)
        fixture = Vertical1DInterpolator(self.x_knots, y)
        result = fixture(self.point)
        expect = self.trapezoid_value
        np.testing.assert_array_almost_equal(expect, result)


class TestInterpolationBounds(unittest.TestCase):
    def setUp(self):
        x = np.array([0., 1., 2., 3., 4.])
        y = np.array([0., 0.5, 1., 0.5, 0.])
        self.fixture = Vertical1DInterpolator(x, y)
        self.point_outside = np.array([5.])

    def test_given_observed_depth_outside_of_model_grid_returns_masked(self):
        result = self.fixture(self.point_outside)
        self.assertAllMasked(result)

    def assertAllMasked(self, result):
        self.assertTrue(result.mask.all())


class TestVertical2DInterpolator(unittest.TestCase):
    def setUp(self):
        depths = np.array([[0, 1, 2, 3, 4],
                           [0, 1, 2, 3, 4]])
        self.field = np.array([[0, 10, 20, 30, 40],
                               [0, 10, 20, 30, 40]])
        self.fixture = Vertical2DInterpolator(depths, self.field)
        self.observed_depths = np.array([[1.5],
                                         [1.5]])

    def test_multiple_water_columns(self):
        result = self.fixture(self.observed_depths)
        expect = np.array([[15],
                           [15]])
        np.testing.assert_array_almost_equal(expect, result)

    def test_call_given_s_levels_returns_smaller_value(self):
        s_levels = np.array([[0, 1, 2, 3, 4],
                             [0, 1.2, 2.4, 3.6, 4.8]])
        fixture = Vertical2DInterpolator(s_levels, self.field)
        result = fixture(self.observed_depths)
        expect = np.array([[15],
                           [12.5]])
        np.testing.assert_array_almost_equal(expect, result)

    def test_call_given_z_levels_reuses_levels(self):
        z_levels = np.array([0, 1, 2, 3, 4])
        fixture = Vertical2DInterpolator(z_levels, self.field)
        result = fixture(self.observed_depths)
        expect = np.array([[15],
                           [15]])
        np.testing.assert_array_almost_equal(expect, result)
