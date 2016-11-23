# pylint: disable=missing-docstring, invalid-name
import unittest
import numpy as np
import obsoper
from obsoper import (exceptions,
                     ObservationOperator)


class TestOperator(unittest.TestCase):
    def setUp(self):
        # Model grid
        self.grid_lons = [[0, 1],
                          [3, 2]]
        self.grid_lats = [[0, 1],
                          [0, 1]]
        self.grid_depths = None

        # Observation positions
        self.obs_lons = [1.5]
        self.obs_lats = [0.5]
        self.obs_depths = None

        # Forecasts/results
        self.analysis = [[1, 2],
                         [3, 4]]
        self.counterparts = [2.5]

    def test_interpolate_given_tripolar_grid(self):
        fixture = obsoper.Operator(self.grid_lons,
                                   self.grid_lats,
                                   self.obs_lons,
                                   self.obs_lats,
                                   layout="tripolar")
        result = fixture.interpolate(self.analysis)
        expect = self.counterparts
        np.testing.assert_array_equal(expect, result)

    def test_interpolate_given_regular_grid(self):
        fixture = obsoper.Operator([100, 200],
                                   [10, 20],
                                   [120],
                                   [12],
                                   layout="regular")
        result = fixture.interpolate(self.analysis)
        expect = [1.6]
        np.testing.assert_array_almost_equal(expect, result)

    def test_interpolate_given_regional_grid(self):
        fixture = obsoper.Operator(self.grid_lons,
                                   self.grid_lats,
                                   self.obs_lons,
                                   self.obs_lats,
                                   layout="regional")
        result = fixture.interpolate(self.analysis)
        expect = self.counterparts
        np.testing.assert_array_equal(expect, result)

    def test_constructor_given_unknown_layout_raises_exception(self):
        with self.assertRaises(exceptions.UnknownLayout):
            obsoper.Operator(None, None, None, None, layout="not a layout")


class TestOperatorRegularGrid(unittest.TestCase):
    def setUp(self):
        self.grid_lons = np.array([0, 1])
        self.grid_lats = np.array([0, 1])
        self.s_levels = np.array([[[0, 1],
                                   [0, 1]],
                                  [[0, 1],
                                   [0, 2]]])
        self.z_levels = np.array([0, 1])
        self.field = np.array([[[11, 12],
                                [13, 14]],
                               [[21, 22],
                                [23, 24]]])
        self.z_counterparts = 17.5
        self.s_counterparts = 17.4

    def make_operator(self, levels):
        obs_lons, obs_lats, obs_depths = [0.5], [0.5], [0.5]
        return obsoper.Operator(self.grid_lons,
                                self.grid_lats,
                                obs_lons,
                                obs_lats,
                                observed_depths=obs_depths,
                                grid_depths=levels)

    def test_vertical_interpolation_given_regular_layout_and_s_levels(self):
        self.check_interpolate(self.s_levels, self.s_counterparts)

    def test_vertical_interpolation_given_regular_layout_and_z_levels(self):
        self.check_interpolate(self.z_levels, self.z_counterparts)

    def check_interpolate(self, levels, expect):
        fixture = self.make_operator(levels)
        result = fixture.interpolate(self.field)
        np.testing.assert_array_almost_equal(expect, result)


class TestObservationOperator(unittest.TestCase):
    def setUp(self):
        self.longitudes = np.arange(3)
        self.latitudes = np.arange(3)
        self.depths = np.tile([10, 20, 30, 40], (3, 3, 1))
        self.fixture = ObservationOperator(self.longitudes, self.latitudes,
                                           self.depths)

        # Pseudo-model field
        self.surface_field = np.ones((3, 3))
        self.full_field = np.ones((3, 3, 4))

        # Sample coordinates
        self.inside_lons = np.array([0.9])
        self.inside_lats = np.array([1.5])
        self.outside_lons = np.array([-0.1])
        self.outside_lats = np.array([3.1])
        self.nan_lons_masked = np.ma.MaskedArray([np.nan], mask=[True])
        self.lons_masked = np.ma.MaskedArray([999., 999.], mask=[False, True])
        self.lats_masked = np.ma.MaskedArray([999., 999.], mask=[False, True])

    def test_interpolate_given_coordinates_and_depths(self):
        observed_depths = np.array([[15]])
        result = self.fixture.interpolate(self.full_field, self.inside_lons,
                                          self.inside_lats, observed_depths)
        expect = np.array([[1]])
        np.testing.assert_array_almost_equal(expect, result)

    def test_horizontal_interpolate(self):
        observed_lats, observed_lons = np.array([1]), np.array([1])
        result = self.fixture.horizontal_interpolate(self.surface_field,
                                                     observed_lons,
                                                     observed_lats)
        expect = [1]
        np.testing.assert_array_almost_equal(expect, result)

    def test_vertical_interpolate_given_section(self):
        model_section = np.array([[1, 2, 3, 4],
                                  [5, 6, 7, 8]])
        model_depths = np.array([[10, 20, 30, 40],
                                 [10, 20, 30, 40]])
        observed_depths = np.array([[15],
                                    [35]])
        result = self.fixture.vertical_interpolate(model_section,
                                                   model_depths,
                                                   observed_depths)
        expect = np.array([[1.5],
                           [7.5]])
        np.testing.assert_array_almost_equal(expect, result)

    def test_horizontal_interpolate_given_data_outside_returns_masked(self):
        result = self.fixture.horizontal_interpolate(self.surface_field,
                                                     self.outside_lons,
                                                     self.outside_lats)
        expect = np.ma.masked_all((1,))
        self.assertMaskedArrayEqual(expect, result)

    def test_horizontal_interpolate_given_masked_lons_nan(self):
        result = self.fixture.horizontal_interpolate(self.surface_field,
                                                     self.nan_lons_masked,
                                                     self.outside_lats)
        expect = np.ma.masked_all((1,))
        self.assertMaskedArrayEqual(expect, result)

    def test_horizontal_interpolate_given_masked_positions(self):
        result = self.fixture.horizontal_interpolate(self.full_field,
                                                     self.lons_masked,
                                                     self.lats_masked)
        expect = np.ma.masked_all((2, 4))
        self.assertMaskedArrayEqual(expect, result)

    def assertMaskedArrayEqual(self, expect, result):
        self.assertEqual(expect.shape, result.shape)
        np.testing.assert_array_almost_equal(expect.compressed(),
                                             result.compressed())
