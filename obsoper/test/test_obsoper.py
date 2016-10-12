# pylint: disable=missing-docstring, invalid-name
import unittest
import numpy as np
from obsoper import ObservationOperator


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

    def test_inside_grid_given_coordinate_inside_returns_true(self):
        result = self.fixture.inside_grid(self.inside_lons, self.inside_lats)
        expect = np.array([True], dtype=np.bool)
        np.testing.assert_array_equal(expect, result)

    def test_inside_grid_given_coordinate_outside_returns_false(self):
        result = self.fixture.inside_grid(self.outside_lons, self.outside_lats)
        expect = np.array([False], dtype=np.bool)
        np.testing.assert_array_equal(expect, result)
