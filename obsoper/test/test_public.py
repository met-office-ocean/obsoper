# pylint: disable=missing-docstring, invalid-name
import unittest
import pkg_resources

HAS_NETCDF4 = True
try:
    import netCDF4
except ImportError:
    HAS_NETCDF4 = False

import numpy as np
import obsoper


if HAS_NETCDF4:
    @unittest.skip("slow test")
    class TestTripolar(unittest.TestCase):
        def setUp(self):
            orca025_grid = pkg_resources.resource_filename("obsoper.test",
                                                           "data/orca025_grid.nc")
            self.grid_longitudes = self.read_variable(orca025_grid, "nav_lon")
            self.grid_latitudes = self.read_variable(orca025_grid, "nav_lat")

            self.observed_longitudes = np.array([100])
            self.observed_latitudes = np.array([10])

            shape = self.grid_longitudes.T.shape
            self.constant = 30.
            self.constant_field = np.full(shape, self.constant)

        def read_variable(self, path, name):
            with netCDF4.Dataset(path) as dataset:
                return np.ma.asarray(dataset.variables[name][:])

        def test_tripolar_interpolation_given_constant_surface_field(self):
            fixture = obsoper.Tripolar(self.grid_longitudes.T,
                                       self.grid_latitudes.T,
                                       self.observed_longitudes,
                                       self.observed_latitudes)
            result = fixture.interpolate(self.constant_field)
            expect = np.array([self.constant])
            np.testing.assert_array_equal(expect, result)


class TestPublicInterface(unittest.TestCase):
    def test_remove_halo_is_accessible_from_library_import(self):
        self.assertEqual(obsoper.orca.remove_halo, obsoper.remove_halo)

    def test_north_fold_is_accessible_from_library_import(self):
        self.assertEqual(obsoper.orca.north_fold, obsoper.north_fold)

    def test_section_is_accessible_from_library_import(self):
        self.assertEqual(obsoper.vertical.Section, obsoper.Section)
